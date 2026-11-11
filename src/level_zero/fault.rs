use super::Status;
use std::cell::RefCell;
use std::sync::{Mutex, mpsc};
use std::time::Duration;

struct Fault {
    call: &'static str,
    status: Status,
    skip: usize,
}

thread_local! {
    static LOCAL: RefCell<Vec<Fault>> = const { RefCell::new(Vec::new()) };
}
static WORKER: Mutex<Option<Fault>> = Mutex::new(None);
/// Faults armed by `nth` on any thread and not consumed yet: while zero, `status` skips the
/// thread-local lookup.
static LOCAL_ARMED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Set once a worker fault or pause was ever armed: until then `status` skips the
/// `thread::current()` name check, which otherwise costs every native call of every test.
static WORKER_ARMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
type Pause = (mpsc::Sender<()>, mpsc::Receiver<()>);
static PAUSE_EXECUTE: Mutex<Option<Pause>> = Mutex::new(None);

pub(crate) fn arm(call: &'static str, status: Status) {
    nth(call, 1, status);
}

pub(crate) fn nth(call: &'static str, occurrence: usize, status: Status) {
    assert!(occurrence != 0);
    LOCAL_ARMED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    LOCAL.with_borrow_mut(|faults| {
        faults.push(Fault {
            call,
            status,
            skip: occurrence - 1,
        })
    });
}

pub(crate) fn consumed() -> bool {
    LOCAL.with_borrow(Vec::is_empty)
}

pub(crate) fn arm_worker(call: &'static str, status: Status) {
    WORKER_ARMED.store(true, std::sync::atomic::Ordering::SeqCst);
    assert!(
        WORKER
            .lock()
            .unwrap()
            .replace(Fault {
                call,
                status,
                skip: 0
            })
            .is_none()
    );
}

pub(crate) fn pause_worker_execute() -> (mpsc::Receiver<()>, mpsc::Sender<()>) {
    WORKER_ARMED.store(true, std::sync::atomic::Ordering::SeqCst);
    let (entered, observed) = mpsc::channel();
    let (resume, paused) = mpsc::channel();
    assert!(
        PAUSE_EXECUTE
            .lock()
            .unwrap()
            .replace((entered, paused))
            .is_none()
    );
    (observed, resume)
}

pub(crate) fn status(call: &str) -> Option<Status> {
    if LOCAL_ARMED.load(std::sync::atomic::Ordering::SeqCst) != 0
        && let Some(status) = LOCAL.with_borrow_mut(|faults| {
        let index = faults.iter().position(|fault| fault.call == call)?;
        if faults[index].skip == 0 {
            LOCAL_ARMED.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            Some(faults.remove(index).status)
        } else {
            faults[index].skip -= 1;
            None
        }
    }) {
        return Some(status);
    }
    if !WORKER_ARMED.load(std::sync::atomic::Ordering::SeqCst)
        || std::thread::current().name() != Some("cuda-batch")
    {
        return None;
    }
    if call == "zeCommandQueueExecuteCommandLists"
        && let Some((entered, paused)) = PAUSE_EXECUTE.lock().unwrap().take()
    {
        let _ = entered.send(());
        let _ = paused.recv_timeout(Duration::from_secs(5));
    }
    let mut fault = WORKER.lock().unwrap();
    if fault.as_ref().is_some_and(|fault| fault.call == call) {
        fault.take().map(|fault| fault.status)
    } else {
        None
    }
}
