use super::Status;
use std::cell::RefCell;
use std::sync::{Mutex, mpsc};
use std::time::Duration;

struct Fault {
    call: &'static str,
    status: Status,
}

thread_local! {
    static LOCAL: RefCell<Vec<Fault>> = const { RefCell::new(Vec::new()) };
}
static WORKER: Mutex<Option<Fault>> = Mutex::new(None);
type Pause = (mpsc::Sender<()>, mpsc::Receiver<()>);
static PAUSE_EXECUTE: Mutex<Option<Pause>> = Mutex::new(None);

pub(crate) fn arm(call: &'static str, status: Status) {
    LOCAL.with_borrow_mut(|faults| faults.push(Fault { call, status }));
}

pub(crate) fn consumed() -> bool {
    LOCAL.with_borrow(Vec::is_empty)
}

pub(crate) fn arm_worker(call: &'static str, status: Status) {
    assert!(
        WORKER
            .lock()
            .unwrap()
            .replace(Fault { call, status })
            .is_none()
    );
}

pub(crate) fn pause_worker_execute() -> (mpsc::Receiver<()>, mpsc::Sender<()>) {
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
    if let Some(status) = LOCAL.with_borrow_mut(|faults| {
        let index = faults.iter().position(|fault| fault.call == call)?;
        Some(faults.remove(index).status)
    }) {
        return Some(status);
    }
    if std::thread::current().name() != Some("cuda-batch") {
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
