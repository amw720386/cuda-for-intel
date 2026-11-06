//! Internal instrumentation used by the hardware tests.
#[cfg(windows)]
use super::{Api, Context, Error, EventDesc, EventPoolDesc, Header, Raw, check};
use std::time::Duration;
#[cfg(windows)]
use std::{cell::Cell, ptr::null_mut};

#[cfg(windows)]
pub(crate) struct Clock {
    append: unsafe extern "system" fn(Raw, *mut u64, Raw, u32, *mut Raw) -> u32,
    sample: unsafe extern "system" fn(Raw, *mut u64, *mut u64) -> u32,
    pub(crate) timer: Timer,
}

#[cfg(windows)]
impl Clock {
    pub(crate) fn new(api: &Api, frequency: Option<u64>, bits: u32) -> Result<Self, Error> {
        let timer =
            Timer::new(frequency.unwrap_or_default(), bits).ok_or(Error::InvalidProperties)?;
        let append = unsafe {
            api._library
                .get(b"zeCommandListAppendWriteGlobalTimestamp\0")
        }
        .map_err(Error::Loader)?;
        let sample =
            unsafe { api._library.get(b"zeDeviceGetGlobalTimestamps\0") }.map_err(Error::Loader)?;
        Ok(Self {
            append: *append,
            sample: *sample,
            timer,
        })
    }

    pub(crate) fn sample(&self, context: &Context) -> Result<(u64, u64), Error> {
        let (mut host, mut device) = (0, 0);
        check("zeDeviceGetGlobalTimestamps", unsafe {
            (self.sample)(context.device.0, &mut host, &mut device)
        })?;
        Ok((host, device))
    }

    // The destination must be device-accessible and live through list completion.
    pub(super) unsafe fn append(&self, list: Raw, dst: *mut u64) -> Result<(), Error> {
        check("zeCommandListAppendWriteGlobalTimestamp", unsafe {
            (self.append)(list, dst, null_mut(), 0, null_mut())
        })
    }

    pub(crate) unsafe fn append_blocking(
        &self,
        context: &Context,
        dst: *mut u64,
    ) -> Result<(), Error> {
        unsafe { self.append(context.commands()?, dst) }
    }
}

#[cfg(windows)]
thread_local! {
    // The shared Forge fixture arms the next copy or launch on this host thread.
    // An operation may already have a dependency signal, which must be preserved.
    static NEXT_EVENT: Cell<Raw> = const { Cell::new(null_mut()) };
}

#[cfg(windows)]
pub(super) fn operation_signal(signal: Raw) -> Raw {
    let event = NEXT_EVENT.with(|next| next.replace(null_mut()));
    if event.is_null() {
        signal
    } else {
        assert!(signal.is_null(), "cannot replace a stream dependency");
        event
    }
}

#[cfg(windows)]
#[repr(C)]
#[derive(Default)]
struct TimestampResult {
    global: [u64; 2],
    context: [u64; 2],
}

#[cfg(windows)]
#[repr(C)]
struct MappedResults {
    header: Header,
    ticks: *mut TimestampResult,
    nanos: *mut TimestampResult,
}

#[cfg(windows)]
type MappedQuery = unsafe extern "system" fn(Raw, Raw, *mut u32, *mut MappedResults) -> u32;

#[cfg(windows)]
pub(crate) struct OperationEvents {
    pool: Raw,
    events: Vec<Raw>,
    query: unsafe extern "system" fn(Raw, *mut TimestampResult) -> u32,
    mapped: Option<MappedQuery>,
    device: Raw,
    pub(crate) timer: Timer,
}

#[cfg(windows)]
impl OperationEvents {
    pub(crate) fn create(
        api: &Api,
        context: &Context,
        count: u32,
        frequency: Option<u64>,
        bits: u32,
        map_to_host: bool,
    ) -> Result<Self, Error> {
        let timer =
            Timer::new(frequency.unwrap_or_default(), bits).ok_or(Error::InvalidProperties)?;
        let query =
            unsafe { api._library.get(b"zeEventQueryKernelTimestamp\0") }.map_err(Error::Loader)?;
        let mapped = if map_to_host {
            Some(
                *unsafe { api._library.get(b"zeEventQueryKernelTimestampsExt\0") }
                    .map_err(Error::Loader)?,
            )
        } else {
            None
        };
        let mut pool = null_mut();
        let desc = EventPoolDesc {
            header: Header::new(0x10),
            // HOST_VISIBLE plus KERNEL_MAPPED_TIMESTAMP or KERNEL_TIMESTAMP.
            flags: 1 | if map_to_host { 1 << 3 } else { 1 << 2 },
            count,
        };
        let mut device = context.device.0;
        check("zeEventPoolCreate (timestamps)", unsafe {
            (api.zeEventPoolCreate)(context.handle.0, &desc, 1, &mut device, &mut pool)
        })?;
        let mut timestamps = Self {
            pool,
            events: Vec::new(),
            query: *query,
            mapped,
            device,
            timer,
        };
        for index in 0..count {
            let desc = EventDesc {
                header: Header::new(0x11),
                index,
                signal: 0,
                wait: 0,
            };
            let mut event = null_mut();
            if let Err(error) = check("zeEventCreate (timestamps)", unsafe {
                (api.zeEventCreate)(pool, &desc, &mut event)
            }) {
                timestamps.destroy(api)?;
                return Err(error);
            }
            timestamps.events.push(event);
        }
        Ok(timestamps)
    }

    pub(crate) fn arm(&self, index: usize) {
        NEXT_EVENT.with(|next| assert!(next.replace(self.events[index]).is_null()));
    }

    pub(crate) fn disarm() -> bool {
        NEXT_EVENT.with(|next| !next.replace(null_mut()).is_null())
    }

    pub(crate) fn wait(
        &self,
        index: usize,
        wait: super::streams::WaitFn,
        timeout: u64,
    ) -> Result<(), Error> {
        check("zeEventHostSynchronize (timestamps)", unsafe {
            wait(self.events[index], timeout)
        })
    }

    pub(crate) fn query(&self, index: usize) -> Result<[u64; 2], Error> {
        let mut result = TimestampResult::default();
        check("zeEventQueryKernelTimestamp", unsafe {
            (self.query)(self.events[index], &mut result)
        })?;
        Ok(result.global)
    }

    pub(crate) fn reset(&self, api: &Api, index: usize) -> Result<(), Error> {
        check("zeEventHostReset (timestamps)", unsafe {
            (api.zeEventHostReset)(self.events[index])
        })
    }

    pub(crate) fn mapped_interval(&self, index: usize) -> Result<(u64, u64), Error> {
        let query = self.mapped.ok_or(Error::InvalidProperties)?;
        let event = self.events[index];
        let mut count = 0;
        check("zeEventQueryKernelTimestampsExt (count)", unsafe {
            query(event, self.device, &mut count, null_mut())
        })?;
        if count == 0 {
            return Err(Error::InvalidProperties);
        }
        let mut timestamps: Vec<_> = (0..count).map(|_| TimestampResult::default()).collect();
        // This Iris Xe driver requires both arrays even though the API marks
        // the raw tick output optional.
        let mut ticks: Vec<_> = (0..count).map(|_| TimestampResult::default()).collect();
        let mut results = MappedResults {
            header: Header::new(0x10012),
            ticks: ticks.as_mut_ptr(),
            nanos: timestamps.as_mut_ptr(),
        };
        check("zeEventQueryKernelTimestampsExt (results)", unsafe {
            query(event, self.device, &mut count, &mut results)
        })?;
        let timestamps = timestamps
            .get(..count as usize)
            .ok_or(Error::InvalidProperties)?;
        if timestamps.is_empty()
            || timestamps
                .iter()
                .any(|time| time.global[0] > time.global[1])
        {
            return Err(Error::InvalidProperties);
        }
        let start = timestamps.iter().map(|time| time.global[0]).min().unwrap();
        let end = timestamps.iter().map(|time| time.global[1]).max().unwrap();
        Ok((start, end))
    }

    pub(crate) fn destroy(&mut self, api: &Api) -> Result<(), Error> {
        while let Some(&event) = self.events.last() {
            check("zeEventDestroy (timestamps)", unsafe {
                (api.zeEventDestroy)(event)
            })?;
            self.events.pop();
        }
        check("zeEventPoolDestroy (timestamps)", unsafe {
            (api.zeEventPoolDestroy)(self.pool)
        })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Timer {
    frequency: u64,
    bits: u32,
}

impl Timer {
    fn new(frequency: u64, bits: u32) -> Option<Self> {
        (frequency > 0 && (1..=64).contains(&bits)).then_some(Self { frequency, bits })
    }

    pub(crate) fn ticks(self, start: u64, end: u64) -> u64 {
        end.wrapping_sub(start) & (u64::MAX >> (64 - self.bits))
    }

    pub(crate) fn seconds(self, ticks: u64) -> f64 {
        ticks as f64 / self.frequency as f64
    }

    pub(crate) fn window(self, start: u64, end: u64, elapsed: Duration) -> Option<u64> {
        // Wall time rules out ambiguous traces long enough for a full rollover.
        let period = 2.0f64.powi(self.bits as i32) / self.frequency as f64;
        (elapsed.as_secs_f64() < period).then(|| self.ticks(start, end))
    }
}

#[test]
fn timestamp_scale_and_rollover() {
    assert!(Timer::new(0, 36).is_none());
    assert!(Timer::new(1, 0).is_none());
    assert!(Timer::new(1, 65).is_none());
    let timer = Timer::new(20_000_000, 32).unwrap();
    assert_eq!(timer.ticks(0xffff_fff8, 0x0010_0000_000c), 20);
    assert_eq!(timer.seconds(20), 1e-6);
    assert_eq!(timer.window(100, 150, Duration::from_micros(3)), Some(50));
    assert_eq!(timer.window(0, 0, Duration::from_secs(300)), None);
    let timer = Timer::new(1, 64).unwrap();
    assert_eq!(timer.ticks(u64::MAX, 1), 2);
}
