//! Development-only sampling profiler (cargo feature `prof`, Linux).
//!
//! A SIGPROF timer fires every millisecond of process CPU time; the handler
//! records the interrupted instruction pointer. `SS_PROF=path` writes the
//! executable's load address and every sample to `path`, for addr2line.
//! Without the feature both functions are empty.

#[cfg(all(feature = "prof", target_os = "linux"))]
mod imp {
    use std::sync::atomic::{AtomicUsize, Ordering};

    const MAX: usize = 1 << 22;
    static N: AtomicUsize = AtomicUsize::new(0);
    static mut BUF: [usize; MAX] = [0; MAX];

    extern "C" fn handler(_sig: libc::c_int, _info: *mut libc::siginfo_t, ctx: *mut libc::c_void) {
        let uc = ctx as *mut libc::ucontext_t;
        let ip = unsafe { (*uc).uc_mcontext.gregs[libc::REG_RIP as usize] as usize };
        let i = N.fetch_add(1, Ordering::Relaxed);
        if i < MAX {
            unsafe { *std::ptr::addr_of_mut!(BUF).cast::<usize>().add(i) = ip };
        }
    }

    pub fn start() {
        if std::env::var_os("SS_PROF").is_none() {
            return;
        }
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = handler as usize;
            sa.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;
            libc::sigaction(libc::SIGPROF, &sa, std::ptr::null_mut());
            let tv = libc::timeval { tv_sec: 0, tv_usec: 1000 };
            let it = libc::itimerval { it_interval: tv, it_value: tv };
            libc::setitimer(libc::ITIMER_PROF, &it, std::ptr::null_mut());
        }
    }

    pub fn dump() {
        let Some(path) = std::env::var_os("SS_PROF") else { return };
        unsafe {
            let it: libc::itimerval = std::mem::zeroed();
            libc::setitimer(libc::ITIMER_PROF, &it, std::ptr::null_mut());
        }
        let exe = std::fs::read_link("/proc/self/exe").unwrap_or_default();
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
        let mut base = 0usize;
        for l in maps.lines() {
            if l.ends_with(exe.to_string_lossy().as_ref()) {
                base = usize::from_str_radix(l.split('-').next().unwrap(), 16).unwrap();
                break;
            }
        }
        let n = N.load(Ordering::Relaxed).min(MAX);
        let mut out = format!("{base:x}\n");
        for i in 0..n {
            let ip = unsafe { *std::ptr::addr_of!(BUF).cast::<usize>().add(i) };
            out.push_str(&format!("{ip:x}\n"));
        }
        let _ = std::fs::write(path, out);
    }
}

#[cfg(not(all(feature = "prof", target_os = "linux")))]
mod imp {
    pub fn start() {}
    pub fn dump() {}
}

pub use imp::{dump, start};
