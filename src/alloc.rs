//! System allocator that asks for transparent huge pages on big blocks, so
//! large tables and partition buffers take fewer TLB misses and page faults.

use std::alloc::{GlobalAlloc, Layout, System};

pub struct HugeAlloc;

#[cfg(target_os = "linux")]
const BIG: usize = 2 << 20;

#[cfg(target_os = "linux")]
#[inline]
unsafe fn advise(p: *mut u8, size: usize) {
    if size < BIG || p.is_null() {
        return;
    }
    const HP: usize = 2 << 20;
    let start = (p as usize + HP - 1) & !(HP - 1);
    let end = (p as usize + size) & !(HP - 1);
    if end > start {
        libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_HUGEPAGE);
    }
}

#[cfg(not(target_os = "linux"))]
#[inline]
unsafe fn advise(_p: *mut u8, _size: usize) {}

unsafe impl GlobalAlloc for HugeAlloc {
    #[inline]
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        advise(p, l.size());
        p
    }
    #[inline]
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(l);
        advise(p, l.size());
        p
    }
    #[inline]
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
    #[inline]
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = System.realloc(p, l, new);
        advise(q, new);
        q
    }
}
