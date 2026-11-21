//! Per-thread bump arena and block lists.
//!
//! The pipeline scatters records into thousands of buckets per thread. As
//! separate `Vec`s, each bucket is too small for transparent huge pages, so
//! every 4 KiB page faults on its own. Here buckets are chains of fixed-size
//! blocks carved from 64 MiB regions aligned to 2 MiB, which the allocator
//! marks for huge pages: a few hundred faults instead of hundreds of thousands.

use std::alloc::{alloc, dealloc, Layout};

const REGION: usize = 64 << 20;
const ALIGN: usize = 2 << 20;
pub const BLOCK_BYTES: usize = 16 << 10;

pub struct Arena {
    regions: Vec<(*mut u8, Layout)>,
    cur: usize,
    end: usize,
}

unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    pub fn new() -> Self {
        Arena { regions: Vec::new(), cur: 0, end: 0 }
    }

    /// `bytes` of uninitialized memory, 64-byte aligned, valid until drop.
    #[inline]
    pub fn alloc(&mut self, bytes: usize) -> *mut u8 {
        let bytes = (bytes + 63) & !63;
        if self.cur + bytes > self.end {
            self.grow(bytes);
        }
        let p = self.cur;
        self.cur += bytes;
        p as *mut u8
    }

    #[cold]
    fn grow(&mut self, bytes: usize) {
        let size = bytes.max(REGION).next_multiple_of(ALIGN);
        let layout = Layout::from_size_align(size, ALIGN).unwrap();
        let p = unsafe { alloc(layout) };
        assert!(!p.is_null(), "out of memory");
        self.regions.push((p, layout));
        self.cur = p as usize;
        self.end = p as usize + size;
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        for &(p, l) in &self.regions {
            unsafe { dealloc(p, l) };
        }
    }
}

/// An append-only list of `T` in arena blocks. The arena it was filled from
/// must outlive every read, and `finish` must run before the first read.
///
/// When `T` divides a cache line, pushes are staged in the list's own 64-byte
/// line and written to the block a full line at a time with non-temporal
/// stores: scattering into thousands of lists then costs no read-for-ownership
/// miss per record.
#[repr(C, align(64))]
pub struct BlockList<T: Copy> {
    stage: [u8; 64],
    blocks: Vec<*mut T>,
    cur: *mut T,
    fill: usize,  // records in the current block
    sfill: usize, // records in the stage
    total: usize,
}

unsafe impl<T: Copy> Send for BlockList<T> {}
unsafe impl<T: Copy> Sync for BlockList<T> {}

impl<T: Copy> BlockList<T> {
    pub const CAP: usize = BLOCK_BYTES / std::mem::size_of::<T>();
    const SPL: usize = 64 / std::mem::size_of::<T>();
    const WC: bool = std::mem::size_of::<T>() <= 64 && 64 % std::mem::size_of::<T>() == 0;

    pub fn new() -> Self {
        BlockList { stage: [0; 64], blocks: Vec::new(), cur: std::ptr::null_mut(), fill: Self::CAP, sfill: 0, total: 0 }
    }

    #[inline(always)]
    pub fn push(&mut self, arena: &mut Arena, v: T) {
        self.total += 1;
        if Self::WC {
            unsafe { (self.stage.as_mut_ptr() as *mut T).add(self.sfill).write(v) };
            self.sfill += 1;
            if self.sfill == Self::SPL {
                self.flush_line(arena);
            }
        } else {
            if self.fill == Self::CAP {
                self.new_block(arena);
            }
            unsafe { self.cur.add(self.fill).write(v) };
            self.fill += 1;
        }
    }

    #[inline(always)]
    fn flush_line(&mut self, arena: &mut Arena) {
        if self.fill == Self::CAP {
            self.new_block(arena);
        }
        unsafe {
            let dst = self.cur.add(self.fill) as *mut u8;
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let src = self.stage.as_ptr() as *const __m128i;
                let d = dst as *mut __m128i;
                _mm_stream_si128(d, _mm_load_si128(src));
                _mm_stream_si128(d.add(1), _mm_load_si128(src.add(1)));
                _mm_stream_si128(d.add(2), _mm_load_si128(src.add(2)));
                _mm_stream_si128(d.add(3), _mm_load_si128(src.add(3)));
            }
            #[cfg(not(target_arch = "x86_64"))]
            std::ptr::copy_nonoverlapping(self.stage.as_ptr(), dst, 64);
        }
        self.fill += Self::SPL;
        self.sfill = 0;
    }

    /// Writes out the staged records and fences the streaming stores.
    pub fn finish(&mut self, arena: &mut Arena) {
        if Self::WC && self.sfill > 0 {
            if self.fill == Self::CAP {
                self.new_block(arena);
            }
            unsafe {
                std::ptr::copy_nonoverlapping(self.stage.as_ptr() as *const T, self.cur.add(self.fill), self.sfill);
            }
            self.fill += self.sfill;
            self.sfill = 0;
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_sfence();
        }
    }

    #[cold]
    #[inline(never)]
    fn new_block(&mut self, arena: &mut Arena) {
        let p = arena.alloc(BLOCK_BYTES) as *mut T;
        self.blocks.push(p);
        self.cur = p;
        self.fill = 0;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.total
    }

    /// The filled blocks as slices, in push order.
    #[inline]
    pub fn chunks(&self) -> impl Iterator<Item = &[T]> + '_ {
        debug_assert!(self.sfill == 0);
        let n = self.blocks.len();
        self.blocks.iter().enumerate().map(move |(i, &b)| {
            let len = if i + 1 == n { self.fill } else { Self::CAP };
            unsafe { std::slice::from_raw_parts(b as *const T, len) }
        })
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        self.chunks().flatten()
    }
}
