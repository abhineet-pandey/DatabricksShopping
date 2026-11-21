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
/// must outlive every read.
pub struct BlockList<T: Copy> {
    blocks: Vec<*mut T>,
    cur: *mut T,
    fill: usize,
    total: usize,
}

unsafe impl<T: Copy> Send for BlockList<T> {}
unsafe impl<T: Copy> Sync for BlockList<T> {}

impl<T: Copy> BlockList<T> {
    pub const CAP: usize = BLOCK_BYTES / std::mem::size_of::<T>();

    pub fn new() -> Self {
        BlockList { blocks: Vec::new(), cur: std::ptr::null_mut(), fill: Self::CAP, total: 0 }
    }

    #[inline(always)]
    pub fn push(&mut self, arena: &mut Arena, v: T) {
        if self.fill == Self::CAP {
            self.new_block(arena);
        }
        unsafe { self.cur.add(self.fill).write(v) };
        self.fill += 1;
        self.total += 1;
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
