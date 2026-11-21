//! Shopping Sessions: native pipeline. See README.md for the design.

mod alloc;
mod arena;
mod csv;
mod ids;
mod parquet;
mod prof;
mod timeparse;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use arena::{Arena, BlockList};
use ids::{Interner, Names};
use timeparse::{fast_iso, fast_ts, parse_price, DAY_MS};

#[global_allocator]
static GLOBAL: alloc::HugeAlloc = alloc::HugeAlloc;

const GAP_MS: i64 = 1_800_000;
const REACH_MS: i64 = 604_800_000;
const CHUNK: usize = 1 << 20;

// ---------------------------------------------------------------- hashing

#[inline(always)]
fn hp(x: u64, bits: u32) -> usize {
    (x.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - bits)) as usize
}
#[inline(always)]
fn hl(x: u64, bits: u32) -> usize {
    ((x ^ (x >> 31)).wrapping_mul(0xD6E8_FEB8_6659_FD93) >> (64 - bits)) as usize
}
#[inline(always)]
fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
}
#[inline(always)]
fn mixh(x: u64) -> u64 {
    (x ^ (x >> 31)).wrapping_mul(0xD6E8_FEB8_6659_FD93)
}
/// Candidate index = block number << CSHIFT | offset in block.
const CSHIFT: u32 = 9;
const _: () = assert!(BlockList::<Cand>::CAP <= 1 << CSHIFT);
#[inline(always)]
fn cand_at(blocks: &[*const Cand], idx: u32) -> &Cand {
    unsafe { &*(*blocks.get_unchecked((idx >> CSHIFT) as usize)).add((idx & ((1 << CSHIFT) - 1)) as usize) }
}
fn table_bits(n: usize) -> u32 {
    let need = (n * 2).max(16);
    usize::BITS - (need - 1).leading_zeros()
}

// ---------------------------------------------------------------- dimensions

const DENSE_CAP: u64 = 1 << 22;
const NONE32: u32 = u32::MAX;

struct Dim {
    prefix: &'static [u8],
    dense: Vec<u32>,
    map: HashMap<Vec<u8>, u32>,
}

impl Dim {
    fn new(prefix: &'static [u8]) -> Self {
        Dim { prefix, dense: Vec::new(), map: HashMap::new() }
    }
    #[inline(always)]
    fn fast(&self, s: &[u8]) -> Option<u64> {
        let p = self.prefix.len();
        if s.len() <= p || s.len() > p + 9 || &s[..p] != self.prefix {
            return None;
        }
        let r = &s[p..];
        if r.len() > 1 && r[0] == b'0' {
            return None;
        }
        let mut v = 0u64;
        for &b in r {
            let d = (b as u64).wrapping_sub(b'0' as u64);
            if d > 9 {
                return None;
            }
            v = v * 10 + d;
        }
        if v < DENSE_CAP {
            Some(v)
        } else {
            None
        }
    }
    fn insert(&mut self, s: &[u8], v: u32) {
        match self.fast(s) {
            Some(n) => {
                let n = n as usize;
                if self.dense.len() <= n {
                    self.dense.resize(n + 1, NONE32);
                }
                self.dense[n] = v;
            }
            None => {
                self.map.insert(s.to_vec(), v);
            }
        }
    }
    #[inline(always)]
    fn get(&self, s: &[u8]) -> Option<u32> {
        match self.fast(s) {
            Some(n) => match self.dense.get(n as usize) {
                Some(&x) if x != NONE32 => Some(x),
                _ => None,
            },
            None => self.map.get(s).copied(),
        }
    }
}

fn read_table(path: &Path) -> Vec<(Vec<u8>, Vec<u8>)> {
    let data = std::fs::read(path).unwrap_or_default();
    let mut out = Vec::new();
    if let Some(nl) = data.iter().position(|&c| c == b'\n') {
        csv::for_each_line_fast::<2, _>(&data[nl + 1..], |f| out.push((f[0].to_vec(), f[1].to_vec())));
    }
    out
}

// ---------------------------------------------------------------- records

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Cand {
    eid: u64,
    user: u64,
    ingest: u32,
    ts: u32,
    pc: u32, // price | category << 24; 0 for view/cart
    valid: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Ev {
    person: u64,
    key: u64, // ts << 32 | pc
}

#[derive(Clone, Copy)]
#[repr(C)]
struct ClickRaw {
    user: u64,
    kid: u64,
    ts: u32,
    camp: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Cl {
    person: u64,
    kid: u64,
    key: u64, // ts << 32 | camp
}

struct Ctx {
    w0: i64,
    w1: i64,
    base: i64,
    ingest_base_secs: i64,
    products: Dim,
    campaigns: Dim,
    interner: Interner,
    pbits: u32,
    unknown_cat: u32,
}

impl Ctx {
    #[inline(always)]
    fn ingest_key(&self, ms: i64) -> u32 {
        let k = ms.div_euclid(1000) - self.ingest_base_secs + (1i64 << 31);
        k.clamp(0, u32::MAX as i64 - 1) as u32
    }
}

// ---------------------------------------------------------------- parsing

fn parse_events(buf: &[u8], ctx: &Ctx, cands: &mut [BlockList<Cand>], arena: &mut Arena) {
    let (w0, w1, base) = (ctx.w0, ctx.w1, ctx.base);
    csv::for_each_line_fast::<7, _>(buf, |f| {
        let e = f[0];
        if e.is_empty() {
            return;
        }
        let eid = ctx.interner.code(e);
        let mut c = Cand { eid, user: 0, ingest: u32::MAX, ts: 0, pc: 0, valid: 0 };
        if let Some(ing) = fast_iso(f[3]) {
            c.ingest = ctx.ingest_key(ing);
            'v: {
                if f[1].is_empty() || f[5].is_empty() {
                    break 'v;
                }
                let ts = match fast_ts(f[2]) {
                    Some(t) if t >= w0 && t < w1 => t,
                    _ => break 'v,
                };
                if ing - ts > DAY_MS {
                    break 'v;
                }
                let ty = f[4];
                let pc = if ty == b"view" || ty == b"cart" {
                    0
                } else if ty == b"purchase" {
                    let price = match parse_price(f[6]) {
                        Some(p) => p,
                        None => break 'v,
                    };
                    let cat = ctx.products.get(f[5]).unwrap_or(ctx.unknown_cat);
                    price | (cat << 24)
                } else {
                    break 'v;
                };
                c.user = ctx.interner.code(f[1]);
                c.ts = (ts - base) as u32;
                c.pc = pc;
                c.valid = 1;
            }
        }
        unsafe { cands.get_unchecked_mut(hp(eid, ctx.pbits)) }.push(arena, c);
    });
}

fn parse_clicks(buf: &[u8], ctx: &Ctx, out: &mut Vec<ClickRaw>) {
    let lo = ctx.w0 - REACH_MS;
    let hi = ctx.w1;
    let base = ctx.base;
    csv::for_each_line_fast::<4, _>(buf, |f| {
        if f[0].is_empty() || f[1].is_empty() {
            return;
        }
        let camp = match ctx.campaigns.get(f[2]) {
            Some(c) => c,
            None => return,
        };
        let ts = match fast_ts(f[3]) {
            Some(t) if t >= lo && t < hi => t,
            _ => return,
        };
        out.push(ClickRaw {
            user: ctx.interner.code(f[1]),
            kid: ctx.interner.code(f[0]),
            ts: (ts - base) as u32,
            camp,
        });
    });
}

fn parse_links(buf: &[u8], ctx: &Ctx, out: &mut Vec<(u64, u64)>) {
    csv::for_each_line_fast::<2, _>(buf, |f| {
        if f[0].is_empty() || f[1].is_empty() {
            return;
        }
        out.push((ctx.interner.code(f[0]), ctx.interner.code(f[1])));
    });
}

// ---------------------------------------------------------------- identity

/// Shared mutable pointer for disjoint parallel writes.
#[derive(Clone, Copy)]
struct SyncPtr<T>(*mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

/// user code -> canonical person code, for every ID that appears in a link.
/// The open-addressing table is split into `1 << pbits` contiguous parts by
/// the top bits of the slot hash; probing wraps inside a part, so each part
/// can be built by one thread.
struct LinkMap {
    bits: u32,
    lbits: u32,
    slots: Vec<(u64, u64)>,
}

impl LinkMap {
    fn new(n_keys: usize) -> Self {
        let bits = table_bits(n_keys);
        let pb = 8.min(bits.saturating_sub(6));
        LinkMap { bits, lbits: bits - pb, slots: vec![(0, 0); 1usize << bits] }
    }
    #[inline(always)]
    fn nparts(&self) -> usize {
        1 << (self.bits - self.lbits)
    }
    #[inline(always)]
    fn part_of(&self, k: u64) -> usize {
        hl(k, self.bits) >> self.lbits
    }
    #[inline(always)]
    fn slot_of(&self, k: u64) -> usize {
        hl(k, self.bits)
    }
    #[inline(always)]
    fn get(&self, k: u64) -> u64 {
        let h = hl(k, self.bits);
        let lmask = (1usize << self.lbits) - 1;
        let base = h & !lmask;
        let mut l = h & lmask;
        loop {
            let s = unsafe { self.slots.get_unchecked(base + l) };
            if s.0 == k {
                return s.1;
            }
            if s.0 == 0 {
                return k;
            }
            l = (l + 1) & lmask;
        }
    }
}

/// Parallel part of the identity build: every distinct linked ID gets a dense
/// node id; each edge endpoint learns its (part-local) node id.
struct LinkPre {
    map: LinkMap,
    part_base: Vec<u32>,
    ep_node: Vec<u32>, // 2 per edge
    node_code: Vec<u64>,
}

fn prepare_links(links: &[Vec<(u64, u64)>], threads: usize) -> LinkPre {
    let n_edges: usize = links.iter().map(|l| l.len()).sum();
    let mut map = LinkMap::new(n_edges * 2);
    let np = map.nparts();
    let mut offs = Vec::with_capacity(links.len());
    let mut acc = 0usize;
    for l in links {
        offs.push(acc);
        acc += l.len();
    }
    // 1. scatter endpoints (code, endpoint index) by part
    let map_ref = &map;
    let scatter_one = |t: usize| {
        let mut out: Vec<Vec<(u64, u32)>> = (0..np).map(|_| Vec::new()).collect();
        let base = offs[t];
        for (i, &(a, b)) in links[t].iter().enumerate() {
            let e = ((base + i) * 2) as u32;
            out[map_ref.part_of(a)].push((a, e));
            out[map_ref.part_of(b)].push((b, e + 1));
        }
        out
    };
    let scattered: Vec<Vec<Vec<(u64, u32)>>> = if threads == 1 {
        (0..links.len()).map(&scatter_one).collect()
    } else {
        run_parallel(links.len(), &scatter_one)
    };
    // 2. per part: insert into its slot range, assign local ids
    let mut ep_node: Vec<u32> = vec![0; n_edges * 2];
    let lbits = map.lbits;
    let bits = map.bits;
    let lmask = (1usize << lbits) - 1;
    let slots_ptr = SyncPtr(map.slots.as_mut_ptr());
    let ep_ptr = SyncPtr(ep_node.as_mut_ptr());
    let next = AtomicUsize::new(0);
    let locals: Vec<Vec<(usize, Vec<u64>)>> = run_parallel(threads, |_| {
        let slots_ptr = slots_ptr;
        let ep_ptr = ep_ptr;
        let mut done = Vec::new();
        loop {
            let p = next.fetch_add(1, Ordering::Relaxed);
            if p >= np {
                break;
            }
            let slots = unsafe { std::slice::from_raw_parts_mut(slots_ptr.0.add(p << lbits), 1 << lbits) };
            let mut codes: Vec<u64> = Vec::new();
            for sc in &scattered {
                for &(k, e) in &sc[p] {
                    let mut l = hl(k, bits) & lmask;
                    let id = loop {
                        let s = unsafe { slots.get_unchecked_mut(l) };
                        if s.0 == k {
                            break s.1 as u32;
                        }
                        if s.0 == 0 {
                            let id = codes.len() as u32;
                            *s = (k, id as u64);
                            codes.push(k);
                            break id;
                        }
                        l = (l + 1) & lmask;
                    };
                    unsafe { *ep_ptr.0.add(e as usize) = id };
                }
            }
            done.push((p, codes));
        }
        done
    });
    drop(scattered);
    let mut per_part: Vec<Vec<u64>> = (0..np).map(|_| Vec::new()).collect();
    for v in locals {
        for (p, c) in v {
            per_part[p] = c;
        }
    }
    let mut part_base = Vec::with_capacity(np);
    let mut node_code = Vec::new();
    for c in &per_part {
        part_base.push(node_code.len() as u32);
        node_code.extend_from_slice(c);
    }
    LinkPre { map, part_base, ep_node, node_code }
}

/// Sequential part: union-find over dense ids, canonical names, final table.
fn finish_links(pre: LinkPre, links: &[Vec<(u64, u64)>], names: &Interner) -> LinkMap {
    let LinkPre { mut map, part_base, mut ep_node, node_code } = pre;
    let n = node_code.len();
    // endpoint local ids -> global ids
    let mut e = 0usize;
    for l in links {
        for &(a, b) in l {
            ep_node[e] += part_base[map.part_of(a)];
            ep_node[e + 1] += part_base[map.part_of(b)];
            e += 2;
        }
    }
    let mut parent: Vec<u32> = (0..n as u32).collect();
    #[inline(always)]
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        unsafe {
            while *parent.get_unchecked(x as usize) != x {
                let p = *parent.get_unchecked(x as usize);
                let gp = *parent.get_unchecked(p as usize);
                *parent.get_unchecked_mut(x as usize) = gp;
                x = gp;
            }
        }
        x
    }
    for pair in ep_node.chunks_exact(2) {
        let ra = find(&mut parent, pair[0]);
        let rb = find(&mut parent, pair[1]);
        if ra != rb {
            if ra < rb {
                parent[rb as usize] = ra;
            } else {
                parent[ra as usize] = rb;
            }
        }
    }
    let mut best: Vec<u64> = vec![0; n];
    for i in 0..n {
        let r = find(&mut parent, i as u32) as usize;
        parent[i] = r as u32;
        let c = node_code[i];
        let b = best[r];
        let less = if (b | c) & ids::FALLBACK == 0 { c < b } else { names.cmp(c, b) == std::cmp::Ordering::Less };
        if b == 0 || less {
            best[r] = c;
        }
    }
    // after the loop above every node points directly at its root
    if n >= 1 << 16 {
        // Rebuild at ~62% load (the build table is sized from the edge count,
        // ~4 slots per ID): half the bytes, so lookups hit cache more often.
        // Parts hold >= 512 slots here, far from overflowing.
        drop(map);
        let bits = usize::BITS - ((n + n / 2 + n / 8) - 1).leading_zeros();
        let pb = 8.min(bits.saturating_sub(6));
        let mut m = LinkMap { bits, lbits: bits - pb, slots: vec![(0, 0); 1usize << bits] };
        let lmask = (1usize << m.lbits) - 1;
        for i in 0..n {
            let k = node_code[i];
            let v = best[parent[i] as usize];
            let h = hl(k, bits);
            let base = h & !lmask;
            let mut l = h & lmask;
            loop {
                let s = unsafe { m.slots.get_unchecked_mut(base + l) };
                if s.0 == 0 {
                    *s = (k, v);
                    break;
                }
                l = (l + 1) & lmask;
            }
        }
        return m;
    }
    let lbits = map.lbits;
    for (p, chunk) in map.slots.chunks_mut(1 << lbits).enumerate() {
        let base = part_base[p];
        for s in chunk {
            if s.0 != 0 {
                let r = parent[(base + s.1 as u32) as usize] as usize;
                s.1 = best[r];
            }
        }
    }
    map
}

// ---------------------------------------------------------------- files

struct Input {
    files: Vec<(std::fs::File, usize)>,
}

fn list_csv(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |x| x == "csv") && p.is_file())
            .collect(),
        Err(_) => Vec::new(),
    };
    v.sort();
    v
}

/// A byte range of one file. The lines that start inside it belong to it.
#[derive(Clone, Copy)]
struct Chunk {
    file: u32,
    kind: u8, // 0 links, 1 events, 2 clicks
    start: usize,
    end: usize,
}

fn make_chunks(input: &mut Input, paths: &[PathBuf], kind: u8, chunk: usize, out: &mut Vec<Chunk>) {
    for p in paths {
        let f = match std::fs::File::open(p) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let len = f.metadata().map(|m| m.len()).unwrap_or(0) as usize;
        if len == 0 {
            continue;
        }
        let fi = input.files.len() as u32;
        input.files.push((f, len));
        let mut start = 0;
        while start < len {
            let end = (start + chunk).min(len);
            out.push(Chunk { file: fi, kind, start, end });
            start = end;
        }
    }
}

/// Reads up to `n` bytes at `off`, appending to `buf`; returns the count.
fn read_at(f: &std::fs::File, off: usize, n: usize, buf: &mut Vec<u8>) -> usize {
    let old = buf.len();
    buf.reserve(n + 64);
    unsafe { buf.set_len(old + n) };
    let mut got = 0;
    while got < n {
        #[cfg(unix)]
        let r = std::os::unix::fs::FileExt::read_at(f, &mut buf[old + got..old + n], (off + got) as u64);
        #[cfg(windows)]
        let r = std::os::windows::fs::FileExt::seek_read(f, &mut buf[old + got..old + n], (off + got) as u64);
        match r {
            Ok(0) => break,
            Ok(k) => got += k,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => panic!("read: {e}"),
        }
    }
    unsafe { buf.set_len(old + got) };
    got
}

/// Loads the whole lines that start in `c` into `buf` and returns them. The
/// header (the file's first line) never belongs to a chunk.
fn load_chunk<'b>(input: &Input, c: &Chunk, buf: &'b mut Vec<u8>) -> &'b [u8] {
    let (f, len) = &input.files[c.file as usize];
    let from = c.start.saturating_sub(1);
    buf.clear();
    read_at(f, from, c.end - from, buf);
    // first line start at or after c.start: just after the first '\n' at or
    // after c.start - 1 (for c.start == 0 that skips the header)
    let begin = match memchr(b'\n', buf) {
        Some(i) => i + 1,
        None => return &[],
    };
    if from + begin >= c.end {
        return &[];
    }
    // extend to the end of the line that starts last
    if buf.last() != Some(&b'\n') {
        let mut at = c.end;
        loop {
            if at >= *len {
                break;
            }
            let k = buf.len();
            let got = read_at(f, at, 1 << 16, buf);
            if got == 0 {
                break;
            }
            at += got;
            if let Some(i) = memchr(b'\n', &buf[k..]) {
                buf.truncate(k + i + 1);
                break;
            }
        }
    }
    &buf[begin..]
}

#[inline]
fn memchr(c: u8, s: &[u8]) -> Option<usize> {
    s.iter().position(|&x| x == c)
}

// ---------------------------------------------------------------- per-person processing

struct Scratch {
    // grouping
    tbits: u32,
    table: Vec<(u64, u32)>,
    persons: Vec<u64>,
    ecount: Vec<u32>,
    eoff: Vec<u32>,
    ccount: Vec<u32>,
    coff: Vec<u32>,
    evs: Vec<u64>,
    cls: Vec<(u64, u64)>,
    // per person
    day_sessions: Vec<i64>,
    day_events: Vec<i64>,
    day_rev: Vec<i64>,
    day_cat: Vec<i64>, // ndays * ncat
    touched: Vec<u32>,
    camp_count: Vec<u32>,
    camp_pos: Vec<u32>,
    active: Vec<u32>,
}

struct UserRows {
    person: Vec<u64>,
    day: Vec<i32>,
    sessions: Vec<i64>,
    events: Vec<i64>,
    revenue: Vec<i64>,
    top: Vec<u32>, // NONE32 for null
}

struct Agg {
    ndays: usize,
    ncat: usize,
    day0: i64,
    base: i64,
    organic: u32,
}

#[inline(always)]
fn ev_ts(k: u64) -> i64 {
    (k >> 32) as i64
}

fn process_partition(
    q: usize,
    ev_parts: &[Vec<BlockList<Ev>>],
    cl_parts: &[Vec<BlockList<Cl>>],
    s: &mut Scratch,
    agg: &Agg,
    names: &Names,
    camp: &mut [[i64; 4]],
    rows: &mut UserRows,
) {
    let ne: usize = ev_parts.iter().map(|p| p[q].len()).sum();
    if ne == 0 {
        return;
    }
    // ---- group events by person
    let bits = table_bits(ne);
    let size = 1usize << bits;
    let mask = size - 1;
    s.table.clear();
    s.table.resize(size, (0, 0));
    s.persons.clear();
    s.ecount.clear();
    let lookup_insert = |t: &mut Vec<(u64, u32)>, persons: &mut Vec<u64>, ecount: &mut Vec<u32>, k: u64| -> usize {
        let mut i = hl(k, bits);
        loop {
            let e = unsafe { t.get_unchecked_mut(i) };
            if e.0 == k {
                return e.1 as usize;
            }
            if e.0 == 0 {
                let id = persons.len();
                *e = (k, id as u32);
                persons.push(k);
                ecount.push(0);
                return id;
            }
            i = (i + 1) & mask;
        }
    };
    for p in ev_parts {
        for e in p[q].iter() {
            let id = lookup_insert(&mut s.table, &mut s.persons, &mut s.ecount, e.person);
            s.ecount[id] += 1;
        }
    }
    let np = s.persons.len();
    s.eoff.clear();
    let mut acc = 0u32;
    for &c in &s.ecount {
        s.eoff.push(acc);
        acc += c;
    }
    s.eoff.push(acc);
    s.evs.clear();
    s.evs.resize(ne, 0);
    {
        // reuse ecount as cursor
        let mut cur: Vec<u32> = s.eoff[..np].to_vec();
        for p in ev_parts {
            for e in p[q].iter() {
                let mut i = hl(e.person, bits);
                let id = loop {
                    let t = unsafe { s.table.get_unchecked(i) };
                    if t.0 == e.person {
                        break t.1 as usize;
                    }
                    i = (i + 1) & mask;
                };
                let at = cur[id];
                cur[id] = at + 1;
                unsafe { *s.evs.get_unchecked_mut(at as usize) = e.key };
            }
        }
    }
    // ---- group clicks of persons with events
    s.ccount.clear();
    s.ccount.resize(np, 0);
    let find = |t: &Vec<(u64, u32)>, k: u64| -> Option<usize> {
        let mut i = hl(k, bits);
        loop {
            let e = unsafe { t.get_unchecked(i) };
            if e.0 == k {
                return Some(e.1 as usize);
            }
            if e.0 == 0 {
                return None;
            }
            i = (i + 1) & mask;
        }
    };
    let mut nc = 0usize;
    for p in cl_parts {
        for c in p[q].iter() {
            if let Some(id) = find(&s.table, c.person) {
                s.ccount[id] += 1;
                nc += 1;
            }
        }
    }
    s.coff.clear();
    let mut acc = 0u32;
    for &c in &s.ccount {
        s.coff.push(acc);
        acc += c;
    }
    s.coff.push(acc);
    s.cls.clear();
    s.cls.resize(nc, (0, 0));
    if nc > 0 {
        let mut cur: Vec<u32> = s.coff[..np].to_vec();
        for p in cl_parts {
            for c in p[q].iter() {
                if let Some(id) = find(&s.table, c.person) {
                    let at = cur[id];
                    cur[id] = at + 1;
                    s.cls[at as usize] = (c.key, c.kid);
                }
            }
        }
    }

    // ---- each person
    let ndays = agg.ndays;
    let ncat = agg.ncat;
    for pi in 0..np {
        let evs = &mut s.evs[s.eoff[pi] as usize..s.eoff[pi + 1] as usize];
        sort_u64(evs);
        let cls = &mut s.cls[s.coff[pi] as usize..s.coff[pi + 1] as usize];
        if cls.len() > 1 {
            cls.sort_unstable_by_key(|c| c.0);
        }
        let cls: &[(u64, u64)] = cls;
        let evs: &[u64] = evs;
        let person = s.persons[pi];

        let mut r = 0usize; // clicks consumed (ts <= start)
        let mut l = 0usize; // clicks expired (ts < start - reach)
        let n = evs.len();
        let mut i = 0usize;
        while i < n {
            let start = ev_ts(evs[i]);
            let mut prev = start;
            let mut count = 0i64;
            let mut rev = 0i64;
            let start_abs = agg.base + start;
            let di = (start_abs.div_euclid(DAY_MS) - agg.day0) as usize;
            if s.day_sessions[di] == 0 {
                s.touched.push(di as u32);
            }
            while i < n {
                let k = evs[i];
                let t = ev_ts(k);
                if t - prev > GAP_MS {
                    break;
                }
                prev = t;
                count += 1;
                let pc = (k & 0xFFFF_FFFF) as u32;
                if pc != 0 {
                    let price = (pc & 0x00FF_FFFF) as i64;
                    rev += price;
                    let cat = (pc >> 24) as usize;
                    s.day_cat[di * ncat + cat] += price;
                }
                i += 1;
            }
            s.day_sessions[di] += 1;
            s.day_events[di] += count;
            s.day_rev[di] += rev;

            // ---- credit and assists
            let credited: u32;
            if cls.is_empty() {
                credited = agg.organic;
            } else {
                while r < cls.len() && ev_ts(cls[r].0) <= start {
                    let c = (cls[r].0 & 0xFFFF_FFFF) as u32 as usize;
                    if s.camp_count[c] == 0 {
                        s.camp_pos[c] = s.active.len() as u32;
                        s.active.push(c as u32);
                    }
                    s.camp_count[c] += 1;
                    r += 1;
                }
                while l < r && ev_ts(cls[l].0) < start - REACH_MS {
                    let c = (cls[l].0 & 0xFFFF_FFFF) as u32 as usize;
                    s.camp_count[c] -= 1;
                    if s.camp_count[c] == 0 {
                        let pos = s.camp_pos[c] as usize;
                        let last = s.active.pop().unwrap();
                        if pos < s.active.len() {
                            s.active[pos] = last;
                            s.camp_pos[last as usize] = pos as u32;
                        }
                    }
                    l += 1;
                }
                if l < r {
                    // latest click group: ts == ts of cls[r-1]
                    let lt = ev_ts(cls[r - 1].0);
                    let mut best = r - 1;
                    let mut j = r - 1;
                    while j > l && ev_ts(cls[j - 1].0) == lt {
                        j -= 1;
                        if names.cmp(cls[j].1, cls[best].1) == std::cmp::Ordering::Less {
                            best = j;
                        }
                    }
                    credited = (cls[best].0 & 0xFFFF_FFFF) as u32;
                    for &a in &s.active {
                        camp[a as usize * ndays + di][3] += 1;
                    }
                } else {
                    credited = agg.organic;
                }
            }
            let row = &mut camp[credited as usize * ndays + di];
            row[0] += 1;
            row[1] += (rev > 0) as i64;
            row[2] += rev;
        }
        // reset click state
        for &a in &s.active {
            s.camp_count[a as usize] = 0;
        }
        s.active.clear();

        // ---- emit user rows
        for &d in &s.touched {
            let d = d as usize;
            let mut top = NONE32;
            if s.day_rev[d] > 0 {
                let mut bestv = 0i64;
                for c in 0..ncat {
                    let v = s.day_cat[d * ncat + c];
                    if v > bestv {
                        bestv = v;
                        top = c as u32;
                    }
                    s.day_cat[d * ncat + c] = 0;
                }
            }
            rows.person.push(person);
            rows.day.push((agg.day0 + d as i64) as i32);
            rows.sessions.push(s.day_sessions[d]);
            rows.events.push(s.day_events[d]);
            rows.revenue.push(s.day_rev[d]);
            rows.top.push(top);
            s.day_sessions[d] = 0;
            s.day_events[d] = 0;
            s.day_rev[d] = 0;
        }
        s.touched.clear();
    }
}

fn sort_u64(v: &mut [u64]) {
    let n = v.len();
    if n <= 1 {
        return;
    }
    if n <= 24 {
        for i in 1..n {
            let x = v[i];
            let mut j = i;
            while j > 0 && v[j - 1] > x {
                v[j] = v[j - 1];
                j -= 1;
            }
            v[j] = x;
        }
        return;
    }
    v.sort_unstable();
}

// ---------------------------------------------------------------- main

fn env_path(k: &str, d: &str) -> PathBuf {
    PathBuf::from(std::env::var(k).unwrap_or_else(|_| d.to_string()))
}

struct Timer {
    t0: Instant,
    on: bool,
}
fn cpu_ms() -> f64 {
    #[cfg(target_os = "linux")]
    unsafe {
        let mut r: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut r);
        let t = |v: libc::timeval| v.tv_sec as f64 * 1e3 + v.tv_usec as f64 / 1e3;
        return t(r.ru_utime) + t(r.ru_stime);
    }
    #[allow(unreachable_code)]
    0.0
}

impl Timer {
    fn mark(&self, what: &str) {
        if self.on {
            eprintln!("[{:8.1} ms wall {:8.1} ms cpu] {what}", self.t0.elapsed().as_secs_f64() * 1e3, cpu_ms());
        }
    }
}

fn run_parallel<R: Send, F: Fn(usize) -> R + Sync>(threads: usize, f: F) -> Vec<R> {
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads).map(|t| {
            let f = &f;
            sc.spawn(move || f(t))
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

fn main() {
    prof::start();
    let timer = Timer { t0: Instant::now(), on: std::env::var("SS_TIMING").is_ok() };
    let args: Vec<String> = std::env::args().collect();
    let (data_dir, out_dir) = if args.len() >= 3 {
        (PathBuf::from(&args[1]), PathBuf::from(&args[2]))
    } else {
        (env_path("DATA_DIR", "/data"), env_path("OUTPUT_DIR", "/output"))
    };
    let threads: usize = std::env::var("NCPU")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(8, |n| n.get()));
    let threads: usize = threads.max(1);
    let threads = std::env::var("SS_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(threads);

    // ---- manifest
    let manifest = std::fs::read_to_string(data_dir.join("manifest.json")).expect("manifest.json");
    let w0 = timeparse::manifest_value(&manifest, "window_start").expect("window_start");
    let w1 = timeparse::manifest_value(&manifest, "window_end").expect("window_end");
    let base = w0 - REACH_MS;
    if w1 - base >= u32::MAX as i64 || w1 <= w0 {
        eprintln!("window too large for this build");
        std::process::exit(2);
    }
    let day0 = w0.div_euclid(DAY_MS);
    let ndays = ((w1 - 1).div_euclid(DAY_MS) - day0 + 1) as usize;

    // ---- dimensions
    let prod_rows = read_table(&data_dir.join("products.csv"));
    let mut cat_names: Vec<Vec<u8>> = prod_rows.iter().map(|r| r.1.clone()).collect();
    cat_names.push(b"unknown".to_vec());
    cat_names.sort();
    cat_names.dedup();
    if cat_names.len() > 255 {
        eprintln!("too many categories for this build");
        std::process::exit(2);
    }
    let cat_rank: HashMap<Vec<u8>, u32> = cat_names.iter().enumerate().map(|(i, c)| (c.clone(), i as u32)).collect();
    let unknown_rank = cat_rank[&b"unknown"[..]];
    let mut products = Dim::new(b"p");
    for (p, c) in &prod_rows {
        products.insert(p, cat_rank[c]);
    }
    let camp_rows = read_table(&data_dir.join("campaigns.csv"));
    let mut campaigns = Dim::new(b"c_");
    let mut camp_ids: Vec<Vec<u8>> = Vec::new();
    let mut camp_channels: Vec<Vec<u8>> = Vec::new();
    for (c, ch) in &camp_rows {
        if campaigns.get(c).is_some() {
            continue;
        }
        campaigns.insert(c, camp_ids.len() as u32);
        camp_ids.push(c.clone());
        camp_channels.push(ch.clone());
    }
    let ncamp = camp_ids.len();
    let organic = ncamp as u32;
    camp_ids.push(b"organic".to_vec());
    camp_channels.push(b"organic".to_vec());

    let pbits: u32 = std::env::var("SS_PBITS").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
    let nparts = 1usize << pbits;
    let ctx = Ctx {
        w0,
        w1,
        base,
        ingest_base_secs: w0.div_euclid(1000),
        products,
        campaigns,
        interner: Interner::default(),
        pbits,
        unknown_cat: unknown_rank,
    };

    // ---- files
    let chunk: usize = std::env::var("SS_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(CHUNK);
    let mut input = Input { files: Vec::new() };
    let mut link_chunks = Vec::new();
    make_chunks(&mut input, &list_csv(&data_dir.join("links")), 0, chunk, &mut link_chunks);
    let mut work_chunks = Vec::new();
    make_chunks(&mut input, &list_csv(&data_dir.join("events")), 1, chunk, &mut work_chunks);
    make_chunks(&mut input, &list_csv(&data_dir.join("clicks")), 2, chunk, &mut work_chunks);
    // larger chunks first is not needed: chunks are similar in size
    timer.mark("setup");

    // ---- phase 1: links
    let next = AtomicUsize::new(0);
    let links: Vec<Vec<(u64, u64)>> = run_parallel(threads, |_| {
        let mut out = Vec::new();
        let mut buf = Vec::new();
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(c) = link_chunks.get(i) else { break };
            parse_links(load_chunk(&input, c, &mut buf), &ctx, &mut out);
        }
        out
    });
    timer.mark("links parsed");

    // ---- phase 2: events + clicks (thread 0 builds the link map first)
    let next = AtomicUsize::new(0);
    let qbits = pbits;
    let nq = 1usize << qbits;
    // Thread 0 builds the whole link map while the others start parsing.
    // Clicks come after every event chunk, so by the time a thread reaches
    // them the map is (almost always) ready, and clicks go straight to their
    // person partition.
    let lm_cell: std::sync::OnceLock<LinkMap> = std::sync::OnceLock::new();
    let parsed: Vec<(Vec<BlockList<Cand>>, Arena, Vec<BlockList<Cl>>, Arena)> = run_parallel(threads, |t| {
        if t == 0 {
            let m = finish_links(prepare_links(&links, 1), &links, &ctx.interner);
            let _ = lm_cell.set(m);
            timer.mark("link map built");
        }
        let mut arena = Arena::new();
        let mut cands: Vec<BlockList<Cand>> = (0..nparts).map(|_| BlockList::new()).collect();
        let mut carena = Arena::new();
        let mut clp: Vec<BlockList<Cl>> = (0..nq).map(|_| BlockList::new()).collect();
        let mut clicks: Vec<ClickRaw> = Vec::new();
        let mut cbuf: Vec<u8> = Vec::new();
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(c) = work_chunks.get(i) else { break };
            let buf = load_chunk(&input, c, &mut cbuf);
            if c.kind == 1 {
                parse_events(buf, &ctx, &mut cands, &mut arena);
            } else {
                clicks.clear();
                parse_clicks(buf, &ctx, &mut clicks);
                let lm = loop {
                    if let Some(m) = lm_cell.get() {
                        break m;
                    }
                    std::thread::yield_now();
                };
                for (j, c) in clicks.iter().enumerate() {
                    if let Some(n) = clicks.get(j + 16) {
                        prefetch(&lm.slots[lm.slot_of(n.user)]);
                    }
                    let person = lm.get(c.user);
                    unsafe { clp.get_unchecked_mut(hp(person, qbits)) }.push(&mut carena, Cl {
                        person,
                        kid: c.kid,
                        key: ((c.ts as u64) << 32) | c.camp as u64,
                    });
                }
            }
        }
        for l in cands.iter_mut() {
            l.finish(&mut arena);
        }
        for l in clp.iter_mut() {
            l.finish(&mut carena);
        }
        (cands, arena, clp, carena)
    });
    let mut cand_parts: Vec<Vec<BlockList<Cand>>> = Vec::with_capacity(threads);
    let mut cand_arenas: Vec<Arena> = Vec::with_capacity(threads);
    let mut cl_parts: Vec<Vec<BlockList<Cl>>> = Vec::with_capacity(threads);
    let mut click_arenas: Vec<Arena> = Vec::with_capacity(threads);
    for (c, a, k, ka) in parsed {
        cand_parts.push(c);
        cand_arenas.push(a);
        cl_parts.push(k);
        click_arenas.push(ka);
    }
    let linkmap = lm_cell.into_inner().unwrap();
    drop(links);
    timer.mark("events+clicks parsed");

    // ---- phase 3: dedup events, map persons, partition by person
    let next = AtomicUsize::new(0);
    let total_tasks = nparts;
    let dbg = timer.on;
    let routed: Vec<(Vec<BlockList<Ev>>, Arena)> = run_parallel(threads, |tid| {
        let (mut t_ins, mut t_emit) = (0f64, 0f64);
        let mut arena = Arena::new();
        let mut evp: Vec<BlockList<Ev>> = (0..nq).map(|_| BlockList::new()).collect();
        let mut ftab: Vec<u64> = Vec::new();
        let mut bptrs: Vec<*const Cand> = Vec::new();
        let mut batch: Vec<(u64, u64)> = Vec::with_capacity(64);
        let flush = |batch: &mut Vec<(u64, u64)>, evp: &mut Vec<BlockList<Ev>>, arena: &mut Arena| {
            for &(u, _) in batch.iter() {
                prefetch(&linkmap.slots[linkmap.slot_of(u)]);
            }
            for &(u, key) in batch.iter() {
                let person = linkmap.get(u);
                unsafe { evp.get_unchecked_mut(hp(person, qbits)) }.push(arena, Ev { person, key });
            }
            batch.clear();
        };
        loop {
            let task = next.fetch_add(1, Ordering::Relaxed);
            if task >= total_tasks {
                break;
            }
            if task < nparts {
                let b = task;
                let n: usize = cand_parts.iter().map(|p| p[b].len()).sum();
                if n == 0 {
                    continue;
                }
                let tq = Instant::now();
                // 8-byte slots: 32-bit fingerprint | candidate index + 1 (0 = empty).
                // Full IDs are compared only when fingerprints match.
                let need = n + n / 2 + 16;
                let bits = usize::BITS - (need - 1).leading_zeros();
                let mask = (1usize << bits) - 1;
                ftab.clear();
                ftab.resize(1usize << bits, 0);
                bptrs.clear();
                for p in cand_parts.iter() {
                    for ch in p[b].chunks() {
                        let bn = bptrs.len() as u32;
                        bptrs.push(ch.as_ptr());
                        for (off, c) in ch.iter().enumerate() {
                            let me = ((bn << CSHIFT) | off as u32) + 1;
                            let hh = mixh(c.eid);
                            let fp = ((hh >> 16) as u32 as u64) << 32;
                            let mut h = (hh >> (64 - bits)) as usize;
                            loop {
                                let e = unsafe { ftab.get_unchecked_mut(h) };
                                if *e == 0 {
                                    *e = fp | me as u64;
                                    break;
                                }
                                if *e & !0xFFFF_FFFF == fp {
                                    let o = cand_at(&bptrs, *e as u32 - 1);
                                    if o.eid == c.eid {
                                        if c.ingest < o.ingest {
                                            *e = fp | me as u64;
                                        }
                                        break;
                                    }
                                }
                                h = (h + 1) & mask;
                            }
                        }
                    }
                }
                let tq2 = Instant::now();
                t_ins += (tq2 - tq).as_secs_f64();
                // Walk the candidates in the same order; a valid one is emitted
                // when its slot names it as the event's kept copy.
                let mut bn = 0u32;
                for p in cand_parts.iter() {
                    for ch in p[b].chunks() {
                        for (off, c) in ch.iter().enumerate() {
                            if c.valid == 0 {
                                continue;
                            }
                            let me = ((bn << CSHIFT) | off as u32) + 1;
                            let hh = mixh(c.eid);
                            let fp = ((hh >> 16) as u32 as u64) << 32;
                            let mut h = (hh >> (64 - bits)) as usize;
                            let kept = loop {
                                let e = unsafe { *ftab.get_unchecked(h) };
                                if e & !0xFFFF_FFFF == fp {
                                    if e as u32 == me {
                                        break true;
                                    }
                                    if cand_at(&bptrs, e as u32 - 1).eid == c.eid {
                                        break false;
                                    }
                                }
                                h = (h + 1) & mask;
                            };
                            if !kept {
                                continue;
                            }
                            batch.push((c.user, ((c.ts as u64) << 32) | c.pc as u64));
                            if batch.len() == 32 {
                                flush(&mut batch, &mut evp, &mut arena);
                            }
                        }
                        bn += 1;
                    }
                }
                flush(&mut batch, &mut evp, &mut arena);
                t_emit += tq2.elapsed().as_secs_f64();
            }
        }
        if dbg && tid < 2 {
            eprintln!("  t{tid}: dedup insert {:.0} ms, emit+lookup {:.0} ms", t_ins * 1e3, t_emit * 1e3);
        }
        for l in evp.iter_mut() {
            l.finish(&mut arena);
        }
        (evp, arena)
    });
    // Freeing hundreds of MiB takes a while; do it off the critical path.
    let _freer = std::thread::spawn(move || {
        drop(cand_parts);
        drop(cand_arenas);
    });
    let mut ev_parts = Vec::with_capacity(threads);
    let mut route_arenas = Vec::with_capacity(threads);
    for (e, a) in routed {
        ev_parts.push(e);
        route_arenas.push(a);
    }
    timer.mark("dedup+route");

    // ---- phase 4: sessions per person
    let names = {
        let g = ctx.interner.inner.lock().unwrap();
        Names { strings: g.strings.clone() }
    };
    let ncat = cat_names.len();
    let agg = Agg { ndays, ncat, day0, base, organic };
    let next = AtomicUsize::new(0);
    let results: Vec<(UserRows, Vec<[i64; 4]>)> = run_parallel(threads, |_| {
        let mut s = Scratch {
            tbits: 0,
            table: Vec::new(),
            persons: Vec::new(),
            ecount: Vec::new(),
            eoff: Vec::new(),
            ccount: Vec::new(),
            coff: Vec::new(),
            evs: Vec::new(),
            cls: Vec::new(),
            day_sessions: vec![0; ndays],
            day_events: vec![0; ndays],
            day_rev: vec![0; ndays],
            day_cat: vec![0; ndays * ncat],
            touched: Vec::new(),
            camp_count: vec![0; ncamp + 1],
            camp_pos: vec![0; ncamp + 1],
            active: Vec::new(),
        };
        let _ = s.tbits;
        let mut camp = vec![[0i64; 4]; (ncamp + 1) * ndays];
        let mut rows = UserRows {
            person: Vec::new(),
            day: Vec::new(),
            sessions: Vec::new(),
            events: Vec::new(),
            revenue: Vec::new(),
            top: Vec::new(),
        };
        loop {
            let q = next.fetch_add(1, Ordering::Relaxed);
            if q >= nq {
                break;
            }
            process_partition(q, &ev_parts, &cl_parts, &mut s, &agg, &names, &mut camp, &mut rows);
        }
        (rows, camp)
    });
    let _freer2 = std::thread::spawn(move || {
        drop(ev_parts);
        drop(cl_parts);
        drop(route_arenas);
        drop(click_arenas);
    });
    timer.mark("sessions");

    // ---- phase 5: write
    let ud = out_dir.join("user_daily");
    let cd = out_dir.join("campaign_daily");
    std::fs::create_dir_all(&ud).expect("mkdir user_daily");
    std::fs::create_dir_all(&cd).expect("mkdir campaign_daily");
    let mut camp_total = vec![[0i64; 4]; (ncamp + 1) * ndays];
    let mut user_rows = Vec::with_capacity(threads);
    for (rows, camp) in results {
        for (a, b) in camp_total.iter_mut().zip(&camp) {
            for k in 0..4 {
                a[k] += b[k];
            }
        }
        user_rows.push(rows);
    }
    let nfiles = user_rows.len();
    std::thread::scope(|sc| {
        for (t, rows) in user_rows.into_iter().enumerate() {
            let ud = &ud;
            let names = &names;
            let cat_names = &cat_names;
            sc.spawn(move || {
                if rows.person.is_empty() && t != 0 {
                    return;
                }
                write_user(&ud.join(format!("part-{t:05}.parquet")), rows, names, cat_names);
            });
        }
        let _ = nfiles;
        // campaign table on this thread
        let mut cid = Vec::new();
        let mut ch = Vec::new();
        let mut day = Vec::new();
        let mut cols: [Vec<i64>; 4] = Default::default();
        let mut n = 0;
        for c in 0..=ncamp {
            for d in 0..ndays {
                let r = &camp_total[c * ndays + d];
                if r[0] > 0 || r[3] > 0 {
                    parquet::push_str(&mut cid, &camp_ids[c]);
                    parquet::push_str(&mut ch, &camp_channels[c]);
                    day.push((day0 + d as i64) as i32);
                    for k in 0..4 {
                        cols[k].push(r[k]);
                    }
                    n += 1;
                }
            }
        }
        let [s0, s1, s2, s3] = cols;
        parquet::write_file(
            &cd.join("part-00000.parquet"),
            n,
            &[
                parquet::Column { name: "campaign_id", values: parquet::Values::Str(cid) },
                parquet::Column { name: "channel", values: parquet::Values::Str(ch) },
                parquet::Column { name: "day", values: parquet::Values::Date(day) },
                parquet::Column { name: "sessions", values: parquet::Values::I64(s0) },
                parquet::Column { name: "purchasing_sessions", values: parquet::Values::I64(s1) },
                parquet::Column { name: "revenue_cents", values: parquet::Values::I64(s2) },
                parquet::Column { name: "assisted_sessions", values: parquet::Values::I64(s3) },
            ],
        )
        .expect("write campaign_daily");
    });
    timer.mark("written");
    prof::dump();
}

fn write_user(path: &Path, rows: UserRows, names: &Names, cat_names: &[Vec<u8>]) {
    let n = rows.person.len();
    let mut ids = Vec::with_capacity(n * 14);
    let mut tmp = Vec::with_capacity(32);
    for &p in &rows.person {
        tmp.clear();
        names.bytes_of(p, &mut tmp);
        parquet::push_str(&mut ids, &tmp);
    }
    let mut cats = Vec::new();
    let mut defs = Vec::with_capacity(n);
    for &t in &rows.top {
        if t == NONE32 {
            defs.push(false);
        } else {
            defs.push(true);
            parquet::push_str(&mut cats, &cat_names[t as usize]);
        }
    }
    parquet::write_file(
        path,
        n,
        &[
            parquet::Column { name: "user_id", values: parquet::Values::Str(ids) },
            parquet::Column { name: "day", values: parquet::Values::Date(rows.day) },
            parquet::Column { name: "sessions", values: parquet::Values::I64(rows.sessions) },
            parquet::Column { name: "events", values: parquet::Values::I64(rows.events) },
            parquet::Column { name: "revenue_cents", values: parquet::Values::I64(rows.revenue) },
            parquet::Column { name: "top_category", values: parquet::Values::OptStr(cats, defs) },
        ],
    )
    .expect("write user_daily");
}
