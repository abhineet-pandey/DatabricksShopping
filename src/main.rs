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

use arena::{Arena, BlockList, ZTable};
use ids::{LocalIds, LongNames};
use timeparse::{fast_iso_secs, fast_ts, parse_price, DAY_MS};

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
fn cand_at(blocks: &[*mut Cand], idx: u32) -> *mut Cand {
    unsafe { (*blocks.get_unchecked((idx >> CSHIFT) as usize)).add((idx & ((1 << CSHIFT) - 1)) as usize) }
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
    pbits: u32,
    unknown_cat: u32,
}

impl Ctx {
    #[inline(always)]
    fn ingest_key(&self, secs: i64) -> u32 {
        let k = secs - self.ingest_base_secs + (1i64 << 31);
        k.clamp(0, u32::MAX as i64 - 1) as u32
    }
}

// ---------------------------------------------------------------- parsing

#[inline(never)]
fn parse_events(buf: &[u8], ctx: &Ctx, cands: &mut [BlockList<Cand>], arena: &mut Arena, ids: &mut LocalIds) {
    let (w0, w1, base) = (ctx.w0, ctx.w1, ctx.base);
    csv::for_each_line_fast::<7, _>(buf, |f| {
        let e = f[0];
        if e.is_empty() {
            return;
        }
        let (eid, fp) = ids.event_code(e);
        // bit 0: valid; bits 1..: second fingerprint of a long event ID
        let mut c = Cand { eid, user: 0, ingest: u32::MAX, ts: 0, pc: 0, valid: fp << 1 };
        if let Some(ing_s) = fast_iso_secs(f[3]) {
            c.ingest = ctx.ingest_key(ing_s);
            let ing = ing_s * 1000;
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
                c.user = ids.code(f[1]);
                c.ts = (ts - base) as u32;
                c.pc = pc;
                c.valid |= 1;
            }
        }
        unsafe { cands.get_unchecked_mut(hp(eid, ctx.pbits)) }.push(arena, c);
    });
}

#[inline(never)]
fn parse_clicks(buf: &[u8], ctx: &Ctx, out: &mut Vec<ClickRaw>, ids: &mut LocalIds) {
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
            user: ids.code(f[1]),
            kid: ids.code(f[0]),
            ts: (ts - base) as u32,
            camp,
        });
    });
}

fn parse_links(buf: &[u8], out: &mut Vec<(u64, u64)>, ids: &mut LocalIds) {
    csv::for_each_line_fast::<2, _>(buf, |f| {
        if f[0].is_empty() || f[1].is_empty() {
            return;
        }
        out.push((ids.code(f[0]), ids.code(f[1])));
    });
}

// ---------------------------------------------------------------- identity

/// user code -> canonical person code, for every ID that appears in a link.
/// The open-addressing table is split into `1 << (bits - lbits)` contiguous
/// parts by the top bits of the slot hash; probing wraps inside a part.
///
/// A blocked Bloom filter (two bits in one 64-bit word per key, ~7 bits per
/// key) answers "not linked" for most unlinked IDs from cache, before the
/// table is touched. It has no false negatives, so it never changes a result.
struct LinkMap {
    bits: u32,
    lbits: u32,
    slots: ZTable<(u64, u64)>,
    filter: Vec<u64>,
    fmask: usize,
}

impl LinkMap {
    fn with_bits(bits: u32) -> Self {
        let pb = 8.min(bits.saturating_sub(6));
        LinkMap { bits, lbits: bits - pb, slots: ZTable::new(1usize << bits), filter: vec![!0u64], fmask: 0 }
    }
    /// (filter word, bit mask) of `k`; independent of the slot hash.
    #[inline(always)]
    fn fpos(&self, k: u64) -> (usize, u64) {
        let m = (k ^ (k >> 29)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        ((m >> 40) as usize & self.fmask, (1u64 << ((m >> 28) & 63)) | (1u64 << ((m >> 34) & 63)))
    }
    /// Builds the filter from every key in the table.
    fn seal(&mut self, keys: &[u64]) {
        let words = (keys.len() * 7 / 64).max(1).next_power_of_two();
        self.filter = vec![0u64; words];
        self.fmask = words - 1;
        for &k in keys {
            let (w, b) = self.fpos(k);
            self.filter[w] |= b;
        }
    }
    #[inline(always)]
    fn part_of(&self, k: u64) -> usize {
        hl(k, self.bits) >> self.lbits
    }
    /// False only when `k` is certainly not in the table.
    #[inline(always)]
    fn maybe(&self, k: u64) -> bool {
        let (w, b) = self.fpos(k);
        unsafe { *self.filter.get_unchecked(w) & b == b }
    }
    #[inline(always)]
    fn prefetch_filter(&self, k: u64) {
        prefetch(unsafe { self.filter.as_ptr().add(self.fpos(k).0) });
    }
    #[inline(always)]
    fn slot_of(&self, k: u64) -> usize {
        hl(k, self.bits)
    }
    #[inline(always)]
    fn get(&self, k: u64) -> u64 {
        if !self.maybe(k) {
            return k;
        }
        self.get_slow(k)
    }
    /// `get` without the filter.
    #[inline(always)]
    fn get_slow(&self, k: u64) -> u64 {
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
    /// The slot holding `k`, or the empty slot where it belongs.
    #[inline(always)]
    fn find_slot(&mut self, k: u64) -> &mut (u64, u64) {
        let h = hl(k, self.bits);
        let lmask = (1usize << self.lbits) - 1;
        let base = h & !lmask;
        let mut l = h & lmask;
        loop {
            let s = unsafe { &*self.slots.as_ptr().add(base + l) };
            if s.0 == k || s.0 == 0 {
                return unsafe { self.slots.get_unchecked_mut(base + l) };
            }
            l = (l + 1) & lmask;
        }
    }
}

/// Builds the identity map on one thread (it runs beside the parsers):
/// dense node ids, union-find, the byte-wise smallest ID per person, and a
/// compact final table. Random table accesses are prefetched a few edges ahead.
// Never inlined: inlined into the parse workers' closure, it made the hot
// parse loop measurably slower.
#[inline(never)]
fn build_links(links: &[Vec<(u64, u64)>], names: &LongNames) -> LinkMap {
    let n_edges: usize = links.iter().map(|l| l.len()).sum();
    build_links_from(links, names, bits_for(n_edges + n_edges / 3))
}

/// Table bits for `n` keys at <= ~62% load.
fn bits_for(n: usize) -> u32 {
    usize::BITS - ((n + n / 2 + n / 8).max(16) - 1).leading_zeros()
}

/// `build_links` starting from a table of `init_bits` bits (tests start small
/// to exercise growth).
fn build_links_from(links: &[Vec<(u64, u64)>], names: &LongNames, init_bits: u32) -> LinkMap {
    const AHEAD: usize = 8;
    /// A table of `node_code[id] -> id` with `bits` bits, and its part fills.
    fn rehash(node_code: &[u64], bits: u32) -> (LinkMap, Vec<u32>) {
        let mut m = LinkMap::with_bits(bits);
        let mut fill = vec![0u32; 1 << (m.bits - m.lbits)];
        for (id, &k) in node_code.iter().enumerate() {
            *m.find_slot(k) = (k, id as u64);
            fill[m.part_of(k)] += 1;
        }
        (m, fill)
    }
    let n_edges: usize = links.iter().map(|l| l.len()).sum();
    // 1. every distinct ID gets a dense node id in first-seen order; the ends
    //    of edge i are ep_node[2i] and ep_node[2i + 1]. The table is sized
    //    for ~4/3 IDs per edge and is also the final one (person codes are
    //    written in place below). It doubles if the IDs outgrow it or a part
    //    gets 7/8 full, so probing always ends.
    let (mut map, mut fill) = rehash(&[], init_bits);
    let mut node_code: Vec<u64> = Vec::with_capacity(n_edges + n_edges / 3);
    let mut ep_node: Vec<u32> = Vec::with_capacity(n_edges * 2);
    for l in links {
        for (i, &(a, b)) in l.iter().enumerate() {
            if let Some(&(na, nb)) = l.get(i + AHEAD) {
                prefetch(&map.slots[map.slot_of(na)]);
                prefetch(&map.slots[map.slot_of(nb)]);
            }
            for k in [a, b] {
                let next = node_code.len() as u64;
                let s = map.find_slot(k);
                let id = if s.0 == k {
                    s.1
                } else {
                    *s = (k, next);
                    node_code.push(k);
                    let p = map.part_of(k);
                    fill[p] += 1;
                    let part_cap = 1usize << map.lbits;
                    if fill[p] as usize * 8 > part_cap * 7 || node_code.len() * 8 > map.slots.len() * 5 {
                        (map, fill) = rehash(&node_code, map.bits + 1);
                    }
                    next
                };
                ep_node.push(id as u32);
            }
        }
    }
    drop(fill);
    let n = node_code.len();
    // 2. union-find over node ids
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
    drop(ep_node);
    // 3. each person's name: the byte-wise smallest of its IDs
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
    // after the loop above every node points directly at its root:
    // write each ID's person code over its node id
    for s in map.slots.iter_mut() {
        if s.0 != 0 {
            s.1 = best[parent[s.1 as usize] as usize];
        }
    }
    map.seal(&node_code);
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
    eperson: Vec<u32>, // person id of each event, in read order
    ekey: Vec<u64>,    // key of each event, in read order
    ra: Vec<u64>,      // radix sort buffers: ts << 32 | read index
    rb: Vec<u64>,
    hist: Vec<u32>,    // 3 radix digit histograms
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
    camp_snap: Vec<i64>, // assisted-session count of the day at activation
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
    names: &LongNames,
    camp: &mut [[i64; 4]],
    rows: &mut UserRows,
) {
    let ne: usize = ev_parts.iter().map(|p| p[q].len()).sum();
    if ne == 0 {
        return;
    }
    // ---- group events by person. A partition holds far fewer persons than
    // events, so the table starts small and doubles whenever it is half full
    // (the same <= 50% load as sizing it from the event count).
    let mut bits = 12u32.min(table_bits(ne));
    s.table.clear();
    s.table.resize(1usize << bits, (0, 0));
    s.persons.clear();
    s.ecount.clear();
    s.eperson.clear();
    s.ekey.clear();
    for p in ev_parts {
        for e in p[q].iter() {
            let k = e.person;
            let mask = (1usize << bits) - 1;
            let mut i = hl(k, bits);
            let id = loop {
                let t = unsafe { s.table.get_unchecked_mut(i) };
                if t.0 == k {
                    break t.1 as usize;
                }
                if t.0 == 0 {
                    let id = s.persons.len();
                    *t = (k, id as u32);
                    s.persons.push(k);
                    s.ecount.push(0);
                    break id;
                }
                i = (i + 1) & mask;
            };
            s.ecount[id] += 1;
            s.eperson.push(id as u32);
            if s.persons.len() * 2 > 1usize << bits {
                bits += 1;
                let mask = (1usize << bits) - 1;
                s.table.clear();
                s.table.resize(1usize << bits, (0, 0));
                for (id, &k) in s.persons.iter().enumerate() {
                    let mut i = hl(k, bits);
                    while s.table[i].0 != 0 {
                        i = (i + 1) & mask;
                    }
                    s.table[i] = (k, id as u32);
                }
            }
        }
    }
    let bits = bits;
    let mask = (1usize << bits) - 1;
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
        // LSD radix sort of the whole partition by ts, then a stable scatter
        // by person: each person's events end up sorted by ts (the order of
        // equal timestamps does not matter).
        let mut cur: Vec<u32> = s.eoff[..np].to_vec();
        let mut ids = s.eperson.iter();
        for p in ev_parts {
            for e in p[q].iter() {
                let id = *ids.next().unwrap() as usize;
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
        let evs: &[u64] = evs;
        let cls = &mut s.cls[s.coff[pi] as usize..s.coff[pi + 1] as usize];
        if cls.len() > 1 {
            cls.sort_unstable_by_key(|c| c.0);
        }
        let cls: &[(u64, u64)] = cls;
        let evs: &[u64] = evs;
        let person = s.persons[pi];

        // Assists are counted lazily: `dsess` counts this person's sessions
        // with an active campaign on day `cur_d`; a campaign is credited
        // `dsess - camp_snap` when it leaves the window or the day changes.
        let mut cur_d = usize::MAX;
        let mut dsess = 0i64;
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
                if di != cur_d {
                    if cur_d != usize::MAX {
                        for &a in &s.active {
                            camp[a as usize * ndays + cur_d][3] += dsess - s.camp_snap[a as usize];
                            s.camp_snap[a as usize] = 0;
                        }
                    }
                    cur_d = di;
                    dsess = 0;
                }
                while r < cls.len() && ev_ts(cls[r].0) <= start {
                    let c = (cls[r].0 & 0xFFFF_FFFF) as u32 as usize;
                    if s.camp_count[c] == 0 {
                        s.camp_pos[c] = s.active.len() as u32;
                        s.active.push(c as u32);
                        s.camp_snap[c] = dsess;
                    }
                    s.camp_count[c] += 1;
                    r += 1;
                }
                while l < r && ev_ts(cls[l].0) < start - REACH_MS {
                    let c = (cls[l].0 & 0xFFFF_FFFF) as u32 as usize;
                    s.camp_count[c] -= 1;
                    if s.camp_count[c] == 0 {
                        camp[c * ndays + cur_d][3] += dsess - s.camp_snap[c];
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
                    dsess += 1;
                } else {
                    credited = agg.organic;
                }
            }
            let row = &mut camp[credited as usize * ndays + di];
            row[0] += 1;
            row[1] += (rev > 0) as i64;
            row[2] += rev;
        }
        // flush assists, reset click state
        for &a in &s.active {
            camp[a as usize * ndays + cur_d][3] += dsess - s.camp_snap[a as usize];
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

const RADIX_BITS: u32 = 11;

/// `ts << 32 | index` for every key, sorted by ts (3 LSD passes of 11 bits;
/// relative timestamps are below 2^31). Returns the sorted slice.
fn radix_by_ts<'a>(keys: &[u64], ra: &'a mut Vec<u64>, rb: &'a mut Vec<u64>, hist: &mut [u32]) -> &'a [u64] {
    const R: usize = 1 << RADIX_BITS;
    const M: u64 = R as u64 - 1;
    let n = keys.len();
    ra.clear();
    ra.extend(keys.iter().enumerate().map(|(i, &k)| (k & !0xFFFF_FFFF) | i as u64));
    rb.clear();
    rb.resize(n, 0);
    hist.fill(0);
    let (h0, rest) = hist.split_at_mut(R);
    let (h1, h2) = rest.split_at_mut(R);
    for &a in ra.iter() {
        unsafe {
            *h0.get_unchecked_mut((a >> 32 & M) as usize) += 1;
            *h1.get_unchecked_mut((a >> (32 + RADIX_BITS) & M) as usize) += 1;
            *h2.get_unchecked_mut((a >> (32 + 2 * RADIX_BITS) & M) as usize) += 1;
        }
    }
    let mut src: &mut Vec<u64> = ra;
    let mut dst: &mut Vec<u64> = rb;
    for (pass, h) in [h0, h1, h2].into_iter().enumerate() {
        let shift = 32 + RADIX_BITS * pass as u32;
        // a digit shared by every key leaves the order unchanged
        if h[(src[0] >> shift & M) as usize] as usize == n {
            continue;
        }
        let mut acc = 0u32;
        for c in h.iter_mut() {
            let v = *c;
            *c = acc;
            acc += v;
        }
        for &a in src.iter() {
            unsafe {
                let o = h.get_unchecked_mut((a >> shift & M) as usize);
                *dst.get_unchecked_mut(*o as usize) = a;
                *o += 1;
            }
        }
        std::mem::swap(&mut src, &mut dst);
    }
    src
}

const RANK_MAX: usize = 48;

/// Branch-free sort of up to `RANK_MAX` keys below 2^63: each key goes to
/// its rank (the number of smaller keys). Equal keys are identical values, so
/// they share a rank and the slots after it are filled with the same value.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn rank_sort(v: &mut [u64]) {
    use std::arch::x86_64::*;
    let n = v.len();
    debug_assert!(n <= RANK_MAX);
    let mut buf = [i64::MAX as u64; RANK_MAX]; // padding never counts as smaller
    buf[..n].copy_from_slice(v);
    let mut out = [u64::MAX; RANK_MAX];
    let chunks = n.div_ceil(4);
    let bp = buf.as_ptr() as *const __m256i;
    for i in 0..n {
        let x = _mm256_set1_epi64x(buf[i] as i64);
        let mut acc = _mm256_setzero_si256();
        for c in 0..chunks {
            acc = _mm256_sub_epi64(acc, _mm256_cmpgt_epi64(x, _mm256_loadu_si256(bp.add(c))));
        }
        let h = _mm_add_epi64(_mm256_castsi256_si128(acc), _mm256_extracti128_si256::<1>(acc));
        let r = (_mm_cvtsi128_si64(h) + _mm_extract_epi64::<1>(h)) as usize;
        *out.get_unchecked_mut(r) = buf[i];
    }
    for k in 1..n {
        if out[k] == u64::MAX {
            out[k] = out[k - 1];
        }
    }
    v.copy_from_slice(&out[..n]);
}

fn sort_u64(v: &mut [u64]) {
    let n = v.len();
    if n <= 1 {
        return;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    if n <= RANK_MAX {
        unsafe { rank_sort(v) };
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
    if n >= RADIX_MIN {
        let (mut ra, mut rb, mut hist) = (Vec::new(), Vec::new(), vec![0u32; 3 << RADIX_BITS]);
        let sorted = radix_by_ts(v, &mut ra, &mut rb, &mut hist);
        let mut out: Vec<u64> = Vec::with_capacity(n);
        out.extend(sorted.iter().map(|&a| unsafe { *v.get_unchecked(a as u32 as usize) }));
        v.copy_from_slice(&out);
        return;
    }
    v.sort_unstable();
}
const RADIX_MIN: usize = 1024;

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
    // Borrowed fields, no per-row allocation: this runs before any thread starts.
    let prod_data = std::fs::read(data_dir.join("products.csv")).unwrap_or_default();
    let mut prod_rows: Vec<(&[u8], &[u8])> = Vec::new();
    if let Some(nl) = memchr(b'\n', &prod_data) {
        csv::for_each_line_fast::<2, _>(&prod_data[nl + 1..], |f| prod_rows.push((f[0], f[1])));
    }
    // Each row's category as an index into `cats` (first-seen order), then
    // remapped to its rank in byte order.
    let mut cats: Vec<&[u8]> = Vec::new();
    let mut row_cat: Vec<u8> = Vec::with_capacity(prod_rows.len());
    {
        let mut last = usize::MAX;
        for &(_, c) in &prod_rows {
            if last == usize::MAX || cats[last] != c {
                last = match cats.iter().position(|&x| x == c) {
                    Some(i) => i,
                    None => {
                        cats.push(c);
                        if cats.len() > 255 {
                            eprintln!("too many categories for this build");
                            std::process::exit(2);
                        }
                        cats.len() - 1
                    }
                };
            }
            row_cat.push(last as u8);
        }
    }
    let mut cat_names: Vec<Vec<u8>> = cats.iter().map(|c| c.to_vec()).collect();
    cat_names.push(b"unknown".to_vec());
    cat_names.sort();
    cat_names.dedup();
    if cat_names.len() > 255 {
        eprintln!("too many categories for this build");
        std::process::exit(2);
    }
    let cat_rank = |c: &[u8]| -> u32 { cat_names.binary_search_by(|x| x.as_slice().cmp(c)).unwrap() as u32 };
    let unknown_rank = cat_rank(b"unknown");
    let rank_of: Vec<u32> = cats.iter().map(|c| cat_rank(c)).collect();
    let mut products = Dim::new(b"p");
    for (&(p, _), &ci) in prod_rows.iter().zip(&row_cat) {
        products.insert(p, rank_of[ci as usize]);
    }
    drop(prod_rows);
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

    // Seed of the long-ID hash; a run that finds a hash collision restarts
    // itself with the next seed (see `restart_on_collision`).
    let id_seed: u64 = std::env::var("SS_ID_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let pbits: u32 = std::env::var("SS_PBITS").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
    let nparts = 1usize << pbits;
    let ctx = Ctx {
        w0,
        w1,
        base,
        ingest_base_secs: w0.div_euclid(1000),
        products,
        campaigns,
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
    let (links, link_ids): (Vec<Vec<(u64, u64)>>, Vec<LocalIds>) = run_parallel(threads, |_| {
        let mut out = Vec::new();
        let mut buf = Vec::new();
        let mut ids = LocalIds::new(id_seed);
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(c) = link_chunks.get(i) else { break };
            parse_links(load_chunk(&input, c, &mut buf), &mut out, &mut ids);
        }
        (out, ids)
    })
    .into_iter()
    .unzip();
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
    // strings of the long IDs seen so far (links first: the link map needs them)
    let long_names: std::sync::Mutex<Option<LongNames>> = std::sync::Mutex::new(None);
    #[allow(clippy::type_complexity)]
    let parsed: Vec<(Vec<BlockList<Cand>>, Arena, Vec<BlockList<Cl>>, Arena, LocalIds)> = run_parallel(threads, |t| {
        if t == 0 {
            let mut g = LongNames::default();
            for l in &link_ids {
                g.merge(l);
            }
            let m = build_links(&links, &g);
            *long_names.lock().unwrap() = Some(g);
            let _ = lm_cell.set(m);
            timer.mark("link map built");
        }
        let mut ids = LocalIds::new(id_seed);
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
                parse_events(buf, &ctx, &mut cands, &mut arena, &mut ids);
            } else {
                clicks.clear();
                parse_clicks(buf, &ctx, &mut clicks, &mut ids);
                let lm = loop {
                    if let Some(m) = lm_cell.get() {
                        break m;
                    }
                    std::thread::yield_now();
                };
                for (j, c) in clicks.iter().enumerate() {
                    if let Some(n) = clicks.get(j + 16) {
                        lm.prefetch_filter(n.user);
                    }
                    if let Some(n) = clicks.get(j + 8) {
                        if lm.maybe(n.user) {
                            prefetch(&lm.slots[lm.slot_of(n.user)]);
                        }
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
        (cands, arena, clp, carena, ids)
    });
    let mut cand_parts: Vec<Vec<BlockList<Cand>>> = Vec::with_capacity(threads);
    let mut cand_arenas: Vec<Arena> = Vec::with_capacity(threads);
    let mut cl_parts: Vec<Vec<BlockList<Cl>>> = Vec::with_capacity(threads);
    let mut click_arenas: Vec<Arena> = Vec::with_capacity(threads);
    let mut local_ids = Vec::with_capacity(threads);
    for (c, a, k, ka, ids) in parsed {
        cand_parts.push(c);
        cand_arenas.push(a);
        cl_parts.push(k);
        click_arenas.push(ka);
        local_ids.push(ids);
    }
    let linkmap = lm_cell.into_inner().unwrap();
    drop(links);
    drop(link_ids);
    // The per-thread string tables are merged beside phase 3; only phase 4
    // needs the names.
    let mut names_acc = long_names.into_inner().unwrap().unwrap();
    let names_job = std::thread::spawn(move || {
        for l in &local_ids {
            names_acc.merge(l);
        }
        drop(local_ids);
        names_acc
    });
    let eid_collision = std::sync::atomic::AtomicBool::new(false);
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
        let mut bptrs: Vec<*mut Cand> = Vec::new();
        let mut batch: Vec<(u64, u64)> = Vec::with_capacity(64);
        let flush = |batch: &mut Vec<(u64, u64)>, evp: &mut Vec<BlockList<Ev>>, arena: &mut Arena| {
            for &(u, _) in batch.iter() {
                linkmap.prefetch_filter(u);
            }
            let mut linked = 0u64; // bit i: batch[i] may be linked
            for (i, &(u, _)) in batch.iter().enumerate() {
                if linkmap.maybe(u) {
                    linked |= 1 << i;
                    prefetch(&linkmap.slots[linkmap.slot_of(u)]);
                }
            }
            for (i, &(u, key)) in batch.iter().enumerate() {
                let person = if linked >> i & 1 != 0 { linkmap.get_slow(u) } else { u };
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
                // This pass writes the `valid` flag of losing copies (partition
                // b belongs to this task alone). Writes go through the raw arena
                // pointers; a copy is never read through `ch` after its write.
                for p in cand_parts.iter() {
                    for (bp, len) in p[b].raw_chunks() {
                        let bn = bptrs.len() as u32;
                        bptrs.push(bp);
                        let ch = unsafe { std::slice::from_raw_parts(bp as *const Cand, len) };
                        for (off, c) in ch.iter().enumerate() {
                            let (c_eid, c_ingest) = (c.eid, c.ingest);
                            let me = ((bn << CSHIFT) | off as u32) + 1;
                            let hh = mixh(c_eid);
                            let fp = ((hh >> 16) as u32 as u64) << 32;
                            let mut h = (hh >> (64 - bits)) as usize;
                            loop {
                                let e = unsafe { ftab.get_unchecked_mut(h) };
                                if *e == 0 {
                                    *e = fp | me as u64;
                                    break;
                                }
                                if *e & !0xFFFF_FFFF == fp {
                                    let op = cand_at(&bptrs, *e as u32 - 1);
                                    let (o_eid, o_ingest, o_valid) = unsafe { ((*op).eid, (*op).ingest, (*op).valid) };
                                    if o_eid == c_eid {
                                        if (o_valid ^ c.valid) >> 1 != 0 {
                                            // equal 63-bit codes, different event IDs
                                            eid_collision.store(true, Ordering::Relaxed);
                                        }
                                        // The losing copy is marked invalid, so the
                                        // emit pass below reads candidates in order.
                                        let loser = if c_ingest < o_ingest {
                                            *e = fp | me as u64;
                                            op
                                        } else {
                                            unsafe { bp.add(off) }
                                        };
                                        unsafe { (*loser).valid &= !1 };
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
                // Every copy that lost dedup is marked invalid now: emit the
                // valid candidates in order.
                for p in cand_parts.iter() {
                    for c in p[b].iter() {
                        if c.valid & 1 == 0 {
                            continue;
                        }
                        batch.push((c.user, ((c.ts as u64) << 32) | c.pc as u64));
                        if batch.len() == 32 {
                            flush(&mut batch, &mut evp, &mut arena);
                        }
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
    let names = names_job.join().unwrap();
    if names.collision || eid_collision.load(Ordering::Relaxed) {
        restart_on_collision(id_seed);
    }
    let ncat = cat_names.len();
    let agg = Agg { ndays, ncat, day0, base, organic };
    // Largest partitions first: one holding a whale or a shared-device person
    // then cannot be picked up last and hold up the whole phase.
    let mut order: Vec<(usize, u32)> = (0..nq)
        .map(|q| (ev_parts.iter().map(|p| p[q].len()).sum::<usize>() + cl_parts.iter().map(|p| p[q].len()).sum::<usize>(), q as u32))
        .collect();
    order.sort_unstable_by(|a, b| b.cmp(a));
    let next = AtomicUsize::new(0);
    let results: Vec<(UserRows, Vec<[i64; 4]>)> = run_parallel(threads, |_| {
        let mut s = Scratch {
            tbits: 0,
            table: Vec::new(),
            persons: Vec::new(),
            ecount: Vec::new(),
            eperson: Vec::new(),
            ekey: Vec::new(),
            ra: Vec::new(),
            rb: Vec::new(),
            hist: vec![0; 3 << RADIX_BITS],
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
            camp_snap: vec![0; ncamp + 1],
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
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(&(_, q)) = order.get(i) else { break };
            process_partition(q as usize, &ev_parts, &cl_parts, &mut s, &agg, &names, &mut camp, &mut rows);
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

/// Two different IDs got the same 63-bit code (odds around 1e-6 per run):
/// nothing has been written yet, so run again with another hash seed.
#[cold]
fn restart_on_collision(seed: u64) -> ! {
    eprintln!("long-ID hash collision with seed {seed}; restarting with seed {}", seed + 1);
    assert!(seed < 16, "hash collisions with 16 seeds");
    let exe = std::env::current_exe().expect("current_exe");
    let status = std::process::Command::new(exe)
        .args(std::env::args_os().skip(1))
        .env("SS_ID_SEED", (seed + 1).to_string())
        .status()
        .expect("restart");
    std::process::exit(status.code().unwrap_or(1));
}

fn write_user(path: &Path, rows: UserRows, names: &LongNames, cat_names: &[Vec<u8>]) {
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

#[cfg(test)]
mod link_tests {
    use super::*;

    /// Union-find over a `HashMap`, naming each person by its smallest code.
    fn reference(edges: &[(u64, u64)]) -> HashMap<u64, u64> {
        fn root(p: &mut HashMap<u64, u64>, x: u64) -> u64 {
            let mut r = x;
            while p[&r] != r {
                r = p[&r];
            }
            r
        }
        let mut p: HashMap<u64, u64> = HashMap::new();
        for &(a, b) in edges {
            p.entry(a).or_insert(a);
            p.entry(b).or_insert(b);
            let (ra, rb) = (root(&mut p, a), root(&mut p, b));
            if ra != rb {
                // the smaller code becomes the root, so roots are the names
                if ra < rb { p.insert(rb, ra) } else { p.insert(ra, rb) };
            }
        }
        let keys: Vec<u64> = p.keys().copied().collect();
        keys.into_iter().map(|k| (k, root(&mut p, k))).collect()
    }

    #[test]
    fn build_links_matches_reference_and_grows() {
        let names = LongNames::default();
        let mut x: u64 = 0x9E3779B97F4A7C15;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for round in 0..40 {
            // few repeats: far more IDs than the 4/3-per-edge estimate, so the
            // table must grow; spread over several input vectors
            let n_edges = [0usize, 1, 5, 300, 5000, 70_000][round % 6];
            let pool = 1 + (rnd() % (3 * n_edges as u64 + 2));
            let mut parts: Vec<Vec<(u64, u64)>> = vec![Vec::new(); 1 + round % 4];
            let mut all = Vec::new();
            for i in 0..n_edges {
                let a = 1 + rnd() % pool;
                let b = match rnd() % 4 {
                    0 => a,                    // self link
                    1 => 1 + (a + 1) % pool,   // chains
                    _ => 1 + rnd() % pool,
                };
                let np = parts.len();
                parts[i % np].push((a, b));
                all.push((a, b));
            }
            let m = if round % 2 == 0 { build_links(&parts, &names) } else { build_links_from(&parts, &names, 4) };
            let want = reference(&all);
            for (&k, &v) in &want {
                assert_eq!(m.get(k), v, "round {round} key {k}");
            }
            for _ in 0..1000 {
                let k = pool + 1 + rnd() % 1_000_000; // never linked
                assert_eq!(m.get(k), k);
            }
        }
    }
}
