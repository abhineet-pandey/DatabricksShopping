//! Exact identifier codes.
//!
//! A string of 1..=12 bytes from `[0-9a-z]` maps to `sum (v(c_i)+1) * 37^(11-i)`
//! with `v('0')=0 .. v('z')=35` and the missing tail padded with 0. The map is
//! injective, the codes are below 2^63, and integer order equals byte order of
//! the strings. Any other string (in v5 data, one ID in eight has a 32-char
//! hex suffix) gets `FALLBACK | 63 hash bits`, with its string kept in a
//! per-thread table; collisions are detected. Codes are never 0, so 0 can
//! mark an empty hash slot.

pub const FALLBACK: u64 = 1 << 63;

const fn pow37() -> [u64; 13] {
    let mut p = [1u64; 13];
    let mut i = 1;
    while i < 13 {
        p[i] = p[i - 1] * 37;
        i += 1;
    }
    p
}
pub const POW37: [u64; 13] = pow37();

const fn char_table() -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut c = 0;
    while c < 10 {
        t[b'0' as usize + c] = 1 + c as u8;
        c += 1;
    }
    let mut c = 0;
    while c < 26 {
        t[b'a' as usize + c] = 11 + c as u8;
        c += 1;
    }
    t
}
static CHAR: [u8; 256] = char_table();
const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";

#[inline(always)]
pub fn fast_code(s: &[u8]) -> Option<u64> {
    let n = s.len();
    if n == 0 || n > 12 {
        return None;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "ssse3", target_feature = "sse4.1"))]
    {
        // A 16-byte load that stays inside the page of `s` cannot fault; the
        // bytes past `n` are masked out and never affect the result.
        if (s.as_ptr() as usize & 4095) <= 4096 - 16 {
            return unsafe { simd_code(s.as_ptr(), n) };
        }
    }
    scalar_code(s)
}

/// `fast_code` for 1..=12 bytes with SSSE3/SSE4.1: characters map to 1..=36,
/// `maddubs`/`madd` fold each group of 4 into `sum v * 37^(3-j)` (< 2^21),
/// and the three groups combine as `g0 * 37^8 + g1 * 37^4 + g2`.
#[cfg(all(target_arch = "x86_64", target_feature = "ssse3", target_feature = "sse4.1"))]
#[inline(always)]
unsafe fn simd_code(p: *const u8, n: usize) -> Option<u64> {
    use std::arch::x86_64::*;
    let v = _mm_loadu_si128(p as *const __m128i);
    let idx = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let inlen = _mm_cmpgt_epi8(_mm_set1_epi8(n as i8), idx);
    let d = _mm_sub_epi8(v, _mm_set1_epi8(b'0' as i8));
    let l = _mm_sub_epi8(v, _mm_set1_epi8(b'a' as i8));
    let isd = _mm_cmpeq_epi8(_mm_min_epu8(d, _mm_set1_epi8(9)), d);
    let isl = _mm_cmpeq_epi8(_mm_min_epu8(l, _mm_set1_epi8(25)), l);
    let bad = _mm_andnot_si128(_mm_or_si128(isd, isl), inlen);
    if _mm_movemask_epi8(bad) != 0 {
        return None;
    }
    let val = _mm_or_si128(
        _mm_and_si128(isd, _mm_add_epi8(d, _mm_set1_epi8(1))),
        _mm_and_si128(isl, _mm_add_epi8(l, _mm_set1_epi8(11))),
    );
    let val = _mm_and_si128(val, inlen);
    let pairs = _mm_maddubs_epi16(val, _mm_set1_epi16(0x0125)); // bytes (37, 1)
    let quads = _mm_madd_epi16(pairs, _mm_set1_epi32(0x0001_0559)); // words (1369, 1)
    let g0 = _mm_cvtsi128_si32(quads) as u32 as u64;
    let g1 = _mm_extract_epi32::<1>(quads) as u32 as u64;
    let g2 = _mm_extract_epi32::<2>(quads) as u32 as u64;
    Some(g0 * POW37[8] + g1 * POW37[4] + g2)
}

#[inline(always)]
fn scalar_code(s: &[u8]) -> Option<u64> {
    let n = s.len();
    if n == 0 || n > 12 {
        return None;
    }
    // sum v_i * 37^(11-i): the multiplies are independent, so only the adds
    // form a dependency chain (same value as Horner's rule times 37^(12-n)).
    let mut c: u64 = 0;
    let mut bad = 0u8;
    for (i, &b) in s.iter().enumerate() {
        let v = CHAR[b as usize];
        bad |= (v == 0) as u8;
        c += v as u64 * POW37[11 - i];
    }
    if bad != 0 {
        return None;
    }
    Some(c)
}

/// Writes the string of a fast code.
pub fn decode_fast(mut code: u64, out: &mut Vec<u8>) {
    let mut buf = [0u8; 12];
    for i in (0..12).rev() {
        buf[i] = (code % 37) as u8;
        code /= 37;
    }
    for &d in &buf {
        if d == 0 {
            break;
        }
        out.push(DIGITS[(d - 1) as usize]);
    }
}

// ---------------------------------------------------------------- opaque IDs

/// Folded 64x64 -> 128 multiply.
#[inline(always)]
fn mum(a: u64, b: u64) -> u64 {
    let r = (a as u128) * (b as u128);
    (r as u64) ^ ((r >> 64) as u64)
}

/// Two independent 64-bit hashes of `s` (keyed by `seed`).
#[inline]
pub fn hash2(s: &[u8], seed: u64) -> (u64, u64) {
    const K1: u64 = 0xa076_1d64_78bd_642f;
    const K2: u64 = 0xe703_7ed1_a0b4_28db;
    const K3: u64 = 0x8ebc_6af0_9c88_c6e3;
    const K4: u64 = 0x5899_65cc_7537_4cc3;
    let n = s.len() as u64;
    let mut a = seed ^ K1 ^ n;
    let mut b = seed.rotate_left(32) ^ K2 ^ n.wrapping_mul(K3);
    let mut c = s.chunks_exact(8);
    for w in &mut c {
        let w = u64::from_le_bytes(w.try_into().unwrap());
        a = mum(a ^ w, K3);
        b = mum(b ^ w.rotate_left(29), K4);
    }
    let r = c.remainder();
    if !r.is_empty() {
        let mut t = [0u8; 8];
        t[..r.len()].copy_from_slice(r);
        let w = u64::from_le_bytes(t);
        a = mum(a ^ w, K3);
        b = mum(b ^ w.rotate_left(29), K4);
    }
    (mum(a ^ K2, K1 ^ n), mum(b ^ K1, K2 ^ n.rotate_left(17)))
}

/// Code of an ID that has no exact fast code: `FALLBACK` | 63 hash bits.
/// Distinct strings with equal codes are detected (see `LocalIds` and
/// `LongNames::merge`), and the run then restarts with another seed.
#[inline(always)]
pub fn long_code(h1: u64) -> u64 {
    FALLBACK | (h1 >> 1)
}

/// Open-addressing string table keyed by long codes over one byte arena.
/// A slot is (code, second fingerprint, offset); the arena holds each string
/// as a length byte pair (u16) followed by the bytes. No locks: one table per
/// thread, merged afterwards.
#[derive(Default)]
pub struct StrTable {
    slots: Vec<(u64, u32, u32)>,
    n: usize,
    bytes: Vec<u8>,
}

impl StrTable {
    #[inline(always)]
    fn home(&self, code: u64) -> usize {
        (code.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 20) as usize & (self.slots.len() - 1)
    }
    fn grow(&mut self) {
        let cap = (self.slots.len() * 2).max(1 << 12);
        let old = std::mem::replace(&mut self.slots, vec![(0, 0, 0); cap]);
        for e in old {
            if e.0 != 0 {
                let mut i = self.home(e.0);
                while self.slots[i].0 != 0 {
                    i = (i + 1) & (cap - 1);
                }
                self.slots[i] = e;
            }
        }
    }
    /// Makes room for `k` more entries without growing during the inserts.
    pub fn reserve(&mut self, k: usize) {
        while (self.n + k) * 2 >= self.slots.len() {
            self.grow();
        }
    }
    #[inline(always)]
    pub fn prefetch(&self, code: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let p = self.slots.as_ptr().add(self.home(code));
            std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
    }
    #[inline(always)]
    fn at(&self, off: u32) -> &[u8] {
        let o = off as usize;
        let n = u16::from_le_bytes([self.bytes[o], self.bytes[o + 1]]) as usize;
        &self.bytes[o + 2..o + 2 + n]
    }
    /// Inserts `s` under (`code`, `fp`) unless `code` is present. Returns
    /// false when `code` is present with another fingerprint (a collision).
    #[inline]
    pub fn insert_fp(&mut self, code: u64, fp: u32, s: &[u8]) -> bool {
        if self.n * 2 >= self.slots.len() {
            self.grow();
        }
        let m = self.slots.len() - 1;
        let mut i = self.home(code);
        loop {
            let e = unsafe { *self.slots.get_unchecked(i) };
            if e.0 == code {
                return e.1 == fp;
            }
            if e.0 == 0 {
                let off = self.bytes.len() as u32;
                let n = s.len().min(u16::MAX as usize);
                self.bytes.extend_from_slice(&(n as u16).to_le_bytes());
                self.bytes.extend_from_slice(&s[..n]);
                self.slots[i] = (code, fp, off);
                self.n += 1;
                return n == s.len();
            }
            i = (i + 1) & m;
        }
    }
    /// Inserts `s` unless present; false when `code` names another string
    /// (compared byte by byte).
    pub fn insert(&mut self, code: u64, fp: u32, s: &[u8]) -> bool {
        if let Some(t) = self.get(code) {
            return t == s;
        }
        self.insert_fp(code, fp, s)
    }
    #[inline]
    pub fn get(&self, code: u64) -> Option<&[u8]> {
        if self.slots.is_empty() {
            return None;
        }
        let m = self.slots.len() - 1;
        let mut i = self.home(code);
        loop {
            let e = unsafe { *self.slots.get_unchecked(i) };
            if e.0 == code {
                return Some(self.at(e.2));
            }
            if e.0 == 0 {
                return None;
            }
            i = (i + 1) & m;
        }
    }
    pub fn len(&self) -> usize {
        self.n
    }
    /// Every (code, fingerprint, string) entry.
    pub fn entries(&self) -> impl Iterator<Item = (u64, u32, &[u8])> + '_ {
        self.slots.iter().filter(|e| e.0 != 0).map(move |e| (e.0, e.1, self.at(e.2)))
    }
}

/// Per-thread ID coder: fast codes when possible, otherwise long codes whose
/// strings are kept (user, click and link IDs: names and tie-breaks need them).
/// Within a thread a repeated code is checked by a second 32-bit fingerprint;
/// across threads, by the bytes.
pub struct LocalIds {
    pub seed: u64,
    pub table: StrTable,
    pub collision: bool,
    /// Inserts waiting for `flush`: (code, fingerprint, string). The strings
    /// point into the caller's current input buffer, so `flush` must run
    /// before that buffer changes. Batching lets the table probes overlap.
    pending: Vec<(u64, u32, *const u8, usize)>,
}

// `pending` is empty whenever a LocalIds leaves its thread (after `flush`).
unsafe impl Send for LocalIds {}
unsafe impl Sync for LocalIds {}

const BATCH: usize = 32;

impl LocalIds {
    pub fn new(seed: u64) -> Self {
        LocalIds { seed, table: StrTable::default(), collision: false, pending: Vec::with_capacity(BATCH) }
    }
    #[inline(always)]
    pub fn code(&mut self, s: &[u8]) -> u64 {
        match fast_code(s) {
            Some(c) => c,
            None => self.long(s),
        }
    }
    #[inline(never)]
    fn long(&mut self, s: &[u8]) -> u64 {
        let (h1, h2) = hash2(s, self.seed);
        let c = long_code(h1);
        self.pending.push((c, (h2 >> 32) as u32, s.as_ptr(), s.len()));
        if self.pending.len() == BATCH {
            self.flush();
        }
        c
    }
    /// Performs the pending inserts. Must be called before the buffer that
    /// the strings passed to `code` live in is changed or freed.
    pub fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        self.table.reserve(self.pending.len());
        for &(c, ..) in &self.pending {
            self.table.prefetch(c);
        }
        for i in 0..self.pending.len() {
            let (c, fp, p, n) = self.pending[i];
            let s = unsafe { std::slice::from_raw_parts(p, n) };
            if !self.table.insert_fp(c, fp, s) {
                self.collision = true;
            }
        }
        self.pending.clear();
    }
    /// Event IDs: no string is kept. Returns the code and a 31-bit second
    /// fingerprint (0 for fast codes) that dedup compares when codes match.
    #[inline(always)]
    pub fn event_code(&self, s: &[u8]) -> (u64, u32) {
        match fast_code(s) {
            Some(c) => (c, 0),
            None => {
                let (h1, h2) = hash2(s, self.seed);
                (long_code(h1), (h2 >> 33) as u32)
            }
        }
    }
}

/// The strings of all long codes, after the per-thread tables are merged.
#[derive(Default)]
pub struct LongNames {
    pub table: StrTable,
    pub collision: bool,
}

impl LongNames {
    pub fn merge(&mut self, l: &LocalIds) {
        self.collision |= l.collision;
        for (c, fp, s) in l.table.entries() {
            if !self.table.insert(c, fp, s) {
                self.collision = true;
            }
        }
    }
}

/// Strings behind codes: fast codes decode, long codes are looked up.
pub trait IdNames {
    fn long_str(&self, code: u64) -> Option<&[u8]>;

    /// The string behind `code`; fast codes are decoded into `buf`.
    #[inline]
    fn str_of<'a>(&'a self, code: u64, buf: &'a mut [u8; 12]) -> &'a [u8] {
        if code & FALLBACK != 0 {
            return self.long_str(code).expect("long ID without its string");
        }
        let mut c = code;
        let mut d = [0u8; 12];
        for i in (0..12).rev() {
            d[i] = (c % 37) as u8;
            c /= 37;
        }
        let mut n = 0;
        while n < 12 && d[n] != 0 {
            buf[n] = DIGITS[(d[n] - 1) as usize];
            n += 1;
        }
        &buf[..n]
    }
    #[inline]
    fn bytes_of(&self, code: u64, out: &mut Vec<u8>) {
        let mut b = [0u8; 12];
        out.extend_from_slice(self.str_of(code, &mut b));
    }
    /// Byte-wise comparison of the strings behind two codes (no allocation).
    fn cmp(&self, a: u64, b: u64) -> std::cmp::Ordering {
        if (a | b) & FALLBACK == 0 {
            return a.cmp(&b);
        }
        let (mut x, mut y) = ([0u8; 12], [0u8; 12]);
        self.str_of(a, &mut x).cmp(self.str_of(b, &mut y))
    }
}

impl IdNames for LongNames {
    #[inline]
    fn long_str(&self, code: u64) -> Option<&[u8]> {
        self.table.get(code)
    }
}

/// The link IDs' strings plus every thread's own table, read in place: no
/// merge is needed to look names up.
pub struct NameView<'a> {
    pub links: &'a LongNames,
    pub locals: &'a [LocalIds],
}

impl IdNames for NameView<'_> {
    #[inline]
    fn long_str(&self, code: u64) -> Option<&[u8]> {
        self.links.table.get(code).or_else(|| self.locals.iter().find_map(|l| l.table.get(code)))
    }
}

/// True if some code names two different strings anywhere: within a thread
/// (fingerprints) or across the link table and the threads (bytes).
pub fn any_collision(links: &LongNames, locals: &[LocalIds]) -> bool {
    if links.collision || locals.iter().any(|l| l.collision) {
        return true;
    }
    let mut seen = StrTable::default();
    for l in locals {
        for (c, fp, s) in l.table.entries() {
            let ok = match links.table.get(c) {
                Some(t) => t == s,
                None => seen.insert(c, fp, s),
            };
            if !ok {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn order_and_roundtrip() {
        let words: Vec<&[u8]> = vec![b"0", b"00", b"1", b"a", b"a0", b"az", b"b", b"u1", b"u10", b"u2", b"zzzzzzzzzzzz", b"k1q2awe54"];
        for w in &words {
            let c = fast_code(w).unwrap();
            assert!(c != 0 && c < FALLBACK);
            let mut o = Vec::new();
            decode_fast(c, &mut o);
            assert_eq!(&o[..], *w);
        }
        for a in &words {
            for b in &words {
                assert_eq!(fast_code(a).unwrap().cmp(&fast_code(b).unwrap()), a.cmp(b));
            }
        }
        assert!(fast_code(b"U1").is_none());
        assert!(fast_code(b" u1").is_none());
        assert!(fast_code(b"").is_none());
        assert!(fast_code(b"aaaaaaaaaaaaa").is_none());
    }
    #[test]
    fn long_ids_roundtrip_and_detect_collisions() {
        let mut l = LocalIds::new(7);
        let a = l.code(b"u5786c0ad7acc294274177835cdbcc8aa");
        let b = l.code(b"u5786c0ad7acc294274177835cdbcc8ab");
        assert!(a & FALLBACK != 0 && b & FALLBACK != 0 && a != b);
        assert_eq!(l.code(b"u5786c0ad7acc294274177835cdbcc8aa"), a);
        assert_eq!(l.code(b"u1"), fast_code(b"u1").unwrap());
        let u = l.code(b"U1");
        l.flush();
        let mut g = LongNames::default();
        g.merge(&l);
        assert!(!g.collision);
        let mut o = Vec::new();
        g.bytes_of(u, &mut o);
        assert_eq!(&o[..], b"U1");
        assert_eq!(g.cmp(u, fast_code(b"a").unwrap()), std::cmp::Ordering::Less);
        // a forced collision: same code, different string
        let mut t = StrTable::default();
        assert!(t.insert(FALLBACK | 5, 1, b"x"));
        assert!(t.insert(FALLBACK | 5, 1, b"x"));
        assert!(!t.insert(FALLBACK | 5, 1, b"y"));
        assert!(!t.insert_fp(FALLBACK | 5, 2, b"x"));
        for i in 0..100_000u64 {
            assert!(t.insert(FALLBACK | (i + 10), 0, format!("id{i}").as_bytes()));
        }
        assert_eq!(t.get(FALLBACK | 99), Some(&b"id89"[..]));
    }
    #[test]
    fn simd_matches_scalar() {
        // bytes around the valid ranges, plus garbage after the string
        let alpha = b"0189:/`azAZ{ \"\x00\xff";
        let mut x: u64 = 0x2545F4914F6CDD1D;
        let mut buf = vec![0u8; 8192];
        for round in 0..400_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let n = (x % 17) as usize;
            let off = if round % 4 == 0 { 4096 - (x >> 8) as usize % 20 } else { (x >> 8) as usize % 4000 };
            for i in 0..32 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                buf[off + i] = if x % 3 == 0 { alpha[(x >> 5) as usize % alpha.len()] } else { b'a' + (x >> 9) as u8 % 26 };
            }
            let s = &buf[off..off + n];
            assert_eq!(fast_code(s), scalar_code(s), "{:?}", String::from_utf8_lossy(s));
        }
    }
}
