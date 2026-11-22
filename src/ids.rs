//! Exact identifier codes.
//!
//! A string of 1..=12 bytes from `[0-9a-z]` maps to `sum (v(c_i)+1) * 37^(11-i)`
//! with `v('0')=0 .. v('z')=35` and the missing tail padded with 0. The map is
//! injective, the codes are below 2^63, and integer order equals byte order of
//! the strings. Any other string is interned and gets `FALLBACK | index`. Codes
//! are never 0, so 0 can mark an empty hash slot.

use std::collections::HashMap;
use std::sync::Mutex;

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

#[derive(Default)]
pub struct InternerInner {
    map: HashMap<Box<[u8]>, u64>,
    pub strings: Vec<Box<[u8]>>,
}

#[derive(Default)]
pub struct Interner {
    pub inner: Mutex<InternerInner>,
}

impl Interner {
    #[cold]
    pub fn intern(&self, s: &[u8]) -> u64 {
        let mut g = self.inner.lock().unwrap();
        if let Some(&c) = g.map.get(s) {
            return c;
        }
        let c = FALLBACK | g.strings.len() as u64;
        g.strings.push(s.into());
        g.map.insert(s.into(), c);
        c
    }

    #[inline(always)]
    pub fn code(&self, s: &[u8]) -> u64 {
        match fast_code(s) {
            Some(c) => c,
            None => self.intern(s),
        }
    }

    pub fn bytes_of(&self, code: u64, out: &mut Vec<u8>) {
        if code & FALLBACK == 0 {
            decode_fast(code, out);
        } else {
            let g = self.inner.lock().unwrap();
            out.extend_from_slice(&g.strings[(code & !FALLBACK) as usize]);
        }
    }

    /// Byte-wise comparison of the strings behind two codes.
    pub fn cmp(&self, a: u64, b: u64) -> std::cmp::Ordering {
        if (a | b) & FALLBACK == 0 {
            return a.cmp(&b);
        }
        let mut x = Vec::new();
        let mut y = Vec::new();
        self.bytes_of(a, &mut x);
        self.bytes_of(b, &mut y);
        x.cmp(&y)
    }
}

/// Read-only view after parsing finished.
pub struct Names {
    pub strings: Vec<Box<[u8]>>,
}

impl Names {
    #[inline]
    pub fn bytes_of(&self, code: u64, out: &mut Vec<u8>) {
        if code & FALLBACK == 0 {
            decode_fast(code, out);
        } else {
            out.extend_from_slice(&self.strings[(code & !FALLBACK) as usize]);
        }
    }
    pub fn cmp(&self, a: u64, b: u64) -> std::cmp::Ordering {
        if (a | b) & FALLBACK == 0 {
            return a.cmp(&b);
        }
        let mut x = Vec::new();
        let mut y = Vec::new();
        self.bytes_of(a, &mut x);
        self.bytes_of(b, &mut y);
        x.cmp(&y)
    }
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
