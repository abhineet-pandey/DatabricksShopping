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
    let mut c: u64 = 0;
    let mut bad = 0u8;
    for &b in s {
        let v = CHAR[b as usize];
        bad |= (v == 0) as u8;
        c = c * 37 + v as u64;
    }
    if bad != 0 {
        return None;
    }
    Some(c * POW37[12 - n])
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
}
