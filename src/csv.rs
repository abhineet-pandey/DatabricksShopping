//! Strict CSV dialect from rules.md, scanned 64 bytes at a time.
//!
//! Only `\n` ends a line. Every `,` separates fields. A field is either free of
//! quotes, or wrapped in exactly one pair of quotes with no quote inside. Any
//! other quote, a wrong field count, or an empty line makes the line malformed.

const NL: u8 = 10;

#[inline(always)]
fn masks(block: &[u8; 64]) -> (u64, u64) {
    // returns (structural = ',' | '\n', quotes)
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        #[cfg(target_feature = "avx2")]
        {
            let p = block.as_ptr() as *const __m256i;
            let a = _mm256_loadu_si256(p);
            let b = _mm256_loadu_si256(p.add(1));
            let comma = _mm256_set1_epi8(b',' as i8);
            let nl = _mm256_set1_epi8(b'\n' as i8);
            let q = _mm256_set1_epi8(b'"' as i8);
            let sa = _mm256_or_si256(_mm256_cmpeq_epi8(a, comma), _mm256_cmpeq_epi8(a, nl));
            let sb = _mm256_or_si256(_mm256_cmpeq_epi8(b, comma), _mm256_cmpeq_epi8(b, nl));
            let s = (_mm256_movemask_epi8(sa) as u32 as u64) | ((_mm256_movemask_epi8(sb) as u32 as u64) << 32);
            let qa = _mm256_movemask_epi8(_mm256_cmpeq_epi8(a, q)) as u32 as u64;
            let qb = _mm256_movemask_epi8(_mm256_cmpeq_epi8(b, q)) as u32 as u64;
            return (s, qa | (qb << 32));
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            let p = block.as_ptr() as *const __m128i;
            let comma = _mm_set1_epi8(b',' as i8);
            let nl = _mm_set1_epi8(b'\n' as i8);
            let q = _mm_set1_epi8(b'"' as i8);
            let mut s = 0u64;
            let mut qq = 0u64;
            for i in 0..4 {
                let v = _mm_loadu_si128(p.add(i));
                let sv = _mm_or_si128(_mm_cmpeq_epi8(v, comma), _mm_cmpeq_epi8(v, nl));
                s |= (_mm_movemask_epi8(sv) as u32 as u64) << (16 * i);
                qq |= (_mm_movemask_epi8(_mm_cmpeq_epi8(v, q)) as u32 as u64) << (16 * i);
            }
            return (s, qq);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut s = 0u64;
        let mut qq = 0u64;
        for (i, &c) in block.iter().enumerate() {
            if c == b',' || c == b'\n' {
                s |= 1 << i;
            }
            if c == b'"' {
                qq |= 1 << i;
            }
        }
        (s, qq)
    }
}

/// Experiment helper: counts structural characters using only the masks.
pub fn count_structural(buf: &[u8]) -> usize {
    let mut n = 0usize;
    let mut base = 0;
    while base + 64 <= buf.len() {
        let block = unsafe { &*(buf.as_ptr().add(base) as *const [u8; 64]) };
        let (s, q) = masks(block);
        n += (s.count_ones() + q.count_ones()) as usize;
        base += 64;
    }
    n
}

/// Calls `cb` with the decoded fields of every well-formed line with exactly
/// `N` fields in `buf` (N <= 8). `buf` must start at a line start. A final line
/// without a newline counts.
#[inline(always)]
pub fn for_each_line<'a, const N: usize, F: FnMut(&[&'a [u8]; N])>(buf: &'a [u8], mut cb: F) {
    assert!(N <= 8);
    let len = buf.len();
    if len == 0 {
        return;
    }
    let mut fields: [&'a [u8]; 8] = [&[]; 8];
    let mut nf: usize = 0; // fields seen in current line
    let mut fs: usize = 0; // current field start
    let mut qs: u32 = 0; // quote count before fs
    let mut ok = true;
    let mut qbase: u32 = 0; // quotes before current block
    let mut base = 0usize;
    let mut tmp: [u8; 64];
    let ptr = buf.as_ptr();

    // Handles the field [fs, p) ended by the structural char at p (or end),
    // without data-dependent branches.
    macro_rules! end_field {
        ($p:expr, $qp:expr) => {{
            let p: usize = $p;
            let qp: u32 = $qp;
            let t = qp - qs;
            let first = unsafe { *ptr.add(fs.min(len - 1)) };
            let last = unsafe { *ptr.add(p.saturating_sub(1)) };
            let quoted = (t == 2) & (first == b'"') & (last == b'"') & (p >= fs + 2);
            let good = (t == 0) | quoted;
            let q = quoted as usize;
            let (a, b) = if good { (fs + q, p - q) } else { (fs, fs) };
            fields[nf & 7] = unsafe { std::slice::from_raw_parts(ptr.add(a), b - a) };
            ok &= good;
            nf += 1;
        }};
    }
    macro_rules! end_line {
        () => {{
            if ok & (nf == N) {
                cb(unsafe { &*(fields.as_ptr() as *const [&'a [u8]; N]) });
            }
            nf = 0;
            ok = true;
        }};
    }

    while base < len {
        let block: &[u8; 64] = if base + 64 <= len {
            unsafe { &*(ptr.add(base) as *const [u8; 64]) }
        } else {
            tmp = [0u8; 64];
            tmp[..len - base].copy_from_slice(&buf[base..]);
            &tmp
        };
        let (mut s, qm) = masks(block);
        while s != 0 {
            let off = s.trailing_zeros();
            s &= s - 1;
            let p = base + off as usize;
            let qp = qbase + (qm & ((1u64 << off) - 1)).count_ones();
            end_field!(p, qp);
            if unsafe { *ptr.add(p) } == NL {
                end_line!();
            }
            fs = p + 1;
            qs = qp;
        }
        qbase += qm.count_ones();
        base += 64;
    }
    if fs < len || nf > 0 {
        end_field!(len, qbase);
        end_line!();
    }
}

/// (comma, newline, quote) masks of the 128 bytes at `p`.
#[inline(always)]
unsafe fn masks128(p: *const u8) -> (u128, u128, u128) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        #[cfg(target_feature = "avx2")]
        {
            let comma = _mm256_set1_epi8(b',' as i8);
            let nl = _mm256_set1_epi8(b'\n' as i8);
            let q = _mm256_set1_epi8(b'"' as i8);
            let (mut c, mut n, mut qq) = (0u128, 0u128, 0u128);
            for i in 0..4 {
                let v = _mm256_loadu_si256((p as *const __m256i).add(i));
                c |= (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, comma)) as u32 as u128) << (32 * i);
                n |= (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, nl)) as u32 as u128) << (32 * i);
                qq |= (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, q)) as u32 as u128) << (32 * i);
            }
            return (c, n, qq);
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            let comma = _mm_set1_epi8(b',' as i8);
            let nl = _mm_set1_epi8(b'\n' as i8);
            let q = _mm_set1_epi8(b'"' as i8);
            let (mut c, mut n, mut qq) = (0u128, 0u128, 0u128);
            for i in 0..8 {
                let v = _mm_loadu_si128((p as *const __m128i).add(i));
                c |= (_mm_movemask_epi8(_mm_cmpeq_epi8(v, comma)) as u32 as u128) << (16 * i);
                n |= (_mm_movemask_epi8(_mm_cmpeq_epi8(v, nl)) as u32 as u128) << (16 * i);
                qq |= (_mm_movemask_epi8(_mm_cmpeq_epi8(v, q)) as u32 as u128) << (16 * i);
            }
            return (c, n, qq);
        }
    }
    #[allow(unreachable_code)]
    {
        let (mut c, mut n, mut qq) = (0u128, 0u128, 0u128);
        for i in 0..128 {
            let b = *p.add(i);
            c |= ((b == b',') as u128) << i;
            n |= ((b == b'\n') as u128) << i;
            qq |= ((b == b'"') as u128) << i;
        }
        (c, n, qq)
    }
}

/// Same contract as `for_each_line`, faster on lines up to 127 bytes: each
/// line is split from one 128-byte window with straight-line code. A field is
/// quoted when it starts and ends with `"` (2+ bytes); the line is clean when
/// it has exactly `N - 1` commas and its quote count is twice the number of
/// quoted fields. Longer lines go through `for_each_line`.
#[inline(always)]
pub fn for_each_line_fast<'a, const N: usize, F: FnMut(&[&'a [u8]; N])>(buf: &'a [u8], mut cb: F) {
    assert!(N >= 2 && N <= 8);
    let len = buf.len();
    let ptr = buf.as_ptr();
    let mut tmp = [0u8; 128];
    let mut l = 0usize;
    while l < len {
        let rem = len - l;
        let (cm, nm, qm) = unsafe {
            if rem >= 128 {
                masks128(ptr.add(l))
            } else {
                tmp = [0u8; 128];
                tmp[..rem].copy_from_slice(&buf[l..]);
                masks128(tmp.as_ptr())
            }
        };
        let end: usize; // line length, excluding the '\n'
        let next: usize;
        if nm != 0 {
            end = nm.trailing_zeros() as usize;
            next = l + end + 1;
        } else if rem < 128 {
            end = rem; // final line without newline
            next = len;
        } else {
            // a line of 128+ bytes: hand it to the general scanner
            let e = match buf[l..].iter().position(|&x| x == NL) {
                Some(i) => l + i + 1,
                None => len,
            };
            for_each_line::<N, _>(&buf[l..e], &mut cb);
            l = e;
            continue;
        }
        let below = if end >= 128 { !0u128 } else { (1u128 << end) - 1 };
        let c = cm & below;
        if c.count_ones() as usize != N - 1 {
            l = next;
            continue;
        }
        let q = qm & below;
        let mut fields: [&'a [u8]; N] = [&[]; N];
        let (mut clo, mut chi) = (c as u64, (c >> 64) as u64);
        let mut fs = 0usize;
        let mut nq = 0u32;
        let lastb = len - 1;
        let mut i = 0;
        while i < N {
            let fe = if i + 1 < N {
                if clo != 0 {
                    let p = clo.trailing_zeros() as usize;
                    clo &= clo - 1;
                    p
                } else {
                    let p = 64 + chi.trailing_zeros() as usize;
                    chi &= chi.wrapping_sub(1);
                    p
                }
            } else {
                end
            };
            let (a, b) = (l + fs, l + fe);
            // branch-free quote strip; loads are clamped inside the buffer
            let first = unsafe { *ptr.add(a.min(lastb)) };
            let last = unsafe { *ptr.add(b.saturating_sub(1)) };
            let quoted = (b >= a + 2) & (first == b'"') & (last == b'"');
            nq += quoted as u32;
            let qd = quoted as usize;
            fields[i] = unsafe { std::slice::from_raw_parts(ptr.add(a + qd), b - a - 2 * qd) };
            fs = fe + 1;
            i += 1;
        }
        if q.count_ones() == 2 * nq {
            cb(&fields);
        }
        l = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn lines<const N: usize>(s: &str) -> Vec<Vec<String>> {
        let mut v = Vec::new();
        for_each_line::<N, _>(s.as_bytes(), |f| {
            v.push(f.iter().map(|x| String::from_utf8_lossy(x).to_string()).collect())
        });
        v
    }
    #[test]
    fn basic() {
        assert_eq!(lines::<2>("a,b\n\"c\",\"\"\n,\n"), vec![vec!["a", "b"], vec!["c", ""], vec!["", ""]]);
        assert_eq!(lines::<2>("a,b\n\nx\na,b,c\n\"a,b\na\"\",b\n\"a\"b,c\n\"\"\"\",x\nlast,one"), vec![vec!["a", "b"], vec!["last", "one"]]);
        assert_eq!(lines::<2>("a,b\r\n"), vec![vec!["a", "b\r"]]);
        assert_eq!(lines::<2>("a,"), vec![vec!["a", ""]]);
        assert_eq!(lines::<2>("\",x\n"), Vec::<Vec<String>>::new());
        let long = format!("{},{}\n{},\"{}\"\n", "x".repeat(100), "y".repeat(70), "z".repeat(63), "w".repeat(64));
        assert_eq!(lines::<2>(&long).len(), 2);
    }
    #[test]
    fn fast_matches_general() {
        let mut x: u64 = 0x9E3779B97F4A7C15;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let alpha = b"ab,,\"\n\r x";
        for round in 0..20000 {
            let n = (rnd() % 600) as usize;
            let mut buf = Vec::with_capacity(n);
            let mode = round % 3;
            while buf.len() < n {
                let r = rnd();
                if mode == 0 {
                    buf.push(alpha[(r % alpha.len() as u64) as usize]);
                } else {
                    // mostly well-formed lines of varying length
                    let nf = 2 + (r % 6) as usize;
                    for f in 0..nf {
                        let w = (rnd() % if mode == 1 { 20 } else { 60 }) as usize;
                        let qd = rnd() % 4 == 0;
                        if qd { buf.push(b'"'); }
                        for _ in 0..w { buf.push(b'a' + (rnd() % 3) as u8); }
                        if rnd() % 50 == 0 { buf.push(b'"'); }
                        if qd && rnd() % 20 != 0 { buf.push(b'"'); }
                        if f + 1 < nf { buf.push(b','); }
                    }
                    if rnd() % 30 != 0 { buf.push(b'\n'); }
                }
            }
            fn run<const N: usize>(buf: &[u8]) {
                let mut a = Vec::new();
                for_each_line::<N, _>(buf, |f| a.push(f.iter().map(|x| x.to_vec()).collect::<Vec<_>>()));
                let mut b = Vec::new();
                for_each_line_fast::<N, _>(buf, |f| b.push(f.iter().map(|x| x.to_vec()).collect::<Vec<_>>()));
                assert_eq!(a, b, "N={N} buf={:?}", String::from_utf8_lossy(buf));
            }
            run::<2>(&buf);
            run::<4>(&buf);
            run::<7>(&buf);
        }
    }
}

#[cfg(test)]
mod bench {
    #[test]
    #[ignore]
    fn split_speed() {
        let p = std::env::var("SPLIT_FILE").unwrap();
        let buf = std::fs::read(p).unwrap();
        let start = buf.iter().position(|&b| b == b'\n').unwrap() + 1;
        let b = &buf[start..];
        for _ in 0..3 {
            let t = std::time::Instant::now();
            let mut n = 0usize;
            super::for_each_line::<7, _>(b, |f| n += f[0].len() + f[6].len());
            let t1 = t.elapsed();
            let t = std::time::Instant::now();
            let mut m = 0usize;
            super::for_each_line_fast::<7, _>(b, |f| m += f[0].len() + f[6].len());
            let t2 = t.elapsed();
            assert_eq!(n, m);
            eprintln!("{} MB: general {:?}  fast {:?}", b.len() >> 20, t1, t2);
            let t = std::time::Instant::now();
            let mut x = 0u128;
            let mut i = 0;
            while i + 128 <= b.len() { let (c, n, q) = unsafe { super::masks128(b.as_ptr().add(i)) }; x ^= c ^ n ^ q; i += 128; }
            let t3 = t.elapsed();
            let t = std::time::Instant::now();
            let mut i = 0; let mut k = 0usize;
            while i + 128 <= b.len() { let (_c, n, _q) = unsafe { super::masks128(b.as_ptr().add(i)) }; let e = n.trailing_zeros() as usize; k += 1; i += e.min(127) + 1; }
            let t4 = t.elapsed();
            eprintln!("   masks stride128 {:?} ({})  line-chase {:?} ({k})", t3, x.count_ones(), t4);
        }
    }
}
