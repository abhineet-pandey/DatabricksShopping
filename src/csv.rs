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
}
