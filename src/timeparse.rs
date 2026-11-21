//! Timestamps (F1, F2, F3) and prices, ASCII digits only.

pub const DAY_MS: i64 = 86_400_000;

pub const fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * ((m + 9) % 12) + 2) / 5 + d - 1;
    era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
}

pub fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[inline(always)]
fn dig(b: u8) -> u32 {
    (b as u32).wrapping_sub(b'0' as u32)
}

#[inline(always)]
fn two(s: &[u8], i: usize) -> Option<i64> {
    let a = dig(s[i]);
    let b = dig(s[i + 1]);
    if a > 9 || b > 9 {
        return None;
    }
    Some((a * 10 + b) as i64)
}

#[inline(always)]
fn is_leap(y: i64) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}

/// Milliseconds since the epoch, or None for a bad timestamp. An F3 value too
/// big for i64 saturates to i64::MAX, which is outside every window.
#[inline]
pub fn parse_ts(s: &[u8]) -> Option<i64> {
    let n = s.len();
    if n == 0 {
        return None;
    }
    if (n == 20 || n == 25) && s[4] == b'-' {
        return parse_iso(s);
    }
    // F3: digits only.
    let mut v: i64 = 0;
    let mut sat = false;
    for &b in s {
        let d = dig(b);
        if d > 9 {
            return None;
        }
        if !sat {
            match v.checked_mul(10).and_then(|x| x.checked_add(d as i64)) {
                Some(x) => v = x,
                None => sat = true,
            }
        }
    }
    Some(if sat { i64::MAX } else { v })
}

#[inline]
pub fn parse_iso(s: &[u8]) -> Option<i64> {
    let n = s.len();
    if n != 20 && n != 25 {
        return None;
    }
    if s[4] != b'-' || s[7] != b'-' || s[10] != b'T' || s[13] != b':' || s[16] != b':' {
        return None;
    }
    let y = two(s, 0)? * 100 + two(s, 2)?;
    let mo = two(s, 5)?;
    let d = two(s, 8)?;
    let h = two(s, 11)?;
    let mi = two(s, 14)?;
    let se = two(s, 17)?;
    if !(1..=12).contains(&mo) || d < 1 || h > 23 || mi > 59 || se > 59 {
        return None;
    }
    let dim = match mo {
        2 => {
            if is_leap(y) {
                29
            } else {
                28
            }
        }
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    if d > dim {
        return None;
    }
    let mut secs = days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60 + se;
    if n == 20 {
        if s[19] != b'Z' {
            return None;
        }
    } else {
        let sign = match s[19] {
            b'+' => 1,
            b'-' => -1,
            _ => return None,
        };
        if s[22] != b':' {
            return None;
        }
        let oh = two(s, 20)?;
        let om = two(s, 23)?;
        if oh > 23 || om > 59 {
            return None;
        }
        secs -= sign * (oh * 3600 + om * 60);
    }
    Some(secs * 1000)
}

/// A purchase price: digits only, value 1..=10_000_000.
#[inline]
pub fn parse_price(s: &[u8]) -> Option<u32> {
    if s.is_empty() {
        return None;
    }
    let mut v: u64 = 0;
    for &b in s {
        let d = dig(b);
        if d > 9 {
            return None;
        }
        v = v * 10 + d as u64;
        if v > 10_000_000 {
            // keep scanning for non-digits is unnecessary: the value is bad either way
            return None;
        }
    }
    if v == 0 {
        None
    } else {
        Some(v as u32)
    }
}

/// Parses the manifest's simple JSON for one key's string value.
pub fn manifest_value(text: &str, key: &str) -> Option<i64> {
    let at = text.find(&format!("\"{key}\""))?;
    let rest = &text[at + key.len() + 2..];
    let colon = rest.find(':')?;
    let rest = &rest[colon + 1..];
    let q1 = rest.find('"')?;
    let rest = &rest[q1 + 1..];
    let q2 = rest.find('"')?;
    parse_ts(rest[..q2].as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn forms() {
        let f1 = parse_ts(b"2026-10-03T23:58:10Z").unwrap();
        assert_eq!(f1, 1791071890000);
        assert_eq!(parse_ts(b"2026-10-03T19:58:10-04:00"), Some(f1));
        assert_eq!(parse_ts(b"1791071890000"), Some(f1));
        assert_eq!(parse_ts(b"0001791071890000"), Some(f1));
        assert_eq!(parse_ts(b"2026-10-04T01:30:00+02:00"), parse_ts(b"2026-10-03T23:30:00Z"));
        assert_eq!(parse_ts(b"2026-10-03T23:30:00-00:00"), parse_ts(b"2026-10-03T23:30:00Z"));
        for bad in [
            &b""[..], b"2026-10-03T23:58:10z", b"2026-10-03t23:58:10Z", b"2026-10-03 23:58:10Z",
            b"2026-10-03T23:58:10", b"1791071890000.0", b"1.79e12", b"-5", b"2026-13-03T23:58:10Z",
            b"2026-02-30T23:58:10Z", b"2027-02-29T23:58:10Z", b"2026-10-03T24:58:10Z", b"2026-10-03T23:58:60Z",
            b"2026-10-03T23:58:10+24:00", b"2026-10-03T23:58:10+05:60", b"2026-10-03T23:58:10+0200",
            b"2026-10-03T23:58:10.000Z",
        ] {
            assert_eq!(parse_ts(bad), None, "{}", String::from_utf8_lossy(bad));
        }
        assert!(parse_ts(b"2028-02-29T00:00:00Z").is_some());
        assert_eq!(parse_ts(b"999999999999999999999"), Some(i64::MAX));
        assert_eq!(parse_price(b"0250"), Some(250));
        assert_eq!(parse_price(b"10000000"), Some(10_000_000));
        assert_eq!(parse_price(b"10000001"), None);
        assert_eq!(parse_price(b"0"), None);
        assert_eq!(parse_price(b"00"), None);
        assert_eq!(parse_price(b"1000\r"), None);
        let (y, m, d) = civil_from_days(days_from_civil(2028, 2, 29));
        assert_eq!((y, m, d), (2028, 2, 29));
    }
}

// ---------------------------------------------------------------- fast paths

#[inline(always)]
fn is_eight_digits(x: u64) -> bool {
    ((x & 0xF0F0_F0F0_F0F0_F0F0) | (((x.wrapping_add(0x0606_0606_0606_0606)) & 0xF0F0_F0F0_F0F0_F0F0) >> 4))
        == 0x3333_3333_3333_3333
}

#[inline(always)]
fn eight_digits(x: u64) -> u64 {
    let mut v = x.wrapping_sub(0x3030_3030_3030_3030);
    v = v.wrapping_mul(10).wrapping_add(v >> 8);
    let mask = 0x0000_00FF_0000_00FF;
    let mul1 = 0x000F_4240_0000_0064;
    let mul2 = 0x0000_2710_0000_0001;
    (((v & mask).wrapping_mul(mul1)).wrapping_add(((v >> 16) & mask).wrapping_mul(mul2))) >> 32
}

/// Digits only (any length), saturating to i64::MAX; None on any non-digit.
#[inline]
pub fn parse_digits(s: &[u8]) -> Option<i64> {
    let n = s.len();
    if n == 0 {
        return None;
    }
    if n <= 18 {
        let lead = n % 8;
        let mut v: u64 = 0;
        for &b in &s[..lead] {
            let d = (b as u64).wrapping_sub(b'0' as u64);
            if d > 9 {
                return None;
            }
            v = v * 10 + d;
        }
        for c in s[lead..].chunks_exact(8) {
            let x = u64::from_le_bytes(c.try_into().unwrap());
            if !is_eight_digits(x) {
                return None;
            }
            v = v * 100_000_000 + eight_digits(x);
        }
        return Some(v as i64);
    }
    let mut v: i64 = 0;
    let mut sat = false;
    for &b in s {
        let d = (b as u32).wrapping_sub(b'0' as u32);
        if d > 9 {
            return None;
        }
        if !sat {
            match v.checked_mul(10).and_then(|x| x.checked_add(d as i64)) {
                Some(x) => v = x,
                None => sat = true,
            }
        }
    }
    Some(if sat { i64::MAX } else { v })
}

// ---------------------------------------------------------------- SWAR F1/F2/F3

const Y0: i64 = 1900;
const NY: usize = 300;

const fn ym_table() -> [i32; NY * 12 + 1] {
    let mut t = [0i32; NY * 12 + 1];
    let mut i = 0;
    while i <= NY * 12 {
        t[i] = days_from_civil(Y0 + (i / 12) as i64, (i % 12) as i64 + 1, 1) as i32;
        i += 1;
    }
    t
}
/// Epoch day of the 1st of each month from 1900-01 on; one extra entry so
/// `YM[i + 1] - YM[i]` is the month's length.
static YM: [i32; NY * 12 + 1] = ym_table();

#[inline(always)]
fn load8(s: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(s[i..i + 8].try_into().unwrap())
}
#[inline(always)]
fn byte(x: u64, i: u32) -> i64 {
    ((x >> (8 * i)) & 0xFF) as i64
}

/// Same results as `parse_iso`, with SWAR validation and a month table.
#[inline(always)]
pub fn fast_iso(s: &[u8]) -> Option<i64> {
    let n = s.len();
    if n != 20 && n != 25 {
        return None;
    }
    let a = load8(s, 0); // YYYY-MM-
    let b = load8(s, 8); // DDThh:mm
    const AS: u64 = 0xFF00_00FF_0000_0000;
    const AV: u64 = (b'-' as u64) << 32 | (b'-' as u64) << 56;
    const BS: u64 = 0x0000_FF00_00FF_0000;
    const BV: u64 = (b'T' as u64) << 16 | (b':' as u64) << 40;
    if (a & AS) != AV || (b & BS) != BV || s[16] != b':' {
        return None;
    }
    let ad = (a & !AS) | (0x30u64 << 32) | (0x30u64 << 56);
    let bd = (b & !BS) | (0x30u64 << 16) | (0x30u64 << 40);
    if !is_eight_digits(ad) || !is_eight_digits(bd) {
        return None;
    }
    let s1 = dig(s[17]) as i64;
    let s2 = dig(s[18]) as i64;
    if s1 > 9 || s2 > 9 {
        return None;
    }
    let a0 = ad.wrapping_sub(0x3030_3030_3030_3030);
    let b0 = bd.wrapping_sub(0x3030_3030_3030_3030);
    let y = byte(a0, 0) * 1000 + byte(a0, 1) * 100 + byte(a0, 2) * 10 + byte(a0, 3);
    let mo = byte(a0, 5) * 10 + byte(a0, 6);
    let d = byte(b0, 0) * 10 + byte(b0, 1);
    let h = byte(b0, 3) * 10 + byte(b0, 4);
    let mi = byte(b0, 6) * 10 + byte(b0, 7);
    let se = s1 * 10 + s2;
    if mo < 1 || mo > 12 || d < 1 || h > 23 || mi > 59 || se > 59 {
        return None;
    }
    let yi = y - Y0;
    if yi < 0 || yi >= NY as i64 {
        return parse_iso(s);
    }
    let i = (yi * 12 + mo - 1) as usize;
    let start = YM[i] as i64;
    if d > YM[i + 1] as i64 - start {
        return None;
    }
    let mut secs = (start + d - 1) * 86_400 + h * 3600 + mi * 60 + se;
    if n == 20 {
        if s[19] != b'Z' {
            return None;
        }
    } else {
        let sign = match s[19] {
            b'+' => 1,
            b'-' => -1,
            _ => return None,
        };
        if s[22] != b':' {
            return None;
        }
        let oh = two(s, 20)?;
        let om = two(s, 23)?;
        if oh > 23 || om > 59 {
            return None;
        }
        secs -= sign * (oh * 3600 + om * 60);
    }
    Some(secs * 1000)
}

/// Same results as `parse_ts`.
#[inline(always)]
pub fn fast_ts(s: &[u8]) -> Option<i64> {
    let n = s.len();
    if (n == 20 || n == 25) && s[4] == b'-' {
        return fast_iso(s);
    }
    if n == 13 {
        let x1 = load8(s, 0);
        let x2 = load8(s, 5);
        if !is_eight_digits(x1) || !is_eight_digits(x2) {
            return None;
        }
        return Some((eight_digits(x1) * 100_000 + eight_digits(x2) % 100_000) as i64);
    }
    parse_digits(s)
}

#[cfg(test)]
mod fast_tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    #[test]
    fn fast_matches_slow() {
        let samples: Vec<&[u8]> = vec![
            b"2026-10-03T23:58:10Z", b"2026-10-03T19:58:10-04:00", b"1791071890000", b"0001791071890000",
            b"2026-02-30T23:58:10Z", b"2027-02-29T23:58:10Z", b"2028-02-29T23:58:10Z", b"2026-13-03T23:58:10Z",
            b"2026-10-03T24:58:10Z", b"2026-10-03T23:58:10+24:00", b"2026-10-03T23:58:10+0200", b"1.79e12",
            b"-5", b"", b"86400000", b"999999999999999999999", b"123456789012345678", b"1234567890123456789",
            b"12345678", b"1234567", b"2026-10-03t23:58:10Z", b"2026-10-03T23:58:10z", b"20261003T23:58:10Z0",
            b"2026-10-03T23:58:10.000Z", b"12345678901234567890123", b"0000000000000000000000001",
            b"1899-12-31T23:59:59Z", b"2200-01-01T00:00:00Z", b"0000-01-01T00:00:00Z", b"9999-12-31T23:59:59Z",
            b"2100-02-29T00:00:00Z", b"2000-02-29T00:00:00Z", b"2026-00-10T00:00:00Z", b"2026-10-00T00:00:00Z",
            b"2026-10-03T23:58:10-00:00", b"2026-10-03T23:58:10+23:59", b"179107189000a", b"a791071890000",
            b"1791071890000\r", b"2026-10-03T23:58:10Z\r",
        ];
        for s in &samples {
            assert_eq!(fast_ts(s), parse_ts(s), "{}", String::from_utf8_lossy(s));
            assert_eq!(fast_iso(s), parse_iso(s), "{}", String::from_utf8_lossy(s));
        }
        // random valid-looking timestamps with random single-byte corruption
        let mut r = Rng(0x1234_5678_9abc_def1);
        let alphabet = b"0123456789-:TZz+ .t\r9";
        for _ in 0..2_000_000 {
            let secs = r.below(400 * 366 * 86_400) as i64 - 50 * 366 * 86_400 + 0x7000_0000;
            let ms = secs * 1000 + r.below(1000) as i64;
            let day = secs.div_euclid(86_400);
            let (y, m, d) = civil_from_days(day);
            let t = secs.rem_euclid(86_400);
            let mut v: Vec<u8> = match r.below(4) {
                0 => format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", t / 3600, t / 60 % 60, t % 60).into_bytes(),
                1 => format!(
                    "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}{}{:02}:{:02}",
                    t / 3600, t / 60 % 60, t % 60,
                    if r.below(2) == 0 { '+' } else { '-' }, r.below(26), r.below(62)
                ).into_bytes(),
                2 => format!("{ms}").into_bytes(),
                _ => format!("{:0w$}", ms, w = 13 + r.below(6) as usize).into_bytes(),
            };
            if r.below(3) == 0 && !v.is_empty() {
                let i = r.below(v.len() as u64) as usize;
                v[i] = alphabet[r.below(alphabet.len() as u64) as usize];
            }
            if r.below(10) == 0 {
                // impossible calendar values
                if v.len() >= 20 && v[4] == b'-' {
                    let i = [5usize, 6, 8, 9, 11, 12, 14, 17][r.below(8) as usize];
                    v[i] = b'0' + r.below(10) as u8;
                }
            }
            assert_eq!(fast_ts(&v), parse_ts(&v), "{}", String::from_utf8_lossy(&v));
            assert_eq!(fast_iso(&v), parse_iso(&v), "{}", String::from_utf8_lossy(&v));
        }
    }
}
