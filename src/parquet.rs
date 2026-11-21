//! A minimal Parquet writer: one row group, one uncompressed PLAIN data page
//! (v1) per column. Enough for the two output tables.

use std::io::Write;

pub enum Values {
    /// Required UTF8 strings: PLAIN-encoded bytes (u32 length + bytes each).
    Str(Vec<u8>),
    /// Optional UTF8 strings: PLAIN bytes of the present values, and one
    /// definition flag per row.
    OptStr(Vec<u8>, Vec<bool>),
    Date(Vec<i32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

pub struct Column {
    pub name: &'static str,
    pub values: Values,
}

pub fn push_str(buf: &mut Vec<u8>, s: &[u8]) {
    buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
    buf.extend_from_slice(s);
}

// ---- thrift compact protocol ----
struct T {
    b: Vec<u8>,
    last: Vec<i16>,
}

const CT_I32: u8 = 5;
const CT_I64: u8 = 6;
const CT_BINARY: u8 = 8;
const CT_LIST: u8 = 9;
const CT_STRUCT: u8 = 12;

impl T {
    fn new() -> Self {
        T { b: Vec::new(), last: vec![0] }
    }
    fn varint(&mut self, mut v: u64) {
        loop {
            if v < 0x80 {
                self.b.push(v as u8);
                return;
            }
            self.b.push((v as u8 & 0x7f) | 0x80);
            v >>= 7;
        }
    }
    fn zz(&mut self, v: i64) {
        self.varint(((v << 1) ^ (v >> 63)) as u64);
    }
    fn field(&mut self, id: i16, ty: u8) {
        let last = *self.last.last().unwrap();
        let delta = id - last;
        if delta > 0 && delta <= 15 {
            self.b.push(((delta as u8) << 4) | ty);
        } else {
            self.b.push(ty);
            self.zz(id as i64);
        }
        *self.last.last_mut().unwrap() = id;
    }
    fn i32f(&mut self, id: i16, v: i32) {
        self.field(id, CT_I32);
        self.zz(v as i64);
    }
    fn i64f(&mut self, id: i16, v: i64) {
        self.field(id, CT_I64);
        self.zz(v);
    }
    fn strf(&mut self, id: i16, s: &[u8]) {
        self.field(id, CT_BINARY);
        self.varint(s.len() as u64);
        self.b.extend_from_slice(s);
    }
    fn begin_struct_field(&mut self, id: i16) {
        self.field(id, CT_STRUCT);
        self.last.push(0);
    }
    fn begin_struct(&mut self) {
        self.last.push(0);
    }
    fn end_struct(&mut self) {
        self.b.push(0);
        self.last.pop();
    }
    fn list_header(&mut self, id: i16, elem: u8, n: usize) {
        self.field(id, CT_LIST);
        if n < 15 {
            self.b.push(((n as u8) << 4) | elem);
        } else {
            self.b.push(0xf0 | elem);
            self.varint(n as u64);
        }
    }
}

// Parquet enums
const TYPE_INT32: i32 = 1;
const TYPE_INT64: i32 = 2;
const TYPE_BYTE_ARRAY: i32 = 6;
const REP_REQUIRED: i32 = 0;
const REP_OPTIONAL: i32 = 1;
const ENC_PLAIN: i32 = 0;
const ENC_RLE: i32 = 3;
const CONV_UTF8: i32 = 0;
const CONV_DATE: i32 = 6;

fn def_levels(defs: &[bool]) -> Vec<u8> {
    // RLE/bit-packed hybrid, bit width 1, all bit-packed, with 4-byte length prefix.
    let groups = defs.len().div_ceil(8);
    let mut body = Vec::with_capacity(groups + 8);
    if groups > 0 {
        let mut hdr = ((groups as u64) << 1) | 1;
        loop {
            if hdr < 0x80 {
                body.push(hdr as u8);
                break;
            }
            body.push((hdr as u8 & 0x7f) | 0x80);
            hdr >>= 7;
        }
        for g in 0..groups {
            let mut byte = 0u8;
            for i in 0..8 {
                let k = g * 8 + i;
                if k < defs.len() && defs[k] {
                    byte |= 1 << i;
                }
            }
            body.push(byte);
        }
    }
    let mut out = Vec::with_capacity(body.len() + 4);
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

fn as_bytes<X: Copy>(v: &[X]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

pub fn write_file(path: &std::path::Path, num_rows: usize, cols: &[Column]) -> std::io::Result<()> {
    let f = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::with_capacity(1 << 22, f);
    w.write_all(b"PAR1")?;
    let mut offset: i64 = 4;
    // (type, data_page_offset, total size, uses_rle)
    let mut metas: Vec<(i32, i64, i64, bool)> = Vec::new();
    for c in cols {
        let (ty, def, data): (i32, Option<Vec<u8>>, &[u8]) = match &c.values {
            Values::Str(b) => (TYPE_BYTE_ARRAY, None, &b[..]),
            Values::OptStr(b, d) => (TYPE_BYTE_ARRAY, Some(def_levels(d)), &b[..]),
            Values::Date(v) | Values::I32(v) => (TYPE_INT32, None, as_bytes(v)),
            Values::I64(v) => (TYPE_INT64, None, as_bytes(v)),
        };
        let page_size = data.len() + def.as_ref().map_or(0, |d| d.len());
        let mut h = T::new();
        h.i32f(1, 0); // DATA_PAGE
        h.i32f(2, page_size as i32);
        h.i32f(3, page_size as i32);
        h.begin_struct_field(5);
        h.i32f(1, num_rows as i32);
        h.i32f(2, ENC_PLAIN);
        h.i32f(3, ENC_RLE);
        h.i32f(4, ENC_RLE);
        h.end_struct();
        h.b.push(0); // end PageHeader
        w.write_all(&h.b)?;
        if let Some(d) = &def {
            w.write_all(d)?;
        }
        w.write_all(data)?;
        let total = (h.b.len() + page_size) as i64;
        metas.push((ty, offset, total, def.is_some()));
        offset += total;
    }

    let mut m = T::new();
    m.i32f(1, 1); // version
    m.list_header(2, CT_STRUCT, cols.len() + 1);
    // root
    m.begin_struct();
    m.strf(4, b"schema");
    m.i32f(5, cols.len() as i32);
    m.end_struct();
    for c in cols {
        m.begin_struct();
        let (ty, rep, conv, logical) = match &c.values {
            Values::Str(_) => (TYPE_BYTE_ARRAY, REP_REQUIRED, Some(CONV_UTF8), Some(1i16)),
            Values::OptStr(..) => (TYPE_BYTE_ARRAY, REP_OPTIONAL, Some(CONV_UTF8), Some(1)),
            Values::Date(_) => (TYPE_INT32, REP_REQUIRED, Some(CONV_DATE), Some(6)),
            Values::I32(_) => (TYPE_INT32, REP_REQUIRED, None, None),
            Values::I64(_) => (TYPE_INT64, REP_REQUIRED, None, None),
        };
        m.i32f(1, ty);
        m.i32f(3, rep);
        m.strf(4, c.name.as_bytes());
        if let Some(cv) = conv {
            m.i32f(6, cv);
        }
        if let Some(l) = logical {
            m.begin_struct_field(10); // logicalType union
            m.begin_struct_field(l); // StringType / DateType: empty struct
            m.end_struct();
            m.end_struct();
        }
        m.end_struct();
    }
    m.i64f(3, num_rows as i64);
    if num_rows == 0 {
        m.list_header(4, CT_STRUCT, 0);
    } else {
        m.list_header(4, CT_STRUCT, 1);
        m.begin_struct(); // RowGroup
        m.list_header(1, CT_STRUCT, cols.len());
        let mut total_bytes = 0i64;
        for (c, &(ty, page_off, total, rle)) in cols.iter().zip(&metas) {
            total_bytes += total;
            m.begin_struct(); // ColumnChunk
            m.i64f(2, page_off);
            m.begin_struct_field(3); // ColumnMetaData
            m.i32f(1, ty);
            if rle {
                m.list_header(2, CT_I32, 2);
                m.zz(ENC_PLAIN as i64);
                m.zz(ENC_RLE as i64);
            } else {
                m.list_header(2, CT_I32, 1);
                m.zz(ENC_PLAIN as i64);
            }
            m.list_header(3, CT_BINARY, 1);
            m.varint(c.name.len() as u64);
            m.b.extend_from_slice(c.name.as_bytes());
            m.i32f(4, 0); // UNCOMPRESSED
            m.i64f(5, num_rows as i64);
            m.i64f(6, total);
            m.i64f(7, total);
            m.i64f(9, page_off);
            m.end_struct();
            m.end_struct();
        }
        m.i64f(2, total_bytes);
        m.i64f(3, num_rows as i64);
        m.end_struct();
    }
    m.strf(6, b"shopping-sessions");
    m.b.push(0);
    w.write_all(&m.b)?;
    w.write_all(&(m.b.len() as u32).to_le_bytes())?;
    w.write_all(b"PAR1")?;
    w.flush()?;
    Ok(())
}
