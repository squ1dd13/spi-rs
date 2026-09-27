//! Decode opacity samples after a record's colour stream.
//!
//! Alpha is one byte per pixel: 0 is transparent and 255 is opaque. The output uses straight alpha, so these bytes do not scale the colour planes. Selector 2 has no alpha stream and obtains 255 during composition. Selectors 4 and 5 use this module.
//!
//! [`decode_slice`] rounds the complete-record colour endpoint up to a byte boundary and starts a contextual reader over the complete record. Each 16 by 16 block selects default copying, explicit motion copying, predictive residual coding, or raw bytes. Methods 2 and 4 are recognised but unsupported. The crate page supplies the bit mapping and explains the padded linear motion addresses.
//!
//! Predictive alpha uses four 8 by 8 arguments, each with one width-8 slot or four width-4 slots. Reference construction uses prepared picture edges and earlier samples in the temporary record plane. Decoded coefficient arrays are applied directly as residual samples; this path does not call the colour inverse transforms. Modes 0 and 1 first accumulate residuals with wrapping `i16` addition. The predictor clips to a byte, then residual addition clips again.
//!
//! [`AlphaState`] persists across records. Its banks start at 128 and its picture plane starts at zero. Preparation on the first local row of every record replaces top references; later rows use retained bottom edges. Successful blocks update left edges, saved top-right samples, and picture storage. Coefficient-mode history and temporary planes are recreated for each record. The state table on the crate page separates these lifetimes.

use crate::{
    bits::ContextBitReader,
    decoder::SliceGeometry,
    entropy::{CallbackKind, CoefficientHistory, ComponentRequest, decode_contextual_component},
    error::{CoefficientError, DecodeError, ErrorContext, SyntaxError},
    format::Record,
    prediction::{
        PredictionReferences, SignedReferences, filter_signed_references, predict_alpha,
        predict_signed_angular, predict_signed_dc, predict_signed_planar,
        prediction_direction_for_mode,
    },
    surface::{Motion, copy_alpha_motion, copy_block16},
};

/// Stores picture-wide alpha history and predictive reference banks.
///
/// The picture plane uses a stride of `16 * block_width + 32` samples. This trailing storage permits the format's linear alpha motion accesses. Availability and bank state persist across record slices.
pub struct AlphaState {
    /// The coded picture width in blocks.
    block_width: usize,
    /// The padded picture height in blocks.
    total_block_rows: usize,
    /// The alpha plane width including 32 padding samples.
    stride: usize,
    /// The current block's left reference and extension.
    left_bank: [u8; 33],
    /// The current block's top reference and extension.
    top_bank: [u8; 33],
    /// Prepared top banks for each block in the next row.
    next_row_top_banks: Vec<[u8; 33]>,
    /// The bottom sample row of the blocks decoded in the current row.
    current_row_bottom: Vec<u8>,
    /// Marks blocks available in the preceding decoded block row.
    block_available: Vec<u8>,
    /// Retains each block's rightmost top-row sample across a slice boundary.
    saved_block_edge: Vec<u8>,
    /// All reconstructed alpha samples for motion references.
    current_picture_plane: Vec<u8>,
}

impl AlphaState {
    /// Allocates and initialises persistent alpha state for one picture.
    ///
    /// Reference edges start at neutral alpha 128, availability starts clear, and the picture plane starts at zero. Empty or overflowing geometry and failed reservations return [`DecodeError`].
    pub(crate) fn new(block_width: usize, total_block_rows: usize) -> Result<Self, DecodeError> {
        if block_width == 0 || total_block_rows == 0 {
            return Err(invalid("empty picture geometry"));
        }
        let stride = block_width
            .checked_add(2)
            .and_then(|v| v.checked_mul(16))
            .ok_or_else(|| allocation("alpha state"))?;
        let height = total_block_rows
            .checked_mul(16)
            .ok_or_else(|| allocation("alpha state"))?;
        let len = stride
            .checked_mul(height)
            .ok_or_else(|| allocation("alpha state"))?;
        block_width
            .checked_mul(total_block_rows)
            .ok_or_else(|| allocation("alpha state"))?;
        let mut next_row_top_banks = Vec::new();
        next_row_top_banks
            .try_reserve_exact(block_width)
            .map_err(|_| allocation("alpha banks"))?;
        next_row_top_banks.resize(block_width, [128; 33]);
        let mut current_row_bottom = Vec::new();
        let edge_len = block_width
            .checked_mul(16)
            .ok_or_else(|| allocation("alpha edges"))?;
        current_row_bottom
            .try_reserve_exact(edge_len)
            .map_err(|_| allocation("alpha edges"))?;
        current_row_bottom.resize(edge_len, 128);
        let mut block_available = Vec::new();
        block_available
            .try_reserve_exact(block_width)
            .map_err(|_| allocation("alpha block state"))?;
        block_available.resize(block_width, 0);
        let mut saved_block_edge = Vec::new();
        saved_block_edge
            .try_reserve_exact(block_width)
            .map_err(|_| allocation("alpha edges"))?;
        saved_block_edge.resize(block_width, 128);
        let mut current_picture_plane = Vec::new();
        current_picture_plane
            .try_reserve_exact(len)
            .map_err(|_| allocation("alpha picture"))?;
        current_picture_plane.resize(len, 0);
        Ok(Self {
            block_width,
            total_block_rows,
            stride,
            left_bank: [128; 33],
            top_bank: [128; 33],
            next_row_top_banks,
            current_row_bottom,
            block_available,
            saved_block_edge,
            current_picture_plane,
        })
    }

    /// Returns heap bytes reserved by persistent alpha vectors.
    ///
    /// Fixed 33-sample banks are inline and covered by the decoder's conservative context allowance.
    pub(crate) fn heap_bytes(&self) -> usize {
        self.next_row_top_banks.capacity() * std::mem::size_of::<[u8; 33]>()
            + self.current_row_bottom.capacity()
            + self.block_available.capacity()
            + self.saved_block_edge.capacity()
            + self.current_picture_plane.capacity()
    }
}

#[derive(Debug)]
/// The decoded alpha plane for one record slice.
///
/// The plane includes horizontal padding and one additional 16-pixel row region used by reconstruction. Only visible rows and columns are composed into output.
pub struct AlphaSlice {
    /// Row-major alpha samples for the slice.
    pub(crate) plane: Vec<u8>,
    /// The padded sample count between rows.
    pub(crate) stride: usize,
    /// The first bit after alpha syntax, relative to the complete record.
    pub(crate) end_bit: usize,
}

impl AlphaSlice {
    /// Returns the allocated byte capacity of the alpha slice plane.
    ///
    /// Alpha samples are bytes, so vector capacity is also the memory-accounting value.
    pub(crate) fn heap_bytes(&self) -> usize {
        self.plane.capacity()
    }
}

/// Creates a decoder allocation error for a named alpha resource.
///
/// Alpha allocation sites use this helper to keep public error classification consistent.
fn allocation(resource: &'static str) -> DecodeError {
    DecodeError::Allocation { resource }
}
/// Creates a malformed-alpha-syntax error with a static detail.
///
/// Positional record and block context is added by the calling decode layer.
fn invalid(detail: &'static str) -> DecodeError {
    DecodeError::Syntax(SyntaxError::Invalid {
        stream: "alpha",
        detail,
    })
}

/// Attaches alpha record, block, and bit positions to a decoder error.
///
/// The bit position is relative to the complete record. Component context is added separately by coefficient decoding when available.
fn at(error: DecodeError, record: &Record<'_>, block: usize, bit: usize) -> DecodeError {
    error.at(ErrorContext {
        record: Some(usize::from(record.index)),
        block: Some(block),
        component: None,
        bit_position: Some(bit),
    })
}

/// Reads the variable-length alpha block method code.
///
/// Prefix `1` selects method 0, `01` selects method 1, and `00xx` selects methods 2 through 5. Truncated prefixes return invalid alpha syntax.
fn read_method(reader: &mut ContextBitReader<'_>) -> Result<u8, DecodeError> {
    if reader
        .read_value(1)
        .ok_or_else(|| invalid("truncated block method"))?
        != 0
    {
        return Ok(0);
    }
    if reader
        .read_value(1)
        .ok_or_else(|| invalid("truncated block method"))?
        != 0
    {
        return Ok(1);
    }
    Ok(2 + reader
        .read_value_u8(2)
        .ok_or_else(|| invalid("truncated block method"))?)
}

/// Reads an alpha motion vector and returns displacement in samples.
///
/// A leading one abbreviates `(0, 16)`. Otherwise, each component uses a unary-width code plus suffix; code parity selects sign, and the decoded magnitude is multiplied by 16. Prefixes of 31 zeroes and truncated values are rejected.
fn read_motion(reader: &mut ContextBitReader<'_>) -> Result<Motion, DecodeError> {
    if reader
        .read_value(1)
        .ok_or_else(|| invalid("truncated motion vector"))?
        != 0
    {
        return Ok(Motion { x: 0, y: 16 });
    }
    let mut values = [0_i32; 2];
    for value in &mut values {
        let mut zeros = 0;
        loop {
            if reader
                .read_value(1)
                .ok_or_else(|| invalid("truncated motion vector"))?
                != 0
            {
                break;
            }
            zeros += 1;
            if zeros >= 31 {
                return Err(invalid("motion prefix is too long"));
            }
        }
        let suffix = if zeros == 0 {
            0
        } else {
            reader
                .read_value(zeros)
                .ok_or_else(|| invalid("truncated motion suffix"))?
        };
        let code = (1_u32 << zeros) + suffix;
        let decoded = if code & 1 != 0 {
            -i32::try_from(code / 2).expect("motion magnitude fits in i32")
        } else {
            i32::try_from(code / 2).expect("motion magnitude fits in i32")
        };
        *value = decoded.saturating_mul(16);
    }
    Ok(Motion {
        x: values[0],
        y: values[1],
    })
}

/// Extends or neutralises alpha edge banks from neighbour availability.
///
/// The four nibbles of `code` represent above-left, above-right, above, and left availability. Each combination applies the implemented fill or corner average.
fn transition(code: u16, top: &mut [u8; 33], left: &mut [u8; 33]) {
    let t = *top;
    let l = *left;
    let avg = |a: u8, b: u8| {
        u8::try_from((u16::from(a) + u16::from(b) + 1) >> 1)
            .expect("average of two bytes fits in u8")
    };
    match code {
        0x0000 => {
            top.fill(128);
            left.fill(128);
        }
        0x0001 => {
            top.fill(l[1]);
            left[17..].fill(l[16]);
        }
        0x0010 => {
            left.fill(t[1]);
            top[17..].fill(t[16]);
        }
        0x0011 => {
            let c = avg(t[1], l[1]);
            top[0] = c;
            left[0] = c;
            top[17..].fill(t[16]);
            left[17..].fill(l[16]);
        }
        0x0100 => {
            top[..17].fill(t[17]);
            left.fill(t[17]);
        }
        0x0101 => {
            let c = avg(t[17], l[1]);
            top[..17].fill(c);
            left[0] = c;
            left[17..].fill(l[16]);
        }
        0x0110 => {
            top[0] = t[1];
            left.fill(t[1]);
        }
        0x0111 => {
            let c = avg(t[1], l[1]);
            top[0] = c;
            left[0] = c;
            left[17..].fill(l[16]);
        }
        0x1000 => {
            top[1..].fill(t[0]);
            left[1..].fill(l[0]);
        }
        0x1001 => {
            top[1..].fill(t[0]);
            left[17..].fill(l[16]);
        }
        0x1010 => {
            top[17..].fill(t[16]);
            left[1..].fill(l[0]);
        }
        0x1011 => {
            top[17..].fill(t[16]);
            left[17..].fill(l[16]);
        }
        0x1100 => {
            let v = avg(t[0], t[17]);
            top[1..17].fill(v);
            left[1..].fill(l[0]);
        }
        0x1101 => {
            let v = avg(t[0], t[17]);
            top[1..17].fill(v);
            left[17..].fill(l[16]);
        }
        0x1110 => left[1..].fill(l[0]),
        0x1111 => left[17..].fill(l[16]),
        _ => {}
    }
}

/// Reads internal alpha reference edges from a partially reconstructed slice.
///
/// The stored origin and component dimensions support the width-four and width-eight constructor layouts without duplicating bounds calculations.
struct AlphaEdgeReader<'data> {
    /// Borrows the current slice plane.
    plane: &'data [u8],
    /// The padded number of samples per row.
    stride: usize,
    /// The component's linear origin in the slice plane.
    origin: usize,
    /// The component's horizontal sample coordinate.
    x: usize,
    /// The component's vertical sample coordinate.
    y: usize,
    /// The component transform width.
    width: usize,
}

impl AlphaEdgeReader<'_> {
    /// Copies an internal left edge into a reference bank.
    ///
    /// The procedure reads `count` samples above-to-bottom and optionally repeats the last sample to the `2 * width` reference extent. It rejects components on the left edge and checked accesses outside the slice.
    fn vertical(&self, out: &mut [u8; 33], count: usize, extend: bool) -> Result<(), DecodeError> {
        if self.x == 0 {
            return Err(invalid("missing left reference"));
        }
        for i in 0..count {
            out[i + 1] = *self
                .plane
                .get(self.origin - 1 + i * self.stride)
                .ok_or_else(|| invalid("left reference outside slice"))?;
        }
        if extend {
            let edge = out[count];
            out[count + 1..2 * self.width + 1].fill(edge);
        }
        Ok(())
    }

    /// Copies an internal top edge into a reference bank.
    ///
    /// The procedure reads `count` contiguous samples and optionally repeats the last sample to the `2 * width` reference extent. It rejects components on the top edge or reads that exceed the padded row.
    fn above(&self, out: &mut [u8; 33], count: usize, extend: bool) -> Result<(), DecodeError> {
        if self.y == 0 || self.x + count > self.stride {
            return Err(invalid("missing top reference"));
        }
        out[1..count + 1].copy_from_slice(
            &self.plane[self.origin - self.stride..self.origin - self.stride + count],
        );
        if extend {
            let edge = out[count];
            out[count + 1..2 * self.width + 1].fill(edge);
        }
        Ok(())
    }
    /// Construct corner-inclusive byte edges for a width-8 alpha slot.
    ///
    /// The caller supplies a quadrant selector and valid sample origin in the temporary record plane. Ranges exclude their upper bound. Internal edge writes start at index one.
    ///
    /// | Selector | Left edge | Top edge | Corner |
    /// | --- | --- | --- | --- |
    /// | 0 | Copy all 33 bank entries | Copy all 33 bank entries | Preserve bank corners separately |
    /// | 1 | 8 internal samples, repeat index 8 through 9..17 | Top bank window 8..25 | Top window's first value supplies both |
    /// | 2 | Left bank window 8..25 | 16 internal top samples | Left window's first value supplies both |
    /// | 3 | 8 internal samples, repeat through 9..17 | 8 internal samples, repeat through 9..17 | Sample immediately above-left supplies both |
    ///
    /// Selectors 1, 2, and 3 retain output indices 17..33. Selector 0 differs from colour's width-8 constructor, which copies only indices 0..17.
    ///
    /// # Errors
    ///
    /// Invalid selectors and checked edge failures return invalid alpha syntax, possibly after earlier output writes.
    ///
    /// # Panics
    ///
    /// Selector 3 directly reads its corner. The caller must supply a non-zero row/column and sufficient plane storage, as the production quadrant layout does.
    fn construct_eight(
        &self,
        selector: usize,
        left_bank: &[u8; 33],
        top_bank: &[u8; 33],
        left: &mut [u8; 33],
        top: &mut [u8; 33],
    ) -> Result<(), DecodeError> {
        match selector {
            0 => {
                *left = *left_bank;
                *top = *top_bank;
            }
            1 => {
                top[..17].copy_from_slice(&top_bank[8..25]);
                left[0] = top[0];
                self.vertical(left, 8, true)?;
            }
            2 => {
                left[..17].copy_from_slice(&left_bank[8..25]);
                top[0] = left[0];
                self.above(top, 16, false)?;
            }
            3 => {
                left[0] = self.plane[self.origin - self.stride - 1];
                top[0] = left[0];
                self.vertical(left, 8, true)?;
                self.above(top, 8, true)?;
            }
            _ => return Err(invalid("invalid width-eight reference selector")),
        }
        Ok(())
    }
}

/// Prepare corner-inclusive byte references for one alpha slot.
///
/// `position` is a sample origin in the temporary record `plane`, whose row spacing is `stride`.
/// The caller retains the two output arrays across slots of one argument. Width-4 selectors 0..15
/// follow `argument*4+slot`; width-8 selectors 0..3 identify the four quadrants. Width 16 copies
/// both complete banks. The width-4 selector rules match the table in
/// [`crate::colour::predictive::construct_colour_refs`], except selector 0 copies all 33 entries
/// rather than nine. Width-8 selector 0 also copies all 33 entries rather than 17.
///
/// Internal left/top values are read from previously reconstructed slots, not persistent picture
/// storage. They start at output index one. A bank window supplies both corners when the other
/// edge is internal. Selector 0 and width 16 preserve each bank's corner separately. With two
/// internal edges, the sample immediately above-left supplies both corners. Some selectors
/// preserve earlier extension entries; clearing the arrays for every slot would change
/// reconstruction.
///
/// # Errors
///
/// Invalid width, selector, overflow, or checked edge accesses return [`DecodeError`], possibly
/// after partial edge writes.
///
/// # Panics
///
/// The caller must provide a complete plane and a selector consistent with the reconstructed slot
/// position. Direct corner accesses for internal selectors and some top slices can panic if those
/// private-call invariants are violated.
fn construct_refs(
    width: usize,
    selector: usize,
    banks: (&[u8; 33], &[u8; 33]),
    plane: &[u8],
    stride: usize,
    position: (usize, usize),
    edges: (&mut [u8; 33], &mut [u8; 33]),
) -> Result<(), DecodeError> {
    let (left_bank, top_bank) = banks;
    let (x, y) = position;
    let (left, top) = edges;
    let origin = y
        .checked_mul(stride)
        .and_then(|v| v.checked_add(x))
        .ok_or_else(|| invalid("reference offset overflow"))?;
    let reader = AlphaEdgeReader {
        plane,
        stride,
        origin,
        x,
        y,
        width,
    };
    match width {
        4 => match selector {
            0 => {
                *left = *left_bank;
                *top = *top_bank;
            }
            1 => {
                top[..9].copy_from_slice(&top_bank[4..13]);
                left[0] = top[0];
                reader.vertical(left, 4, true)?;
            }
            2 => {
                left[..9].copy_from_slice(&left_bank[4..13]);
                top[0] = left[0];
                reader.above(top, 8, false)?;
            }
            3 => {
                left[0] = plane[origin - stride - 1];
                top[0] = left[0];
                reader.vertical(left, 4, true)?;
                reader.above(top, 4, false)?;
            }
            4 => {
                top[..9].copy_from_slice(&top_bank[8..17]);
                left[0] = top[0];
                reader.vertical(left, 8, false)?;
            }
            5 => {
                top[..9].copy_from_slice(&top_bank[12..21]);
                left[0] = top[0];
                reader.vertical(left, 4, true)?;
            }
            6 | 9 | 14 => {
                left[0] = plane[origin - stride - 1];
                top[0] = left[0];
                reader.vertical(left, 4, true)?;
                reader.above(top, 8, false)?;
            }
            7 | 11 | 13 | 15 => {
                left[0] = plane[origin - stride - 1];
                top[0] = left[0];
                reader.vertical(left, 4, true)?;
                reader.above(top, 4, true)?;
            }
            8 => {
                left[..9].copy_from_slice(&left_bank[8..17]);
                top[0] = left[0];
                reader.above(top, 8, false)?;
            }
            10 => {
                left[..9].copy_from_slice(&left_bank[12..21]);
                top[0] = left[0];
                reader.above(top, 8, false)?;
            }
            12 => {
                left[0] = plane[origin - stride - 1];
                top[0] = left[0];
                reader.vertical(left, 8, false)?;
                reader.above(top, 8, false)?;
            }
            _ => return Err(invalid("invalid width-four reference selector")),
        },
        8 => reader.construct_eight(selector, left_bank, top_bank, left, top)?,
        16 => {
            *left = *left_bank;
            *top = *top_bank;
        }
        _ => return Err(invalid("invalid component width")),
    }
    Ok(())
}

/// Applies alpha prediction mode 4 in signed reference space.
///
/// The predictor extends the top reference backwards from the left edge with the fixed inverse
/// angle and then interpolates each row. Width must be 4, 8, or 16.
fn alpha_signed_four(
    left: &[i16; 33],
    top: &[i16; 33],
    width: usize,
    signed: &mut [i16; 256],
) -> Result<(), DecodeError> {
    let mut main = [0_i16; 65];
    main[32..65].copy_from_slice(top);
    let mut sum = 128;
    let last: isize = match width {
        4 => -2,
        8 => -3,
        16 => -6,
        _ => return Err(invalid("invalid component width")),
    };
    for index in (last..=-1_isize).rev() {
        sum += 630;
        main[usize::try_from(32_isize + index).expect("angular index is nonnegative")] =
            left[usize::try_from(sum >> 8).expect("angular index is nonnegative")];
    }
    for row in 0..width {
        let d = -((i32::try_from(row).expect("block width is at most 16") + 1) * 13);
        let integer = d >> 5;
        let frac = d & 31;
        for column in 0..width {
            let main_index = usize::try_from(
                32 + i32::try_from(column).expect("block width is at most 16") + integer + 1,
            )
            .expect("angular index is nonnegative");
            signed[row * width + column] = i16::try_from(
                ((32 - frac) * i32::from(main[main_index])
                    + frac * i32::from(main[main_index + 1])
                    + 16)
                    >> 5,
            )
            .expect("interpolated byte fits in i16");
        }
    }
    Ok(())
}

/// Applies alpha prediction mode 7 in signed reference space.
///
/// Width 4 uses the codec's dedicated inverse-angle interpolation. Widths 8 and 16 delegate to the
/// general signed angular predictor with direction 11.
fn alpha_signed_seven(
    l: &[i16; 33],
    t: &[i16; 33],
    width: usize,
    signed: &mut [i16; 256],
) -> Result<(), DecodeError> {
    if width == 4 {
        let mut main = [0_i16; 34];
        main[16..33].copy_from_slice(&l[..17]);
        for i in (-16_isize..=-1_isize).rev() {
            let side =
                usize::try_from((-i * 630 + 128) >> 8).expect("angular side index is nonnegative");
            main[usize::try_from(16_isize + i).expect("angular index is nonnegative")] =
                t[side.min(16) + 1];
        }
        for r in 0..4 {
            for c in 0..4 {
                let d = -((i32::try_from(c).expect("column is below four") + 1) * 13);
                let integer = -((-d + 31) / 32);
                let frac = d - integer * 32;
                let idx = usize::try_from(
                    16 + i32::try_from(r).expect("row is below four") + integer + 1,
                )
                .expect("angular index is nonnegative");
                signed[r * 4 + c] = i16::try_from(
                    ((32 - frac) * i32::from(main[idx]) + frac * i32::from(main[idx + 1]) + 16)
                        >> 5,
                )
                .expect("interpolated byte fits in i16");
            }
        }
    } else {
        predict_signed_angular(l, t, width, 11, &mut signed[..width * width])
            .map_err(|_| invalid("left angular prediction failed"))?;
    }
    Ok(())
}

/// Dispatches one alpha predictor over signed reference samples.
///
/// Modes 0 and 1 copy edges, 2 is DC, 3 is diagonal, 4 through 8 use fixed directional formulae, 9
/// through 16 are angular, and 17 is planar. This helper does not clip; `predict` clips its signed
/// result to bytes before `apply_component` adds residuals.
///
/// Invalid widths, unsupported modes, and lower-level prediction failures return invalid alpha
/// syntax.
fn predict_signed_mode(
    left: &[i16; 33],
    top: &[i16; 33],
    left_reference: &[i16; 33],
    top_reference: &[i16; 33],
    width: usize,
    mode: u8,
    signed: &mut [i16; 256],
) -> Result<(), DecodeError> {
    match mode {
        0 | 1 => {
            for r in 0..width {
                for c in 0..width {
                    signed[r * width + c] = if mode == 0 {
                        top_reference[c + 1]
                    } else {
                        left_reference[r + 1]
                    };
                }
            }
        }
        2 => predict_signed_dc(left, top, width, &mut signed[..width * width])
            .map_err(|_| invalid("DC prediction failed"))?,
        3 => {
            for r in 0..width {
                for c in 0..width {
                    signed[r * width + c] = if c >= r {
                        top_reference[c - r]
                    } else {
                        left_reference[r - c]
                    };
                }
            }
        }
        4 => alpha_signed_four(left_reference, top_reference, width, signed)?,
        5 => {
            for r in 0..width {
                let d = (r + 1) * 13;
                let i = d >> 5;
                let f = i32::try_from(d & 31).expect("fraction is at most 31");
                for c in 0..width {
                    let n = c + i + 1;
                    signed[r * width + c] = i16::try_from(
                        ((32 - f) * i32::from(top_reference[n])
                            + f * i32::from(top_reference[n + 1])
                            + 16)
                            >> 5,
                    )
                    .expect("interpolated byte fits in i16");
                }
            }
        }
        6 => {
            for r in 0..width {
                for c in 0..width {
                    signed[r * width + c] = if width == 4 {
                        let i = r + c;
                        if i >= 6 {
                            top_reference[8]
                        } else {
                            (top_reference[i + 1]
                                + 2 * top_reference[i + 2]
                                + top_reference[i + 3]
                                + 2)
                                >> 2
                        }
                    } else {
                        top_reference[r + c + 2]
                    };
                }
            }
        }
        7 => alpha_signed_seven(left_reference, top_reference, width, signed)?,
        8 => {
            for r in 0..width {
                for c in 0..width {
                    let d = (c + 1) * 13;
                    let i = r + (d >> 5) + 1;
                    let f = i32::try_from(d & 31).expect("fraction is at most 31");
                    signed[r * width + c] = i16::try_from(
                        ((32 - f) * i32::from(left_reference[i])
                            + f * i32::from(left_reference[i + 1])
                            + 16)
                            >> 5,
                    )
                    .expect("interpolated byte fits in i16");
                }
            }
        }
        9..=16 => {
            predict_signed_angular_mode(left_reference, top_reference, width, mode, signed)?;
        }
        17 => {
            let mut refs = SignedReferences::default();
            refs.left[..32].copy_from_slice(&left[1..33]);
            refs.top[..32].copy_from_slice(&top[1..33]);
            predict_signed_planar(&refs, width, &mut signed[..width * width])
                .map_err(|_| invalid("planar prediction failed"))?;
        }
        _ => return Err(invalid("unsupported prediction mode")),
    }
    Ok(())
}

/// Maps an alpha mode to a general signed angular direction and predicts it.
///
/// Only modes with a direction-table entry are accepted. The output slice covers exactly `width *
/// width` samples.
fn predict_signed_angular_mode(
    left_reference: &[i16; 33],
    top_reference: &[i16; 33],
    width: usize,
    mode: u8,
    signed: &mut [i16; 256],
) -> Result<(), DecodeError> {
    let direction =
        prediction_direction_for_mode(mode).ok_or_else(|| invalid("invalid angular mode"))?;
    predict_signed_angular(
        left_reference,
        top_reference,
        width,
        direction,
        &mut signed[..width * width],
    )
    .map_err(|_| invalid("angular prediction failed"))
}

/// Predict a square from corner-inclusive byte banks.
///
/// The caller supplies width 4,8,16 and prepared indices 0..=2*width. Width-4 modes
/// 0,1,2,4,5,6,7,8,17 convert to [`PredictionReferences`]: drop the bank corner from the edge
/// arrays, copy eight neighbours, and repeat the final value through the remaining capacity. Mode
/// 4 uses the top corner; other such modes use the left corner. Both first-row flags remain false
/// because bank preparation has already replaced unavailable references.
///
/// Other cases convert active references to `i16`. Filter width-4 modes 3/9, width-8 modes 3..8,
/// and width-16 modes 3..16 before dispatch. Mode 2 uses unfiltered DC and mode 17 uses unfiltered
/// planar references. Clip the signed result to 0..255 here, before residual addition. Writes only
/// `out[..width*width]`; inputs and later output entries are unchanged.
///
/// # Errors
///
/// Unsupported modes, widths, or filter/prediction failures return [`DecodeError`]. The caller
/// must enforce supported dimensions before the fixed-capacity slices are formed; this is a
/// private production dispatcher.
fn predict(
    left_u8: &[u8; 33],
    top_u8: &[u8; 33],
    width: usize,
    mode: u8,
    out: &mut [u8; 256],
) -> Result<(), DecodeError> {
    if width == 4 && ((mode <= 8 && mode != 3) || mode == 17) {
        let mut references = PredictionReferences {
            top_left: if mode == 4 { top_u8[0] } else { left_u8[0] },
            ..Default::default()
        };
        references.left[..8].copy_from_slice(&left_u8[1..9]);
        references.top[..8].copy_from_slice(&top_u8[1..9]);
        let last_left = references.left[7];
        let last_top = references.top[7];
        references.left[8..].fill(last_left);
        references.top[8..].fill(last_top);
        predict_alpha(&references, width, mode, &mut out[..width * width], false)
            .map_err(|_| invalid("width-four alpha prediction failed"))?;
        return Ok(());
    }
    let mut left = [0_i16; 33];
    let mut top = [0_i16; 33];
    for i in 0..=2 * width {
        left[i] = i16::from(left_u8[i]);
        top[i] = i16::from(top_u8[i]);
    }
    let mut fl = [0_i16; 33];
    let mut ft = [0_i16; 33];
    let filtered = mode != 17
        && ((width == 4 && matches!(mode, 3 | 6 | 9))
            || (width == 8 && (3..=8).contains(&mode))
            || (width == 16 && (3..=16).contains(&mode)));
    if filtered {
        filter_signed_references(&left, &top, width, &mut fl, &mut ft)
            .map_err(|_| invalid("reference filtering failed"))?;
    }
    let l = if filtered { &fl } else { &left };
    let t = if filtered { &ft } else { &top };
    let mut signed = [0_i16; 256];
    predict_signed_mode(&left, &top, l, t, width, mode, &mut signed)?;
    for i in 0..width * width {
        out[i] = u8::try_from(signed[i].clamp(0, 255)).expect("clamped sample fits in u8");
    }
    Ok(())
}

/// Add decoded alpha values to a prepared byte prediction.
///
/// The caller supplies width 4 or 8 in production and row-major arrays with capacity 256. Only the
/// first `width*width` output entries are written; later entries are unchanged. The input arrays
/// remain unchanged. No inverse transform or profile scaling occurs here.
///
/// Copy `coeff` into local signed storage. Mode 0 adds the preceding row's already accumulated
/// value to each later row; mode 1 adds the preceding column's value across each row. These
/// additions wrap in `i16`. Other modes leave values as decoded. Inactive callbacks contribute
/// zero regardless of that local array; both active callback kinds use it directly. Add each
/// residual to the byte prediction in `i32` and clip to 0..255.
fn apply_component(
    width: usize,
    mode: u8,
    callback: CallbackKind,
    coeff: &[i16; 256],
    prediction: &[u8; 256],
    output: &mut [u8; 256],
) {
    let mut values = *coeff;
    if mode == 0 {
        for r in 1..width {
            for c in 0..width {
                values[r * width + c] =
                    values[r * width + c].wrapping_add(values[(r - 1) * width + c]);
            }
        }
    } else if mode == 1 {
        for r in 0..width {
            for c in 1..width {
                values[r * width + c] =
                    values[r * width + c].wrapping_add(values[r * width + c - 1]);
            }
        }
    }
    for i in 0..width * width {
        let residual = if callback == CallbackKind::Inactive {
            0
        } else {
            i32::from(values[i])
        };
        output[i] = u8::try_from((i32::from(prediction[i]) + residual).clamp(0, 255))
            .expect("clamped sample fits in u8");
    }
}

/// Groups mutable state needed to decode one alpha block.
///
/// The context is passed by value to keep the block procedure's argument list manageable while
/// retaining distinct data and record lifetimes.
struct AlphaBlockContext<'data, 'record> {
    /// Borrows the framed record for raw reads and diagnostics.
    record: &'data Record<'record>,
    /// Current slice and picture geometry.
    geometry: SliceGeometry,
    /// Persistent alpha history.
    state: &'data mut AlphaState,
    /// Reads contextual alpha syntax from the record.
    reader: &'data mut ContextBitReader<'record>,
    /// Contextual coefficient history for this slice.
    history: &'data mut CoefficientHistory,
    /// Receives reconstructed slice alpha samples.
    plane: &'data mut [u8],
    /// The padded alpha sample count per row.
    stride: usize,
}

/// Locates one alpha block in diagnostic, slice, and picture coordinates.
///
/// Pixel coordinates start at the left edge of the alpha plane and use its padded stride; block
/// and bit values identify the syntax that produced the block.
#[derive(Clone, Copy)]
struct AlphaBlockPosition {
    /// The zero-based record-local block number.
    block: usize,
    /// The record-relative bit position at the block start.
    bit: usize,
    /// The block column.
    block_x: usize,
    /// The slice-local block row.
    block_y: usize,
    /// The horizontal destination sample.
    dst_x: usize,
    /// The slice-local vertical destination sample.
    local_y: usize,
    /// The picture-global vertical destination sample.
    global_y: usize,
}

/// Reconstruct one alpha block from four coded arguments.
///
/// The next bit must be one; zero or absent syntax returns an error. Each argument covers an 8 by
/// 8 quadrant in top-left, top-right, bottom-left, bottom-right order and selects one width-8 slot
/// or four width-4 slots. The caller has prepared picture banks and bounded the 16 by 16
/// destination in the temporary record plane.
///
/// Decode mode and coefficient syntax, construct references from banks and earlier output,
/// predict, then apply the coded values directly as residuals. There is no inverse transform. Each
/// reconstructed slot is written immediately to the temporary plane. Edge constructor arrays are
/// zeroed per argument and retained between its slots; coefficient-mode history updates during
/// syntax decoding.
///
/// # Errors
///
/// Missing initialisation, coefficient syntax, invalid references, or prediction failures return
/// [`DecodeError`]. Earlier slots, banks, mode history, and the contextual cursor can already have
/// changed. Coefficient errors include argument and complete-record bit context. Persistent
/// picture samples are committed by the outer block procedure after success.
fn decode_alpha_predictive(
    record: &Record<'_>,
    state: &mut AlphaState,
    reader: &mut ContextBitReader<'_>,
    history: &mut CoefficientHistory,
    plane: &mut [u8],
    stride: usize,
    position: AlphaBlockPosition,
) -> Result<(), DecodeError> {
    let AlphaBlockPosition {
        block,
        bit,
        block_x,
        dst_x,
        local_y,
        ..
    } = position;

    if reader.read_value(1) != Some(1) {
        return Err(at(
            invalid("predictive transform callback is not initialised"),
            record,
            block,
            bit,
        ));
    }
    for argument in 0..4_u32 {
        let component = decode_contextual_component(
            reader,
            history,
            ComponentRequest {
                argument,
                horizontal_offset: u16::try_from(block_x * 16)
                    .map_err(|_| invalid("predictive block offset overflow"))?,
                maximum_width: 8,
                fixed_width: 0,
                colour_long_symbols: false,
                width_selector_already_read: false,
            },
        )
        .map_err(|error| {
            DecodeError::Coefficients(CoefficientError::Syntax(Box::new(error))).at(ErrorContext {
                record: Some(usize::from(record.index)),
                block: Some(block),
                component: Some(argument as usize),
                bit_position: Some(reader.logical_bit()),
            })
        })?;
        let width = usize::from(component.width);
        // Constructors may write only part of each side. Their remaining samples persist across
        // slots in one argument.
        let mut left = [0_u8; 33];
        let mut top = [0_u8; 33];
        for slot in 0..component.count {
            let base_x =
                dst_x + (argument as usize & 1) * 8 + if width == 4 { (slot & 1) * 4 } else { 0 };
            let base_y = local_y
                + (argument as usize >> 1) * 8
                + if width == 4 { (slot >> 1) * 4 } else { 0 };
            let selector = if width == 4 {
                argument as usize * 4 + slot
            } else {
                argument as usize
            };
            construct_refs(
                width,
                selector,
                (&state.left_bank, &state.top_bank),
                plane,
                stride,
                (base_x, base_y),
                (&mut left, &mut top),
            )
            .map_err(|e| at(e, record, block, reader.logical_bit()))?;
            let mut pred = [0_u8; 256];
            predict(&left, &top, width, component.modes[slot], &mut pred)
                .map_err(|e| at(e, record, block, reader.logical_bit()))?;
            let mut reconstructed = [0_u8; 256];
            apply_component(
                width,
                component.modes[slot],
                component.callbacks[slot],
                &component.values[slot],
                &pred,
                &mut reconstructed,
            );
            for r in 0..width {
                let out = (base_y + r) * stride + base_x;
                plane[out..out + width]
                    .copy_from_slice(&reconstructed[r * width..r * width + width]);
            }
        }
    }
    Ok(())
}

/// Prepares top and left alpha banks before one block is decoded.
///
/// Top state comes from the preceding block row, the previous block in a slice's first row, or
/// neutral 128. For later slice-local rows, a four-neighbour availability code extends unavailable
/// portions of both banks.
fn prepare_alpha_banks(
    state: &mut AlphaState,
    geometry: SliceGeometry,
    block_x: usize,
    block_y: usize,
    absolute_row: usize,
    previous: &[u8],
    current: &[u8],
) {
    if block_y != 0 {
        state.top_bank = state.next_row_top_banks[block_x];
    } else if block_x != 0 {
        state.top_bank.fill(state.saved_block_edge[block_x - 1]);
    } else {
        state.top_bank.fill(128);
    }
    let top_available = absolute_row != 0 && previous[block_x] != 0;
    let left_available = block_x != 0 && current[block_x - 1] != 0;
    if block_y != 0 {
        let code = (u16::from(block_x > 0 && previous[block_x - 1] != 0) << 12)
            | (u16::from(block_x + 1 < geometry.block_width && previous[block_x + 1] != 0) << 8)
            | (u16::from(top_available) << 4)
            | u16::from(left_available);
        transition(code, &mut state.top_bank, &mut state.left_bank);
    } else if block_x == 0 {
        state.left_bank.fill(128);
    }
}

/// Commits one decoded alpha block to persistent picture and edge state.
///
/// The block is copied from the temporary slice plane, marked available, and used to update the
/// saved right edge, left bank, and current block-row bottom samples for subsequent blocks.
fn finish_alpha_block(
    state: &mut AlphaState,
    plane: &[u8],
    stride: usize,
    position: AlphaBlockPosition,
    current: &mut [u8],
) {
    let AlphaBlockPosition {
        block_x,
        block_y,
        dst_x,
        local_y,
        global_y,
        ..
    } = position;
    for row in 0..16 {
        let src = (local_y + row) * stride + dst_x;
        let dest = (global_y + row) * stride + dst_x;
        state.current_picture_plane[dest..dest + 16].copy_from_slice(&plane[src..src + 16]);
    }
    current[block_x] = 1;
    state.saved_block_edge[block_x] = plane[local_y * stride + dst_x + 15];
    state.left_bank[0] = if block_y == 0 {
        128
    } else {
        state.top_bank[16]
    };
    for i in 0..16 {
        state.left_bank[i + 1] = plane[(local_y + i) * stride + dst_x + 15];
        state.current_row_bottom[dst_x + i] = plane[(local_y + 15) * stride + dst_x + i];
    }
    let left_edge = state.left_bank[16];
    state.left_bank[17..].fill(left_edge);
}

/// Reconstructs an alpha block by spatial or explicit motion copying.
///
/// Method 0 selects the block above in the first column and the block to the left elsewhere.
/// Method 1 reads an explicit vector. The source is copied within persistent padded picture
/// history and then into the current slice plane.
///
/// Out-of-range motion and plane-copy failures become contextual invalid-alpha errors.
fn decode_alpha_motion(
    record: &Record<'_>,
    state: &mut AlphaState,
    reader: &mut ContextBitReader<'_>,
    plane: &mut [u8],
    stride: usize,
    position: AlphaBlockPosition,
    method: u8,
) -> Result<(), DecodeError> {
    let AlphaBlockPosition {
        block,
        bit,
        block_x,
        dst_x,
        local_y,
        global_y,
        ..
    } = position;

    let motion = if method == 0 {
        if block_x == 0 {
            Motion { x: 0, y: 16 }
        } else {
            Motion { x: 16, y: 0 }
        }
    } else {
        read_motion(reader).map_err(|e| at(e, record, block, bit))?
    };
    copy_alpha_motion(
        &mut state.current_picture_plane,
        stride,
        state.total_block_rows * 16,
        dst_x,
        global_y,
        motion,
    )
    .map_err(|_| {
        at(
            invalid("motion source outside padded picture"),
            record,
            block,
            bit,
        )
    })?;
    copy_block16(
        &state.current_picture_plane,
        stride,
        (dst_x, global_y),
        plane,
        stride,
        (dst_x, local_y),
    )
    .map_err(|_| at(invalid("alpha block copy failed"), record, block, bit))?;
    Ok(())
}

/// Reads one byte-aligned raw alpha block.
///
/// The procedure rounds the contextual endpoint up to a byte boundary, copies exactly 256 bytes in
/// row-major order, then resets the accumulator after the raw bytes. Alignment bits are not
/// required to be zero.
///
/// A range that overflows or exceeds the complete record returns a contextual truncation error
/// without updating reader state.
fn decode_raw_alpha_block(
    record: &Record<'_>,
    reader: &mut ContextBitReader<'_>,
    plane: &mut [u8],
    stride: usize,
    position: AlphaBlockPosition,
) -> Result<(), DecodeError> {
    let AlphaBlockPosition {
        block,
        dst_x,
        local_y,
        ..
    } = position;

    let raw_bit = (reader.logical_bit() + 7) & !7;
    let offset = raw_bit / 8;
    let end = offset
        .checked_add(256)
        .filter(|&v| v <= record.data.len())
        .ok_or_else(|| at(invalid("truncated raw alpha block"), record, block, raw_bit))?;
    for row in 0..16 {
        let out = (local_y + row) * stride + dst_x;
        plane[out..out + 16]
            .copy_from_slice(&record.data[offset + row * 16..offset + row * 16 + 16]);
    }
    reader.byte_offset = end;
    reader.accumulator = 0;
    reader.available = 0;
    Ok(())
}

/// Decodes and commits one alpha block selected by its method prefix.
///
/// Methods 0 and 1 copy motion references, method 3 uses contextual predictive coding, and method
/// 5 reads raw samples. Methods 2 and 4 are recognised but unsupported. Bank preparation changes
/// state before dispatch. Successful reconstruction then updates saved edges and picture storage.
///
/// Method and motion failures use the block start bit. Predictive failures can use the current
/// contextual endpoint, and raw failures use the aligned raw start. All use record-local block
/// numbers and complete-record bit origins.
fn decode_alpha_block(
    context: AlphaBlockContext<'_, '_>,
    block_x: usize,
    block_y: usize,
    absolute_row: usize,
    previous: &[u8],
    current: &mut [u8],
) -> Result<(), DecodeError> {
    let AlphaBlockContext {
        record,
        geometry,
        state,
        reader,
        history,
        plane,
        stride,
    } = context;
    let block = block_y * geometry.block_width + block_x;
    let bit = reader.logical_bit();
    prepare_alpha_banks(
        state,
        geometry,
        block_x,
        block_y,
        absolute_row,
        previous,
        current,
    );
    let method = read_method(reader).map_err(|e| at(e, record, block, bit))?;
    if matches!(method, 2 | 4) {
        return Err(at(
            DecodeError::Syntax(SyntaxError::UnsupportedMethod {
                stream: "alpha",
                method,
            }),
            record,
            block,
            bit,
        ));
    }
    let dst_x = block_x * 16;
    let local_y = block_y * 16;
    let global_y = absolute_row * 16;
    let position = AlphaBlockPosition {
        block,
        bit,
        block_x,
        block_y,
        dst_x,
        local_y,
        global_y,
    };
    match method {
        0 | 1 => decode_alpha_motion(record, state, reader, plane, stride, position, method)?,
        3 => decode_alpha_predictive(record, state, reader, history, plane, stride, position)?,
        5 => decode_raw_alpha_block(record, reader, plane, stride, position)?,
        _ => {
            return Err(at(
                invalid("unknown alpha coding method"),
                record,
                block,
                bit,
            ));
        }
    }
    finish_alpha_block(state, plane, stride, position, current);
    Ok(())
}

/// Finalises availability and top banks after one alpha block row.
///
/// The current availability vector becomes history for the next row. Each 33-sample top bank
/// receives a corner and 32 samples from the completed bottom edge, with right-side values clamped
/// to the final visible picture sample.
fn finish_alpha_row(state: &mut AlphaState, geometry: SliceGeometry, current: &[u8]) {
    state.block_available.copy_from_slice(current);
    for x in 0..geometry.block_width {
        let origin = x * 16;
        let bank = &mut state.next_row_top_banks[x];
        bank[0] = if origin == 0 {
            0
        } else {
            state.current_row_bottom[origin - 1]
        };
        for i in 0..32 {
            let px = origin + i;
            bank[i + 1] = if px < state.current_row_bottom.len() && px < geometry.picture_width {
                state.current_row_bottom[px]
            } else {
                state.current_row_bottom[geometry.picture_width.saturating_sub(1)]
            };
        }
    }
}

/// Decodes the alpha stream that follows one record's colour stream.
///
/// Alpha begins at the next byte boundary after `colour_end_bit`. The decoder creates contextual
/// coefficient history and a padded temporary plane, then visits blocks in row-major order while
/// maintaining picture-wide motion and edge state. The returned endpoint is relative to the
/// complete record.
///
/// The contextual reader requires a four-byte initial window. Invalid geometry, insufficient
/// input, malformed or unsupported methods, invalid references, coefficient failures, and
/// allocation failures return [`DecodeError`].
pub fn decode_slice(
    record: &Record<'_>,
    geometry: SliceGeometry,
    colour_end_bit: usize,
    state: &mut AlphaState,
) -> Result<AlphaSlice, DecodeError> {
    if geometry.block_width != state.block_width
        || geometry
            .absolute_block_row
            .checked_add(geometry.block_rows)
            .filter(|&v| v <= state.total_block_rows)
            .is_none()
        || geometry.block_rows == 0
    {
        return Err(invalid("slice geometry does not match alpha state"));
    }
    let start = (colour_end_bit + 7) & !7;
    let mut reader = ContextBitReader::new(record.data);
    if !reader.initialise_at_bit(start) {
        return Err(at(
            invalid("alpha stream has no bounded reader window"),
            record,
            0,
            start,
        ));
    }
    let history_stride = geometry
        .block_width
        .checked_mul(4)
        .and_then(|v| v.checked_add(1))
        .and_then(|v| u16::try_from(v).ok())
        .ok_or_else(|| allocation("alpha coefficient history"))?;
    let rows =
        u16::try_from(geometry.block_rows).map_err(|_| allocation("alpha coefficient history"))?;
    let mut history = CoefficientHistory::new(history_stride, rows).map_err(|error| {
        crate::error::coefficient_history_error(error, "alpha coefficient history")
    })?;
    let stride = state.stride;
    let out_rows = geometry
        .block_rows
        .checked_mul(16)
        .and_then(|v| v.checked_add(16))
        .ok_or_else(|| allocation("alpha slice"))?;
    let len = stride
        .checked_mul(out_rows)
        .ok_or_else(|| allocation("alpha slice"))?;
    let mut plane = Vec::new();
    plane
        .try_reserve_exact(len)
        .map_err(|_| allocation("alpha slice"))?;
    plane.resize(len, 0);
    let mut previous = Vec::new();
    previous
        .try_reserve_exact(geometry.block_width)
        .map_err(|_| allocation("alpha block state"))?;
    previous.resize(geometry.block_width, 0);
    let mut current = Vec::new();
    current
        .try_reserve_exact(geometry.block_width)
        .map_err(|_| allocation("alpha block state"))?;
    current.resize(geometry.block_width, 0);
    for block_y in 0..geometry.block_rows {
        history
            .begin_alpha_block_row(u16::try_from(block_y).expect("block row fits history"))
            .map_err(|_| invalid("coefficient row history failed"))?;
        let absolute_row = geometry.absolute_block_row + block_y;
        previous.copy_from_slice(&state.block_available);
        current.fill(0);
        for block_x in 0..geometry.block_width {
            decode_alpha_block(
                AlphaBlockContext {
                    record,
                    geometry,
                    state,
                    reader: &mut reader,
                    history: &mut history,
                    plane: &mut plane,
                    stride,
                },
                block_x,
                block_y,
                absolute_row,
                &previous,
                &mut current,
            )?;
        }
        finish_alpha_row(state, geometry, &current);
    }
    Ok(AlphaSlice {
        plane,
        stride,
        end_bit: reader.logical_bit(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::RecordHeader;

    fn record(data: &[u8]) -> Record<'_> {
        Record {
            data,
            payload: &data[14..],
            payload_offset: 14,
            index: 0,
            stream_offset: 0,
            header: RecordHeader {
                opaque_byte_b: 0,
                index: 0,
                opaque_byte_a: 0,
                opaque_flag: false,
                opaque_bits: 0,
                opaque_byte_c: 0,
                opaque_byte_d: 0,
                opaque_byte_e: 0,
            },
        }
    }

    fn geometry(row: usize, rows: usize) -> SliceGeometry {
        SliceGeometry {
            block_width: 1,
            block_rows: rows,
            absolute_block_row: row,
            picture_width: 16,
            picture_height: 32,
        }
    }

    fn raw_record(value: u8) -> Vec<u8> {
        let mut bytes = vec![0_u8; 14 + 1 + 256];
        bytes[14] = 0x30;
        bytes[15..].fill(value);
        bytes
    }

    #[test]
    fn raw_alpha_block_is_byte_aligned_and_advances_record_bit_position() {
        let bytes = raw_record(0x5a);
        let mut state = AlphaState::new(1, 1).unwrap();
        let decoded = decode_slice(&record(&bytes), geometry(0, 1), 112, &mut state).unwrap();
        assert_eq!(decoded.plane[..16], [0x5a; 16]);
        assert_eq!(decoded.end_bit, 2168);
    }

    #[test]
    fn unsupported_alpha_methods_have_typed_errors() {
        for (method, byte) in [(2, 0x00), (4, 0x20)] {
            let mut bytes = vec![0_u8; 18];
            bytes[14] = byte;
            let mut state = AlphaState::new(1, 1).unwrap();
            let error = decode_slice(&record(&bytes), geometry(0, 1), 112, &mut state).unwrap_err();
            assert!(
                matches!(error,DecodeError::At{source,..} if matches!(*source,DecodeError::Syntax(SyntaxError::UnsupportedMethod{stream:"alpha",method:m}) if m==method))
            );
        }
    }

    #[test]
    fn picture_alpha_bank_and_motion_history_survive_record_boundaries() {
        let first = raw_record(77);
        let mut state = AlphaState::new(1, 2).unwrap();
        decode_slice(&record(&first), geometry(0, 1), 112, &mut state).unwrap();
        assert_eq!(&state.next_row_top_banks[0][1..17], &[77; 16]);

        let mut second = vec![0_u8; 18];
        second[14] = 0x60; // motion method, default vector (0, 16)
        let decoded = decode_slice(&record(&second), geometry(1, 1), 112, &mut state).unwrap();
        assert_eq!(&decoded.plane[..16], &[77; 16]);
    }

    #[test]
    fn motion_code_parity_selects_sign() {
        let mut bits = ContextBitReader::new(&[0x26, 0, 0, 0]);
        assert!(bits.initialise_at_bit(0));
        assert_eq!(read_motion(&mut bits).unwrap(), Motion { x: 16, y: -16 });
    }
}
