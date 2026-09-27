//! Estimate block samples from neighbours in the same picture.
//!
//! A sample is one component value at a pixel position. Intra prediction means that references come from the same picture. Callers prepare samples above and to the left of a square block, with replacements where a neighbour is unavailable. A prediction mode selects an estimate from those references. Reconstruction then adds a residual, which is a signed change. This module does not decode residuals or update picture storage.
//!
//! Use zero-based column `x` and row `y` within a square of width `w`. Output is row-major at `output[y*w+x]`. Most predictors support widths 4, 8, 16 and write exactly `w*w` elements. The rest of a longer output slice is unchanged. Width and buffer checks precede output writes; mode checks also precede writes where a mode is accepted.
//!
//! # Reference representations
//!
//! ```text
//!                 corner    first top neighbour           extension
//! bank indices:      0          1 ... w                  w+1 ... 2*w
//! coordinates:    (-1,-1)     (0,-1) ... (w-1,-1)        (w,-1) ...
//! left bank:         0          1 ... w                  w+1 ... 2*w
//! coordinates:    (-1,-1)     (-1,0) ... (-1,w-1)        (-1,w) ...
//! ```
//!
//! Raw signed arrays and production colour/alpha banks include the corner. [`PredictionReferences`] instead stores it separately in `top_left`; its `top[0]` and `left[0]` are immediate neighbours. [`SignedReferences`] excludes the corner and is used for planar prediction. [`build_signed_references`] drops the input corner and copies 32 values. The test-only `build_prediction_references` is a reference calculation, not the production builder. Production construction is in `colour::predictive` and `alpha`.
//!
//! “Signed” describes `i16` storage, independent of direction sign. Colour supplies L in 0..255 and biased difference values in 0..511; alpha supplies 0..255. Angular displacement uses units of 1/32 sample. A step of 13 means 13/32 sample per output row or column, not 13 degrees. Interpolation uses two weights summing to 32 and adds 16 before an arithmetic shift of five. Negative projections extend the main edge backwards using the other edge.
//!
//! # Dispatch and preparation
//!
//! The crate page lists all modes. Modes 0/1 copy an edge, 2 averages edges, 3 uses a diagonal, 4..16 select directions, and 17 blends planar ramps. DC prediction means an edge-average estimate; it is distinct from the constant DC coefficient in a transform. A direction-table index is not a mode number. The production dispatchers handle direct-copy and DC cases before calling the general angular helper.
//!
//! Colour filters width-4 modes 3/6/9, width-8 modes 3..8, and width-16 modes 3..16. [`filter_signed_references`] applies the integer 1,2,1 kernel. Width-4 mode 8 is unfiltered despite its helper parameter name. Alpha width 4 dispatches modes other than 3 and 9..16 through [`predict_alpha`]; the production caller leaves its first-row flags false because edge replacements were already made. In particular, the byte mode-6 endpoint rule differs from the colour mode-6 call.
//!
//! DC and planar calculations use integer division, which truncates towards zero for signed values. Filters and angular interpolation use signed shifts, which round negative values downwards. Do not replace one operation with the other. The crate page gives a complete numerical angular block; local contracts describe filter ranges and dedicated direction extensions.

use std::cmp::Ordering;

/// Invalid spatial prediction arguments.
///
/// Predictors accept only 4, 8, or 16 sample square blocks and require an output region large enough for the complete block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PredictionError {
    /// A block width outside 4, 8, and 16.
    #[error("invalid prediction block width")]
    InvalidWidth,
    /// A mode or angular direction unsupported by the selected predictor.
    #[error("unsupported prediction mode")]
    InvalidMode,
    /// Test-only reference construction leaves the source plane.
    #[error("prediction reference is out of bounds")]
    #[cfg(test)]
    OutOfBounds,
    /// Output cannot hold `width * width` samples.
    #[error("prediction output buffer is too small")]
    BufferTooSmall,
}

/// Tests whether a prediction width is supported.
///
/// SPI prediction operates on square blocks of width 4, 8, or 16 only.
fn valid_width(w: usize) -> bool {
    w == 4 || w == 8 || w == 16
}

/// Converts a proven byte-weighted average to `u8`.
///
/// Predictor formulae and byte inputs guarantee the value is in range; violation of that internal invariant panics.
fn byte_sample(value: u32) -> u8 {
    u8::try_from(value).expect("weighted average of bytes fits in u8")
}

/// Converts a proven signed byte-weighted average to `u8`.
///
/// Predictor formulae guarantee a non-negative byte result; violation of that internal invariant panics.
fn byte_sample_signed(value: i32) -> u8 {
    u8::try_from(value).expect("weighted average of bytes fits in u8")
}

/// Converts a proven signed predictor value to `i16`.
///
/// Reference ranges and weighted-average formulae guarantee the conversion; violation of that internal invariant panics.
fn signed_sample(value: i32) -> i16 {
    i16::try_from(value).expect("weighted average of signed samples fits in i16")
}

/// Converts a validated signed reference index to `usize`.
///
/// Callers establish non-negativity through width and angle checks; violation of that internal invariant panics.
fn block_index(value: i32) -> usize {
    usize::try_from(value).expect("validated prediction index is nonnegative")
}

/// Converts a small prediction coordinate to `i32`.
///
/// Coordinates are at most 32 for supported block widths; violation of that internal invariant panics.
fn block_coordinate(value: usize) -> i32 {
    i32::try_from(value).expect("prediction coordinate is at most 32")
}

/// Byte references for a square alpha prediction block.
///
/// `top_left` is the corner at (-1,-1). `top[i]` represents (i,-1) and `left[i]` represents (-1,i); neither array includes the corner. Callers supply extensions beyond the square where a mode needs them. Production width-4 alpha copies eight neighbours and repeats the last value through the remaining capacity.
///
/// `first_row_values[y]` supplies the row value for top-copy mode when `first_row` is true. These flags support the test/reference helper; production alpha leaves `first_row` false after it prepares record-edge replacements in its banks.
#[derive(Clone, Copy, Debug)]
pub struct PredictionReferences {
    /// The sample diagonally above and left of the block.
    pub top_left: u8,
    /// Top-edge samples and their extension.
    pub top: [u8; 33],
    /// Left-edge samples and their extension.
    pub left: [u8; 33],
    /// Replacement values used by top-copy mode on the first row.
    pub first_row_values: [u8; 16],
    /// Enables the byte helper's first-row rules; production alpha leaves this false.
    pub first_row: bool,
}
impl Default for PredictionReferences {
    fn default() -> Self {
        Self {
            top_left: 0,
            top: [0; 33],
            left: [0; 33],
            first_row_values: [0; 16],
            first_row: false,
        }
    }
}

/// Builds the top and left samples available at a block position.
///
/// This test-only reference implementation observes 16-pixel block and 8-pixel sub-block boundaries when extending unavailable samples. It also supports the special first-row edge repetition used by alpha prediction.
///
/// Returns [`PredictionError::InvalidWidth`] for invalid geometry and [`PredictionError::OutOfBounds`] when the plane has too few rows.
#[cfg(test)]
pub fn build_prediction_references(
    plane: &[u8],
    stride: usize,
    x: usize,
    y: usize,
    width: usize,
    repeat_first_row_edge: bool,
    first_row_top: u8,
) -> Result<PredictionReferences, PredictionError> {
    if !valid_width(width) || stride == 0 || x.checked_add(width).filter(|&v| v <= stride).is_none()
    {
        return Err(PredictionError::InvalidWidth);
    }
    let rows = plane.len() / stride;
    if rows < y.saturating_add(width) {
        return Err(PredictionError::OutOfBounds);
    }

    let mut r = PredictionReferences {
        first_row: y == 0,
        top_left: if x == 0 || y == 0 {
            128
        } else {
            plane[(y - 1) * stride + x - 1]
        },
        ..Default::default()
    };

    if repeat_first_row_edge && y == 0 && x != 0 {
        let edge = plane[x - 1];
        r.top_left = edge;
        r.top.fill(edge);
        for i in 0..17 {
            r.left[i] = plane[(y + i.min(width - 1)) * stride + x - 1];
        }
        for row in 0..width {
            r.first_row_values[row] = plane[y * stride + row * stride + x - 1];
        }
        return Ok(r);
    }
    let block_x = (x / 16) * 16;
    let block_y = (y / 16) * 16;
    let top_limit = if y.is_multiple_of(16) {
        stride - 1
    } else if width == 4 && y % 8 == 4 {
        ((x / 8) * 8 + 7).min(stride - 1)
    } else {
        block_x.saturating_add(15).min(stride - 1)
    };
    let left_limit = if width == 4 && x.is_multiple_of(8) {
        ((y / 8) * 8 + 7).min(rows - 1)
    } else if x.is_multiple_of(16) {
        block_y.saturating_add(15).min(rows - 1)
    } else {
        (y + width - 1).min(rows - 1)
    };
    for i in 0..17 {
        r.top[i] = if y == 0 {
            if x == 0 { 128 } else { first_row_top }
        } else {
            plane[(y - 1) * stride + (x + i).min(top_limit)]
        };
        r.left[i] = if x == 0 {
            128
        } else {
            plane[(y + i).min(left_limit) * stride + x - 1]
        };
    }
    for row in 0..width {
        let start = (x / 16) * 16;
        r.first_row_values[row] = if start == 0 {
            128
        } else {
            plane[(y + row) * stride + start - 1]
        };
    }
    Ok(r)
}

/// Applies the three-tap `1, 2, 1` edge filter to byte samples.
///
/// The sum includes a rounding bias of two before division by four.
pub fn filter3(a: u8, b: u8, c: u8) -> u8 {
    byte_sample(u32::from(
        (u16::from(a) + 2 * u16::from(b) + u16::from(c) + 2) >> 2,
    ))
}

/// Returns one raw or filtered top or left edge sample.
///
/// Width-four prediction uses the middle sample unchanged. Larger widths apply [`filter3`], with `top_left` supplying the preceding sample at index zero.
fn edge_filter(references: &PredictionReferences, top: bool, index: usize, width: usize) -> u8 {
    let a = if top {
        if index == 0 {
            references.top_left
        } else {
            references.top[index - 1]
        }
    } else if index == 0 {
        references.top_left
    } else {
        references.left[index - 1]
    };
    let m = if top {
        references.top[index]
    } else {
        references.left[index]
    };
    let z = if top {
        references.top[index + 1]
    } else {
        references.left[index + 1]
    };
    if width == 4 { m } else { filter3(a, m, z) }
}

/// Calculates one alpha DC-predicted sample.
///
/// The base value averages `width` top and left samples with codec rounding. Width four then filters the first row and column towards their adjacent references.
fn alpha_dc(references: &PredictionReferences, width: usize, row: usize, col: usize) -> u8 {
    let mut sum = u32::try_from(width).expect("validated width is at most 16");
    for i in 0..width {
        sum += u32::from(references.top[i]) + u32::from(references.left[i]);
    }
    let dc = sum / u32::try_from(2 * width).expect("validated width is at most 16");
    if width != 4 {
        byte_sample(dc)
    } else if row == 0 && col == 0 {
        byte_sample(
            (3 * (u32::from(references.left[0]) + u32::from(references.top[0])) + 2 * dc + 4) >> 3,
        )
    } else if row == 0 {
        byte_sample((3 * u32::from(references.top[col]) + 5 * dc + 4) >> 3)
    } else if col == 0 {
        byte_sample((3 * u32::from(references.left[row]) + 5 * dc + 4) >> 3)
    } else {
        byte_sample(dc)
    }
}

/// Calculates one top-oriented inverse-angular alpha sample.
///
/// The procedure constructs an extended main reference with the fixed 630 inverse angle and interpolates with a direction step of negative 13.
fn alpha_angular_top(
    references: &PredictionReferences,
    width: usize,
    row: usize,
    col: usize,
) -> u8 {
    let mut main = [0_u8; 35];
    let origin = 17;
    main[origin] = references.top_left;
    for i in 0..17 {
        main[origin + i + 1] = references.top[i];
    }
    let mut sum = 128_i32;
    for n in 1..=width {
        sum += 630;
        let idx = usize::try_from(sum >> 8).expect("angular index is nonnegative");
        main[origin - n] = if idx == 0 {
            references.top_left
        } else {
            references.left[idx - 1]
        };
    }
    let d = -((block_coordinate(row) + 1) * 13);
    let integer = d >> 5;
    let frac = d & 31;
    let i = block_coordinate(col) + integer + 1;
    byte_sample_signed(
        ((32 - frac) * i32::from(main[block_index(block_coordinate(origin) + i)])
            + frac * i32::from(main[block_index(block_coordinate(origin) + i + 1)])
            + 16)
            >> 5,
    )
}

/// Calculates one left-oriented inverse-angular alpha sample.
///
/// The procedure uses raw references at width four and filtered references otherwise, extends the main reference from the top edge, and interpolates with a direction step of negative 13.
fn alpha_angular_left(
    references: &PredictionReferences,
    width: usize,
    row: usize,
    col: usize,
) -> u8 {
    let corner = if width == 4 {
        references.top_left
    } else {
        filter3(references.left[0], references.top_left, references.top[0])
    };
    let mut main = [0_u8; 35];
    let origin = 17;
    main[origin] = corner;
    for i in 0..17 {
        main[origin + i + 1] = edge_filter(references, false, i, width);
    }
    for n in 1_usize..=16 {
        let idx = ((n * 630 + 128) >> 8).saturating_sub(1).min(16);
        main[origin - n] = edge_filter(references, true, idx, width);
    }
    let d = -((block_coordinate(col) + 1) * 13);
    let integer = if d < 0 { -((-d + 31) / 32) } else { d / 32 };
    let frac = d - integer * 32;
    let i = block_coordinate(row) + integer + 1;
    byte_sample_signed(
        ((32 - frac) * i32::from(main[block_index(block_coordinate(origin) + i)])
            + frac * i32::from(main[block_index(block_coordinate(origin) + i + 1)])
            + 16)
            >> 5,
    )
}

/// Calculates one filtered diagonal alpha sample.
///
/// Samples above the diagonal come from the top edge, samples below it come from the left edge, and diagonal samples filter across the top-left corner.
fn alpha_diagonal(references: &PredictionReferences, row: usize, col: usize) -> u8 {
    match col.cmp(&row) {
        Ordering::Greater => {
            let top_index = col - row - 1;
            filter3(
                if top_index == 0 {
                    references.top_left
                } else {
                    references.top[top_index - 1]
                },
                references.top[top_index],
                references.top[top_index + 1],
            )
        }
        Ordering::Less => {
            let left_index = row - col - 1;
            filter3(
                if left_index == 0 {
                    references.top_left
                } else {
                    references.left[left_index - 1]
                },
                references.left[left_index],
                references.left[left_index + 1],
            )
        }
        Ordering::Equal => filter3(references.left[0], references.top_left, references.top[0]),
    }
}

/// Calculates one bilinear planar alpha sample.
///
/// Horizontal and vertical ramps meet the far left and top samples. When requested for the first row, unavailable top contributions are replaced with zero.
fn alpha_planar(
    references: &PredictionReferences,
    width: usize,
    row: usize,
    col: usize,
    first_row_top_unavailable: bool,
) -> u8 {
    let e = width - 1;
    let te = if references.first_row && first_row_top_unavailable {
        0
    } else {
        references.top[e]
    };
    let tv = if references.first_row && first_row_top_unavailable {
        0
    } else {
        references.top[col]
    };
    let h = u32::try_from(e - col).expect("validated width is at most 16")
        * u32::from(references.left[row])
        + u32::try_from(col + 1).expect("validated width is at most 16") * u32::from(te);
    let v = u32::try_from(e - row).expect("validated width is at most 16") * u32::from(tv)
        + u32::try_from(row + 1).expect("validated width is at most 16")
            * u32::from(references.left[e]);
    byte_sample(
        (h + v + u32::try_from(width).expect("validated width is at most 16"))
            / u32::try_from(2 * width).expect("validated width is at most 16"),
    )
}

/// Predict byte alpha samples from corner-excluding references.
///
/// Widths 4,8,16 and modes 0..8,17 are accepted. Output is row-major; only the first `width*width` entries change. Mode 0 copies top samples, except `first_row` selects `first_row_values[y]` across each row. Mode 1 copies left samples. Mode 2 averages both edges and adjusts boundaries only at width 4. Mode 3 filters across the corner diagonal. Modes 4/7 project negatively along top/left; modes 5/8 project positively by 13/32 sample per row/column. Mode 6 filters the advancing top diagonal and uses the final extension directly at its last endpoint. Mode 17 blends planar ramps.
///
/// If both first-row flags are set, mode 2 returns zero at width 8, mode 3 returns zero at all widths, and mode 17 replaces its top contributions with zero. These options are exercised by reference tests; production width-4 alpha leaves both flags false. Production dispatch uses this helper for width-4 modes 0,1,2,4,5,6,7,8,17, and shared signed helpers for other cases.
///
/// # Errors
///
/// Unsupported width or mode, or short output, returns [`PredictionError`] before writes.
pub fn predict_alpha(
    references: &PredictionReferences,
    width: usize,
    mode: u8,
    output: &mut [u8],
    first_row_top_unavailable: bool,
) -> Result<(), PredictionError> {
    if !valid_width(width) {
        return Err(PredictionError::InvalidWidth);
    }
    if output.len() < width * width {
        return Err(PredictionError::BufferTooSmall);
    }
    if !(mode <= 8 || mode == 17) {
        return Err(PredictionError::InvalidMode);
    }
    for row in 0..width {
        for col in 0..width {
            let v = match mode {
                0 => {
                    if references.first_row {
                        references.first_row_values[row]
                    } else {
                        references.top[col]
                    }
                }
                1 => references.left[row],
                2 if width == 8 && references.first_row && first_row_top_unavailable => 0,
                2 => alpha_dc(references, width, row, col),
                3 if references.first_row && first_row_top_unavailable => 0,
                3 => alpha_diagonal(references, row, col),
                4 => alpha_angular_top(references, width, row, col),
                5 => {
                    let d = (row + 1) * 13;
                    let i = col + (d >> 5);
                    let f = d & 31;
                    byte_sample(
                        ((u32::try_from(32 - f).expect("fraction is at most 31"))
                            * u32::from(edge_filter(references, true, i, width))
                            + u32::try_from(f).expect("fraction is nonnegative")
                                * u32::from(edge_filter(references, true, i + 1, width))
                            + 16)
                            >> 5,
                    )
                }
                6 => {
                    let i = col + row;
                    if i >= 2 * width - 2 {
                        references.top[2 * width - 1]
                    } else {
                        filter3(
                            references.top[i],
                            references.top[i + 1],
                            references.top[i + 2],
                        )
                    }
                }
                7 => alpha_angular_left(references, width, row, col),
                8 => {
                    let d = (col + 1) * 13;
                    let i = row + (d >> 5);
                    let f = d & 31;
                    byte_sample(
                        (u32::try_from(32 - f).expect("fraction is at most 31")
                            * u32::from(edge_filter(references, false, i, width))
                            + u32::try_from(f).expect("fraction is nonnegative")
                                * u32::from(edge_filter(references, false, i + 1, width))
                            + 16)
                            >> 5,
                    )
                }
                17 => alpha_planar(references, width, row, col, first_row_top_unavailable),
                _ => return Err(PredictionError::InvalidMode),
            };
            output[row * width + col] = v;
        }
    }
    Ok(())
}

/// Corner-excluding signed edges for planar prediction.
///
/// `top[0]` and `left[0]` are the immediate neighbours above and left of (0,0). Planar prediction reads indices 0..width and uses index `width-1` as the far endpoint. The constructor copies 32 samples from corner-inclusive histories and leaves index 32 zero. “Signed” refers to `i16` storage; production values include biased colour differences up to 511.
#[derive(Clone, Copy, Debug)]
pub struct SignedReferences {
    /// 33 top-edge samples.
    pub top: [i16; 33],
    /// 33 left-edge samples.
    pub left: [i16; 33],
}
impl Default for SignedReferences {
    fn default() -> Self {
        Self {
            top: [0; 33],
            left: [0; 33],
        }
    }
}

/// Copies persistent histories into the layout used by planar prediction.
///
/// History element zero is the corner and is dropped; the following 32 values become zero-based edge arrays.
pub fn build_signed_references(
    left_history: &[i16; 33],
    top_history: &[i16; 33],
) -> SignedReferences {
    let mut refs = SignedReferences::default();
    refs.left[..32].copy_from_slice(&left_history[1..]);
    refs.top[..32].copy_from_slice(&top_history[1..]);
    refs
}

/// Predict colour mode 4 from a top edge with negative displacement.
///
/// Both inputs are corner-inclusive arrays. Colour passes raw references at width 4 and filtered references at widths 8 and 16. Copy the complete top array to main-reference indices 0..32. For negative indices -1 down to -2, -3, or -6 at widths 4, 8, or 16, start an accumulator at 128, add 630 per index, and select `left[accumulator >> 8]`. Other temporary entries remain zero. The value 630 is an inverse projection factor in units of 1/256; it is not degrees.
///
/// At output column `x`, row `y`, use displacement `d=-(y+1)*13`, integer part `d>>5`, and fraction `d&31`. Interpolate the two main-reference values at `x+(d>>5)+1` and the next index with weights `32-f` and `f`, add 16, then shift right by five in `i32`. This projects the top edge by -13/32 sample per row. Write the first `width*width` elements in row order; leave later output unchanged.
///
/// # Errors
///
/// Invalid width (not 4, 8, 16) or short output returns [`PredictionError`] before writes.
pub fn predict_signed_wide_angular_reference_a(
    filtered_left: &[i16; 33],
    filtered_top: &[i16; 33],
    width: usize,
    output: &mut [i16],
) -> Result<(), PredictionError> {
    if !valid_width(width) {
        return Err(PredictionError::InvalidWidth);
    }
    if output.len() < width * width {
        return Err(PredictionError::BufferTooSmall);
    }
    let mut main = [0_i16; 65];
    const ORIGIN: i32 = 32;
    for i in 0..33 {
        main[block_index(ORIGIN + block_coordinate(i))] = filtered_top[i];
    }
    let last_negative = match width {
        4 => -2,
        8 => -3,
        _ => -6,
    };
    let mut inverse_sum = 128_i32;
    for offset in 1..=(-last_negative) {
        let index = -offset;
        inverse_sum += 630;
        main[block_index(ORIGIN + index)] = filtered_left[block_index(inverse_sum >> 8)];
    }
    for row in 0..width {
        let displacement = -((block_coordinate(row) + 1) * 13);
        let integer = displacement >> 5;
        let fraction = displacement & 31;
        for col in 0..width {
            let first = block_index(ORIGIN + block_coordinate(col) + integer + 1);
            let value = (32 - fraction) * i32::from(main[first])
                + fraction * i32::from(main[first + 1])
                + 16;
            output[row * width + col] = signed_sample(value >> 5);
        }
    }
    Ok(())
}

/// Predicts the signed top-edge directional mode.
///
/// Each row shifts the top reference one sample farther right, so output at `(row, column)` uses top index `row + column + 2`.
///
/// Returns [`PredictionError`] for an invalid width or short output.
pub fn predict_signed_top_edge_directional_signed(
    filtered_top: &[i16; 33],
    width: usize,
    output: &mut [i16],
) -> Result<(), PredictionError> {
    if !valid_width(width) {
        return Err(PredictionError::InvalidWidth);
    }
    if output.len() < width * width {
        return Err(PredictionError::BufferTooSmall);
    }
    for row in 0..width {
        for col in 0..width {
            output[row * width + col] = filtered_top[row + col + 2];
        }
    }
    Ok(())
}

/// Predict colour mode 7 from a filtered left edge.
///
/// Width must be 8 or 16. Inputs include the corner at zero. Main index zero uses `filtered_top[0]`; positive indices 1..32 use left indices 1..32. For a negative main index `-n`, use top index `1 + min(max(((n*630+128)>>8)-1,0),31)`. The temporary negative extent is the block width. Here 630 is an inverse projection factor scaled by 256.
///
/// For output `(x,y)`, displacement is `-(x+1)*13` in units of 1/32 sample. Round its integer part downwards, obtain a fraction 0..31, and interpolate main indices `y+integer+1` and the next index with a rounding offset of 16 before `>>5`. Products use `i32`. Writes only the first `width*width` output elements in row order.
///
/// # Errors
///
/// Returns [`PredictionError`] before writes for a width other than 8 or 16, or a short output slice.
pub fn predict_signed_left_edge_directional(
    filtered_left: &[i16; 33],
    filtered_top: &[i16; 33],
    width: usize,
    output: &mut [i16],
) -> Result<(), PredictionError> {
    if width != 8 && width != 16 {
        return Err(PredictionError::InvalidWidth);
    }
    if output.len() < width * width {
        return Err(PredictionError::BufferTooSmall);
    }
    const ORIGIN: i32 = 32;
    let mut main = [0_i16; 65];
    main[block_index(ORIGIN)] = filtered_top[0];
    for i in 0..32 {
        main[block_index(ORIGIN) + i + 1] = filtered_left[i + 1];
    }
    for index in -block_coordinate(width)..0 {
        let side_index = block_index((-index * 630 + 128) >> 8)
            .saturating_sub(1)
            .min(31);
        main[block_index(ORIGIN + index)] = filtered_top[side_index + 1];
    }
    for row in 0..width {
        for col in 0..width {
            let displacement = -((block_coordinate(col) + 1) * 13);
            let integer = -((-displacement + 31) / 32);
            let fraction = displacement - integer * 32;
            let first = block_index(ORIGIN + block_coordinate(row) + integer + 1);
            let value = (32 - fraction) * i32::from(main[first])
                + fraction * i32::from(main[first + 1])
                + 16;
            output[row * width + col] = signed_sample(value >> 5);
        }
    }
    Ok(())
}

/// Predict a width-4 square for modes 5, 6, or 7.
///
/// Inputs include a corner at index zero. Production colour uses this helper for unfiltered modes 5 and 7; its mode-6 path uses another helper. Mode 5 projects along the top array by `(y+1)*13/32` samples and interpolates adjacent values. Mode 6 uses `(top[x+y+1]+2*top[x+y+2]+top[x+y+3]+2)>>2`, except the last diagonal uses `top[8]` directly.
///
/// Mode 7 requires `left`. Main index zero is `left[0]`, positive indices 1..17 are left indices 1..17. For `n=1..=16`, negative index `-n` uses `top[1+min(max(((n*630+128)>>8)-1,0),16)]`. Displacement is `-(x+1)*13/32` along the left edge. Both interpolating modes use weights summing to 32, add 16, then shift right by five in `i32`. The first 16 output elements are written in row order; later elements are unchanged.
///
/// # Errors
///
/// A short output, unsupported mode, or missing mode-7 left edge returns [`PredictionError`] before writes.
pub fn predict_signed_width4_angular(
    top: &[i16; 33],
    mode: u8,
    left: Option<&[i16; 33]>,
    output: &mut [i16],
) -> Result<(), PredictionError> {
    if output.len() < 16 {
        return Err(PredictionError::BufferTooSmall);
    }
    if !matches!(mode, 5..=7) || (mode == 7 && left.is_none()) {
        return Err(PredictionError::InvalidMode);
    }
    let mut main = [0_i16; 34];
    if mode == 7 {
        let l = left.expect("validated above");
        const ORIGIN: i32 = 16;
        main[block_index(ORIGIN)] = l[0];
        for i in 0..17 {
            main[block_index(ORIGIN) + i + 1] = l[i + 1];
        }
        for index in (-16_i32..=-1).rev() {
            let side_index = block_index((-index * 630 + 128) >> 8)
                .saturating_sub(1)
                .min(16);
            main[block_index(ORIGIN + index)] = top[side_index + 1];
        }
    }
    for row in 0..4 {
        for col in 0..4 {
            let value = if mode == 5 {
                let displacement = (row + 1) * 13;
                let reference = col + (displacement >> 5);
                let fraction = displacement & 31;
                (block_coordinate(32 - fraction) * i32::from(top[reference + 1])
                    + block_coordinate(fraction) * i32::from(top[reference + 2])
                    + 16)
                    >> 5
            } else if mode == 6 {
                let reference = col + row;
                if reference >= 6 {
                    i32::from(top[8])
                } else {
                    (i32::from(top[reference + 1])
                        + 2 * i32::from(top[reference + 2])
                        + i32::from(top[reference + 3])
                        + 2)
                        >> 2
                }
            } else {
                const ORIGIN: i32 = 16;
                let displacement = -((block_coordinate(col) + 1) * 13);
                let integer = -((-displacement + 31) / 32);
                let fraction = displacement - integer * 32;
                let first = block_index(ORIGIN + block_coordinate(row) + integer + 1);
                ((32 - fraction) * i32::from(main[first])
                    + fraction * i32::from(main[first + 1])
                    + 16)
                    >> 5
            };
            output[row * 4 + col] = signed_sample(value);
        }
    }
    Ok(())
}

/// Predicts the dedicated width-four signed diagonal mode.
///
/// Values above and below the diagonal come from top and left references respectively. The diagonal repeats the left corner. Output always contains 16 samples.
///
/// Returns [`PredictionError::BufferTooSmall`] for fewer than 16 output elements.
pub fn predict_signed_width4_diagonal(
    filtered_left: &[i16; 33],
    filtered_top: &[i16; 33],
    output: &mut [i16],
) -> Result<(), PredictionError> {
    if output.len() < 16 {
        return Err(PredictionError::BufferTooSmall);
    }
    for row in 0..4 {
        for col in 0..4 {
            output[row * 4 + col] = match col.cmp(&row) {
                std::cmp::Ordering::Greater => filtered_top[col - row],
                std::cmp::Ordering::Less => filtered_left[row - col],
                std::cmp::Ordering::Equal => filtered_left[0],
            };
        }
    }
    Ok(())
}

/// Filter corner-inclusive signed edges before selected prediction modes.
///
/// Inputs use index 0 for the corner, indices 1..=width for immediate edge samples, and indices width+1..=2*width for extensions. For each interior index `i` from 1 to `2*width-1`, write `(edge[i-1]+2*edge[i]+edge[i+1]+2)>>2`. Arithmetic uses `i32`, then a checked `i16` conversion. For example, neighbours 100, 120, 140 give 120.
///
/// Both output corners use `(left[1]+2*left[0]+top[1]+2)>>2`; the input `top[0]` does not contribute to this shared corner. Copy index `2*width` unchanged. Leave larger output indices unchanged. Filtering thus changes a bounded active prefix, not the whole 33-element capacity. Inputs are not modified.
///
/// # Errors
///
/// Returns [`PredictionError::InvalidWidth`] before writes unless width is 4, 8, or 16.
pub fn filter_signed_references(
    left: &[i16; 33],
    top: &[i16; 33],
    width: usize,
    out_left: &mut [i16; 33],
    out_top: &mut [i16; 33],
) -> Result<(), PredictionError> {
    if !valid_width(width) {
        return Err(PredictionError::InvalidWidth);
    }
    let end = 2 * width;
    let corner = (i32::from(left[1]) + 2 * i32::from(left[0]) + i32::from(top[1]) + 2) >> 2;
    out_left[0] = signed_sample(corner);
    out_top[0] = signed_sample(corner);
    for i in 1..end {
        out_left[i] = signed_sample(
            (i32::from(left[i - 1]) + 2 * i32::from(left[i]) + i32::from(left[i + 1]) + 2) >> 2,
        );
        out_top[i] = signed_sample(
            (i32::from(top[i - 1]) + 2 * i32::from(top[i]) + i32::from(top[i + 1]) + 2) >> 2,
        );
    }
    out_left[end] = left[end];
    out_top[end] = top[end];
    Ok(())
}

/// Fill a square with an edge average and adjusted boundary samples.
///
/// Inputs include a corner at index zero, which is excluded from the average. Sum indices 1..=width of both edges, add `width`, then divide by `2*width` in `i32` to obtain `dc`. Fill the square with that value, then adjust the corner and first row and column. Integer division truncates towards zero.
///
/// | Width | Output corner | Other first-row sample for top neighbour `t` |
/// | --- | --- | --- |
/// | 4 | `(3*(left[1]+top[1])+2*dc+4)>>3` | `(3*t+5*dc+4)>>3` |
/// | 8 | `(left[1]+top[1]+2*dc+2)>>2` | `(t+3*dc+2)>>2` |
/// | 16 | `(left[1]+top[1]+6*dc+4)>>3` | `(t+7*dc+4)>>3` |
///
/// The first column uses the same rule with its left neighbour. For width 4 with left samples 10 and top samples 20, `dc=15`, output corner is 15, the remaining first row is 17, the remaining first column is 13, and the interior is 15. Writes the first `width*width` row-major entries only.
///
/// # Errors
///
/// Returns [`PredictionError`] for unsupported width or short output before writes.
pub fn predict_signed_dc(
    left: &[i16; 33],
    top: &[i16; 33],
    width: usize,
    output: &mut [i16],
) -> Result<(), PredictionError> {
    if !valid_width(width) {
        return Err(PredictionError::InvalidWidth);
    }
    if output.len() < width * width {
        return Err(PredictionError::BufferTooSmall);
    }
    let mut total = block_coordinate(width);
    for i in 1..=width {
        total += i32::from(left[i]) + i32::from(top[i]);
    }
    let dc = total / block_coordinate(2 * width);
    output[..width * width].fill(signed_sample(dc));
    output[0] = if width == 4 {
        signed_sample((3 * (i32::from(left[1]) + i32::from(top[1])) + 2 * dc + 4) >> 3)
    } else if width == 8 {
        signed_sample((i32::from(left[1]) + i32::from(top[1]) + 2 * dc + 2) >> 2)
    } else {
        signed_sample((i32::from(left[1]) + i32::from(top[1]) + 6 * dc + 4) >> 3)
    };
    let (ew, er, shift, sw) = if width == 4 {
        (5, 4, 3, 3)
    } else if width == 8 {
        (3, 2, 2, 1)
    } else {
        (7, 4, 3, 1)
    };
    for i in 1..width {
        output[i] = signed_sample((sw * i32::from(top[i + 1]) + ew * dc + er) >> shift);
        output[i * width] = signed_sample((sw * i32::from(left[i + 1]) + ew * dc + er) >> shift);
    }
    Ok(())
}

/// Map a prediction mode to its direction-table index.
///
/// For modes 0..16 the indices are `[5,13,0,1,3,7,9,11,15,17,2,4,6,8,10,12,14]`. Mode 17 and larger values return `None`. This table does not mean every result can be passed to [`predict_signed_angular`]: indices 0, 5, and 13 represent cases handled by separate predictors. Production dispatch uses the general angular helper for modes 9..16.
pub fn prediction_direction_for_mode(mode: u8) -> Option<u8> {
    const D: [u8; 17] = [5, 13, 0, 1, 3, 7, 9, 11, 15, 17, 2, 4, 6, 8, 10, 12, 14];
    D.get(mode as usize).copied()
}

/// Predicts the filtered signed diagonal mode for width 8 or 16.
///
/// Samples on and above the diagonal use the top edge, while samples below it use the left edge.
///
/// Returns [`PredictionError`] for another width or short output.
pub fn predict_signed_filtered_diagonal(
    left: &[i16; 33],
    top: &[i16; 33],
    width: usize,
    output: &mut [i16],
) -> Result<(), PredictionError> {
    if width != 8 && width != 16 {
        return Err(PredictionError::InvalidWidth);
    }
    if output.len() < width * width {
        return Err(PredictionError::BufferTooSmall);
    }
    for row in 0..width {
        for col in 0..width {
            output[row * width + col] = if col >= row {
                top[col - row]
            } else {
                left[row - col]
            };
        }
    }
    Ok(())
}

/// Predict colour mode 5 by positive top-oriented interpolation.
///
/// This width-8-or-16 mode advances by 13/32 sample for each row and linearly interpolates adjacent top references.
///
/// Returns [`PredictionError`] for another width or short output.
pub fn predict_signed_angular_reference_b(
    top: &[i16; 33],
    width: usize,
    output: &mut [i16],
) -> Result<(), PredictionError> {
    if width != 8 && width != 16 {
        return Err(PredictionError::InvalidWidth);
    }
    if output.len() < width * width {
        return Err(PredictionError::BufferTooSmall);
    }
    for row in 0..width {
        let displacement = (row + 1) * 13;
        let integer = displacement >> 5;
        let fraction = displacement & 31;
        for col in 0..width {
            let i = col + integer + 1;
            let value = block_coordinate(32 - fraction) * i32::from(top[i])
                + block_coordinate(fraction) * i32::from(top[i + 1])
                + 16;
            output[row * width + col] = signed_sample(value >> 5);
        }
    }
    Ok(())
}

/// Fill a square with estimates derived from its left reference samples.
///
/// Input and output use `i16`; “signed” describes sample storage, not direction. The positive direction moves down the left reference array as the output column moves right. The input includes a corner at index 0 and the first immediate left neighbour at index 1. The caller supplies the required extension below the block.
///
/// Let `x` and `y` be zero-based column and row inside the output square. The displacement along the left edge is `(x+1)*13/32` samples. The constant 13 is a step in units of 1/32 sample, not degrees. Calculate `d=(x+1)*13`, `i=y+(d>>5)+1`, and `f=d&31`. The output is `((32-f)*left[i] + f*left[i+1] + 16) >> 5`, with products and sums in `i32`. Adding 16 supplies the rounding offset before the signed shift. For (0,0), weights are 19 and 13. Values 100 and 132 give `(1900+1716+16)>>5 = 113`.
///
/// This helper does not filter. Colour mode 8 passes unfiltered references at width 4 and filtered references at widths 8 and 16. The parameter name `filtered_left` therefore does not describe all calls. [`filter_signed_references`] specifies the filter and endpoint rules.
///
/// Writes exactly `width*width` samples in row order at `output[y*width+x]`; later elements are unchanged. Inputs remain unchanged.
///
/// # Errors
///
/// Returns [`PredictionError::InvalidWidth`] unless width is 4, 8, or 16, or [`PredictionError::BufferTooSmall`] for short output. Both checks occur before writes.
pub fn predict_signed_side_directional(
    filtered_left: &[i16; 33],
    width: usize,
    output: &mut [i16],
) -> Result<(), PredictionError> {
    if !valid_width(width) {
        return Err(PredictionError::InvalidWidth);
    }
    if output.len() < width * width {
        return Err(PredictionError::BufferTooSmall);
    }
    for row in 0..width {
        for col in 0..width {
            let displacement = (col + 1) * 13;
            let index = row + (displacement >> 5) + 1;
            let fraction = displacement & 31;
            let value = block_coordinate(32 - fraction) * i32::from(filtered_left[index])
                + block_coordinate(fraction) * i32::from(filtered_left[index + 1])
                + 16;
            output[row * width + col] = signed_sample(value >> 5);
        }
    }
    Ok(())
}

/// Predict a square from a direction-table index and corner-inclusive edges.
///
/// For a direction below 10, the main edge is top and the side edge is left; subtract 5 from the direction. For a direction of 10 or more, the main edge is left and the side is top; subtract 13. The absolute difference 1,2,3,4 selects displacement magnitude 5,13,21,32 in units of 1/32 sample. The difference's sign selects the displacement sign. Other differences are invalid.
///
/// Main indices 0..32 copy the supplied corner and neighbours. For negative displacement, construct indices -1..-width: start `sum=128`, add inverse factor 1638,630,390,256 for the selected magnitude, then copy `side[min(sum>>8,32)]`. Clamping the side index repeats its final sample. The temporary array places logical index zero at physical index 32.
///
/// For top orientation, axis is row and cross-coordinate is column; for left orientation these are column and row. Calculate `d=(axis+1)*step`, `integer=d>>5`, and `f=d-integer*32`. Interpolate main indices `cross+integer+1` and its successor with weights `32-f`, `f`, offset 16, then arithmetic `>>5`. Products use `i32`. Writes only the first `width*width` row-major output elements.
///
/// # Errors
///
/// Invalid width, short output, or unsupported direction returns [`PredictionError`] before any writes. Valid directions are 1..4, 6..9, 10..12, and 14..17.
pub fn predict_signed_angular(
    left: &[i16; 33],
    top: &[i16; 33],
    width: usize,
    direction: u8,
    output: &mut [i16],
) -> Result<(), PredictionError> {
    if !valid_width(width) {
        return Err(PredictionError::InvalidWidth);
    }
    if output.len() < width * width {
        return Err(PredictionError::BufferTooSmall);
    }
    if direction == 0 || direction > 17 {
        return Err(PredictionError::InvalidMode);
    }
    const ANG: [i32; 9] = [0, 5, 13, 21, 32, 0, 0, 0, 0];
    const INV: [i32; 5] = [0, 1638, 630, 390, 256];
    let horizontal = direction >= 10;
    let delta = i32::from(direction) - if horizontal { 13 } else { 5 };
    let ai = delta.unsigned_abs() as usize;
    if ai >= ANG.len() || ANG[ai] == 0 {
        return Err(PredictionError::InvalidMode);
    }
    let angle = if delta < 0 { -ANG[ai] } else { ANG[ai] };
    let main = if horizontal { left } else { top };
    let side = if horizontal { top } else { left };
    let mut refs = [0_i16; 65];
    let origin = 32_i32;
    refs[32..65].copy_from_slice(main);
    if angle < 0 {
        let mut sum = 128_i32;
        for i in 1..=width {
            sum += INV[ai];
            // Large inverse angles can project past the persistent 33-sample
            // side history even when the predictor never consumes that far.
            // Extend by repeating the final available sample.
            let side_index = block_index(sum >> 8).min(side.len() - 1);
            refs[block_index(origin - block_coordinate(i))] = side[side_index];
        }
    }
    for r in 0..width {
        for c in 0..width {
            let axis = if horizontal { c } else { r };
            let cross = if horizontal { r } else { c };
            let d = (block_coordinate(axis) + 1) * angle;
            let integer = d >> 5;
            let frac = d - integer * 32;
            let i = block_index(origin + block_coordinate(cross) + integer + 1);
            let val = (32 - frac) * i32::from(refs[i]) + frac * i32::from(refs[i + 1]) + 16;
            output[r * width + c] = signed_sample(val >> 5);
        }
    }
    Ok(())
}

/// Blend horizontal and vertical ramps across a square.
///
/// `refs` excludes the corner. Let `e=width-1`, and let `x,y` be zero-based column and row. The horizontal contribution is `(e-x)*left[y] + (x+1)*top[e]`. The vertical contribution is `(e-y)*top[x] + (y+1)*left[e]`. Add both and `width`, then divide by `2*width` in `i32`. Division truncates towards zero; the far endpoints are the final in-width neighbours, not extension samples.
///
/// For constant edges of value 20, each ramp contributes `20*width`, so every prediction is 20. The first `width*width` output entries are written in row order; later entries are unchanged.
///
/// # Errors
///
/// Returns [`PredictionError`] for width outside 4,8,16 or short output before writes.
pub fn predict_signed_planar(
    refs: &SignedReferences,
    width: usize,
    output: &mut [i16],
) -> Result<(), PredictionError> {
    if !valid_width(width) {
        return Err(PredictionError::InvalidWidth);
    }
    if output.len() < width * width {
        return Err(PredictionError::BufferTooSmall);
    }
    let e = width - 1;
    for r in 0..width {
        for c in 0..width {
            let h = block_coordinate(e - c) * i32::from(refs.left[r])
                + block_coordinate(c + 1) * i32::from(refs.top[e]);
            let v = block_coordinate(e - r) * i32::from(refs.top[c])
                + block_coordinate(r + 1) * i32::from(refs.left[e]);
            output[r * width + c] =
                signed_sample((h + v + block_coordinate(width)) / block_coordinate(2 * width));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_dc_edges_match_fixed_rounding_vector() {
        let mut left = [0_i16; 33];
        let mut top = [0_i16; 33];
        left[1..=4].fill(10);
        top[1..=4].fill(20);
        let mut out = [0_i16; 16];
        predict_signed_dc(&left, &top, 4, &mut out).unwrap();
        assert_eq!(out[0], 15);
        assert_eq!(out[1], 17);
        assert_eq!(out[4], 13);
        assert_eq!(out[5], 15);
    }

    #[test]
    fn alpha_top_mode_uses_first_row_side_values() {
        let refs = PredictionReferences {
            first_row: true,
            top: [99; 33],
            first_row_values: [7, 8, 9, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ..PredictionReferences::default()
        };
        let mut out = [0_u8; 16];
        predict_alpha(&refs, 4, 0, &mut out, false).unwrap();
        assert_eq!(out, [7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10]);
    }

    #[test]
    fn reference_builder_rejects_short_plane() {
        assert!(matches!(
            build_prediction_references(&[0; 15], 4, 0, 0, 4, false, 0),
            Err(PredictionError::OutOfBounds)
        ));
    }

    #[test]
    fn width4_angular_callbacks_have_fixed_vectors() {
        let top = core::array::from_fn(|i| i16::try_from(i * 10).expect("test index fits i16"));
        let left = [100_i16; 33];
        let mut out = [0_i16; 16];
        predict_signed_width4_angular(&top, 5, None, &mut out).unwrap();
        assert_eq!(out[0], 14);
        predict_signed_width4_angular(&top, 6, None, &mut out).unwrap();
        assert_eq!(out[0], 20);
        assert_eq!(out[15], 80);
        predict_signed_width4_angular(&[0; 33], 7, Some(&left), &mut out).unwrap();
        assert_eq!(out[0], 100);
        assert_eq!(out[3], 38);
        predict_signed_width4_angular(&top, 7, Some(&left), &mut out).unwrap();
        assert_eq!(out[3], 50);
    }

    #[test]
    fn signed_reference_a_and_top_edge_keep_halfword_values() {
        let left = [100_i16; 33];
        let top = [0_i16; 33];
        let mut out = [0_i16; 64];
        predict_signed_wide_angular_reference_a(&left, &top, 8, &mut out).unwrap();
        assert_eq!(out[0], 0);
        assert_eq!(out[7 * 8], 100);
        let ramp = core::array::from_fn(|i| i16::try_from(i).expect("test index fits i16") * 10);
        predict_signed_top_edge_directional_signed(&ramp, 8, &mut out).unwrap();
        assert_eq!(out[0], 20);
        assert_eq!(out[7 * 8 + 7], 160);
        predict_signed_top_edge_directional_signed(&ramp, 4, &mut out).unwrap();
        assert_eq!(out[15], 80);
    }

    #[test]
    fn signed_left_edge_directional_interpolates_without_clipping() {
        let left = [100_i16; 33];
        let top = [0_i16; 33];
        let mut out = [0_i16; 64];
        predict_signed_left_edge_directional(&left, &top, 8, &mut out).unwrap();
        assert_eq!(out[0], 59);
        assert_eq!(out[7 * 8], 100);
    }

    #[test]
    fn angular_inverse_reference_extends_past_history_without_panicking() {
        let left = [100_i16; 33];
        let top = [0_i16; 33];
        let mut out = [0_i16; 256];
        predict_signed_angular(&left, &top, 16, 4, &mut out).unwrap();
        assert_eq!(out[15 * 16], 100);
    }

    #[test]
    fn signed_constructor_drops_corner_history_entry() {
        let left = core::array::from_fn(|i| i16::try_from(i).expect("test index fits i16"));
        let top = core::array::from_fn(|i| i16::try_from(100 + i).expect("test index fits i16"));
        let refs = build_signed_references(&left, &top);
        assert_eq!(refs.left[0], 1);
        assert_eq!(refs.top[0], 101);
        assert_eq!(refs.left[31], 32);
    }
}
