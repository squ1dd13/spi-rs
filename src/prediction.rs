//! Reference construction and integer intra prediction.

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PredictionError {
    #[error("invalid prediction block width")]
    InvalidWidth,
    #[error("unsupported prediction mode")]
    InvalidMode,
    #[error("prediction reference is out of bounds")]
    OutOfBounds,
    #[error("prediction output buffer is too small")]
    BufferTooSmall,
}

fn valid_width(w: usize) -> bool {
    w == 4 || w == 8 || w == 16
}

#[derive(Clone, Copy, Debug)]
pub struct PredictionReferences {
    pub top_left: u8,
    pub top: [u8; 33],
    pub left: [u8; 33],
    pub first_row_values: [u8; 16],
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

/// Build the top and left samples available at a block position.
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
    let mut r = PredictionReferences::default();
    r.first_row = y == 0;
    r.top_left = if x == 0 || y == 0 {
        128
    } else {
        plane[(y - 1) * stride + x - 1]
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
    let top_limit = if y % 16 == 0 {
        stride - 1
    } else if width == 4 && y % 8 == 4 {
        ((x / 8) * 8 + 7).min(stride - 1)
    } else {
        block_x.saturating_add(15).min(stride - 1)
    };
    let left_limit = if width == 4 && x % 8 == 0 {
        ((y / 8) * 8 + 7).min(rows - 1)
    } else if x % 16 == 0 {
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

pub fn filter3(a: u8, b: u8, c: u8) -> u8 {
    ((u16::from(a) + 2 * u16::from(b) + u16::from(c) + 2) >> 2) as u8
}

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

/// Predict alpha modes supported by the observed callback: top, left, DC,
/// diagonal, angular families 4 through 8, and planar (17).
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
                2 => {
                    let mut sum = width as u32;
                    for i in 0..width {
                        sum += u32::from(references.top[i]) + u32::from(references.left[i]);
                    }
                    let dc = sum / (2 * width) as u32;
                    if width != 4 {
                        dc as u8
                    } else if row == 0 && col == 0 {
                        ((3 * (u32::from(references.left[0]) + u32::from(references.top[0]))
                            + 2 * dc
                            + 4)
                            >> 3) as u8
                    } else if row == 0 {
                        ((3 * u32::from(references.top[col]) + 5 * dc + 4) >> 3) as u8
                    } else if col == 0 {
                        ((3 * u32::from(references.left[row]) + 5 * dc + 4) >> 3) as u8
                    } else {
                        dc as u8
                    }
                }
                3 if references.first_row && first_row_top_unavailable => 0,
                3 => {
                    if col > row {
                        let i = col - row - 1;
                        filter3(
                            if i == 0 {
                                references.top_left
                            } else {
                                references.top[i - 1]
                            },
                            references.top[i],
                            references.top[i + 1],
                        )
                    } else if row > col {
                        let i = row - col - 1;
                        filter3(
                            if i == 0 {
                                references.top_left
                            } else {
                                references.left[i - 1]
                            },
                            references.left[i],
                            references.left[i + 1],
                        )
                    } else {
                        filter3(references.left[0], references.top_left, references.top[0])
                    }
                }
                4 => {
                    let mut main = [0_u8; 35];
                    let origin = 17;
                    main[origin] = references.top_left;
                    for i in 0..17 {
                        main[origin + i + 1] = references.top[i];
                    }
                    let mut sum = 128_i32;
                    for n in 1..=width {
                        sum += 630;
                        let idx = (sum >> 8) as usize;
                        main[origin - n] = if idx == 0 {
                            references.top_left
                        } else {
                            references.left[idx - 1]
                        };
                    }
                    let d = -((row as i32 + 1) * 13);
                    let integer = d >> 5;
                    let frac = d & 31;
                    let i = (col as i32) + integer + 1;
                    (((32 - frac) * i32::from(main[(origin as i32 + i) as usize])
                        + frac * i32::from(main[(origin as i32 + i + 1) as usize])
                        + 16)
                        >> 5) as u8
                }
                5 => {
                    let d = (row + 1) * 13;
                    let i = col + (d >> 5);
                    let f = d & 31;
                    ((((32 - f) as u32) * u32::from(edge_filter(references, true, i, width))
                        + (f as u32) * u32::from(edge_filter(references, true, i + 1, width))
                        + 16)
                        >> 5) as u8
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
                7 => {
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
                    let d = -((col as i32 + 1) * 13);
                    let integer = if d < 0 { -((-d + 31) / 32) } else { d / 32 };
                    let frac = d - integer * 32;
                    let i = row as i32 + integer + 1;
                    (((32 - frac) * i32::from(main[(origin as i32 + i) as usize])
                        + frac * i32::from(main[(origin as i32 + i + 1) as usize])
                        + 16)
                        >> 5) as u8
                }
                8 => {
                    let d = (col + 1) * 13;
                    let i = row + (d >> 5);
                    let f = d & 31;
                    ((((32 - f) as u32) * u32::from(edge_filter(references, false, i, width))
                        + (f as u32) * u32::from(edge_filter(references, false, i + 1, width))
                        + 16)
                        >> 5) as u8
                }
                17 => {
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
                    let h = (e - col) as u32 * u32::from(references.left[row])
                        + (col + 1) as u32 * u32::from(te);
                    let v = (e - row) as u32 * u32::from(tv)
                        + (row + 1) as u32 * u32::from(references.left[e]);
                    ((h + v + width as u32) / (2 * width) as u32) as u8
                }
                _ => return Err(PredictionError::InvalidMode),
            };
            output[row * width + col] = v;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct SignedReferences {
    pub top: [i16; 33],
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

/// Copy the persistent histories into the constructor layout used by planar prediction.
pub fn build_signed_references(
    left_history: &[i16; 33],
    top_history: &[i16; 33],
) -> SignedReferences {
    let mut refs = SignedReferences::default();
    refs.left[..32].copy_from_slice(&left_history[1..]);
    refs.top[..32].copy_from_slice(&top_history[1..]);
    refs
}

/// Signed angular Reference A. Width four uses the observed shortened inverse
/// extension; larger widths use their corresponding 630-angle extension.
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
        main[(ORIGIN + i as i32) as usize] = filtered_top[i];
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
        main[(ORIGIN + index) as usize] = filtered_left[(inverse_sum >> 8) as usize];
    }
    for row in 0..width {
        let displacement = -((row as i32 + 1) * 13);
        let integer = displacement >> 5;
        let fraction = displacement & 31;
        for col in 0..width {
            let first = ORIGIN + col as i32 + integer + 1;
            let value = (32 - fraction) * i32::from(main[first as usize])
                + fraction * i32::from(main[first as usize + 1])
                + 16;
            output[row * width + col] = (value >> 5) as i16;
        }
    }
    Ok(())
}

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
    main[ORIGIN as usize] = filtered_top[0];
    for i in 0..32 {
        main[ORIGIN as usize + i + 1] = filtered_left[i + 1];
    }
    for index in -(width as i32)..0 {
        let side_index = (((-index * 630 + 128) >> 8) as usize)
            .saturating_sub(1)
            .min(31);
        main[(ORIGIN + index) as usize] = filtered_top[side_index + 1];
    }
    for row in 0..width {
        for col in 0..width {
            let displacement = -((col as i32 + 1) * 13);
            let integer = -((-displacement + 31) / 32);
            let fraction = displacement - integer * 32;
            let first = ORIGIN + row as i32 + integer + 1;
            let value = (32 - fraction) * i32::from(main[first as usize])
                + fraction * i32::from(main[first as usize + 1])
                + 16;
            output[row * width + col] = (value >> 5) as i16;
        }
    }
    Ok(())
}

/// Recovered width-four angular callback for modes 5, 6, and 7.
pub fn predict_signed_width4_angular(
    top: &[i16; 33],
    mode: u8,
    left: Option<&[i16; 33]>,
    output: &mut [i16],
) -> Result<(), PredictionError> {
    if output.len() < 16 {
        return Err(PredictionError::BufferTooSmall);
    }
    if !matches!(mode, 5 | 6 | 7) || (mode == 7 && left.is_none()) {
        return Err(PredictionError::InvalidMode);
    }
    let left = left;
    let mut main = [0_i16; 34];
    if mode == 7 {
        let l = left.expect("validated above");
        const ORIGIN: i32 = 16;
        main[ORIGIN as usize] = l[0];
        for i in 0..17 {
            main[ORIGIN as usize + i + 1] = l[i + 1];
        }
        for index in (-16_i32..=-1).rev() {
            let side_index = (((-index * 630 + 128) >> 8) as usize)
                .saturating_sub(1)
                .min(16);
            main[(ORIGIN + index) as usize] = top[side_index + 1];
        }
    }
    for row in 0..4 {
        for col in 0..4 {
            let value = if mode == 5 {
                let displacement = (row + 1) * 13;
                let reference = col + (displacement >> 5);
                let fraction = displacement & 31;
                ((32 - fraction) as i32 * i32::from(top[reference + 1])
                    + fraction as i32 * i32::from(top[reference + 2])
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
                let displacement = -((col as i32 + 1) * 13);
                let integer = -((-displacement + 31) / 32);
                let fraction = displacement - integer * 32;
                let first = ORIGIN + row as i32 + integer + 1;
                ((32 - fraction) * i32::from(main[first as usize])
                    + fraction * i32::from(main[first as usize + 1])
                    + 16)
                    >> 5
            };
            output[row * 4 + col] = value as i16;
        }
    }
    Ok(())
}

/// The width-four diagonal callback uses the left corner on the diagonal.
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
            output[row * 4 + col] = if col > row {
                filtered_top[col - row]
            } else if row > col {
                filtered_left[row - col]
            } else {
                filtered_left[0]
            };
        }
    }
    Ok(())
}

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
    out_left[0] = corner as i16;
    out_top[0] = corner as i16;
    for i in 1..end {
        out_left[i] =
            ((i32::from(left[i - 1]) + 2 * i32::from(left[i]) + i32::from(left[i + 1]) + 2) >> 2)
                as i16;
        out_top[i] = ((i32::from(top[i - 1]) + 2 * i32::from(top[i]) + i32::from(top[i + 1]) + 2)
            >> 2) as i16;
    }
    out_left[end] = left[end];
    out_top[end] = top[end];
    Ok(())
}

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
    let mut total = width as i32;
    for i in 1..=width {
        total += i32::from(left[i]) + i32::from(top[i]);
    }
    let dc = total / (2 * width) as i32;
    output[..width * width].fill(dc as i16);
    output[0] = if width == 4 {
        ((3 * (i32::from(left[1]) + i32::from(top[1])) + 2 * dc + 4) >> 3) as i16
    } else if width == 8 {
        ((i32::from(left[1]) + i32::from(top[1]) + 2 * dc + 2) >> 2) as i16
    } else {
        ((i32::from(left[1]) + i32::from(top[1]) + 6 * dc + 4) >> 3) as i16
    };
    let (ew, er, shift, sw) = if width == 4 {
        (5, 4, 3, 3)
    } else if width == 8 {
        (3, 2, 2, 1)
    } else {
        (7, 4, 3, 1)
    };
    for i in 1..width {
        output[i] = ((sw * i32::from(top[i + 1]) + ew * dc + er) >> shift) as i16;
        output[i * width] = ((sw * i32::from(left[i + 1]) + ew * dc + er) >> shift) as i16;
    }
    Ok(())
}

pub fn prediction_direction_for_mode(mode: u8) -> Option<u8> {
    const D: [u8; 17] = [5, 13, 0, 1, 3, 7, 9, 11, 15, 17, 2, 4, 6, 8, 10, 12, 14];
    D.get(mode as usize).copied()
}

pub fn predict_signed_top_or_left(
    left: &[i16; 33],
    top: &[i16; 33],
    top_mode: bool,
    output: &mut [i16],
) -> Result<(), PredictionError> {
    if output.len() < 256 {
        return Err(PredictionError::BufferTooSmall);
    }
    for row in 0..16 {
        for col in 0..16 {
            output[row * 16 + col] = if top_mode {
                top[col + 1]
            } else {
                left[row + 1]
            };
        }
    }
    Ok(())
}

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
            let value = (32 - fraction) as i32 * i32::from(top[i])
                + fraction as i32 * i32::from(top[i + 1])
                + 16;
            output[row * width + col] = (value >> 5) as i16;
        }
    }
    Ok(())
}

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
            let value = (32 - fraction) as i32 * i32::from(filtered_left[index])
                + fraction as i32 * i32::from(filtered_left[index + 1])
                + 16;
            output[row * width + col] = (value >> 5) as i16;
        }
    }
    Ok(())
}

/// Predict one signed angular mode from the recovered direction table.
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
    for i in 0..33 {
        refs[32 + i] = main[i];
    }
    if angle < 0 {
        let mut sum = 128_i32;
        for i in 1..=width {
            sum += INV[ai];
            // Large inverse angles can project past the persistent 33-sample
            // side history even when the predictor never consumes that far.
            // Extend by repeating the final available sample.
            let side_index = ((sum >> 8) as usize).min(side.len() - 1);
            refs[(origin - i as i32) as usize] = side[side_index];
        }
    }
    for r in 0..width {
        for c in 0..width {
            let axis = if horizontal { c } else { r };
            let cross = if horizontal { r } else { c };
            let d = (axis as i32 + 1) * angle;
            let integer = d >> 5;
            let frac = d - integer * 32;
            let i = origin + cross as i32 + integer + 1;
            let val = (32 - frac) * i32::from(refs[i as usize])
                + frac * i32::from(refs[i as usize + 1])
                + 16;
            output[r * width + c] = (val >> 5) as i16;
        }
    }
    Ok(())
}

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
            let h =
                (e - c) as i32 * i32::from(refs.left[r]) + (c + 1) as i32 * i32::from(refs.top[e]);
            let v =
                (e - r) as i32 * i32::from(refs.top[c]) + (r + 1) as i32 * i32::from(refs.left[e]);
            output[r * width + c] = ((h + v + width as i32) / (2 * width) as i32) as i16;
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
        let top = core::array::from_fn(|i| (i * 10) as i16);
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
    }

    #[test]
    fn signed_reference_a_and_top_edge_keep_halfword_values() {
        let left = [100_i16; 33];
        let top = [0_i16; 33];
        let mut out = [0_i16; 64];
        predict_signed_wide_angular_reference_a(&left, &top, 8, &mut out).unwrap();
        assert_eq!(out[0], 0);
        assert_eq!(out[7 * 8], 100);
        let ramp = core::array::from_fn(|i| (i as i16) * 10);
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
        let left = core::array::from_fn(|i| i as i16);
        let top = core::array::from_fn(|i| (100 + i) as i16);
        let refs = build_signed_references(&left, &top);
        assert_eq!(refs.left[0], 1);
        assert_eq!(refs.top[0], 101);
        assert_eq!(refs.left[31], 32);
    }
}
