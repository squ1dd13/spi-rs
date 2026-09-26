//! Alpha block stream decoding and picture-wide reference state.

use crate::{
    bits::ContextBitReader,
    decoder::SliceGeometry,
    entropy::{CoefficientHistory, ComponentRequest, decode_contextual_component},
    error::{CoefficientError, DecodeError, ErrorContext, SyntaxError},
    format::Record,
    prediction::{
        PredictionReferences, SignedReferences, filter_signed_references, predict_alpha,
        predict_signed_angular, predict_signed_dc, predict_signed_planar,
        prediction_direction_for_mode,
    },
    surface::{Motion, copy_alpha_motion, copy_block16},
};

/// Picture-wide alpha history. The planes include 32 pixels of horizontal padding.
pub(crate) struct AlphaState {
    block_width: usize,
    total_block_rows: usize,
    stride: usize,
    left_bank: [u8; 33],
    top_bank: [u8; 33],
    next_row_top_banks: Vec<[u8; 33]>,
    current_row_bottom: Vec<u8>,
    block_available: Vec<u8>,
    saved_block_edge: Vec<u8>,
    current_picture_plane: Vec<u8>,
}

impl AlphaState {
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

    pub(crate) fn heap_bytes(&self) -> usize {
        self.next_row_top_banks.capacity() * std::mem::size_of::<[u8; 33]>()
            + self.current_row_bottom.capacity()
            + self.block_available.capacity()
            + self.saved_block_edge.capacity()
            + self.current_picture_plane.capacity()
    }
}

#[derive(Debug)]
pub(crate) struct AlphaSlice {
    pub(crate) plane: Vec<u8>,
    pub(crate) stride: usize,
    pub(crate) end_bit: usize,
}

impl AlphaSlice {
    pub(crate) fn heap_bytes(&self) -> usize {
        self.plane.capacity()
    }
}

fn allocation(resource: &'static str) -> DecodeError {
    DecodeError::Allocation { resource }
}
fn invalid(detail: &'static str) -> DecodeError {
    DecodeError::Syntax(SyntaxError::Invalid {
        stream: "alpha",
        detail,
    })
}

fn at(error: DecodeError, record: &Record<'_>, block: usize, bit: usize) -> DecodeError {
    error.at(ErrorContext {
        record: Some(usize::from(record.index)),
        block: Some(block),
        component: None,
        bit_position: Some(bit),
    })
}

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
        .read_value(2)
        .ok_or_else(|| invalid("truncated block method"))? as u8)
}

fn read_motion(reader: &mut ContextBitReader<'_>) -> Result<Motion, DecodeError> {
    if reader
        .read_value(1)
        .ok_or_else(|| invalid("truncated motion vector"))?
        != 0
    {
        return Ok(Motion { x: 0, y: 16 });
    }
    let mut values = [0i32; 2];
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
        let code = (1u32 << zeros) + suffix;
        let decoded = if code & 1 != 0 {
            -((code / 2) as i32)
        } else {
            (code / 2) as i32
        };
        *value = decoded.saturating_mul(16);
    }
    Ok(Motion {
        x: values[0],
        y: values[1],
    })
}

fn transition(code: u16, top: &mut [u8; 33], left: &mut [u8; 33]) {
    let t = *top;
    let l = *left;
    let avg = |a: u8, b: u8| ((u16::from(a) + u16::from(b) + 1) >> 1) as u8;
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

fn construct_refs(
    width: usize,
    selector: usize,
    left_bank: &[u8; 33],
    top_bank: &[u8; 33],
    plane: &[u8],
    stride: usize,
    x: usize,
    y: usize,
) -> Result<([u8; 33], [u8; 33]), DecodeError> {
    let mut left = [0u8; 33];
    let mut top = [0u8; 33];
    let origin = y
        .checked_mul(stride)
        .and_then(|v| v.checked_add(x))
        .ok_or_else(|| invalid("reference offset overflow"))?;
    let vertical = |out: &mut [u8; 33], count: usize, extend: bool| -> Result<(), DecodeError> {
        if x == 0 {
            return Err(invalid("missing left reference"));
        }
        for i in 0..count {
            out[i + 1] = *plane
                .get(origin - 1 + i * stride)
                .ok_or_else(|| invalid("left reference outside slice"))?;
        }
        if extend {
            let edge = out[count];
            out[count + 1..2 * width + 1].fill(edge);
        }
        Ok(())
    };
    let above = |out: &mut [u8; 33], count: usize, extend: bool| -> Result<(), DecodeError> {
        if y == 0 || x + count > stride {
            return Err(invalid("missing top reference"));
        }
        out[1..count + 1].copy_from_slice(&plane[origin - stride..origin - stride + count]);
        if extend {
            let edge = out[count];
            out[count + 1..2 * width + 1].fill(edge);
        }
        Ok(())
    };
    match width {
        4 => match selector {
            0 => {
                left = *left_bank;
                top = *top_bank;
            }
            1 => {
                top[..9].copy_from_slice(&top_bank[4..13]);
                left[0] = top[0];
                vertical(&mut left, 4, true)?;
            }
            2 => {
                left[..9].copy_from_slice(&left_bank[4..13]);
                top[0] = left[0];
                above(&mut top, 8, false)?;
            }
            3 => {
                left[0] = plane[origin - stride - 1];
                top[0] = left[0];
                vertical(&mut left, 4, true)?;
                above(&mut top, 4, false)?;
            }
            4 => {
                top[..9].copy_from_slice(&top_bank[8..17]);
                left[0] = top[0];
                vertical(&mut left, 8, false)?;
            }
            5 => {
                top[..9].copy_from_slice(&top_bank[12..21]);
                left[0] = top[0];
                vertical(&mut left, 4, true)?;
            }
            6 | 9 | 14 => {
                left[0] = plane[origin - stride - 1];
                top[0] = left[0];
                vertical(&mut left, 4, true)?;
                above(&mut top, 8, false)?;
            }
            7 | 11 | 13 | 15 => {
                left[0] = plane[origin - stride - 1];
                top[0] = left[0];
                vertical(&mut left, 4, true)?;
                above(&mut top, 4, true)?;
            }
            8 => {
                left[..9].copy_from_slice(&left_bank[8..17]);
                top[0] = left[0];
                above(&mut top, 8, false)?;
            }
            10 => {
                left[..9].copy_from_slice(&left_bank[12..21]);
                top[0] = left[0];
                above(&mut top, 8, false)?;
            }
            12 => {
                left[0] = plane[origin - stride - 1];
                top[0] = left[0];
                vertical(&mut left, 8, false)?;
                above(&mut top, 8, false)?;
            }
            _ => return Err(invalid("invalid width-four reference selector")),
        },
        8 => match selector {
            0 => {
                left = *left_bank;
                top = *top_bank;
            }
            1 => {
                top[..17].copy_from_slice(&top_bank[8..25]);
                left[0] = top[0];
                vertical(&mut left, 8, true)?;
            }
            2 => {
                left[..17].copy_from_slice(&left_bank[8..25]);
                top[0] = left[0];
                above(&mut top, 16, false)?;
            }
            3 => {
                left[0] = plane[origin - stride - 1];
                top[0] = left[0];
                vertical(&mut left, 8, true)?;
                above(&mut top, 8, true)?;
            }
            _ => return Err(invalid("invalid width-eight reference selector")),
        },
        16 => {
            left = *left_bank;
            top = *top_bank;
        }
        _ => return Err(invalid("invalid component width")),
    }
    Ok((left, top))
}

fn predict(
    left_u8: &[u8; 33],
    top_u8: &[u8; 33],
    width: usize,
    mode: u8,
    out: &mut [u8; 256],
) -> Result<(), DecodeError> {
    if width == 4 && ((mode <= 8 && mode != 3) || mode == 17) {
        let mut references = PredictionReferences::default();
        references.top_left = if mode == 4 { top_u8[0] } else { left_u8[0] };
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
    let mut left = [0i16; 33];
    let mut top = [0i16; 33];
    for i in 0..=2 * width {
        left[i] = i16::from(left_u8[i]);
        top[i] = i16::from(top_u8[i]);
    }
    let mut fl = [0i16; 33];
    let mut ft = [0i16; 33];
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
    let mut signed = [0i16; 256];
    match mode {
        0 | 1 => {
            for r in 0..width {
                for c in 0..width {
                    signed[r * width + c] = if mode == 0 { t[c + 1] } else { l[r + 1] };
                }
            }
        }
        2 => predict_signed_dc(&left, &top, width, &mut signed[..width * width])
            .map_err(|_| invalid("DC prediction failed"))?,
        3 => {
            for r in 0..width {
                for c in 0..width {
                    signed[r * width + c] = if c >= r { t[c - r] } else { l[r - c] };
                }
            }
        }
        4 => {
            let mut main = [0i16; 65];
            for i in 0..33 {
                main[32 + i] = t[i];
            }
            let mut sum = 128;
            let last = if width == 4 {
                -2
            } else if width == 8 {
                -3
            } else {
                -6
            };
            for i in (last as isize..=-1isize).rev() {
                sum += 630;
                main[(32isize + i) as usize] = l[(sum >> 8) as usize];
            }
            for r in 0..width {
                let d = -((r as i32 + 1) * 13);
                let integer = d >> 5;
                let frac = d & 31;
                for c in 0..width {
                    let i = (32 + c as i32 + integer + 1) as usize;
                    signed[r * width + c] =
                        (((32 - frac) * i32::from(main[i]) + frac * i32::from(main[i + 1]) + 16)
                            >> 5) as i16;
                }
            }
        }
        5 => {
            for r in 0..width {
                let d = (r + 1) * 13;
                let i = d >> 5;
                let f = (d & 31) as i32;
                for c in 0..width {
                    let n = c + i + 1;
                    signed[r * width + c] =
                        (((32 - f) * i32::from(t[n]) + f * i32::from(t[n + 1]) + 16) >> 5) as i16;
                }
            }
        }
        6 => {
            for r in 0..width {
                for c in 0..width {
                    signed[r * width + c] = if width == 4 {
                        let i = r + c;
                        if i >= 6 {
                            t[8]
                        } else {
                            (t[i + 1] + 2 * t[i + 2] + t[i + 3] + 2) >> 2
                        }
                    } else {
                        t[r + c + 2]
                    };
                }
            }
        }
        7 => {
            if width == 4 {
                let mut main = [0i16; 34];
                for i in 0..17 {
                    main[16 + i] = l[i];
                }
                for i in (-16isize..=-1isize).rev() {
                    let side = ((-i * 630 + 128) >> 8) as usize;
                    main[(16isize + i) as usize] = t[side.min(16) + 1];
                }
                for r in 0..4 {
                    for c in 0..4 {
                        let d = -((c as i32 + 1) * 13);
                        let integer = -((-d + 31) / 32);
                        let frac = d - integer * 32;
                        let idx = (16 + r as i32 + integer + 1) as usize;
                        signed[r * 4 + c] = (((32 - frac) * i32::from(main[idx])
                            + frac * i32::from(main[idx + 1])
                            + 16)
                            >> 5) as i16;
                    }
                }
            } else {
                predict_signed_angular(l, t, width, 11, &mut signed[..width * width])
                    .map_err(|_| invalid("left angular prediction failed"))?;
            }
        }
        8 => {
            for r in 0..width {
                for c in 0..width {
                    let d = (c + 1) * 13;
                    let i = r + (d >> 5) + 1;
                    let f = (d & 31) as i32;
                    signed[r * width + c] =
                        (((32 - f) * i32::from(l[i]) + f * i32::from(l[i + 1]) + 16) >> 5) as i16;
                }
            }
        }
        9..=16 => {
            let direction = prediction_direction_for_mode(mode)
                .ok_or_else(|| invalid("invalid angular mode"))?;
            predict_signed_angular(l, t, width, direction, &mut signed[..width * width])
                .map_err(|_| invalid("angular prediction failed"))?;
        }
        17 => {
            let mut refs = SignedReferences::default();
            for i in 0..32 {
                refs.left[i] = left[i + 1];
                refs.top[i] = top[i + 1];
            }
            predict_signed_planar(&refs, width, &mut signed[..width * width])
                .map_err(|_| invalid("planar prediction failed"))?;
        }
        _ => return Err(invalid("unsupported prediction mode")),
    }
    for i in 0..width * width {
        out[i] = signed[i].clamp(0, 255) as u8;
    }
    Ok(())
}

fn apply_component(
    width: usize,
    mode: u8,
    _full: bool,
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
        output[i] = (i32::from(prediction[i]) + i32::from(values[i])).clamp(0, 255) as u8;
    }
}

pub(crate) fn decode_slice(
    record: &Record<'_>,
    geometry: SliceGeometry,
    color_end_bit: usize,
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
    let start = (color_end_bit + 7) & !7;
    let mut reader = ContextBitReader::new(record.data);
    if !reader.initialize_at_bit(start) {
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
    let mut history = CoefficientHistory::new(history_stride, rows)
        .map_err(|_| invalid("coefficient history allocation failed"))?;
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
            .begin_alpha_block_row(block_y as u16)
            .map_err(|_| invalid("coefficient row history failed"))?;
        let absolute_row = geometry.absolute_block_row + block_y;
        previous.copy_from_slice(&state.block_available);
        current.fill(0);
        for block_x in 0..geometry.block_width {
            let block = block_y * geometry.block_width + block_x;
            let bit = reader.logical_bit();
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
                let code = ((block_x > 0 && previous[block_x - 1] != 0) as u16) << 12
                    | ((block_x + 1 < geometry.block_width && previous[block_x + 1] != 0) as u16)
                        << 8
                    | (top_available as u16) << 4
                    | left_available as u16;
                transition(code, &mut state.top_bank, &mut state.left_bank);
            } else if block_x == 0 {
                state.left_bank.fill(128);
            }
            let method = read_method(&mut reader).map_err(|e| at(e, record, block, bit))?;
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
            match method {
                0 | 1 => {
                    let motion = if method == 0 {
                        if block_x == 0 {
                            Motion { x: 0, y: 16 }
                        } else {
                            Motion { x: 16, y: 0 }
                        }
                    } else {
                        read_motion(&mut reader).map_err(|e| at(e, record, block, bit))?
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
                        dst_x,
                        global_y,
                        &mut plane,
                        stride,
                        dst_x,
                        local_y,
                    )
                    .map_err(|_| at(invalid("alpha block copy failed"), record, block, bit))?;
                }
                3 => {
                    if reader.read_value(1) != Some(1) {
                        return Err(at(
                            invalid("predictive transform callback is not initialized"),
                            record,
                            block,
                            bit,
                        ));
                    }
                    for argument in 0..4u32 {
                        let component = decode_contextual_component(
                            &mut reader,
                            &mut history,
                            ComponentRequest {
                                argument,
                                horizontal_offset: (block_x * 16) as u16,
                                maximum_width: 8,
                                fixed_width: 0,
                                color_long_symbols: false,
                                width_selector_already_read: false,
                            },
                        )
                        .map_err(|error| {
                            DecodeError::Coefficients(CoefficientError::Syntax(Box::new(error))).at(
                                ErrorContext {
                                    record: Some(usize::from(record.index)),
                                    block: Some(block),
                                    component: Some(argument as usize),
                                    bit_position: Some(reader.logical_bit()),
                                },
                            )
                        })?;
                        let width = usize::from(component.width);
                        for slot in 0..component.count {
                            let base_x = dst_x
                                + (argument as usize & 1) * 8
                                + if width == 4 { (slot & 1) * 4 } else { 0 };
                            let base_y = local_y
                                + (argument as usize >> 1) * 8
                                + if width == 4 { (slot >> 1) * 4 } else { 0 };
                            let selector = if width == 4 {
                                argument as usize * 4 + slot
                            } else {
                                argument as usize
                            };
                            let (left, top) = construct_refs(
                                width,
                                selector,
                                &state.left_bank,
                                &state.top_bank,
                                &plane,
                                stride,
                                base_x,
                                base_y,
                            )
                            .map_err(|e| at(e, record, block, reader.logical_bit()))?;
                            let mut pred = [0u8; 256];
                            predict(&left, &top, width, component.modes[slot], &mut pred)
                                .map_err(|e| at(e, record, block, reader.logical_bit()))?;
                            let mut reconstructed = [0u8; 256];
                            apply_component(
                                width,
                                component.modes[slot],
                                component.callback_full_transform[slot],
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
                }
                5 => {
                    let raw_bit = (reader.logical_bit() + 7) & !7;
                    let offset = raw_bit / 8;
                    let end = offset
                        .checked_add(256)
                        .filter(|&v| v <= record.data.len())
                        .ok_or_else(|| {
                            at(invalid("truncated raw alpha block"), record, block, raw_bit)
                        })?;
                    for row in 0..16 {
                        let out = (local_y + row) * stride + dst_x;
                        plane[out..out + 16].copy_from_slice(
                            &record.data[offset + row * 16..offset + row * 16 + 16],
                        );
                    }
                    reader.byte_offset = end;
                    reader.accumulator = 0;
                    reader.available = 0;
                }
                _ => {
                    return Err(at(
                        invalid("unknown alpha coding method"),
                        record,
                        block,
                        bit,
                    ));
                }
            }
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
        state.block_available.copy_from_slice(&current);
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
                bank[i + 1] = if px < state.current_row_bottom.len() && px < geometry.picture_width
                {
                    state.current_row_bottom[px]
                } else {
                    state.current_row_bottom[geometry.picture_width.saturating_sub(1)]
                };
            }
        }
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
                terminal: 0,
                end: false,
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
        let mut bytes = vec![0u8; 14 + 1 + 256];
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
            let mut bytes = vec![0u8; 18];
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

        let mut second = vec![0u8; 18];
        second[14] = 0x60; // motion method, default vector (0, 16)
        let decoded = decode_slice(&record(&second), geometry(1, 1), 112, &mut state).unwrap();
        assert_eq!(&decoded.plane[..16], &[77; 16]);
    }

    #[test]
    fn motion_code_parity_selects_sign() {
        let mut bits = ContextBitReader::new(&[0x26, 0, 0, 0]);
        assert!(bits.initialize_at_bit(0));
        assert_eq!(read_motion(&mut bits).unwrap(), Motion { x: 16, y: -16 });
    }
}
