use super::{predictive, references};
use crate::{
    bits::{BitReader, ContextBitReader},
    color::{ColorSlice, ColorState, allocation, invalid},
    decoder::SliceGeometry,
    entropy::{CoefficientHistory, decode_contextual_indices, decode_rle_plane},
    error::{CoefficientError, DecodeError},
    format::Record,
    surface::{Motion, copy_color_motion},
};

struct Palette {
    entries: usize,
    prefix: u32,
    bytes: Vec<u8>,
}

pub(super) fn decode_slice(
    record: &Record<'_>,
    geometry: SliceGeometry,
    state: &mut ColorState,
) -> Result<ColorSlice, DecodeError> {
    validate_geometry(geometry, state)?;
    if record.payload_offset.checked_mul(8).is_none()
        || record.data.len() < record.payload_offset
        || record.payload.is_empty()
    {
        return Err(invalid("record has no color payload"));
    }
    let stride = geometry
        .block_width
        .checked_mul(16)
        .ok_or_else(|| allocation("color slice"))?;
    let plane_len = stride
        .checked_mul(
            geometry
                .block_rows
                .checked_mul(16)
                .ok_or_else(|| allocation("color slice"))?,
        )
        .ok_or_else(|| allocation("color slice"))?;
    let mut planes = [Vec::new(), Vec::new(), Vec::new()];
    for plane in &mut planes {
        plane
            .try_reserve_exact(plane_len)
            .map_err(|_| allocation("color slice"))?;
        plane.resize(plane_len, 0);
    }
    let mut bits = BitReader::new(record.payload);
    let mut palette = Palette {
        entries: 0,
        prefix: 0,
        bytes: Vec::new(),
    };
    let mut coefficient_histories: Option<[CoefficientHistory; 3]> = None;
    for local_y in 0..geometry.block_rows {
        let global_block_y = state.global_row(local_y, geometry.absolute_block_row)?;
        if local_y != 0
            && let Some(histories) = coefficient_histories.as_mut()
        {
            for (plane, history) in histories.iter_mut().enumerate() {
                history
                    .begin_block_row(local_y as u16, plane != 0)
                    .map_err(|error| {
                        DecodeError::Coefficients(crate::error::CoefficientError::Syntax(Box::new(
                            error,
                        )))
                    })?;
            }
        }
        for block_x in 0..geometry.block_width {
            let block_index = local_y * geometry.block_width + block_x;
            let method_start = bits.bit_position();
            let method = read_method(&mut bits).map_err(|error| {
                at_block(
                    error,
                    record,
                    block_index,
                    record.payload_offset * 8 + bits.bit_position(),
                )
            })?;
            let available_index = global_block_y * state.block_width + block_x;
            state.available[available_index] = 1;
            let result = (|| -> Result<(), DecodeError> {
                match method {
                    0 => spatial_copy(
                        state,
                        &mut planes,
                        geometry,
                        block_x,
                        local_y,
                        global_block_y,
                    )?,
                    1 => {
                        let (dx, dy) = read_motion(&mut bits)?;
                        motion_copy(
                            state,
                            &mut planes,
                            geometry,
                            block_x,
                            local_y,
                            global_block_y,
                            dx,
                            dy,
                        )?;
                    }
                    2 => residual(
                        &mut bits,
                        state,
                        &mut planes,
                        geometry,
                        block_x,
                        local_y,
                        global_block_y,
                    )?,
                    3 => {
                        predictive::decode_block(
                            &mut bits,
                            record,
                            state,
                            &mut planes,
                            geometry,
                            block_x,
                            local_y,
                            global_block_y,
                            block_index,
                            method_start,
                            &mut coefficient_histories,
                        )?;
                    }
                    4 => palette_block(
                        &mut bits,
                        record.payload,
                        record.payload_offset,
                        &mut palette,
                        state,
                        &mut planes,
                        geometry,
                        block_x,
                        local_y,
                        global_block_y,
                    )?,
                    5 => raw(
                        &mut bits,
                        state,
                        &mut planes,
                        geometry,
                        block_x,
                        local_y,
                        global_block_y,
                    )?,
                    _ => unreachable!(),
                }
                Ok(())
            })();
            result.map_err(|error| {
                at_block(
                    error,
                    record,
                    block_index,
                    record.payload_offset * 8 + bits.bit_position(),
                )
            })?;
        }
    }
    Ok(ColorSlice {
        planes,
        stride,
        end_bit: record
            .payload_offset
            .checked_mul(8)
            .and_then(|base| base.checked_add(bits.bit_position()))
            .ok_or_else(|| invalid("color bit position overflow"))?,
    })
}

fn at_block(error: DecodeError, record: &Record<'_>, block: usize, bit: usize) -> DecodeError {
    error.at(crate::ErrorContext {
        record: Some(record.index as usize),
        block: Some(block),
        component: None,
        bit_position: Some(bit),
    })
}

fn validate_geometry(g: SliceGeometry, s: &ColorState) -> Result<(), DecodeError> {
    if g.block_width != s.block_width
        || g.block_rows == 0
        || g.absolute_block_row
            .checked_add(g.block_rows)
            .is_none_or(|end| end > s.total_block_rows)
    {
        return Err(invalid("slice geometry does not match color history"));
    }
    let padded_width = g
        .block_width
        .checked_mul(16)
        .ok_or_else(|| invalid("slice width overflow"))?;
    if padded_width != s.stride {
        return Err(invalid("slice stride does not match picture"));
    }
    Ok(())
}

fn read_method(bits: &mut BitReader<'_>) -> Result<u8, DecodeError> {
    let first = bits
        .read_bit()
        .map_err(|_| invalid("truncated block method"))?;
    if first != 0 {
        return Ok(0);
    }
    let second = bits
        .read_bit()
        .map_err(|_| invalid("truncated block method"))?;
    if second != 0 {
        return Ok(1);
    }
    Ok(bits
        .read_bits(2)
        .map_err(|_| invalid("truncated block method"))? as u8
        + 2)
}

fn read_unary(bits: &mut BitReader<'_>, limit: u32) -> Result<u32, DecodeError> {
    for n in 0..limit {
        if bits
            .read_bit()
            .map_err(|_| invalid("truncated unary code"))?
            != 0
        {
            return Ok(n);
        }
    }
    Err(invalid("unary code exceeds limit"))
}

fn read_motion(bits: &mut BitReader<'_>) -> Result<(i32, i32), DecodeError> {
    if bits
        .read_bit()
        .map_err(|_| invalid("truncated motion syntax"))?
        != 0
    {
        return Ok((0, 16));
    }
    let mut result = [0_i32; 2];
    for value in &mut result {
        let zeros = read_unary(bits, 31)?;
        if zeros >= 31 {
            return Err(invalid("motion code is too large"));
        }
        let suffix = if zeros == 0 {
            0
        } else {
            bits.read_bits(zeros)
                .map_err(|_| invalid("truncated motion code"))?
        };
        let code = (1_u32 << zeros) + suffix;
        let signed = if code & 1 != 0 {
            -((code >> 1) as i64)
        } else {
            (code >> 1) as i64
        };
        *value = i32::try_from(
            signed
                .checked_mul(16)
                .ok_or_else(|| invalid("motion overflow"))?,
        )
        .map_err(|_| invalid("motion overflow"))?;
    }
    Ok((result[0], result[1]))
}

fn raw(
    bits: &mut BitReader<'_>,
    state: &mut ColorState,
    out: &mut [Vec<u8>; 3],
    g: SliceGeometry,
    bx: usize,
    ly: usize,
    gy: usize,
) -> Result<(), DecodeError> {
    let pad = (8 - bits.bit_position() % 8) % 8;
    if pad != 0 {
        bits.read_bits(pad as u32)
            .map_err(|_| invalid("truncated raw block alignment"))?;
    }
    let mut scratch = [[0_u8; 256]; 3];
    for plane in &mut scratch {
        for sample in plane {
            *sample = bits
                .read_byte()
                .map_err(|_| invalid("truncated raw block"))?;
        }
    }
    write_block(state, out, g, bx, ly, gy, &scratch);
    Ok(())
}

fn residual(
    bits: &mut BitReader<'_>,
    state: &mut ColorState,
    out: &mut [Vec<u8>; 3],
    g: SliceGeometry,
    bx: usize,
    ly: usize,
    gy: usize,
) -> Result<(), DecodeError> {
    let predicted = bits
        .read_bit()
        .map_err(|_| invalid("truncated residual prediction flag"))?
        == 0;
    references::update(state, bx, ly, gy)?;
    let mut scratch = [[0_u8; 512]; 3];
    for plane in &mut scratch {
        decode_rle_plane(bits, plane, 256).map_err(|_| invalid("invalid residual plane"))?;
    }
    if predicted {
        for p in 0..3 {
            for i in 0..256 {
                let reference = if i < 16 {
                    state.banks[p][i + 1]
                } else {
                    scratch[p][i - 16]
                };
                let encoded = scratch[p][i];
                let delta = i32::from(encoded >> 1) ^ -i32::from(encoded & 1);
                let mut value = i32::from(reference) + delta * 4;
                if value > 257 {
                    value -= 260;
                } else if value < -2 {
                    value += 260;
                }
                scratch[p][i] = value.clamp(0, 255) as u8;
            }
        }
    }
    let packed: [[u8; 256]; 3] = [
        scratch[0][..256].try_into().unwrap(),
        scratch[1][..256].try_into().unwrap(),
        scratch[2][..256].try_into().unwrap(),
    ];
    write_block(state, out, g, bx, ly, gy, &packed);
    Ok(())
}

fn palette_block(
    bits: &mut BitReader<'_>,
    stream: &[u8],
    payload_offset: usize,
    palette: &mut Palette,
    state: &mut ColorState,
    out: &mut [Vec<u8>; 3],
    g: SliceGeometry,
    bx: usize,
    ly: usize,
    gy: usize,
) -> Result<(), DecodeError> {
    let marker = bits
        .read_bit()
        .map_err(|_| invalid("truncated palette marker"))?;
    if marker == 0 {
        if palette.prefix == 0 {
            return Err(invalid("palette continuation has no active palette"));
        }
    } else {
        let variant = bits
            .read_bit()
            .map_err(|_| invalid("truncated palette variant"))?;
        let prefix = bits
            .read_bits(4)
            .map_err(|_| invalid("truncated palette prefix"))?;
        if prefix == 0 {
            ensure_capacity(&mut palette.bytes, 3, "color palette")?;
            palette.bytes.clear();
            for _ in 0..3 {
                palette.bytes.push(
                    bits.read_byte()
                        .map_err(|_| invalid("truncated single-entry palette"))?,
                );
            }
            palette.entries = 1;
            palette.prefix = 0;
        } else {
            let count = bits
                .read_bits(prefix)
                .map_err(|_| invalid("truncated palette size"))? as usize
                + 1;
            let byte_count = count
                .checked_mul(3)
                .ok_or_else(|| allocation("color palette"))?;
            if variant != 0 {
                palette.bytes.clear();
            }
            let old_len = if variant == 0 {
                palette
                    .entries
                    .checked_mul(3)
                    .ok_or_else(|| allocation("color palette"))?
            } else {
                0
            };
            if palette.bytes.len() > byte_count {
                palette.bytes.truncate(byte_count);
            }
            if byte_count > old_len {
                let add = byte_count - old_len;
                ensure_capacity(&mut palette.bytes, add, "color palette")?;
                for _ in 0..add {
                    palette.bytes.push(
                        bits.read_byte()
                            .map_err(|_| invalid("truncated palette entries"))?,
                    );
                }
            }
            palette.entries = count;
            palette.prefix = prefix;
        }
    }
    if palette.prefix != 0 {
        let mut reader = ContextBitReader::new(stream);
        if !reader.initialize_at_bit(bits.bit_position()) {
            return Err(invalid("invalid palette index reader"));
        }
        let mut indices = [0_u8; 256];
        decode_contextual_indices(&mut reader, palette.prefix, &mut indices).map_err(|error| {
            DecodeError::Coefficients(CoefficientError::Syntax(Box::new(error))).at(
                crate::ErrorContext {
                    bit_position: Some(payload_offset * 8 + reader.logical_bit()),
                    ..crate::ErrorContext::default()
                },
            )
        })?;
        let final_bit = reader.logical_bit();
        while bits.bit_position() < final_bit {
            let take = (final_bit - bits.bit_position()).min(32) as u32;
            bits.read_bits(take)
                .map_err(|_| invalid("palette index reader passed stream end"))?;
        }
        let mut samples = [[0_u8; 256]; 3];
        for i in 0..256 {
            let index = usize::from(indices[i]);
            if index >= palette.entries {
                return Err(invalid("palette index exceeds palette size"));
            }
            for p in 0..3 {
                samples[p][i] = palette.bytes[index * 3 + p];
            }
        }
        write_block(state, out, g, bx, ly, gy, &samples);
    } else {
        let samples = [
            [palette.bytes[0]; 256],
            [palette.bytes[1]; 256],
            [palette.bytes[2]; 256],
        ];
        write_block(state, out, g, bx, ly, gy, &samples);
    }
    Ok(())
}

fn ensure_capacity(
    vec: &mut Vec<u8>,
    additional: usize,
    resource: &'static str,
) -> Result<(), DecodeError> {
    vec.try_reserve(additional)
        .map_err(|_| allocation(resource))
}

fn spatial_copy(
    state: &mut ColorState,
    out: &mut [Vec<u8>; 3],
    g: SliceGeometry,
    bx: usize,
    ly: usize,
    gy: usize,
) -> Result<(), DecodeError> {
    let (dx, dy) = if bx == 0 { (0, 16) } else { (16, 0) };
    motion_copy(state, out, g, bx, ly, gy, dx, dy)
}

fn motion_copy(
    state: &mut ColorState,
    out: &mut [Vec<u8>; 3],
    g: SliceGeometry,
    bx: usize,
    ly: usize,
    gy: usize,
    dx: i32,
    dy: i32,
) -> Result<(), DecodeError> {
    let x = bx * 16;
    let global_y = gy * 16;
    let mut scratch = [[0_u8; 256]; 3];
    for p in 0..3 {
        copy_color_motion(
            &mut state.planes[p],
            state.stride,
            state.stride,
            state.total_block_rows * 16,
            x,
            global_y,
            Motion { x: dx, y: dy },
        )
        .map_err(|_| invalid("motion source is outside picture history"))?;
        for row in 0..16 {
            let src = (global_y + row) * state.stride + x;
            scratch[p][row * 16..row * 16 + 16].copy_from_slice(&state.planes[p][src..src + 16]);
        }
    }
    write_block(state, out, g, bx, ly, gy, &scratch);
    Ok(())
}

fn write_block<const N: usize>(
    state: &mut ColorState,
    out: &mut [Vec<u8>; 3],
    _g: SliceGeometry,
    bx: usize,
    ly: usize,
    gy: usize,
    samples: &[[u8; N]; 3],
) {
    let x = bx * 16;
    for p in 0..3 {
        for row in 0..16 {
            let local = (ly * 16 + row) * state.stride + x;
            let global = (gy * 16 + row) * state.stride + x;
            out[p][local..local + 16].copy_from_slice(&samples[p][row * 16..row * 16 + 16]);
            state.planes[p][global..global + 16]
                .copy_from_slice(&samples[p][row * 16..row * 16 + 16]);
        }
    }
}
