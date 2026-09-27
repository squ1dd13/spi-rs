//! Read colour methods and reconstruct one record's block grid.
//!
//! The ordinary bit cursor starts at payload bit zero. Method prefixes select a copy, byte residuals, predictive coefficients, a palette, or raw planes. Every successful block writes 256 samples to each of the G, B, R planes in temporary record storage and persistent picture storage. [`decode_slice`] adds the record header offset to the final cursor.
//!
//! The palette starts empty for each record and can be reused by later palette blocks in that record. Coefficient-mode history is allocated on the first predictive block and reset at record end. Persistent reference banks are prepared by residual and predictive methods only. The availability flag is set after a method is read, before its reconstruction; it is not a transaction marker and is not checked by motion copying.
//!
//! The crate page's “Block methods” section gives the encoded mapping and byte residual arithmetic. This module owns the method dispatch and ordinary/contextual reader hand-off; [`predictive`](../predictive/index.html) owns coefficient reconstruction.

use super::{predictive, references};
use crate::{
    bits::{BitReader, ContextBitReader},
    colour::{BlockPosition, ColourMemoryBudget, ColourSlice, ColourState, allocation, invalid},
    decoder::SliceGeometry,
    entropy::{CoefficientHistory, decode_contextual_indices, decode_rle_plane},
    error::{CoefficientError, DecodeError},
    format::Record,
    surface::{Motion, copy_colour_motion},
};

/// Retains the palette coding state shared by consecutive colour blocks.
///
/// A non-zero prefix gives the bit width of contextual palette indices. Palette bytes are interleaved in green-blue-red order and can be replaced, extended, or reused by later palette blocks.
struct Palette {
    /// The number of complete three-byte entries.
    entries: usize,
    /// The contextual index width, or zero for a single entry.
    prefix: u32,
    /// Interleaved green-blue-red palette entries.
    bytes: Vec<u8>,
    /// Tracks this vector's capacity against the live-memory limit.
    budget: ColourMemoryBudget,
}

/// Decodes all colour blocks in one record slice.
///
/// This validates geometry, allocates three padded slice planes, and visits blocks in row-major order. Palette state and lazily allocated coefficient histories are shared within the slice, while reconstructed samples are also committed to picture-wide `state`.
///
/// The returned endpoint includes the 14-byte record framing and syntax headers. Invalid syntax, references, allocations, or memory use return [`DecodeError`].
pub(super) fn decode_slice(
    record: &Record<'_>,
    geometry: SliceGeometry,
    state: &mut ColourState,
    budget: ColourMemoryBudget,
) -> Result<ColourSlice, DecodeError> {
    validate_geometry(geometry, state)?;
    if record.payload_offset.checked_mul(8).is_none()
        || record.data.len() < record.payload_offset
        || record.payload.is_empty()
    {
        return Err(invalid("record has no colour payload"));
    }
    let stride = geometry
        .block_width
        .checked_mul(16)
        .ok_or_else(|| allocation("colour slice"))?;
    let plane_len = stride
        .checked_mul(
            geometry
                .block_rows
                .checked_mul(16)
                .ok_or_else(|| allocation("colour slice"))?,
        )
        .ok_or_else(|| allocation("colour slice"))?;
    let mut planes = [Vec::new(), Vec::new(), Vec::new()];
    for plane in &mut planes {
        plane
            .try_reserve_exact(plane_len)
            .map_err(|_| allocation("colour slice"))?;
        plane.resize(plane_len, 0);
    }
    let mut bits = BitReader::new(record.payload);
    let mut palette = Palette {
        entries: 0,
        prefix: 0,
        bytes: Vec::new(),
        budget,
    };
    let mut coefficient_histories: Option<[CoefficientHistory; 3]> = None;
    palette.budget.check(&planes, &coefficient_histories)?;
    for local_y in 0..geometry.block_rows {
        let global_block_y = state.global_row(local_y, geometry.absolute_block_row)?;
        if local_y != 0
            && let Some(histories) = coefficient_histories.as_mut()
        {
            for (plane, history) in histories.iter_mut().enumerate() {
                history
                    .begin_block_row(
                        u16::try_from(local_y).expect("block row fits history"),
                        plane != 0,
                    )
                    .map_err(|error| {
                        DecodeError::Coefficients(crate::error::CoefficientError::Syntax(Box::new(
                            error,
                        )))
                    })?;
            }
        }
        for block_x in 0..geometry.block_width {
            decode_colour_block(
                &mut bits,
                record,
                state,
                &mut planes,
                BlockPosition {
                    geometry,
                    x: block_x,
                    local_y,
                    global_y: global_block_y,
                },
                &mut palette,
                &mut coefficient_histories,
            )?;
        }
    }
    Ok(ColourSlice {
        planes,
        stride,
        end_bit: record
            .payload_offset
            .checked_mul(8)
            .and_then(|base| base.checked_add(bits.bit_position()))
            .ok_or_else(|| invalid("colour bit position overflow"))?,
    })
}

/// Decodes and commits one colour block selected by its method prefix.
///
/// Method 0 spatially copies, 1 motion-copies, 2 decodes byte residuals, 3 uses transform prediction, 4 expands a palette, and 5 reads raw planar samples. The block becomes available before reconstruction so method-specific validation determines whether its references are legal.
///
/// Any failure is wrapped with record-relative bit position and record-local block number.
fn decode_colour_block(
    bits: &mut BitReader<'_>,
    record: &Record<'_>,
    state: &mut ColourState,
    planes: &mut [Vec<u8>; 3],
    position: BlockPosition,
    palette: &mut Palette,
    coefficient_histories: &mut Option<[CoefficientHistory; 3]>,
) -> Result<(), DecodeError> {
    let BlockPosition {
        geometry,
        x: block_x,
        local_y,
        global_y: global_block_y,
    } = position;
    let block_index = local_y * geometry.block_width + block_x;
    let method = read_method(bits).map_err(|error| {
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
            0 => spatial_copy(state, planes, position)?,
            1 => {
                let (dx, dy) = read_motion(bits)?;
                motion_copy(state, planes, position, Motion { x: dx, y: dy })?;
            }
            2 => residual(bits, state, planes, position)?,
            3 => {
                predictive::decode_block(
                    bits,
                    record,
                    state,
                    planes,
                    position,
                    coefficient_histories,
                    palette.budget,
                )?;
            }
            4 => palette_block(
                bits,
                record,
                palette,
                state,
                planes,
                position,
                coefficient_histories,
            )?,
            5 => raw(bits, state, planes, position)?,
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
    Ok(())
}

/// Attaches colour block and bit positions to a decoder error.
///
/// The record index comes from the parsed syntax header. Component context remains absent until a component decoder adds it.
fn at_block(error: DecodeError, record: &Record<'_>, block: usize, bit: usize) -> DecodeError {
    error.at(crate::ErrorContext {
        record: Some(record.index as usize),
        block: Some(block),
        component: None,
        bit_position: Some(bit),
    })
}

/// Validates slice geometry against persistent colour history.
///
/// The block width and derived stride must match the picture, and the non-empty row range must fit inside persistent history. Invalid values become colour syntax errors.
fn validate_geometry(geometry: SliceGeometry, state: &ColourState) -> Result<(), DecodeError> {
    if geometry.block_width != state.block_width
        || geometry.block_rows == 0
        || geometry
            .absolute_block_row
            .checked_add(geometry.block_rows)
            .is_none_or(|end| end > state.total_block_rows)
    {
        return Err(invalid("slice geometry does not match colour history"));
    }
    let padded_width = geometry
        .block_width
        .checked_mul(16)
        .ok_or_else(|| invalid("slice width overflow"))?;
    if padded_width != state.stride {
        return Err(invalid("slice stride does not match picture"));
    }
    Ok(())
}

/// Reads the variable-length colour block method code.
///
/// Prefix `1` selects method 0, `01` selects method 1, and `00xx` selects methods 2 through 5. Truncated prefixes return invalid colour syntax.
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
        .read_u8(2)
        .map_err(|_| invalid("truncated block method"))?
        + 2)
}

/// Reads a one-terminated unary value below `limit`.
///
/// The return value is the number of leading zero bits. Truncation or a prefix without a terminating one within the limit is invalid colour syntax.
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

/// Reads a colour motion vector and returns displacement in samples.
///
/// A leading one abbreviates `(0, 16)`. Otherwise, each component is an Exp-Golomb-like magnitude whose low code bit selects sign, then is multiplied by 16. Excessive or truncated codes are rejected.
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
            -i64::from(code >> 1)
        } else {
            i64::from(code >> 1)
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

/// Reads one byte-aligned raw green-blue-red colour block.
///
/// Padding advances to the next byte boundary without requiring zero bits. Exactly 256 bytes are read for each plane and committed only after the complete block is available, so truncation cannot partially update sample planes; the outer caller has already set availability.
fn raw(
    bits: &mut BitReader<'_>,
    state: &mut ColourState,
    out: &mut [Vec<u8>; 3],
    position: BlockPosition,
) -> Result<(), DecodeError> {
    let pad = (8 - bits.bit_position() % 8) % 8;
    if pad != 0 {
        bits.read_bits(u32::try_from(pad).expect("padding has at most seven bits"))
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
    write_block(state, out, position, &scratch);
    Ok(())
}

/// Decodes three run-length-coded residual planes.
///
/// A zero prediction flag enables vertical delta reconstruction. Each signed folded residual is multiplied by four, added to the top reference or previous output row, adjusted once by subtracting 260 above 257 or adding 260 below -2, then clipped to a byte. A set flag leaves decoded bytes as direct samples.
fn residual(
    bits: &mut BitReader<'_>,
    state: &mut ColourState,
    out: &mut [Vec<u8>; 3],
    position: BlockPosition,
) -> Result<(), DecodeError> {
    let predicted = bits
        .read_bit()
        .map_err(|_| invalid("truncated residual prediction flag"))?
        == 0;
    references::update(state, position.x, position.local_y, position.global_y)?;
    let mut scratch = [[0_u8; 256]; 3];
    for plane in &mut scratch {
        decode_rle_plane(bits, plane, 256).map_err(|_| invalid("invalid residual plane"))?;
    }
    if predicted {
        for (plane_index, plane) in scratch.iter_mut().enumerate() {
            for sample_index in 0..256 {
                let reference = if sample_index < 16 {
                    state.banks[plane_index][sample_index + 1]
                } else {
                    plane[sample_index - 16]
                };
                let encoded = plane[sample_index];
                let delta = i32::from(encoded >> 1) ^ -i32::from(encoded & 1);
                let mut value = i32::from(reference) + delta * 4;
                if value > 257 {
                    value -= 260;
                } else if value < -2 {
                    value += 260;
                }
                plane[sample_index] =
                    u8::try_from(value.clamp(0, 255)).expect("clamped sample fits in u8");
            }
        }
    }
    write_block(state, out, position, &scratch);
    Ok(())
}

/// Reads or reuses the palette definition for a palette block.
///
/// A zero marker continues the active palette with a non-zero prefix. A set marker carries a replacement or extension variant and a four-bit index prefix. Prefix zero defines one green-blue-red entry; otherwise the prefix encodes the entry count field width.
///
/// Continuation without active state, truncated fields, size overflow, allocation failure, and memory-budget violations return [`DecodeError`].
fn read_palette(
    bits: &mut BitReader<'_>,
    palette: &mut Palette,
    planes: &[Vec<u8>; 3],
    histories: &Option<[CoefficientHistory; 3]>,
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
            palette.bytes.clear();
            reserve_palette(palette, 3, planes, histories)?;
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
                .ok_or_else(|| allocation("colour palette"))?;
            if variant != 0 {
                palette.bytes.clear();
            }
            let old_len = if variant == 0 {
                palette
                    .entries
                    .checked_mul(3)
                    .ok_or_else(|| allocation("colour palette"))?
            } else {
                0
            };
            if palette.bytes.len() > byte_count {
                palette.bytes.truncate(byte_count);
            }
            if byte_count > old_len {
                let add = byte_count - old_len;
                reserve_palette(palette, add, planes, histories)?;
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
    Ok(())
}

/// Reserves palette bytes while enforcing the live-memory budget.
///
/// The prospective required capacity is checked before allocation and the allocator's actual capacity is checked afterwards. This handles allocators that reserve more than requested.
fn reserve_palette(
    palette: &mut Palette,
    additional: usize,
    planes: &[Vec<u8>; 3],
    histories: &Option<[CoefficientHistory; 3]>,
) -> Result<(), DecodeError> {
    let required_capacity = palette
        .bytes
        .len()
        .checked_add(additional)
        .ok_or_else(|| allocation("colour palette"))?;
    palette
        .budget
        .with_palette_capacity(required_capacity.max(palette.bytes.capacity()))
        .check(planes, histories)?;
    palette
        .bytes
        .try_reserve_exact(additional)
        .map_err(|_| allocation("colour palette"))?;
    palette.budget = palette
        .budget
        .with_palette_capacity(palette.bytes.capacity());
    palette.budget.check(planes, histories)
}

/// Expands one palette-coded block into three sample planes.
///
/// A zero-prefix palette fills every sample directly. For a non-zero prefix, the decoder uses the contextual reader at the ordinary cursor, decodes 256 indices, advances the ordinary cursor to the same endpoint, validates each index, and expands interleaved entries. A non-zero prefix can describe one entry and still carries coded indices.
///
/// Contextual entropy failures include their record-relative bit position. Invalid initial lookahead, out-of-range indices, or desynchronised endpoints are rejected.
fn palette_block(
    bits: &mut BitReader<'_>,
    record: &Record<'_>,
    palette: &mut Palette,
    state: &mut ColourState,
    out: &mut [Vec<u8>; 3],
    position: BlockPosition,
    histories: &Option<[CoefficientHistory; 3]>,
) -> Result<(), DecodeError> {
    read_palette(bits, palette, out, histories)?;
    if palette.prefix != 0 {
        let mut reader = ContextBitReader::new(record.payload);
        if !reader.initialise_at_bit(bits.bit_position()) {
            return Err(invalid("invalid palette index reader"));
        }
        let mut indices = [0_u8; 256];
        decode_contextual_indices(&mut reader, palette.prefix, &mut indices).map_err(|error| {
            DecodeError::Coefficients(CoefficientError::Syntax(Box::new(error))).at(
                crate::ErrorContext {
                    bit_position: Some(record.payload_offset * 8 + reader.logical_bit()),
                    ..crate::ErrorContext::default()
                },
            )
        })?;
        let final_bit = reader.logical_bit();
        while bits.bit_position() < final_bit {
            let take = u32::try_from((final_bit - bits.bit_position()).min(32))
                .expect("read length is at most 32 bits");
            bits.read_bits(take)
                .map_err(|_| invalid("palette index reader passed stream end"))?;
        }
        let mut samples = [[0_u8; 256]; 3];
        for sample_index in 0..256 {
            let index = usize::from(indices[sample_index]);
            if index >= palette.entries {
                return Err(invalid("palette index exceeds palette size"));
            }
            for (plane_index, sample) in samples.iter_mut().enumerate() {
                sample[sample_index] = palette.bytes[index * 3 + plane_index];
            }
        }
        write_block(state, out, position, &samples);
    } else {
        let samples = [
            [palette.bytes[0]; 256],
            [palette.bytes[1]; 256],
            [palette.bytes[2]; 256],
        ];
        write_block(state, out, position, &samples);
    }
    Ok(())
}

/// Copies the default neighbouring colour block into the current position.
///
/// Blocks in the first column copy from the block above; all other blocks copy from the block to the left. The shared motion-copy path validates that the selected source exists.
fn spatial_copy(
    state: &mut ColourState,
    out: &mut [Vec<u8>; 3],
    position: BlockPosition,
) -> Result<(), DecodeError> {
    let (dx, dy) = if position.x == 0 { (0, 16) } else { (16, 0) };
    motion_copy(state, out, position, Motion { x: dx, y: dy })
}

/// Copies a motion-referenced block from persistent picture history.
///
/// The same vector is applied to all three colour planes. Reconstructed samples are read back into scratch and then written to both the slice output and persistent state so reference-bank updates see one consistent block.
fn motion_copy(
    state: &mut ColourState,
    out: &mut [Vec<u8>; 3],
    position: BlockPosition,
    motion: Motion,
) -> Result<(), DecodeError> {
    let x = position.x * 16;
    let global_y = position.global_y * 16;
    let mut scratch = [[0_u8; 256]; 3];
    for (plane_index, plane) in scratch.iter_mut().enumerate() {
        copy_colour_motion(
            &mut state.planes[plane_index],
            state.stride,
            state.stride,
            state.total_block_rows * 16,
            x,
            global_y,
            motion,
        )
        .map_err(|_| invalid("motion source is outside picture history"))?;
        for row in 0..16 {
            let src = (global_y + row) * state.stride + x;
            plane[row * 16..row * 16 + 16]
                .copy_from_slice(&state.planes[plane_index][src..src + 16]);
        }
    }
    write_block(state, out, position, &scratch);
    Ok(())
}

/// Writes one reconstructed colour block to slice and picture planes.
///
/// `position` and allocation checks performed by the slice decoder guarantee all 16 by 16 destinations are in bounds. Plane order remains green, blue, red.
fn write_block(
    state: &mut ColourState,
    out: &mut [Vec<u8>; 3],
    position: BlockPosition,
    samples: &[[u8; 256]; 3],
) {
    let x = position.x * 16;
    for plane_index in 0..3 {
        for row in 0..16 {
            let local = (position.local_y * 16 + row) * state.stride + x;
            let global = (position.global_y * 16 + row) * state.stride + x;
            out[plane_index][local..local + 16]
                .copy_from_slice(&samples[plane_index][row * 16..row * 16 + 16]);
            state.planes[plane_index][global..global + 16]
                .copy_from_slice(&samples[plane_index][row * 16..row * 16 + 16]);
        }
    }
}
