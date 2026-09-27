//! Predictive colour blocks use signed histories and fixed-size reconstruction scratch.

use crate::{
    bits::{BitReader, ContextBitReader},
    colour::{ColourState, allocation, invalid, references},
    decoder::SliceGeometry,
    entropy::{
        CallbackKind, CoefficientHistory, ComponentRequest, ComponentResult,
        decode_contextual_component,
    },
    error::{CoefficientError, DecodeError, SyntaxError},
    format::Record,
    prediction::{
        build_signed_references, filter_signed_references, predict_signed_angular,
        predict_signed_angular_reference_b, predict_signed_dc, predict_signed_filtered_diagonal,
        predict_signed_left_edge_directional, predict_signed_planar,
        predict_signed_side_directional, predict_signed_top_edge_directional_signed,
        predict_signed_wide_angular_reference_a, predict_signed_width4_angular,
        predict_signed_width4_diagonal, prediction_direction_for_mode,
    },
    transform::{
        inverse_transform_16x16, reconstruct_component_residual, truncate_signed_halfword,
    },
};

fn low_byte(value: i16) -> u8 {
    value.to_le_bytes()[0]
}

pub(super) fn decode_block(
    bits: &mut BitReader<'_>,
    record: &Record<'_>,
    state: &mut ColourState,
    output: &mut [Vec<u8>; 3],
    geometry: SliceGeometry,
    block_x: usize,
    local_y: usize,
    global_y: usize,
    block_index: usize,
    _method_start: usize,
    histories: &mut Option<[CoefficientHistory; 3]>,
) -> Result<(), DecodeError> {
    ensure_histories(histories, state.block_width, geometry.block_rows, local_y)?;
    let histories = histories.as_mut().expect("histories initialised");
    references::update(state, block_x, local_y, global_y)?;

    let mut reader = ContextBitReader::new(record.payload);
    if !reader.initialise_at_bit(bits.bit_position()) {
        return Err(invalid("invalid predictive reader position"));
    }
    let initialised = reader
        .read_value(1)
        .ok_or_else(|| invalid("truncated predictive state"))?;
    let luma_subdivided = reader
        .read_value(1)
        .ok_or_else(|| invalid("truncated luma layout"))?
        != 0;
    let mut signed = [[0_i16; 256]; 3];
    if luma_subdivided {
        for component in 0..4 {
            let result = decode_component(
                &mut reader,
                &mut histories[0],
                component,
                block_x,
                8,
                0,
                false,
                component,
                record.payload_offset,
            )?;
            reconstruct_component(
                &result,
                initialised == 0,
                &state.histories[3],
                &state.histories[0],
                0,
                component,
                4,
                &mut signed[0],
            )?;
        }
    } else {
        let result = decode_component(
            &mut reader,
            &mut histories[0],
            0,
            block_x,
            16,
            0,
            true,
            0,
            record.payload_offset,
        )?;
        reconstruct_component(
            &result,
            initialised == 0,
            &state.histories[3],
            &state.histories[0],
            0,
            0,
            1,
            &mut signed[0],
        )?;
    }
    for plane in 1..3 {
        let subdivided = reader
            .read_value(1)
            .ok_or_else(|| invalid("truncated chroma layout"))?
            != 0;
        if subdivided {
            for component in 0..4 {
                let result = decode_component(
                    &mut reader,
                    &mut histories[plane],
                    component,
                    block_x,
                    8,
                    0,
                    false,
                    plane * 4 + component,
                    record.payload_offset,
                )?;
                reconstruct_component(
                    &result,
                    initialised == 0,
                    &state.histories[plane + 3],
                    &state.histories[plane],
                    plane,
                    component,
                    4,
                    &mut signed[plane],
                )?;
            }
        } else {
            let result = decode_component(
                &mut reader,
                &mut histories[plane],
                0,
                block_x,
                16,
                0,
                true,
                plane * 4,
                record.payload_offset,
            )?;
            reconstruct_component(
                &result,
                initialised == 0,
                &state.histories[plane + 3],
                &state.histories[plane],
                plane,
                0,
                1,
                &mut signed[plane],
            )?;
        }
    }
    let final_bit = reader.logical_bit();
    while bits.bit_position() < final_bit {
        let take = u32::try_from((final_bit - bits.bit_position()).min(32))
            .expect("read length is at most 32 bits");
        bits.read_bits(take)
            .map_err(|_| invalid("predictive reader passed stream end"))?;
    }

    let mut converted = [[0_u8; 256]; 3];
    for i in 0..256 {
        let co = i32::from(signed[1][i]) - 256;
        let luma = u32::from(signed[0][i].cast_unsigned());
        let w2 = luma.wrapping_sub(co.cast_unsigned() >> 1);
        let red_raw = truncate_signed_halfword(w2.wrapping_add(co.cast_unsigned()));
        let red = if (red_raw.cast_unsigned() & 0xff00) != 0 {
            low_byte((-red_raw) >> 15)
        } else {
            u8::try_from(red_raw).expect("red sample is below 256")
        };
        let cg = u32::try_from(i32::from(signed[2][i]) + 0x1ff00)
            .expect("chroma bias makes the value nonnegative");
        let green_raw = truncate_signed_halfword(w2.wrapping_sub(cg >> 1));
        let green = if (green_raw.cast_unsigned() & 0xff00) != 0 {
            u8::try_from(((-green_raw) >> 15) & 0xff).expect("masked green sample fits u8")
        } else {
            u8::try_from(green_raw).expect("green sample is below 256")
        };
        let blue_raw = truncate_signed_halfword(
            u32::from(signed[2][i].cast_unsigned())
                .wrapping_add(u32::from(green))
                .wrapping_add(0xff00),
        );
        let blue = if blue_raw.cast_unsigned() < 0x100 {
            u8::try_from(blue_raw).expect("blue sample is below 256")
        } else {
            low_byte((-blue_raw) >> 15)
        };
        converted[0][i] = red;
        converted[1][i] = green;
        converted[2][i] = blue;
    }
    let x = block_x * 16;
    let local_y_px = local_y * 16;
    let global_y_px = global_y * 16;
    for p in 0..3 {
        for row in 0..16 {
            let src = row * 16;
            let dst = (local_y_px + row) * state.stride + x;
            let hist = (global_y_px + row) * state.stride + x;
            output[p][dst..dst + 16].copy_from_slice(&converted[p][src..src + 16]);
            state.planes[p][hist..hist + 16].copy_from_slice(&converted[p][src..src + 16]);
        }
    }
    let _ = (geometry, block_index);
    Ok(())
}

fn ensure_histories(
    histories: &mut Option<[CoefficientHistory; 3]>,
    block_width: usize,
    rows: usize,
    local_y: usize,
) -> Result<(), DecodeError> {
    if histories.is_none() {
        let stride = block_width
            .checked_mul(4)
            .and_then(|v| v.checked_add(1))
            .and_then(|v| u16::try_from(v).ok())
            .ok_or_else(|| allocation("colour coefficient history"))?;
        let rows = u16::try_from(rows).map_err(|_| allocation("colour coefficient history"))?;
        let make = || CoefficientHistory::new(stride, rows).map_err(coefficient_error);
        *histories = Some([make()?, make()?, make()?]);
        for y in 1..=local_y {
            for (plane, history) in histories
                .as_mut()
                .expect("set above")
                .iter_mut()
                .enumerate()
            {
                history
                    .begin_block_row(
                        u16::try_from(y).expect("block row fits history"),
                        plane != 0,
                    )
                    .map_err(coefficient_error)?;
            }
        }
    }
    Ok(())
}

fn decode_component(
    reader: &mut ContextBitReader<'_>,
    history: &mut CoefficientHistory,
    argument: usize,
    block_x: usize,
    maximum_width: u8,
    fixed_width: u8,
    width_selector_already_read: bool,
    component_index: usize,
    payload_offset: usize,
) -> Result<ComponentResult, DecodeError> {
    let result = decode_contextual_component(
        reader,
        history,
        ComponentRequest {
            argument: u32::try_from(argument).expect("component argument is at most four"),
            horizontal_offset: u16::try_from(block_x * 16)
                .map_err(|_| invalid("predictive block offset overflow"))?,
            maximum_width,
            fixed_width,
            colour_long_symbols: true,
            width_selector_already_read,
        },
    );
    result.map_err(|error| {
        coefficient_error(error).at(crate::ErrorContext {
            component: Some(component_index),
            bit_position: Some(payload_offset * 8 + reader.logical_bit()),
            ..crate::ErrorContext::default()
        })
    })
}

fn reconstruct_component(
    component: &ComponentResult,
    alternate_profile: bool,
    left: &[i16; 33],
    top: &[i16; 33],
    plane: usize,
    argument: usize,
    component_count: usize,
    output: &mut [i16; 256],
) -> Result<(), DecodeError> {
    if component.count == 0
        || component.count > 4
        || component.width as usize * component.width as usize > 256
    {
        return Err(invalid("invalid predictive component dimensions"));
    }
    let width = component.width as usize;
    let mut residual = [0_i16; 256];
    if width == 16 {
        if component.callbacks[0] != CallbackKind::Inactive {
            inverse_transform_16x16(
                &component.values[0],
                if alternate_profile { 160 } else { 144 },
                &mut residual,
            )
            .map_err(|e| DecodeError::Coefficients(CoefficientError::Transform(Box::new(e))))?;
        }
    } else {
        let mut coeffs = [[0_i16; 64]; 4];
        let mut callbacks = [0_u8; 4];
        for slot in 0..component.count {
            coeffs[slot][..width * width].copy_from_slice(&component.values[slot][..width * width]);
            callbacks[slot] = match component.callbacks[slot] {
                CallbackKind::Inactive => 0,
                CallbackKind::ConstantResidual => 1,
                CallbackKind::FullBlockTransform => 7,
            };
        }
        reconstruct_component_residual(
            width,
            component.count,
            &coeffs,
            &callbacks,
            alternate_profile,
            &mut residual,
        )
        .map_err(|e| DecodeError::Coefficients(CoefficientError::Transform(Box::new(e))))?;
    }
    let offset_x = if component_count == 4 {
        (argument & 1) * 8
    } else {
        0
    };
    let offset_y = if component_count == 4 {
        (argument >> 1) * 8
    } else {
        0
    };
    // The source stream keeps each component argument's constructor buffers
    // alive across its slots. Some selectors write only part of a reference
    // edge, so later slots can consume values written by earlier slots.
    let mut slot_left = [0_i16; 33];
    let mut slot_top = [0_i16; 33];
    for slot in 0..component.count {
        let width = component.width as usize;
        let slot_x = offset_x + if width == 4 { (slot & 1) * 4 } else { 0 };
        let slot_y = offset_y + if width == 4 { (slot >> 1) * 4 } else { 0 };
        let selector = if width == 4 {
            argument * 4 + slot
        } else {
            argument
        };
        construct_colour_refs(
            width,
            selector,
            left,
            top,
            output,
            slot_x,
            slot_y,
            &mut slot_left,
            &mut slot_top,
        )?;
        let mut pred = [0_i16; 256];
        predict(
            component.modes[slot],
            width,
            &slot_left,
            &slot_top,
            &mut pred,
        )?;
        for row in 0..width {
            for col in 0..width {
                let i = row * width + col;
                let dst = (slot_y + row) * 16 + slot_x + col;
                let cap = if plane == 0 { 255 } else { 511 };
                output[dst] = i16::try_from(
                    (i32::from(pred[i]) + i32::from(residual[slot * width * width + i]))
                        .clamp(0, cap),
                )
                .expect("clamped colour sample fits i16");
            }
        }
    }
    Ok(())
}

fn construct_colour_refs(
    width: usize,
    selector: usize,
    left_bank: &[i16; 33],
    top_bank: &[i16; 33],
    workspace: &[i16; 256],
    component_x: usize,
    component_y: usize,
    left: &mut [i16; 33],
    top: &mut [i16; 33],
) -> Result<(), DecodeError> {
    if width == 16 {
        *left = *left_bank;
        *top = *top_bank;
        return Ok(());
    }
    if width == 8 {
        let origin = component_y * 16 + component_x;
        match selector {
            0 => {
                left[..17].copy_from_slice(&left_bank[..17]);
                top[..17].copy_from_slice(&top_bank[..17]);
            }
            1 if component_x != 0 => {
                top[..17].copy_from_slice(&top_bank[8..25]);
                left[0] = top[0];
                for i in 0..8 {
                    left[i + 1] = workspace[origin - 1 + i * 16];
                }
                let edge = left[8];
                left[9..17].fill(edge);
            }
            2 if component_y != 0 => {
                left[..17].copy_from_slice(&left_bank[8..25]);
                top[0] = left[0];
                top[1..17].copy_from_slice(&workspace[origin - 16..origin]);
            }
            3 if component_x != 0 && component_y != 0 => {
                left[0] = workspace[origin - 17];
                top[0] = left[0];
                for i in 0..8 {
                    left[i + 1] = workspace[origin - 1 + i * 16];
                    top[i + 1] = workspace[origin - 16 + i];
                }
                let left_edge = left[8];
                let top_edge = top[8];
                left[9..17].fill(left_edge);
                top[9..17].fill(top_edge);
            }
            _ => return Err(invalid("invalid width-eight colour reference selector")),
        }
        return Ok(());
    }
    if width != 4 || selector > 15 {
        return Err(invalid("invalid width-four colour reference selector"));
    }
    let origin = component_y * 16 + component_x;
    let vertical = |count: usize, extend: bool, out: &mut [i16; 33]| -> bool {
        if component_x == 0 {
            return false;
        }
        for i in 0..count {
            out[i + 1] = workspace[origin - 1 + i * 16];
        }
        if extend {
            let edge = out[count];
            out[count + 1..9].fill(edge);
        }
        true
    };
    let above = |count: usize, extend: bool, out: &mut [i16; 33]| -> bool {
        if component_y == 0 {
            return false;
        }
        for i in 0..count {
            out[i + 1] = workspace[origin - 16 + i];
        }
        if extend {
            let edge = out[count];
            out[count + 1..9].fill(edge);
        }
        true
    };
    let ok = match selector {
        0 => {
            left[..9].copy_from_slice(&left_bank[..9]);
            top[..9].copy_from_slice(&top_bank[..9]);
            true
        }
        1 => {
            top[..9].copy_from_slice(&top_bank[4..13]);
            left[0] = top[0];
            vertical(4, true, left)
        }
        2 => {
            left[..9].copy_from_slice(&left_bank[4..13]);
            top[0] = left[0];
            above(8, false, top)
        }
        3 if component_x != 0 && component_y != 0 => {
            left[0] = workspace[origin - 17];
            top[0] = left[0];
            vertical(4, true, left) && above(4, false, top)
        }
        4 => {
            top[..9].copy_from_slice(&top_bank[8..17]);
            left[0] = top[0];
            vertical(8, false, left)
        }
        5 => {
            top[..9].copy_from_slice(&top_bank[12..21]);
            left[0] = top[0];
            vertical(4, true, left)
        }
        6 | 9 | 14 if component_x != 0 && component_y != 0 => {
            left[0] = workspace[origin - 17];
            top[0] = left[0];
            vertical(4, true, left) && above(8, false, top)
        }
        7 | 11 | 13 | 15 if component_x != 0 && component_y != 0 => {
            left[0] = workspace[origin - 17];
            top[0] = left[0];
            vertical(4, true, left) && above(4, true, top)
        }
        8 => {
            left[..9].copy_from_slice(&left_bank[8..17]);
            top[0] = left[0];
            above(8, false, top)
        }
        10 => {
            left[..9].copy_from_slice(&left_bank[12..21]);
            top[0] = left[0];
            above(8, false, top)
        }
        12 if component_x != 0 && component_y != 0 => {
            left[0] = workspace[origin - 17];
            top[0] = left[0];
            vertical(8, false, left) && above(8, false, top)
        }
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(invalid("colour reference is unavailable for constructor"))
    }
}

fn predict(
    mode: u8,
    width: usize,
    left: &[i16; 33],
    top: &[i16; 33],
    out: &mut [i16; 256],
) -> Result<(), DecodeError> {
    let prediction_error = |e| {
        DecodeError::Prediction(crate::error::PredictionError {
            source: Box::new(e),
        })
    };
    if mode == 0 {
        for row in 0..width {
            for col in 0..width {
                out[row * width + col] = top[col + 1];
            }
        }
        return Ok(());
    }
    if mode == 1 {
        for row in 0..width {
            for col in 0..width {
                out[row * width + col] = left[row + 1];
            }
        }
        return Ok(());
    }
    if mode == 2 {
        return predict_signed_dc(left, top, width, out).map_err(prediction_error);
    }
    if mode == 17 {
        let refs = build_signed_references(left, top);
        return predict_signed_planar(&refs, width, out).map_err(prediction_error);
    }
    let mut filtered_left = [0_i16; 33];
    let mut filtered_top = [0_i16; 33];
    let needs_filter = if width == 4 {
        matches!(mode, 3 | 6 | 9)
    } else if width == 8 {
        (3..=8).contains(&mode)
    } else {
        (3..=16).contains(&mode)
    };
    if needs_filter {
        filter_signed_references(left, top, width, &mut filtered_left, &mut filtered_top)
            .map_err(prediction_error)?;
    }
    if width == 4 && mode == 3 {
        return predict_signed_width4_diagonal(&filtered_left, &filtered_top, out)
            .map_err(prediction_error);
    }
    if width == 4 && mode == 4 {
        return predict_signed_wide_angular_reference_a(left, top, 4, out)
            .map_err(prediction_error);
    }
    if width == 4 && matches!(mode, 5 | 7) {
        return predict_signed_width4_angular(
            top,
            mode,
            if mode == 7 { Some(left) } else { None },
            out,
        )
        .map_err(prediction_error);
    }
    if width == 4 && mode == 6 {
        return predict_signed_top_edge_directional_signed(&filtered_top, 4, out)
            .map_err(prediction_error);
    }
    if width == 4 && mode == 8 {
        return predict_signed_side_directional(left, 4, out).map_err(prediction_error);
    }
    if (width == 8 || width == 16) && mode == 3 {
        return predict_signed_filtered_diagonal(&filtered_left, &filtered_top, width, out)
            .map_err(prediction_error);
    }
    if (width == 8 || width == 16) && mode == 4 {
        return predict_signed_wide_angular_reference_a(&filtered_left, &filtered_top, width, out)
            .map_err(prediction_error);
    }
    if (width == 8 || width == 16) && mode == 5 {
        return predict_signed_angular_reference_b(&filtered_top, width, out)
            .map_err(prediction_error);
    }
    if (width == 8 || width == 16) && mode == 6 {
        return predict_signed_top_edge_directional_signed(&filtered_top, width, out)
            .map_err(prediction_error);
    }
    if (width == 8 || width == 16) && mode == 7 {
        return predict_signed_left_edge_directional(&filtered_left, &filtered_top, width, out)
            .map_err(prediction_error);
    }
    if (width == 8 || width == 16) && mode == 8 {
        return predict_signed_side_directional(&filtered_left, width, out)
            .map_err(prediction_error);
    }
    if (9..=16).contains(&mode) {
        let direction = prediction_direction_for_mode(mode)
            .ok_or_else(|| invalid("unsupported signed prediction mode"))?;
        let (l, t) = if needs_filter {
            (&filtered_left, &filtered_top)
        } else {
            (left, top)
        };
        return predict_signed_angular(l, t, width, direction, out).map_err(prediction_error);
    }
    Err(DecodeError::Syntax(SyntaxError::Invalid {
        stream: "colour",
        detail: "unsupported signed prediction mode",
    }))
}

fn coefficient_error(error: crate::entropy::EntropyError) -> DecodeError {
    DecodeError::Coefficients(CoefficientError::Syntax(Box::new(error)))
}
