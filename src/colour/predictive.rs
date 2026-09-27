//! Reconstruct colour method 3 from predictions and coded coefficients.
//!
//! A predictive block has three transformed components: L, U+256, and V+256. L combines colour channels; U and V are differences. They are not the final G, B, R planes. Each component occupies a 16 by 16 workspace. A subdivision bit selects one width-16 argument or four width-8 arguments; each width-8 argument can contain one width-8 slot or four width-4 slots.
//!
//! For each argument, entropy decoding returns prediction modes and coefficient arrays. [`reconstruct_residual`](fn.reconstruct_residual.html) scales and inverse-transforms those coefficients. [`construct_colour_refs`](fn.construct_colour_refs.html) obtains corner-inclusive top and left arrays from picture banks and earlier samples in the component workspace. [`predict`](fn.predict.html) supplies estimates, and [`reconstruct_component`](fn.reconstruct_component.html) adds residuals and clips to the component range. Constructor arrays persist across slots because some selectors leave an extension unchanged.
//!
//! After all components finish, [`decode_block`](fn.decode_block.html) removes the difference biases and writes byte planes. Local variables called `red`, `green`, and `blue` are stored in plane order G, B, R; their names must not change the interpretation of the output. The crate page states the exact conversion order and gives a predictive-slot example.
//!
//! Errors are not rolled back. Reference banks and mode histories can change before coefficient or prediction failure. The ordinary cursor is synchronised to the contextual endpoint only after all components reconstruct successfully; picture samples are committed after conversion.

use crate::{
    bits::{BitReader, ContextBitReader},
    colour::{BlockPosition, ColourMemoryBudget, ColourState, allocation, invalid, references},
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

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "pixel reconstruction keeps the low eight bits"
)]
/// Returns the low byte of a wrapping signed reconstruction value.
///
/// Predictive colour conversion deliberately keeps the original codec's narrowing behaviour instead of saturating the value.
fn low_byte(value: i16) -> u8 {
    value as u8
}

/// Describes one plane passed from block layout to coefficient decoding.
///
/// The request selects its history column, profile, subdivision, and record offset without duplicating the larger mutable decoder context.
struct PlaneRequest {
    /// The zero-based SPI plane number.
    plane: usize,
    /// The containing block column.
    block_x: usize,
    /// The codec payload's byte offset in the complete record.
    payload_offset: usize,
    /// Selects alternate dequantisation weights when true.
    alternate_profile: bool,
    /// Selects four 8 by 8 arguments instead of one 16 by 16 argument.
    subdivided: bool,
}

/// Decodes all contextual components for one predictive colour plane.
///
/// A subdivided plane carries four arguments with an 8-pixel maximum width; an undivided plane carries one 16-pixel argument. Each result is immediately reconstructed into its location in the 16 by 16 signed workspace.
///
/// Entropy errors include component and record-relative bit context, while invalid reconstruction dimensions or prediction references return typed decoder errors.
fn decode_predictive_plane(
    reader: &mut ContextBitReader<'_>,
    history: &mut CoefficientHistory,
    state: &ColourState,
    request: PlaneRequest,
    output: &mut [i16; 256],
) -> Result<(), DecodeError> {
    let count = if request.subdivided { 4 } else { 1 };
    let horizontal_offset = u16::try_from(request.block_x * 16)
        .map_err(|_| invalid("predictive block offset overflow"))?;
    for component in 0..count {
        let result = decode_component(
            reader,
            history,
            ComponentRequest {
                argument: u32::try_from(component).expect("component argument is at most four"),
                horizontal_offset,
                maximum_width: if request.subdivided { 8 } else { 16 },
                fixed_width: 0,
                colour_long_symbols: true,
                width_selector_already_read: !request.subdivided,
            },
            request.plane * 4 + component,
            request.payload_offset,
        )?;
        reconstruct_component(
            &result,
            request.alternate_profile,
            (
                &state.histories[request.plane + 3],
                &state.histories[request.plane],
            ),
            (request.plane, component, count),
            output,
        )?;
    }
    Ok(())
}

/// Decodes one transform-based predictive colour block.
///
/// The procedure lazily creates coefficient histories, updates picture-edge references, switches to the contextual accumulator, and decodes a luma-like plane plus two signed chroma planes. It then converts those signed values to the decoder's three byte planes with the codec's wrapping and clipping rules and commits them to slice and picture storage.
///
/// On success, the ordinary bit reader advances to the contextual endpoint. On earlier failure it can remain at entry while banks and mode history have changed. Invalid history, syntax, prediction, transform, allocation, and memory-limit conditions return [`DecodeError`].
pub(super) fn decode_block(
    bits: &mut BitReader<'_>,
    record: &Record<'_>,
    state: &mut ColourState,
    output: &mut [Vec<u8>; 3],
    position: BlockPosition,
    histories: &mut Option<[CoefficientHistory; 3]>,
    budget: ColourMemoryBudget,
) -> Result<(), DecodeError> {
    let BlockPosition {
        geometry,
        x: block_x,
        local_y,
        global_y,
    } = position;
    ensure_histories(
        histories,
        state.block_width,
        geometry.block_rows,
        local_y,
        output,
        budget,
    )?;
    let histories = histories.as_mut().expect("histories initialised");
    references::update(state, block_x, local_y, global_y)?;

    let mut reader = ContextBitReader::new(record.payload);
    if !reader.initialise_at_bit(bits.bit_position()) {
        return Err(invalid("invalid predictive reader position"));
    }
    let initialised = reader
        .read_value(1)
        .ok_or_else(|| invalid("truncated predictive state"))?;
    let mut signed = [[0_i16; 256]; 3];
    for plane in 0..3 {
        let subdivided = reader.read_value(1).ok_or_else(|| {
            invalid(if plane == 0 {
                "truncated luma layout"
            } else {
                "truncated chroma layout"
            })
        })? != 0;
        decode_predictive_plane(
            &mut reader,
            &mut histories[plane],
            state,
            PlaneRequest {
                plane,
                block_x,
                payload_offset: record.payload_offset,
                alternate_profile: initialised == 0,
                subdivided,
            },
            &mut signed[plane],
        )?;
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
    Ok(())
}

/// Lazily allocates the three colour coefficient histories.
///
/// History stride is four columns per picture block plus one boundary column. The prospective value and accounted neutral-flag storage is checked against the memory budget before allocation, actual capacities are checked afterwards, and skipped block rows are initialised when the first predictive block occurs below row zero.
fn ensure_histories(
    histories: &mut Option<[CoefficientHistory; 3]>,
    block_width: usize,
    rows: usize,
    local_y: usize,
    output: &[Vec<u8>; 3],
    budget: ColourMemoryBudget,
) -> Result<(), DecodeError> {
    if histories.is_none() {
        let stride = block_width
            .checked_mul(4)
            .and_then(|v| v.checked_add(1))
            .and_then(|v| u16::try_from(v).ok())
            .ok_or_else(|| allocation("colour coefficient history"))?;
        let rows = u16::try_from(rows).map_err(|_| allocation("colour coefficient history"))?;
        let history_len = usize::from(stride)
            .checked_mul(1 + usize::from(rows) * 4 + 4)
            .ok_or_else(|| allocation("colour coefficient history"))?;
        let additional = history_len
            .checked_mul(3)
            .and_then(|bytes| {
                history_len
                    .div_ceil(8)
                    .checked_mul(3)
                    .and_then(|neutral| bytes.checked_add(neutral))
            })
            .ok_or_else(|| allocation("colour coefficient history"))?;
        budget.check_with_additional(output, histories, additional)?;
        let make = || CoefficientHistory::new(stride, rows).map_err(coefficient_error);
        *histories = Some([make()?, make()?, make()?]);
        budget.check(output, histories)?;
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

/// Decodes one coefficient component and adds positional error context.
///
/// The context identifies the plane argument and translates the contextual reader position from payload-relative to complete-record bits.
fn decode_component(
    reader: &mut ContextBitReader<'_>,
    history: &mut CoefficientHistory,
    request: ComponentRequest,
    component_index: usize,
    payload_offset: usize,
) -> Result<ComponentResult, DecodeError> {
    let result = decode_contextual_component(reader, history, request);
    result.map_err(|error| {
        coefficient_error(error).at(crate::ErrorContext {
            component: Some(component_index),
            bit_position: Some(payload_offset * 8 + reader.logical_bit()),
            ..crate::ErrorContext::default()
        })
    })
}

/// Write packed colour residual slots from decoded coefficients.
///
/// `component.width` is 4,8,16, and `component.count` describes one or four slots. The caller passes a zeroed 256-value workspace. A small slot occupies `slot*width*width..(slot+1)*width*width`; this packed layout is distinct from its location in the 16 by 16 component plane.
///
/// Width 16 uses the matrix transform for any active callback, with scale 144 normally or 160 for alternate profile. Widths 4/8 map inactive, constant, and full callbacks to selectors 0,1,7 of [`reconstruct_component_residual`]. Inactive regions remain unchanged, so zero initialisation is required for zero residuals.
///
/// # Errors
///
/// Transform validation failures become coefficient reconstruction errors. Checks within each transform precede writes, but callers must not assume that an enclosing multi-slot operation rolls back earlier work.
fn reconstruct_residual(
    component: &ComponentResult,
    alternate_profile: bool,
    residual: &mut [i16; 256],
) -> Result<(), DecodeError> {
    let width = usize::from(component.width);
    if width == 16 {
        if component.callbacks[0] != CallbackKind::Inactive {
            inverse_transform_16x16(
                &component.values[0],
                if alternate_profile { 160 } else { 144 },
                residual,
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
            residual,
        )
        .map_err(|e| DecodeError::Coefficients(CoefficientError::Transform(Box::new(e))))?;
    }
    Ok(())
}

/// Reconstruct one argument into a 16 by 16 transformed component workspace.
///
/// `position` is `(plane, argument, component_count)`. Plane 0 is L; planes 1/2 are U+256 and V+256. An undivided component has one argument at (0,0). Four-argument components place argument `a` at `((a&1)*8,(a>>1)*8)`. Width-4 slots add `((slot&1)*4,(slot>>1)*4)` to that origin; width-8 and width-16 slots add zero.
///
/// First reconstruct packed residual slots. For each slot, construct corner-inclusive edges, predict a square, and add its packed residual in `i32`. Clip to 0..255 for plane 0 or 0..511 for planes 1/2. Write each sample to `output[(origin_y+y)*16+origin_x+x]`; other workspace entries retain their values for later reference construction. Constructor arrays start at zero per argument and persist across its slots.
///
/// # Errors
///
/// Invalid component count or dimensions, missing constructor edges, or prediction/transform failures return [`DecodeError`]. Earlier slots can already have modified the workspace. Callers must provide the production argument/count geometry described above; this helper is not a general bounds-checked layout API.
fn reconstruct_component(
    component: &ComponentResult,
    alternate_profile: bool,
    banks: (&[i16; 33], &[i16; 33]),
    position: (usize, usize, usize),
    output: &mut [i16; 256],
) -> Result<(), DecodeError> {
    let (left, top) = banks;
    let (plane, argument, component_count) = position;
    if component.count == 0
        || component.count > 4
        || component.width as usize * component.width as usize > 256
    {
        return Err(invalid("invalid predictive component dimensions"));
    }
    let width = component.width as usize;
    let mut residual = [0_i16; 256];
    reconstruct_residual(component, alternate_profile, &mut residual)?;
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
            (left, top),
            output,
            (slot_x, slot_y),
            (&mut slot_left, &mut slot_top),
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

/// Prepare corner-inclusive signed references for a width-8 slot.
///
/// `position` is the sample origin inside the row-major 16 by 16 workspace. The caller supplies the quadrant corresponding to selector 0 through 3. Ranges below exclude their upper bound. A bank window includes a corner; internal edge samples start at output index one.
///
/// | Selector | Left edge | Top edge | Corner |
/// | --- | --- | --- | --- |
/// | 0 | Copy bank 0..17 | Copy bank 0..17 | Preserve each bank's corner independently |
/// | 1 | Read 8 internal samples; repeat index 8 through 9..17 | Copy top bank 8..25 | Top window's first value supplies both |
/// | 2 | Copy left bank 8..25 | Read 16 internal top samples | Left window's first value supplies both |
/// | 3 | Read 8 internal samples; repeat through 9..17 | Read 8 internal samples; repeat through 9..17 | Workspace sample at origin-17 supplies both |
///
/// Only indices 0..17 are written. Larger entries retain their previous values. This is eight immediate neighbours and eight extension neighbours, plus the corner.
///
/// # Errors
///
/// Invalid selector or a missing required internal origin returns invalid colour syntax. Callers must bound the workspace origin and all referenced samples; this private helper relies on the production quadrant geometry.
fn construct_colour_refs_eight(
    selector: usize,
    banks: (&[i16; 33], &[i16; 33]),
    workspace: &[i16; 256],
    position: (usize, usize),
    edges: (&mut [i16; 33], &mut [i16; 33]),
) -> Result<(), DecodeError> {
    let (left_bank, top_bank) = banks;
    let (component_x, component_y) = position;
    let (left, top) = edges;
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
    Ok(())
}

/// Reads already reconstructed internal edges from a 16 by 16 workspace.
///
/// The reader records the component origin so width-four reference selectors can obtain left and top samples without repeating index calculations.
struct ColourEdgeReader<'data> {
    /// Borrows the partially reconstructed signed component.
    workspace: &'data [i16; 256],
    /// The component's linear origin in `workspace`.
    origin: usize,
    /// The component's horizontal offset.
    component_x: usize,
    /// The component's vertical offset.
    component_y: usize,
}

impl ColourEdgeReader<'_> {
    /// Copies an internal left edge into a reference array.
    ///
    /// Returns `false` when the component has no reconstructed pixels to its left. When `extend` is set, the final copied sample fills the remaining width-four reference extent.
    fn vertical(&self, count: usize, extend: bool, out: &mut [i16; 33]) -> bool {
        if self.component_x == 0 {
            return false;
        }
        for i in 0..count {
            out[i + 1] = self.workspace[self.origin - 1 + i * 16];
        }
        if extend {
            let edge = out[count];
            out[count + 1..9].fill(edge);
        }
        true
    }

    /// Copies an internal top edge into a reference array.
    ///
    /// Returns `false` when the component has no reconstructed pixels above it. When `extend` is set, the final copied sample fills the remaining width-four reference extent.
    fn above(&self, count: usize, extend: bool, out: &mut [i16; 33]) -> bool {
        if self.component_y == 0 {
            return false;
        }
        for i in 0..count {
            out[i + 1] = self.workspace[self.origin - 16 + i];
        }
        if extend {
            let edge = out[count];
            out[count + 1..9].fill(edge);
        }
        true
    }
}

/// Construct a slot's corner-inclusive signed edges.
///
/// `workspace` is a partially reconstructed 16 by 16 component in row order. `position` gives the slot's sample origin inside it. `banks` are the containing block's left and top edges; `edges` are caller-owned arrays retained across slots of one argument. Width 16 copies complete banks. Width 8 uses selectors 0..3; width 4 uses selectors 0..15 in argument-major order (`argument*4+slot`). The following table describes width-4 writes; unspecified extension entries retain their previous values.
///
/// | Selector | Left preparation | Top preparation |
/// | --- | --- | --- |
/// | 0 | Bank indices 0..9 | Bank indices 0..9 |
/// | 1 | 4 internal samples, repeat last through index 8 | Bank window 4..13 |
/// | 2 | Bank window 4..13 | 8 internal samples |
/// | 3 | 4 internal samples, repeat last | 4 internal samples; retain indices 5..8 |
/// | 4 | 8 internal samples | Bank window 8..17 |
/// | 5 | 4 internal samples, repeat last | Bank window 12..21 |
/// | 6,9,14 | 4 internal samples, repeat last | 8 internal samples |
/// | 7,11,13,15 | 4 internal samples, repeat last | 4 internal samples, repeat last |
/// | 8 | Bank window 8..17 | 8 internal samples |
/// | 10 | Bank window 12..21 | 8 internal samples |
/// | 12 | 8 internal samples | 8 internal samples |
///
/// Ranges in this table are Rust half-open ranges. Copied windows start at output index zero. Internal samples start at index one. When one edge uses a bank window and the other is internal, that window supplies both corners. Selector 0 and width 16 preserve the left and top bank corners separately; they can differ. When both edges are internal, the sample at `origin-17` supplies the corner. Width-8 selectors use the same quadrant principle with 8 immediate samples and an extension through index 16; see [`construct_colour_refs_eight`].
///
/// # Errors
///
/// Invalid width/selector or a missing required left/top origin returns invalid colour syntax. An error can follow a partial edge write. The caller must provide valid workspace coordinates for the selector; colour's internal edge reads rely on its argument geometry.
fn construct_colour_refs(
    width: usize,
    selector: usize,
    banks: (&[i16; 33], &[i16; 33]),
    workspace: &[i16; 256],
    position: (usize, usize),
    edges: (&mut [i16; 33], &mut [i16; 33]),
) -> Result<(), DecodeError> {
    let (left_bank, top_bank) = banks;
    let (component_x, component_y) = position;
    let (left, top) = edges;
    if width == 16 {
        *left = *left_bank;
        *top = *top_bank;
        return Ok(());
    }
    if width == 8 {
        return construct_colour_refs_eight(
            selector,
            (left_bank, top_bank),
            workspace,
            (component_x, component_y),
            (left, top),
        );
    }
    if width != 4 || selector > 15 {
        return Err(invalid("invalid width-four colour reference selector"));
    }
    let origin = component_y * 16 + component_x;
    let edge_reader = ColourEdgeReader {
        workspace,
        origin,
        component_x,
        component_y,
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
            edge_reader.vertical(4, true, left)
        }
        2 => {
            left[..9].copy_from_slice(&left_bank[4..13]);
            top[0] = left[0];
            edge_reader.above(8, false, top)
        }
        3 if component_x != 0 && component_y != 0 => {
            left[0] = workspace[origin - 17];
            top[0] = left[0];
            edge_reader.vertical(4, true, left) && edge_reader.above(4, false, top)
        }
        4 => {
            top[..9].copy_from_slice(&top_bank[8..17]);
            left[0] = top[0];
            edge_reader.vertical(8, false, left)
        }
        5 => {
            top[..9].copy_from_slice(&top_bank[12..21]);
            left[0] = top[0];
            edge_reader.vertical(4, true, left)
        }
        6 | 9 | 14 if component_x != 0 && component_y != 0 => {
            left[0] = workspace[origin - 17];
            top[0] = left[0];
            edge_reader.vertical(4, true, left) && edge_reader.above(8, false, top)
        }
        7 | 11 | 13 | 15 if component_x != 0 && component_y != 0 => {
            left[0] = workspace[origin - 17];
            top[0] = left[0];
            edge_reader.vertical(4, true, left) && edge_reader.above(4, true, top)
        }
        8 => {
            left[..9].copy_from_slice(&left_bank[8..17]);
            top[0] = left[0];
            edge_reader.above(8, false, top)
        }
        10 => {
            left[..9].copy_from_slice(&left_bank[12..21]);
            top[0] = left[0];
            edge_reader.above(8, false, top)
        }
        12 if component_x != 0 && component_y != 0 => {
            left[0] = workspace[origin - 17];
            top[0] = left[0];
            edge_reader.vertical(8, false, left) && edge_reader.above(8, false, top)
        }
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(invalid("colour reference is unavailable for constructor"))
    }
}

/// Fills a block by repeating its top or left reference edge.
///
/// Mode 0 repeats the top edge down every row; mode 1 repeats the left edge across every column. The caller passes a valid width no greater than 16.
fn predict_edge(mode: u8, width: usize, left: &[i16; 33], top: &[i16; 33], out: &mut [i16; 256]) {
    for row in 0..width {
        for col in 0..width {
            out[row * width + col] = if mode == 0 {
                top[col + 1]
            } else {
                left[row + 1]
            };
        }
    }
}

/// Predict one colour slot from prepared corner-inclusive signed edges.
///
/// The caller supplies width 4,8,16 and component-range values (0..255 or 0..511). Modes 0/1 copy top/left, 2 uses DC, and 17 uses corner-excluding planar references. The module and crate pages define the coordinate system and mode families.
///
/// | Width | Modes whose input edges are filtered here |
/// | --- | --- |
/// | 4 | 3,6,9 |
/// | 8 | 3..8 |
/// | 16 | 3..16 |
///
/// Mode 8 therefore passes raw `left` to `predict_signed_side_directional` at width 4, despite that helper's parameter name. Filtering is this dispatcher's responsibility. Dedicated width-4 diagonal and angular paths retain their own endpoint rules. Modes 9..16 use a direction-table index, not the mode itself.
///
/// Writes the first `width*width` output entries in row order. Later entries and inputs are unchanged. The caller has already validated component dimensions; the direct-copy branches rely on that precondition.
///
/// # Errors
///
/// Unsupported mode combinations produce colour syntax errors. Checked predictor failures become [`DecodeError::Prediction`]. Prediction checks precede their writes.
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
    if mode <= 1 {
        predict_edge(mode, width, left, top, out);
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

/// Converts an entropy-history failure into a colour coefficient error.
///
/// Allocation failure remains classified as allocation; other entropy failures become coefficient syntax.
fn coefficient_error(error: crate::entropy::EntropyError) -> DecodeError {
    crate::error::coefficient_history_error(error, "colour coefficient history")
}
