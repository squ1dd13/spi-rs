//! Decode runs, palette indices, component layouts, and signed coefficient values.
//!
//! Entropy syntax represents values with variable-length bit sequences. A symbol is a decoded
//! choice, such as a mode or a magnitude with a position change. A coding context is stored
//! information used to interpret that choice. This implementation uses a 32-bit contextual
//! accumulator and neighbouring mode history; “contextual” does not mean a standard arithmetic
//! coder.
//!
//! A component is one value per pixel. An argument identifies a component region inside a 16 by 16
//! block. A slot is a square region with its own prediction mode and coefficient array.
//! [`ComponentRequest`] selects argument geometry and stream rules. [`ComponentResult`] returns
//! slot width, layout count, modes, coefficient activity, scan variants, and row-major values. A
//! layout slot can have no coefficients and still require prediction.
//!
//! # Decoding order
//!
//! [`decode_contextual_component`] first selects width and slot count, then reads mode symbols
//! with [`CoefficientHistory`]. It next reads an activity-table code. For each active slot it
//! reads a coefficient count and that many signed symbols. The result classifies no coefficients
//! as inactive, a single coefficient at scan position zero as constant-only, and other
//! arrangements as a full block. Colour uses these classes to select transforms. Alpha uses the
//! returned values directly as residuals; the class only distinguishes active from inactive there.
//!
//! The usual width bit gives maximum width with one slot for zero, or half width with four slots
//! for one. Colour undivided blocks have already read their layout selector and obtain width 16
//! without the special `fixed_width == 16` branch. That special branch skips seven bits, forces
//! mode 4 and diagonal scan, and has no production caller in this checkout.
//!
//! # Prefixes, symbols, and scans
//!
//! An unsigned contextual index has `z` leading zeroes, a terminating one, and `z` suffix bits.
//! Its value is `2^z - 1 + suffix`. The reader routines also preserve the accumulator's
//! replacement-refill semantics; they must not read beyond bounded lookahead. Coefficient counts
//! are one plus this index.
//!
//! For a coefficient, prefix width `z` is followed by `z+1` payload bits. The low payload bit is
//! the sign (one means negative); `code = 2^z + (payload >> 1)`. For `z <= 6`, [`SHORT_SYMBOLS`]
//! at `code-1` supplies magnitude and scan delta. For a long colour code, `p=code-1`, delta is `p
//! & 255`, and magnitude is `(p >> 8)-1`. For long alpha codes, `p=code-128`; the low 4, 6, or 8
//! bits give delta for widths 4, 8, or 16, and the remaining value plus one gives magnitude.
//! Magnitudes above 32767 are rejected.
//!
//! A scan is the order in which coded values visit a square. The scan cursor starts at zero. Add
//! the delta, map that cursor to a row-major array index, write the signed value, then increment
//! the cursor. Colour always uses diagonal scan. Alpha uses the mode mapping
//! `[1,2,0,0,1,1,0,2,2,0,0,1,1,0,0,2,2]` for modes 0..16, where 0 is diagonal, 1 row order, and 2
//! column order; mode 17 uses diagonal scan. The first width-4 diagonal positions are 0,4,1,2,5,8.
//! Thus scan order and array layout are separate concepts.
//!
//! For example, coefficient bits `0001 0110` have prefix width 3 and payload 6. They select
//! short-table entry 10, magnitude 5, delta 0, positive sign. Starting at cursor zero, this writes
//! value 5 at position zero and advances the cursor to one. This calculation is not a complete
//! encoded component.
//!
//! # Mode history and partial effects
//!
//! [`CoefficientHistory`] stores mode symbols in four-pixel cells, plus neutral-state flags. It
//! stores no sample values. Its stride is four cells per picture block plus one boundary cell.
//! Initially all symbols are 2 and neutral flags are true. Row transitions retain a delayed row
//! and reset the active four rows. Colour has a first-exposure normalisation rule; alpha copies
//! the last active row directly. Histories are created anew for each record.
//!
//! A leading mode bit of one selects the left or delayed upper candidate; a further bit selects
//! the delayed candidate when they differ. A leading zero selects explicit syntax. Prefix `01`
//! then selects mode 2 or symbol 17 with one bit; prefix `00` reads four bits and skips values
//! reserved for the predicted choices. Symbol 17 aliases mode 2 only during neighbour selection;
//! its final prediction mode is planar 17.
//!
//! History is updated during layout decoding, before coefficient parsing can fail. Contextual
//! reads and output writes can also have partial effects. Callers must discard a failed operation,
//! not retry as if the reader and history were unchanged. The crate page explains byte runs,
//! palette runs, and record-relative bit accounting.

use thiserror::Error;

use crate::bits::ContextBitReader;

#[expect(
    clippy::cast_possible_truncation,
    reason = "only the low eight bits form the decoded byte"
)]
/// Returns the low byte of a decoded entropy value.
///
/// Relevant syntax fields use an eight-bit register and intentionally discard higher bits.
fn low_byte(value: u32) -> u8 {
    value as u8
}

/// Malformed entropy syntax, invalid history, or output overflow.
///
/// Allocation failure is kept distinct so the public decoder can preserve its resource category.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum EntropyError {
    /// A truncated, impossible, or out-of-table entropy code.
    #[error("entropy code is truncated or invalid")]
    InvalidCode,
    /// Decoded symbols exceed the caller's declared output capacity.
    #[error("entropy output exceeds its declared capacity")]
    OutputOverflow,
    /// A coefficient-history address or layout is invalid.
    #[error("context history is too small for this component")]
    InvalidHistory,
    /// A coefficient scan index outside its transform block.
    #[error("coefficient scan position is outside the block")]
    InvalidScan,
    /// Coefficient-history storage could not be reserved.
    #[error("could not allocate contextual coefficient history")]
    AllocationFailure,
}

/// Reads the zero count of a unary prefix from the ordinary bit reader.
///
/// A terminating one must occur before `max_zeros`. Truncation or a missing terminator returns
/// [`EntropyError::InvalidCode`].
pub fn read_unary_prefix(
    bits: &mut crate::bits::BitReader<'_>,
    max_zeros: u32,
) -> Result<u32, EntropyError> {
    for zeros in 0..max_zeros {
        match bits.read_bit() {
            Ok(1) => return Ok(zeros),
            Ok(_) => {}
            Err(_) => return Err(EntropyError::InvalidCode),
        }
    }
    Err(EntropyError::InvalidCode)
}

/// Decodes repeated-byte RLE until at least `output_size` bytes exist.
///
/// A run is present when the preceding two literals are equal. Run and literal values use
/// unary-prefixed codes, with an escape to a raw byte for large literal prefixes. The stream's run
/// counter wraps to eight bits and total logical production is limited to 512 bytes.
///
/// The returned logical count includes all repeated values and the literal consumed after the run,
/// even if they pass `output_size`. Writes stop at `output_size`; extra scratch capacity is not
/// filled with excess logical samples. A literal is still consumed after a run reaches the
/// requested size. For requested size 3, two literal 5s, a run of one 5, and a final literal 7
/// give output `[5,5,5]` and logical count 4.
///
/// A short output is rejected before reading. Invalid syntax or logical production above 512 can
/// fail after earlier cursor advances and writes; no rollback occurs. A zero requested size
/// returns zero without reading or writing.
pub fn decode_rle_plane(
    bits: &mut crate::bits::BitReader<'_>,
    output: &mut [u8],
    output_size: usize,
) -> Result<usize, EntropyError> {
    if output_size > output.len() {
        return Err(EntropyError::OutputOverflow);
    }
    let (mut previous_previous, mut previous) = (0xff_u8, 0_u8);
    let mut produced = 0_usize;
    while produced < output_size {
        if previous == previous_previous {
            let prefix = read_unary_prefix(bits, 31)?;
            let suffix = if prefix == 0 {
                0
            } else {
                bits.read_bits(prefix)
                    .map_err(|_| EntropyError::InvalidCode)?
            };
            // The stream stores this run count in an 8-bit register.
            let run = usize::from(low_byte(((1_u32 << prefix) - 1) + suffix));
            let end = produced
                .checked_add(run)
                .ok_or(EntropyError::OutputOverflow)?;
            if end > 512 {
                return Err(EntropyError::OutputOverflow);
            }
            if produced < output_size {
                output[produced..end.min(output_size)].fill(previous_previous);
            }
            produced = end;
        }
        let prefix = read_unary_prefix(bits, 31)?;
        let remainder = bits.read_bits(2).map_err(|_| EntropyError::InvalidCode)?;
        let literal = if prefix < 21 {
            u8::try_from(prefix * 4 + remainder).expect("literal prefix is below 21")
        } else if prefix == 21 && remainder <= 2 {
            u8::try_from(remainder | 0x54).expect("literal is at most 0x56")
        } else {
            bits.read_byte().map_err(|_| EntropyError::InvalidCode)?
        };
        if produced < output_size {
            output[produced] = literal;
        }
        produced = produced
            .checked_add(1)
            .ok_or(EntropyError::OutputOverflow)?;
        if produced > 512 {
            return Err(EntropyError::OutputOverflow);
        }
        previous_previous = previous;
        previous = literal;
    }
    Ok(produced)
}

/// Shifts a contextual accumulator left with wrapping semantics.
///
/// A shift of 32 or more produces the target's wrapping-shift result, matching the codec reader
/// operations.
fn lz(value: u32, shift: u32) -> u32 {
    value.wrapping_shl(shift)
}

/// Counts leading zero bits in one lookahead byte.
///
/// This wrapper keeps prefix-scanning code expressed in the format's byte-oriented stages.
fn leading_zeroes(value: u8) -> u32 {
    value.leading_zeros()
}

/// Decodes the width of a contextual Exp-Golomb prefix.
///
/// The procedure reproduces SPI's accumulator alignment and lookahead refill behaviour, including
/// a bounded peek into the following byte. It consumes the zero prefix and terminating one, but
/// not the suffix.
///
/// Returns [`EntropyError::InvalidCode`] when refill fails or the derived width is outside 0
/// through 31.
pub fn read_context_prefix(reader: &mut ContextBitReader<'_>) -> Result<u32, EntropyError> {
    let old_available = reader.available;
    let lookahead = 24 | (old_available & 7);
    let mut available_for_lookahead = old_available;
    if old_available < lookahead {
        let missing = 32 - old_available;
        if missing >= 8 && reader.byte_offset < reader.size {
            let old_accumulator = reader.accumulator;
            let old_available_for_merge = reader.available;
            if !reader.refill(missing >> 3) {
                return Err(EntropyError::InvalidCode);
            }
            reader.accumulator = (reader.accumulator >> old_available_for_merge) | old_accumulator;
            reader.available += old_available_for_merge;
            available_for_lookahead = reader.available;
        }
    }

    let mut candidate = reader.accumulator >> (32 - lookahead);
    let next_byte_shift = i32::try_from(lookahead).expect("lookahead is at most 31")
        - i32::try_from(available_for_lookahead).expect("available bits are at most 32");
    if (1..=8).contains(&next_byte_shift)
        && let Some(next_byte) = reader.peek_next_byte()
    {
        candidate |= u32::from(next_byte)
            >> (8 - u32::try_from(next_byte_shift).expect("shift is positive"));
    }
    let candidate_bytes = candidate.to_be_bytes();
    let mut zeroes = leading_zeroes(candidate_bytes[0]);
    if zeroes == 8 {
        zeroes = leading_zeroes(candidate_bytes[1]) + 8;
        if zeroes == 16 {
            zeroes = leading_zeroes(candidate_bytes[2]) + 16;
            if zeroes == 24 {
                zeroes = leading_zeroes(candidate_bytes[3]) + 24;
            }
        }
    }
    let aligned = (old_available | 0xffff_fff8).cast_signed();
    let signed_width = aligned + i32::try_from(zeroes).expect("prefix has at most 32 zeroes");
    if !(0..=31).contains(&signed_width) {
        return Err(EntropyError::InvalidCode);
    }
    let width = u32::try_from(signed_width).expect("checked nonnegative width");
    let mut bits_for_code = width + 1;
    if available_for_lookahead <= width {
        if !reader.refill(4) {
            return Err(EntropyError::InvalidCode);
        }
        bits_for_code -= available_for_lookahead;
    }
    if !reader.read_bits(bits_for_code) {
        return Err(EntropyError::InvalidCode);
    }
    Ok(width)
}

/// Decodes one contextual Exp-Golomb index.
///
/// The returned tuple is `(width, suffix, index)`, where `index` is `(2^width - 1) + suffix`.
/// Suffix fragments can cross a replacement refill and are merged explicitly.
///
/// Truncated prefix or suffix data returns [`EntropyError::InvalidCode`].
pub fn read_context_index(
    reader: &mut ContextBitReader<'_>,
) -> Result<(u32, u32, u32), EntropyError> {
    let width = read_context_prefix(reader)?;
    let suffix = if width == 0 {
        0
    } else if width > reader.available {
        let prefix = reader.accumulator >> (32 - width);
        let old_available = reader.available;
        if !reader.refill(4) {
            return Err(EntropyError::InvalidCode);
        }
        let extra = width - old_available;
        if reader.available < extra {
            return Err(EntropyError::InvalidCode);
        }
        let suffix = (reader.accumulator >> (32 - extra)) | prefix;
        reader.accumulator = lz(reader.accumulator, extra);
        reader.available -= extra;
        suffix
    } else {
        let suffix = reader.accumulator >> (32 - width);
        if !reader.read_bits(width) {
            return Err(EntropyError::InvalidCode);
        }
        suffix
    };
    let index = ((1_u32 << width) - 1) + suffix;
    Ok((width, suffix, index))
}

/// Decodes one fixed-width literal used by contextual palette indices.
///
/// Literal bits can straddle the current accumulator and a replacement refill. The procedure
/// merges both fragments and leaves subsequent bits left-aligned.
fn read_contextual_literal(
    reader: &mut ContextBitReader<'_>,
    prefix_bits: u32,
) -> Result<u32, EntropyError> {
    let old_accumulator = reader.accumulator;
    let old_available = reader.available;
    let mut bits_for_literal = prefix_bits.wrapping_sub(old_available);
    let literal = if bits_for_literal.cast_signed() <= 0 {
        bits_for_literal = prefix_bits;
        reader.accumulator = lz(old_accumulator, bits_for_literal);
        reader.available = old_available - bits_for_literal;
        old_accumulator >> (32 - bits_for_literal)
    } else {
        if !reader.refill(4) {
            return Err(EntropyError::InvalidCode);
        }
        if reader.available < bits_for_literal {
            return Err(EntropyError::InvalidCode);
        }
        let prefix_fragment = old_accumulator >> (32 - prefix_bits);
        let suffix_shift = 32 - bits_for_literal;
        let new_accumulator = reader.accumulator;
        reader.accumulator = lz(new_accumulator, bits_for_literal);
        reader.available -= bits_for_literal;
        (new_accumulator >> suffix_shift) | prefix_fragment
    };
    Ok(literal)
}

/// Decodes the run-control code after a contextual palette literal.
///
/// The codec uses prefix lookahead similar to [`read_context_prefix`] but returns the run-control
/// value, code width, and suffix separately. Accumulator fragments are merged across refills as
/// required.
///
/// Invalid widths or incomplete refills return [`EntropyError::InvalidCode`].
fn read_contextual_run_control(
    reader: &mut ContextBitReader<'_>,
) -> Result<(u32, u32, u32), EntropyError> {
    let lookahead_bits = 0x18 | (reader.available & 7);
    let pre_lookahead_available = reader.available;
    let mut available_after_lookahead = pre_lookahead_available;
    if pre_lookahead_available < lookahead_bits {
        let missing_bits = 32 - pre_lookahead_available;
        if missing_bits >= 8 {
            let refill_bytes = missing_bits >> 3;
            let refill_accumulator = reader.accumulator;
            let refill_available = reader.available;
            if !reader.refill(refill_bytes) {
                return Err(EntropyError::InvalidCode);
            }
            reader.accumulator = (reader.accumulator >> refill_available) | refill_accumulator;
            reader.available += refill_available;
            available_after_lookahead = reader.available;
        }
    }
    let mut candidate = reader.accumulator >> (32 - lookahead_bits);
    let next_shift = i32::try_from(lookahead_bits).expect("lookahead is at most 31")
        - i32::try_from(available_after_lookahead).expect("available bits are at most 32");
    if (1..=8).contains(&next_shift)
        && let Some(next) = reader.peek_next_byte()
    {
        candidate |= u32::from(next) >> (8 - u32::try_from(next_shift).expect("shift is positive"));
    }
    let candidate_bytes = candidate.to_be_bytes();
    let mut code_width = leading_zeroes(candidate_bytes[0]);
    if code_width == 8 {
        code_width = leading_zeroes(candidate_bytes[1]) + 8;
        if code_width == 16 {
            code_width = leading_zeroes(candidate_bytes[2]) + 16;
            if code_width == 24 {
                code_width = leading_zeroes(candidate_bytes[3]) + 24;
            }
        }
    }
    let aligned_available = pre_lookahead_available | 0xffff_fff8;
    let code_width = aligned_available.wrapping_add(code_width);
    if code_width >= 32 {
        return Err(EntropyError::InvalidCode);
    }
    let mut bits_for_code = code_width.wrapping_add(1);
    if available_after_lookahead
        .wrapping_sub(code_width)
        .cast_signed()
        <= 0
    {
        if !reader.refill(4) {
            return Err(EntropyError::InvalidCode);
        }
        bits_for_code = bits_for_code.wrapping_sub(available_after_lookahead);
    }
    if reader.available < bits_for_code {
        return Err(EntropyError::InvalidCode);
    }
    if bits_for_code == 32 {
        reader.accumulator = 0;
        reader.available = 0;
    } else {
        reader.accumulator = lz(reader.accumulator, bits_for_code);
        reader.available -= bits_for_code;
    }
    let mut code_value = 0_u32;
    if code_width != 0 {
        let current_accumulator = reader.accumulator;
        let current_available = reader.available;
        let refill = i32::try_from(code_width).expect("code width is at most 32")
            - i32::try_from(current_available).expect("available bits are at most 32");
        if refill > 0 {
            if !reader.refill(4) {
                return Err(EntropyError::InvalidCode);
            }
            let prefix_value = current_accumulator >> (32 - code_width);
            let new_accumulator = reader.accumulator;
            let extra = u32::try_from(refill).expect("refill is positive");
            let new_available = reader.available;
            if new_available < extra {
                return Err(EntropyError::InvalidCode);
            }
            reader.accumulator = lz(new_accumulator, extra);
            reader.available = new_available - extra;
            code_value = (new_accumulator >> (32 - extra)) | prefix_value;
        } else {
            let code_accumulator = reader.accumulator;
            reader.accumulator = lz(code_accumulator, code_width);
            reader.available -= code_width;
            code_value = code_accumulator >> (32 - code_width);
        }
    }
    let mask = if code_width == 0 {
        0
    } else {
        (1_u32 << code_width) - 1
    };
    let run_control = code_value.wrapping_add(mask);
    Ok((run_control, code_width, code_value))
}

/// Decodes 256 contextual palette indices.
///
/// Each literal is followed by a run-control code that can repeat it. `initial_prefix_bits` gives
/// the fixed literal width and must be from 1 through 31. The destination can be larger than 256
/// but only the first 256 entries carry syntax output.
///
/// Returns [`EntropyError`] for invalid prefix width, a short destination, malformed run syntax,
/// or a run beyond capacity.
pub fn decode_contextual_indices(
    reader: &mut ContextBitReader<'_>,
    initial_prefix_bits: u32,
    indices: &mut [u8],
) -> Result<(), EntropyError> {
    if indices.len() < 256 || initial_prefix_bits == 0 || initial_prefix_bits > 31 {
        return Err(EntropyError::InvalidCode);
    }
    indices.fill(0);
    let mut cursor = 0_u32;
    while cursor < 256 {
        let literal = read_contextual_literal(reader, initial_prefix_bits)?;
        indices[cursor as usize] = low_byte(literal);

        let (run_control, code_width, code_value) = read_contextual_run_control(reader)?;
        if run_control.wrapping_add(cursor) > 255 {
            return Err(EntropyError::InvalidCode);
        }
        if run_control >= 1 {
            let advance = (1_u32 << code_width).wrapping_add(code_value);
            if usize::try_from(cursor + advance).map_or(true, |end| end > indices.len()) {
                return Err(EntropyError::OutputOverflow);
            }
            indices[cursor as usize..(cursor + advance) as usize].fill(low_byte(literal));
            cursor += advance;
        } else {
            cursor += 1;
        }
    }
    Ok(())
}

/// Maps short coefficient codes to magnitude and scan-delta pairs.
const SHORT_SYMBOLS: [(u8, u8); 128] = [
    (1, 0),
    (1, 1),
    (2, 0),
    (1, 2),
    (3, 0),
    (1, 3),
    (4, 0),
    (1, 4),
    (2, 1),
    (1, 5),
    (5, 0),
    (1, 6),
    (6, 0),
    (1, 7),
    (3, 1),
    (2, 2),
    (1, 8),
    (7, 0),
    (1, 9),
    (8, 0),
    (1, 10),
    (9, 0),
    (4, 1),
    (1, 11),
    (2, 3),
    (1, 12),
    (10, 0),
    (1, 13),
    (3, 2),
    (11, 0),
    (1, 14),
    (5, 1),
    (2, 4),
    (12, 0),
    (1, 15),
    (6, 1),
    (13, 0),
    (2, 5),
    (1, 16),
    (14, 0),
    (1, 17),
    (4, 2),
    (3, 3),
    (7, 1),
    (15, 0),
    (1, 18),
    (2, 6),
    (1, 20),
    (1, 19),
    (16, 0),
    (1, 21),
    (8, 1),
    (17, 0),
    (1, 24),
    (2, 7),
    (1, 22),
    (5, 2),
    (1, 25),
    (3, 4),
    (18, 0),
    (9, 1),
    (1, 23),
    (2, 8),
    (19, 0),
    (4, 3),
    (10, 1),
    (1, 26),
    (20, 0),
    (6, 2),
    (2, 9),
    (21, 0),
    (3, 5),
    (11, 1),
    (1, 27),
    (22, 0),
    (2, 10),
    (7, 2),
    (1, 31),
    (12, 1),
    (1, 30),
    (23, 0),
    (1, 28),
    (1, 29),
    (5, 3),
    (4, 4),
    (1, 32),
    (2, 11),
    (24, 0),
    (13, 1),
    (8, 2),
    (3, 6),
    (25, 0),
    (2, 13),
    (2, 12),
    (14, 1),
    (26, 0),
    (1, 34),
    (1, 33),
    (6, 3),
    (15, 1),
    (27, 0),
    (4, 5),
    (1, 35),
    (2, 14),
    (9, 2),
    (5, 4),
    (28, 0),
    (3, 7),
    (16, 1),
    (29, 0),
    (10, 2),
    (17, 1),
    (30, 0),
    (7, 3),
    (3, 8),
    (31, 0),
    (18, 1),
    (1, 36),
    (11, 2),
    (32, 0),
    (1, 37),
    (3, 9),
    (33, 0),
    (2, 15),
    (19, 1),
    (6, 4),
    (34, 0),
    (1, 0),
];

/// Diagonal, row-major, and column-major scans for 4 by 4 coefficients.
const SCAN4: [[u16; 16]; 3] = [
    [0, 4, 1, 2, 5, 8, 12, 9, 6, 3, 7, 10, 13, 14, 11, 15],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15],
];
/// Diagonal, row-major, and column-major scans for 8 by 8 coefficients.
const SCAN8: [[u16; 64]; 3] = [
    [
        0, 8, 1, 2, 9, 16, 24, 17, 10, 3, 4, 11, 18, 25, 32, 40, 33, 26, 19, 12, 5, 6, 13, 20, 27,
        34, 41, 48, 56, 49, 42, 35, 28, 21, 14, 7, 15, 22, 29, 36, 43, 50, 57, 58, 51, 44, 37, 30,
        23, 31, 38, 45, 52, 59, 60, 53, 46, 39, 47, 54, 61, 62, 55, 63,
    ],
    [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    ],
    [
        0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50, 58,
        3, 11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53,
        61, 6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63,
    ],
];

/// Maps a coefficient scan cursor to a row-major transform position.
///
/// Variants 0, 1, and 2 are diagonal, row-major, and column-major. Widths 4 and 8 use fixed
/// tables; width 16 calculates its diagonal scan and directly calculates the linear variants.
///
/// Returns [`EntropyError::InvalidScan`] for an unsupported width or variant, or an index outside
/// the square block.
pub fn coefficient_scan_position(
    width: u8,
    variant: u8,
    scan_index: u32,
) -> Result<u16, EntropyError> {
    if variant >= 3
        || !matches!(width, 4 | 8 | 16)
        || scan_index >= u32::from(width) * u32::from(width)
    {
        return Err(EntropyError::InvalidScan);
    }
    if width == 4 {
        return Ok(SCAN4[variant as usize][scan_index as usize]);
    }
    if width == 8 {
        return Ok(SCAN8[variant as usize][scan_index as usize]);
    }
    if variant == 1 {
        return Ok(u16::try_from(scan_index).expect("validated scan index is below 256"));
    }
    if variant == 2 {
        return Ok(u16::try_from((scan_index % 16) * 16 + scan_index / 16)
            .expect("validated scan position is below 256"));
    }
    let mut cursor = 0_u32;
    for diagonal in 0..31_u32 {
        let first = diagonal.saturating_sub(15);
        let last = diagonal.min(15);
        if diagonal & 1 != 0 {
            for row in (first..=last).rev() {
                if cursor == scan_index {
                    return Ok(u16::try_from(row * 16 + diagonal - row)
                        .expect("scan position is below 256"));
                }
                cursor += 1;
            }
        } else {
            for row in first..=last {
                if cursor == scan_index {
                    return Ok(u16::try_from(row * 16 + diagonal - row)
                        .expect("scan position is below 256"));
                }
                cursor += 1;
            }
        }
    }
    Err(EntropyError::InvalidScan)
}

/// Decodes one contextual coefficient symbol and advances its scan cursor.
///
/// Short symbols use [`SHORT_SYMBOLS`]. Long colour symbols pack an eight-bit scan delta; other
/// long symbols use `partition_shift`. The payload low bit gives coefficient sign. Scan deltas are
/// applied before mapping to row-major position.
///
/// Returns [`EntropyError`] for invalid partitions, truncated codes, excessive magnitude, scan
/// overflow, or a position outside the transform.
pub fn decode_scanned_coefficient(
    reader: &mut ContextBitReader<'_>,
    partition_shift: u32,
    block_width: u8,
    scan_variant: u8,
    scan_cursor: &mut u32,
    colour_long_symbol: bool,
) -> Result<(u16, i16), EntropyError> {
    if partition_shift == 0 || partition_shift >= 16 {
        return Err(EntropyError::InvalidCode);
    }
    let width = read_context_prefix(reader)?;
    if width >= 31 {
        return Err(EntropyError::InvalidCode);
    }
    let payload = reader
        .read_value(width + 1)
        .ok_or(EntropyError::InvalidCode)?;
    let code = (1_u32 << width) + (payload >> 1);
    let (delta, magnitude) = if width > 6 {
        let partition = code
            .checked_sub(if colour_long_symbol { 1 } else { 0x80 })
            .ok_or(EntropyError::InvalidCode)?;
        if colour_long_symbol {
            (
                (partition & 0xff),
                (partition >> 8)
                    .checked_sub(1)
                    .ok_or(EntropyError::InvalidCode)?,
            )
        } else {
            (
                partition & ((1_u32 << partition_shift) - 1),
                (partition >> partition_shift) + 1,
            )
        }
    } else {
        let (magnitude, delta) = *SHORT_SYMBOLS
            .get(code.checked_sub(1).ok_or(EntropyError::InvalidCode)? as usize)
            .ok_or(EntropyError::InvalidCode)?;
        (u32::from(delta), u32::from(magnitude))
    };
    let scan_index = scan_cursor
        .checked_add(delta)
        .ok_or(EntropyError::InvalidScan)?;
    let position = coefficient_scan_position(block_width, scan_variant, scan_index)?;
    if magnitude > 0x7fff {
        return Err(EntropyError::InvalidCode);
    }
    *scan_cursor = scan_index + 1;
    let signed = if payload & 1 != 0 {
        -i32::try_from(magnitude).expect("checked magnitude fits i32")
    } else {
        i32::try_from(magnitude).expect("checked magnitude fits i32")
    };
    Ok((
        position,
        i16::try_from(signed).expect("checked magnitude fits i16"),
    ))
}

/// Stores spatial syntax-symbol history for contextual coefficients.
///
/// Byte values hold previous mode symbols. A parallel boolean plane marks neutral 16-wide symbols
/// so colour row transitions can normalise ambiguous zero modes correctly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoefficientHistory {
    /// Mode symbols in a padded row grid.
    pub(crate) bytes: Vec<u8>,
    /// Marks history cells that still carry neutral initial state.
    neutral: Vec<bool>,
    /// The first cell of the current four-row history window.
    pub(crate) origin: usize,
    /// The cell count between history rows.
    pub(crate) stride: usize,
    /// Neutral state can be exposed on the next colour row transition.
    neutral_pending: bool,
}

impl CoefficientHistory {
    /// Returns reserved heap storage in bytes.
    ///
    /// The current calculation charges `ceil(neutral.capacity()/8)` bytes. Rust `Vec<bool>` stores
    /// byte-sized elements, so this undercounts that vector. This method describes the existing
    /// account, not exact allocated storage.
    pub(crate) fn heap_bytes(&self) -> usize {
        self.bytes.capacity() + self.neutral.capacity().div_ceil(8)
    }

    /// Allocates neutral coefficient history for one slice.
    ///
    /// Storage includes one preceding row, four cells per slice block row, and four trailing rows.
    /// Bytes initialise to mode 2 and neutral flags initialise true.
    ///
    /// Returns [`EntropyError::InvalidHistory`] for zero or overflowing geometry and
    /// [`EntropyError::AllocationFailure`] when either vector cannot reserve storage.
    pub(crate) fn new(row_stride: u16, slice_rows: u16) -> Result<Self, EntropyError> {
        let stride = usize::from(row_stride);
        if stride == 0 {
            return Err(EntropyError::InvalidHistory);
        }
        let len = stride
            .checked_mul(1 + usize::from(slice_rows) * 4 + 4)
            .ok_or(EntropyError::InvalidHistory)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(len)
            .map_err(|_| EntropyError::AllocationFailure)?;
        bytes.resize(len, 2);
        let mut neutral = Vec::new();
        neutral
            .try_reserve_exact(len)
            .map_err(|_| EntropyError::AllocationFailure)?;
        neutral.resize(len, true);
        Ok(Self {
            bytes,
            neutral,
            origin: stride,
            stride,
            neutral_pending: true,
        })
    }

    /// Advances colour history to a new block row.
    ///
    /// The final symbol from each uniform four-row column becomes delayed history. On the first
    /// exposure of neutral state, luma and chroma normalisation rules adjust ambiguous modes. The
    /// active four-row window then resets to neutral mode 2.
    ///
    /// Returns [`EntropyError::InvalidHistory`] when the active window exceeds allocated storage.
    pub(crate) fn begin_block_row(
        &mut self,
        block_y: u16,
        normalise_zero: bool,
    ) -> Result<(), EntropyError> {
        let end = self.origin + self.stride * 4;
        if end > self.bytes.len() {
            return Err(EntropyError::InvalidHistory);
        }
        self.origin = self.stride;
        if block_y != 0 {
            let expose = self.neutral_pending;
            for column in 0..self.stride {
                let first = self.bytes[self.origin + column];
                let uniform =
                    (0..4).all(|row| self.bytes[self.origin + row * self.stride + column] == first);
                let neutral =
                    (0..4).all(|row| self.neutral[self.origin + row * self.stride + column]);
                let mut value = self.bytes[self.origin + 3 * self.stride + column];
                if expose && uniform && neutral && normalise_zero && matches!(value, 0 | 3) {
                    value = 1;
                } else if expose && uniform && neutral && !normalise_zero && value == 1 {
                    value = 0;
                }
                self.bytes[self.origin - self.stride + column] = value;
            }
            self.neutral_pending = false;
            self.bytes[self.origin..end].fill(2);
            self.neutral[self.origin..end].fill(true);
        }
        Ok(())
    }

    /// Advances alpha history to a new block row.
    ///
    /// For rows after zero, the final active row is copied into delayed history and the four-row
    /// active window resets to neutral mode 2.
    ///
    /// Returns [`EntropyError::InvalidHistory`] when origin or active window layout is
    /// inconsistent.
    pub(crate) fn begin_alpha_block_row(&mut self, block_y: u16) -> Result<(), EntropyError> {
        let end = self.origin + self.stride * 4;
        if self.origin != self.stride || end > self.bytes.len() {
            return Err(EntropyError::InvalidHistory);
        }
        if block_y != 0 {
            let source = self.origin + 3 * self.stride;
            self.bytes.copy_within(source..source + self.stride, 0);
            self.neutral.copy_within(source..source + self.stride, 0);
            self.bytes[self.origin..end].fill(2);
            self.neutral[self.origin..end].fill(true);
        }
        Ok(())
    }

    /// Stores one decoded history symbol over its component footprint.
    ///
    /// Width-four components occupy one history cell. Wider components fill a square measured in
    /// four-pixel cells, using `maximum_width` to determine side length.
    ///
    /// Returns [`EntropyError::InvalidHistory`] when the origin, offset, or footprint leaves
    /// storage.
    fn store(
        &mut self,
        offset: usize,
        value: u8,
        maximum_width: u8,
        decoded_width: u8,
        is_neutral: bool,
    ) -> Result<(), EntropyError> {
        let slot = self
            .origin
            .checked_add(offset)
            .ok_or(EntropyError::InvalidHistory)?;
        if slot >= self.bytes.len() {
            return Err(EntropyError::InvalidHistory);
        }
        if decoded_width == 4 {
            self.bytes[slot] = value;
            self.neutral[slot] = is_neutral;
            return Ok(());
        }
        let side = usize::from(maximum_width >> 2);
        if side == 0 || slot + (side - 1) * self.stride + side > self.bytes.len() {
            return Err(EntropyError::InvalidHistory);
        }
        for row in 0..side {
            let start = slot + row * self.stride;
            self.bytes[start..start + side].fill(value);
            self.neutral[start..start + side].fill(is_neutral);
        }
        Ok(())
    }
}

/// Identifies the history symbol that aliases mode 2 during neighbour-mode selection.
const HISTORY_SPECIAL_SYMBOL: u8 = 0x11;

/// Maps a stored history symbol to the mode used for neighbour selection.
///
/// The special symbol `0x11` aliases mode 2; every other symbol maps to itself.
fn history_mode(symbol: u8) -> u8 {
    if symbol == HISTORY_SPECIAL_SYMBOL {
        2
    } else {
        symbol
    }
}

/// Decodes one mode symbol using previous and delayed spatial history.
///
/// The leading decision chooses either a compact explicit symbol or neighbour prediction. Explicit
/// syntax skips values that would duplicate predicted modes. Predicted syntax can select between
/// left and delayed history and propagates neutral state for 16-wide components.
///
/// The decoded value is stored across its component footprint. Truncated syntax or invalid history
/// returns [`EntropyError`].
fn read_context_history_symbol(
    reader: &mut ContextBitReader<'_>,
    history: &mut CoefficientHistory,
    offset: usize,
    maximum_width: u8,
    decoded_width: u8,
) -> Result<u8, EntropyError> {
    let slot = history
        .origin
        .checked_add(offset)
        .ok_or(EntropyError::InvalidHistory)?;
    if slot >= history.bytes.len() || slot < history.stride {
        return Err(EntropyError::InvalidHistory);
    }
    let previous = history.bytes[slot - 1];
    let delayed = history.bytes[slot - history.stride];
    let previous_neutral = history.neutral[slot - 1];
    let delayed_neutral = history.neutral[slot - history.stride];
    let mode = history_mode(previous);
    if reader.available == 0 && !reader.refill(4) {
        return Err(EntropyError::InvalidCode);
    }
    let old_accumulator = reader.accumulator;
    reader.accumulator = lz(old_accumulator, 1);
    reader.available -= 1;
    let mut output;
    let mut is_neutral = false;
    if old_accumulator & 0x8000_0000 == 0 {
        if reader.available == 0 && !reader.refill(4) {
            return Err(EntropyError::InvalidCode);
        }
        let accumulator1 = reader.accumulator;
        let available1 = reader.available;
        reader.accumulator = lz(accumulator1, 1);
        reader.available -= 1;
        if accumulator1 & 0x8000_0000 == 0 {
            let accumulator2 = reader.accumulator;
            let available2 = reader.available;
            if available1 < 5 {
                if !reader.refill(4) {
                    return Err(EntropyError::InvalidCode);
                }
                output = (reader.accumulator >> (available1 + 27)) | (accumulator2 >> 28);
                let shift = 5 - available1;
                reader.accumulator = lz(reader.accumulator, shift);
                reader.available -= shift;
            } else {
                output = accumulator2 >> 28;
                reader.accumulator = lz(accumulator2, 4);
                reader.available = available2 - 4;
            }
            if output & 0xff > 1 || u32::from(mode) <= (output & 0xff) {
                output += 1;
            }
            if output & 0xff > 1 && mode != 2 && u32::from(mode) <= (output & 0xff) {
                output += 1;
            }
        } else {
            if reader.available == 0 && !reader.refill(4) {
                return Err(EntropyError::InvalidCode);
            }
            output = if reader.accumulator & 0x8000_0000 != 0 {
                u32::from(HISTORY_SPECIAL_SYMBOL)
            } else {
                2
            };
            reader.accumulator = lz(reader.accumulator, 1);
            reader.available -= 1;
        }
    } else {
        let candidate = history_mode(delayed);
        output = u32::from(mode);
        if mode != candidate {
            if reader.available == 0 && !reader.refill(4) {
                return Err(EntropyError::InvalidCode);
            }
            let select_delayed = reader.accumulator & 0x8000_0000 != 0;
            output = u32::from(if select_delayed { candidate } else { mode });
            is_neutral = decoded_width == 16
                && if select_delayed {
                    delayed_neutral
                } else {
                    previous_neutral
                };
            reader.accumulator = lz(reader.accumulator, 1);
            reader.available -= 1;
        } else {
            is_neutral = decoded_width == 16 && previous_neutral && delayed_neutral;
        }
    }
    let output = low_byte(output);
    history.store(offset, output, maximum_width, decoded_width, is_neutral)?;
    Ok(output)
}

/// Describes one contextual coefficient component request.
///
/// Geometry and stream flags control layout decoding, history addressing, symbol tables, and
/// coefficient long-code partitioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComponentRequest {
    /// The component argument within the containing 16 by 16 block.
    pub argument: u32,
    /// Its picture-horizontal offset in pixels.
    pub horizontal_offset: u16,
    /// The largest permitted transform width.
    pub maximum_width: u8,
    /// Zero selects a width from the stream. Values 4, 8, or 16 force a width.
    pub fixed_width: u8,
    /// Selects the colour form of long coefficient symbols and width-four state table.
    pub colour_long_symbols: bool,
    /// An outer layout bit already selected subdivision.
    pub width_selector_already_read: bool,
}

/// Classify a slot's decoded values for the reconstruction caller.
///
/// No coefficients gives `Inactive`. Exactly one value at scan position zero gives
/// `ConstantResidual`; any other active arrangement gives `FullBlockTransform`. Colour maps these
/// classes to its reconstruction selectors. Alpha only tests for `Inactive` and otherwise uses the
/// array directly; it does not run an inverse transform or expand a constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallbackKind {
    /// The slot has no coded coefficients.
    Inactive,
    /// One coded value at scan position zero; colour expands it as a constant residual.
    ConstantResidual,
    /// Active values that require a full transform for colour; alpha applies them directly.
    FullBlockTransform,
}

/// Layout and decoded values for one component argument.
///
/// `count` is the number of layout slots, one or four in production. It does not count active
/// coefficient slots. Each slot has a prediction mode even if its callback is inactive. Values
/// occupy the first `width*width` elements of a 256-element row-major array; unused elements start
/// at zero.
///
/// Colour interprets the values as transform coefficients. Alpha applies the same arrays directly
/// as residuals. Callback classification is derived from the coded count and scan endpoint, not
/// read as an independent selector field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentResult {
    /// The square transform width for every active slot.
    pub width: u8,
    /// Number of layout slots, including slots without coefficients.
    pub count: usize,
    /// The spatial prediction mode for each slot.
    pub modes: [u8; 4],
    /// The residual reconstruction kind for each slot.
    pub callbacks: [CallbackKind; 4],
    /// The coefficient scan variant for each slot.
    pub scan_variants: [u8; 4],
    /// Row-major signed coefficients for each slot.
    pub values: [[i16; 256]; 4],
}

/// Maps width-four alpha/luma state indices to slot-activity bit fields.
const STATE4_LUMA: [u8; 64] = [
    0x3f, 0x3c, 0x3e, 0x3d, 0x1c, 0x34, 0x2c, 0x38, 0x1f, 0x2f, 0x37, 0x3b, 0x0c, 0x30, 0x1e, 0x36,
    0x2e, 0x3a, 0x14, 0x1d, 0x35, 0x2d, 0x28, 0x39, 0x33, 0x0f, 0x00, 0x10, 0x04, 0x0e, 0x32, 0x20,
    0x08, 0x0d, 0x31, 0x24, 0x18, 0x16, 0x2b, 0x17, 0x2a, 0x15, 0x29, 0x26, 0x22, 0x1a, 0x12, 0x06,
    0x27, 0x0a, 0x02, 0x21, 0x11, 0x1b, 0x05, 0x25, 0x23, 0x01, 0x19, 0x09, 0x13, 0x07, 0x0b, 0x03,
];
/// Maps width-four colour state indices to slot-activity bit fields.
const STATE4_COLOUR: [u8; 64] = [
    0x3c, 0x3f, 0x3e, 0x3d, 0x1c, 0x34, 0x2c, 0x38, 0x0c, 0x30, 0x14, 0x28, 0x1f, 0x2f, 0x3b, 0x10,
    0x37, 0x04, 0x20, 0x08, 0x1e, 0x36, 0x2e, 0x3a, 0x1d, 0x00, 0x24, 0x35, 0x2d, 0x18, 0x39, 0x0f,
    0x33, 0x32, 0x0e, 0x0d, 0x31, 0x16, 0x2b, 0x2a, 0x17, 0x15, 0x29, 0x22, 0x12, 0x26, 0x0a, 0x06,
    0x1a, 0x11, 0x21, 0x27, 0x09, 0x05, 0x23, 0x25, 0x1b, 0x19, 0x13, 0x0b, 0x02, 0x07, 0x01, 0x03,
];
/// Maps width-eight prefix widths to slot-activity bit fields.
const STATE8: [u8; 64] = [
    0x04, 0x00, 0x06, 0x07, 0x05, 0x02, 0x01, 0x03, 0x3f, 0x3d, 0x3c, 0x3e, 0x35, 0x3b, 0x37, 0x39,
    0x1d, 0x2d, 0x2f, 0x1f, 0x34, 0x1c, 0x38, 0x2c, 0x36, 0x3a, 0x30, 0x1e, 0x31, 0x0d, 0x2e, 0x14,
    0x15, 0x29, 0x33, 0x0f, 0x0c, 0x2b, 0x28, 0x17, 0x32, 0x0e, 0x16, 0x2a, 0x25, 0x19, 0x18, 0x24,
    0x20, 0x08, 0x10, 0x11, 0x27, 0x04, 0x1a, 0x09, 0x1b, 0x21, 0x05, 0x0b, 0x26, 0x07, 0x0a, 0x13,
];

/// Decodes transform width, slot count, and prediction modes for one component.
///
/// A fixed width creates one slot. Otherwise, an optional state bit selects maximum width or
/// half-width with one or four slots. History offsets use four-pixel cells and account for
/// argument quadrant and horizontal picture position.
///
/// Returns [`EntropyError`] for invalid widths, slot counts, history addresses, or truncated mode
/// syntax.
fn read_component_layout(
    reader: &mut ContextBitReader<'_>,
    history: &mut CoefficientHistory,
    request: ComponentRequest,
) -> Result<(u8, usize, [u8; 4]), EntropyError> {
    let state_bit = if request.fixed_width == 0 && !request.width_selector_already_read {
        reader.read_value(1).ok_or(EntropyError::InvalidCode)?
    } else {
        0
    };
    let width = if request.fixed_width >= 16 {
        16
    } else if request.fixed_width != 0 {
        request.fixed_width
    } else {
        request.maximum_width >> state_bit
    };
    if !matches!(width, 4 | 8 | 16) {
        return Err(EntropyError::InvalidCode);
    }
    let component_count = if request.fixed_width != 0 {
        1_usize
    } else {
        ((state_bit + 1) << state_bit) as usize
    };
    if component_count > 4 || request.maximum_width == 0 {
        return Err(EntropyError::InvalidCode);
    }
    let base = usize::try_from(request.argument & !1)
        .ok()
        .and_then(|argument| argument.checked_mul(history.stride))
        .and_then(|base| {
            base.checked_add(
                (usize::from(request.horizontal_offset) + ((request.argument as usize & 1) << 3))
                    >> 2,
            )
        })
        .ok_or(EntropyError::InvalidHistory)?;
    let offsets = [
        base,
        base + 1,
        base + history.stride,
        base + history.stride + 1,
    ];
    let mut modes = [0; 4];
    if request.fixed_width == 16 {
        if !reader.read_bits(7) {
            return Err(EntropyError::InvalidCode);
        }
        modes[0] = 4;
    } else {
        for component in 0..component_count {
            modes[component] = read_context_history_symbol(
                reader,
                history,
                offsets[component],
                request.maximum_width,
                width,
            )?;
        }
    }

    Ok((width, component_count, modes))
}

/// Decodes one contextual component's layout and coefficient slots.
///
/// The special `fixed_width == 16` branch skips seven bits, forces mode 4, and uses diagonal
/// coefficient scan. It has no production caller. Production undivided colour obtains width 16
/// with `fixed_width == 0` and an already-read layout selector; it follows the ordinary
/// state-table path. Other widths decode a state table that marks active slots, derive scan
/// variants from prediction modes, and classify a single coefficient at scan position zero as a
/// constant residual.
///
/// Returns [`EntropyError`] for malformed layout, state, coefficient count, symbol, history, or
/// scan syntax.
pub fn decode_contextual_component(
    reader: &mut ContextBitReader<'_>,
    history: &mut CoefficientHistory,
    request: ComponentRequest,
) -> Result<ComponentResult, EntropyError> {
    let (width, component_count, modes) = read_component_layout(reader, history, request)?;
    let mut result = ComponentResult {
        width,
        count: component_count,
        modes: [0; 4],
        callbacks: [CallbackKind::Inactive; 4],
        scan_variants: [0; 4],
        values: [[0; 256]; 4],
    };
    result.modes[..component_count].copy_from_slice(&modes[..component_count]);
    if request.fixed_width == 16 {
        let (count_width, suffix, _) = read_context_index(reader)?;
        if count_width >= 31 {
            return Err(EntropyError::InvalidCode);
        }
        let count = (1_u32 << count_width) + suffix;
        let mut cursor = 0_u32;
        for _ in 0..count {
            let (position, value) =
                decode_scanned_coefficient(reader, 8, 16, 0, &mut cursor, true)?;
            result.values[0][position as usize] = value;
        }
        result.callbacks[0] = if count == 1 && cursor == 1 {
            CallbackKind::ConstantResidual
        } else {
            CallbackKind::FullBlockTransform
        };
        return Ok(result);
    }

    let state_value = if width == 4 {
        let (_, _, state_index) = read_context_index(reader)?;
        let table = if request.colour_long_symbols {
            &STATE4_COLOUR
        } else {
            &STATE4_LUMA
        };
        *table
            .get(state_index as usize)
            .ok_or(EntropyError::InvalidCode)?
    } else {
        let state_width = read_context_prefix(reader)?;
        *STATE8
            .get(state_width as usize)
            .ok_or(EntropyError::InvalidCode)?
    };
    const MODE_SCAN_VARIANT: [u8; 17] = [1, 2, 0, 0, 1, 1, 0, 2, 2, 0, 0, 1, 1, 0, 0, 2, 2];
    for slot in 0..component_count {
        let state_bit_index = (component_count + 1) - slot;
        if (u32::from(state_value) >> state_bit_index) & 1 == 0 {
            continue;
        }
        let (count_width, suffix, _) = read_context_index(reader)?;
        if count_width >= 31 {
            return Err(EntropyError::InvalidCode);
        }
        let coefficient_count = (1_u32 << count_width) + suffix;
        let raw_mode = result.modes[slot];
        let scan_variant = if request.colour_long_symbols || raw_mode == HISTORY_SPECIAL_SYMBOL {
            0
        } else {
            *MODE_SCAN_VARIANT
                .get(raw_mode as usize)
                .ok_or(EntropyError::InvalidCode)?
        };
        result.scan_variants[slot] = scan_variant;
        let partition_shift = match width {
            4 => 4,
            8 => 6,
            16 => 8,
            _ => unreachable!(),
        };
        let mut cursor = 0_u32;
        for _ in 0..coefficient_count {
            let (position, value) = decode_scanned_coefficient(
                reader,
                partition_shift,
                width,
                scan_variant,
                &mut cursor,
                request.colour_long_symbols,
            )?;
            result.values[slot][position as usize] = value;
        }
        result.callbacks[slot] = if coefficient_count == 1 && cursor == 1 {
            CallbackKind::ConstantResidual
        } else {
            CallbackKind::FullBlockTransform
        };
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rle_consumes_literal_after_terminal_repetition() {
        // Two literal 5s, then a run of one 5, followed by the required literal 7.
        let mut bits = crate::bits::BitReader::new(&[0b0101_0101, 0b0100_1110]);
        let mut output = [0; 3];
        let logical = decode_rle_plane(&mut bits, &mut output, 3).unwrap();
        assert_eq!(logical, 4);
        assert_eq!(output, [5, 5, 5]);
        assert_eq!(bits.bit_position(), 15);
    }

    #[test]
    fn contextual_prefix_uses_terminal_lookahead_without_reading_past_end() {
        let mut reader = ContextBitReader::new(&[0, 0, 0, 0]);
        assert!(reader.initialise_at_bit(0));
        assert!(reader.read_bits(32));
        assert_eq!(
            read_context_prefix(&mut reader),
            Err(EntropyError::InvalidCode)
        );
        assert_eq!(reader.byte_offset, reader.size);
    }

    #[test]
    fn partial_refills_do_not_consume_missing_bits() {
        let data = [0_u8; 5];
        let mut literal_reader = ContextBitReader::new(&data);
        literal_reader.byte_offset = 4;
        literal_reader.available = 1;
        literal_reader.accumulator = 0x8000_0000;
        assert_eq!(
            read_contextual_literal(&mut literal_reader, 31),
            Err(EntropyError::InvalidCode)
        );

        let mut run_reader = ContextBitReader::new(&data);
        run_reader.byte_offset = 4;
        assert_eq!(
            read_contextual_run_control(&mut run_reader),
            Err(EntropyError::InvalidCode)
        );

        for available in 0..=8 {
            for accumulator in [0, 0x1000_0000, 0x8000_0000, u32::MAX] {
                let mut index_reader = ContextBitReader::new(&data);
                index_reader.byte_offset = 4;
                index_reader.available = available;
                index_reader.accumulator = accumulator;
                let _ = read_context_index(&mut index_reader);
            }
        }
    }
}
