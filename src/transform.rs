//! Convert colour coefficients into residual samples.
//!
//! A coefficient is a coded weight for a spatial pattern. The coefficient at position zero is the
//! DC coefficient: it supplies a constant contribution across the square. Dequantisation here
//! means multiplication by selected integer weights, with the shifts and narrowing listed below.
//! An inverse transform combines those patterns to obtain a residual at each sample position.
//! Colour reconstruction adds this residual to a prediction, then clips the component value. This
//! module produces signed residuals only; it does not add predictions, convert colour channels, or
//! process predictive alpha.
//!
//! Coefficient and residual arrays use row-major squares. Widths 4 and 8 have up to four packed
//! slots; slot `s` occupies `s*w*w..(s+1)*w*w` in output. Width 16 has a separate 256-value
//! function. A callback selector identifies inactive, constant-only, or full-transform
//! reconstruction. It is an integer choice in this implementation, not a callback supplied by an
//! API user.
//!
//! # Integer stages
//!
//! | Path | Scaling before transform | Later full-transform operations (selector 1 is described below) |
//! | --- | --- | --- |
//! | Width 4, normal generic | `(W4[i]*c + 1) >> 1`, narrow to `i16` | Column pass, row pass with `+32` then `>>6`; transpose output |
//! | Width 4, alternate generic | `A4[i]*c`, narrow to `i16` | Same generic passes |
//! | Width 4, selector 7 | `CALLBACK7_W4[i]*c` normally, `A4[i]*c` otherwise; narrow | Dedicated passes and transposed output; final unsigned wrapping addition and logical shift |
//! | Width 8, normal | `(W8[i]*c + 4) >> 3`, narrow | Add 32 to element 0 with `i16` wrapping; column and row passes; final arithmetic `>>6`; transpose |
//! | Width 8, alternate | `(A8[i]*c + 2) >> 2`, narrow | Same width-8 passes |
//! | Width 16 | Matrix coefficient product times scale 144 or 160 in production | First pass `+128`, `>>8`, narrow to `i32`; second `+2048`, `>>12`, narrow to `i32` then `i16` |
//!
//! A separable transform performs the one-dimensional calculation on columns, then rows. Narrowing
//! keeps low bits and interprets them as signed; it is not clipping. For example, low-halfword
//! extraction from `0x1ffff` gives -1. Signed right shifts preserve the sign and round negative
//! values downwards. The rounding offset is added before the shift. The selector-7 final
//! expression instead casts to `u32`, adds 32 with wrapping, shifts logically, then narrows;
//! retain that exact order.
//!
//! Selector 1 uses only the scaled coefficient at position zero: `(value+32)>>6` fills the slot.
//! Normal width-4 coefficient 5 therefore produces 11 everywhere. Inactive slots leave output
//! unchanged; colour callers initialise the workspace to zero. Width-16 active slots always use
//! the matrix transform, including a constant-only classification. The crate page connects these
//! residuals to prediction and final pixels.

/// Invalid inverse-transform arguments.
///
/// Coefficient values intentionally use wrapping halfword arithmetic and do not cause errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TransformError {
    /// A transform width other than 4 or 8.
    #[error("invalid transform block width")]
    InvalidWidth,
    /// More coefficient slots than the four-element storage can hold.
    #[error("invalid transform coefficient slot count")]
    InvalidCount,
    /// An input or output slice lacks the required samples.
    #[error("transform buffer is too small")]
    BufferTooSmall,
}

/// Wraps an integer to a signed 16-bit value.
///
/// The implemented reconstruction keeps the low 16 bits and interprets them as signed. Values
/// above `i16::MAX` therefore wrap rather than saturate.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the coefficient format keeps the low signed halfword"
)]
pub fn truncate_signed_halfword(value: u32) -> i16 {
    value as i16
}

/// Wraps an intermediate signed word to a signed halfword.
///
/// Small inverse transforms rely on two's-complement narrowing after arithmetic stages, so this
/// conversion wraps instead of saturating.
#[expect(
    clippy::cast_possible_truncation,
    reason = "intermediate coefficients wrap to a signed halfword"
)]
fn truncate_i32(value: i32) -> i16 {
    value as i16
}

/// Rounds and shifts a transform accumulator into a wrapping signed word.
///
/// The caller supplies a positive shift. Adding the half-unit implements the codec's integer
/// rounding before arithmetic right shift.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the rounded transform result wraps to a signed word"
)]
fn round_shift(value: i64, shift: u32) -> i32 {
    ((value + (1_i64 << (shift - 1))) >> shift) as i32
}

/// Reconstruct 256 signed residual samples from a 16 by 16 coefficient square.
///
/// Input and output are row-major; only their first 256 entries participate. The fixed matrix
/// `T[k][r]` weights coefficient row `k` for output row `r`. First calculate
/// `sum(T[k][r]*coeff[k*16+c]*transform_scale)` in `i64`, add 128, shift arithmetically by eight,
/// and narrow to `i32`. Then combine each intermediate row with `T[k][c]`, add 2048, shift by
/// twelve, narrow to `i32`, then keep the signed low 16 bits. The output is not clipped.
///
/// Colour supplies scale 144 for normal profile and 160 for alternate profile. This helper accepts
/// any `i16` scale. With coefficient zero equal to 64, all others zero, and scale 64, every
/// residual is 16. Inputs and output elements after 255 are unchanged.
///
/// # Errors
///
/// Returns [`TransformError::BufferTooSmall`] before writes if either slice has fewer than 256
/// elements.
pub fn inverse_transform_16x16(
    coefficients: &[i16],
    transform_scale: i16,
    output: &mut [i16],
) -> Result<(), TransformError> {
    const T: [[i16; 16]; 16] = [
        [
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        ],
        [
            90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90,
        ],
        [
            89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89,
        ],
        [
            87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87,
        ],
        [
            83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83,
        ],
        [
            80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80,
        ],
        [
            75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75,
        ],
        [
            70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70,
        ],
        [
            64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64,
        ],
        [
            57, -80, -25, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 25, 80, -57,
        ],
        [
            50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50,
        ],
        [
            43, -90, 57, 25, -87, 70, 9, -80, 80, -9, -70, 87, -25, -57, 90, -43,
        ],
        [
            36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36,
        ],
        [
            25, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -25,
        ],
        [
            18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18,
        ],
        [
            9, -25, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 25, -9,
        ],
    ];
    if coefficients.len() < 256 || output.len() < 256 {
        return Err(TransformError::BufferTooSmall);
    }
    let mut intermediate = [0_i32; 256];
    for row in 0..16 {
        for col in 0..16 {
            let mut sum = 0_i64;
            for k in 0..16 {
                sum += i64::from(T[k][row])
                    * i64::from(coefficients[k * 16 + col])
                    * i64::from(transform_scale);
            }
            intermediate[row * 16 + col] = round_shift(sum, 8);
        }
    }
    for row in 0..16 {
        for col in 0..16 {
            let mut sum = 0_i64;
            for k in 0..16 {
                sum += i64::from(intermediate[row * 16 + k]) * i64::from(T[k][col]);
            }
            output[row * 16 + col] = truncate_i32(round_shift(sum, 12));
        }
    }
    Ok(())
}

/// Normal-profile dequantisation weights for a 4 by 4 transform.
const W4: [i16; 16] = [
    288, 368, 288, 368, 368, 464, 368, 464, 288, 368, 288, 368, 368, 464, 368, 464,
];
/// Alternate-profile dequantisation weights for a 4 by 4 transform.
const A4: [i16; 16] = [
    160, 208, 160, 208, 208, 256, 208, 256, 160, 208, 160, 208, 208, 256, 208, 256,
];
/// Normal-profile weights for the callback-7 4 by 4 transform.
const CALLBACK7_W4: [i16; 16] = [
    144, 184, 144, 184, 184, 232, 184, 232, 144, 184, 144, 184, 184, 232, 184, 232,
];
/// Normal-profile dequantisation weights for an 8 by 8 transform.
const W8: [i16; 64] = [
    576, 544, 736, 544, 576, 544, 736, 544, 544, 512, 688, 512, 544, 512, 688, 512, 736, 688, 928,
    688, 736, 688, 928, 688, 544, 512, 688, 512, 544, 512, 688, 512, 576, 544, 736, 544, 576, 544,
    736, 544, 544, 512, 688, 512, 544, 512, 688, 512, 736, 688, 928, 688, 736, 688, 928, 688, 544,
    512, 688, 512, 544, 512, 688, 512,
];
/// Alternate-profile dequantisation weights for an 8 by 8 transform.
const A8: [i16; 64] = [
    320, 304, 400, 304, 320, 304, 400, 304, 304, 288, 384, 288, 304, 288, 384, 288, 400, 384, 512,
    384, 400, 384, 512, 384, 304, 288, 384, 288, 304, 288, 384, 288, 320, 304, 400, 304, 320, 304,
    400, 304, 304, 288, 384, 288, 304, 288, 384, 288, 400, 384, 512, 384, 400, 384, 512, 384, 304,
    288, 384, 288, 304, 288, 384, 288,
];

/// Applies the codec's separable 4 by 4 integer transform in place.
///
/// Callback 1 represents a DC-only slot and scales only the first value. Other callback values use
/// the full transform; the caller guarantees a 16-element slice.
fn transform4(values: &mut [i16], callback: u8) {
    if callback == 1 {
        values[0] = truncate_i32((i32::from(values[0]) + 32) >> 6);
        return;
    }
    let mut temp = [0_i16; 16];
    for col in 0..4 {
        let (x0, x1, x2, x3) = (
            i32::from(values[col]),
            i32::from(values[4 + col]),
            i32::from(values[8 + col]),
            i32::from(values[12 + col]),
        );
        let (a0, a1, a2, a3) = (x0 + x2, x0 - x2, x1 + (x3 >> 1), (x1 >> 1) - x3);
        temp[col] = truncate_i32(a0 + a2);
        temp[4 + col] = truncate_i32(a1 + a3);
        temp[8 + col] = truncate_i32(a1 - a3);
        temp[12 + col] = truncate_i32(a0 - a2);
    }
    for row in 0..4 {
        let s = &temp[row * 4..row * 4 + 4];
        let (x0, x1, x2, x3) = (
            i32::from(s[0]),
            i32::from(s[1]),
            i32::from(s[2]),
            i32::from(s[3]),
        );
        let (a0, a1, a2, a3) = (x0 + x2, x0 - x2, x1 + (x3 >> 1), (x1 >> 1) - x3);
        for (j, v) in [a0 + a2, a1 + a3, a1 - a3, a0 - a2].into_iter().enumerate() {
            values[row * 4 + j] = truncate_i32((v + 32) >> 6);
        }
    }
}

/// Applies the codec's separable 8 by 8 integer transform in place.
///
/// Callback 1 represents a DC-only slot and scales only the first value. The full path adds its
/// rounding bias before column and row passes and uses wrapping halfword intermediates.
fn transform8(values: &mut [i16], callback: u8) {
    if callback == 1 {
        values[0] = truncate_i32((i32::from(values[0]) + 32) >> 6);
        return;
    }
    /// Applies one dimension of the 8-point integer transform.
    ///
    /// `stride` and `ds` select input and output spacing. The caller provides eight accessible
    /// values in each direction; `round` enables the final six-bit scaling pass.
    fn one(src: &[i16], stride: usize, dst: &mut [i16], ds: usize, round: bool) {
        let x = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| i32::from(src[i * stride]));
        let (e0, e1, e2, e3) = (
            x[0] + x[4],
            x[0] - x[4],
            x[2] + (x[6] >> 1),
            (x[2] >> 1) - x[6],
        );
        let (a0, a1, a2, a3) = (
            x[5] + x[3] + x[1] + (x[1] >> 1),
            x[5] - (x[3] + x[7] + (x[7] >> 1)),
            x[7] - (x[3] + (x[3] >> 1)) + x[1],
            x[7] + x[5] + (x[5] >> 1) - x[1],
        );
        let (o0, o1, o2, o3) = (
            a0 - (a1 >> 2),
            a2 + (a3 >> 2),
            a3 - (a2 >> 2),
            a1 + (a0 >> 2),
        );
        let ev = [e0 + e2, e1 + e3, e1 - e3, e0 - e2];
        let out = [
            ev[0] + o0,
            ev[1] - o2,
            ev[2] + o1,
            ev[3] + o3,
            ev[3] - o3,
            ev[2] - o1,
            ev[1] + o2,
            ev[0] - o0,
        ];
        for i in 0..8 {
            dst[i * ds] = truncate_i32(if round { out[i] >> 6 } else { out[i] });
        }
    }
    let mut tmp = [0_i16; 64];
    values[0] = values[0].wrapping_add(32);
    for col in 0..8 {
        let mut d = [0_i16; 8];
        one(&values[col..], 8, &mut d, 1, false);
        for i in 0..8 {
            tmp[i * 8 + col] = d[i];
        }
    }
    for row in 0..8 {
        let mut d = [0_i16; 8];
        one(&tmp[row * 8..], 1, &mut d, 1, true);
        values[row * 8..row * 8 + 8].copy_from_slice(&d);
    }
}

/// Reconstructs callback-7 coefficients into one row-major 4 by 4 slot.
///
/// Callback 7 has dedicated weights and a transposed placement that differs from the generic 4 by
/// 4 path. The fixed-size input guarantees all coefficient accesses; the caller provides at least
/// 16 output elements.
fn reconstruct_callback7(coefficients: &[i16; 64], alternate_profile: bool, output: &mut [i16]) {
    let weights = if alternate_profile {
        &A4
    } else {
        &CALLBACK7_W4
    };
    let mut q = [0_i16; 16];
    for i in 0..16 {
        q[i] = truncate_signed_halfword(
            (i32::from(coefficients[i]) * i32::from(weights[i])).cast_unsigned(),
        );
    }
    let mut t = [0_i16; 16];
    for c in 0..4 {
        let (x0, x1, x2, x3) = (
            i32::from(q[c]),
            i32::from(q[4 + c]),
            i32::from(q[8 + c]),
            i32::from(q[12 + c]),
        );
        let (a0, a1, a2, a3) = (x0 + x2, x0 - x2, x1 + (x3 >> 1), (x1 >> 1) - x3);
        t[c] = truncate_i32(a0 + a2);
        t[4 + c] = truncate_i32(a1 + a3);
        t[8 + c] = truncate_i32(a1 - a3);
        t[12 + c] = truncate_i32(a0 - a2);
    }
    for r in 0..4 {
        let s = &t[r * 4..r * 4 + 4];
        let (x0, x1, x2, x3) = (
            i32::from(s[0]),
            i32::from(s[1]),
            i32::from(s[2]),
            i32::from(s[3]),
        );
        let (a0, a1, a2, a3) = (x0 + x2, x0 - x2, x1 + (x3 >> 1), (x1 >> 1) - x3);
        for c in 0..4 {
            output[c * 4 + r] = truncate_signed_halfword(
                ([a0 + a2, a1 + a3, a1 - a3, a0 - a2][c]
                    .cast_unsigned()
                    .wrapping_add(32))
                    >> 6,
            );
        }
    }
}

/// Reconstruct packed width-4 or width-8 residual slots.
///
/// `count` is 0..=4. Slot `s` reads `coefficients[s][..width*width]` in row order and writes
/// `output[s*width*width..(s+1)*width*width]`. It does not place that square in a larger image.
/// The colour caller places it later and supplies a zeroed workspace so inactive slots contribute
/// zero.
///
/// Selector 0 leaves a slot unchanged. Selector 1 scales coefficient zero and fills the slot with
/// its constant residual. Selector 7 at width 4 uses the dedicated weights and transform; other
/// non-zero selectors use the generic full transform. At width 8, selector 7 also uses that
/// generic full transform. The module table specifies profile weights, rounding, wrapping, and
/// final transpose. All inputs and output after `count*width*width` remain unchanged.
///
/// # Errors
///
/// Returns [`TransformError::InvalidWidth`] unless width is 4 or 8,
/// [`TransformError::InvalidCount`] for count above four, or [`TransformError::BufferTooSmall`]
/// for insufficient output. All validation precedes slot writes.
pub fn reconstruct_component_residual(
    width: usize,
    count: usize,
    coefficients: &[[i16; 64]; 4],
    callback_modes: &[u8; 4],
    alternate_profile: bool,
    output: &mut [i16],
) -> Result<(), TransformError> {
    if width != 4 && width != 8 {
        return Err(TransformError::InvalidWidth);
    }
    if count > 4 {
        return Err(TransformError::InvalidCount);
    }
    let slot_len = width * width;
    if output.len() < slot_len * count {
        return Err(TransformError::BufferTooSmall);
    }
    for slot in 0..count {
        let mode = callback_modes[slot];
        if mode == 0 {
            continue;
        }
        let mut v = [0_i16; 64];
        v[..slot_len].copy_from_slice(&coefficients[slot][..slot_len]);
        if width == 4 && mode == 7 {
            reconstruct_callback7(
                &coefficients[slot],
                alternate_profile,
                &mut output[slot * slot_len..(slot + 1) * slot_len],
            );
            continue;
        }
        let weights = if width == 4 && alternate_profile {
            &A4[..]
        } else if width == 4 {
            &W4[..]
        } else if alternate_profile {
            &A8[..]
        } else {
            &W8[..]
        };
        if width == 4 && alternate_profile {
            for i in 0..slot_len {
                v[i] = truncate_i32(i32::from(weights[i]) * i32::from(v[i]));
            }
        } else {
            let shift = if width == 4 {
                1
            } else if alternate_profile {
                2
            } else {
                3
            };
            let round = 1_i32 << (shift - 1);
            for i in 0..slot_len {
                v[i] = truncate_i32((round + i32::from(weights[i]) * i32::from(v[i])) >> shift);
            }
        }
        if width == 4 {
            transform4(&mut v[..16], mode);
        } else {
            transform8(&mut v[..64], mode);
        }
        if mode == 1 {
            output[slot * slot_len..(slot + 1) * slot_len].fill(v[0]);
        } else {
            for r in 0..width {
                for c in 0..width {
                    output[slot * slot_len + r * width + c] = v[c * width + r];
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inverse16_dc_has_fixed_scaled_vector() {
        let mut c = [0_i16; 256];
        c[0] = 64;
        let mut out = [0_i16; 256];
        inverse_transform_16x16(&c, 64, &mut out).unwrap();
        assert_eq!(out, [16_i16; 256]);
    }
    #[test]
    fn constant_4x4_callback_dequantizes_and_fills() {
        let mut c = [[0_i16; 64]; 4];
        c[0][0] = 5;
        let mut out = [-1_i16; 16];
        reconstruct_component_residual(4, 1, &c, &[1, 0, 0, 0], false, &mut out).unwrap();
        assert_eq!(out, [11_i16; 16]);
    }
    #[test]
    fn signed_narrowing_wraps() {
        assert_eq!(truncate_signed_halfword(0x1_ffff), -1);
        assert_eq!(truncate_i32(0x1_8000), i16::MIN);
        assert_eq!(round_shift(i64::from(i32::MAX) + 1, 1), 0x4000_0000);
        assert_eq!(round_shift((i64::from(i32::MAX) + 1) * 2, 1), i32::MIN);
    }

    #[test]
    fn callback7_normal_profile_uses_144_dc_weight() {
        let mut coefficients = [[0_i16; 64]; 4];
        coefficients[0][0] = 1;
        let mut output = [0_i16; 16];
        reconstruct_component_residual(4, 1, &coefficients, &[7, 0, 0, 0], false, &mut output)
            .unwrap();
        assert_eq!(output, [2_i16; 16]);
    }

    #[test]
    fn alternate_width4_generic_path_multiplies_without_rounding_shift() {
        let mut coefficients = [[0_i16; 64]; 4];
        coefficients[0][0] = 1;
        let mut output = [0_i16; 16];
        reconstruct_component_residual(4, 1, &coefficients, &[1, 0, 0, 0], true, &mut output)
            .unwrap();
        assert_eq!(output, [3_i16; 16]);
    }
}
