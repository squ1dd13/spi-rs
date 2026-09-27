//! Integer coefficient reconstruction primitives.

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TransformError {
    #[error("invalid transform block width")]
    InvalidWidth,
    #[error("invalid transform coefficient slot count")]
    InvalidCount,
    #[error("transform buffer is too small")]
    BufferTooSmall,
}

/// Wrap an integer to a signed 16-bit value, matching a narrowing halfword cast.
pub fn truncate_signed_halfword(value: u32) -> i16 {
    let bytes = value.to_le_bytes();
    i16::from_le_bytes([bytes[0], bytes[1]])
}

fn truncate_i32(value: i32) -> i16 {
    let bytes = value.to_le_bytes();
    i16::from_le_bytes([bytes[0], bytes[1]])
}

fn round_shift(value: i64, shift: u32) -> i32 {
    let bytes = ((value + (1_i64 << (shift - 1))) >> shift).to_le_bytes();
    i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// Apply the separable 16 by 16 inverse transform.
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

const W4: [i16; 16] = [
    288, 368, 288, 368, 368, 464, 368, 464, 288, 368, 288, 368, 368, 464, 368, 464,
];
const A4: [i16; 16] = [
    160, 208, 160, 208, 208, 256, 208, 256, 160, 208, 160, 208, 208, 256, 208, 256,
];
const CALLBACK7_W4: [i16; 16] = [
    144, 184, 144, 184, 184, 232, 184, 232, 144, 184, 144, 184, 184, 232, 184, 232,
];
const W8: [i16; 64] = [
    576, 544, 736, 544, 576, 544, 736, 544, 544, 512, 688, 512, 544, 512, 688, 512, 736, 688, 928,
    688, 736, 688, 928, 688, 544, 512, 688, 512, 544, 512, 688, 512, 576, 544, 736, 544, 576, 544,
    736, 544, 544, 512, 688, 512, 544, 512, 688, 512, 736, 688, 928, 688, 736, 688, 928, 688, 544,
    512, 688, 512, 544, 512, 688, 512,
];
const A8: [i16; 64] = [
    320, 304, 400, 304, 320, 304, 400, 304, 304, 288, 384, 288, 304, 288, 384, 288, 400, 384, 512,
    384, 400, 384, 512, 384, 304, 288, 384, 288, 304, 288, 384, 288, 320, 304, 400, 304, 320, 304,
    400, 304, 304, 288, 384, 288, 304, 288, 384, 288, 400, 384, 512, 384, 400, 384, 512, 384, 304,
    288, 384, 288, 304, 288, 384, 288,
];

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

fn transform8(values: &mut [i16], callback: u8) {
    if callback == 1 {
        values[0] = truncate_i32((i32::from(values[0]) + 32) >> 6);
        return;
    }
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

/// Reconstruct one component's callback slots into a row-major packed output.
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

/// Each slot occupies `width * width` samples. Callback 7 uses a dedicated 4 by 4 path.
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
