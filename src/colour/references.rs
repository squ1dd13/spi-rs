//! Prepare colour samples above and to the left of the next block.
//!
//! Reference banks include a corner at index 0, immediate neighbours at 1..=16, and extensions at
//! 17..=32. Top banks are `banks[0..3]` and left banks are `banks[4..7]`, in G, B, R order. Slots
//! 3, 7, and 8 do not supply these edges.
//!
//! [`update`] runs before byte residual and predictive methods. It reads persistent picture
//! planes, applies the record-first-row rules or the four-neighbour availability transition, then
//! converts all 33 samples in both orientations to L, U+256, V+256. These transformed values feed
//! predictive colour; byte residuals use the byte top banks. The conversion equations are on the
//! crate page.
//!
//! Unavailable extensions are not an instruction to read arbitrary neighbouring storage.
//! [`transition`] fills or repeats specific prepared values. Its code has one availability bit in
//! each nibble: above-left at bit 12, above-right at bit 8, above at bit 4, left at bit 0. The
//! rules describe this implementation; no independent specification establishes why each
//! replacement was selected.

use crate::{
    DecodeError,
    colour::{ColourState, invalid},
};

/// Updates byte and signed colour reference banks for the current block.
///
/// The procedure refreshes physical top and left edges from the persistent planes, synthesises
/// unavailable extensions according to the four-neighbour availability code, then converts
/// green-blue-red references into luma-like and biased signed-chroma histories.
///
/// Returns invalid colour syntax when the block position lies outside picture history.
pub(super) fn update(
    state: &mut ColourState,
    block_x: usize,
    local_y: usize,
    global_y: usize,
) -> Result<(), DecodeError> {
    if block_x >= state.block_width || global_y >= state.total_block_rows {
        return Err(invalid("colour boundary is outside picture"));
    }
    refresh(state, block_x, local_y, global_y);
    if local_y == 0 {
        if block_x == 0 {
            for p in 0..3 {
                state.banks[p].fill(128);
                state.banks[p + 4].fill(128);
            }
        } else if state.available[global_y * state.block_width + block_x - 1] != 0 {
            for p in 0..3 {
                let left_one = state.banks[p + 4][1];
                let left_sixteen = state.banks[p + 4][16];
                state.banks[p].fill(left_one);
                state.banks[p + 4][17..].fill(left_sixteen);
            }
        } else {
            for p in 0..3 {
                state.banks[p].fill(128);
                state.banks[p + 4].fill(128);
            }
        }
    } else {
        let left = block_x != 0 && state.available[global_y * state.block_width + block_x - 1] != 0;
        let above = state.available[(global_y - 1) * state.block_width + block_x] != 0;
        let above_left =
            block_x != 0 && state.available[(global_y - 1) * state.block_width + block_x - 1] != 0;
        let above_right = block_x + 1 < state.block_width
            && state.available[(global_y - 1) * state.block_width + block_x + 1] != 0;
        let code = (u16::from(above_left) << 12)
            | (u16::from(above_right) << 8)
            | (u16::from(above) << 4)
            | u16::from(left);
        let (top, lower) = state.banks.split_at_mut(4);
        for p in 0..3 {
            transition(code, &mut top[p], &mut lower[p]);
        }
    }
    for orientation in 0..2 {
        let bank = if orientation == 0 {
            [0, 1, 2]
        } else {
            [4, 5, 6]
        };
        let history = orientation * 3;
        for i in 0..33 {
            let a = i32::from(state.banks[bank[0]][i]);
            let b = i32::from(state.banks[bank[1]][i]);
            let c = i32::from(state.banks[bank[2]][i]);
            let delta = c - b;
            let middle = b + (delta >> 1);
            let residual = a - middle;
            state.histories[history][i] =
                i16::try_from(middle + (residual >> 1)).expect("byte colour history fits i16");
            state.histories[history + 1][i] =
                i16::try_from(residual + 0x100).expect("byte colour history fits i16");
            state.histories[history + 2][i] =
                i16::try_from(delta + 0x100).expect("byte colour history fits i16");
        }
    }
    Ok(())
}

/// Loads available top and left samples from the reconstructed picture planes.
///
/// On the first local row, this helper leaves top banks unchanged; `update` subsequently replaces
/// them with 128 or left-derived values. The first column sets left neighbours to 128. Left corner
/// is 128 on the first local row or first column. Later rows load top neighbours and clamp their
/// extensions to padded width.
fn refresh(state: &mut ColourState, block_x: usize, local_y: usize, global_y: usize) {
    let x = block_x * 16;
    let y = global_y * 16;
    if local_y != 0 {
        for p in 0..3 {
            if block_x == 0 {
                state.banks[p][0] = 128;
            } else {
                state.banks[p][0] = state.planes[p][(y - 1) * state.stride + x - 1];
            }
            for i in 1..33 {
                let index = x + i - 1;
                state.banks[p][i] =
                    state.planes[p][(y - 1) * state.stride + index.min(state.stride - 1)];
            }
        }
    }
    for p in 0..3 {
        state.banks[p + 4][0] = if local_y == 0 || block_x == 0 {
            128
        } else {
            state.planes[p][(y - 1) * state.stride + x - 1]
        };
        if block_x == 0 {
            state.banks[p + 4][1..17].fill(128);
        } else {
            for row in 0..16 {
                state.banks[p + 4][row + 1] = state.planes[p][(y + row) * state.stride + x - 1];
            }
        }
    }
}

/// Extends or neutralises edge banks from a neighbour-availability code.
///
/// The four nibbles represent above-left, above-right, above, and left availability. Each listed
/// combination applies the implemented fill or corner average; unrecognised codes leave refreshed
/// samples unchanged.
fn transition(code: u16, top: &mut [u8; 33], left: &mut [u8; 33]) {
    let t = *top;
    let l = *left;
    let avg = |a: u8, b: u8| {
        u8::try_from((u16::from(a) + u16::from(b) + 1) >> 1).expect("average of two bytes fits u8")
    };
    match code {
        0 => {
            top.fill(128);
            left.fill(128);
        }
        1 => {
            top.fill(l[1]);
            left[17..].fill(l[16]);
        }
        0x10 => {
            left.fill(t[1]);
            top[17..].fill(t[16]);
        }
        0x11 => {
            let c = avg(t[1], l[1]);
            top[0] = c;
            left[0] = c;
            top[17..].fill(t[16]);
            left[17..].fill(l[16]);
        }
        0x100 => {
            top[..17].fill(t[17]);
            left.fill(t[17]);
        }
        0x101 => {
            let c = avg(t[17], l[1]);
            top[..17].fill(c);
            left[0] = c;
            left[17..].fill(l[16]);
        }
        0x110 => {
            top[0] = t[1];
            left.fill(t[1]);
        }
        0x111 => {
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
            top[1..17].fill(avg(t[0], t[17]));
            left[1..].fill(l[0]);
        }
        0x1101 => {
            top[1..17].fill(avg(t[0], t[17]));
            left[17..].fill(l[16]);
        }
        0x1110 => left[1..].fill(l[0]),
        0x1111 => left[17..].fill(l[16]),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_rgb_banks_map_to_biased_signed_chroma() {
        let mut state = ColourState::new(1, 2).unwrap();
        for plane in &mut state.planes {
            plane.fill(128);
        }
        state.available[0] = 1;
        state.available[1] = 1;

        update(&mut state, 0, 1, 1).unwrap();

        for orientation in [0, 3] {
            for index in [1, 32] {
                assert_eq!(state.histories[orientation][index], 128);
                assert_eq!(state.histories[orientation + 1][index], 256);
                assert_eq!(state.histories[orientation + 2][index], 256);
            }
        }
    }

    #[test]
    fn slice_first_row_uses_unavailable_left_corner() {
        let mut state = ColourState::new(2, 2).unwrap();
        for plane in &mut state.planes {
            plane.fill(7);
        }
        state.available[2] = 1;
        state.available[3] = 1;

        update(&mut state, 1, 0, 1).unwrap();

        for plane in 0..3 {
            assert_eq!(state.banks[plane + 4][0], 128);
        }
    }
}
