//! Retain reconstructed colour samples for the whole picture.
//!
//! [`ColourState`] contains three byte planes in G, B, R order. Its stride is the padded width in samples, and its height is the padded number of block rows times 16. These planes supply spatial and displacement-based copies across record boundaries. Availability flags supply edge preparation, but do not restrict motion reads.
//!
//! The byte banks hold the current prepared top and left edges. [`references`](../references/index.html) fills six transformed banks from those byte values before residual or predictive blocks. The transformed banks contain top L, biased U, biased V, then left L, biased U, biased V. They are sample data, distinct from the record-local mode symbols in [`crate::entropy::CoefficientHistory`].
//!
//! This object lasts for one picture, but that lifetime does not imply that all samples remain usable as top references across records. Preparation on each record's first local row overrides the relevant edges. The crate page gives the state table and conversion equations.

use crate::{DecodeError, colour::allocation};

/// Picture-wide G, B, R samples and prepared reference arrays.
///
/// The three planes are row-major bytes with padded stride and padded height. They start at zero. Availability flags start at zero and are set before a block reconstructs; reference preparation uses them, but motion copying does not. This state persists through every record of one decode.
///
/// Byte bank indices 0,1,2 are top G,B,R, and 4,5,6 are left G,B,R. Each used bank has corner at 0 and 32 neighbours or extension samples at 1..33. Indices 3 and 7 remain unused zero banks; index 8 starts at 128 and is not used by the current edge producer. The six `histories` arrays are transformed sample banks, not entropy mode history. Preparation before residual and predictive methods refreshes them.
pub struct ColourState {
    /// The padded picture width in blocks.
    pub(crate) block_width: usize,
    /// The padded picture height in blocks.
    pub(crate) total_block_rows: usize,
    /// The padded plane width in samples.
    pub(crate) stride: usize,
    /// Complete reconstructed green, blue, and red picture planes.
    pub(crate) planes: [Vec<u8>; 3],
    /// Edge-preparation availability per picture block; motion copies do not check it.
    pub(crate) available: Vec<u8>,
    /// Top G/B/R at 0..3 and left G/B/R at 4..7; each used bank includes a corner and 32 samples.
    pub(crate) banks: [[u8; 33]; 9],
    /// Six transformed sample arrays: top L/U+256/V+256, then left L/U+256/V+256; corner at 0, neighbours at 1..33. Produced by reference preparation and consumed by predictive colour.
    pub(crate) histories: [[i16; 33]; 6],
}

impl ColourState {
    /// Allocates and initialises persistent colour state for one picture.
    ///
    /// Plane dimensions are padded to whole 16-pixel blocks. Colour planes and availability start at zero; unused byte bank 8 starts at 128; the six transformed arrays start at zero. Empty or overflowing geometry and failed reservations return [`DecodeError`].
    pub(crate) fn new(block_width: usize, total_block_rows: usize) -> Result<Self, DecodeError> {
        if block_width == 0 || total_block_rows == 0 {
            return Err(crate::colour::invalid("empty picture geometry"));
        }
        let stride = block_width
            .checked_mul(16)
            .ok_or_else(|| allocation("colour history"))?;
        let height = total_block_rows
            .checked_mul(16)
            .ok_or_else(|| allocation("colour history"))?;
        let len = stride
            .checked_mul(height)
            .ok_or_else(|| allocation("colour history"))?;
        let mut planes = [Vec::new(), Vec::new(), Vec::new()];
        for plane in &mut planes {
            plane
                .try_reserve_exact(len)
                .map_err(|_| allocation("colour history"))?;
            plane.resize(len, 0);
        }
        let blocks = block_width
            .checked_mul(total_block_rows)
            .ok_or_else(|| allocation("colour block state"))?;
        let mut available = Vec::new();
        available
            .try_reserve_exact(blocks)
            .map_err(|_| allocation("colour block state"))?;
        available.resize(blocks, 0);
        let mut banks = [[0_u8; 33]; 9];
        banks[8].fill(128);
        Ok(Self {
            block_width,
            total_block_rows,
            stride,
            planes,
            available,
            banks,
            histories: [[0; 33]; 6],
        })
    }

    /// Returns heap bytes reserved by persistent colour vectors.
    ///
    /// Fixed-size banks and histories live inline and are covered by the decoder's conservative context allowance.
    pub(crate) fn heap_bytes(&self) -> usize {
        self.planes.iter().map(Vec::capacity).sum::<usize>() + self.available.capacity()
    }

    /// Converts a slice-local block row into a checked picture row.
    ///
    /// Addition overflow or a result beyond `total_block_rows` is reported as invalid colour syntax.
    pub(crate) fn global_row(&self, local_row: usize, origin: usize) -> Result<usize, DecodeError> {
        let y = origin
            .checked_add(local_row)
            .ok_or_else(|| crate::colour::invalid("slice row overflow"))?;
        if y >= self.total_block_rows {
            return Err(crate::colour::invalid("slice exceeds picture rows"));
        }
        Ok(y)
    }
}
