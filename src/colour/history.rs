use crate::{DecodeError, colour::allocation};

/// Picture-wide padded planes and the boundary data used by predictive blocks.
pub struct ColourState {
    pub(crate) block_width: usize,
    pub(crate) total_block_rows: usize,
    pub(crate) stride: usize,
    pub(crate) planes: [Vec<u8>; 3],
    pub(crate) available: Vec<u8>,
    pub(crate) banks: [[u8; 33]; 9],
    pub(crate) histories: [[i16; 33]; 6],
}

impl ColourState {
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

    pub(crate) fn heap_bytes(&self) -> usize {
        self.planes.iter().map(Vec::capacity).sum::<usize>() + self.available.capacity()
    }

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
