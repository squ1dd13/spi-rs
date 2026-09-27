use bitstream_io::{BigEndian, BitRead, BitReader as IoBitReader};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum BitError {
    #[error("requested bit range exceeds the input")]
    EndOfInput,
    #[error("bit count must be between 1 and 32")]
    InvalidCount,
}

/// A checked, MSB-first reader for ordinary bounded syntax.
pub(crate) struct BitReader<'a> {
    input: &'a [u8],
    inner: IoBitReader<&'a [u8], BigEndian>,
    bit_position: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            inner: IoBitReader::new(input),
            bit_position: 0,
        }
    }

    pub(crate) fn bit_position(&self) -> usize {
        self.bit_position
    }

    pub(crate) fn can_read(&self, count: u32) -> bool {
        self.input.len().checked_mul(8).is_some_and(|total| {
            self.bit_position <= total && count as usize <= total - self.bit_position
        })
    }

    pub(crate) fn read_bits(&mut self, count: u32) -> Result<u32, BitError> {
        if count == 0 || count > 32 {
            return Err(BitError::InvalidCount);
        }
        if !self.can_read(count) {
            return Err(BitError::EndOfInput);
        }
        let value = self
            .inner
            .read_var::<u32>(count)
            .map_err(|_| BitError::EndOfInput)?;
        self.bit_position += count as usize;
        Ok(value)
    }

    pub(crate) fn read_u8(&mut self, count: u32) -> Result<u8, BitError> {
        if count == 0 || count > 8 {
            return Err(BitError::InvalidCount);
        }
        self.read_bits(count)
            .map(|value| u8::try_from(value).expect("at most eight bits fit in u8"))
    }

    pub(crate) fn read_u16(&mut self, count: u32) -> Result<u16, BitError> {
        if count == 0 || count > 16 {
            return Err(BitError::InvalidCount);
        }
        self.read_bits(count)
            .map(|value| u16::try_from(value).expect("at most sixteen bits fit in u16"))
    }

    pub(crate) fn read_bit(&mut self) -> Result<u8, BitError> {
        #[allow(clippy::cast_possible_truncation)]
        self.read_bits(1).map(|v| v as u8)
    }

    pub(crate) fn read_byte(&mut self) -> Result<u8, BitError> {
        #[allow(clippy::cast_possible_truncation)]
        self.read_bits(8).map(|v| v as u8)
    }
}

/// Context reader that matches the format's 32-bit accumulator refill rules.
/// Lookahead may inspect a bounded following byte through `peek_next_byte`.
#[derive(Debug, Clone)]
pub(crate) struct ContextBitReader<'a> {
    pub(crate) data: &'a [u8],
    pub(crate) size: usize,
    pub(crate) byte_offset: usize,
    pub(crate) accumulator: u32,
    pub(crate) available: u32,
}

impl<'a> ContextBitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            size: data.len(),
            byte_offset: 0,
            accumulator: 0,
            available: 0,
        }
    }

    pub(crate) fn initialize_at_bit(&mut self, bit: usize) -> bool {
        if self.size > usize::MAX / 8
            || bit > self.size * 8
            || bit / 8 > self.size
            || self.size - bit / 8 < 4
        {
            return false;
        }
        let source_offset = bit / 8;
        let shift = u32::try_from(bit & 7).expect("bit offset fits in u32");
        let mut word = 0_u32;
        for byte in &self.data[source_offset..source_offset + 4] {
            word = (word << 8) | u32::from(*byte);
        }
        self.accumulator = word << shift;
        self.available = 32 - shift;
        self.byte_offset = source_offset + 4;
        true
    }

    pub(crate) fn logical_bit(&self) -> usize {
        self.byte_offset * 8 - self.available as usize
    }

    /// Replace the accumulator with up to `count` subsequent bytes, left aligned.
    pub(crate) fn refill(&mut self, count: u32) -> bool {
        if count == 0 || count > 4 || self.byte_offset >= self.size {
            return false;
        }
        let take = usize::try_from(count)
            .expect("refill count is at most four")
            .min(self.size - self.byte_offset);
        let take_bits = u32::try_from(take).expect("refill count is at most four");
        let mut word = 0_u32;
        for byte in &self.data[self.byte_offset..self.byte_offset + take] {
            word = (word << 8) | u32::from(*byte);
        }
        self.accumulator = word << (8 * (4 - take_bits));
        self.available = take_bits * 8;
        self.byte_offset += take;
        true
    }

    pub(crate) fn read_bits(&mut self, count: u32) -> bool {
        if count > 32 {
            return false;
        }
        let mut remaining = count;
        while remaining != 0 {
            if self.available == 0 && !self.refill(4) {
                return false;
            }
            let take = remaining.min(self.available);
            self.accumulator = if take == 32 {
                0
            } else {
                self.accumulator << take
            };
            self.available -= take;
            remaining -= take;
        }
        true
    }

    pub(crate) fn read_value(&mut self, count: u32) -> Option<u32> {
        if count > 32 {
            return None;
        }
        let mut remaining = count;
        let mut result = 0_u32;
        while remaining != 0 {
            if self.available == 0 && !self.refill(4) {
                return None;
            }
            let take = remaining.min(self.available);
            let fragment = self.accumulator >> (32 - take);
            result = if take == 32 {
                fragment
            } else {
                (result << take) | fragment
            };
            self.accumulator = if take == 32 {
                0
            } else {
                self.accumulator << take
            };
            self.available -= take;
            remaining -= take;
        }
        Some(result)
    }

    pub(crate) fn read_value_u8(&mut self, count: u32) -> Option<u8> {
        if count > 8 {
            return None;
        }
        self.read_value(count)
            .map(|value| u8::try_from(value).expect("at most eight bits fit in u8"))
    }

    pub(crate) fn peek_next_byte(&self) -> Option<u8> {
        self.data.get(self.byte_offset).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn ordinary_reader_is_checked_and_msb_first() {
        let mut bits = BitReader::new(&[0b1011_0010]);
        assert_eq!(bits.read_bits(3), Ok(0b101));
        assert_eq!(bits.bit_position(), 3);
        assert_eq!(bits.read_bits(6), Err(BitError::EndOfInput));
        assert_eq!(bits.bit_position(), 3);
    }

    #[test]
    fn contextual_refill_is_left_aligned_and_clamped() {
        let mut bits = ContextBitReader::new(&[0x12, 0x34, 0x56]);
        assert!(bits.refill(4));
        assert_eq!(bits.accumulator, 0x1234_5600);
        assert_eq!(bits.available, 24);
        assert_eq!(bits.byte_offset, 3);
        assert!(!bits.refill(1));
    }

    #[test]
    fn contextual_initializer_tracks_unaligned_position() {
        let mut bits = ContextBitReader::new(&[0x80, 0x12, 0x34, 0x56]);
        assert!(bits.initialize_at_bit(1));
        assert_eq!(bits.accumulator, 0x0024_68ac);
        assert_eq!(bits.available, 31);
        assert_eq!(bits.logical_bit(), 1);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn small_ordinary_reader_inputs_match_msb_reference(
            input in prop::collection::vec(any::<u8>(), 0..24),
            count in 0_u32..33,
        ) {
            let mut bits = BitReader::new(&input);
            let result = bits.read_bits(count);
            if count == 0 {
                prop_assert_eq!(result, Err(BitError::InvalidCount));
            } else if input.len().saturating_mul(8) < count as usize {
                prop_assert_eq!(result, Err(BitError::EndOfInput));
            } else {
                let mut expected = 0_u32;
                for position in 0..count as usize {
                    expected = (expected << 1)
                        | u32::from((input[position / 8] >> (7 - (position & 7))) & 1);
                }
                prop_assert_eq!(result, Ok(expected));
                prop_assert_eq!(bits.bit_position(), count as usize);
            }
        }
    }
}
