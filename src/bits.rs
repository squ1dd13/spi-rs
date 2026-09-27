//! Read fields from a bounded byte array, most-significant bit first.
//!
//! [`BitReader`] is the ordinary cursor used for colour method codes, raw samples, and byte runs.
//! It checks a requested field before reading it. A failed ordinary read leaves that field's start
//! position unchanged; it does not undo earlier reads in the enclosing operation.
//!
//! [`ContextBitReader`] stores up to 32 unconsumed bits in the high end of an accumulator. Its
//! byte offset records loaded bytes, while [`ContextBitReader::logical_bit`] subtracts the
//! unconsumed bits. Refills replace the accumulator; entropy routines preserve and merge fragments
//! explicitly. A contextual read can consume a prefix before it fails. This reader is part of the
//! implemented syntax, not an arithmetic probability decoder.
//!
//! Colour creates both readers over `Record::payload` and adds the 14-byte header offset when it
//! reports positions. Alpha creates a contextual reader over `Record::data`, so its positions
//! already include the header. Predictive and palette colour paths synchronise the ordinary cursor
//! after successful contextual decoding; palette index validation can still fail after that
//! synchronisation. See the crate page's “Bit readers and positions” section for the alpha
//! hand-off.

use bitstream_io::{BigEndian, BitRead, BitReader as IoBitReader};
use thiserror::Error;

/// An invalid request to the ordinary bit reader.
///
/// Failed reads do not advance the logical bit position.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BitError {
    /// The requested bits extend beyond the input.
    #[error("requested bit range exceeds the input")]
    EndOfInput,
    /// A read width is zero or exceeds the destination type.
    #[error("bit count must be between 1 and 32")]
    InvalidCount,
}

/// Reads ordinary bounded syntax most-significant bit first.
///
/// The wrapper tracks an explicit bit position and checks bounds before delegating to
/// `bitstream-io`, which keeps failures non-consuming.
pub struct BitReader<'data> {
    /// Retains the original input for bounds checks.
    input: &'data [u8],
    /// Performs the underlying most-significant-bit-first extraction.
    inner: IoBitReader<&'data [u8], BigEndian>,
    /// The number of bits consumed from `input`.
    bit_position: usize,
}

impl<'data> BitReader<'data> {
    /// Creates an ordinary reader at the first bit of `input`.
    ///
    /// SPI bit fields are read from the most significant bit of each byte to the least significant
    /// bit.
    pub(crate) fn new(input: &'data [u8]) -> Self {
        Self {
            input,
            inner: IoBitReader::new(input),
            bit_position: 0,
        }
    }

    /// Returns the number of bits consumed by successful reads.
    ///
    /// Failed reads leave this position unchanged.
    pub(crate) fn bit_position(&self) -> usize {
        self.bit_position
    }

    /// Tests whether `count` bits remain in the input.
    ///
    /// The check treats an overflowing byte-to-bit length as unreadable and does not modify reader
    /// state. It does not reject a zero count; typed read methods perform that validation.
    pub(crate) fn can_read(&self, count: u32) -> bool {
        self.input.len().checked_mul(8).is_some_and(|total| {
            self.bit_position <= total && count as usize <= total - self.bit_position
        })
    }

    /// Reads between 1 and 32 bits as a `u32`.
    ///
    /// Bits are returned in stream order with the first bit becoming the most significant bit of
    /// the requested-width value. Invalid widths and truncated input return [`BitError`] without
    /// advancing the tracked position.
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

    /// Reads between 1 and 8 bits as a `u8`.
    ///
    /// This applies the narrower width bound before using [`BitReader::read_bits`].
    pub(crate) fn read_u8(&mut self, count: u32) -> Result<u8, BitError> {
        if count == 0 || count > 8 {
            return Err(BitError::InvalidCount);
        }
        self.read_bits(count)
            .map(|value| u8::try_from(value).expect("at most eight bits fit in u8"))
    }

    /// Reads between 1 and 16 bits as a `u16`.
    ///
    /// This applies the narrower width bound before using [`BitReader::read_bits`].
    pub(crate) fn read_u16(&mut self, count: u32) -> Result<u16, BitError> {
        if count == 0 || count > 16 {
            return Err(BitError::InvalidCount);
        }
        self.read_bits(count)
            .map(|value| u16::try_from(value).expect("at most sixteen bits fit in u16"))
    }

    /// Reads one bit as zero or one.
    ///
    /// End of input is reported as [`BitError::EndOfInput`].
    pub(crate) fn read_bit(&mut self) -> Result<u8, BitError> {
        self.read_u8(1)
    }

    /// Reads eight stream bits as one byte.
    ///
    /// The read can start at any bit position and remains most-significant-bit first.
    pub(crate) fn read_byte(&mut self) -> Result<u8, BitError> {
        self.read_u8(8)
    }
}

/// Implements the 32-bit accumulator used by contextual SPI entropy syntax.
///
/// Available bits are left-aligned in `accumulator`. Refills replace the accumulator with up to
/// four subsequent bytes rather than appending to it; entropy routines save and merge fragments
/// where the format requires that behaviour. Lookahead can inspect one bounded following byte
/// through [`ContextBitReader::peek_next_byte`].
#[derive(Debug, Clone)]
pub struct ContextBitReader<'data> {
    /// Input bytes whose start defines logical bit zero: colour payload or complete alpha record.
    pub(crate) data: &'data [u8],
    /// Caches `data.len()` for parity with the format's reader state.
    pub(crate) size: usize,
    /// The byte after the current accumulator load.
    pub(crate) byte_offset: usize,
    /// Available bits left-aligned in a 32-bit word.
    pub(crate) accumulator: u32,
    /// The number of valid high bits in `accumulator`.
    pub(crate) available: u32,
}

impl<'data> ContextBitReader<'data> {
    /// Creates an empty contextual accumulator over `data`.
    ///
    /// The caller must initialise or refill the reader before consuming values.
    pub(crate) fn new(data: &'data [u8]) -> Self {
        Self {
            data,
            size: data.len(),
            byte_offset: 0,
            accumulator: 0,
            available: 0,
        }
    }

    /// Initialises the accumulator at an arbitrary logical bit position.
    ///
    /// The operation requires four source bytes beginning at `bit / 8`. It loads them in
    /// big-endian order, discards the preceding bits within the first byte, and returns `false`
    /// without changing existing state when the position or lookahead is out of range.
    pub(crate) fn initialise_at_bit(&mut self, bit: usize) -> bool {
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

    /// Returns the logical bit position represented by the accumulator state.
    ///
    /// The calculation subtracts valid, unconsumed accumulator bits from the end of the latest
    /// byte load.
    pub(crate) fn logical_bit(&self) -> usize {
        self.byte_offset * 8 - self.available as usize
    }

    /// Replaces the accumulator with up to `count` subsequent bytes, left aligned.
    ///
    /// `count` must be from 1 through 4. A final short refill succeeds when at least one byte
    /// remains and records only the bits that were loaded. Returns `false` without changing state
    /// for count zero, count above four, or no remaining byte.
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

    /// Discards `count` bits across as many accumulator refills as required.
    ///
    /// A zero-width discard succeeds. The method returns `false` for widths above 32 or when the
    /// input ends before all requested bits are consumed; a partial failed operation can have
    /// advanced state.
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

    /// Reads up to 32 bits across accumulator refills.
    ///
    /// The first stream bit becomes the highest bit within the requested-width result. A
    /// zero-width request returns `Some(0)` without changing state. The method returns `None`
    /// without changing state for a width above 32; truncated input can fail after a partial
    /// advance.
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

    /// Reads up to eight contextual bits as a `u8`.
    ///
    /// A zero-width read returns zero. Wider requests and truncated input return `None`.
    pub(crate) fn read_value_u8(&mut self, count: u32) -> Option<u8> {
        if count > 8 {
            return None;
        }
        self.read_value(count)
            .map(|value| u8::try_from(value).expect("at most eight bits fit in u8"))
    }

    /// Returns the byte immediately after the current accumulator load.
    ///
    /// This bounded lookahead does not consume data and returns `None` at the end of the input.
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
    fn contextual_initialiser_tracks_unaligned_position() {
        let mut bits = ContextBitReader::new(&[0x80, 0x12, 0x34, 0x56]);
        assert!(bits.initialise_at_bit(1));
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
