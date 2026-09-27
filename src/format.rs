//! Locate the sections and records in a complete SPI container.
//!
//! Parsing here produces borrowed byte slices; it does not reconstruct samples. The outer two
//! lengths are little-endian and exclude their own four bytes. Picture fields and record lengths
//! are big-endian. A record length includes its six framing bytes. [`parse_container`] requires
//! exact outer consumption, [`parse_picture`] permits extra picture-section bytes, and
//! [`parse_records`] requires complete framed records.
//!
//! The crate page's “Container and records” section gives the byte and bit tables. The [`Record`]
//! view keeps both the complete record and its payload so codec readers can use the required input
//! origin. Colour starts at payload bit zero; diagnostics add `payload_offset * 8`. Alpha reads
//! the complete record. `stream_offset` instead locates a record inside the record section and is
//! not a bit-reader origin.
//!
//! [`parse_records`] checks markers, lengths, and the terminal zero bits. It does not check that
//! record indices are consecutive or that records cover the picture; [`crate::decoder`] performs
//! those checks after it derives the padded geometry. Opaque fields are retained with their widths
//! but have no established meaning here.

use std::io::Cursor;

use binrw::{BinRead, Endian};
use thiserror::Error;

/// Malformed SPI framing and fixed headers.
///
/// These errors describe byte-level container defects. Codec syntax errors inside a record use the
/// higher-level decoder error types.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum FormatError {
    /// The first section length is incomplete.
    #[error("container is missing a length field")]
    TruncatedContainer,
    /// A section length overflows or extends beyond the input.
    #[error("container section length exceeds the input")]
    InvalidContainerLength,
    /// Bytes remain after the declared record section.
    #[error("container has trailing bytes")]
    TrailingContainerBytes,
    /// The picture section has fewer than 16 bytes.
    #[error("picture header is truncated")]
    TruncatedPictureHeader,
    /// The picture section does not start with `AA 01` or cannot be read.
    #[error("picture header has an invalid identifier")]
    InvalidPictureIdentifier,
    /// The coded width or height is zero.
    #[error("picture dimensions must be non-zero")]
    InvalidPictureDimensions,
    /// A record marker, syntax header, or reserved terminal bit is invalid.
    #[error("record {record} has an invalid header")]
    InvalidRecordHeader {
        /// The zero-based record number in the record section.
        record: usize,
    },
    /// A record length is too small, overflows, or exceeds the section.
    #[error("record {record} declares an invalid length")]
    InvalidRecordLength {
        /// The zero-based record number in the record section.
        record: usize,
    },
    /// The number of records exceeds the caller's bound.
    #[error("record count exceeds the configured limit")]
    TooManyRecords,
    /// Fewer than six bytes remain for a final record header.
    #[error("record payload ends with an incomplete record")]
    TruncatedRecord,
    /// Storage for the record views could not be reserved.
    #[error("could not reserve memory for record descriptors")]
    AllocationFailure,
}

/// Borrows the two payload sections of an SPI container.
///
/// The slices exclude their four-byte little-endian length fields. `records` still contains each
/// record's own framing header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Container<'data> {
    /// Borrows the picture-header section.
    pub picture: &'data [u8],
    /// Borrows the concatenated record section.
    pub records: &'data [u8],
}

/// Parses the two length-prefixed sections of an SPI container.
///
/// The parser requires both four-byte little-endian lengths and exact consumption of `input`. It
/// borrows the section payloads and does not inspect their contents.
///
/// # Errors
///
/// Returns [`FormatError::TruncatedContainer`] when the first length is absent,
/// [`FormatError::InvalidContainerLength`] when either section is incomplete or its bounds
/// overflow, and [`FormatError::TrailingContainerBytes`] when data follows the record section.
pub fn parse_container(input: &[u8]) -> Result<Container<'_>, FormatError> {
    if input.len() < 4 {
        return Err(FormatError::TruncatedContainer);
    }
    let picture_len = u32::from_le_bytes(input[..4].try_into().expect("length checked")) as usize;
    let picture_end = 4_usize
        .checked_add(picture_len)
        .ok_or(FormatError::InvalidContainerLength)?;
    if picture_end > input.len() || input.len() - picture_end < 4 {
        return Err(FormatError::InvalidContainerLength);
    }
    let records_len = u32::from_le_bytes(
        input[picture_end..picture_end + 4]
            .try_into()
            .expect("length checked"),
    ) as usize;
    let records_start = picture_end + 4;
    let records_end = records_start
        .checked_add(records_len)
        .ok_or(FormatError::InvalidContainerLength)?;
    if records_end > input.len() {
        return Err(FormatError::InvalidContainerLength);
    }
    if records_end != input.len() {
        return Err(FormatError::TrailingContainerBytes);
    }
    Ok(Container {
        picture: &input[4..picture_end],
        records: &input[records_start..records_end],
    })
}

/// The understood fields from the fixed picture header.
///
/// SPI stores dimensions as big-endian 16-bit integers. Unknown ancillary fields are intentionally
/// not exposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PictureInfo {
    /// The visible picture width in pixels.
    pub width: u16,
    /// The visible picture height in pixels.
    pub height: u16,
    /// The coded colour mode.
    pub colour_mode: u8,
    /// The coded output selector, which determines whether alpha is present.
    pub selector: u8,
}

/// Maps the fixed, big-endian picture header before validation.
///
/// Fields prefixed with an underscore have unknown semantics and are retained only to place the
/// understood fields at their correct offsets.
#[derive(BinRead)]
#[br(magic = b"\xaa\x01")]
struct PictureFields {
    /// The unknown four-byte word after the identifier.
    _record_word: u32,
    /// The coded colour mode.
    colour_mode: u8,
    /// The first unknown two-byte ancillary value.
    _ancillary_a: u16,
    /// The second unknown two-byte ancillary value.
    _ancillary_b: u16,
    /// The visible width.
    width: u16,
    /// The visible height.
    height: u16,
    /// The output selector.
    selector: u8,
}

/// Parses the understood part of a picture header.
///
/// The header must contain at least 16 bytes, start with `AA 01`, and declare non-zero dimensions.
/// Extra picture-section bytes are permitted because their format is not yet understood.
///
/// # Errors
///
/// Returns a truncated-header, invalid-identifier, or invalid-dimensions [`FormatError`] as
/// applicable.
pub fn parse_picture(input: &[u8]) -> Result<PictureInfo, FormatError> {
    if input.len() < 16 {
        return Err(FormatError::TruncatedPictureHeader);
    }
    let fields = PictureFields::read_options(&mut Cursor::new(input), Endian::Big, ())
        .map_err(|_| FormatError::InvalidPictureIdentifier)?;
    if fields.width == 0 || fields.height == 0 {
        return Err(FormatError::InvalidPictureDimensions);
    }
    Ok(PictureInfo {
        width: fields.width,
        height: fields.height,
        colour_mode: fields.colour_mode,
        selector: fields.selector,
    })
}

/// The bit-packed syntax fields before a record's codec payload.
///
/// Only `index` currently has known decoder semantics. The other values are exposed to keep
/// reverse-engineering observations separate from assumptions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordHeader {
    /// The first opaque byte in bitstream order.
    pub opaque_byte_b: u8,
    /// The zero-based vertical record index.
    pub index: u16,
    /// The second opaque byte in bitstream order.
    pub opaque_byte_a: u8,
    /// The opaque one-bit flag.
    pub opaque_flag: bool,
    /// The opaque two-bit value.
    pub opaque_bits: u8,
    /// The third opaque byte in bitstream order.
    pub opaque_byte_c: u8,
    /// The fourth opaque byte in bitstream order.
    pub opaque_byte_d: u8,
    /// The fifth opaque byte in bitstream order.
    pub opaque_byte_e: u8,
}

/// Borrows one framed record and its codec payload.
///
/// A record consists of a six-byte `AA 02` framing header, an eight-byte bit-packed syntax header,
/// and the colour and optional alpha streams. Offsets permit codec errors to refer to the complete
/// record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Record<'data> {
    /// The complete record, including its 6-byte framing header.
    pub data: &'data [u8],
    /// Codec payload after the 6-byte framing and 8-byte syntax header.
    pub payload: &'data [u8],
    /// The payload's byte offset from the start of `data`.
    pub payload_offset: usize,
    /// The record index copied from `header`.
    pub index: u16,
    /// The record's byte offset in the complete record section.
    pub stream_offset: usize,
    /// The parsed bit-packed syntax header.
    pub header: RecordHeader,
}

/// Parses the eight-byte bit-packed syntax header of one record.
///
/// Fields are read most-significant bit first. Four reserved terminal bits and the following end
/// flag must all be zero; an incomplete field or non-zero reserved value makes the entire header
/// invalid.
///
/// # Errors
///
/// Returns [`FormatError::InvalidRecordHeader`] tagged with `record_number` when any bit is
/// missing or a reserved terminal value is set.
fn parse_record_header(
    header_bytes: &[u8],
    record_number: usize,
) -> Result<RecordHeader, FormatError> {
    let mut bits = crate::bits::BitReader::new(header_bytes);
    let (header, terminal, end_flag) = (|| -> Result<_, crate::bits::BitError> {
        let header = RecordHeader {
            opaque_byte_b: bits.read_byte()?,
            index: bits.read_u16(16)?,
            opaque_byte_a: bits.read_byte()?,
            opaque_flag: bits.read_bit()? != 0,
            opaque_bits: bits.read_u8(2)?,
            opaque_byte_c: bits.read_byte()?,
            opaque_byte_d: bits.read_byte()?,
            opaque_byte_e: bits.read_byte()?,
        };
        Ok((header, bits.read_u8(4)?, bits.read_bit()? != 0))
    })()
    .map_err(|_| FormatError::InvalidRecordHeader {
        record: record_number,
    })?;
    if terminal != 0 || end_flag {
        return Err(FormatError::InvalidRecordHeader {
            record: record_number,
        });
    }
    Ok(header)
}

/// Splits a record section into checked, borrowed record views.
///
/// Each record must start with `AA 02`; its big-endian 32-bit length includes the six-byte framing
/// header and must be at least 14 bytes to include the syntax header. Parsing continues until the
/// section is consumed exactly. This function validates framing only and does not require record
/// indices to be consecutive.
///
/// `max_records` is both a resource bound and a format-context bound supplied by the decoder.
/// Storage reservation is limited by the smaller of that value and the number of minimum-size
/// records that can fit in `payload`.
///
/// # Errors
///
/// Returns [`FormatError`] for an invalid marker, length, syntax header, incomplete final framing
/// header, excess record count, or descriptor allocation failure.
pub fn parse_records(payload: &[u8], max_records: usize) -> Result<Vec<Record<'_>>, FormatError> {
    let mut records = Vec::new();
    records
        .try_reserve(max_records.min(payload.len() / 14))
        .map_err(|_| FormatError::AllocationFailure)?;
    let mut offset = 0_usize;
    while offset < payload.len() {
        if records.len() == max_records {
            return Err(FormatError::TooManyRecords);
        }
        if payload.len() - offset < 6 {
            return Err(FormatError::TruncatedRecord);
        }
        if payload[offset] != 0xaa || payload[offset + 1] != 2 {
            return Err(FormatError::InvalidRecordHeader {
                record: records.len(),
            });
        }
        let length = u32::from_be_bytes(
            payload[offset + 2..offset + 6]
                .try_into()
                .expect("record header checked"),
        ) as usize;
        if length < 6 {
            return Err(FormatError::InvalidRecordLength {
                record: records.len(),
            });
        }
        let end = offset
            .checked_add(length)
            .ok_or(FormatError::InvalidRecordLength {
                record: records.len(),
            })?;
        if end > payload.len() {
            return Err(FormatError::InvalidRecordLength {
                record: records.len(),
            });
        }
        if length < 14 {
            return Err(FormatError::InvalidRecordLength {
                record: records.len(),
            });
        }
        let header = parse_record_header(&payload[offset + 6..offset + 14], records.len())?;
        records.push(Record {
            data: &payload[offset..end],
            payload: &payload[offset + 14..end],
            payload_offset: 14,
            index: header.index,
            stream_offset: offset,
            header,
        });
        offset = end;
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn container_borrows_both_length_prefixed_sections() {
        let input = [2, 0, 0, 0, 0xaa, 1, 0, 0, 0, 0];
        let container = parse_container(&input).unwrap();
        assert_eq!(container.picture, &[0xaa, 1]);
        assert!(container.records.is_empty());
    }

    #[test]
    fn record_parser_keeps_declared_record_boundary() {
        let mut bytes = vec![0xaa, 2, 0, 0, 0, 14];
        // One-byte opaque fields and zero terminal/end bits complete the 8-byte syntax header.
        bytes.extend_from_slice(&[0x12, 0x00, 0x01, 0, 0, 0, 0, 0]);
        let records = parse_records(&bytes, 1).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].stream_offset, 0);
        assert_eq!(records[0].index, 1);
        assert_eq!(records[0].data, bytes);
        assert!(records[0].payload.is_empty());
        assert_eq!(records[0].payload_offset, 14);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn bounded_container_inputs_never_panic(input in prop::collection::vec(any::<u8>(), 0..64)) {
            if let Ok(container) = parse_container(&input) {
                prop_assert!(container.picture.len() + container.records.len() + 8 <= input.len());
                prop_assert!(container.picture.as_ptr() >= input.as_ptr());
                prop_assert!(container.records.as_ptr() >= input.as_ptr());
            }
        }
    }
}
