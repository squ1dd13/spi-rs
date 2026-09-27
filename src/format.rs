use std::io::Cursor;

use binrw::{BinRead, Endian};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum FormatError {
    #[error("container is missing a length field")]
    TruncatedContainer,
    #[error("container section length exceeds the input")]
    InvalidContainerLength,
    #[error("container has trailing bytes")]
    TrailingContainerBytes,
    #[error("picture header is truncated")]
    TruncatedPictureHeader,
    #[error("picture header has an invalid identifier")]
    InvalidPictureIdentifier,
    #[error("picture dimensions must be non-zero")]
    InvalidPictureDimensions,
    #[error("record {record} has an invalid header")]
    InvalidRecordHeader { record: usize },
    #[error("record {record} declares an invalid length")]
    InvalidRecordLength { record: usize },
    #[error("record count exceeds the configured limit")]
    TooManyRecords,
    #[error("record payload ends with an incomplete record")]
    TruncatedRecord,
    #[error("could not reserve memory for record descriptors")]
    AllocationFailure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Container<'a> {
    pub picture: &'a [u8],
    pub records: &'a [u8],
}

pub(crate) fn parse_container(input: &[u8]) -> Result<Container<'_>, FormatError> {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PictureInfo {
    pub width: u16,
    pub height: u16,
    pub color_mode: u8,
    pub selector: u8,
}

#[derive(BinRead)]
#[br(magic = b"\xaa\x01")]
struct PictureFields {
    _record_word: u32,
    color_mode: u8,
    _ancillary_a: u16,
    _ancillary_b: u16,
    width: u16,
    height: u16,
    selector: u8,
}

pub(crate) fn parse_picture(input: &[u8]) -> Result<PictureInfo, FormatError> {
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
        color_mode: fields.color_mode,
        selector: fields.selector,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RecordHeader {
    pub opaque_byte_b: u8,
    pub index: u16,
    pub opaque_byte_a: u8,
    pub opaque_flag: bool,
    pub opaque_bits: u8,
    pub opaque_byte_c: u8,
    pub opaque_byte_d: u8,
    pub opaque_byte_e: u8,
    pub terminal: u8,
    pub end: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Record<'a> {
    /// The complete record, including its 6-byte framing header.
    pub data: &'a [u8],
    /// Codec payload after the 6-byte framing and 8-byte syntax header.
    pub payload: &'a [u8],
    pub payload_offset: usize,
    pub index: u16,
    pub stream_offset: usize,
    pub header: RecordHeader,
}

pub(crate) fn parse_records(
    payload: &[u8],
    max_records: usize,
) -> Result<Vec<Record<'_>>, FormatError> {
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
        let header_bytes = &payload[offset + 6..offset + 14];
        let bits = crate::bits::BitReader::new(header_bytes);
        let mut bits = bits;
        let opaque_byte_b = bits
            .read_byte()
            .map_err(|_| FormatError::InvalidRecordHeader {
                record: records.len(),
            })?;
        let index = bits
            .read_bits(16)
            .map_err(|_| FormatError::InvalidRecordHeader {
                record: records.len(),
            })? as u16;
        let opaque_byte_a = bits
            .read_byte()
            .map_err(|_| FormatError::InvalidRecordHeader {
                record: records.len(),
            })?;
        let opaque_flag = bits
            .read_bit()
            .map_err(|_| FormatError::InvalidRecordHeader {
                record: records.len(),
            })?
            != 0;
        let opaque_bits = bits
            .read_bits(2)
            .map_err(|_| FormatError::InvalidRecordHeader {
                record: records.len(),
            })? as u8;
        let opaque_byte_c = bits
            .read_byte()
            .map_err(|_| FormatError::InvalidRecordHeader {
                record: records.len(),
            })?;
        let opaque_byte_d = bits
            .read_byte()
            .map_err(|_| FormatError::InvalidRecordHeader {
                record: records.len(),
            })?;
        let opaque_byte_e = bits
            .read_byte()
            .map_err(|_| FormatError::InvalidRecordHeader {
                record: records.len(),
            })?;
        let terminal = bits
            .read_bits(4)
            .map_err(|_| FormatError::InvalidRecordHeader {
                record: records.len(),
            })? as u8;
        let end_flag = bits
            .read_bit()
            .map_err(|_| FormatError::InvalidRecordHeader {
                record: records.len(),
            })?
            != 0;
        if terminal != 0 || end_flag {
            return Err(FormatError::InvalidRecordHeader {
                record: records.len(),
            });
        }
        records.push(Record {
            data: &payload[offset..end],
            payload: &payload[offset + 14..end],
            payload_offset: 14,
            index,
            stream_offset: offset,
            header: RecordHeader {
                opaque_byte_b,
                index,
                opaque_byte_a,
                opaque_flag,
                opaque_bits,
                opaque_byte_c,
                opaque_byte_d,
                opaque_byte_e,
                terminal,
                end: end_flag,
            },
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
