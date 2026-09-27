//! Colour block decoding and picture-wide colour history.

mod blocks;
mod history;
mod predictive;
mod references;

use crate::{
    decoder::SliceGeometry,
    entropy::CoefficientHistory,
    error::{DecodeError, SyntaxError},
    format::Record,
};

pub use history::ColourState;

#[derive(Debug)]
pub struct ColourSlice {
    pub planes: [Vec<u8>; 3],
    pub stride: usize,
    pub end_bit: usize,
}

impl ColourSlice {
    pub(crate) fn heap_bytes(&self) -> usize {
        self.planes.iter().map(Vec::capacity).sum()
    }
}

#[derive(Clone, Copy)]
pub struct BlockPosition {
    pub geometry: SliceGeometry,
    pub x: usize,
    pub local_y: usize,
    pub global_y: usize,
}

#[derive(Clone, Copy)]
pub struct ColourMemoryBudget {
    pub base_live_bytes: usize,
    pub limit: usize,
    pub palette_capacity: usize,
}

impl ColourMemoryBudget {
    #[cfg(test)]
    const UNLIMITED: Self = Self {
        base_live_bytes: 0,
        limit: usize::MAX,
        palette_capacity: 0,
    };

    pub fn with_palette_capacity(self, palette_capacity: usize) -> Self {
        Self {
            palette_capacity,
            ..self
        }
    }

    pub fn check(
        self,
        planes: &[Vec<u8>; 3],
        histories: &Option<[CoefficientHistory; 3]>,
    ) -> Result<(), DecodeError> {
        self.check_with_additional(planes, histories, 0)
    }

    pub fn check_with_additional(
        self,
        planes: &[Vec<u8>; 3],
        histories: &Option<[CoefficientHistory; 3]>,
        additional: usize,
    ) -> Result<(), DecodeError> {
        let mut requested = self.base_live_bytes;
        for capacity in planes.iter().map(Vec::capacity).chain(
            histories
                .iter()
                .flat_map(|histories| histories.iter().map(CoefficientHistory::heap_bytes)),
        ) {
            requested = requested
                .checked_add(capacity)
                .ok_or(DecodeError::ResourceLimit {
                    resource: "decoder memory",
                    requested: usize::MAX,
                    limit: self.limit,
                })?;
        }
        requested = requested
            .checked_add(self.palette_capacity)
            .and_then(|requested| requested.checked_add(additional))
            .ok_or(DecodeError::ResourceLimit {
                resource: "decoder memory",
                requested: usize::MAX,
                limit: self.limit,
            })?;
        if requested > self.limit {
            return Err(DecodeError::ResourceLimit {
                resource: "decoder memory",
                requested,
                limit: self.limit,
            });
        }
        Ok(())
    }
}

pub fn decode_slice(
    record: &Record<'_>,
    geometry: SliceGeometry,
    state: &mut ColourState,
    budget: ColourMemoryBudget,
) -> Result<ColourSlice, DecodeError> {
    blocks::decode_slice(record, geometry, state, budget)
}

fn allocation(resource: &'static str) -> DecodeError {
    DecodeError::Allocation { resource }
}

fn invalid(detail: &'static str) -> DecodeError {
    DecodeError::Syntax(SyntaxError::Invalid {
        stream: "colour",
        detail,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        decoder::SliceGeometry,
        format::{Record, RecordHeader},
    };

    fn geometry() -> SliceGeometry {
        SliceGeometry {
            block_width: 1,
            block_rows: 1,
            absolute_block_row: 0,
            picture_width: 16,
            picture_height: 16,
        }
    }

    fn record_data(stream: &[u8]) -> Vec<u8> {
        let mut data = vec![0_u8; 14];
        data.extend_from_slice(stream);
        data
    }

    fn view(data: &[u8]) -> Record<'_> {
        Record {
            data,
            payload: &data[14..],
            payload_offset: 14,
            index: 0,
            stream_offset: 0,
            header: RecordHeader {
                opaque_byte_b: 0,
                index: 0,
                opaque_byte_a: 0,
                opaque_flag: false,
                opaque_bits: 0,
                opaque_byte_c: 0,
                opaque_byte_d: 0,
                opaque_byte_e: 0,
            },
        }
    }

    fn push_bits(bits: &mut Vec<bool>, value: u32, count: usize) {
        for shift in (0..count).rev() {
            bits.push((value >> shift) & 1 != 0);
        }
    }

    fn pack_bits(bits: &[bool]) -> Vec<u8> {
        let mut bytes = vec![0; bits.len().div_ceil(8)];
        for (index, bit) in bits.iter().enumerate() {
            if *bit {
                bytes[index / 8] |= 1 << (7 - index % 8);
            }
        }
        bytes
    }

    #[test]
    fn raw_block_starts_on_byte_boundary_and_keeps_planes() {
        // raw method 5 is 00 11; skip to the next byte, then 3 x 256 samples.
        let mut stream = vec![0_u8; 769];
        stream[0] = 0b0011_0000;
        stream[1..257].fill(12);
        stream[257..513].fill(34);
        stream[513..769].fill(56);
        let data = record_data(&stream);
        let record = view(&data);
        let mut state = ColourState::new(1, 1).unwrap();
        let decoded = decode_slice(
            &record,
            geometry(),
            &mut state,
            ColourMemoryBudget::UNLIMITED,
        )
        .unwrap();
        assert_eq!(decoded.planes[0][0], 12);
        assert_eq!(decoded.planes[1][0], 34);
        assert_eq!(decoded.planes[2][0], 56);
        assert_eq!(decoded.end_bit, 112 + 8 + 3 * 256 * 8);
    }

    #[test]
    fn residual_block_decodes_rle_and_signed_delta() {
        // Method 2 is followed by a zero prediction flag and three zero RLE planes.
        // This case checks the raw (non-predicted) residual plane path.
        let mut encoded = Vec::new();
        push_bits(&mut encoded, 0, 5); // residual method and prediction flag
        for _ in 0..3 {
            push_bits(&mut encoded, 0b100, 3);
            push_bits(&mut encoded, 0, 8);
            push_bits(&mut encoded, 1, 1);
            push_bits(&mut encoded, 0, 8);
            push_bits(&mut encoded, 0b100, 3);
        }
        let stream = pack_bits(&encoded);
        let data = record_data(&stream);
        let record = view(&data);
        let mut state = ColourState::new(1, 1).unwrap();
        let decoded = decode_slice(
            &record,
            geometry(),
            &mut state,
            ColourMemoryBudget::UNLIMITED,
        )
        .unwrap();
        assert_eq!(decoded.planes[0].len(), 256);
        assert_eq!(decoded.planes[0][0], 128);
    }

    #[test]
    fn palette_single_colour_expands_to_all_samples() {
        // Palette method 4 is 00 10, followed by marker 1, variant 0,
        // prefix width 0, then one RGB triplet.
        // 0010 method, then palette marker/variant/prefix, then packed colours.
        let data = record_data(&[0b0010_1000, 0x08, 0xd1, 0x59, 0xc0]);
        let record = view(&data);
        let mut state = ColourState::new(1, 1).unwrap();
        let decoded = decode_slice(
            &record,
            geometry(),
            &mut state,
            ColourMemoryBudget::UNLIMITED,
        )
        .unwrap();
        assert_eq!(decoded.planes[0], vec![0x23; 256]);
        assert_eq!(decoded.planes[1], vec![0x45; 256]);
        assert_eq!(decoded.planes[2], vec![0x67; 256]);
    }

    #[test]
    fn palette_growth_obeys_the_live_memory_limit() {
        let single = record_data(&[0b0010_1000, 0x08, 0xd1, 0x59, 0xc0]);
        let mut limited_state = ColourState::new(1, 1).unwrap();
        let base_live_bytes = limited_state.heap_bytes();
        let too_small = ColourMemoryBudget {
            base_live_bytes,
            limit: base_live_bytes + 3 * 256 + 2,
            palette_capacity: 0,
        };
        let small_error =
            decode_slice(&view(&single), geometry(), &mut limited_state, too_small).unwrap_err();
        assert_eq!(small_error.kind(), crate::ErrorKind::ResourceLimit);

        let mut fitting_state = ColourState::new(1, 1).unwrap();
        let fits = ColourMemoryBudget {
            base_live_bytes: fitting_state.heap_bytes(),
            limit: fitting_state.heap_bytes() + 3 * 256 + 3,
            palette_capacity: 0,
        };
        assert!(decode_slice(&view(&single), geometry(), &mut fitting_state, fits).is_ok());

        let mut bits = Vec::new();
        push_bits(&mut bits, 0b0010, 4);
        push_bits(&mut bits, 1, 1);
        push_bits(&mut bits, 1, 1);
        push_bits(&mut bits, 15, 4);
        push_bits(&mut bits, (1 << 15) - 1, 15);
        let large = record_data(&pack_bits(&bits));
        let mut large_state = ColourState::new(1, 1).unwrap();
        let budget = ColourMemoryBudget {
            base_live_bytes: large_state.heap_bytes(),
            limit: large_state.heap_bytes() + 10_000,
            palette_capacity: 0,
        };
        let large_error =
            decode_slice(&view(&large), geometry(), &mut large_state, budget).unwrap_err();
        assert_eq!(large_error.kind(), crate::ErrorKind::ResourceLimit);
    }

    #[test]
    fn raw_picture_history_is_used_by_next_slice() {
        let first_data = record_data(&{
            let mut bytes = vec![0_u8; 769];
            bytes[0] = 0b0011_0000;
            bytes[1..257].fill(91);
            bytes[257..513].fill(92);
            bytes[513..769].fill(93);
            bytes
        });
        let first = view(&first_data);
        let second_data = record_data(&[0b1000_0000]);
        let second = view(&second_data);
        let mut state = ColourState::new(1, 2).unwrap();
        let mut first_geometry = geometry();
        first_geometry.block_rows = 1;
        decode_slice(
            &first,
            first_geometry,
            &mut state,
            ColourMemoryBudget::UNLIMITED,
        )
        .unwrap();
        assert_eq!(state.planes[0][0], 91);
        let second_geometry = SliceGeometry {
            absolute_block_row: 1,
            ..first_geometry
        };
        let decoded = decode_slice(
            &second,
            second_geometry,
            &mut state,
            ColourMemoryBudget::UNLIMITED,
        )
        .unwrap();
        assert_eq!(decoded.planes[0][0], 91);
    }
}
