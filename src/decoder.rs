//! Picture sequencing, resource accounting, and output composition.

use crate::{DecodeError, DecodeLimits, ErrorContext, ImageInfo, error::ParseError, format};

/// Geometry shared by one record's colour and alpha streams. Block coordinates
/// use the padded 16-pixel grid; visible dimensions clip the last record.
#[derive(Debug, Clone, Copy)]
pub struct SliceGeometry {
    pub block_width: usize,
    pub block_rows: usize,
    pub absolute_block_row: usize,
    pub picture_width: usize,
    pub picture_height: usize,
}

impl SliceGeometry {
    pub(crate) fn visible_height(self) -> usize {
        self.picture_height
            .saturating_sub(self.absolute_block_row * 16)
            .min(self.block_rows * 16)
    }

    pub(crate) fn pixel_row(self) -> usize {
        self.absolute_block_row * 16
    }
}

fn checked_mul(
    a: usize,
    b: usize,
    resource: &'static str,
    limit: usize,
) -> Result<usize, DecodeError> {
    let result = a.checked_mul(b).ok_or(DecodeError::ResourceLimit {
        resource,
        requested: usize::MAX,
        limit,
    })?;
    Ok(result)
}

fn checked_add(
    a: usize,
    b: usize,
    resource: &'static str,
    limit: usize,
) -> Result<usize, DecodeError> {
    a.checked_add(b).ok_or(DecodeError::ResourceLimit {
        resource,
        requested: usize::MAX,
        limit,
    })
}

/// Checks a conservative upper bound before any large decoder allocation. The
/// block allowance covers reference banks, availability masks, and coefficient
/// contexts in addition to the explicitly sized output and plane buffers.
fn check_memory(
    dimensions: (usize, usize),
    blocks_x: usize,
    blocks_y: usize,
    slice_rows: usize,
    records: usize,
    selector: u8,
    limit: usize,
) -> Result<usize, DecodeError> {
    let (width, height) = dimensions;
    let resource = "decoder memory";
    let output = checked_mul(
        checked_mul(width, height, resource, limit)?,
        4,
        resource,
        limit,
    )?;
    let padded_width = checked_mul(blocks_x, 16, resource, limit)?;
    let padded_height = checked_mul(blocks_y, 16, resource, limit)?;
    let colour_picture = checked_mul(
        checked_mul(padded_width, padded_height, resource, limit)?,
        3,
        resource,
        limit,
    )?;
    let slice_height = checked_mul(slice_rows, 16, resource, limit)?;
    let colour_slice = checked_mul(
        checked_mul(padded_width, slice_height, resource, limit)?,
        3,
        resource,
        limit,
    )?;
    let alpha_stride = checked_mul(
        checked_add(blocks_x, 2, resource, limit)?,
        16,
        resource,
        limit,
    )?;
    let (alpha_picture, alpha_slice) = if selector == 2 {
        (0, 0)
    } else {
        (
            checked_mul(
                alpha_stride,
                checked_add(padded_height, 16, resource, limit)?,
                resource,
                limit,
            )?,
            checked_mul(
                alpha_stride,
                checked_add(slice_height, 16, resource, limit)?,
                resource,
                limit,
            )?,
        )
    };
    let block_count = checked_mul(blocks_x, blocks_y, resource, limit)?;
    let context = checked_mul(block_count, 4096, resource, limit)?;
    let descriptors = checked_mul(
        records,
        std::mem::size_of::<format::Record<'_>>(),
        resource,
        limit,
    )?;
    let total = [
        output,
        colour_picture,
        colour_slice,
        alpha_picture,
        alpha_slice,
        context,
        descriptors,
    ]
    .into_iter()
    .try_fold(0_usize, |sum, n| checked_add(sum, n, resource, limit))?;
    if total > limit {
        return Err(DecodeError::ResourceLimit {
            resource,
            requested: total,
            limit,
        });
    }
    Ok(output)
}

fn check_live_memory(parts: &[usize], limit: usize) -> Result<usize, DecodeError> {
    let total = parts.iter().try_fold(0_usize, |sum, part| {
        checked_add(sum, *part, "decoder memory", limit)
    })?;
    if total > limit {
        return Err(DecodeError::ResourceLimit {
            resource: "decoder memory",
            requested: total,
            limit,
        });
    }
    Ok(total)
}

fn compose_rows(
    output: &mut [u8],
    width: usize,
    geometry: SliceGeometry,
    colour: &crate::colour::ColourSlice,
    alpha: Option<&crate::alpha::AlphaSlice>,
) -> Result<(), DecodeError> {
    let height = geometry.visible_height();
    let padded_width = geometry.block_width * 16;
    if colour.stride < padded_width
        || colour
            .planes
            .iter()
            .any(|plane| plane.len() < colour.stride * height)
    {
        return Err(DecodeError::Coverage {
            detail: "colour plane does not cover visible rows",
        });
    }
    if let Some(alpha) = alpha
        && (alpha.stride < width || alpha.plane.len() < alpha.stride * height)
    {
        return Err(DecodeError::Coverage {
            detail: "alpha plane does not cover visible rows",
        });
    }
    for row in 0..height {
        let output_row = (geometry.pixel_row() + row) * width * 4;
        let colour_row = row * colour.stride;
        let alpha_row = alpha.map_or(0, |plane| row * plane.stride);
        for x in 0..width {
            let pixel = output_row + x * 4;
            // Stored source planes are green, blue, red; the output is RGBA.
            output[pixel] = colour.planes[2][colour_row + x];
            output[pixel + 1] = colour.planes[0][colour_row + x];
            output[pixel + 2] = colour.planes[1][colour_row + x];
            output[pixel + 3] = alpha.map_or(255, |plane| plane.plane[alpha_row + x]);
        }
    }
    Ok(())
}

struct RecordDecodeContext<'data> {
    output: &'data mut Vec<u8>,
    colour_state: &'data mut crate::colour::ColourState,
    alpha_state: &'data mut Option<crate::alpha::AlphaState>,
    width: usize,
    height: usize,
    blocks_x: usize,
    blocks_y: usize,
    slice_rows: usize,
    record_storage: usize,
    memory_limit: usize,
}

impl RecordDecodeContext<'_> {
    fn record_geometry(
        &self,
        record: &format::Record<'_>,
        index: usize,
        covered_rows: usize,
    ) -> Result<SliceGeometry, DecodeError> {
        if usize::from(record.index) != index {
            return Err(DecodeError::Coverage {
                detail: "record indices are not consecutive",
            }
            .at(ErrorContext {
                record: Some(index),
                ..ErrorContext::default()
            }));
        }
        let origin = index
            .checked_mul(self.slice_rows)
            .ok_or(DecodeError::Coverage {
                detail: "record origin overflows",
            })?;
        if origin >= self.blocks_y {
            return Err(DecodeError::Coverage {
                detail: "record origin exceeds the picture",
            }
            .at(ErrorContext {
                record: Some(index),
                ..ErrorContext::default()
            }));
        }
        let geometry = SliceGeometry {
            block_width: self.blocks_x,
            block_rows: self.slice_rows.min(self.blocks_y - origin),
            absolute_block_row: origin,
            picture_width: self.width,
            picture_height: self.height,
        };
        if geometry.pixel_row() != covered_rows {
            return Err(DecodeError::Coverage {
                detail: "slice rows have a gap",
            });
        }
        Ok(geometry)
    }

    fn decode_records(&mut self, records: &[format::Record<'_>]) -> Result<(), DecodeError> {
        let mut covered_rows = 0_usize;
        for (index, record) in records.iter().enumerate() {
            let geometry = self.record_geometry(record, index, covered_rows)?;
            let base_live_bytes = check_live_memory(
                &[
                    self.output.capacity(),
                    self.record_storage,
                    self.colour_state.heap_bytes(),
                    self.alpha_state
                        .as_ref()
                        .map_or(0, crate::alpha::AlphaState::heap_bytes),
                ],
                self.memory_limit,
            )?;
            let budget = crate::colour::ColourMemoryBudget {
                base_live_bytes,
                limit: self.memory_limit,
                palette_capacity: 0,
            };
            let colour = crate::colour::decode_slice(record, geometry, self.colour_state, budget)
                .map_err(|error| {
                error.at(ErrorContext {
                    record: Some(index),
                    ..ErrorContext::default()
                })
            })?;
            log::debug!("record {index}: colour endpoint {} bits", colour.end_bit);
            let aligned =
                colour
                    .end_bit
                    .checked_add(7)
                    .map(|v| v & !7)
                    .ok_or(DecodeError::Coverage {
                        detail: "colour bit position overflows",
                    })?;
            let record_bits = record
                .data
                .len()
                .checked_mul(8)
                .ok_or(DecodeError::Coverage {
                    detail: "record bit length overflows",
                })?;
            if aligned > record_bits {
                return Err(DecodeError::Coverage {
                    detail: "colour stream exceeds its record",
                });
            }
            let alpha = if let Some(state) = self.alpha_state.as_mut() {
                Some(
                    crate::alpha::decode_slice(record, geometry, colour.end_bit, state).map_err(
                        |error| {
                            error.at(ErrorContext {
                                record: Some(index),
                                ..ErrorContext::default()
                            })
                        },
                    )?,
                )
            } else {
                None
            };
            check_live_memory(
                &[
                    self.output.capacity(),
                    self.record_storage,
                    self.colour_state.heap_bytes(),
                    self.alpha_state
                        .as_ref()
                        .map_or(0, crate::alpha::AlphaState::heap_bytes),
                    colour.heap_bytes(),
                    alpha
                        .as_ref()
                        .map_or(0, crate::alpha::AlphaSlice::heap_bytes),
                ],
                self.memory_limit,
            )?;
            let end_bit = alpha.as_ref().map_or(aligned, |stream| stream.end_bit);
            log::debug!("record {index}: final endpoint {end_bit} bits");
            if end_bit < aligned || end_bit > record_bits || record_bits - end_bit >= 8 {
                return Err(DecodeError::Coverage {
                    detail: "record stream has incomplete or extra syntax",
                }
                .at(ErrorContext {
                    record: Some(index),
                    bit_position: Some(end_bit),
                    ..ErrorContext::default()
                }));
            }
            compose_rows(self.output, self.width, geometry, &colour, alpha.as_ref())?;
            covered_rows += geometry.visible_height();
        }
        if covered_rows != self.height {
            return Err(DecodeError::Coverage {
                detail: "picture rows are incomplete",
            });
        }
        Ok(())
    }
}

fn validate_decode_picture(picture: format::PictureInfo) -> Result<(), DecodeError> {
    if picture.colour_mode != 0 {
        return Err(DecodeError::UnsupportedColourMode {
            value: picture.colour_mode,
        });
    }
    if !matches!(picture.selector, 2 | 4 | 5) {
        return Err(DecodeError::UnsupportedOutputFormat {
            selector: picture.selector,
        });
    }
    Ok(())
}

pub fn decode_with_limits(
    input: &[u8],
    limits: DecodeLimits,
) -> Result<image::RgbaImage, DecodeError> {
    if input.len() > limits.max_input_bytes {
        return Err(DecodeError::ResourceLimit {
            resource: "input",
            requested: input.len(),
            limit: limits.max_input_bytes,
        });
    }
    let container =
        format::parse_container(input).map_err(|error| ParseError::Container(Box::new(error)))?;
    let picture = format::parse_picture(container.picture)
        .map_err(|error| ParseError::Picture(Box::new(error)))?;
    validate_decode_picture(picture)?;
    let width = usize::from(picture.width);
    let height = usize::from(picture.height);
    let blocks_x = width.div_ceil(16);
    let blocks_y = height.div_ceil(16);
    let descriptor_limit = checked_mul(
        blocks_y,
        std::mem::size_of::<format::Record<'_>>(),
        "record descriptors",
        limits.max_memory_bytes,
    )?;
    if descriptor_limit > limits.max_memory_bytes {
        return Err(DecodeError::ResourceLimit {
            resource: "record descriptors",
            requested: descriptor_limit,
            limit: limits.max_memory_bytes,
        });
    }
    let records = format::parse_records(container.records, blocks_y)
        .map_err(|error| ParseError::Record(Box::new(error)))?;
    if records.is_empty() {
        return Err(DecodeError::Coverage {
            detail: "picture has no records",
        });
    }
    // Each record starts at a fixed span. The last record can cover fewer rows.
    let slice_rows = blocks_y.div_ceil(records.len());
    let output_len = check_memory(
        (width, height),
        blocks_x,
        blocks_y,
        slice_rows,
        records.len(),
        picture.selector,
        limits.max_memory_bytes,
    )?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .map_err(|_| DecodeError::Allocation { resource: "output" })?;
    output.resize(output_len, 0);
    let mut colour_state = crate::colour::ColourState::new(blocks_x, blocks_y)?;
    let mut alpha_state = if picture.selector == 2 {
        None
    } else {
        Some(crate::alpha::AlphaState::new(blocks_x, blocks_y)?)
    };
    let record_storage = checked_mul(
        records.capacity(),
        std::mem::size_of::<format::Record<'_>>(),
        "record descriptors",
        limits.max_memory_bytes,
    )?;
    let alpha_state_bytes = alpha_state
        .as_ref()
        .map_or(0, crate::alpha::AlphaState::heap_bytes);
    check_live_memory(
        &[
            output.capacity(),
            record_storage,
            colour_state.heap_bytes(),
            alpha_state_bytes,
        ],
        limits.max_memory_bytes,
    )?;
    RecordDecodeContext {
        output: &mut output,
        colour_state: &mut colour_state,
        alpha_state: &mut alpha_state,
        width,
        height,
        blocks_x,
        blocks_y,
        slice_rows,
        record_storage,
        memory_limit: limits.max_memory_bytes,
    }
    .decode_records(&records)?;
    let info = ImageInfo {
        width: u32::from(picture.width),
        height: u32::from(picture.height),
        colour_mode: picture.colour_mode,
        selector: picture.selector,
    };
    image::RgbaImage::from_raw(info.width, info.height, output).ok_or(DecodeError::Coverage {
        detail: "output buffer has an invalid size",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn framed(width: u16, height: u16, selector: u8, record: &[u8]) -> Vec<u8> {
        let mut picture = vec![0xaa, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        picture.extend_from_slice(&width.to_be_bytes());
        picture.extend_from_slice(&height.to_be_bytes());
        picture.push(selector);
        picture.extend_from_slice(&[0; 4]);
        let mut input = Vec::new();
        input.extend_from_slice(
            &u32::try_from(picture.len())
                .expect("test picture fits u32")
                .to_le_bytes(),
        );
        input.extend_from_slice(&picture);
        input.extend_from_slice(
            &u32::try_from(record.len())
                .expect("test record fits u32")
                .to_le_bytes(),
        );
        input.extend_from_slice(record);
        input
    }

    fn raw_record_rows(index: u16, colours: &[[u8; 3]], alpha: Option<u8>) -> Vec<u8> {
        let mut body = Vec::new();
        for colour in colours {
            body.push(0x30); // method 5, then byte alignment
            for value in colour {
                body.extend_from_slice(&[*value; 256]);
            }
        }
        if let Some(value) = alpha {
            for _ in colours {
                body.push(0x30); // alpha method 5, then byte alignment
                body.extend_from_slice(&[value; 256]);
            }
        }
        let len = 14 + body.len();
        let mut record = vec![0xaa, 2];
        record.extend_from_slice(
            &u32::try_from(len)
                .expect("test record fits u32")
                .to_be_bytes(),
        );
        record.push(0); // first opaque header byte
        record.extend_from_slice(&index.to_be_bytes());
        record.extend_from_slice(&[0; 5]); // remaining 40 header bits
        record.extend_from_slice(&body);
        record
    }

    fn raw_record(index: u16, colour: [u8; 3], alpha: Option<u8>) -> Vec<u8> {
        raw_record_rows(index, &[colour], alpha)
    }

    #[test]
    fn header_keeps_raw_selector_and_big_endian_dimensions() {
        let input = framed(0x1234, 0x0102, 7, &[]);
        let info = crate::read_header(&input).unwrap();
        assert_eq!(
            (info.width, info.height, info.selector),
            (0x1234, 0x0102, 7)
        );
        assert_eq!(
            crate::decode(&input).unwrap_err().kind(),
            crate::ErrorKind::UnsupportedOutputFormat
        );
    }

    #[test]
    fn input_and_memory_limits_are_classified() {
        let record = [0xaa, 2, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0];
        let input = framed(16, 16, 2, &record);
        let small_input = DecodeLimits {
            max_input_bytes: input.len() - 1,
            ..DecodeLimits::default()
        };
        assert_eq!(
            crate::decode_with_limits(&input, small_input)
                .unwrap_err()
                .kind(),
            crate::ErrorKind::ResourceLimit
        );
        let small_memory = DecodeLimits {
            max_memory_bytes: 1024,
            ..DecodeLimits::default()
        };
        assert_eq!(
            crate::decode_with_limits(&input, small_memory)
                .unwrap_err()
                .kind(),
            crate::ErrorKind::ResourceLimit
        );
    }

    #[test]
    fn raw_blocks_compose_clipped_rgba_for_all_supported_selectors() {
        for selector in [2, 4, 5] {
            let record = raw_record(0, [11, 22, 33], (selector != 2).then_some(44));
            let pixels = crate::decode(&framed(7, 5, selector, &record)).unwrap();
            assert_eq!((pixels.width(), pixels.height()), (7, 5));
            assert_eq!(pixels.as_raw().len(), 7 * 5 * 4);
            for pixel in pixels.as_raw().chunks_exact(4) {
                assert_eq!(pixel, &[33, 11, 22, if selector == 2 { 255 } else { 44 }]);
            }
        }
    }

    #[test]
    fn multiple_raw_slices_cover_visible_rows() {
        let mut records = raw_record(0, [1, 2, 3], Some(4));
        records.extend_from_slice(&raw_record(1, [5, 6, 7], Some(8)));
        let pixels = crate::decode(&framed(16, 17, 4, &records)).unwrap();
        assert_eq!(&pixels.as_raw()[0..4], &[3, 1, 2, 4]);
        assert_eq!(
            &pixels.as_raw()[16 * 16 * 4..16 * 16 * 4 + 4],
            &[7, 5, 6, 8]
        );
    }

    #[test]
    fn nominal_slice_span_covers_a_short_final_slice() {
        let mut records = raw_record_rows(0, &[[1, 2, 3], [4, 5, 6]], None);
        records.extend_from_slice(&raw_record_rows(1, &[[7, 8, 9], [10, 11, 12]], None));
        records.extend_from_slice(&raw_record_rows(2, &[[13, 14, 15]], None));
        let pixels = crate::decode(&framed(16, 65, 2, &records)).unwrap();
        for (row, expected) in [
            (0, [3, 1, 2, 255]),
            (16, [6, 4, 5, 255]),
            (32, [9, 7, 8, 255]),
            (48, [12, 10, 11, 255]),
            (64, [15, 13, 14, 255]),
        ] {
            assert_eq!(&pixels.as_raw()[row * 16 * 4..row * 16 * 4 + 4], &expected);
        }
    }

    #[test]
    fn unsupported_alpha_method_has_stable_kind() {
        let mut record = raw_record(0, [1, 2, 3], None);
        record.extend_from_slice(&[0; 4]); // alpha method 2 plus reader window
        let new_len = u32::try_from(record.len())
            .expect("test record fits u32")
            .to_be_bytes();
        record[2..6].copy_from_slice(&new_len);
        let error = crate::decode(&framed(16, 16, 4, &record)).unwrap_err();
        assert_eq!(error.kind(), crate::ErrorKind::UnsupportedSyntax);
    }
}
