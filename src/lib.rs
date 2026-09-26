//! Native SPI picture decoding.
//!
//! Input is borrowed for one decode call. Output pixels are tightly packed,
//! top-to-bottom RGBA with straight alpha. RGB values remain unchanged when
//! alpha is zero. All offsets in errors are relative to the start of the
//! complete record named by that error.

#![forbid(unsafe_code)]

mod alpha;
mod bits;
mod color;
mod decoder;
mod entropy;
mod error;
mod format;
mod prediction;
mod surface;
mod transform;

pub use error::{
    CoefficientError, DecodeError, ErrorContext, ErrorKind, MotionError, ParseError,
    PredictionError, SyntaxError,
};

/// Picture metadata from the first container payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub color_mode: u8,
    pub selector: u8,
}

/// Limits for one decode call. The memory limit covers decoder-owned output,
/// persistent planes, slice storage, record views, and context storage. It
/// excludes the caller's input buffer, allocator overhead, and PNG encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeLimits {
    pub max_input_bytes: usize,
    pub max_memory_bytes: usize,
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 256 * 1024 * 1024,
            max_memory_bytes: 512 * 1024 * 1024,
        }
    }
}

/// Inspect the framed container and picture header without decoding records.
/// An unsupported output selector is still readable metadata.
pub fn read_header(input: &[u8]) -> Result<ImageInfo, DecodeError> {
    let container = format::parse_container(input)
        .map_err(|error| error::ParseError::Container(Box::new(error)))?;
    let picture = format::parse_picture(container.picture)
        .map_err(|error| error::ParseError::Picture(Box::new(error)))?;
    Ok(ImageInfo {
        width: u32::from(picture.width),
        height: u32::from(picture.height),
        color_mode: picture.color_mode,
        selector: picture.selector,
    })
}

/// Decode one complete picture with default resource limits.
pub fn decode(input: &[u8]) -> Result<image::RgbaImage, DecodeError> {
    decode_with_limits(input, DecodeLimits::default())
}

/// Decode one complete picture with explicit resource limits.
pub fn decode_with_limits(
    input: &[u8],
    limits: DecodeLimits,
) -> Result<image::RgbaImage, DecodeError> {
    decoder::decode_with_limits(input, limits)
}
