//! Native SPI picture decoding.
//!
//! Input is borrowed for one decode call. Output pixels are tightly packed,
//! top-to-bottom RGBA with straight alpha. RGB values remain unchanged when
//! alpha is zero. All offsets in errors are relative to the start of the
//! complete record named by that error.

#![forbid(unsafe_code)]
#![warn(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_ptr_alignment,
    clippy::cast_sign_loss,
    clippy::char_lit_as_u8,
    clippy::checked_conversions,
    clippy::dbg_macro,
    clippy::debug_assert_with_mut_call,
    clippy::doc_link_with_quotes,
    clippy::doc_markdown,
    clippy::empty_line_after_outer_attr,
    clippy::float_cmp_const,
    clippy::float_cmp,
    clippy::float_equality_without_abs,
    clippy::missing_panics_doc,
    clippy::mod_module_files,
    clippy::option_if_let_else,
    clippy::redundant_pub_crate,
    clippy::semicolon_if_nothing_returned,
    clippy::shadow_unrelated,
    clippy::similar_names,
    clippy::suspicious_operation_groupings,
    clippy::todo,
    clippy::too_many_lines,
    clippy::unnecessary_cast,
    clippy::unneeded_field_pattern,
    clippy::unnested_or_patterns,
    clippy::unseparated_literal_suffix,
    clippy::unused_self,
    clippy::used_underscore_binding,
    clippy::useless_let_if_seq,
    clippy::wildcard_dependencies,
    clippy::wildcard_imports,
    keyword_idents,
    non_ascii_idents,
    noop_method_call,
    unused_crate_dependencies,
    unused_extern_crates,
    unused_import_braces
)]

mod alpha;
mod bits;
mod colour;
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
    pub colour_mode: u8,
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
        colour_mode: picture.colour_mode,
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
