//! Classify decode failures and retain the positions known at each layer.
//!
//! [`ErrorKind`] supplies stable names for callers and corpus expectations. Detailed error
//! messages can give more information, but they are not the stable classification interface.
//! [`DecodeError::kind`] follows nested position wrappers to the underlying category.
//!
//! A component layer can report a contextual cursor, a block layer can report a block and ordinary
//! cursor, and the record layer can add its sequence number. [`DecodeError::at`] retains each as a
//! separate nested wrapper; it does not merge fields or replace earlier positions. Bit offsets
//! start at the complete record, including all 14 header bytes. Block numbers are record-local and
//! row-major. Colour component numbers are `plane*4+argument`; alpha uses the argument number.
//!
//! Unsupported selectors, colour modes, and recognised methods are distinct from malformed syntax.
//! Internal wrappers do not guarantee that every low-level surface or predictor failure retains
//! its own category: stream dispatchers can convert it to a syntax error. The crate page explains
//! error positions and the limits of memory accounting.

use std::error::Error;

use thiserror::Error;

/// Stable categories used by callers and corpus manifests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// Invalid container or fixed-header framing.
    Parse,
    /// A well-formed but unsupported output selector.
    UnsupportedOutputFormat,
    /// A well-formed but unsupported colour mode.
    UnsupportedColourMode,
    /// A recognised stream that selects an unsupported coding method.
    UnsupportedSyntax,
    /// Invalid or truncated ordinary codec syntax.
    Bitstream,
    /// Invalid entropy-coded coefficients or transform reconstruction.
    Coefficients,
    /// An invalid spatial prediction request.
    Prediction,
    /// An invalid motion-copy request.
    Motion,
    /// Records or decoded planes that do not cover the picture correctly.
    Coverage,
    /// A configured input or memory limit violation.
    ResourceLimit,
    /// An allocator failure while reserving decoder storage.
    Allocation,
}

impl ErrorKind {
    /// Returns the stable lower-case name of this error category.
    ///
    /// The returned names are suitable for logs and corpus manifests. They are independent of the
    /// more detailed human-readable error messages.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Parse => "parse",
            Self::UnsupportedOutputFormat => "unsupported_output_format",
            Self::UnsupportedColourMode => "unsupported_colour_mode",
            Self::UnsupportedSyntax => "unsupported_syntax",
            Self::Bitstream => "bitstream",
            Self::Coefficients => "coefficients",
            Self::Prediction => "prediction",
            Self::Motion => "motion",
            Self::Coverage => "coverage",
            Self::ResourceLimit => "resource_limit",
            Self::Allocation => "allocation",
        }
    }
}

impl std::fmt::Display for ErrorKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Identifies optional positions associated with a decode error.
///
/// A record number is its zero-based order in the second container payload. A block number is
/// local to that record, in row-major order. A bit position starts at the first bit of the
/// complete `AA 02` record, including its six-byte prefix. A component number starts at zero; it
/// identifies an alpha argument or a colour plane component, according to the stream.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ErrorContext {
    /// The zero-based record number, when known.
    pub record: Option<usize>,
    /// The zero-based, record-local block number, when known.
    pub block: Option<usize>,
    /// The zero-based colour component or alpha argument, when known.
    pub component: Option<usize>,
    /// The record-relative bit position, when known.
    pub bit_position: Option<usize>,
}

/// Errors from container and header parsing.
#[derive(Debug, Error)]
pub enum ParseError {
    /// Wraps an error in the outer two-section container.
    #[error("invalid container: {0}")]
    Container(#[source] Box<dyn Error + Send + Sync>),
    /// Wraps an error in the fixed picture header.
    #[error("invalid picture header: {0}")]
    Picture(#[source] Box<dyn Error + Send + Sync>),
    /// Wraps an error in record framing or a record syntax header.
    #[error("invalid record framing: {0}")]
    Record(#[source] Box<dyn Error + Send + Sync>),
}

/// Errors from compressed syntax and ordinary bit reads.
#[derive(Debug, Error)]
pub enum SyntaxError {
    /// Wraps a low-level bit-reading failure.
    #[error("invalid bitstream: {0}")]
    Bits(#[source] Box<dyn Error + Send + Sync>),
    /// A method value that this decoder does not implement.
    #[error("unsupported {stream} method {method}")]
    UnsupportedMethod {
        /// Names the colour or alpha stream.
        stream: &'static str,
        /// The decoded method number.
        method: u8,
    },
    /// Malformed syntax with a static diagnostic.
    #[error("invalid {stream} syntax: {detail}")]
    Invalid {
        /// Names the colour or alpha stream.
        stream: &'static str,
        /// Describes the violated syntax rule.
        detail: &'static str,
    },
}

/// Failures while entropy-coded coefficients are decoded or reconstructed.
///
/// This distinction lets callers separate malformed coefficient syntax from an invalid
/// inverse-transform request.
#[derive(Debug, Error)]
pub enum CoefficientError {
    /// Wraps an entropy or coefficient-history syntax error.
    #[error("invalid coefficient syntax: {0}")]
    Syntax(#[source] Box<dyn Error + Send + Sync>),
    /// Wraps an inverse-transform or residual-reconstruction error.
    #[error("invalid coefficient reconstruction: {0}")]
    Transform(#[source] Box<dyn Error + Send + Sync>),
}

/// Wraps an error raised while building or applying spatial prediction.
///
/// The source is type-erased so internal prediction implementations can change without expanding
/// the public error API.
#[derive(Debug, Error)]
#[error("prediction failed: {source}")]
pub struct PredictionError {
    #[source]
    /// The internal prediction failure.
    pub source: Box<dyn Error + Send + Sync>,
}

/// Wraps an error raised while copying a motion-referenced block.
///
/// The source normally describes invalid plane dimensions or an out-of-bounds source or
/// destination rectangle.
#[derive(Debug, Error)]
#[error("motion copy failed: {source}")]
pub struct MotionError {
    #[source]
    /// The internal surface-copy failure.
    pub source: Box<dyn Error + Send + Sync>,
}

/// A public SPI decoding failure.
///
/// The [`DecodeError::At`] wrapper adds positions without changing the stable [`ErrorKind`]
/// returned by [`DecodeError::kind`].
#[derive(Debug, Error)]
pub enum DecodeError {
    /// Invalid container, picture-header, or record framing.
    #[error(transparent)]
    Parse(#[from] ParseError),
    /// A selector outside the supported values 2, 4, and 5.
    #[error("unsupported output selector {selector}")]
    UnsupportedOutputFormat {
        /// The unsupported picture-header selector.
        selector: u8,
    },
    /// A colour mode other than the supported value 0.
    #[error("unsupported colour mode {value}")]
    UnsupportedColourMode {
        /// The unsupported picture-header colour mode.
        value: u8,
    },
    /// Malformed or unsupported colour or alpha syntax.
    #[error(transparent)]
    Syntax(#[from] SyntaxError),
    /// A coefficient entropy or reconstruction failure.
    #[error(transparent)]
    Coefficients(#[from] CoefficientError),
    /// A spatial prediction failure.
    #[error(transparent)]
    Prediction(#[from] PredictionError),
    /// An invalid motion reference or copy.
    #[error(transparent)]
    Motion(#[from] MotionError),
    /// Incomplete, overlapping, or otherwise invalid picture coverage.
    #[error("invalid slice coverage: {detail}")]
    Coverage {
        /// Describes the coverage invariant that failed.
        detail: &'static str,
    },
    /// That decoded resource requirements exceed a caller-supplied limit.
    #[error("{resource} requires {requested} bytes, above the {limit}-byte limit")]
    ResourceLimit {
        /// Names the bounded resource.
        resource: &'static str,
        /// The required byte count, or [`usize::MAX`] after arithmetic overflow.
        requested: usize,
        /// The configured maximum byte count.
        limit: usize,
    },
    /// That the allocator could not reserve a decoder buffer.
    #[error("allocation failed for {resource}")]
    Allocation {
        /// Names the buffer or state that could not be allocated.
        resource: &'static str,
    },
    /// Adds record, block, component, or bit context to another error.
    #[error("{source} at {context:?}")]
    At {
        /// All positions known at the wrapping layer.
        context: ErrorContext,
        #[source]
        /// The underlying classified decoder error.
        source: Box<DecodeError>,
    },
}

impl DecodeError {
    /// Returns the stable category for this failure.
    ///
    /// Position wrappers are traversed recursively, so adding [`ErrorContext`] with
    /// [`DecodeError::at`] never changes the result.
    pub const fn kind(&self) -> ErrorKind {
        match self {
            Self::Parse(_) => ErrorKind::Parse,
            Self::UnsupportedOutputFormat { .. } => ErrorKind::UnsupportedOutputFormat,
            Self::UnsupportedColourMode { .. } => ErrorKind::UnsupportedColourMode,
            Self::Syntax(SyntaxError::UnsupportedMethod { .. }) => ErrorKind::UnsupportedSyntax,
            Self::Syntax(_) => ErrorKind::Bitstream,
            Self::Coefficients(_) => ErrorKind::Coefficients,
            Self::Prediction(_) => ErrorKind::Prediction,
            Self::Motion(_) => ErrorKind::Motion,
            Self::Coverage { .. } => ErrorKind::Coverage,
            Self::ResourceLimit { .. } => ErrorKind::ResourceLimit,
            Self::Allocation { .. } => ErrorKind::Allocation,
            Self::At { source, .. } => source.kind(),
        }
    }

    /// Attaches positional context to this failure.
    ///
    /// Existing context is preserved as a nested source. Callers normally add information as an
    /// error crosses component, block, and record boundaries.
    pub fn at(self, context: ErrorContext) -> Self {
        Self::At {
            context,
            source: Box::new(self),
        }
    }
}

/// Converts a coefficient-history error into its public decoder category.
///
/// Allocation failure remains an allocation error associated with `resource`; every other
/// entropy-history failure becomes coefficient syntax.
pub fn coefficient_history_error(
    error: crate::entropy::EntropyError,
    resource: &'static str,
) -> DecodeError {
    match error {
        crate::entropy::EntropyError::AllocationFailure => DecodeError::Allocation { resource },
        other => DecodeError::Coefficients(CoefficientError::Syntax(Box::new(other))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coefficient_history_errors_keep_their_category() {
        assert_eq!(
            coefficient_history_error(
                crate::entropy::EntropyError::AllocationFailure,
                "coefficient history",
            )
            .kind(),
            ErrorKind::Allocation
        );
        assert_eq!(
            coefficient_history_error(
                crate::entropy::EntropyError::InvalidHistory,
                "coefficient history",
            )
            .kind(),
            ErrorKind::Coefficients
        );
    }
}
