//! Stable error categories and context for failed SPI decoding.

use std::error::Error;

use thiserror::Error;

/// Stable categories used by callers and corpus manifests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    Parse,
    UnsupportedOutputFormat,
    UnsupportedColourMode,
    UnsupportedSyntax,
    Bitstream,
    Coefficients,
    Prediction,
    Motion,
    Coverage,
    ResourceLimit,
    Allocation,
}

impl ErrorKind {
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

/// Optional positions for a decode error. A record number is its zero-based order in the second container payload. A block number is local to that record, in row-major order. A bit position starts at the first bit of the complete `AA 02` record, including its six-byte prefix. A component number starts at zero and refers to an internal plane.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ErrorContext {
    pub record: Option<usize>,
    pub block: Option<usize>,
    pub component: Option<usize>,
    pub bit_position: Option<usize>,
}

/// Errors from container and header parsing.
#[derive(Debug, Error)]
pub enum ParseError {
    #[error("invalid container: {0}")]
    Container(#[source] Box<dyn Error + Send + Sync>),
    #[error("invalid picture header: {0}")]
    Picture(#[source] Box<dyn Error + Send + Sync>),
    #[error("invalid record framing: {0}")]
    Record(#[source] Box<dyn Error + Send + Sync>),
}

/// Errors from compressed syntax and ordinary bit reads.
#[derive(Debug, Error)]
pub enum SyntaxError {
    #[error("invalid bitstream: {0}")]
    Bits(#[source] Box<dyn Error + Send + Sync>),
    #[error("unsupported {stream} method {method}")]
    UnsupportedMethod { stream: &'static str, method: u8 },
    #[error("invalid {stream} syntax: {detail}")]
    Invalid {
        stream: &'static str,
        detail: &'static str,
    },
}

#[derive(Debug, Error)]
pub enum CoefficientError {
    #[error("invalid coefficient syntax: {0}")]
    Syntax(#[source] Box<dyn Error + Send + Sync>),
    #[error("invalid coefficient reconstruction: {0}")]
    Transform(#[source] Box<dyn Error + Send + Sync>),
}

#[derive(Debug, Error)]
#[error("prediction failed: {source}")]
pub struct PredictionError {
    #[source]
    pub source: Box<dyn Error + Send + Sync>,
}

#[derive(Debug, Error)]
#[error("motion copy failed: {source}")]
pub struct MotionError {
    #[source]
    pub source: Box<dyn Error + Send + Sync>,
}

/// Public failure type. `At` adds positions without changing the stable kind.
#[derive(Debug, Error)]
pub enum DecodeError {
    #[error(transparent)]
    Parse(#[from] ParseError),
    #[error("unsupported output selector {selector}")]
    UnsupportedOutputFormat { selector: u8 },
    #[error("unsupported colour mode {value}")]
    UnsupportedColourMode { value: u8 },
    #[error(transparent)]
    Syntax(#[from] SyntaxError),
    #[error(transparent)]
    Coefficients(#[from] CoefficientError),
    #[error(transparent)]
    Prediction(#[from] PredictionError),
    #[error(transparent)]
    Motion(#[from] MotionError),
    #[error("invalid slice coverage: {detail}")]
    Coverage { detail: &'static str },
    #[error("{resource} requires {requested} bytes, above the {limit}-byte limit")]
    ResourceLimit {
        resource: &'static str,
        requested: usize,
        limit: usize,
    },
    #[error("allocation failed for {resource}")]
    Allocation { resource: &'static str },
    #[error("{source} at {context:?}")]
    At {
        context: ErrorContext,
        #[source]
        source: Box<DecodeError>,
    },
}

impl DecodeError {
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

    pub fn at(self, context: ErrorContext) -> Self {
        Self::At {
            context,
            source: Box::new(self),
        }
    }
}
