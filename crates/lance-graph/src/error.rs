// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Error types for the Lance graph query engine
//!
//! ## Error construction macros
//!
//! Use these macros instead of manual `Location::new(file!(), line!(), column!())`:
//!
//! ```ignore
//! // Before (verbose, 200+ chars per error):
//! GraphError::PlanError {
//!     message: format!("..."),
//!     location: snafu::Location::new(file!(), line!(), column!()),
//! }
//!
//! // After (zero-cost #[track_caller] via std::panic::Location):
//! plan_err!("Failed to plan: {}", reason)
//! config_err!("Invalid config: {}", detail)
//! exec_err!("Execution failed: {}", detail)
//! ```

use snafu::{prelude::*, Location};

pub type Result<T> = std::result::Result<T, GraphError>;

// =============================================================================
// Zero-cost error construction via #[track_caller]
// =============================================================================

/// Create a PlanError with zero-cost caller location capture.
///
/// `#[track_caller]` makes the compiler insert the call-site location
/// at compile time — 0 runtime cycles (Gate 4). Uses `std::panic::Location`
/// internally (Gate 7), bridged to `snafu::Location` for compatibility
/// with the existing error enum.
#[track_caller]
pub fn plan_err_at(message: String) -> GraphError {
    let loc = std::panic::Location::caller();
    GraphError::PlanError {
        message,
        location: Location::new(loc.file(), loc.line(), loc.column()),
    }
}

/// Create a ConfigError with zero-cost caller location capture.
#[track_caller]
pub fn config_err_at(message: String) -> GraphError {
    let loc = std::panic::Location::caller();
    GraphError::ConfigError {
        message,
        location: Location::new(loc.file(), loc.line(), loc.column()),
    }
}

/// Create an ExecutionError with zero-cost caller location capture.
#[track_caller]
pub fn exec_err_at(message: String) -> GraphError {
    let loc = std::panic::Location::caller();
    GraphError::ExecutionError {
        message,
        location: Location::new(loc.file(), loc.line(), loc.column()),
    }
}

/// Create a PlanError with zero-cost location capture.
///
/// Uses `#[track_caller]` via `plan_err_at()` — the compiler inserts the
/// call-site file/line/column at 0 runtime cycles. No `file!()` / `line!()`
/// macros needed.
///
/// # Example
/// ```ignore
/// use lance_graph::error::plan_err;
/// let err = plan_err!("Cannot join {} to {}", left, right);
/// ```
#[macro_export]
macro_rules! plan_err {
    ($($arg:tt)*) => {
        $crate::error::plan_err_at(format!($($arg)*))
    };
}

/// Create a ConfigError with zero-cost location capture.
#[macro_export]
macro_rules! config_err {
    ($($arg:tt)*) => {
        $crate::error::config_err_at(format!($($arg)*))
    };
}

/// Create an ExecutionError with zero-cost location capture.
#[macro_export]
macro_rules! exec_err {
    ($($arg:tt)*) => {
        $crate::error::exec_err_at(format!($($arg)*))
    };
}

/// Errors that can occur during graph query processing
#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum GraphError {
    /// Error parsing Cypher query syntax
    #[snafu(display("Cypher parse error at position {position}: {message}"))]
    ParseError {
        message: String,
        position: usize,
        location: Location,
    },

    /// Error with graph configuration
    #[snafu(display("Graph configuration error: {message}"))]
    ConfigError { message: String, location: Location },

    /// Error during query planning
    #[snafu(display("Query planning error: {message}"))]
    PlanError { message: String, location: Location },

    /// Error during query execution
    #[snafu(display("Query execution error: {message}"))]
    ExecutionError { message: String, location: Location },

    /// Unsupported Cypher feature
    #[snafu(display("Unsupported Cypher feature: {feature}"))]
    UnsupportedFeature { feature: String, location: Location },

    /// Invalid graph pattern
    #[snafu(display("Invalid graph pattern: {message}"))]
    InvalidPattern { message: String, location: Location },

    /// DataFusion integration error
    #[snafu(display("DataFusion error: {source}"))]
    DataFusion {
        source: datafusion_common::DataFusionError,
        location: Location,
    },

    /// Lance core error
    #[snafu(display("Lance core error: {source}"))]
    LanceCore {
        source: lance::Error,
        location: Location,
    },

    /// Arrow error
    #[snafu(display("Arrow error: {source}"))]
    Arrow {
        source: arrow::error::ArrowError,
        location: Location,
    },
}

impl From<datafusion_common::DataFusionError> for GraphError {
    fn from(source: datafusion_common::DataFusionError) -> Self {
        Self::DataFusion {
            source,
            location: Location::new(file!(), line!(), column!()),
        }
    }
}

impl From<lance::Error> for GraphError {
    fn from(source: lance::Error) -> Self {
        Self::LanceCore {
            source,
            location: Location::new(file!(), line!(), column!()),
        }
    }
}

impl From<arrow::error::ArrowError> for GraphError {
    fn from(source: arrow::error::ArrowError) -> Self {
        Self::Arrow {
            source,
            location: Location::new(file!(), line!(), column!()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_err_carries_location() {
        let err = plan_err!("cannot plan join: {} to {}", "left", "right");
        match err {
            GraphError::PlanError { message, location } => {
                assert_eq!(message, "cannot plan join: left to right");
                // #[track_caller] captures the call site — this file.
                assert!(
                    location.file.contains("error.rs"),
                    "location should point to this file, got: {}",
                    location.file
                );
                assert!(location.line > 0, "line should be non-zero");
            }
            other => panic!("expected PlanError, got: {:?}", other),
        }
    }

    #[test]
    fn test_config_err_carries_location() {
        let err = config_err!("invalid config: {}", "missing field");
        match err {
            GraphError::ConfigError { message, location } => {
                assert_eq!(message, "invalid config: missing field");
                assert!(location.file.contains("error.rs"));
            }
            other => panic!("expected ConfigError, got: {:?}", other),
        }
    }

    #[test]
    fn test_exec_err_carries_location() {
        let err = exec_err!("execution failed at step {}", 3);
        match err {
            GraphError::ExecutionError { message, location } => {
                assert_eq!(message, "execution failed at step 3");
                assert!(location.file.contains("error.rs"));
            }
            other => panic!("expected ExecutionError, got: {:?}", other),
        }
    }

    #[test]
    fn test_plan_err_display() {
        let err = plan_err!("test error");
        let display = format!("{}", err);
        assert!(display.contains("test error"));
    }
}
