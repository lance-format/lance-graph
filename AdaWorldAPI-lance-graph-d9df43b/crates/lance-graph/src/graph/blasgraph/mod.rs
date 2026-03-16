// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # BlasGraph — Sparse Matrix Algebra with Semiring Operations
//!
//! A GraphBLAS-inspired sparse matrix algebra for graph computations.
//! Uses hyperdimensional binary vectors (16384-bit) as the fundamental
//! data type, enabling algebraic graph algorithms via semiring operations.
//!
//! ## Semirings
//!
//! | Name | Multiply | Add | Use Case |
//! |------|----------|-----|----------|
//! | XOR_BUNDLE | XOR | Bundle (majority) | Path composition |
//! | BIND_FIRST | XOR | First non-empty | BFS traversal |
//! | HAMMING_MIN | Hamming dist | Min | Shortest path |
//! | SIMILARITY_MAX | Similarity | Max | Best match |
//! | RESONANCE | XOR bind | Best density | Query expansion |
//! | BOOLEAN | AND | OR | Reachability |
//! | XOR_FIELD | XOR | XOR | GF(2) algebra |

pub mod hdr;
pub mod descriptor;
pub mod matrix;
pub mod ops;
pub mod semiring;
pub mod sparse;
pub mod types;
pub mod vector;

pub use descriptor::{Descriptor, GrBDesc};
pub use matrix::GrBMatrix;
pub use ops::*;
pub use semiring::{HdrSemiring, Semiring};
pub use sparse::{CooStorage, CsrStorage, SparseFormat};
pub use types::*;
pub use vector::GrBVector;

/// GraphBLAS status codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum GrBInfo {
    /// Operation completed successfully.
    Success = 0,
    /// No value at specified position.
    NoValue = 1,
    /// Invalid value provided.
    InvalidValue = 2,
    /// Index out of bounds.
    InvalidIndex = 3,
    /// Type domain mismatch.
    DomainMismatch = 4,
    /// Dimension mismatch between operands.
    DimensionMismatch = 5,
    /// Output container not empty.
    OutputNotEmpty = 6,
    /// Memory allocation failed.
    OutOfMemory = 7,
    /// Invalid object state.
    InvalidObject = 8,
    /// Null pointer.
    NullPointer = 9,
}

/// Initialize the GraphBLAS context (no-op in Rust).
pub fn grb_init() -> GrBInfo {
    GrBInfo::Success
}

/// Finalize the GraphBLAS context (no-op in Rust).
pub fn grb_finalize() -> GrBInfo {
    GrBInfo::Success
}

/// Get library version (GraphBLAS 2.0 compatible).
pub fn grb_version() -> (u32, u32, u32) {
    (2, 0, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grb_info() {
        assert_eq!(grb_init(), GrBInfo::Success);
        assert_eq!(grb_finalize(), GrBInfo::Success);
        assert_eq!(grb_version(), (2, 0, 0));
    }
}
