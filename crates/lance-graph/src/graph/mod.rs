// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Graph primitives: fingerprinting, sparse bitmaps, and SPO triple store.
//!
//! This module provides the low-level graph data structures that sit beneath
//! the Cypher query engine. While the Cypher layer operates on property graphs
//! via DataFusion, this layer provides direct fingerprint-based graph operations.

pub mod fingerprint;
pub mod sparse;
pub mod spo;

/// Container geometry identifiers for graph storage layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ContainerGeometry {
    /// Flat record batch (default).
    Flat = 0,
    /// Adjacency list.
    AdjList = 1,
    /// CSR (Compressed Sparse Row).
    Csr = 2,
    /// CSC (Compressed Sparse Column).
    Csc = 3,
    /// COO (Coordinate list).
    Coo = 4,
    /// Hybrid (mixed format).
    Hybrid = 5,
    /// SPO (Subject-Predicate-Object triple store).
    Spo = 6,
}
