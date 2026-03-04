// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! SPO (Subject-Predicate-Object) triple store for fingerprint-based graph queries.
//!
//! This module provides:
//! - [`TruthValue`] / [`TruthGate`]: NARS-style confidence values and filters
//! - [`SpoBuilder`]: Constructs edge records from fingerprints
//! - [`SpoStore`]: In-memory triple store with bitmap ANN queries
//! - [`SpoSemiring`] / [`HammingMin`]: Semiring algebra for chain traversal
//! - [`MerkleRoot`] / [`BindSpace`]: Integrity verification for graph nodes

pub mod builder;
pub mod merkle;
pub mod semiring;
pub mod store;
pub mod truth;

pub use builder::{SpoBuilder, SpoRecord};
pub use merkle::{BindSpace, ClamPath, MerkleRoot, VerifyStatus};
pub use semiring::{HammingMin, SpoSemiring, TraversalHop};
pub use store::{SpoHit, SpoStore};
pub use truth::{TruthGate, TruthValue};
