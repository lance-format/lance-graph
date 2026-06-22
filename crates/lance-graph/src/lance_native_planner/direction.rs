// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Traversal direction for the native CSR expand operators.

/// Direction a single-hop expand traverses. `Undirected` is intentionally
/// absent — undirected expands fall back to the DataFusion join planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativeDirection {
    /// Follow edges source -> destination (CSR).
    Outgoing,
    /// Follow edges destination -> source (CSC / reversed).
    Incoming,
}
