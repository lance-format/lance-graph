// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # Neighborhood Vectors — Scope-Based Graph Representation
//!
//! Each node stores a **neighborhood vector**: an array of ZeckF64 edges
//! to all other nodes in its scope (up to 10,000 nodes).
//!
//! ```text
//! Position i = ZeckF64 edge to scope node i.
//! 0x0000000000000000 = no edge.
//! Position IS the address. No separate ID column.
//! One scope table maps position -> global node_id.
//! ```
//!
//! ## Storage Options
//!
//! | Mode            | Per Node   | Per 10K Scope | Precision |
//! |-----------------|------------|---------------|-----------|
//! | Scent only      | 10 KB      | 100 MB        | ρ ≈ 0.94  |
//! | Scent + fine    | 20 KB      | 200 MB        | ρ ≈ 0.96  |
//! | Full progressive| 80 KB      | 800 MB        | ρ ≈ 0.98  |

use super::types::BitVec;
use super::zeckf64;

/// Maximum number of nodes in a scope.
pub const MAX_SCOPE_SIZE: usize = 10_000;

/// A scope definition: maps positional indices to global node IDs.
#[derive(Clone, Debug)]
pub struct Scope {
    /// Unique scope identifier.
    pub scope_id: u64,
    /// Global node IDs, indexed by position. Length ≤ MAX_SCOPE_SIZE.
    pub node_ids: Vec<u64>,
}

impl Scope {
    /// Create a new scope.
    pub fn new(scope_id: u64, node_ids: Vec<u64>) -> Self {
        assert!(
            node_ids.len() <= MAX_SCOPE_SIZE,
            "scope size {} exceeds maximum {}",
            node_ids.len(),
            MAX_SCOPE_SIZE
        );
        Self { scope_id, node_ids }
    }

    /// Number of nodes in this scope.
    pub fn len(&self) -> usize {
        self.node_ids.len()
    }

    /// Whether the scope is empty.
    pub fn is_empty(&self) -> bool {
        self.node_ids.is_empty()
    }

    /// Look up the positional index for a global node ID.
    pub fn position_of(&self, global_id: u64) -> Option<usize> {
        self.node_ids.iter().position(|&id| id == global_id)
    }

    /// Get the global node ID at a positional index.
    pub fn global_id(&self, position: usize) -> Option<u64> {
        self.node_ids.get(position).copied()
    }
}

/// A neighborhood vector for a single node within a scope.
///
/// Contains one ZeckF64 edge per scope member. Position `i` holds the
/// ZeckF64 encoding of the edge from this node to scope node `i`.
/// A value of 0 means "no edge" (or self-edge).
#[derive(Clone, Debug)]
pub struct NeighborhoodVector {
    /// Global ID of the node this neighborhood belongs to.
    pub node_id: u64,
    /// Scope this neighborhood is within.
    pub scope_id: u64,
    /// ZeckF64 edges, one per scope position.
    pub edges: Vec<u64>,
}

impl NeighborhoodVector {
    /// Number of non-zero edges.
    pub fn edge_count(&self) -> u16 {
        self.edges.iter().filter(|&&e| e != 0).count() as u16
    }

    /// Extract scent bytes (byte 0 of each edge) for column-pruned storage.
    pub fn scent_column(&self) -> Vec<u8> {
        zeckf64::extract_scent(&self.edges)
    }

    /// Extract resolution bytes (byte 1 of each edge) for column-pruned storage.
    pub fn resolution_column(&self) -> Vec<u8> {
        zeckf64::extract_resolution(&self.edges)
    }

    /// Get the ZeckF64 edge to a specific scope position.
    pub fn edge_to(&self, position: usize) -> u64 {
        self.edges.get(position).copied().unwrap_or(0)
    }
}

/// Build all neighborhood vectors for a scope.
///
/// Given a set of node IDs and their SPO planes, compute the full
/// pairwise ZeckF64 encoding for every node pair.
///
/// # Arguments
///
/// - `scope_id` — Unique identifier for this scope
/// - `node_ids` — Global IDs of nodes in the scope
/// - `planes` — SPO triples (subject, predicate, object) for each node
///
/// # Returns
///
/// A `(Scope, Vec<NeighborhoodVector>)` pair.
pub fn build_scope(
    scope_id: u64,
    node_ids: &[u64],
    planes: &[(BitVec, BitVec, BitVec)],
) -> (Scope, Vec<NeighborhoodVector>) {
    assert_eq!(
        node_ids.len(),
        planes.len(),
        "node_ids and planes must have equal length"
    );
    assert!(
        node_ids.len() <= MAX_SCOPE_SIZE,
        "scope size {} exceeds maximum {}",
        node_ids.len(),
        MAX_SCOPE_SIZE
    );

    let n = node_ids.len();
    let scope = Scope::new(scope_id, node_ids.to_vec());

    // Build plane references for batch computation
    let plane_refs: Vec<(&BitVec, &BitVec, &BitVec)> =
        planes.iter().map(|(s, p, o)| (s, p, o)).collect();

    let neighborhoods: Vec<NeighborhoodVector> = (0..n)
        .map(|i| {
            let edges = zeckf64::compute_neighborhood(i, &plane_refs);
            NeighborhoodVector {
                node_id: node_ids[i],
                scope_id,
                edges,
            }
        })
        .collect();

    (scope, neighborhoods)
}

/// Compact scent-only representation for memory-efficient search.
///
/// Stores only byte 0 of each ZeckF64 edge, reducing memory from 80KB
/// to 10KB per node at ρ ≈ 0.94 precision.
#[derive(Clone, Debug)]
pub struct ScentVector {
    /// Global ID of the node.
    pub node_id: u64,
    /// Scent bytes, one per scope position.
    pub scents: Vec<u8>,
}

impl ScentVector {
    /// Create from a full NeighborhoodVector by extracting scent bytes.
    pub fn from_neighborhood(nv: &NeighborhoodVector) -> Self {
        Self {
            node_id: nv.node_id,
            scents: nv.scent_column(),
        }
    }

    /// Number of non-zero scent entries.
    pub fn edge_count(&self) -> u16 {
        self.scents.iter().filter(|&&s| s != 0).count() as u16
    }
}

/// Compact u16 progressive representation (scent + SPO quantile).
///
/// 20KB per node at ρ ≈ 0.96 precision. Recommended starting point.
#[derive(Clone, Debug)]
pub struct ProgressiveU16Vector {
    /// Global ID of the node.
    pub node_id: u64,
    /// Progressive u16 values (scent in low byte, SPO quantile in high byte).
    pub values: Vec<u16>,
}

impl ProgressiveU16Vector {
    /// Create from a full NeighborhoodVector.
    pub fn from_neighborhood(nv: &NeighborhoodVector) -> Self {
        Self {
            node_id: nv.node_id,
            values: nv.edges.iter().map(|&e| zeckf64::progressive_u16(e)).collect(),
        }
    }

    /// Number of non-zero entries.
    pub fn edge_count(&self) -> u16 {
        self.values.iter().filter(|&&v| v != 0).count() as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_triple(s: u64, p: u64, o: u64) -> (BitVec, BitVec, BitVec) {
        (BitVec::random(s), BitVec::random(p), BitVec::random(o))
    }

    #[test]
    fn test_scope_creation() {
        let ids: Vec<u64> = (0..100).collect();
        let scope = Scope::new(1, ids.clone());
        assert_eq!(scope.len(), 100);
        assert_eq!(scope.position_of(42), Some(42));
        assert_eq!(scope.global_id(42), Some(42));
        assert_eq!(scope.position_of(999), None);
    }

    #[test]
    fn test_build_scope() {
        let n = 50;
        let ids: Vec<u64> = (0..n as u64).collect();
        let planes: Vec<_> = (0..n)
            .map(|i| random_triple(i as u64 * 3, i as u64 * 3 + 1, i as u64 * 3 + 2))
            .collect();

        let (scope, neighborhoods) = build_scope(1, &ids, &planes);
        assert_eq!(scope.len(), n);
        assert_eq!(neighborhoods.len(), n);

        // Each neighborhood should have n edges
        for nv in &neighborhoods {
            assert_eq!(nv.edges.len(), n);
        }

        // Self-edges should be 0
        for (i, nv) in neighborhoods.iter().enumerate() {
            assert_eq!(nv.edges[i], 0, "self-edge should be 0 for node {}", i);
        }

        // Non-self edges should be non-zero
        for nv in &neighborhoods {
            assert!(nv.edge_count() > 0);
        }
    }

    #[test]
    fn test_neighborhood_vector_columns() {
        let planes = vec![
            random_triple(0, 1, 2),
            random_triple(3, 4, 5),
            random_triple(6, 7, 8),
        ];
        let ids = vec![100, 200, 300];
        let (_, neighborhoods) = build_scope(1, &ids, &planes);

        let nv = &neighborhoods[0];
        let scents = nv.scent_column();
        let resolutions = nv.resolution_column();

        assert_eq!(scents.len(), 3);
        assert_eq!(resolutions.len(), 3);
        assert_eq!(nv.node_id, 100);
        assert_eq!(nv.scope_id, 1);
    }

    #[test]
    fn test_scent_vector() {
        let planes = vec![random_triple(0, 1, 2), random_triple(3, 4, 5)];
        let ids = vec![10, 20];
        let (_, neighborhoods) = build_scope(1, &ids, &planes);

        let sv = ScentVector::from_neighborhood(&neighborhoods[0]);
        assert_eq!(sv.node_id, 10);
        assert_eq!(sv.scents.len(), 2);
    }

    #[test]
    fn test_progressive_u16_vector() {
        let planes = vec![random_triple(0, 1, 2), random_triple(3, 4, 5)];
        let ids = vec![10, 20];
        let (_, neighborhoods) = build_scope(1, &ids, &planes);

        let pv = ProgressiveU16Vector::from_neighborhood(&neighborhoods[0]);
        assert_eq!(pv.node_id, 10);
        assert_eq!(pv.values.len(), 2);
        // Self-edge should be 0
        assert_eq!(pv.values[0], 0);
    }

    #[test]
    #[should_panic(expected = "scope size")]
    fn test_scope_too_large() {
        let ids: Vec<u64> = (0..10_001).collect();
        Scope::new(1, ids);
    }
}
