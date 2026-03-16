// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Semiring algebra for SPO graph traversal.
//!
//! A semiring over (⊕, ⊗) provides:
//! - ⊕ (combine): how to merge parallel paths
//! - ⊗ (extend): how to chain sequential hops
//!
//! `HammingMin` uses Hamming distance as the cost metric:
//! - ⊕ = min (take the shortest path)
//! - ⊗ = add (distances accumulate through chain)

use super::truth::TruthValue;

/// A semiring for graph traversal cost computation.
pub trait SpoSemiring {
    /// The cost type (e.g., u32 for Hamming distance).
    type Cost: Copy + Ord + Default;

    /// Identity element for ⊗ (extend). Zero hops = zero cost.
    fn one() -> Self::Cost;

    /// Identity element for ⊕ (combine). Worst possible cost.
    fn zero() -> Self::Cost;

    /// Combine parallel paths: keep the best one.
    fn combine(a: Self::Cost, b: Self::Cost) -> Self::Cost;

    /// Extend a path by one hop: accumulate cost.
    fn extend(path: Self::Cost, hop: Self::Cost) -> Self::Cost;
}

/// Hamming distance semiring: min-plus over bit distances.
///
/// - combine = min (shortest semantic path)
/// - extend = saturating_add (distances accumulate)
pub struct HammingMin;

impl SpoSemiring for HammingMin {
    type Cost = u32;

    fn one() -> u32 {
        0
    }

    fn zero() -> u32 {
        u32::MAX
    }

    fn combine(a: u32, b: u32) -> u32 {
        a.min(b)
    }

    fn extend(path: u32, hop: u32) -> u32 {
        path.saturating_add(hop)
    }
}

/// A hop in a traversal chain.
#[derive(Debug, Clone)]
pub struct TraversalHop {
    /// The target entity fingerprint hash (dn_hash of the target).
    pub target_key: u64,
    /// Hamming distance of this hop.
    pub distance: u32,
    /// Truth value of the edge traversed.
    pub truth: TruthValue,
    /// Cumulative distance from the start.
    pub cumulative_distance: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_min_combine() {
        assert_eq!(HammingMin::combine(5, 3), 3);
        assert_eq!(HammingMin::combine(3, 5), 3);
    }

    #[test]
    fn test_hamming_min_extend() {
        assert_eq!(HammingMin::extend(10, 5), 15);
    }

    #[test]
    fn test_hamming_min_extend_saturating() {
        assert_eq!(HammingMin::extend(u32::MAX, 1), u32::MAX);
    }

    #[test]
    fn test_hamming_min_identity() {
        let cost = 42u32;
        assert_eq!(HammingMin::extend(HammingMin::one(), cost), cost);
        assert_eq!(HammingMin::combine(HammingMin::zero(), cost), cost);
    }
}
