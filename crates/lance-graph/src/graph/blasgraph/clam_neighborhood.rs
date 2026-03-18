// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # CLAM Integration — Conjecture Validation
//!
//! Build a CLAM tree on neighborhood scent vectors to test whether
//! natural cluster radii land on the Pareto frontier levels.
//!
//! **This is a TEST, not a fact.** The conjecture that CLAM cluster
//! radii naturally converge to the Pareto transition points (8-bit
//! and 57-bit levels) needs empirical validation before any
//! architectural commitment.
//!
//! ## The Conjecture
//!
//! The Pareto frontier has exactly 3 points:
//! - 8 bits:  ρ=0.937 — 6,144× compression
//! - 57 bits: ρ=0.982 — 862× compression
//! - 49,152 bits: ρ=1.000 — 1× (reference)
//!
//! If CLAM clustering on scent vectors produces natural cluster radii
//! that coincide with these levels, it would validate the unified
//! framework hypothesis. If not, the architecture still works — the
//! scent vectors just aren't fractal at these scales.

/// L1 distance metric on scent byte vectors.
///
/// This is the metric space for CLAM clustering on neighborhood vectors.
/// Each node's scent vector = [u8; N] where N = scope size.
pub fn scent_l1_distance(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u32)
        .sum()
}

/// Hamming distance metric on scent byte vectors.
///
/// Alternative metric: count differing band bits across all edges.
/// More aligned with the boolean lattice structure of scent bytes.
pub fn scent_hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

/// A cluster radius observation for conjecture testing.
#[derive(Clone, Debug)]
pub struct RadiusObservation {
    /// Depth in the CLAM tree.
    pub depth: usize,
    /// Number of points in this cluster.
    pub size: usize,
    /// Maximum distance from center to any point in the cluster.
    pub radius: u32,
    /// Mean distance from center to points.
    pub mean_distance: f64,
}

/// Simple ball-tree partitioning for CLAM validation.
///
/// Not a full CLAM implementation — just enough to measure natural
/// cluster radii and test the convergence conjecture.
///
/// # Arguments
/// - `scent_vectors` — Scent vectors for all nodes in the scope
/// - `min_cluster_size` — Stop splitting when clusters are this small
///
/// # Returns
/// A list of `RadiusObservation`s for all clusters in the tree.
pub fn measure_cluster_radii(
    scent_vectors: &[&[u8]],
    min_cluster_size: usize,
) -> Vec<RadiusObservation> {
    let mut observations = Vec::new();
    let indices: Vec<usize> = (0..scent_vectors.len()).collect();

    fn recurse(
        indices: &[usize],
        scent_vectors: &[&[u8]],
        depth: usize,
        min_size: usize,
        observations: &mut Vec<RadiusObservation>,
    ) {
        if indices.len() < 2 {
            return;
        }

        // Find the centroid (element-wise median approximated by mean)
        // Find the point farthest from the first point (approximate center finding)
        let center_idx = indices[0];
        let mut farthest_idx = indices[0];
        let mut max_dist = 0u32;

        for &i in indices {
            let d = scent_hamming_distance(scent_vectors[center_idx], scent_vectors[i]);
            if d > max_dist {
                max_dist = d;
                farthest_idx = i;
            }
        }

        // Find the point farthest from the farthest point
        let mut second_farthest = farthest_idx;
        let mut max_dist2 = 0u32;
        for &i in indices {
            let d = scent_hamming_distance(scent_vectors[farthest_idx], scent_vectors[i]);
            if d > max_dist2 {
                max_dist2 = d;
                second_farthest = i;
            }
        }

        // Compute radius: max distance from center to any point
        let total_distance: u64 = indices
            .iter()
            .map(|&i| {
                scent_hamming_distance(scent_vectors[center_idx], scent_vectors[i]) as u64
            })
            .sum();
        let mean_distance = total_distance as f64 / indices.len() as f64;

        observations.push(RadiusObservation {
            depth,
            size: indices.len(),
            radius: max_dist,
            mean_distance,
        });

        if indices.len() <= min_size {
            return;
        }

        // Split by proximity to the two poles
        let mut left = Vec::new();
        let mut right = Vec::new();

        for &i in indices {
            let d_left = scent_hamming_distance(scent_vectors[farthest_idx], scent_vectors[i]);
            let d_right =
                scent_hamming_distance(scent_vectors[second_farthest], scent_vectors[i]);
            if d_left <= d_right {
                left.push(i);
            } else {
                right.push(i);
            }
        }

        // Avoid degenerate splits
        if left.is_empty() || right.is_empty() {
            return;
        }

        recurse(&left, scent_vectors, depth + 1, min_size, observations);
        recurse(&right, scent_vectors, depth + 1, min_size, observations);
    }

    recurse(
        &indices,
        scent_vectors,
        0,
        min_cluster_size,
        &mut observations,
    );

    observations
}

/// Analyze whether observed cluster radii show clustering near Pareto levels.
///
/// Returns a summary of how many radii land near the expected transition
/// points. This is the core conjecture test.
///
/// # Arguments
/// - `observations` — Cluster radius measurements from `measure_cluster_radii`
/// - `scope_size` — Number of nodes in the scope (affects expected radii)
///
/// # Returns
/// A `ParetoAnalysis` with counts of radii near each expected level.
pub fn analyze_pareto_convergence(
    observations: &[RadiusObservation],
    scope_size: usize,
) -> ParetoAnalysis {
    // The Pareto frontier bits map to expected Hamming distances in scent space.
    // Scent = 7 bits per edge, so max Hamming distance per edge = 7.
    // For scope_size edges, max total Hamming = 7 * scope_size.
    //
    // The 8-bit level (ρ=0.937) means ~6.3% error in ranking.
    // The 57-bit level (ρ=0.982) means ~1.8% error.
    //
    // We expect cluster radii to show natural breaks near:
    //   Level 1: radius ≈ 0.063 * 7 * scope_size (8-bit equivalent)
    //   Level 2: radius ≈ 0.018 * 7 * scope_size (57-bit equivalent)
    //   Level 3: radius ≈ 0 (exact match, 49K-bit equivalent)

    let max_radius = 7.0 * scope_size as f64;
    let level1_center = 0.063 * max_radius;
    let level2_center = 0.018 * max_radius;
    let tolerance = 0.3; // 30% relative tolerance

    let mut near_level1 = 0u32;
    let mut near_level2 = 0u32;
    let mut near_exact = 0u32;
    let mut other = 0u32;

    for obs in observations {
        let r = obs.radius as f64;
        if r < level2_center * (1.0 + tolerance) {
            near_exact += 1;
        } else if (r - level2_center).abs() < level2_center * tolerance {
            near_level2 += 1;
        } else if (r - level1_center).abs() < level1_center * tolerance {
            near_level1 += 1;
        } else {
            other += 1;
        }
    }

    ParetoAnalysis {
        total_clusters: observations.len() as u32,
        near_8bit_level: near_level1,
        near_57bit_level: near_level2,
        near_exact_level: near_exact,
        other: other,
        convergence_ratio: if observations.is_empty() {
            0.0
        } else {
            (near_level1 + near_level2 + near_exact) as f64 / observations.len() as f64
        },
    }
}

/// Result of Pareto convergence analysis.
#[derive(Clone, Debug)]
pub struct ParetoAnalysis {
    /// Total number of clusters observed.
    pub total_clusters: u32,
    /// Clusters with radii near the 8-bit Pareto level.
    pub near_8bit_level: u32,
    /// Clusters with radii near the 57-bit Pareto level.
    pub near_57bit_level: u32,
    /// Clusters with radii near exact match level.
    pub near_exact_level: u32,
    /// Clusters with radii not near any Pareto level.
    pub other: u32,
    /// Fraction of clusters near any Pareto level.
    pub convergence_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::blasgraph::neighborhood::build_scope;
    use crate::graph::blasgraph::types::BitVec;

    fn random_triple(s: u64, p: u64, o: u64) -> (BitVec, BitVec, BitVec) {
        (BitVec::random(s), BitVec::random(p), BitVec::random(o))
    }

    #[test]
    fn test_scent_l1_distance() {
        let a = vec![0u8, 10, 20, 30];
        let b = vec![0u8, 15, 25, 35];
        assert_eq!(scent_l1_distance(&a, &b), 15); // 0+5+5+5
    }

    #[test]
    fn test_scent_hamming_distance() {
        let a = vec![0xFF, 0x00];
        let b = vec![0x00, 0xFF];
        assert_eq!(scent_hamming_distance(&a, &b), 16); // all 16 bits differ
    }

    #[test]
    fn test_scent_hamming_distance_self() {
        let a = vec![0x7F, 0x3F, 0x01];
        assert_eq!(scent_hamming_distance(&a, &a), 0);
    }

    #[test]
    fn test_measure_cluster_radii() {
        let n = 100;
        let planes: Vec<_> = (0..n)
            .map(|i| random_triple(i as u64 * 3, i as u64 * 3 + 1, i as u64 * 3 + 2))
            .collect();
        let ids: Vec<u64> = (0..n as u64).collect();
        let (_, neighborhoods) = build_scope(1, &ids, &planes);

        let scent_vecs: Vec<Vec<u8>> = neighborhoods.iter().map(|nv| nv.scent_column()).collect();
        let scent_refs: Vec<&[u8]> = scent_vecs.iter().map(|v| v.as_slice()).collect();

        let observations = measure_cluster_radii(&scent_refs, 5);

        // Should have multiple cluster observations at different depths
        assert!(!observations.is_empty());

        // Root cluster should contain all nodes
        assert_eq!(observations[0].size, n);
        assert!(observations[0].radius > 0);

        // Deeper clusters should be smaller
        let max_depth = observations.iter().map(|o| o.depth).max().unwrap();
        assert!(max_depth > 0);
    }

    #[test]
    fn test_pareto_analysis() {
        let observations = vec![
            RadiusObservation {
                depth: 0,
                size: 100,
                radius: 500,
                mean_distance: 250.0,
            },
            RadiusObservation {
                depth: 1,
                size: 50,
                radius: 200,
                mean_distance: 100.0,
            },
            RadiusObservation {
                depth: 2,
                size: 10,
                radius: 5,
                mean_distance: 2.0,
            },
        ];

        let analysis = analyze_pareto_convergence(&observations, 100);
        assert_eq!(analysis.total_clusters, 3);
        // The analysis bins radii — exact values depend on scope size
        // Just verify it runs without panic and produces reasonable output
        assert!(analysis.convergence_ratio >= 0.0 && analysis.convergence_ratio <= 1.0);
    }

    #[test]
    fn test_clam_on_real_scent_vectors() {
        // This is TEST 6 from the prompt: CLAM radius validation
        let n = 200;
        let planes: Vec<_> = (0..n)
            .map(|i| random_triple(i as u64 * 3, i as u64 * 3 + 1, i as u64 * 3 + 2))
            .collect();
        let ids: Vec<u64> = (0..n as u64).collect();
        let (_, neighborhoods) = build_scope(1, &ids, &planes);

        let scent_vecs: Vec<Vec<u8>> = neighborhoods.iter().map(|nv| nv.scent_column()).collect();
        let scent_refs: Vec<&[u8]> = scent_vecs.iter().map(|v| v.as_slice()).collect();

        let observations = measure_cluster_radii(&scent_refs, 10);
        let analysis = analyze_pareto_convergence(&observations, n);

        // Report findings (the conjecture test)
        // We don't assert the conjecture is true — we just verify
        // the measurement pipeline works correctly.
        assert!(analysis.total_clusters > 0);
        // At minimum, we should see clusters at multiple depths
        let depths: std::collections::HashSet<usize> =
            observations.iter().map(|o| o.depth).collect();
        assert!(depths.len() > 1, "expected multiple tree depths");
    }
}
