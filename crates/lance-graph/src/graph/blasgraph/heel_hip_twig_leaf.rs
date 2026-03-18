// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # Heel → Hip → Twig → Leaf — Progressive Neighborhood Search
//!
//! A 4-stage cascade that searches a graph via neighborhood vectors,
//! loading progressively more data at each stage.
//!
//! ```text
//! HEEL (1 vector):
//!   Load MY neighborhood scent [u8; N] from neighborhoods.lance.
//!   Compare against all N scope entries.
//!   Cost: 10KB loaded, 10K byte comparisons, ~20μs.
//!   Survivors: ~50 nodes (top matches by scent).
//!
//! HIP (50 vectors):
//!   Load 50 survivors' neighborhood scents.
//!   For each: compare against THEIR N scope entries.
//!   Each survivor opens a NEW 90° window into the graph.
//!   Cost: 50 × 10KB = 500KB, 500K comparisons, ~500μs.
//!   Survivors: ~50 nodes from union of ~50K unique explored.
//!
//! TWIG (50 vectors):
//!   Same operation, third hop.
//!   Cost: 500KB loaded, 500K comparisons, ~500μs.
//!   Total explored: ~200K unique nodes across 3 hops.
//!   Survivors: ~50 candidates.
//!
//! LEAF (50 candidates):
//!   Load cold data for final verification.
//!   L1: integrated 16Kbit fingerprints (rho=0.834).
//!   L2: exact S+P+O planes for top 10 (rho=1.000).
//!   Cost: 160KB loaded, ~100μs.
//!
//! TOTAL: ~1.2MB loaded, ~1.1ms, 200K nodes explored, 3 hops.
//! ```

use std::collections::HashSet;

use super::zeckf64;

/// Result of a search stage: (scope_position, distance_score).
pub type SearchHit = (usize, u32);

/// Configuration for the search cascade.
#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// Number of survivors to keep at each stage.
    pub k: usize,
    /// Whether to use progressive u16 distance (bytes 0-1) instead of scent only.
    pub use_progressive: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            k: 50,
            use_progressive: false,
        }
    }
}

// ---------------------------------------------------------------------------
// HEEL: first hop — search my own neighborhood
// ---------------------------------------------------------------------------

/// HEEL: Find top-K closest neighbors from a single neighborhood vector.
///
/// Uses scent (byte 0) only for maximum speed. Each entry is scored by
/// popcount-similarity to the query scent pattern.
///
/// # Arguments
/// - `query_scent` — The scent byte pattern to match against
/// - `my_scents` — Scent bytes of my neighborhood (one per scope position)
/// - `k` — Number of survivors to return
///
/// # Returns
/// Sorted `Vec<SearchHit>` of (position, distance) pairs, closest first.
pub fn heel_search(query_scent: u8, my_scents: &[u8], k: usize) -> Vec<SearchHit> {
    let mut hits: Vec<SearchHit> = my_scents
        .iter()
        .enumerate()
        .filter(|(_, &s)| s != 0) // skip empty edges
        .map(|(i, &s)| {
            let dist = (s ^ query_scent).count_ones();
            (i, dist)
        })
        .collect();

    hits.sort_by_key(|&(_, d)| d);
    hits.truncate(k);
    hits
}

/// HEEL with full ZeckF64 vectors (uses scent byte for scoring).
///
/// Convenience function that extracts scent from the neighborhood edges.
pub fn heel_search_edges(my_edges: &[u64], k: usize) -> Vec<SearchHit> {
    // Score by inverse scent: more "close" bands = better match = lower distance
    let mut hits: Vec<SearchHit> = my_edges
        .iter()
        .enumerate()
        .filter(|(_, &e)| e != 0)
        .map(|(i, &e)| {
            let s = zeckf64::scent(e);
            // Invert: 7 - popcount(bands) so that "all close" = distance 0
            let bands = s & 0x7F;
            let dist = 7 - bands.count_ones();
            (i, dist)
        })
        .collect();

    hits.sort_by_key(|&(_, d)| d);
    hits.truncate(k);
    hits
}

// ---------------------------------------------------------------------------
// HIP: second hop — explore survivors' neighborhoods
// ---------------------------------------------------------------------------

/// HIP: Given heel survivors, load their scent vectors and find next-hop survivors.
///
/// Each survivor's neighborhood opens a new window into the graph.
/// Scopes may overlap (shared neighbors) or diverge (new territory).
///
/// # Arguments
/// - `survivor_positions` — Scope positions of heel survivors
/// - `all_scents` — Scent vectors for all scope nodes, indexed by position.
///   `all_scents[i]` is the scent vector for scope node `i`.
/// - `target_scent` — The scent pattern we're looking for
/// - `k` — Number of survivors to return
///
/// # Returns
/// Sorted `Vec<SearchHit>` of (position, best_distance) pairs.
pub fn hip_search(
    survivor_positions: &[usize],
    all_scents: &[&[u8]],
    target_scent: u8,
    k: usize,
) -> Vec<SearchHit> {
    let mut seen = HashSet::new();
    let mut hits: Vec<SearchHit> = Vec::new();

    for &idx in survivor_positions {
        if idx >= all_scents.len() {
            continue;
        }
        let neighbor_scents = all_scents[idx];

        for (j, &s) in neighbor_scents.iter().enumerate() {
            if s == 0 {
                continue;
            }
            if !seen.insert(j) {
                // Already seen — update if better distance
                let dist = (s ^ target_scent).count_ones();
                if let Some(existing) = hits.iter_mut().find(|h| h.0 == j) {
                    if dist < existing.1 {
                        existing.1 = dist;
                    }
                }
                continue;
            }
            let dist = (s ^ target_scent).count_ones();
            hits.push((j, dist));
        }
    }

    hits.sort_by_key(|&(_, d)| d);
    hits.truncate(k);
    hits
}

/// HIP with full ZeckF64 edges.
pub fn hip_search_edges(
    survivor_positions: &[usize],
    all_edges: &[&[u64]],
    k: usize,
) -> Vec<SearchHit> {
    let mut seen = HashSet::new();
    let mut hits: Vec<SearchHit> = Vec::new();

    for &idx in survivor_positions {
        if idx >= all_edges.len() {
            continue;
        }
        for (j, &e) in all_edges[idx].iter().enumerate() {
            if e == 0 {
                continue;
            }
            let bands = (zeckf64::scent(e) & 0x7F).count_ones();
            let dist = 7 - bands;
            if !seen.insert(j) {
                if let Some(existing) = hits.iter_mut().find(|h| h.0 == j) {
                    if dist < existing.1 {
                        existing.1 = dist;
                    }
                }
                continue;
            }
            hits.push((j, dist));
        }
    }

    hits.sort_by_key(|&(_, d)| d);
    hits.truncate(k);
    hits
}

// ---------------------------------------------------------------------------
// TWIG: third hop — identical to hip, one more hop
// ---------------------------------------------------------------------------

/// TWIG: Third hop, identical operation to HIP.
///
/// Takes hip survivors, loads their neighborhoods, finds next-hop survivors.
/// After twig, ~200K unique nodes have been explored across 3 hops.
pub fn twig_search(
    survivor_positions: &[usize],
    all_scents: &[&[u8]],
    target_scent: u8,
    k: usize,
) -> Vec<SearchHit> {
    // Structurally identical to hip_search
    hip_search(survivor_positions, all_scents, target_scent, k)
}

/// TWIG with full ZeckF64 edges.
pub fn twig_search_edges(
    survivor_positions: &[usize],
    all_edges: &[&[u64]],
    k: usize,
) -> Vec<SearchHit> {
    hip_search_edges(survivor_positions, all_edges, k)
}

// ---------------------------------------------------------------------------
// LEAF: final verification stage
// ---------------------------------------------------------------------------

/// Result of leaf verification with exact distance.
#[derive(Clone, Debug)]
pub struct LeafResult {
    /// Scope position of the verified node.
    pub position: usize,
    /// Global node ID (resolved from scope).
    pub node_id: u64,
    /// Approximate distance from scent search.
    pub scent_distance: u32,
    /// Exact Hamming distance on 16Kbit integrated plane (if available).
    pub integrated_distance: Option<u32>,
    /// Exact SPO distance (sum of S+P+O plane distances, if available).
    pub exact_spo_distance: Option<u32>,
}

/// LEAF: Final verification of twig candidates using cold data.
///
/// Two verification levels:
/// - L1: Compare 16Kbit integrated fingerprints (ρ=0.834, 2KB per node)
/// - L2: Compare exact S+P+O planes for top candidates (ρ=1.000, 6KB per node)
///
/// # Arguments
/// - `candidates` — Twig search results (position, scent_distance)
/// - `scope` — Scope for resolving positions to global IDs
/// - `query_integrated` — Query node's 16Kbit integrated fingerprint
/// - `candidate_integrated` — 16Kbit fingerprints for candidates, keyed by position
/// - `top_n_exact` — Number of top L1 candidates to verify with exact SPO
/// - `query_planes` — Query node's exact S+P+O planes
/// - `candidate_planes` — Exact S+P+O planes for top candidates, keyed by position
pub fn leaf_verify(
    candidates: &[SearchHit],
    scope_node_ids: &[u64],
    query_integrated: Option<&super::types::BitVec>,
    candidate_integrated: &[(usize, &super::types::BitVec)],
    top_n_exact: usize,
    query_planes: Option<(&super::types::BitVec, &super::types::BitVec, &super::types::BitVec)>,
    candidate_planes: &[(
        usize,
        &super::types::BitVec,
        &super::types::BitVec,
        &super::types::BitVec,
    )],
) -> Vec<LeafResult> {
    let mut results: Vec<LeafResult> = Vec::with_capacity(candidates.len());

    // L1: integrated fingerprint comparison
    for &(pos, scent_dist) in candidates {
        let node_id = scope_node_ids.get(pos).copied().unwrap_or(0);
        let integrated_distance = if let Some(qi) = query_integrated {
            candidate_integrated
                .iter()
                .find(|(p, _)| *p == pos)
                .map(|(_, ci)| qi.hamming_distance(ci))
        } else {
            None
        };

        results.push(LeafResult {
            position: pos,
            node_id,
            scent_distance: scent_dist,
            integrated_distance,
            exact_spo_distance: None,
        });
    }

    // Sort by integrated distance (or scent distance if no integrated)
    results.sort_by_key(|r| r.integrated_distance.unwrap_or(r.scent_distance * 1000));

    // L2: exact SPO verification for top N
    if let Some((qs, qp, qo)) = query_planes {
        for result in results.iter_mut().take(top_n_exact) {
            if let Some((_, cs, cp, co)) = candidate_planes
                .iter()
                .find(|(p, _, _, _)| *p == result.position)
            {
                let ds = qs.hamming_distance(cs);
                let dp = qp.hamming_distance(cp);
                let d_o = qo.hamming_distance(co);
                result.exact_spo_distance = Some(ds + dp + d_o);
            }
        }
    }

    // Final sort by best available distance
    results.sort_by_key(|r| {
        r.exact_spo_distance
            .unwrap_or_else(|| r.integrated_distance.unwrap_or(r.scent_distance * 1000))
    });

    results
}

// ---------------------------------------------------------------------------
// Full cascade
// ---------------------------------------------------------------------------

/// Run the full Heel → Hip → Twig cascade on scent vectors.
///
/// Returns the twig survivors (position, distance) ready for leaf verification.
///
/// # Arguments
/// - `my_scents` — Scent vector of the query node
/// - `query_scent` — Target scent pattern (typically the query node's best scent)
/// - `all_scents` — All scope nodes' scent vectors
/// - `config` — Search configuration
pub fn cascade_search(
    my_scents: &[u8],
    query_scent: u8,
    all_scents: &[&[u8]],
    config: &SearchConfig,
) -> Vec<SearchHit> {
    // HEEL
    let heel_survivors = heel_search(query_scent, my_scents, config.k);
    if heel_survivors.is_empty() {
        return Vec::new();
    }

    // HIP
    let heel_positions: Vec<usize> = heel_survivors.iter().map(|&(p, _)| p).collect();
    let hip_survivors = hip_search(&heel_positions, all_scents, query_scent, config.k);
    if hip_survivors.is_empty() {
        return heel_survivors;
    }

    // TWIG
    let hip_positions: Vec<usize> = hip_survivors.iter().map(|&(p, _)| p).collect();
    let twig_survivors = twig_search(&hip_positions, all_scents, query_scent, config.k);
    if twig_survivors.is_empty() {
        return hip_survivors;
    }

    twig_survivors
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
    fn test_heel_search_basic() {
        // Create scent vector with known patterns
        let scents = vec![
            0u8,   // self (no edge)
            0x7F,  // all bands close (distance 0 from 0x7F)
            0x3F,  // 6 bands close
            0x01,  // 1 band close
            0x00,  // no bands close
        ];

        let hits = heel_search(0x7F, &scents, 3);
        assert_eq!(hits.len(), 3);
        // First hit should be position 1 (exact match)
        assert_eq!(hits[0].0, 1);
        assert_eq!(hits[0].1, 0); // 0 distance
    }

    #[test]
    fn test_heel_search_edges() {
        let planes: Vec<_> = (0..20)
            .map(|i| random_triple(i * 3, i * 3 + 1, i * 3 + 2))
            .collect();
        let ids: Vec<u64> = (0..20).collect();
        let (_, neighborhoods) = build_scope(1, &ids, &planes);

        let hits = heel_search_edges(&neighborhoods[0].edges, 5);
        assert_eq!(hits.len(), 5);
        // Hits should be sorted by distance
        for i in 1..hits.len() {
            assert!(hits[i].1 >= hits[i - 1].1);
        }
    }

    #[test]
    fn test_hip_search() {
        let n = 50;
        let planes: Vec<_> = (0..n)
            .map(|i| random_triple(i as u64 * 3, i as u64 * 3 + 1, i as u64 * 3 + 2))
            .collect();
        let ids: Vec<u64> = (0..n as u64).collect();
        let (_, neighborhoods) = build_scope(1, &ids, &planes);

        // Run heel first
        let heel_hits = heel_search_edges(&neighborhoods[0].edges, 10);
        let survivor_positions: Vec<usize> = heel_hits.iter().map(|&(p, _)| p).collect();

        // Run hip with scent vectors
        let scent_vecs: Vec<Vec<u8>> = neighborhoods.iter().map(|nv| nv.scent_column()).collect();
        let scent_refs: Vec<&[u8]> = scent_vecs.iter().map(|v| v.as_slice()).collect();

        let target_scent = zeckf64::scent(neighborhoods[0].edges[1]); // scent toward node 1
        let hip_hits = hip_search(&survivor_positions, &scent_refs, target_scent, 10);

        // Should find some hits
        assert!(!hip_hits.is_empty());
        // Hits should be sorted
        for i in 1..hip_hits.len() {
            assert!(hip_hits[i].1 >= hip_hits[i - 1].1);
        }
    }

    #[test]
    fn test_cascade_search() {
        let n = 100;
        let planes: Vec<_> = (0..n)
            .map(|i| random_triple(i as u64 * 3, i as u64 * 3 + 1, i as u64 * 3 + 2))
            .collect();
        let ids: Vec<u64> = (0..n as u64).collect();
        let (_, neighborhoods) = build_scope(1, &ids, &planes);

        let scent_vecs: Vec<Vec<u8>> = neighborhoods.iter().map(|nv| nv.scent_column()).collect();
        let scent_refs: Vec<&[u8]> = scent_vecs.iter().map(|v| v.as_slice()).collect();

        let config = SearchConfig { k: 10, use_progressive: false };
        let query_scent = zeckf64::scent(neighborhoods[0].edges[1]);
        let results = cascade_search(&scent_vecs[0], query_scent, &scent_refs, &config);

        assert!(!results.is_empty());
        assert!(results.len() <= 10);
    }

    #[test]
    fn test_leaf_verify_without_cold_data() {
        let candidates = vec![(1usize, 2u32), (3, 3), (5, 4)];
        let scope_ids = vec![100u64, 200, 300, 400, 500, 600];

        let results = leaf_verify(
            &candidates,
            &scope_ids,
            None,
            &[],
            0,
            None,
            &[],
        );

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].node_id, 200);
        assert_eq!(results[0].scent_distance, 2);
    }

    #[test]
    fn test_leaf_verify_with_integrated() {
        let candidates = vec![(0usize, 5u32), (1, 3)];
        let scope_ids = vec![100u64, 200];

        let query_int = BitVec::random(42);
        let cand0_int = BitVec::random(43); // far from query
        let cand1_int = query_int.clone(); // identical to query

        let candidate_integrated: Vec<(usize, &BitVec)> =
            vec![(0, &cand0_int), (1, &cand1_int)];

        let results = leaf_verify(
            &candidates,
            &scope_ids,
            Some(&query_int),
            &candidate_integrated,
            0,
            None,
            &[],
        );

        // Node 1 should rank first (identical integrated fingerprint)
        assert_eq!(results[0].node_id, 200);
        assert_eq!(results[0].integrated_distance, Some(0));
    }

    #[test]
    fn test_three_hop_traversal() {
        // Build 3 "scopes" (here just one big scope simulating overlapping scopes)
        let n = 200;
        let planes: Vec<_> = (0..n)
            .map(|i| random_triple(i as u64 * 3, i as u64 * 3 + 1, i as u64 * 3 + 2))
            .collect();
        let ids: Vec<u64> = (0..n as u64).collect();
        let (_, neighborhoods) = build_scope(1, &ids, &planes);

        let scent_vecs: Vec<Vec<u8>> = neighborhoods.iter().map(|nv| nv.scent_column()).collect();
        let scent_refs: Vec<&[u8]> = scent_vecs.iter().map(|v| v.as_slice()).collect();

        // HEEL from node 0
        let heel = heel_search(0x7F, &scent_vecs[0], 20);
        assert!(!heel.is_empty());

        // HIP
        let heel_pos: Vec<usize> = heel.iter().map(|&(p, _)| p).collect();
        let hip = hip_search(&heel_pos, &scent_refs, 0x7F, 20);

        // TWIG
        let hip_pos: Vec<usize> = hip.iter().map(|&(p, _)| p).collect();
        let twig = twig_search(&hip_pos, &scent_refs, 0x7F, 20);

        // Collect all explored nodes across 3 hops
        let mut explored = HashSet::new();
        for &(p, _) in &heel {
            explored.insert(p);
        }
        for &(p, _) in &hip {
            explored.insert(p);
        }
        for &(p, _) in &twig {
            explored.insert(p);
        }

        // Should have explored a significant portion
        assert!(
            explored.len() > 10,
            "only explored {} nodes across 3 hops",
            explored.len()
        );
    }
}
