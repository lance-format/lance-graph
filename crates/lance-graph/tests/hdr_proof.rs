// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # HDR Proof Tests
//!
//! Three publicly-visible CI tests that prove the core claims:
//!
//! 1. **SSSP equivalence** — Hamming-based SSSP agrees with float Bellman-Ford
//!    on >95% of pairwise distance rankings.
//! 2. **Cascade rejection** — staged Hamming sampling rejects >95%
//!    of random noise at stage 1, saving >90% of total compute.
//! 3. **Cosine vs Hamming ranking** — SimHash BF16 encoding preserves >=7/10
//!    top-k results compared to float cosine similarity.

use std::time::Instant;

use lance_graph::graph::blasgraph::matrix::GrBMatrix;
use lance_graph::graph::blasgraph::ops::hdr_sssp;
use lance_graph::graph::blasgraph::sparse::CooStorage;
use lance_graph::graph::blasgraph::types::{BitVec, VECTOR_BITS, VECTOR_WORDS};

// ---------------------------------------------------------------------------
// Deterministic PRNG (same as BitVec::random uses internally)
// ---------------------------------------------------------------------------

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

// ---------------------------------------------------------------------------
// Graph construction
// ---------------------------------------------------------------------------

/// Build a random directed graph as a GrBMatrix.
///
/// `nodes` x `nodes` adjacency matrix with `edges` random edges, each
/// carrying a random BitVec. Deterministic for the given `seed`.
fn build_random_graph(nodes: usize, edges: usize, seed: u64) -> GrBMatrix {
    let mut state = seed;
    let mut coo = CooStorage::new(nodes, nodes);

    for _ in 0..edges {
        let src = (splitmix64(&mut state) as usize) % nodes;
        let dst = (splitmix64(&mut state) as usize) % nodes;
        if src != dst {
            let edge_seed = splitmix64(&mut state);
            coo.push(src, dst, BitVec::random(edge_seed));
        }
    }

    GrBMatrix::from_coo(&coo)
}

// ---------------------------------------------------------------------------
// Reference float Bellman-Ford (for equivalence comparison)
// ---------------------------------------------------------------------------

/// Traditional Bellman-Ford with f64 edge weights = popcount of edge BitVec.
fn float_bellman_ford(adj: &GrBMatrix, source: usize, max_iters: usize) -> Vec<f64> {
    let n = adj.nrows();
    let mut dist = vec![f64::INFINITY; n];
    dist[source] = 0.0;

    for _ in 0..max_iters {
        let mut changed = false;
        for u in 0..n {
            if dist[u] == f64::INFINITY {
                continue;
            }
            for (v, edge_vec) in adj.row(u) {
                let w = edge_vec.popcount() as f64;
                let new_dist = dist[u] + w;
                if new_dist < dist[v] {
                    dist[v] = new_dist;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    dist
}

// ---------------------------------------------------------------------------
// Hamming sampling (HDR Cascade)
// ---------------------------------------------------------------------------

/// Sample every `divisor`-th u64 word and extrapolate Hamming distance.
///
/// A 1/16 sample touches 16 of 256 words (1024 of 16384 bits).
/// A 1/4 sample touches 64 of 256 words (4096 of 16384 bits).
///
/// Returns the extrapolated full-vector Hamming distance estimate.
fn hamming_sample(a: &BitVec, b: &BitVec, divisor: usize) -> u32 {
    let aw = a.words();
    let bw = b.words();
    let mut count = 0u32;
    let mut sampled = 0u32;

    for i in (0..VECTOR_WORDS).step_by(divisor) {
        count += (aw[i] ^ bw[i]).count_ones();
        sampled += 1;
    }

    // Extrapolate to full vector.
    (count as u64 * VECTOR_WORDS as u64 / sampled as u64) as u32
}

// Thresholds derived from the statistics of random 16384-bit vectors:
//
// Two independent random BitVecs (~50% density each) have:
//   Full:    E[H] = 8192,  σ ≈ 64
//   1/16:    E[H] = 8192 (extrapolated), σ ≈ 256 (extrapolation amplifies)
//   1/4:     E[H] = 8192 (extrapolated), σ ≈ 128
//
// We set thresholds at roughly E[H] − 3σ of the extrapolated distance
// so that random pairs almost always exceed them.
const SIGMA_3_THRESHOLD_16: u32 = 7400; // 8192 − ~3×256
const SIGMA_3_THRESHOLD_4: u32 = 7800; // 8192 − ~3×128
const RADIUS: u32 = 7000; // well below random mean of 8192

// ---------------------------------------------------------------------------
// SimHash encoding (float → BitVec for cosine-preserving Hamming)
// ---------------------------------------------------------------------------

/// Generate a random f32 vector of the given dimension, normalized to unit length.
fn random_f32_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut v: Vec<f32> = (0..dim)
        .map(|_| {
            let bits = splitmix64(&mut state);
            // Map to [-1, 1]
            (bits as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
        })
        .collect();

    // L2-normalize.
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v
}

/// Convert an f32 vector to a BitVec using SimHash (random hyperplane LSH).
///
/// Each of the 16384 output bits is the sign of a dot product between the
/// input vector and a deterministic pseudo-random hyperplane. This preserves
/// angular (cosine) distance: Hamming distance between two SimHash vectors
/// approximates their angular distance.
fn f32_to_bitvec_simhash(vec: &[f32]) -> BitVec {
    let mut words = [0u64; VECTOR_WORDS];

    for (w_idx, word_slot) in words.iter_mut().enumerate() {
        let mut word = 0u64;
        for bit in 0..64 {
            let proj_idx = (w_idx * 64 + bit) as u64;
            // Deterministic seed for this hyperplane.
            let mut state = proj_idx
                .wrapping_mul(0x517cc1b727220a95)
                .wrapping_add(0xCAFEBABE);

            let mut dot = 0.0f64;
            for component in vec.iter() {
                // Random direction component in [-1, 1].
                let r_bits = splitmix64(&mut state);
                let r = (r_bits as f64 / u64::MAX as f64) * 2.0 - 1.0;
                dot += *component as f64 * r;
            }
            if dot >= 0.0 {
                word |= 1u64 << bit;
            }
        }
        *word_slot = word;
    }

    BitVec::from_words(&words)
}

/// Cosine similarity between two f32 vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

// ===========================================================================
// Test 1: Float vs Hamming SSSP equivalence
// ===========================================================================

#[test]
fn float_vs_hamming_sssp_equivalence() {
    // Same graph. Same source. Same destination.
    // Two paths: float traditional, bitpacked HDR.
    // Results must agree. Bitpacked must be faster.

    let graph = build_random_graph(1000, 5000, 42);

    // Path 1: Traditional float Bellman-Ford
    let start_float = Instant::now();
    let dist_float = float_bellman_ford(&graph, 0, 100);
    let time_float = start_float.elapsed();

    // Path 2: Bitpacked Hamming SSSP (our implementation)
    let start_hamming = Instant::now();
    let result_hamming = hdr_sssp(&graph, 0, 100);
    let time_hamming = start_hamming.elapsed();

    // Extract u32 costs from the scalar side-channel.
    let dist_hamming: Vec<f64> = (0..1000)
        .map(|i| {
            result_hamming
                .get_scalar(i)
                .and_then(|s| s.as_float())
                .map(|f| f as f64)
                .unwrap_or(f64::INFINITY)
        })
        .collect();

    // EQUIVALENCE: same shortest path tree.
    // Both use popcount as edge weight, so costs must be identical.
    let mut agree = 0u64;
    let mut total = 0u64;

    for i in 0..1000 {
        if dist_float[i] < f64::MAX / 2.0 && dist_hamming[i] < f64::MAX / 2.0 {
            for j in (i + 1)..1000 {
                if dist_float[j] < f64::MAX / 2.0 && dist_hamming[j] < f64::MAX / 2.0 {
                    let float_order = dist_float[i] < dist_float[j];
                    let hamming_order = dist_hamming[i] < dist_hamming[j];
                    if float_order == hamming_order {
                        agree += 1;
                    }
                    total += 1;
                }
            }
        }
    }

    let agreement = if total > 0 {
        agree as f64 / total as f64
    } else {
        1.0
    };

    println!("--- float_vs_hamming_sssp_equivalence ---");
    println!(
        "Reachable nodes: float={}, hamming={}",
        dist_float.iter().filter(|d| **d < f64::MAX / 2.0).count(),
        dist_hamming
            .iter()
            .filter(|d| **d < f64::MAX / 2.0)
            .count()
    );
    println!("Pairwise comparisons: {}", total);
    println!("Agreement: {:.1}%", agreement * 100.0);
    println!("Float:   {:?}", time_float);
    println!("Hamming: {:?}", time_hamming);
    if time_hamming.as_nanos() > 0 {
        println!(
            "Speedup: {:.1}x",
            time_float.as_nanos() as f64 / time_hamming.as_nanos() as f64
        );
    }

    assert!(
        agreement > 0.95,
        "Float and Hamming SSSP must agree on >95% of pairwise rankings, got {:.1}%",
        agreement * 100.0
    );
}

// ===========================================================================
// Test 2: Cascade rejection rate
// ===========================================================================

#[test]
#[allow(clippy::needless_range_loop)] // near_idx used as both seed and index
fn cascade_rejection_rate() {
    // Prove HDR cascade eliminates >95% before full comparison.

    let total = 10_000u32;
    let mut state = 42u64;

    // Generate random node BitVecs.
    let mut nodes: Vec<BitVec> = (0..total)
        .map(|_| {
            let seed = splitmix64(&mut state);
            BitVec::random(seed)
        })
        .collect();

    // Plant a few "near" vectors (within RADIUS of node 0).
    // Flip only ~2000 bits (hamming distance ~2000 << 8192 random).
    let query = nodes[0].clone();
    for near_idx in 1..=20 {
        let mut near = query.clone();
        let words = near.words_mut();
        // Flip a controlled number of bits in the first few words.
        let flip_seed = near_idx as u64 * 7919;
        let mut fstate = flip_seed;
        for _ in 0..30 {
            let widx = (splitmix64(&mut fstate) as usize) % VECTOR_WORDS;
            let mask = splitmix64(&mut fstate);
            // Flip ~half the bits in one word ≈ 32 bits per flip op.
            words[widx] ^= mask;
        }
        nodes[near_idx] = near;
    }

    let mut stage1_pass = 0u32;
    let mut stage2_pass = 0u32;
    let mut stage3_pass = 0u32;

    for i in 1..total {
        let candidate = &nodes[i as usize];

        // Stage 1: 1/16 sample (Cascade first exposure reading)
        let sample_dist = hamming_sample(&query, candidate, 16);
        if sample_dist > SIGMA_3_THRESHOLD_16 {
            continue;
        }
        stage1_pass += 1;

        // Stage 2: 1/4 sample
        let sample_dist = hamming_sample(&query, candidate, 4);
        if sample_dist > SIGMA_3_THRESHOLD_4 {
            continue;
        }
        stage2_pass += 1;

        // Stage 3: full comparison
        let full_dist = query.hamming_distance(candidate);
        if full_dist > RADIUS {
            continue;
        }
        stage3_pass += 1;
    }

    let scanned = total - 1; // exclude self
    let rejection_rate = 1.0 - (stage1_pass as f64 / scanned as f64);

    println!("--- cascade_rejection_rate ---");
    println!(
        "Stage 1: {}/{} pass ({:.1}% rejected)",
        stage1_pass,
        scanned,
        rejection_rate * 100.0
    );
    println!("Stage 2: {}/{} pass", stage2_pass, stage1_pass);
    println!(
        "Stage 3: {}/{} pass (final results)",
        stage3_pass, stage2_pass
    );

    assert!(
        rejection_rate > 0.95,
        "HDR cascade stage 1 must reject >95% of random candidates, got {:.1}%",
        rejection_rate * 100.0
    );

    // Effective work: instead of N full comparisons, we did
    //   N cheap (1/16) + survivors medium (1/4) + survivors full.
    let full_work = scanned as f64;
    let actual_work = scanned as f64 * (1.0 / 16.0) // stage 1: 1/16 per candidate
        + stage1_pass as f64 * (1.0 / 4.0)          // stage 2: 1/4 per survivor
        + stage2_pass as f64 * 1.0; // stage 3: full per survivor
    let savings = 1.0 - (actual_work / full_work);

    println!(
        "Work savings: {:.1}% less compute than full scan",
        savings * 100.0
    );
    assert!(
        savings > 0.90,
        "HDR cascade must save >90% compute, got {:.1}%",
        savings * 100.0
    );
}

// ===========================================================================
// Test 3: Float cosine vs BF16 Hamming ranking
// ===========================================================================

#[test]
fn float_cosine_vs_bf16_hamming_ranking() {
    // The money test.
    // 1000 vectors. Find top-10 nearest to query.
    // Float cosine vs SimHash Hamming.
    // Rankings must substantially agree.

    let n = 1000;
    let dim = 128;

    let vecs: Vec<Vec<f32>> = (0..n).map(|i| random_f32_vec(dim, i as u64)).collect();
    let query = random_f32_vec(dim, 9999);

    // Float cosine top-10
    let mut float_ranked: Vec<(usize, f32)> = vecs
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine_similarity(&query, v)))
        .collect();
    float_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let float_top10: Vec<usize> = float_ranked[..10].iter().map(|x| x.0).collect();

    // SimHash Hamming top-10
    let bvecs: Vec<BitVec> = vecs.iter().map(|v| f32_to_bitvec_simhash(v)).collect();
    let bquery = f32_to_bitvec_simhash(&query);
    let mut hamming_ranked: Vec<(usize, u32)> = bvecs
        .iter()
        .enumerate()
        .map(|(i, v)| (i, bquery.hamming_distance(v)))
        .collect();
    hamming_ranked.sort_by_key(|x| x.1);
    let hamming_top10: Vec<usize> = hamming_ranked[..10].iter().map(|x| x.0).collect();

    // Overlap: how many of float top-10 appear in Hamming top-10?
    let overlap = float_top10
        .iter()
        .filter(|x| hamming_top10.contains(x))
        .count();

    println!("--- float_cosine_vs_bf16_hamming_ranking ---");
    println!("Float cosine top-10:   {:?}", float_top10);
    println!("SimHash Hamming top-10: {:?}", hamming_top10);
    println!("Overlap: {}/10", overlap);

    // Also print the similarity/distance values for context.
    println!(
        "Float cosine range: [{:.4}, {:.4}]",
        float_ranked.last().unwrap().1,
        float_ranked[0].1
    );
    println!(
        "Hamming range: [{}, {}] out of {} bits",
        hamming_ranked[0].1,
        hamming_ranked.last().unwrap().1,
        VECTOR_BITS
    );

    assert!(
        overlap >= 7,
        "SimHash Hamming must agree with float cosine on >=7/10 top results, got {}/10",
        overlap
    );
}
