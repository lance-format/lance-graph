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

// ---------------------------------------------------------------------------
// Integration tests: spo_redisgraph_parity
//
// These tests model a RedisGraph-style social graph (nodes: persons, edges:
// KNOWS/CREATES/HELPS) and exercise all SPO layers end-to-end:
// fingerprinting, bitmap packing, truth values, forward/reverse/relation
// queries, gated queries, chain traversal, merkle integrity, and the
// HammingMin semiring.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod spo_redisgraph_parity {
    use crate::graph::fingerprint::{dn_hash, hamming_distance, label_fp, ZERO_FP};
    use crate::graph::spo::builder::SpoBuilder;
    use crate::graph::spo::merkle::{BindSpace, MerkleRoot, VerifyStatus};
    use crate::graph::spo::semiring::{HammingMin, SpoSemiring};
    use crate::graph::spo::store::SpoStore;
    use crate::graph::spo::truth::{TruthGate, TruthValue};

    /// Helper: build a store with a small social graph.
    ///
    ///   Jan -[KNOWS]-> Ada   (truth 0.9, 0.8)
    ///   Ada -[KNOWS]-> Max   (truth 0.8, 0.7)
    ///   Jan -[CREATES]-> Project  (truth 1.0, 1.0)
    ///   Max -[HELPS]-> Jan   (truth 0.6, 0.3)
    fn social_graph() -> SpoStore {
        let mut store = SpoStore::new();

        let jan = label_fp("Jan");
        let ada = label_fp("Ada");
        let max = label_fp("Max");
        let project = label_fp("Project");
        let knows = label_fp("KNOWS");
        let creates = label_fp("CREATES");
        let helps = label_fp("HELPS");

        store.insert(
            dn_hash("edge:jan-knows-ada"),
            &SpoBuilder::build_edge(&jan, &knows, &ada, TruthValue::new(0.9, 0.8)),
        );
        store.insert(
            dn_hash("edge:ada-knows-max"),
            &SpoBuilder::build_edge(&ada, &knows, &max, TruthValue::new(0.8, 0.7)),
        );
        store.insert(
            dn_hash("edge:jan-creates-project"),
            &SpoBuilder::build_edge(&jan, &creates, &project, TruthValue::certain()),
        );
        store.insert(
            dn_hash("edge:max-helps-jan"),
            &SpoBuilder::build_edge(&max, &helps, &jan, TruthValue::new(0.6, 0.3)),
        );

        store
    }

    // -----------------------------------------------------------------------
    // Test 1: Forward query — MATCH (Jan)-[:KNOWS]->(?) returns Ada
    // -----------------------------------------------------------------------
    #[test]
    fn test_1_forward_query_jan_knows() {
        let store = social_graph();
        let jan = label_fp("Jan");
        let knows = label_fp("KNOWS");
        let ada = label_fp("Ada");

        let hits = store.query_forward(&jan, &knows, 200);
        assert!(!hits.is_empty(), "Should find Jan-[KNOWS]->Ada");

        // The closest hit should have the Ada fingerprint as object
        let best = &hits[0];
        assert_eq!(best.record.object, ada);
    }

    // -----------------------------------------------------------------------
    // Test 2: Reverse query — MATCH (?)-[:KNOWS]->(Max) returns Ada
    // -----------------------------------------------------------------------
    #[test]
    fn test_2_reverse_query_who_knows_max() {
        let store = social_graph();
        let knows = label_fp("KNOWS");
        let max = label_fp("Max");
        let ada = label_fp("Ada");

        let hits = store.query_reverse(&knows, &max, 200);
        assert!(!hits.is_empty(), "Should find Ada-[KNOWS]->Max");

        let best = &hits[0];
        assert_eq!(best.record.subject, ada);
    }

    // -----------------------------------------------------------------------
    // Test 3: Relation query — MATCH (Jan)-[?]->(Ada) returns KNOWS
    // -----------------------------------------------------------------------
    #[test]
    fn test_3_relation_query_jan_to_ada() {
        let store = social_graph();
        let jan = label_fp("Jan");
        let ada = label_fp("Ada");
        let knows = label_fp("KNOWS");

        let hits = store.query_relation(&jan, &ada, 200);
        assert!(!hits.is_empty(), "Should find the KNOWS predicate");

        let best = &hits[0];
        assert_eq!(best.record.predicate, knows);
    }

    // -----------------------------------------------------------------------
    // Test 4: Truth gated forward — STRONG gate filters out weak edges
    // -----------------------------------------------------------------------
    #[test]
    fn test_4_truth_gated_forward_strong() {
        let store = social_graph();
        let jan = label_fp("Jan");
        let knows = label_fp("KNOWS");

        // Jan-[KNOWS]->Ada has truth (0.9, 0.8), expectation = 0.82 → passes STRONG (0.75)
        let strong_hits = store.query_forward_gated(&jan, &knows, 200, TruthGate::STRONG);
        assert!(!strong_hits.is_empty(), "STRONG gate should pass 0.82 expectation");

        // Max-[HELPS]->Jan has truth (0.6, 0.3), expectation = 0.53 → fails STRONG
        let max = label_fp("Max");
        let helps = label_fp("HELPS");
        let strong_helps = store.query_forward_gated(&max, &helps, 200, TruthGate::STRONG);
        assert!(
            strong_helps.is_empty(),
            "STRONG gate should reject 0.53 expectation"
        );
    }

    // -----------------------------------------------------------------------
    // Test 5: Truth gated forward — CERTAIN gate rejects all except perfect edges
    // -----------------------------------------------------------------------
    #[test]
    fn test_5_truth_gated_certain() {
        let store = social_graph();
        let jan = label_fp("Jan");
        let creates = label_fp("CREATES");

        // Jan-[CREATES]->Project has truth (1.0, 1.0), expectation = 1.0 → passes CERTAIN (0.9)
        let certain_hits = store.query_forward_gated(&jan, &creates, 200, TruthGate::CERTAIN);
        assert!(
            !certain_hits.is_empty(),
            "CERTAIN gate should pass (1.0, 1.0) edge"
        );

        // Jan-[KNOWS]->Ada has expectation 0.82 → fails CERTAIN
        let knows = label_fp("KNOWS");
        let certain_knows = store.query_forward_gated(&jan, &knows, 200, TruthGate::CERTAIN);
        assert!(
            certain_knows.is_empty(),
            "CERTAIN gate should reject 0.82 expectation"
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: Merkle integrity — verify_lineage gap vs verify_integrity
    // -----------------------------------------------------------------------
    #[test]
    fn test_6_merkle_integrity_gap() {
        let mut space = BindSpace::new();
        let fp = label_fp("Jan");
        let addr = space.write_dn_path("person:jan", fp, 1);

        // Both checks pass on fresh node
        assert_eq!(space.verify_lineage(addr), VerifyStatus::Consistent);
        assert_eq!(space.verify_integrity(addr), VerifyStatus::Consistent);

        // Corrupt a fingerprint word
        space.read_mut(addr).unwrap().fingerprint[3] ^= 0xDEAD;

        // verify_lineage misses it (documented gap — structural check only)
        assert_eq!(space.verify_lineage(addr), VerifyStatus::Consistent);
        // verify_integrity catches it
        assert_eq!(space.verify_integrity(addr), VerifyStatus::Corrupted);

        // Out-of-bounds address returns NotFound
        assert_eq!(space.verify_integrity(999), VerifyStatus::NotFound);
    }

    // -----------------------------------------------------------------------
    // Test 7: Chain traversal — walk Jan→Ada→Max via KNOWS edges
    // -----------------------------------------------------------------------
    #[test]
    fn test_7_chain_traversal_forward() {
        let store = social_graph();
        let jan = label_fp("Jan");

        let path = store.walk_chain_forward(&jan, 200, 5);
        // Should find at least 1 hop (Jan→Ada)
        assert!(
            !path.is_empty(),
            "Chain traversal should find at least one hop from Jan"
        );

        // Cumulative distances should be non-decreasing
        for window in path.windows(2) {
            assert!(
                window[1].cumulative_distance >= window[0].cumulative_distance,
                "Cumulative distance should be non-decreasing"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 8: HammingMin semiring properties — associativity, identity, absorption
    // -----------------------------------------------------------------------
    #[test]
    fn test_8_hamming_min_semiring_properties() {
        // Identity: extend(one, x) = x
        assert_eq!(HammingMin::extend(HammingMin::one(), 42), 42);

        // Identity: combine(zero, x) = x
        assert_eq!(HammingMin::combine(HammingMin::zero(), 42), 42);

        // Absorption: combine(one, x) = one (min of 0 and any positive = 0)
        assert_eq!(HammingMin::combine(HammingMin::one(), 42), 0);

        // Associativity of extend
        let a = 10u32;
        let b = 20u32;
        let c = 30u32;
        assert_eq!(
            HammingMin::extend(HammingMin::extend(a, b), c),
            HammingMin::extend(a, HammingMin::extend(b, c))
        );

        // Associativity of combine
        assert_eq!(
            HammingMin::combine(HammingMin::combine(a, b), c),
            HammingMin::combine(a, HammingMin::combine(b, c))
        );

        // Saturation: extend(MAX, 1) = MAX
        assert_eq!(HammingMin::extend(u32::MAX, 1), u32::MAX);
    }

    // -----------------------------------------------------------------------
    // Test 9: Truth value revision — combining evidence increases confidence
    // -----------------------------------------------------------------------
    #[test]
    fn test_9_truth_revision_combines_evidence() {
        let a = TruthValue::new(0.8, 0.5);
        let b = TruthValue::new(0.7, 0.5);

        let revised = a.revision(&b);

        // Confidence should increase with more evidence
        assert!(
            revised.confidence > a.confidence,
            "Revised confidence {} should exceed input {}",
            revised.confidence,
            a.confidence
        );

        // Frequency should be between the two inputs
        assert!(revised.frequency >= 0.7 && revised.frequency <= 0.8);

        // Expectation of revised should be meaningful
        let e = revised.expectation();
        assert!(e > 0.5 && e < 1.0);

        // Strength ordering: certain > revised > unknown
        let certain = TruthValue::certain();
        let unknown = TruthValue::unknown();
        assert!(certain.strength() > revised.strength());
        assert!(revised.strength() > unknown.strength());
    }

    // -----------------------------------------------------------------------
    // Test 10: Fingerprint uniqueness and triangle inequality
    // -----------------------------------------------------------------------
    #[test]
    fn test_10_fingerprint_hamming_triangle_inequality() {
        let jan = label_fp("Jan");
        let ada = label_fp("Ada");
        let max = label_fp("Max");

        // Self-distance is zero
        assert_eq!(hamming_distance(&jan, &jan), 0);

        // Symmetry
        assert_eq!(hamming_distance(&jan, &ada), hamming_distance(&ada, &jan));

        // Triangle inequality: d(Jan, Max) <= d(Jan, Ada) + d(Ada, Max)
        let d_jm = hamming_distance(&jan, &max);
        let d_ja = hamming_distance(&jan, &ada);
        let d_am = hamming_distance(&ada, &max);
        assert!(
            d_jm <= d_ja + d_am,
            "Triangle inequality violated: d(Jan,Max)={} > d(Jan,Ada)={} + d(Ada,Max)={}",
            d_jm,
            d_ja,
            d_am
        );

        // Distance from any label to zero fingerprint is non-zero
        assert!(hamming_distance(&jan, &ZERO_FP) > 0);

        // Different labels produce different fingerprints
        assert_ne!(jan, ada);
        assert_ne!(ada, max);
        assert_ne!(jan, max);

        // MerkleRoot from different fingerprints should differ
        let root_jan = MerkleRoot::from_fingerprint(&jan);
        let root_ada = MerkleRoot::from_fingerprint(&ada);
        assert_ne!(root_jan, root_ada);
    }
}
