// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Ground truth integration tests for the SPO triple store stack.
//!
//! These tests prove the stack works end-to-end, not just that individual
//! functions compile. They cover:
//!
//! 1. SPO hydration round-trip (insert + forward/reverse query)
//! 2. 2³ projection verbs consistency (SxP2O, SxO2P, PxO2S)
//! 3. TruthGate filtering (OPEN, STRONG, CERTAIN)
//! 4. Belichtung prefilter rejection rate
//! 5. Semiring chain traversal
//! 6. ClamPath + MerkleRoot integrity (documents verify_lineage gap)
//! 7. Cross-convergence: Cypher vs projection verb

use lance_graph::graph::fingerprint::{dn_hash, label_fp, FINGERPRINT_WORDS};
use lance_graph::graph::spo::{
    BindSpace, MerkleRoot, SpoBuilder, SpoStore, TruthGate, TruthValue, VerifyStatus,
};

// =========================================================================
// Test 1: SPO Hydration Round-Trip
// =========================================================================

#[test]
fn test_spo_hydration_round_trip() {
    // 1. Build an edge: Jan KNOWS Ada
    let jan = label_fp("Jan");
    let knows = label_fp("KNOWS");
    let ada = label_fp("Ada");
    let record = SpoBuilder::build_edge(&jan, &knows, &ada, TruthValue::new(0.9, 0.8));

    // 2. Insert into SpoStore
    let mut store = SpoStore::new();
    store.insert(dn_hash("edge:jan-knows-ada"), &record);

    // 3. Forward query: Jan KNOWS ? → should find Ada
    let hits = store.query_forward(&jan, &knows, 100);
    assert!(!hits.is_empty(), "Forward query should find Ada");
    assert!(
        hits[0].distance < 50,
        "Best hit should be close, got distance={}",
        hits[0].distance
    );

    // 4. Reverse query: ? KNOWS Ada → should find Jan
    let hits = store.query_reverse(&knows, &ada, 100);
    assert!(!hits.is_empty(), "Reverse query should find Jan");
}

// =========================================================================
// Test 2: 2³ Projection Verbs
// =========================================================================

#[test]
fn test_projection_verbs_consistency() {
    // Build: Jan CREATES Ada, Jan KNOWS Bob, Ada HELPS Bob
    let jan_fp = label_fp("Jan");
    let ada_fp = label_fp("Ada");
    let bob_fp = label_fp("Bob");
    let creates_fp = label_fp("CREATES");
    let knows_fp = label_fp("KNOWS");
    let helps_fp = label_fp("HELPS");

    let mut store = SpoStore::new();

    let r1 = SpoBuilder::build_edge(&jan_fp, &creates_fp, &ada_fp, TruthValue::new(0.9, 0.9));
    store.insert(dn_hash("edge:jan-creates-ada"), &r1);

    let r2 = SpoBuilder::build_edge(&jan_fp, &knows_fp, &bob_fp, TruthValue::new(0.8, 0.7));
    store.insert(dn_hash("edge:jan-knows-bob"), &r2);

    let r3 = SpoBuilder::build_edge(&ada_fp, &helps_fp, &bob_fp, TruthValue::new(0.7, 0.6));
    store.insert(dn_hash("edge:ada-helps-bob"), &r3);

    // SxP2O: Jan CREATES ? → Ada
    let sxp2o = store.query_forward(&jan_fp, &creates_fp, 100);
    assert!(!sxp2o.is_empty(), "SxP2O: Jan CREATES ? should find Ada");

    // SxO2P: Jan ? Ada → CREATES
    // (bind S and O, find P — what verb connects Jan to Ada?)
    let sxo2p = store.query_relation(&jan_fp, &ada_fp, 100);
    assert!(
        !sxo2p.is_empty(),
        "SxO2P: Jan ? Ada should find CREATES"
    );

    // PxO2S: CREATES ? Ada → Jan
    // (bind P and O, find S — who CREATES Ada?)
    let pxo2s = store.query_reverse(&creates_fp, &ada_fp, 100);
    assert!(
        !pxo2s.is_empty(),
        "PxO2S: CREATES ? Ada should find Jan"
    );

    // All three should agree on the Jan-CREATES-Ada triple
    // Verify the forward query found the right record
    assert_eq!(sxp2o[0].record.subject, jan_fp);
    assert_eq!(sxp2o[0].record.predicate, creates_fp);
    assert_eq!(sxp2o[0].record.object, ada_fp);
}

// =========================================================================
// Test 3: TruthGate Filtering
// =========================================================================

#[test]
fn test_truth_gate_filters_low_confidence() {
    let mut store = SpoStore::new();

    let a = label_fp("entity_A");
    let verb = label_fp("RELATES");
    let b = label_fp("entity_B");
    let c = label_fp("entity_C");

    // Insert high-confidence edge
    let record_high = SpoBuilder::build_edge(&a, &verb, &b, TruthValue::new(0.9, 0.8));
    store.insert(1, &record_high);

    // Insert low-confidence edge
    let record_low = SpoBuilder::build_edge(&a, &verb, &c, TruthValue::new(0.3, 0.2));
    store.insert(2, &record_low);

    // OPEN gate: both found
    let open = store.query_forward_gated(&a, &verb, 200, TruthGate::OPEN);
    assert_eq!(
        open.len(),
        2,
        "OPEN gate should find both edges, found {}",
        open.len()
    );

    // STRONG gate: only high-confidence found
    // TruthValue(0.9, 0.8).expectation() = 0.82 > 0.75 ✓
    // TruthValue(0.3, 0.2).expectation() = 0.46 < 0.75 ✗
    let strong = store.query_forward_gated(&a, &verb, 200, TruthGate::STRONG);
    assert_eq!(
        strong.len(),
        1,
        "STRONG gate should find only high-confidence edge, found {}",
        strong.len()
    );

    // CERTAIN gate: only very high confidence
    // TruthValue(0.9, 0.8).expectation() = 0.82 < 0.9 — also filtered!
    let certain = store.query_forward_gated(&a, &verb, 200, TruthGate::CERTAIN);
    assert_eq!(
        certain.len(),
        0,
        "CERTAIN gate (0.9 threshold) should filter expectation=0.82, found {}",
        certain.len()
    );
}

// =========================================================================
// Test 4: Belichtung Prefilter Rejection Rate
// =========================================================================

#[test]
fn test_belichtung_rejection_rate() {
    let mut store = SpoStore::new();

    // Insert 100 random edges
    for i in 0..100 {
        let s = label_fp(&format!("entity_{}", i));
        let p = label_fp("RELATES");
        let o = label_fp(&format!("target_{}", i));
        let record = SpoBuilder::build_edge(&s, &p, &o, TruthValue::new(0.5, 0.5));
        store.insert(i as u64, &record);
    }

    // Query with tight radius — belichtung should reject most
    let query_s = label_fp("entity_42");
    let query_p = label_fp("RELATES");
    let hits = store.query_forward(&query_s, &query_p, 30);

    // Should find entity_42's edge, maybe 1-2 others. Not 50+.
    assert!(
        hits.len() < 10,
        "Belichtung should reject most non-matches, got {}",
        hits.len()
    );

    // The exact match (entity_42 → target_42) should be present
    // (its S and P fingerprints are exact matches)
    let has_exact = hits
        .iter()
        .any(|h| h.record.subject == query_s && h.record.predicate == query_p);
    assert!(
        has_exact,
        "Exact match for entity_42 should be in the results"
    );
}

// =========================================================================
// Test 5: Semiring Traversal
// =========================================================================

#[test]
fn test_semiring_walk_chain() {
    let mut store = SpoStore::new();

    // Build chain: A→B→C→D
    let a = label_fp("node_A");
    let b = label_fp("node_B");
    let c = label_fp("node_C");
    let d = label_fp("node_D");
    let next = label_fp("NEXT");

    let r1 = SpoBuilder::build_edge(&a, &next, &b, TruthValue::new(0.9, 0.9));
    let r2 = SpoBuilder::build_edge(&b, &next, &c, TruthValue::new(0.8, 0.8));
    let r3 = SpoBuilder::build_edge(&c, &next, &d, TruthValue::new(0.7, 0.7));

    store.insert(dn_hash("a-next-b"), &r1);
    store.insert(dn_hash("b-next-c"), &r2);
    store.insert(dn_hash("c-next-d"), &r3);

    // Walk with HammingMin semiring (shortest semantic path)
    let path = store.walk_chain_forward(&a, 100, 3);

    assert_eq!(
        path.len(),
        3,
        "Should find 3 hops in A→B→C→D chain, found {}",
        path.len()
    );

    // Each hop should have increasing cumulative distance
    // (distances accumulate through chain)
    for i in 1..path.len() {
        assert!(
            path[i].cumulative_distance >= path[i - 1].cumulative_distance,
            "Cumulative distance should increase: hop {} ({}) < hop {} ({})",
            i - 1,
            path[i - 1].cumulative_distance,
            i,
            path[i].cumulative_distance
        );
    }
}

// =========================================================================
// Test 6: ClamPath + MerkleRoot Integrity
// =========================================================================

/// This test DOCUMENTS the verify_lineage no-op gap.
///
/// After corrupting the fingerprint, `verify_lineage` still returns
/// `Consistent` because it only checks structural presence (non-zero root),
/// not content integrity. `verify_integrity` correctly detects the corruption.
///
/// This is a **known gap**, not a bug to fix silently.
#[test]
fn test_clam_merkle_integrity() {
    let mut space = BindSpace::new();
    let fp = [0xDEAD_u64; FINGERPRINT_WORDS];

    let addr = space.write_dn_path("agent:test:node", fp, 3);

    // Read back ClamPath + MerkleRoot
    let (path, root) = space.clam_merkle(addr).unwrap();
    assert!(!root.is_zero(), "MerkleRoot should be stamped");
    assert_eq!(path.depth, 3);

    // Verify the root matches the original fingerprint
    let expected_root = MerkleRoot::from_fingerprint(&fp);
    assert_eq!(*root, expected_root, "Root should match original fingerprint");

    // Corrupt the fingerprint
    space.read_mut(addr).unwrap().fingerprint[5] ^= 0xFFFF; // flip some bits

    // verify_lineage still says Consistent — THIS IS THE KNOWN GAP
    let status = space.verify_lineage(addr);
    assert_eq!(
        status,
        VerifyStatus::Consistent,
        "verify_lineage has known gap: does not re-hash content. \
         It should detect corruption but currently doesn't."
    );

    // verify_integrity correctly detects the corruption
    let status = space.verify_integrity(addr);
    assert_eq!(
        status,
        VerifyStatus::Corrupted,
        "verify_integrity should detect bit-flip corruption"
    );

    // The merkle root stored in the node is still the ORIGINAL root
    // (stamped at write time, not updated on corruption)
    let (_, root_after) = space.clam_merkle(addr).unwrap();
    assert_eq!(
        *root_after, expected_root,
        "Root should still be the original (stamped at write time)"
    );
}

// =========================================================================
// Test 7: Cross-Convergence (Cypher vs Projection Verb)
// =========================================================================

/// Convergence proof: SPO projection path and Cypher path must return
/// the same results for equivalent queries.
///
/// Currently only validates the SPO side. Full convergence requires
/// DataFusion wiring, which is future work. The SPO side must work first.
#[test]
fn test_cypher_vs_projection_convergence() {
    let mut store = SpoStore::new();

    // Insert: Jan CREATES Ada
    let jan_fp = label_fp("Jan");
    let creates_fp = label_fp("CREATES");
    let ada_fp = label_fp("Ada");

    let record = SpoBuilder::build_edge(&jan_fp, &creates_fp, &ada_fp, TruthValue::new(0.9, 0.8));
    store.insert(dn_hash("edge:jan-creates-ada"), &record);

    // SPO path: SxP2O(Jan, CREATES) → Ada
    let spo_hits = store.query_forward(&jan_fp, &creates_fp, 100);

    // Cypher path equivalent: MATCH (a)-[:CREATES]->(b) WHERE a = Jan
    // (This uses cypher_to_sql → DataFusion → same store)
    // For now: just verify the SPO path returns something.
    // Full convergence test needs DataFusion wired, which is future work.
    // But the SPO side must work first.
    assert!(
        !spo_hits.is_empty(),
        "SPO path must find the edge for convergence"
    );

    // Verify the result is the correct triple
    let hit = &spo_hits[0];
    assert_eq!(hit.record.subject, jan_fp, "Subject should be Jan");
    assert_eq!(
        hit.record.predicate, creates_fp,
        "Predicate should be CREATES"
    );
    assert_eq!(hit.record.object, ada_fp, "Object should be Ada");

    // Verify truth value was preserved through the round-trip
    assert!(
        (hit.record.truth.frequency - 0.9).abs() < 0.001,
        "Frequency should be preserved"
    );
    assert!(
        (hit.record.truth.confidence - 0.8).abs() < 0.001,
        "Confidence should be preserved"
    );

    // Future: When DataFusion SPO UDF is wired, add:
    // let cypher_result = CypherQuery::new("MATCH (a:Entity)-[:CREATES]->(b) RETURN b")
    //     .execute_with_spo_store(&store)
    //     .await;
    // assert_eq!(cypher_result, spo_hits);
}
