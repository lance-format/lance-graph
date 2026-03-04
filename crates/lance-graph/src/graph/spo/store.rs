// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! In-memory SPO triple store with bitmap-based ANN queries.
//!
//! `SpoStore` provides an in-memory implementation of the SPO triple store.
//! Records are keyed by u64 address (from `dn_hash`) and queried via
//! Hamming distance on packed fingerprint bitmaps.
//!
//! This is the development/testing implementation. Production will use
//! Lance ANN indices for the vector search.

use std::collections::HashMap;

use crate::graph::fingerprint::{hamming_distance, Fingerprint};
use crate::graph::sparse::{bitmap_hamming, Bitmap};

use super::builder::{SpoBuilder, SpoRecord};
use super::semiring::{HammingMin, SpoSemiring, TraversalHop};
use super::truth::TruthGate;

/// A query hit from the SPO store.
#[derive(Debug, Clone)]
pub struct SpoHit {
    /// The key (dn_hash) of the matched record.
    pub key: u64,
    /// Hamming distance from the query vector to the packed record.
    pub distance: u32,
    /// The matched record.
    pub record: SpoRecord,
}

/// In-memory SPO triple store.
///
/// Stores SPO records indexed by u64 keys and supports bitmap-based
/// nearest-neighbor queries for the 2³ projection verbs.
pub struct SpoStore {
    records: HashMap<u64, SpoRecord>,
}

impl SpoStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    /// Insert a record at the given key.
    pub fn insert(&mut self, key: u64, record: &SpoRecord) {
        self.records.insert(key, record.clone());
    }

    /// Number of records in the store.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    // =========================================================================
    // Core query methods (brute-force scan for dev/test)
    // =========================================================================

    /// Raw bitmap query: find records closest to the query vector.
    ///
    /// Returns up to `radius` hits, sorted by ascending Hamming distance.
    fn query_bitmap(&self, query: &Bitmap, radius: u32) -> Vec<SpoHit> {
        let mut hits: Vec<SpoHit> = self
            .records
            .iter()
            .map(|(&key, record)| SpoHit {
                key,
                distance: bitmap_hamming(query, &record.packed),
                record: record.clone(),
            })
            .filter(|hit| hit.distance <= radius)
            .collect();

        hits.sort_by_key(|h| h.distance);
        hits
    }

    /// Raw bitmap query with truth gate filtering.
    fn query_bitmap_gated(
        &self,
        query: &Bitmap,
        radius: u32,
        gate: TruthGate,
    ) -> Vec<SpoHit> {
        let mut hits: Vec<SpoHit> = self
            .records
            .iter()
            .map(|(&key, record)| SpoHit {
                key,
                distance: bitmap_hamming(query, &record.packed),
                record: record.clone(),
            })
            .filter(|hit| hit.distance <= radius && gate.passes(&hit.record.truth))
            .collect();

        hits.sort_by_key(|h| h.distance);
        hits
    }

    // =========================================================================
    // 2³ Projection Verbs
    // =========================================================================

    /// SxP2O: Forward query — given Subject and Predicate, find Object.
    ///
    /// `MATCH (s)-[:P]->(?) WHERE s = Subject`
    pub fn query_forward(
        &self,
        subject: &Fingerprint,
        predicate: &Fingerprint,
        radius: u32,
    ) -> Vec<SpoHit> {
        let query = SpoBuilder::build_forward_query(subject, predicate);
        let hits = self.query_bitmap(&query, radius);

        // Post-filter: subject and predicate must closely match
        hits.into_iter()
            .filter(|h| {
                hamming_distance(subject, &h.record.subject) < radius / 2 + 1
                    && hamming_distance(predicate, &h.record.predicate) < radius / 2 + 1
            })
            .collect()
    }

    /// SxP2O with truth gate.
    pub fn query_forward_gated(
        &self,
        subject: &Fingerprint,
        predicate: &Fingerprint,
        radius: u32,
        gate: TruthGate,
    ) -> Vec<SpoHit> {
        let query = SpoBuilder::build_forward_query(subject, predicate);
        let hits = self.query_bitmap_gated(&query, radius, gate);

        hits.into_iter()
            .filter(|h| {
                hamming_distance(subject, &h.record.subject) < radius / 2 + 1
                    && hamming_distance(predicate, &h.record.predicate) < radius / 2 + 1
            })
            .collect()
    }

    /// PxO2S: Reverse query — given Predicate and Object, find Subject.
    ///
    /// `MATCH (?)-[:P]->(o) WHERE o = Object`
    pub fn query_reverse(
        &self,
        predicate: &Fingerprint,
        object: &Fingerprint,
        radius: u32,
    ) -> Vec<SpoHit> {
        let query = SpoBuilder::build_reverse_query(predicate, object);
        let hits = self.query_bitmap(&query, radius);

        hits.into_iter()
            .filter(|h| {
                hamming_distance(predicate, &h.record.predicate) < radius / 2 + 1
                    && hamming_distance(object, &h.record.object) < radius / 2 + 1
            })
            .collect()
    }

    /// SxO2P: Relation query — given Subject and Object, find Predicate.
    ///
    /// `MATCH (s)-[?]->(o)` — what verb connects s to o?
    pub fn query_relation(
        &self,
        subject: &Fingerprint,
        object: &Fingerprint,
        radius: u32,
    ) -> Vec<SpoHit> {
        let query = SpoBuilder::build_relation_query(subject, object);
        let hits = self.query_bitmap(&query, radius);

        hits.into_iter()
            .filter(|h| {
                hamming_distance(subject, &h.record.subject) < radius / 2 + 1
                    && hamming_distance(object, &h.record.object) < radius / 2 + 1
            })
            .collect()
    }

    // =========================================================================
    // Chain traversal (semiring-based)
    // =========================================================================

    /// Walk a chain of forward hops from a starting subject.
    ///
    /// Uses `HammingMin` semiring: costs accumulate, best path wins.
    /// Follows edges greedily by picking the closest match at each hop.
    pub fn walk_chain_forward(
        &self,
        start_subject: &Fingerprint,
        radius: u32,
        max_hops: usize,
    ) -> Vec<TraversalHop> {
        let mut path = Vec::new();
        let mut current_subject = *start_subject;
        let mut cumulative = HammingMin::one();
        let mut visited = std::collections::HashSet::new();

        for _ in 0..max_hops {
            // Find all edges from current_subject (any predicate)
            let mut best_hit: Option<SpoHit> = None;
            for record in self.records.values() {
                let d = hamming_distance(&current_subject, &record.subject);
                if d < radius / 2 + 1 && !visited.contains(&self.key_for_object(&record.object)) {
                    match &best_hit {
                        Some(existing) if d >= existing.distance => {}
                        _ => {
                            best_hit = Some(SpoHit {
                                key: self.key_for_object(&record.object),
                                distance: d,
                                record: record.clone(),
                            });
                        }
                    }
                }
            }

            match best_hit {
                Some(hit) => {
                    cumulative = HammingMin::extend(cumulative, hit.distance);
                    visited.insert(hit.key);
                    path.push(TraversalHop {
                        target_key: hit.key,
                        distance: hit.distance,
                        truth: hit.record.truth,
                        cumulative_distance: cumulative,
                    });
                    current_subject = hit.record.object;
                }
                None => break,
            }
        }

        path
    }

    /// Find the key for a given object fingerprint (reverse lookup).
    fn key_for_object(&self, object: &Fingerprint) -> u64 {
        // Hash the object fingerprint to get a stable key
        let mut h: u64 = 0xcbf29ce484222325;
        for &w in object.iter() {
            h ^= w;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
}

impl Default for SpoStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::fingerprint::{dn_hash, label_fp};
    use crate::graph::spo::TruthValue;

    #[test]
    fn test_store_insert_and_len() {
        let mut store = SpoStore::new();
        assert!(store.is_empty());

        let s = label_fp("Jan");
        let p = label_fp("KNOWS");
        let o = label_fp("Ada");
        let record = SpoBuilder::build_edge(&s, &p, &o, TruthValue::new(0.9, 0.8));
        store.insert(dn_hash("edge:jan-knows-ada"), &record);

        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_forward_query() {
        let mut store = SpoStore::new();
        let jan = label_fp("Jan");
        let knows = label_fp("KNOWS");
        let ada = label_fp("Ada");
        let record = SpoBuilder::build_edge(&jan, &knows, &ada, TruthValue::new(0.9, 0.8));
        store.insert(dn_hash("edge:jan-knows-ada"), &record);

        let hits = store.query_forward(&jan, &knows, 200);
        assert!(!hits.is_empty(), "Forward query should find the edge");
    }

    #[test]
    fn test_reverse_query() {
        let mut store = SpoStore::new();
        let jan = label_fp("Jan");
        let knows = label_fp("KNOWS");
        let ada = label_fp("Ada");
        let record = SpoBuilder::build_edge(&jan, &knows, &ada, TruthValue::new(0.9, 0.8));
        store.insert(dn_hash("edge:jan-knows-ada"), &record);

        let hits = store.query_reverse(&knows, &ada, 200);
        assert!(!hits.is_empty(), "Reverse query should find the edge");
    }
}
