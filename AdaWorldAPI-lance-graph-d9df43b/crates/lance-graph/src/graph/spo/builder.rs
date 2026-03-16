// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Builder for SPO edge records.
//!
//! An SPO record packs Subject, Predicate, Object fingerprints together
//! with a truth value into a structure that can be stored in an SpoStore
//! and queried via ANN search.

use crate::graph::fingerprint::{Fingerprint, FINGERPRINT_WORDS};
use crate::graph::sparse::{pack_axes, Bitmap, BITMAP_WORDS};

use super::truth::TruthValue;

/// An SPO record representing a single edge in the graph.
///
/// Contains the packed search vector (for ANN queries) and the individual
/// components (for result interpretation).
#[derive(Debug, Clone)]
pub struct SpoRecord {
    /// Subject fingerprint.
    pub subject: Fingerprint,
    /// Predicate fingerprint.
    pub predicate: Fingerprint,
    /// Object fingerprint.
    pub object: Fingerprint,
    /// Packed bitmap: S|P|O for ANN similarity search.
    pub packed: Bitmap,
    /// Truth value of this edge.
    pub truth: TruthValue,
}

/// Builder for constructing SPO edge records.
pub struct SpoBuilder;

impl SpoBuilder {
    /// Build an edge record from S, P, O fingerprints and a truth value.
    ///
    /// The packed bitmap is the OR of all three fingerprints, used as
    /// the search vector for ANN queries in Lance.
    pub fn build_edge(
        subject: &Fingerprint,
        predicate: &Fingerprint,
        object: &Fingerprint,
        truth: TruthValue,
    ) -> SpoRecord {
        // Ensure sizes match (compile-time guarantee via type aliases,
        // but assert at runtime for safety during development).
        debug_assert_eq!(FINGERPRINT_WORDS, BITMAP_WORDS);

        let packed = pack_axes(subject, predicate, object);

        SpoRecord {
            subject: *subject,
            predicate: *predicate,
            object: *object,
            packed,
            truth,
        }
    }

    /// Build a forward query vector: S|P (looking for O).
    ///
    /// For SxP2O queries: given Subject and Predicate, find Object.
    pub fn build_forward_query(subject: &Fingerprint, predicate: &Fingerprint) -> Bitmap {
        let zero = [0u64; BITMAP_WORDS];
        pack_axes(subject, predicate, &zero)
    }

    /// Build a reverse query vector: P|O (looking for S).
    ///
    /// For PxO2S queries: given Predicate and Object, find Subject.
    pub fn build_reverse_query(predicate: &Fingerprint, object: &Fingerprint) -> Bitmap {
        let zero = [0u64; BITMAP_WORDS];
        pack_axes(&zero, predicate, object)
    }

    /// Build a relation query vector: S|O (looking for P).
    ///
    /// For SxO2P queries: given Subject and Object, find Predicate.
    pub fn build_relation_query(subject: &Fingerprint, object: &Fingerprint) -> Bitmap {
        let zero = [0u64; BITMAP_WORDS];
        pack_axes(subject, &zero, object)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::fingerprint::label_fp;

    #[test]
    fn test_build_edge() {
        let s = label_fp("Jan");
        let p = label_fp("KNOWS");
        let o = label_fp("Ada");
        let record = SpoBuilder::build_edge(&s, &p, &o, TruthValue::new(0.9, 0.8));

        assert_eq!(record.subject, s);
        assert_eq!(record.predicate, p);
        assert_eq!(record.object, o);
        assert_eq!(record.truth.frequency, 0.9);
        // Packed should be S|P|O
        for i in 0..BITMAP_WORDS {
            assert_eq!(record.packed[i], s[i] | p[i] | o[i]);
        }
    }

    #[test]
    fn test_forward_query_vector() {
        let s = label_fp("Jan");
        let p = label_fp("KNOWS");
        let query = SpoBuilder::build_forward_query(&s, &p);
        // Should contain bits from both S and P
        for i in 0..BITMAP_WORDS {
            assert_eq!(query[i], s[i] | p[i]);
        }
    }
}
