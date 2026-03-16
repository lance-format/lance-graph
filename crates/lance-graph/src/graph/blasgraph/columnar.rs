// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance columnar schema for cognitive types.
//!
//! Nodes, edges, and fingerprints stored as Lance dataset columns.
//! Plane as FixedSizeBinary, Node as 3 columns, BF16 edge weights.

use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

/// Number of bytes per fingerprint plane (16384 bits / 8 = 2048 bytes).
const PLANE_BYTES: i32 = 2048;

/// Number of bytes per Merkle seal (48 bits / 8 = 6 bytes).
const SEAL_BYTES: i32 = 6;

/// Lance column schema for node storage.
///
/// Each node stores three fingerprint planes (subject, predicate, object),
/// their Merkle seals, and an encounter counter:
///
/// - `node_id`: UInt32 -- unique node identifier
/// - `plane_s`: FixedSizeBinary(2048) -- subject fingerprint bits
/// - `plane_p`: FixedSizeBinary(2048) -- predicate fingerprint bits
/// - `plane_o`: FixedSizeBinary(2048) -- object fingerprint bits
/// - `seal_s`: FixedSizeBinary(6) -- subject MerkleRoot 48 bits
/// - `seal_p`: FixedSizeBinary(6) -- predicate MerkleRoot 48 bits
/// - `seal_o`: FixedSizeBinary(6) -- object MerkleRoot 48 bits
/// - `encounters`: UInt32 -- number of times this node has been observed
pub struct NodeSchema;

impl NodeSchema {
    /// Build the Arrow schema for node storage.
    pub fn arrow_schema() -> Schema {
        Schema::new(vec![
            Field::new("node_id", DataType::UInt32, false),
            Field::new("plane_s", DataType::FixedSizeBinary(PLANE_BYTES), false),
            Field::new("plane_p", DataType::FixedSizeBinary(PLANE_BYTES), false),
            Field::new("plane_o", DataType::FixedSizeBinary(PLANE_BYTES), false),
            Field::new("seal_s", DataType::FixedSizeBinary(SEAL_BYTES), false),
            Field::new("seal_p", DataType::FixedSizeBinary(SEAL_BYTES), false),
            Field::new("seal_o", DataType::FixedSizeBinary(SEAL_BYTES), false),
            Field::new("encounters", DataType::UInt32, false),
        ])
    }

    /// Return the schema wrapped in an Arc for use with RecordBatch builders.
    pub fn arrow_schema_ref() -> Arc<Schema> {
        Arc::new(Self::arrow_schema())
    }
}

/// Lance column schema for edge storage.
///
/// Edges connect two nodes with a BF16 weight and a fingerprint label:
///
/// - `src_id`: UInt32 -- source node identifier
/// - `dst_id`: UInt32 -- destination node identifier
/// - `weight`: Float16 -- BF16 edge weight
/// - `label`: FixedSizeBinary(2048) -- edge label fingerprint
pub struct EdgeSchema;

impl EdgeSchema {
    /// Build the Arrow schema for edge storage.
    pub fn arrow_schema() -> Schema {
        Schema::new(vec![
            Field::new("src_id", DataType::UInt32, false),
            Field::new("dst_id", DataType::UInt32, false),
            Field::new("weight", DataType::Float16, false),
            Field::new(
                "label",
                DataType::FixedSizeBinary(PLANE_BYTES),
                true,
            ),
        ])
    }

    /// Return the schema wrapped in an Arc for use with RecordBatch builders.
    pub fn arrow_schema_ref() -> Arc<Schema> {
        Arc::new(Self::arrow_schema())
    }
}

/// Lance column schema for fingerprint-only storage.
///
/// A flat table of fingerprints for bulk Hamming search:
///
/// - `id`: UInt32 -- fingerprint identifier
/// - `fingerprint`: FixedSizeBinary(2048) -- the fingerprint bits
pub struct FingerprintSchema;

impl FingerprintSchema {
    /// Build the Arrow schema for fingerprint storage.
    pub fn arrow_schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            Field::new(
                "fingerprint",
                DataType::FixedSizeBinary(PLANE_BYTES),
                false,
            ),
        ])
    }

    /// Return the schema wrapped in an Arc for use with RecordBatch builders.
    pub fn arrow_schema_ref() -> Arc<Schema> {
        Arc::new(Self::arrow_schema())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_schema_fields() {
        let schema = NodeSchema::arrow_schema();
        assert_eq!(schema.fields().len(), 8);
        assert_eq!(schema.field(0).name(), "node_id");
        assert_eq!(*schema.field(0).data_type(), DataType::UInt32);
        assert_eq!(
            *schema.field(1).data_type(),
            DataType::FixedSizeBinary(2048)
        );
        assert_eq!(schema.field(1).name(), "plane_s");
        assert_eq!(
            *schema.field(4).data_type(),
            DataType::FixedSizeBinary(6)
        );
        assert_eq!(schema.field(4).name(), "seal_s");
        assert_eq!(schema.field(7).name(), "encounters");
    }

    #[test]
    fn test_edge_schema_fields() {
        let schema = EdgeSchema::arrow_schema();
        assert_eq!(schema.fields().len(), 4);
        assert_eq!(schema.field(0).name(), "src_id");
        assert_eq!(schema.field(1).name(), "dst_id");
        assert_eq!(*schema.field(2).data_type(), DataType::Float16);
        assert!(schema.field(3).is_nullable());
    }

    #[test]
    fn test_fingerprint_schema_fields() {
        let schema = FingerprintSchema::arrow_schema();
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(
            *schema.field(1).data_type(),
            DataType::FixedSizeBinary(2048)
        );
    }

    #[test]
    fn test_schema_refs_are_arc() {
        let node_ref = NodeSchema::arrow_schema_ref();
        assert_eq!(Arc::strong_count(&node_ref), 1);

        let edge_ref = EdgeSchema::arrow_schema_ref();
        assert_eq!(Arc::strong_count(&edge_ref), 1);

        let fp_ref = FingerprintSchema::arrow_schema_ref();
        assert_eq!(Arc::strong_count(&fp_ref), 1);
    }
}
