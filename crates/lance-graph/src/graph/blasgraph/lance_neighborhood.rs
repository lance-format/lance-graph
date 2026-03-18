// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # Lance Neighborhood Storage
//!
//! Extension tables for persisting neighborhood vectors and scopes in Lance.
//! These are **extension tables only** — upstream `graph_nodes.lance` and
//! `graph_rels.lance` formats are untouched.
//!
//! ## Schema
//!
//! ```text
//! scopes.lance:
//!   scope_id          Int64
//!   node_ids          FixedSizeBinary(80000)   -- [Int64; 10000]
//!   node_count        UInt16
//!   created_at        Timestamp
//!
//! neighborhoods.lance:
//!   node_id           Int64
//!   scope_id          Int64
//!   scent             FixedSizeBinary(10000)   -- [u8; 10000] byte 0
//!   resolution        FixedSizeBinary(10000)   -- [u8; 10000] byte 1
//!   edge_count        UInt16
//!   updated_at        Timestamp
//!   NOTE: Lance column pruning means reading only scent never loads resolution.
//!
//! cognitive_nodes.lance:
//!   node_id           Int64
//!   zeckf16_self      UInt8
//!   integrated_16k    FixedSizeBinary(2048)    -- 16Kbit cascade L1
//!   subject_plane     FixedSizeBinary(2048)
//!   predicate_plane   FixedSizeBinary(2048)
//!   object_plane      FixedSizeBinary(2048)
//!   truth_freq        Float32
//!   truth_conf        Float32
//!   merkle_root       FixedSizeBinary(6)       -- Blake3 integrity only
//! ```

use arrow_schema::{DataType, Field, Schema, TimeUnit};
use std::sync::Arc;

use super::neighborhood::{NeighborhoodVector, Scope, MAX_SCOPE_SIZE};

/// Arrow schema for the `scopes` table.
pub fn scopes_schema() -> Schema {
    Schema::new(vec![
        Field::new("scope_id", DataType::Int64, false),
        Field::new(
            "node_ids",
            DataType::FixedSizeBinary((MAX_SCOPE_SIZE * 8) as i32),
            false,
        ),
        Field::new("node_count", DataType::UInt16, false),
        Field::new(
            "created_at",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
    ])
}

/// Arrow schema for the `neighborhoods` table.
pub fn neighborhoods_schema() -> Schema {
    Schema::new(vec![
        Field::new("node_id", DataType::Int64, false),
        Field::new("scope_id", DataType::Int64, false),
        Field::new(
            "scent",
            DataType::FixedSizeBinary(MAX_SCOPE_SIZE as i32),
            false,
        ),
        Field::new(
            "resolution",
            DataType::FixedSizeBinary(MAX_SCOPE_SIZE as i32),
            false,
        ),
        Field::new("edge_count", DataType::UInt16, false),
        Field::new(
            "updated_at",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
    ])
}

/// Arrow schema for the `cognitive_nodes` cold verification table.
pub fn cognitive_nodes_schema() -> Schema {
    Schema::new(vec![
        Field::new("node_id", DataType::Int64, false),
        Field::new("zeckf16_self", DataType::UInt8, false),
        Field::new(
            "integrated_16k",
            DataType::FixedSizeBinary(2048),
            false,
        ),
        Field::new(
            "subject_plane",
            DataType::FixedSizeBinary(2048),
            false,
        ),
        Field::new(
            "predicate_plane",
            DataType::FixedSizeBinary(2048),
            false,
        ),
        Field::new(
            "object_plane",
            DataType::FixedSizeBinary(2048),
            false,
        ),
        Field::new("truth_freq", DataType::Float32, true),
        Field::new("truth_conf", DataType::Float32, true),
        Field::new("merkle_root", DataType::FixedSizeBinary(6), true),
    ])
}

/// Serialize a scope's node_ids to a fixed-size byte buffer for Arrow storage.
///
/// Produces an 80,000-byte buffer (10,000 × 8 bytes) with little-endian u64s.
/// Unused positions are zeroed.
pub fn serialize_scope_node_ids(scope: &Scope) -> Vec<u8> {
    let mut buf = vec![0u8; MAX_SCOPE_SIZE * 8];
    for (i, &id) in scope.node_ids.iter().enumerate() {
        let offset = i * 8;
        buf[offset..offset + 8].copy_from_slice(&id.to_le_bytes());
    }
    buf
}

/// Deserialize scope node_ids from a fixed-size byte buffer.
///
/// Returns the first `node_count` IDs from the buffer.
pub fn deserialize_scope_node_ids(buf: &[u8], node_count: u16) -> Vec<u64> {
    let n = node_count as usize;
    assert!(n <= MAX_SCOPE_SIZE);
    assert!(buf.len() >= n * 8);

    (0..n)
        .map(|i| {
            let offset = i * 8;
            u64::from_le_bytes(buf[offset..offset + 8].try_into().unwrap())
        })
        .collect()
}

/// Serialize a neighborhood's scent column to a fixed-size buffer for Arrow.
///
/// Produces a 10,000-byte buffer. Unused positions are zeroed.
pub fn serialize_scent(nv: &NeighborhoodVector) -> Vec<u8> {
    let scents = nv.scent_column();
    let mut buf = vec![0u8; MAX_SCOPE_SIZE];
    let n = scents.len().min(MAX_SCOPE_SIZE);
    buf[..n].copy_from_slice(&scents[..n]);
    buf
}

/// Serialize a neighborhood's resolution column to a fixed-size buffer.
pub fn serialize_resolution(nv: &NeighborhoodVector) -> Vec<u8> {
    let resolutions = nv.resolution_column();
    let mut buf = vec![0u8; MAX_SCOPE_SIZE];
    let n = resolutions.len().min(MAX_SCOPE_SIZE);
    buf[..n].copy_from_slice(&resolutions[..n]);
    buf
}

/// Deserialize scent bytes from a fixed-size buffer.
pub fn deserialize_scent(buf: &[u8], edge_count: u16) -> Vec<u8> {
    let n = (edge_count as usize).min(buf.len());
    buf[..n].to_vec()
}

/// Build an Arrow `RecordBatch` for a batch of neighborhoods.
///
/// Uses the `neighborhoods_schema()`. Each row is one node's neighborhood.
///
/// Returns the schema and column arrays suitable for `RecordBatch::try_new()`.
pub fn build_neighborhood_arrays(
    neighborhoods: &[NeighborhoodVector],
) -> (Arc<Schema>, Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let schema = Arc::new(neighborhoods_schema());
    let scent_bufs: Vec<Vec<u8>> = neighborhoods.iter().map(|nv| serialize_scent(nv)).collect();
    let resolution_bufs: Vec<Vec<u8>> = neighborhoods
        .iter()
        .map(|nv| serialize_resolution(nv))
        .collect();
    (schema, scent_bufs, resolution_bufs)
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
    fn test_schemas_are_valid() {
        let s1 = scopes_schema();
        assert_eq!(s1.fields().len(), 4);

        let s2 = neighborhoods_schema();
        assert_eq!(s2.fields().len(), 6);

        let s3 = cognitive_nodes_schema();
        assert_eq!(s3.fields().len(), 9);
    }

    #[test]
    fn test_scope_serialization_roundtrip() {
        let ids: Vec<u64> = (100..150).collect();
        let scope = Scope::new(1, ids.clone());

        let buf = serialize_scope_node_ids(&scope);
        assert_eq!(buf.len(), MAX_SCOPE_SIZE * 8);

        let recovered = deserialize_scope_node_ids(&buf, scope.len() as u16);
        assert_eq!(recovered, ids);
    }

    #[test]
    fn test_neighborhood_serialization() {
        let planes = vec![
            random_triple(0, 1, 2),
            random_triple(3, 4, 5),
            random_triple(6, 7, 8),
        ];
        let ids = vec![10u64, 20, 30];
        let (_, neighborhoods) = build_scope(1, &ids, &planes);

        let scent_buf = serialize_scent(&neighborhoods[0]);
        assert_eq!(scent_buf.len(), MAX_SCOPE_SIZE);

        let resolution_buf = serialize_resolution(&neighborhoods[0]);
        assert_eq!(resolution_buf.len(), MAX_SCOPE_SIZE);

        // First 3 bytes should have data, rest zeroed
        let recovered_scent = deserialize_scent(&scent_buf, 3);
        assert_eq!(recovered_scent.len(), 3);
    }

    #[test]
    fn test_build_neighborhood_arrays() {
        let planes = vec![random_triple(0, 1, 2), random_triple(3, 4, 5)];
        let ids = vec![10u64, 20];
        let (_, neighborhoods) = build_scope(1, &ids, &planes);

        let (schema, scent_bufs, resolution_bufs) = build_neighborhood_arrays(&neighborhoods);
        assert_eq!(schema.fields().len(), 6);
        assert_eq!(scent_bufs.len(), 2);
        assert_eq!(resolution_bufs.len(), 2);
    }
}
