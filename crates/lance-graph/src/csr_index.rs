// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! CSR (Compressed Sparse Row) adjacency index for native graph traversal
//!
//! Instead of translating graph traversals into SQL joins, this module provides
//! a CSR-based adjacency index that enables O(1) neighbor lookup. Inspired by
//! [GraphAr](https://graphar.apache.org)'s approach of encoding CSR offset
//! tables alongside columnar edge data.
//!
//! # Layout
//!
//! ```text
//! Offset Array:  [0, 3, 5, 5, 9, ...]   (one entry per vertex + 1 sentinel)
//! Neighbor Array: [2, 5, 7, 1, 4, 0, 3, 6, 8, ...]  (destination vertex IDs)
//! ```
//!
//! For vertex `v`, its neighbors are `neighbors[offsets[v]..offsets[v+1]]`.

use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{GraphError, Result};

/// In-memory CSR adjacency index for fast neighbor lookup.
///
/// Stores graph topology in two arrays:
/// - `offsets[v]` = start position of vertex v's neighbors in the neighbor array
/// - `neighbors[offsets[v]..offsets[v+1]]` = destination vertex IDs
#[derive(Debug, Clone)]
pub struct CsrIndex {
    offsets: Vec<u64>,
    neighbors: Vec<u64>,
    num_vertices: u64,
}

impl CsrIndex {
    /// Look up all neighbors of a vertex. Returns an empty slice for vertices
    /// with no outgoing edges or vertex IDs beyond the index range.
    pub fn neighbors(&self, vertex_id: u64) -> &[u64] {
        let v = vertex_id as usize;
        if v >= self.offsets.len() - 1 {
            return &[];
        }
        let start = self.offsets[v] as usize;
        let end = self.offsets[v + 1] as usize;
        &self.neighbors[start..end]
    }

    /// Return the out-degree of a vertex (number of outgoing edges).
    pub fn degree(&self, vertex_id: u64) -> u32 {
        self.neighbors(vertex_id).len() as u32
    }

    /// Return the total number of vertices in the index.
    pub fn num_vertices(&self) -> u64 {
        self.num_vertices
    }

    /// Return the total number of edges in the index.
    pub fn num_edges(&self) -> u64 {
        self.neighbors.len() as u64
    }

    /// Export the CSR index as an Arrow RecordBatch (offset table).
    ///
    /// Schema: `vertex_id: u64, offset: u64, degree: u64`
    ///
    /// This can be persisted as a Lance dataset for later loading.
    pub fn to_record_batch(&self) -> Result<RecordBatch> {
        let n = self.num_vertices as usize;
        let vertex_ids: Vec<u64> = (0..n as u64).collect();
        let offsets: Vec<u64> = self.offsets[..n].to_vec();
        let degrees: Vec<u64> = (0..n)
            .map(|i| self.offsets[i + 1] - self.offsets[i])
            .collect();

        let schema = Arc::new(Schema::new(vec![
            Field::new("vertex_id", DataType::UInt64, false),
            Field::new("offset", DataType::UInt64, false),
            Field::new("degree", DataType::UInt64, false),
        ]));

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vertex_ids)),
                Arc::new(UInt64Array::from(offsets)),
                Arc::new(UInt64Array::from(degrees)),
            ],
        )
        .map_err(|e| GraphError::PlanError {
            message: format!("Failed to create CSR offset RecordBatch: {}", e),
            location: snafu::Location::new(file!(), line!(), column!()),
        })
    }

    /// Export the neighbor (destination) array as an Arrow RecordBatch.
    ///
    /// Schema: `dst_id: u64`
    ///
    /// This is the edge list sorted by source vertex, suitable for sequential scans.
    pub fn neighbors_to_record_batch(&self) -> Result<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "dst_id",
            DataType::UInt64,
            false,
        )]));

        RecordBatch::try_new(
            schema,
            vec![Arc::new(UInt64Array::from(self.neighbors.clone()))],
        )
        .map_err(|e| GraphError::PlanError {
            message: format!("Failed to create CSR neighbors RecordBatch: {}", e),
            location: snafu::Location::new(file!(), line!(), column!()),
        })
    }

    /// Perform a k-hop BFS traversal from a starting vertex.
    ///
    /// Returns all vertices reachable within `max_hops` hops, organized by
    /// distance. Each entry in the returned Vec is the set of vertices at
    /// that hop distance (index 0 = starting vertex, index 1 = 1-hop neighbors, etc.).
    pub fn bfs(&self, start: u64, max_hops: u32) -> Vec<Vec<u64>> {
        let mut visited = vec![false; self.num_vertices as usize];
        let mut levels: Vec<Vec<u64>> = Vec::with_capacity(max_hops as usize + 1);

        if (start as usize) >= self.num_vertices as usize {
            return levels;
        }

        visited[start as usize] = true;
        levels.push(vec![start]);

        for _ in 0..max_hops {
            let frontier = levels.last().unwrap();
            let mut next_level = Vec::new();

            for &v in frontier {
                for &neighbor in self.neighbors(v) {
                    let n = neighbor as usize;
                    if n < visited.len() && !visited[n] {
                        visited[n] = true;
                        next_level.push(neighbor);
                    }
                }
            }

            if next_level.is_empty() {
                break;
            }
            levels.push(next_level);
        }

        levels
    }

    /// Find the shortest path between two vertices using BFS.
    ///
    /// Returns `None` if no path exists, or `Some(path)` where path is the
    /// sequence of vertex IDs from `start` to `end` (inclusive).
    pub fn shortest_path(&self, start: u64, end: u64) -> Option<Vec<u64>> {
        if start == end {
            return Some(vec![start]);
        }

        let n = self.num_vertices as usize;
        if start as usize >= n || end as usize >= n {
            return None;
        }

        let mut visited = vec![false; n];
        let mut parent: Vec<Option<u64>> = vec![None; n];
        let mut queue = std::collections::VecDeque::new();

        visited[start as usize] = true;
        queue.push_back(start);

        while let Some(current) = queue.pop_front() {
            for &neighbor in self.neighbors(current) {
                let ni = neighbor as usize;
                if ni < n && !visited[ni] {
                    visited[ni] = true;
                    parent[ni] = Some(current);

                    if neighbor == end {
                        let mut path = vec![end];
                        let mut node = end;
                        while let Some(p) = parent[node as usize] {
                            path.push(p);
                            node = p;
                        }
                        path.reverse();
                        return Some(path);
                    }
                    queue.push_back(neighbor);
                }
            }
        }

        None
    }
}

/// Builder for constructing a CSR index from edge data.
///
/// Accepts edges as (source, destination) pairs and builds the compressed
/// sparse row representation.
#[derive(Debug)]
pub struct CsrIndexBuilder {
    edges: Vec<(u64, u64)>,
    num_vertices: Option<u64>,
}

impl CsrIndexBuilder {
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            num_vertices: None,
        }
    }

    /// Set the total number of vertices explicitly. If not set, it is inferred
    /// from the maximum vertex ID seen in the edges.
    pub fn with_num_vertices(mut self, n: u64) -> Self {
        self.num_vertices = Some(n);
        self
    }

    /// Add a single directed edge from `src` to `dst`.
    pub fn add_edge(mut self, src: u64, dst: u64) -> Self {
        self.edges.push((src, dst));
        self
    }

    /// Add edges from an Arrow RecordBatch with `src_id` and `dst_id` columns.
    pub fn add_edges_from_batch(self, batch: &RecordBatch) -> Result<Self> {
        self.add_edges_from_batch_with_columns(batch, "src_id", "dst_id")
    }

    /// Add edges from an Arrow RecordBatch, reading source vertex ids from
    /// `src_col` and destination vertex ids from `dst_col`.
    ///
    /// Both columns must be `UInt64`. To build a reversed (incoming/CSC) index,
    /// pass the destination column name as `src_col` and vice versa.
    pub fn add_edges_from_batch_with_columns(
        mut self,
        batch: &RecordBatch,
        src_col: &str,
        dst_col: &str,
    ) -> Result<Self> {
        let src_array = batch
            .column_by_name(src_col)
            .ok_or_else(|| GraphError::PlanError {
                message: format!("Edge batch missing '{}' column", src_col),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| GraphError::PlanError {
                message: format!("'{}' column must be UInt64", src_col),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        let dst_array = batch
            .column_by_name(dst_col)
            .ok_or_else(|| GraphError::PlanError {
                message: format!("Edge batch missing '{}' column", dst_col),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| GraphError::PlanError {
                message: format!("'{}' column must be UInt64", dst_col),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        for i in 0..batch.num_rows() {
            self.edges.push((src_array.value(i), dst_array.value(i)));
        }

        Ok(self)
    }

    /// Build the CSR index.
    ///
    /// Sorts edges by source vertex, then builds offset and neighbor arrays.
    pub fn build(mut self) -> CsrIndex {
        let num_vertices = self.num_vertices.unwrap_or_else(|| {
            self.edges
                .iter()
                .flat_map(|&(s, d)| [s, d])
                .max()
                .map(|m| m + 1)
                .unwrap_or(0)
        });

        // Sort by source vertex for CSR construction
        self.edges.sort_unstable_by_key(|&(src, _)| src);

        // Build offset and neighbor arrays
        let mut offsets = vec![0u64; num_vertices as usize + 1];
        let mut neighbors = Vec::with_capacity(self.edges.len());

        // Count degrees
        let mut degree_map: HashMap<u64, u64> = HashMap::new();
        for &(src, _) in &self.edges {
            *degree_map.entry(src).or_insert(0) += 1;
        }

        // Build prefix-sum offsets
        let mut running = 0u64;
        for v in 0..num_vertices {
            offsets[v as usize] = running;
            running += degree_map.get(&v).copied().unwrap_or(0);
        }
        offsets[num_vertices as usize] = running;

        // Fill neighbor array
        for &(_, dst) in &self.edges {
            neighbors.push(dst);
        }

        CsrIndex {
            offsets,
            neighbors,
            num_vertices,
        }
    }
}

impl Default for CsrIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Build both outgoing (CSR) and incoming (CSC) adjacency indices from edge data.
///
/// Returns `(outgoing_index, incoming_index)`.
pub fn build_bidirectional_index(edges: &[(u64, u64)], num_vertices: u64) -> (CsrIndex, CsrIndex) {
    let mut outgoing_builder = CsrIndexBuilder::new().with_num_vertices(num_vertices);
    let mut incoming_builder = CsrIndexBuilder::new().with_num_vertices(num_vertices);

    for &(src, dst) in edges {
        outgoing_builder = outgoing_builder.add_edge(src, dst);
        incoming_builder = incoming_builder.add_edge(dst, src);
    }

    (outgoing_builder.build(), incoming_builder.build())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test graph:
    //   0 -> 1, 2, 3
    //   1 -> 2
    //   2 -> 3
    //   3 -> (none)
    fn sample_index() -> CsrIndex {
        CsrIndexBuilder::new()
            .with_num_vertices(4)
            .add_edge(0, 1)
            .add_edge(0, 2)
            .add_edge(0, 3)
            .add_edge(1, 2)
            .add_edge(2, 3)
            .build()
    }

    #[test]
    fn test_basic_neighbor_lookup() {
        let idx = sample_index();

        assert_eq!(idx.neighbors(0), &[1, 2, 3]);
        assert_eq!(idx.neighbors(1), &[2]);
        assert_eq!(idx.neighbors(2), &[3]);
        assert_eq!(idx.neighbors(3), &[] as &[u64]);
    }

    #[test]
    fn test_degree() {
        let idx = sample_index();
        assert_eq!(idx.degree(0), 3);
        assert_eq!(idx.degree(1), 1);
        assert_eq!(idx.degree(2), 1);
        assert_eq!(idx.degree(3), 0);
    }

    #[test]
    fn test_metadata() {
        let idx = sample_index();
        assert_eq!(idx.num_vertices(), 4);
        assert_eq!(idx.num_edges(), 5);
    }

    #[test]
    fn test_out_of_range_vertex() {
        let idx = sample_index();
        assert_eq!(idx.neighbors(99), &[] as &[u64]);
        assert_eq!(idx.degree(99), 0);
    }

    #[test]
    fn test_empty_graph() {
        let idx = CsrIndexBuilder::new().with_num_vertices(0).build();
        assert_eq!(idx.num_vertices(), 0);
        assert_eq!(idx.num_edges(), 0);
        assert_eq!(idx.neighbors(0), &[] as &[u64]);
    }

    #[test]
    fn test_isolated_vertices() {
        let idx = CsrIndexBuilder::new()
            .with_num_vertices(5)
            .add_edge(1, 3)
            .build();

        assert_eq!(idx.neighbors(0), &[] as &[u64]);
        assert_eq!(idx.neighbors(1), &[3]);
        assert_eq!(idx.neighbors(2), &[] as &[u64]);
        assert_eq!(idx.neighbors(3), &[] as &[u64]);
        assert_eq!(idx.neighbors(4), &[] as &[u64]);
    }

    #[test]
    fn test_build_from_record_batch() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("src_id", DataType::UInt64, false),
            Field::new("dst_id", DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vec![0, 0, 1, 2])),
                Arc::new(UInt64Array::from(vec![1, 2, 2, 0])),
            ],
        )
        .unwrap();

        let idx = CsrIndexBuilder::new()
            .add_edges_from_batch(&batch)
            .unwrap()
            .build();

        assert_eq!(idx.neighbors(0), &[1, 2]);
        assert_eq!(idx.neighbors(1), &[2]);
        assert_eq!(idx.neighbors(2), &[0]);
    }

    #[test]
    fn test_to_record_batch() {
        let idx = sample_index();
        let batch = idx.to_record_batch().unwrap();

        assert_eq!(batch.num_rows(), 4);
        assert_eq!(batch.num_columns(), 3);

        let offsets = batch
            .column_by_name("offset")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(offsets.value(0), 0);
        assert_eq!(offsets.value(1), 3);
        assert_eq!(offsets.value(2), 4);
        assert_eq!(offsets.value(3), 5);

        let degrees = batch
            .column_by_name("degree")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(degrees.value(0), 3);
        assert_eq!(degrees.value(1), 1);
        assert_eq!(degrees.value(2), 1);
        assert_eq!(degrees.value(3), 0);
    }

    #[test]
    fn test_neighbors_to_record_batch() {
        let idx = sample_index();
        let batch = idx.neighbors_to_record_batch().unwrap();

        assert_eq!(batch.num_rows(), 5);
        let dst = batch
            .column_by_name("dst_id")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(dst.value(0), 1);
        assert_eq!(dst.value(1), 2);
        assert_eq!(dst.value(2), 3);
        assert_eq!(dst.value(3), 2);
        assert_eq!(dst.value(4), 3);
    }

    #[test]
    fn test_bidirectional_index() {
        let edges = vec![(0, 1), (0, 2), (1, 2)];
        let (outgoing, incoming) = build_bidirectional_index(&edges, 3);

        // Outgoing
        assert_eq!(outgoing.neighbors(0), &[1, 2]);
        assert_eq!(outgoing.neighbors(1), &[2]);
        assert_eq!(outgoing.neighbors(2), &[] as &[u64]);

        // Incoming (reversed edges)
        assert_eq!(incoming.neighbors(0), &[] as &[u64]);
        assert_eq!(incoming.neighbors(1), &[0]);
        assert_eq!(incoming.neighbors(2), &[0, 1]);
    }

    #[test]
    fn test_bfs_traversal() {
        // Graph: 0->1, 0->2, 1->3, 2->3, 3->4
        let idx = CsrIndexBuilder::new()
            .with_num_vertices(5)
            .add_edge(0, 1)
            .add_edge(0, 2)
            .add_edge(1, 3)
            .add_edge(2, 3)
            .add_edge(3, 4)
            .build();

        let levels = idx.bfs(0, 3);
        assert_eq!(levels.len(), 4);
        assert_eq!(levels[0], vec![0]); // start
        assert_eq!(levels[1], vec![1, 2]); // 1-hop
        assert_eq!(levels[2], vec![3]); // 2-hop (3 reached from both 1 and 2, but visited once)
        assert_eq!(levels[3], vec![4]); // 3-hop
    }

    #[test]
    fn test_bfs_limited_hops() {
        let idx = CsrIndexBuilder::new()
            .with_num_vertices(4)
            .add_edge(0, 1)
            .add_edge(1, 2)
            .add_edge(2, 3)
            .build();

        let levels = idx.bfs(0, 1);
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0], vec![0]);
        assert_eq!(levels[1], vec![1]);
    }

    #[test]
    fn test_bfs_disconnected() {
        let idx = CsrIndexBuilder::new()
            .with_num_vertices(4)
            .add_edge(0, 1)
            // 2, 3 are disconnected
            .build();

        let levels = idx.bfs(0, 10);
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0], vec![0]);
        assert_eq!(levels[1], vec![1]);
    }

    #[test]
    fn test_bfs_invalid_start() {
        let idx = CsrIndexBuilder::new().with_num_vertices(3).build();
        let levels = idx.bfs(99, 5);
        assert!(levels.is_empty());
    }

    #[test]
    fn test_shortest_path_direct() {
        let idx = CsrIndexBuilder::new()
            .with_num_vertices(3)
            .add_edge(0, 1)
            .add_edge(1, 2)
            .add_edge(0, 2)
            .build();

        // Direct edge 0->2 is shorter than 0->1->2
        let path = idx.shortest_path(0, 2).unwrap();
        assert_eq!(path, vec![0, 2]);
    }

    #[test]
    fn test_shortest_path_multi_hop() {
        let idx = CsrIndexBuilder::new()
            .with_num_vertices(4)
            .add_edge(0, 1)
            .add_edge(1, 2)
            .add_edge(2, 3)
            .build();

        let path = idx.shortest_path(0, 3).unwrap();
        assert_eq!(path, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_shortest_path_same_vertex() {
        let idx = sample_index();
        let path = idx.shortest_path(2, 2).unwrap();
        assert_eq!(path, vec![2]);
    }

    #[test]
    fn test_shortest_path_unreachable() {
        let idx = CsrIndexBuilder::new()
            .with_num_vertices(3)
            .add_edge(0, 1)
            // no path from 0 to 2
            .build();

        assert!(idx.shortest_path(0, 2).is_none());
    }

    #[test]
    fn test_shortest_path_invalid_vertices() {
        let idx = sample_index();
        assert!(idx.shortest_path(99, 0).is_none());
        assert!(idx.shortest_path(0, 99).is_none());
    }

    #[test]
    fn test_auto_inferred_num_vertices() {
        let idx = CsrIndexBuilder::new().add_edge(0, 5).add_edge(3, 7).build();

        // Should infer num_vertices = 8 (max ID 7 + 1)
        assert_eq!(idx.num_vertices(), 8);
        assert_eq!(idx.neighbors(0), &[5]);
        assert_eq!(idx.neighbors(3), &[7]);
    }

    #[test]
    fn test_self_loops() {
        let idx = CsrIndexBuilder::new()
            .with_num_vertices(3)
            .add_edge(0, 0)
            .add_edge(1, 1)
            .add_edge(0, 1)
            .build();

        assert_eq!(idx.neighbors(0), &[0, 1]);
        assert_eq!(idx.neighbors(1), &[1]);
    }

    #[test]
    fn test_parallel_edges() {
        let idx = CsrIndexBuilder::new()
            .with_num_vertices(2)
            .add_edge(0, 1)
            .add_edge(0, 1)
            .add_edge(0, 1)
            .build();

        // CSR preserves multi-edges
        assert_eq!(idx.neighbors(0), &[1, 1, 1]);
        assert_eq!(idx.degree(0), 3);
    }

    #[test]
    fn test_build_from_record_batch_custom_columns() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("src_person_id", DataType::UInt64, false),
            Field::new("dst_person_id", DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vec![0, 0, 1])),
                Arc::new(UInt64Array::from(vec![1, 2, 2])),
            ],
        )
        .unwrap();

        // Forward (outgoing): src -> dst
        let idx = CsrIndexBuilder::new()
            .add_edges_from_batch_with_columns(&batch, "src_person_id", "dst_person_id")
            .unwrap()
            .build();
        assert_eq!(idx.neighbors(0), &[1, 2]);
        assert_eq!(idx.neighbors(1), &[2]);

        // Reversed (incoming): swap the column args -> dst -> src
        let rev = CsrIndexBuilder::new()
            .add_edges_from_batch_with_columns(&batch, "dst_person_id", "src_person_id")
            .unwrap()
            .build();
        assert_eq!(rev.neighbors(2), &[0, 1]);
        assert_eq!(rev.neighbors(1), &[0]);
    }
}
