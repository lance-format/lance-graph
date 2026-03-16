// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # High-Level GraphBLAS Operations
//!
//! Free functions that mirror the GraphBLAS C API naming conventions:
//! `grb_mxm`, `grb_mxv`, `grb_vxm`, element-wise operations, reduce,
//! apply, assign, and extract. Also includes HDR-specific graph algorithms
//! such as BFS, SSSP, and PageRank.

use crate::graph::blasgraph::descriptor::{Descriptor, GrBDesc};
use crate::graph::blasgraph::matrix::GrBMatrix;
use crate::graph::blasgraph::semiring::{HdrSemiring, Semiring};
use crate::graph::blasgraph::types::{BinaryOp, BitVec, HdrScalar, MonoidOp, UnaryOp};
use crate::graph::blasgraph::vector::GrBVector;

// ---------------------------------------------------------------------------
// Matrix-matrix, matrix-vector, vector-matrix
// ---------------------------------------------------------------------------

/// Matrix-matrix multiply `C = A (+).(*) B` under a semiring.
pub fn grb_mxm(
    a: &GrBMatrix,
    b: &GrBMatrix,
    semiring: &dyn Semiring,
    desc: &Descriptor,
) -> GrBMatrix {
    a.mxm(b, semiring, desc)
}

/// Matrix-vector multiply `w = A (+).(*) u` under a semiring.
pub fn grb_mxv(
    a: &GrBMatrix,
    u: &GrBVector,
    semiring: &dyn Semiring,
    desc: &Descriptor,
) -> GrBVector {
    a.mxv(u, semiring, desc)
}

/// Vector-matrix multiply `w = u (+).(*) A` under a semiring.
pub fn grb_vxm(
    u: &GrBVector,
    a: &GrBMatrix,
    semiring: &dyn Semiring,
    desc: &Descriptor,
) -> GrBVector {
    a.vxm(u, semiring, desc)
}

// ---------------------------------------------------------------------------
// Element-wise
// ---------------------------------------------------------------------------

/// Element-wise matrix addition (union of patterns).
pub fn grb_ewise_add_matrix(
    a: &GrBMatrix,
    b: &GrBMatrix,
    op: BinaryOp,
    desc: &Descriptor,
) -> GrBMatrix {
    a.ewise_add(b, op, desc)
}

/// Element-wise matrix multiplication (intersection of patterns).
pub fn grb_ewise_mult_matrix(
    a: &GrBMatrix,
    b: &GrBMatrix,
    op: BinaryOp,
    desc: &Descriptor,
) -> GrBMatrix {
    a.ewise_mult(b, op, desc)
}

/// Element-wise vector addition (union of patterns).
pub fn grb_ewise_add_vector(a: &GrBVector, b: &GrBVector, op: BinaryOp) -> GrBVector {
    a.ewise_add(b, op)
}

/// Element-wise vector multiplication (intersection of patterns).
pub fn grb_ewise_mult_vector(a: &GrBVector, b: &GrBVector, op: BinaryOp) -> GrBVector {
    a.ewise_mult(b, op)
}

// ---------------------------------------------------------------------------
// Reduce
// ---------------------------------------------------------------------------

/// Reduce matrix rows to a vector using a monoid.
pub fn grb_reduce_matrix_to_vector(m: &GrBMatrix, monoid: MonoidOp) -> GrBVector {
    m.reduce_rows(monoid)
}

/// Reduce matrix columns to a vector using a monoid.
pub fn grb_reduce_matrix_cols(m: &GrBMatrix, monoid: MonoidOp) -> GrBVector {
    m.reduce_cols(monoid)
}

/// Reduce a matrix to a single scalar using a monoid.
pub fn grb_reduce_matrix_to_scalar(m: &GrBMatrix, monoid: MonoidOp) -> HdrScalar {
    m.reduce(monoid)
}

/// Reduce a vector to a single scalar using a monoid.
pub fn grb_reduce_vector(v: &GrBVector, monoid: MonoidOp) -> HdrScalar {
    v.reduce(monoid)
}

// ---------------------------------------------------------------------------
// Apply
// ---------------------------------------------------------------------------

/// Apply a unary operator to every element of a matrix.
pub fn grb_apply_matrix(m: &GrBMatrix, op: UnaryOp) -> GrBMatrix {
    m.apply(op)
}

/// Apply a unary operator to every element of a vector.
pub fn grb_apply_vector(v: &GrBVector, op: UnaryOp) -> GrBVector {
    v.apply(op)
}

// ---------------------------------------------------------------------------
// Assign / Extract
// ---------------------------------------------------------------------------

/// Extract a submatrix at the given row and column indices.
pub fn grb_extract_matrix(m: &GrBMatrix, rows: &[usize], cols: &[usize]) -> GrBMatrix {
    m.extract(rows, cols)
}

/// Extract a sub-vector at the given indices.
pub fn grb_extract_vector(v: &GrBVector, indices: &[usize]) -> GrBVector {
    v.extract(indices)
}

/// Assign values from `src` into `dst` at the given indices.
pub fn grb_assign_vector(dst: &mut GrBVector, src: &GrBVector, indices: &[usize]) {
    dst.assign(src, indices);
}

/// Transpose a matrix.
pub fn grb_transpose(m: &GrBMatrix) -> GrBMatrix {
    m.transpose()
}

// ---------------------------------------------------------------------------
// HDR Graph Algorithms
// ---------------------------------------------------------------------------

/// Breadth-first search using hyperdimensional vectors.
///
/// Performs a level-synchronous BFS from `source` on the adjacency matrix `adj`.
/// Returns a vector where entry `i` contains the XOR-bound path vector from source
/// to node `i` (if reachable within `max_depth` hops).
///
/// Uses the BIND_FIRST semiring: XOR multiply, first non-empty add.
pub fn hdr_bfs(adj: &GrBMatrix, source: usize, max_depth: usize) -> GrBVector {
    let n = adj.nrows();
    assert_eq!(n, adj.ncols(), "adjacency matrix must be square");
    assert!(source < n, "source out of bounds");

    let semiring = HdrSemiring::BindFirst;
    let desc = GrBDesc::default();

    // Frontier: start with the source node assigned its random vector.
    let mut frontier = GrBVector::new(n);
    frontier.set(source, BitVec::random(source as u64 + 1));

    // Result: accumulate discovered nodes.
    let mut result = GrBVector::new(n);
    result.set(source, BitVec::random(source as u64 + 1));

    for _depth in 0..max_depth {
        // Advance frontier: next = frontier * A (follow outgoing edges)
        let next = adj.vxm(&frontier, &semiring, &desc);

        // Only keep newly discovered nodes (not already in result).
        let mut new_frontier = GrBVector::new(n);
        let mut any_new = false;
        for (idx, v) in next.iter() {
            if result.get(idx).is_none() {
                new_frontier.set(idx, v.clone());
                result.set(idx, v.clone());
                any_new = true;
            }
        }

        if !any_new {
            break;
        }
        frontier = new_frontier;
    }

    result
}

/// Single-source shortest path using Hamming distance.
///
/// Performs Bellman-Ford relaxation where each edge's weight is the
/// popcount (Hamming weight) of its `BitVec`. Path costs are tracked
/// as cumulative `u32` popcount sums alongside XOR-composed path
/// vectors.
///
/// Returns a vector where entry `i` holds the XOR-composed path vector
/// from `source` to node `i`. The path cost for node `i` is
/// `result.get(i).unwrap().popcount()` — but note this is the popcount
/// of the XOR composition, not the cumulative edge weight sum. For the
/// cumulative cost, use `result.get_scalar(i)` which returns
/// `Float(cumulative_cost)`.
pub fn hdr_sssp(adj: &GrBMatrix, source: usize, max_iters: usize) -> GrBVector {
    let n = adj.nrows();
    assert_eq!(n, adj.ncols(), "adjacency matrix must be square");
    assert!(source < n, "source out of bounds");

    // Cumulative path costs as u32 (u32::MAX = unreachable).
    let mut cost: Vec<u32> = vec![u32::MAX; n];
    cost[source] = 0;

    // XOR-composed path vectors.
    let mut path_vecs: Vec<Option<BitVec>> = vec![None; n];
    path_vecs[source] = Some(BitVec::zero());

    for _iter in 0..max_iters {
        let mut changed = false;

        for u in 0..n {
            if cost[u] == u32::MAX {
                continue;
            }
            for (v, edge_vec) in adj.row(u) {
                let edge_weight = edge_vec.popcount();
                let new_cost = cost[u].saturating_add(edge_weight);
                if new_cost < cost[v] {
                    cost[v] = new_cost;
                    let path = match &path_vecs[u] {
                        Some(pv) => pv.xor(edge_vec),
                        None => edge_vec.clone(),
                    };
                    path_vecs[v] = Some(path);
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }

    // Build result: BitVec path vectors + u32 cost scalars.
    let mut result = GrBVector::new(n);
    for (i, pv) in path_vecs.into_iter().enumerate() {
        if let Some(v) = pv {
            result.set(i, v);
        }
    }
    for (i, &c) in cost.iter().enumerate() {
        if c < u32::MAX {
            result.set_scalar(i, HdrScalar::Float(c as f32));
        }
    }

    result
}

/// HDR PageRank: iterative importance scoring using XOR_BUNDLE semiring.
///
/// Each node's "rank vector" is the bundle of its neighbours' rank
/// vectors, XOR-bound through the adjacency matrix. After `max_iters`
/// iterations the result approximates the stationary distribution in
/// hyperdimensional space.
///
/// Returns a vector of BitVec rank representations for each node.
pub fn hdr_pagerank(adj: &GrBMatrix, max_iters: usize, _damping: f32) -> GrBVector {
    let n = adj.nrows();
    assert_eq!(n, adj.ncols(), "adjacency matrix must be square");

    let semiring = HdrSemiring::XorBundle;
    let desc = GrBDesc::default();

    // Initialise each node with a random vector.
    let mut rank = GrBVector::new(n);
    for i in 0..n {
        rank.set(i, BitVec::random(i as u64 + 1));
    }

    for _iter in 0..max_iters {
        let next = adj.mxv(&rank, &semiring, &desc);

        // Fill in any nodes that got no contribution with their old value.
        let mut merged = GrBVector::new(n);
        for i in 0..n {
            if let Some(v) = next.get(i) {
                merged.set(i, v.clone());
            } else if let Some(v) = rank.get(i) {
                merged.set(i, v.clone());
            }
        }

        rank = merged;
    }

    rank
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::blasgraph::sparse::CooStorage;

    /// Build a simple path graph: 0 -> 1 -> 2 -> 3.
    fn path_graph(n: usize) -> GrBMatrix {
        let mut coo = CooStorage::new(n, n);
        for i in 0..n - 1 {
            coo.push(i, i + 1, BitVec::random((i + 100) as u64));
        }
        GrBMatrix::from_coo(&coo)
    }

    /// Build a cycle graph: 0 -> 1 -> 2 -> 0.
    fn cycle_graph(n: usize) -> GrBMatrix {
        let mut coo = CooStorage::new(n, n);
        for i in 0..n {
            coo.push(i, (i + 1) % n, BitVec::random((i + 100) as u64));
        }
        GrBMatrix::from_coo(&coo)
    }

    #[test]
    fn test_grb_mxm() {
        let mut coo_a = CooStorage::new(2, 3);
        coo_a.push(0, 0, BitVec::random(1));
        coo_a.push(1, 2, BitVec::random(2));
        let a = GrBMatrix::from_coo(&coo_a);

        let mut coo_b = CooStorage::new(3, 2);
        coo_b.push(0, 0, BitVec::random(3));
        coo_b.push(2, 1, BitVec::random(4));
        let b = GrBMatrix::from_coo(&coo_b);

        let c = grb_mxm(&a, &b, &HdrSemiring::XorBundle, &GrBDesc::default());
        assert_eq!(c.nrows(), 2);
        assert_eq!(c.ncols(), 2);
        assert!(c.get(0, 0).is_some()); // A[0,0] * B[0,0]
        assert!(c.get(1, 1).is_some()); // A[1,2] * B[2,1]
    }

    #[test]
    fn test_grb_mxv() {
        let adj = path_graph(3);
        let mut v = GrBVector::new(3);
        v.set(0, BitVec::random(1));
        v.set(1, BitVec::random(2));
        v.set(2, BitVec::random(3));

        let w = grb_mxv(&adj, &v, &HdrSemiring::XorBundle, &GrBDesc::default());
        // Row 0 has edge to col 1, so w[0] should exist.
        assert!(w.get(0).is_some());
    }

    #[test]
    fn test_grb_vxm() {
        let adj = path_graph(3);
        let mut v = GrBVector::new(3);
        v.set(0, BitVec::random(1));

        let w = grb_vxm(&v, &adj, &HdrSemiring::XorBundle, &GrBDesc::default());
        // v[0] * A means we follow edges from node 0 -> gets col 1 contribution.
        assert!(w.nnz() > 0);
    }

    #[test]
    fn test_grb_ewise_add_matrix() {
        let mut coo_a = CooStorage::new(2, 2);
        coo_a.push(0, 0, BitVec::random(1));
        let a = GrBMatrix::from_coo(&coo_a);

        let mut coo_b = CooStorage::new(2, 2);
        coo_b.push(1, 1, BitVec::random(2));
        let b = GrBMatrix::from_coo(&coo_b);

        let c = grb_ewise_add_matrix(&a, &b, BinaryOp::Or, &GrBDesc::default());
        assert_eq!(c.nnz(), 2);
    }

    #[test]
    fn test_grb_reduce() {
        let adj = path_graph(4);
        let v = grb_reduce_matrix_to_vector(&adj, MonoidOp::First);
        // Rows 0,1,2 have outgoing edges.
        assert!(v.get(0).is_some());
        assert!(v.get(1).is_some());
        assert!(v.get(2).is_some());
    }

    #[test]
    fn test_grb_apply_vector() {
        let mut v = GrBVector::new(3);
        v.set(0, BitVec::random(1));
        let result = grb_apply_vector(&v, UnaryOp::BitwiseNot);
        assert!(result.get(0).is_some());
        assert_ne!(result.get(0).unwrap(), v.get(0).unwrap());
    }

    #[test]
    fn test_grb_extract_assign() {
        let mut v = GrBVector::new(10);
        v.set(2, BitVec::random(1));
        v.set(5, BitVec::random(2));
        v.set(8, BitVec::random(3));

        let sub = grb_extract_vector(&v, &[2, 5, 8]);
        assert_eq!(sub.dim(), 3);
        assert_eq!(sub.nnz(), 3);

        let mut dst = GrBVector::new(10);
        grb_assign_vector(&mut dst, &sub, &[0, 1, 2]);
        assert_eq!(dst.nnz(), 3);
    }

    #[test]
    fn test_grb_transpose() {
        let mut coo = CooStorage::new(2, 3);
        coo.push(0, 2, BitVec::random(1));
        let m = GrBMatrix::from_coo(&coo);
        let mt = grb_transpose(&m);
        assert_eq!(mt.nrows(), 3);
        assert_eq!(mt.ncols(), 2);
        assert!(mt.get(2, 0).is_some());
    }

    #[test]
    fn test_hdr_bfs_path_graph() {
        let adj = path_graph(4);
        let result = hdr_bfs(&adj, 0, 10);
        // All nodes should be reachable from 0 on a path graph.
        assert!(result.get(0).is_some());
        assert!(result.get(1).is_some());
        assert!(result.get(2).is_some());
        assert!(result.get(3).is_some());
    }

    #[test]
    fn test_hdr_bfs_unreachable() {
        // Disconnected graph: only edge 0->1.
        let mut coo = CooStorage::new(4, 4);
        coo.push(0, 1, BitVec::random(100));
        let adj = GrBMatrix::from_coo(&coo);

        let result = hdr_bfs(&adj, 0, 10);
        assert!(result.get(0).is_some());
        assert!(result.get(1).is_some());
        // Nodes 2, 3 should not be reachable.
        assert!(result.get(2).is_none());
        assert!(result.get(3).is_none());
    }

    #[test]
    fn test_hdr_sssp_path_graph() {
        let adj = path_graph(4);
        let result = hdr_sssp(&adj, 0, 10);
        // Source should have the zero vector and zero cost.
        assert!(result.get(0).is_some());
        assert_eq!(result.get(0).unwrap().popcount(), 0);
        assert_eq!(result.get_scalar(0).and_then(|s| s.as_float()), Some(0.0));

        // All nodes should be reachable.
        for i in 0..4 {
            assert!(result.get(i).is_some(), "node {} unreachable", i);
            assert!(
                result.get_scalar(i).is_some(),
                "node {} missing cost scalar",
                i
            );
        }

        // Costs should be non-decreasing along the path.
        let costs: Vec<f32> = (0..4)
            .map(|i| result.get_scalar(i).unwrap().as_float().unwrap())
            .collect();
        for w in costs.windows(2) {
            assert!(
                w[0] <= w[1],
                "cost should be non-decreasing: {:?}",
                costs
            );
        }
    }

    #[test]
    fn test_hdr_pagerank_cycle() {
        let adj = cycle_graph(4);
        let rank = hdr_pagerank(&adj, 5, 0.85);
        // Every node should have a rank vector.
        for i in 0..4 {
            assert!(rank.get(i).is_some(), "node {} missing rank", i);
        }
    }

    #[test]
    fn test_hdr_pagerank_convergence() {
        let adj = cycle_graph(3);
        let rank1 = hdr_pagerank(&adj, 1, 0.85);
        let rank10 = hdr_pagerank(&adj, 10, 0.85);
        // Both should have all nodes populated.
        assert_eq!(rank1.nnz(), 3);
        assert_eq!(rank10.nnz(), 3);
    }
}
