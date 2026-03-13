// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # Sparse Vector of Hyperdimensional Vectors
//!
//! [`GrBVector`] is a sparse vector whose elements are [`BitVec`] values.
//! Supports element-wise operations, apply, reduce, and nearest-neighbour
//! queries via Hamming distance.

use crate::graph::blasgraph::semiring::{apply_binary_op, apply_monoid};
use crate::graph::blasgraph::sparse::SparseVec;
use crate::graph::blasgraph::types::{
    hamming_to_similarity, BinaryOp, BitVec, HdrScalar, MonoidOp, UnaryOp,
};

/// A sparse vector of [`BitVec`] elements.
///
/// Elements are stored sorted by index. Absent positions are structural
/// zeros and occupy no memory.
#[derive(Clone, Debug)]
pub struct GrBVector {
    /// Underlying sparse storage.
    storage: SparseVec,
}

impl GrBVector {
    /// Create an empty vector of the given dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            storage: SparseVec::new(dim),
        }
    }

    /// Dimension (logical length) of the vector.
    pub fn dim(&self) -> usize {
        self.storage.dim
    }

    /// Number of stored (non-zero) entries.
    pub fn nnz(&self) -> usize {
        self.storage.nnz()
    }

    /// Set an element at the given index.
    pub fn set(&mut self, index: usize, value: BitVec) {
        self.storage.set(index, value);
    }

    /// Get the element at the given index, if present.
    pub fn get(&self, index: usize) -> Option<&BitVec> {
        self.storage.get(index)
    }

    /// Iterate over `(index, &BitVec)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &BitVec)> {
        self.storage.iter()
    }

    /// Element-wise addition with another vector under a binary op.
    ///
    /// Produces the union of indices. Where both have an entry the binary
    /// op is applied; otherwise the lone entry is kept.
    pub fn ewise_add(&self, other: &GrBVector, op: BinaryOp) -> GrBVector {
        assert_eq!(self.dim(), other.dim(), "dimension mismatch");
        let mut result = GrBVector::new(self.dim());

        let mut ai = self.storage.entries.iter().peekable();
        let mut bi = other.storage.entries.iter().peekable();

        loop {
            match (ai.peek(), bi.peek()) {
                (Some(a), Some(b)) => {
                    if a.index < b.index {
                        result.set(a.index, a.value.clone());
                        ai.next();
                    } else if a.index > b.index {
                        result.set(b.index, b.value.clone());
                        bi.next();
                    } else {
                        let val = apply_binary_op(
                            op,
                            &HdrScalar::Vector(a.value.clone()),
                            &HdrScalar::Vector(b.value.clone()),
                        );
                        if let HdrScalar::Vector(v) = val {
                            result.set(a.index, v);
                        }
                        ai.next();
                        bi.next();
                    }
                }
                (Some(a), None) => {
                    result.set(a.index, a.value.clone());
                    ai.next();
                }
                (None, Some(b)) => {
                    result.set(b.index, b.value.clone());
                    bi.next();
                }
                (None, None) => break,
            }
        }

        result
    }

    /// Element-wise multiplication with another vector under a binary op.
    ///
    /// Produces the intersection of indices.
    pub fn ewise_mult(&self, other: &GrBVector, op: BinaryOp) -> GrBVector {
        assert_eq!(self.dim(), other.dim(), "dimension mismatch");
        let mut result = GrBVector::new(self.dim());

        for (idx, a_val) in self.iter() {
            if let Some(b_val) = other.get(idx) {
                let val = apply_binary_op(
                    op,
                    &HdrScalar::Vector(a_val.clone()),
                    &HdrScalar::Vector(b_val.clone()),
                );
                if let HdrScalar::Vector(v) = val {
                    result.set(idx, v);
                }
            }
        }

        result
    }

    /// Apply a unary operator to every stored element.
    pub fn apply(&self, op: UnaryOp) -> GrBVector {
        let mut result = GrBVector::new(self.dim());
        for (idx, v) in self.iter() {
            let new_v = match op {
                UnaryOp::BitwiseNot => v.not(),
                UnaryOp::Identity => v.clone(),
                UnaryOp::Permute => v.permute(1),
                UnaryOp::Negate | UnaryOp::Density => v.clone(),
            };
            result.set(idx, new_v);
        }
        result
    }

    /// Reduce all stored elements using a monoid, producing a scalar.
    pub fn reduce(&self, monoid: MonoidOp) -> HdrScalar {
        let mut acc = HdrScalar::Empty;
        for (_idx, v) in self.iter() {
            let scalar = HdrScalar::Vector(v.clone());
            acc = apply_monoid(monoid, &acc, &scalar);
        }
        acc
    }

    /// Find the `k` nearest stored entries to `query` by Hamming distance.
    ///
    /// Returns `(index, distance)` pairs sorted by ascending distance.
    pub fn find_nearest(&self, query: &BitVec, k: usize) -> Vec<(usize, u32)> {
        let mut distances: Vec<(usize, u32)> = self
            .iter()
            .map(|(idx, v)| (idx, v.hamming_distance(query)))
            .collect();
        distances.sort_by_key(|&(_, d)| d);
        distances.truncate(k);
        distances
    }

    /// Find all stored entries within `max_distance` Hamming distance of `query`.
    ///
    /// Returns `(index, distance)` pairs sorted by ascending distance.
    pub fn find_within(&self, query: &BitVec, max_distance: u32) -> Vec<(usize, u32)> {
        let mut results: Vec<(usize, u32)> = self
            .iter()
            .filter_map(|(idx, v)| {
                let d = v.hamming_distance(query);
                if d <= max_distance {
                    Some((idx, d))
                } else {
                    None
                }
            })
            .collect();
        results.sort_by_key(|&(_, d)| d);
        results
    }

    /// Find the most similar stored entry to `query`.
    ///
    /// Returns `(index, similarity)` where similarity is in `[0.0, 1.0]`.
    pub fn find_most_similar(&self, query: &BitVec) -> Option<(usize, f32)> {
        self.iter()
            .map(|(idx, v)| {
                let d = v.hamming_distance(query);
                (idx, hamming_to_similarity(d))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Extract a sub-vector at the given indices.
    pub fn extract(&self, indices: &[usize]) -> GrBVector {
        let mut result = GrBVector::new(indices.len());
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            if let Some(v) = self.get(old_idx) {
                result.set(new_idx, v.clone());
            }
        }
        result
    }

    /// Assign values from `src` into this vector at the given indices.
    pub fn assign(&mut self, src: &GrBVector, indices: &[usize]) {
        for (src_idx, &dst_idx) in indices.iter().enumerate() {
            if let Some(v) = src.get(src_idx) {
                self.set(dst_idx, v.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_vector() {
        let v = GrBVector::new(10);
        assert_eq!(v.dim(), 10);
        assert_eq!(v.nnz(), 0);
    }

    #[test]
    fn test_set_get() {
        let mut v = GrBVector::new(5);
        let bv = BitVec::random(42);
        v.set(3, bv.clone());
        assert_eq!(v.get(3).unwrap(), &bv);
        assert!(v.get(0).is_none());
        assert_eq!(v.nnz(), 1);
    }

    #[test]
    fn test_ewise_add() {
        let mut a = GrBVector::new(5);
        a.set(0, BitVec::random(1));
        a.set(2, BitVec::random(2));

        let mut b = GrBVector::new(5);
        b.set(1, BitVec::random(3));
        b.set(2, BitVec::random(4));

        let result = a.ewise_add(&b, BinaryOp::Or);
        assert_eq!(result.nnz(), 3);
        assert!(result.get(0).is_some());
        assert!(result.get(1).is_some());
        assert!(result.get(2).is_some());
    }

    #[test]
    fn test_ewise_mult() {
        let mut a = GrBVector::new(5);
        a.set(0, BitVec::random(1));
        a.set(2, BitVec::random(2));

        let mut b = GrBVector::new(5);
        b.set(1, BitVec::random(3));
        b.set(2, BitVec::random(4));

        let result = a.ewise_mult(&b, BinaryOp::Xor);
        assert_eq!(result.nnz(), 1);
        assert!(result.get(2).is_some());
    }

    #[test]
    fn test_apply() {
        let mut v = GrBVector::new(3);
        let bv = BitVec::random(1);
        v.set(0, bv.clone());
        let result = v.apply(UnaryOp::BitwiseNot);
        assert_eq!(result.get(0).unwrap(), &bv.not());
    }

    #[test]
    fn test_reduce() {
        let mut v = GrBVector::new(3);
        v.set(0, BitVec::random(1));
        v.set(2, BitVec::random(2));
        let result = v.reduce(MonoidOp::First);
        assert!(result.as_vector().is_some());
    }

    #[test]
    fn test_find_nearest() {
        let mut v = GrBVector::new(5);
        let query = BitVec::random(100);
        v.set(0, query.clone());
        v.set(1, BitVec::random(200));
        v.set(2, BitVec::random(300));

        let nearest = v.find_nearest(&query, 2);
        assert_eq!(nearest.len(), 2);
        assert_eq!(nearest[0].0, 0);
        assert_eq!(nearest[0].1, 0);
    }

    #[test]
    fn test_find_within() {
        let mut v = GrBVector::new(5);
        let query = BitVec::random(100);
        v.set(0, query.clone());
        v.set(1, query.not());

        let within = v.find_within(&query, 100);
        assert_eq!(within.len(), 1);
        assert_eq!(within[0].0, 0);
    }

    #[test]
    fn test_find_most_similar() {
        let mut v = GrBVector::new(3);
        let query = BitVec::random(1);
        v.set(0, query.clone());
        v.set(1, BitVec::random(2));

        let (idx, sim) = v.find_most_similar(&query).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(sim, 1.0);
    }

    #[test]
    fn test_find_most_similar_empty() {
        let v = GrBVector::new(5);
        assert!(v.find_most_similar(&BitVec::random(1)).is_none());
    }

    #[test]
    fn test_extract() {
        let mut v = GrBVector::new(10);
        v.set(3, BitVec::random(1));
        v.set(7, BitVec::random(2));

        let sub = v.extract(&[3, 5, 7]);
        assert_eq!(sub.dim(), 3);
        assert!(sub.get(0).is_some());
        assert!(sub.get(1).is_none());
        assert!(sub.get(2).is_some());
    }

    #[test]
    fn test_assign() {
        let mut dst = GrBVector::new(10);
        let mut src = GrBVector::new(2);
        let v1 = BitVec::random(1);
        let v2 = BitVec::random(2);
        src.set(0, v1.clone());
        src.set(1, v2.clone());

        dst.assign(&src, &[5, 8]);
        assert_eq!(dst.get(5).unwrap(), &v1);
        assert_eq!(dst.get(8).unwrap(), &v2);
    }

    #[test]
    fn test_iter() {
        let mut v = GrBVector::new(5);
        v.set(1, BitVec::random(10));
        v.set(3, BitVec::random(20));
        let indices: Vec<usize> = v.iter().map(|(i, _)| i).collect();
        assert_eq!(indices, vec![1, 3]);
    }
}
