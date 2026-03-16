// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # Sparse Matrix of Hyperdimensional Vectors
//!
//! [`GrBMatrix`] is a sparse matrix whose elements are [`BitVec`] values.
//! It supports GraphBLAS-style operations: mxm, mxv, vxm, element-wise
//! add/mult, transpose, extract, apply, reduce_rows, reduce_cols, and reduce.

use crate::graph::blasgraph::descriptor::Descriptor;
use crate::graph::blasgraph::semiring::{apply_binary_op, apply_monoid, Semiring};
use crate::graph::blasgraph::sparse::{CooStorage, CsrStorage};
use crate::graph::blasgraph::types::{BinaryOp, BitVec, HdrScalar, MonoidOp, UnaryOp};
use crate::graph::blasgraph::vector::GrBVector;

/// A sparse matrix of [`BitVec`] elements stored in CSR format.
///
/// Rows and columns are zero-indexed. Structural zeros (absent entries)
/// are never stored.
#[derive(Clone, Debug)]
pub struct GrBMatrix {
    /// Underlying CSR storage.
    storage: CsrStorage,
}

impl GrBMatrix {
    /// Create an empty matrix with the given dimensions.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            storage: CsrStorage::new(nrows, ncols),
        }
    }

    /// Build a matrix from COO (triplet) data.
    pub fn from_coo(coo: &CooStorage) -> Self {
        Self {
            storage: coo.to_csr(),
        }
    }

    /// Build a matrix from CSR storage.
    pub fn from_csr(csr: CsrStorage) -> Self {
        Self { storage: csr }
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.storage.nrows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.storage.ncols
    }

    /// Number of stored (non-zero) entries.
    pub fn nnz(&self) -> usize {
        self.storage.nnz()
    }

    /// Get the element at `(row, col)`, if present.
    pub fn get(&self, row: usize, col: usize) -> Option<&BitVec> {
        self.storage.get(row, col)
    }

    /// Set an element at `(row, col)`. Rebuilds CSR from COO.
    ///
    /// For bulk construction, prefer building a [`CooStorage`] and calling
    /// [`GrBMatrix::from_coo`].
    pub fn set(&mut self, row: usize, col: usize, value: BitVec) {
        assert!(row < self.nrows(), "row out of bounds");
        assert!(col < self.ncols(), "col out of bounds");

        let mut coo = self.storage.to_coo();
        // Remove existing entry at (row, col) if any.
        let mut found = false;
        for i in 0..coo.rows.len() {
            if coo.rows[i] == row && coo.cols[i] == col {
                coo.vals[i] = value.clone();
                found = true;
                break;
            }
        }
        if !found {
            coo.push(row, col, value);
        }
        self.storage = coo.to_csr();
    }

    /// Borrow the underlying CSR storage.
    pub fn csr(&self) -> &CsrStorage {
        &self.storage
    }

    /// Iterate over the entries in a specific row as `(col, &BitVec)`.
    pub fn row(&self, i: usize) -> impl Iterator<Item = (usize, &BitVec)> {
        self.storage.row(i)
    }

    /// Iterate over all entries as `(row, col, &BitVec)`.
    pub fn iter(&self) -> MatrixIter<'_> {
        MatrixIter {
            matrix: self,
            row: 0,
            pos: 0,
        }
    }

    /// Transpose the matrix, returning a new matrix.
    pub fn transpose(&self) -> GrBMatrix {
        GrBMatrix {
            storage: self.storage.transpose(),
        }
    }

    /// Matrix-matrix multiply: `C = A * B` under the given semiring.
    ///
    /// Dimensions: `A` is `m x k`, `B` is `k x n`, result is `m x n`.
    pub fn mxm(&self, other: &GrBMatrix, semiring: &dyn Semiring, desc: &Descriptor) -> GrBMatrix {
        let a = if desc.transpose_input0() {
            self.transpose()
        } else {
            self.clone()
        };
        let b = if desc.transpose_input1() {
            other.transpose()
        } else {
            other.clone()
        };

        assert_eq!(
            a.ncols(),
            b.nrows(),
            "dimension mismatch: A cols ({}) != B rows ({})",
            a.ncols(),
            b.nrows()
        );

        let bt = b.transpose();
        let mut coo = CooStorage::new(a.nrows(), b.ncols());

        for i in 0..a.nrows() {
            for j in 0..b.ncols() {
                let mut acc = semiring.zero();
                for (k, a_val) in a.row(i) {
                    if let Some(b_val) = bt.get(j, k) {
                        let prod = semiring.multiply(
                            &HdrScalar::Vector(a_val.clone()),
                            &HdrScalar::Vector(b_val.clone()),
                        );
                        acc = semiring.add(&acc, &prod);
                    }
                }
                if let HdrScalar::Vector(v) = acc {
                    if !v.is_zero() {
                        coo.push(i, j, v);
                    }
                }
            }
        }

        GrBMatrix::from_coo(&coo)
    }

    /// Matrix-vector multiply: `w = A * u` under the given semiring.
    pub fn mxv(&self, vec: &GrBVector, semiring: &dyn Semiring, desc: &Descriptor) -> GrBVector {
        let a = if desc.transpose_input0() {
            self.transpose()
        } else {
            self.clone()
        };

        assert_eq!(
            a.ncols(),
            vec.dim(),
            "dimension mismatch: A cols ({}) != vector dim ({})",
            a.ncols(),
            vec.dim()
        );

        let mut result = GrBVector::new(a.nrows());

        for i in 0..a.nrows() {
            let mut acc = semiring.zero();
            for (j, a_val) in a.row(i) {
                if let Some(v_val) = vec.get(j) {
                    let prod = semiring.multiply(
                        &HdrScalar::Vector(a_val.clone()),
                        &HdrScalar::Vector(v_val.clone()),
                    );
                    acc = semiring.add(&acc, &prod);
                }
            }
            if let HdrScalar::Vector(v) = acc {
                if !v.is_zero() {
                    result.set(i, v);
                }
            }
        }

        result
    }

    /// Vector-matrix multiply: `w = u * A` under the given semiring.
    pub fn vxm(&self, vec: &GrBVector, semiring: &dyn Semiring, desc: &Descriptor) -> GrBVector {
        let at = self.transpose();
        at.mxv(vec, semiring, desc)
    }

    /// Element-wise addition: `C = A + B` under the given binary op.
    ///
    /// The result has the union of non-zero patterns.
    pub fn ewise_add(&self, other: &GrBMatrix, op: BinaryOp, _desc: &Descriptor) -> GrBMatrix {
        assert_eq!(self.nrows(), other.nrows(), "row dimension mismatch");
        assert_eq!(self.ncols(), other.ncols(), "col dimension mismatch");

        let mut coo = CooStorage::new(self.nrows(), self.ncols());

        for i in 0..self.nrows() {
            let mut a_map: std::collections::BTreeMap<usize, &BitVec> =
                std::collections::BTreeMap::new();
            for (c, v) in self.row(i) {
                a_map.insert(c, v);
            }
            let mut b_map: std::collections::BTreeMap<usize, &BitVec> =
                std::collections::BTreeMap::new();
            for (c, v) in other.row(i) {
                b_map.insert(c, v);
            }

            let mut all_cols: Vec<usize> = a_map.keys().chain(b_map.keys()).copied().collect();
            all_cols.sort_unstable();
            all_cols.dedup();

            for c in all_cols {
                let val = match (a_map.get(&c), b_map.get(&c)) {
                    (Some(a), Some(b)) => apply_binary_op(
                        op,
                        &HdrScalar::Vector((*a).clone()),
                        &HdrScalar::Vector((*b).clone()),
                    ),
                    (Some(a), None) => HdrScalar::Vector((*a).clone()),
                    (None, Some(b)) => HdrScalar::Vector((*b).clone()),
                    (None, None) => unreachable!(),
                };
                if let HdrScalar::Vector(v) = val {
                    coo.push(i, c, v);
                }
            }
        }

        GrBMatrix::from_coo(&coo)
    }

    /// Element-wise multiplication: `C = A .* B` under the given binary op.
    ///
    /// The result has the intersection of non-zero patterns.
    pub fn ewise_mult(&self, other: &GrBMatrix, op: BinaryOp, _desc: &Descriptor) -> GrBMatrix {
        assert_eq!(self.nrows(), other.nrows(), "row dimension mismatch");
        assert_eq!(self.ncols(), other.ncols(), "col dimension mismatch");

        let mut coo = CooStorage::new(self.nrows(), self.ncols());

        for i in 0..self.nrows() {
            for (c, a_val) in self.row(i) {
                if let Some(b_val) = other.get(i, c) {
                    let val = apply_binary_op(
                        op,
                        &HdrScalar::Vector(a_val.clone()),
                        &HdrScalar::Vector(b_val.clone()),
                    );
                    if let HdrScalar::Vector(v) = val {
                        coo.push(i, c, v);
                    }
                }
            }
        }

        GrBMatrix::from_coo(&coo)
    }

    /// Extract a submatrix at the given row and column indices.
    pub fn extract(&self, rows: &[usize], cols: &[usize]) -> GrBMatrix {
        let mut coo = CooStorage::new(rows.len(), cols.len());
        for (ri, &r) in rows.iter().enumerate() {
            for (ci, &c) in cols.iter().enumerate() {
                if let Some(v) = self.get(r, c) {
                    coo.push(ri, ci, v.clone());
                }
            }
        }
        GrBMatrix::from_coo(&coo)
    }

    /// Apply a unary operator to every stored element.
    pub fn apply(&self, op: UnaryOp) -> GrBMatrix {
        let mut coo = CooStorage::new(self.nrows(), self.ncols());
        for (r, c, v) in self.iter() {
            let new_v = apply_unary(op, v);
            coo.push(r, c, new_v);
        }
        GrBMatrix::from_coo(&coo)
    }

    /// Reduce each row using a monoid, producing a vector of length `nrows`.
    pub fn reduce_rows(&self, monoid: MonoidOp) -> GrBVector {
        let mut result = GrBVector::new(self.nrows());
        for i in 0..self.nrows() {
            let mut acc = HdrScalar::Empty;
            for (_c, v) in self.row(i) {
                let scalar = HdrScalar::Vector(v.clone());
                acc = apply_monoid(monoid, &acc, &scalar);
            }
            if let HdrScalar::Vector(v) = acc {
                result.set(i, v);
            }
        }
        result
    }

    /// Reduce each column using a monoid, producing a vector of length `ncols`.
    pub fn reduce_cols(&self, monoid: MonoidOp) -> GrBVector {
        let t = self.transpose();
        t.reduce_rows(monoid)
    }

    /// Reduce the entire matrix to a single scalar using a monoid.
    pub fn reduce(&self, monoid: MonoidOp) -> HdrScalar {
        let mut acc = HdrScalar::Empty;
        for (_r, _c, v) in self.iter() {
            let scalar = HdrScalar::Vector(v.clone());
            acc = apply_monoid(monoid, &acc, &scalar);
        }
        acc
    }
}

/// Apply a unary operator to a single [`BitVec`].
fn apply_unary(op: UnaryOp, v: &BitVec) -> BitVec {
    match op {
        UnaryOp::BitwiseNot => v.not(),
        UnaryOp::Identity => v.clone(),
        UnaryOp::Permute => v.permute(1),
        UnaryOp::Negate | UnaryOp::Density => v.clone(),
    }
}

/// Iterator over all `(row, col, &BitVec)` entries in a [`GrBMatrix`].
pub struct MatrixIter<'a> {
    matrix: &'a GrBMatrix,
    row: usize,
    pos: usize,
}

impl<'a> Iterator for MatrixIter<'a> {
    type Item = (usize, usize, &'a BitVec);

    fn next(&mut self) -> Option<Self::Item> {
        let csr = self.matrix.csr();
        while self.row < csr.nrows {
            let end = csr.row_ptrs[self.row + 1];
            if self.pos < end {
                let col = csr.col_indices[self.pos];
                let val = &csr.values[self.pos];
                self.pos += 1;
                return Some((self.row, col, val));
            }
            self.row += 1;
            if self.row < csr.nrows {
                self.pos = csr.row_ptrs[self.row];
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::blasgraph::descriptor::GrBDesc;
    use crate::graph::blasgraph::semiring::HdrSemiring;

    fn make_identity_matrix(n: usize) -> GrBMatrix {
        let mut coo = CooStorage::new(n, n);
        for i in 0..n {
            coo.push(i, i, BitVec::random((i + 1) as u64));
        }
        GrBMatrix::from_coo(&coo)
    }

    #[test]
    fn test_new_matrix() {
        let m = GrBMatrix::new(3, 4);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 4);
        assert_eq!(m.nnz(), 0);
    }

    #[test]
    fn test_set_get() {
        let mut m = GrBMatrix::new(3, 3);
        let v = BitVec::random(42);
        m.set(1, 2, v.clone());
        assert_eq!(m.get(1, 2).unwrap(), &v);
        assert!(m.get(0, 0).is_none());
        assert_eq!(m.nnz(), 1);
    }

    #[test]
    fn test_from_coo() {
        let mut coo = CooStorage::new(2, 2);
        coo.push(0, 0, BitVec::random(1));
        coo.push(1, 1, BitVec::random(2));
        let m = GrBMatrix::from_coo(&coo);
        assert_eq!(m.nnz(), 2);
    }

    #[test]
    fn test_transpose() {
        let mut coo = CooStorage::new(2, 3);
        let v = BitVec::random(1);
        coo.push(0, 2, v.clone());
        let m = GrBMatrix::from_coo(&coo);
        let mt = m.transpose();
        assert_eq!(mt.nrows(), 3);
        assert_eq!(mt.ncols(), 2);
        assert_eq!(mt.get(2, 0).unwrap(), &v);
    }

    #[test]
    fn test_mxm_identity() {
        let ident = make_identity_matrix(3);
        let mut coo = CooStorage::new(3, 3);
        let v = BitVec::random(100);
        coo.push(0, 1, v.clone());
        let m = GrBMatrix::from_coo(&coo);
        let result = m.mxm(&ident, &HdrSemiring::XorBundle, &GrBDesc::default());
        assert!(result.nnz() > 0);
        assert!(result.get(0, 1).is_some());
    }

    #[test]
    fn test_mxv() {
        let mut coo = CooStorage::new(2, 3);
        coo.push(0, 0, BitVec::random(1));
        coo.push(0, 1, BitVec::random(2));
        coo.push(1, 2, BitVec::random(3));
        let m = GrBMatrix::from_coo(&coo);

        let mut v = GrBVector::new(3);
        v.set(0, BitVec::random(10));
        v.set(1, BitVec::random(20));
        v.set(2, BitVec::random(30));

        let result = m.mxv(&v, &HdrSemiring::XorBundle, &GrBDesc::default());
        assert!(result.get(0).is_some());
        assert!(result.get(1).is_some());
    }

    #[test]
    fn test_ewise_add() {
        let mut coo_a = CooStorage::new(2, 2);
        coo_a.push(0, 0, BitVec::random(1));
        let a = GrBMatrix::from_coo(&coo_a);

        let mut coo_b = CooStorage::new(2, 2);
        coo_b.push(0, 1, BitVec::random(2));
        let b = GrBMatrix::from_coo(&coo_b);

        let result = a.ewise_add(&b, BinaryOp::Or, &GrBDesc::default());
        assert_eq!(result.nnz(), 2);
    }

    #[test]
    fn test_ewise_mult() {
        let mut coo_a = CooStorage::new(2, 2);
        coo_a.push(0, 0, BitVec::random(1));
        coo_a.push(0, 1, BitVec::random(2));
        let a = GrBMatrix::from_coo(&coo_a);

        let mut coo_b = CooStorage::new(2, 2);
        coo_b.push(0, 0, BitVec::random(3));
        coo_b.push(1, 1, BitVec::random(4));
        let b = GrBMatrix::from_coo(&coo_b);

        let result = a.ewise_mult(&b, BinaryOp::Xor, &GrBDesc::default());
        assert_eq!(result.nnz(), 1);
    }

    #[test]
    fn test_extract_submatrix() {
        let mut coo = CooStorage::new(4, 4);
        coo.push(1, 2, BitVec::random(1));
        coo.push(2, 3, BitVec::random(2));
        coo.push(0, 0, BitVec::random(3));
        let m = GrBMatrix::from_coo(&coo);

        let sub = m.extract(&[1, 2], &[2, 3]);
        assert_eq!(sub.nrows(), 2);
        assert_eq!(sub.ncols(), 2);
        assert!(sub.get(0, 0).is_some());
        assert!(sub.get(1, 1).is_some());
    }

    #[test]
    fn test_apply_not() {
        let mut coo = CooStorage::new(2, 2);
        let v = BitVec::random(1);
        coo.push(0, 0, v.clone());
        let m = GrBMatrix::from_coo(&coo);
        let result = m.apply(UnaryOp::BitwiseNot);
        assert_eq!(*result.get(0, 0).unwrap(), v.not());
    }

    #[test]
    fn test_reduce_rows() {
        let mut coo = CooStorage::new(2, 3);
        coo.push(0, 0, BitVec::random(1));
        coo.push(0, 1, BitVec::random(2));
        coo.push(1, 2, BitVec::random(3));
        let m = GrBMatrix::from_coo(&coo);
        let result = m.reduce_rows(MonoidOp::First);
        assert!(result.get(0).is_some());
        assert!(result.get(1).is_some());
    }

    #[test]
    fn test_reduce_cols() {
        let mut coo = CooStorage::new(3, 2);
        coo.push(0, 0, BitVec::random(1));
        coo.push(1, 0, BitVec::random(2));
        coo.push(2, 1, BitVec::random(3));
        let m = GrBMatrix::from_coo(&coo);
        let result = m.reduce_cols(MonoidOp::First);
        assert_eq!(result.dim(), 2);
        assert!(result.get(0).is_some());
        assert!(result.get(1).is_some());
    }

    #[test]
    fn test_reduce_scalar() {
        let mut coo = CooStorage::new(2, 2);
        coo.push(0, 0, BitVec::random(1));
        coo.push(1, 1, BitVec::random(2));
        let m = GrBMatrix::from_coo(&coo);
        let result = m.reduce(MonoidOp::First);
        assert!(result.as_vector().is_some());
    }

    #[test]
    fn test_iterator() {
        let mut coo = CooStorage::new(3, 3);
        coo.push(0, 1, BitVec::random(1));
        coo.push(2, 0, BitVec::random(2));
        let m = GrBMatrix::from_coo(&coo);
        let entries: Vec<(usize, usize)> = m.iter().map(|(r, c, _)| (r, c)).collect();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains(&(0, 1)));
        assert!(entries.contains(&(2, 0)));
    }
}
