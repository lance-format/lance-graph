// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # Sparse Storage Formats
//!
//! Provides Compressed Sparse Row (CSR) and Coordinate (COO) storage
//! for sparse matrices and vectors of [`BitVec`] elements.

use crate::graph::blasgraph::types::BitVec;

/// A single entry in a sparse structure: `(index, value)`.
#[derive(Clone, Debug)]
pub struct SparseEntry {
    /// Row or column index of this entry.
    pub index: usize,
    /// The hyperdimensional vector stored at this position.
    pub value: BitVec,
}

/// A sparse vector stored as sorted `(index, value)` pairs.
#[derive(Clone, Debug)]
pub struct SparseVec {
    /// Dimension (logical length) of the vector.
    pub dim: usize,
    /// Sorted entries.
    pub entries: Vec<SparseEntry>,
}

impl SparseVec {
    /// Create an empty sparse vector of the given dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            entries: Vec::new(),
        }
    }

    /// Number of stored (non-zero) entries.
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    /// Insert or replace an entry. Maintains sorted order.
    pub fn set(&mut self, index: usize, value: BitVec) {
        assert!(
            index < self.dim,
            "index {} out of bounds (dim={})",
            index,
            self.dim
        );
        match self.entries.binary_search_by_key(&index, |e| e.index) {
            Ok(pos) => self.entries[pos].value = value,
            Err(pos) => self.entries.insert(pos, SparseEntry { index, value }),
        }
    }

    /// Get the value at the given index, if present.
    pub fn get(&self, index: usize) -> Option<&BitVec> {
        self.entries
            .binary_search_by_key(&index, |e| e.index)
            .ok()
            .map(|pos| &self.entries[pos].value)
    }

    /// Iterate over `(index, &BitVec)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &BitVec)> {
        self.entries.iter().map(|e| (e.index, &e.value))
    }
}

/// Coordinate (triplet) storage format for sparse matrices.
///
/// Stores `(row, col, value)` triplets. Suitable for incremental
/// construction before converting to CSR.
#[derive(Clone, Debug)]
pub struct CooStorage {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Row indices.
    pub rows: Vec<usize>,
    /// Column indices.
    pub cols: Vec<usize>,
    /// Values.
    pub vals: Vec<BitVec>,
}

impl CooStorage {
    /// Create an empty COO storage with the given dimensions.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            rows: Vec::new(),
            cols: Vec::new(),
            vals: Vec::new(),
        }
    }

    /// Number of stored entries.
    pub fn nnz(&self) -> usize {
        self.vals.len()
    }

    /// Add a triplet `(row, col, value)`.
    pub fn push(&mut self, row: usize, col: usize, value: BitVec) {
        assert!(
            row < self.nrows,
            "row {} out of bounds (nrows={})",
            row,
            self.nrows
        );
        assert!(
            col < self.ncols,
            "col {} out of bounds (ncols={})",
            col,
            self.ncols
        );
        self.rows.push(row);
        self.cols.push(col);
        self.vals.push(value);
    }

    /// Convert to CSR format.
    pub fn to_csr(&self) -> CsrStorage {
        let mut csr = CsrStorage::new(self.nrows, self.ncols);

        // Sort triplets by (row, col)
        let mut indices: Vec<usize> = (0..self.nnz()).collect();
        indices.sort_by(|&a, &b| {
            self.rows[a]
                .cmp(&self.rows[b])
                .then(self.cols[a].cmp(&self.cols[b]))
        });

        let mut row_ptrs = vec![0usize; self.nrows + 1];
        let mut col_indices = Vec::with_capacity(self.nnz());
        let mut values = Vec::with_capacity(self.nnz());

        for &idx in &indices {
            let r = self.rows[idx];
            row_ptrs[r + 1] += 1;
            col_indices.push(self.cols[idx]);
            values.push(self.vals[idx].clone());
        }

        // Prefix sum
        for i in 1..=self.nrows {
            row_ptrs[i] += row_ptrs[i - 1];
        }

        csr.row_ptrs = row_ptrs;
        csr.col_indices = col_indices;
        csr.values = values;
        csr
    }

    /// Iterate over `(row, col, &BitVec)` triplets.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, &BitVec)> {
        self.rows
            .iter()
            .zip(self.cols.iter())
            .zip(self.vals.iter())
            .map(|((&r, &c), v)| (r, c, v))
    }
}

/// Compressed Sparse Row (CSR) storage format.
///
/// The standard format for efficient row-wise access and matrix-vector
/// multiplication.
#[derive(Clone, Debug)]
pub struct CsrStorage {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Row pointers: `row_ptrs[i]..row_ptrs[i+1]` are the entries for row `i`.
    pub row_ptrs: Vec<usize>,
    /// Column indices for each non-zero entry.
    pub col_indices: Vec<usize>,
    /// Values for each non-zero entry.
    pub values: Vec<BitVec>,
}

impl CsrStorage {
    /// Create an empty CSR storage with the given dimensions.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            row_ptrs: vec![0; nrows + 1],
            col_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Number of stored entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get the entries for a specific row as `(col, &BitVec)` pairs.
    pub fn row(&self, i: usize) -> impl Iterator<Item = (usize, &BitVec)> {
        let start = self.row_ptrs[i];
        let end = self.row_ptrs[i + 1];
        self.col_indices[start..end]
            .iter()
            .zip(self.values[start..end].iter())
            .map(|(&c, v)| (c, v))
    }

    /// Get the value at `(row, col)` if present.
    pub fn get(&self, row: usize, col: usize) -> Option<&BitVec> {
        let start = self.row_ptrs[row];
        let end = self.row_ptrs[row + 1];
        let slice = &self.col_indices[start..end];
        slice
            .binary_search(&col)
            .ok()
            .map(|pos| &self.values[start + pos])
    }

    /// Convert back to COO format.
    pub fn to_coo(&self) -> CooStorage {
        let mut coo = CooStorage::new(self.nrows, self.ncols);
        for row in 0..self.nrows {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];
            for idx in start..end {
                coo.push(row, self.col_indices[idx], self.values[idx].clone());
            }
        }
        coo
    }

    /// Transpose the CSR matrix, returning a new CSR.
    pub fn transpose(&self) -> CsrStorage {
        let coo = self.to_coo();
        let mut transposed = CooStorage::new(self.ncols, self.nrows);
        for (r, c, v) in coo.iter() {
            transposed.push(c, r, v.clone());
        }
        transposed.to_csr()
    }
}

/// Selector for the sparse storage format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseFormat {
    /// Compressed Sparse Row.
    Csr,
    /// Coordinate / triplet.
    Coo,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_vec_basic() {
        let mut sv = SparseVec::new(10);
        assert_eq!(sv.nnz(), 0);
        assert!(sv.get(0).is_none());

        sv.set(3, BitVec::random(1));
        sv.set(7, BitVec::random(2));
        assert_eq!(sv.nnz(), 2);
        assert!(sv.get(3).is_some());
        assert!(sv.get(7).is_some());
        assert!(sv.get(5).is_none());
    }

    #[test]
    fn test_sparse_vec_overwrite() {
        let mut sv = SparseVec::new(5);
        sv.set(2, BitVec::random(10));
        let v2 = BitVec::random(20);
        sv.set(2, v2.clone());
        assert_eq!(sv.nnz(), 1);
        assert_eq!(sv.get(2).unwrap(), &v2);
    }

    #[test]
    fn test_sparse_vec_sorted_order() {
        let mut sv = SparseVec::new(10);
        sv.set(5, BitVec::random(1));
        sv.set(1, BitVec::random(2));
        sv.set(8, BitVec::random(3));
        let indices: Vec<usize> = sv.iter().map(|(i, _)| i).collect();
        assert_eq!(indices, vec![1, 5, 8]);
    }

    #[test]
    fn test_coo_basic() {
        let mut coo = CooStorage::new(3, 3);
        coo.push(0, 1, BitVec::random(1));
        coo.push(1, 2, BitVec::random(2));
        coo.push(2, 0, BitVec::random(3));
        assert_eq!(coo.nnz(), 3);
    }

    #[test]
    fn test_coo_to_csr_roundtrip() {
        let mut coo = CooStorage::new(3, 4);
        let v1 = BitVec::random(10);
        let v2 = BitVec::random(20);
        let v3 = BitVec::random(30);
        coo.push(0, 1, v1.clone());
        coo.push(0, 3, v2.clone());
        coo.push(2, 2, v3.clone());

        let csr = coo.to_csr();
        assert_eq!(csr.nrows, 3);
        assert_eq!(csr.ncols, 4);
        assert_eq!(csr.nnz(), 3);

        assert_eq!(csr.get(0, 1).unwrap(), &v1);
        assert_eq!(csr.get(0, 3).unwrap(), &v2);
        assert_eq!(csr.get(2, 2).unwrap(), &v3);
        assert!(csr.get(1, 0).is_none());
    }

    #[test]
    fn test_csr_row_iteration() {
        let mut coo = CooStorage::new(2, 3);
        coo.push(0, 0, BitVec::random(1));
        coo.push(0, 2, BitVec::random(2));
        coo.push(1, 1, BitVec::random(3));

        let csr = coo.to_csr();

        let row0: Vec<usize> = csr.row(0).map(|(c, _)| c).collect();
        assert_eq!(row0, vec![0, 2]);

        let row1: Vec<usize> = csr.row(1).map(|(c, _)| c).collect();
        assert_eq!(row1, vec![1]);
    }

    #[test]
    fn test_csr_transpose() {
        let mut coo = CooStorage::new(2, 3);
        let v1 = BitVec::random(1);
        coo.push(0, 2, v1.clone());
        coo.push(1, 0, BitVec::random(2));

        let csr = coo.to_csr();
        let trans = csr.transpose();
        assert_eq!(trans.nrows, 3);
        assert_eq!(trans.ncols, 2);
        assert_eq!(trans.get(2, 0).unwrap(), &v1);
    }

    #[test]
    fn test_csr_to_coo() {
        let mut coo = CooStorage::new(2, 2);
        coo.push(0, 0, BitVec::random(1));
        coo.push(1, 1, BitVec::random(2));

        let csr = coo.to_csr();
        let coo2 = csr.to_coo();
        assert_eq!(coo2.nnz(), 2);
    }

    #[test]
    fn test_empty_csr() {
        let csr = CsrStorage::new(5, 5);
        assert_eq!(csr.nnz(), 0);
        assert!(csr.get(0, 0).is_none());
        let row: Vec<_> = csr.row(0).collect();
        assert!(row.is_empty());
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_coo_out_of_bounds_row() {
        let mut coo = CooStorage::new(2, 2);
        coo.push(5, 0, BitVec::zero());
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_sparse_vec_out_of_bounds() {
        let mut sv = SparseVec::new(3);
        sv.set(10, BitVec::zero());
    }
}
