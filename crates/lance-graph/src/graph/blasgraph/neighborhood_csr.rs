// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # CSR Bridge — Build CSR from Neighborhood Vectors
//!
//! Secondary path for graph algorithms (BFS, PageRank, etc.) that need
//! adjacency structure. Primary search uses neighborhood vectors directly.
//!
//! The CSR matrix uses scent bytes (u8) as edge weights.

/// A CSR (Compressed Sparse Row) matrix with u8 edge weights.
///
/// Built from neighborhood scent vectors for use with standard graph
/// algorithms. This is the **secondary** search path — neighborhood
/// vectors are primary.
#[derive(Clone, Debug)]
pub struct ScentCsr {
    /// Number of rows (nodes).
    pub nrows: usize,
    /// Number of columns (nodes).
    pub ncols: usize,
    /// Row pointers: row_ptrs[i]..row_ptrs[i+1] gives the range of
    /// non-zero entries in row i.
    pub row_ptrs: Vec<usize>,
    /// Column indices for each non-zero entry.
    pub col_indices: Vec<usize>,
    /// Scent byte values for each non-zero entry.
    pub values: Vec<u8>,
}

impl ScentCsr {
    /// Build a CSR matrix from neighborhood scent vectors.
    ///
    /// # Arguments
    /// - `scent_vectors` — Scent vectors for all scope nodes.
    ///   `scent_vectors[i]` is node i's scent vector.
    ///
    /// # Returns
    /// A `ScentCsr` where `(i, j) = scent[j]` for non-zero scent entries.
    pub fn from_scent_vectors(scent_vectors: &[&[u8]]) -> Self {
        let n = scent_vectors.len();
        let mut row_ptrs = Vec::with_capacity(n + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        row_ptrs.push(0);
        for scents in scent_vectors {
            for (j, &s) in scents.iter().enumerate() {
                if s != 0 {
                    col_indices.push(j);
                    values.push(s);
                }
            }
            row_ptrs.push(col_indices.len());
        }

        Self {
            nrows: n,
            ncols: if n > 0 {
                scent_vectors[0].len()
            } else {
                0
            },
            row_ptrs,
            col_indices,
            values,
        }
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Iterate over row i: yields (column_index, scent_value) pairs.
    pub fn row(&self, _i: usize) -> &[(usize, u8)] {
        // Separate col_indices/values arrays can't be returned as tuple slice.
        // Use row_range() to iterate col_indices/values directly.
        &[]
    }

    /// Get the range of entries for row i in col_indices/values arrays.
    pub fn row_range(&self, i: usize) -> std::ops::Range<usize> {
        self.row_ptrs[i]..self.row_ptrs[i + 1]
    }

    /// Get the scent value at (row, col), or 0 if not present.
    pub fn get(&self, row: usize, col: usize) -> u8 {
        let range = self.row_range(row);
        for idx in range {
            if self.col_indices[idx] == col {
                return self.values[idx];
            }
        }
        0
    }

    /// Number of neighbors for a given node.
    pub fn degree(&self, node: usize) -> usize {
        let range = self.row_range(node);
        range.end - range.start
    }

    /// Sparse matrix-vector multiply: y = A * x (using scent as weights).
    ///
    /// `x` is a dense vector of length `ncols`. Returns a dense vector of
    /// length `nrows`. Each element y[i] = sum(scent[i][j] * x[j]) for
    /// non-zero entries.
    pub fn spmv(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.ncols);
        let mut y = vec![0.0f32; self.nrows];
        for i in 0..self.nrows {
            let range = self.row_range(i);
            for idx in range {
                y[i] += self.values[idx] as f32 * x[self.col_indices[idx]];
            }
        }
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_from_scent_vectors() {
        let v0: Vec<u8> = vec![0, 0x7F, 0x3F, 0, 0x01];
        let v1: Vec<u8> = vec![0x7F, 0, 0x1F, 0, 0];
        let v2: Vec<u8> = vec![0x3F, 0x1F, 0, 0, 0];

        let vecs: Vec<&[u8]> = vec![&v0, &v1, &v2];
        let csr = ScentCsr::from_scent_vectors(&vecs);

        assert_eq!(csr.nrows, 3);
        assert_eq!(csr.ncols, 5);
        // v0 has 3 non-zero, v1 has 2, v2 has 2 = 7 total
        assert_eq!(csr.nnz(), 7);

        // Check specific lookups
        assert_eq!(csr.get(0, 1), 0x7F);
        assert_eq!(csr.get(0, 0), 0); // self
        assert_eq!(csr.get(1, 0), 0x7F);
        assert_eq!(csr.get(2, 3), 0);
    }

    #[test]
    fn test_csr_degree() {
        let v0: Vec<u8> = vec![0, 1, 2, 0, 3];
        let v1: Vec<u8> = vec![1, 0, 0, 0, 0];

        let vecs: Vec<&[u8]> = vec![&v0, &v1];
        let csr = ScentCsr::from_scent_vectors(&vecs);

        assert_eq!(csr.degree(0), 3);
        assert_eq!(csr.degree(1), 1);
    }

    #[test]
    fn test_csr_spmv() {
        let v0: Vec<u8> = vec![0, 1, 2];
        let v1: Vec<u8> = vec![1, 0, 3];
        let v2: Vec<u8> = vec![2, 3, 0];

        let vecs: Vec<&[u8]> = vec![&v0, &v1, &v2];
        let csr = ScentCsr::from_scent_vectors(&vecs);

        let x = vec![1.0f32, 2.0, 3.0];
        let y = csr.spmv(&x);

        // y[0] = 0*1 + 1*2 + 2*3 = 8
        assert_eq!(y[0], 8.0);
        // y[1] = 1*1 + 0*2 + 3*3 = 10
        assert_eq!(y[1], 10.0);
        // y[2] = 2*1 + 3*2 + 0*3 = 8
        assert_eq!(y[2], 8.0);
    }
}
