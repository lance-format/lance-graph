// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Vector Search API for lance-graph
//!
//! This module provides a flexible API for vector similarity search that can work with:
//! - In-memory RecordBatches (brute-force search)
//! - Lance datasets (ANN search with indices)
//!
//! This is distinct from the UDF-based vector search (`vector_distance()`, `vector_similarity()`)
//! which is integrated into Cypher queries. This API provides explicit two-step search for
//! GraphRAG workflows where you want to:
//! 1. Use Cypher for graph traversal and filtering
//! 2. Use VectorSearch for similarity ranking with Lance ANN indices
//!
//! # Example
//!
//! ```ignore
//! use lance_graph::lance_vector_search::VectorSearch;
//! use lance_graph::ast::DistanceMetric;
//!
//! // Step 1: Run Cypher query to get candidates
//! let candidates = query.execute(datasets, None).await?;
//!
//! // Step 2: Rerank by vector similarity
//! let results = VectorSearch::new("embedding")
//!     .query_vector(vec![0.1, 0.2, 0.3])
//!     .metric(DistanceMetric::Cosine)
//!     .top_k(10)
//!     .search(&candidates)
//!     .await?;
//! ```

use crate::ast::DistanceMetric;
use crate::datafusion_planner::vector_ops;
use crate::error::{GraphError, Result};
use arrow::array::{Array, ArrayRef, Float32Array, UInt32Array};
use arrow::compute::take;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

/// Builder for vector similarity search operations
///
/// Supports both brute-force search on RecordBatches and ANN search on Lance datasets.
#[derive(Debug, Clone)]
pub struct VectorSearch {
    /// Name of the vector column to search
    column: String,
    /// Query vector for similarity computation
    query_vector: Option<Vec<f32>>,
    /// Distance metric (L2, Cosine, Dot)
    metric: DistanceMetric,
    /// Number of results to return
    top_k: usize,
    /// Whether to include distance/similarity scores in output
    include_distance: bool,
    /// Name for the distance column (default: "_distance")
    distance_column_name: String,
}

impl VectorSearch {
    /// Create a new VectorSearch builder for the specified column
    ///
    /// # Arguments
    /// * `column` - Name of the vector column in the data
    ///
    /// # Example
    /// ```ignore
    /// let search = VectorSearch::new("embedding");
    /// ```
    pub fn new(column: &str) -> Self {
        Self {
            column: column.to_string(),
            query_vector: None,
            metric: DistanceMetric::L2,
            top_k: 10,
            include_distance: true,
            distance_column_name: "_distance".to_string(),
        }
    }

    /// Set the query vector for similarity search
    ///
    /// # Arguments
    /// * `vec` - The query vector (must match dimension of data vectors)
    pub fn query_vector(mut self, vec: Vec<f32>) -> Self {
        self.query_vector = Some(vec);
        self
    }

    /// Set the distance metric
    ///
    /// # Arguments
    /// * `metric` - Distance metric (L2, Cosine, or Dot)
    pub fn metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set the number of results to return
    ///
    /// # Arguments
    /// * `k` - Maximum number of results
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Whether to include distance scores in the output
    ///
    /// # Arguments
    /// * `include` - If true, adds a distance column to results
    pub fn include_distance(mut self, include: bool) -> Self {
        self.include_distance = include;
        self
    }

    /// Set the name for the distance column
    ///
    /// # Arguments
    /// * `name` - Column name for distance values (default: "_distance")
    pub fn distance_column_name(mut self, name: &str) -> Self {
        self.distance_column_name = name.to_string();
        self
    }

    /// Perform brute-force vector search on a RecordBatch
    ///
    /// This method computes distances for all vectors in the batch and returns
    /// the top-k results sorted by distance (ascending).
    ///
    /// # Arguments
    /// * `data` - RecordBatch containing the vector column
    ///
    /// # Returns
    /// A new RecordBatch with the top-k rows, optionally including a distance column
    ///
    /// # Example
    /// ```ignore
    /// let results = VectorSearch::new("embedding")
    ///     .query_vector(query_vec)
    ///     .metric(DistanceMetric::Cosine)
    ///     .top_k(10)
    ///     .search(&candidates)
    ///     .await?;
    /// ```
    pub async fn search(&self, data: &RecordBatch) -> Result<RecordBatch> {
        let query_vector = self
            .query_vector
            .as_ref()
            .ok_or_else(|| GraphError::ConfigError {
                message: "Query vector is required for search".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        // Find the vector column
        let schema = data.schema();
        let column_idx = schema
            .index_of(&self.column)
            .map_err(|_| GraphError::ConfigError {
                message: format!("Vector column '{}' not found in data", self.column),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        let vector_column = data.column(column_idx);

        // Extract vectors and compute distances
        let vectors = vector_ops::extract_vectors(vector_column)?;
        let distances = vector_ops::compute_vector_distances(&vectors, query_vector, &self.metric);

        // Get top-k indices (sorted by distance ascending)
        let top_k_indices = self.get_top_k_indices(&distances);

        // Build result batch
        self.build_result_batch(data, &top_k_indices, &distances)
    }

    /// Perform ANN vector search on a Lance dataset
    ///
    /// This method uses Lance's native ANN search via `scan().nearest()`,
    /// which leverages vector indices (IVF_PQ, IVF_HNSW, etc.) when available.
    ///
    /// # Arguments
    /// * `dataset` - Lance dataset with vector column
    ///
    /// # Returns
    /// A RecordBatch with the top-k nearest neighbors
    ///
    /// # Example
    /// ```ignore
    /// let dataset = lance::Dataset::open("data.lance").await?;
    /// let results = VectorSearch::new("embedding")
    ///     .query_vector(query_vec)
    ///     .metric(DistanceMetric::L2)
    ///     .top_k(10)
    ///     .search_lance(&dataset)
    ///     .await?;
    /// ```
    pub async fn search_lance(&self, dataset: &lance::Dataset) -> Result<RecordBatch> {
        use arrow::compute::concat_batches;
        use futures::TryStreamExt;

        let query_vector = self
            .query_vector
            .as_ref()
            .ok_or_else(|| GraphError::ConfigError {
                message: "Query vector is required for search".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        // Convert metric to Lance's DistanceType
        let lance_metric = match self.metric {
            DistanceMetric::L2 => lance_linalg::distance::DistanceType::L2,
            DistanceMetric::Cosine => lance_linalg::distance::DistanceType::Cosine,
            DistanceMetric::Dot => lance_linalg::distance::DistanceType::Dot,
        };

        // Create query array
        let query_array = Float32Array::from(query_vector.clone());

        // Build scanner with ANN search
        let mut scanner = dataset.scan();
        scanner
            .nearest(&self.column, &query_array as &dyn Array, self.top_k)
            .map_err(|e| GraphError::ExecutionError {
                message: format!("Failed to configure nearest neighbor search: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?
            .distance_metric(lance_metric);

        // Execute scan and collect results
        let stream = scanner
            .try_into_stream()
            .await
            .map_err(|e| GraphError::ExecutionError {
                message: format!("Failed to create scan stream: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        let batches: Vec<RecordBatch> =
            stream
                .try_collect()
                .await
                .map_err(|e| GraphError::ExecutionError {
                    message: format!("Failed to collect scan results: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

        if batches.is_empty() {
            // Return empty batch with dataset schema
            let lance_schema = dataset.schema();
            let arrow_schema: Schema = lance_schema.into();
            return Ok(RecordBatch::new_empty(Arc::new(arrow_schema)));
        }

        // Concatenate batches
        let schema = batches[0].schema();
        concat_batches(&schema, &batches).map_err(|e| GraphError::ExecutionError {
            message: format!("Failed to concatenate result batches: {}", e),
            location: snafu::Location::new(file!(), line!(), column!()),
        })
    }

    /// Get indices of top-k smallest distances
    fn get_top_k_indices(&self, distances: &[f32]) -> Vec<u32> {
        // Create (index, distance) pairs
        let mut indexed: Vec<(usize, f32)> = distances.iter().cloned().enumerate().collect();

        // Sort by distance (ascending)
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-k indices
        indexed
            .into_iter()
            .take(self.top_k)
            .map(|(idx, _)| idx as u32)
            .collect()
    }

    /// Build result RecordBatch from original data and top-k indices
    fn build_result_batch(
        &self,
        data: &RecordBatch,
        indices: &[u32],
        distances: &[f32],
    ) -> Result<RecordBatch> {
        let indices_array = UInt32Array::from(indices.to_vec());

        // Take rows from each column
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(data.num_columns() + 1);
        let mut fields: Vec<Field> = Vec::with_capacity(data.num_columns() + 1);

        for (i, field) in data.schema().fields().iter().enumerate() {
            let column = data.column(i);
            let taken = take(column.as_ref(), &indices_array, None).map_err(|e| {
                GraphError::ExecutionError {
                    message: format!("Failed to select rows: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                }
            })?;
            columns.push(taken);
            fields.push(field.as_ref().clone());
        }

        // Add distance column if requested
        if self.include_distance {
            let selected_distances: Vec<f32> =
                indices.iter().map(|&i| distances[i as usize]).collect();
            let distance_array = Arc::new(Float32Array::from(selected_distances)) as ArrayRef;
            columns.push(distance_array);
            fields.push(Field::new(
                &self.distance_column_name,
                DataType::Float32,
                false,
            ));
        }

        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, columns).map_err(|e| GraphError::ExecutionError {
            message: format!("Failed to create result batch: {}", e),
            location: snafu::Location::new(file!(), line!(), column!()),
        })
    }
}

/// Result of a vector search operation with metadata
#[derive(Debug)]
pub struct VectorSearchResult {
    /// The result data
    pub data: RecordBatch,
    /// Whether ANN index was used (vs brute-force)
    pub used_ann_index: bool,
    /// Number of vectors scanned (for brute-force)
    pub vectors_scanned: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{FixedSizeListArray, Int64Array, StringArray};
    use arrow::datatypes::FieldRef;

    fn create_test_batch() -> RecordBatch {
        // Create schema with 3D embeddings
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 3),
                false,
            ),
        ]));

        // Create test data with embeddings
        let embedding_data = vec![
            1.0, 0.0, 0.0, // Alice - closest to [1,0,0]
            0.9, 0.1, 0.0, // Bob - second closest
            0.0, 1.0, 0.0, // Carol - orthogonal
            0.0, 0.0, 1.0, // David - orthogonal
            0.5, 0.5, 0.0, // Eve - medium distance
        ];

        let field = Arc::new(Field::new("item", DataType::Float32, true)) as FieldRef;
        let values = Arc::new(Float32Array::from(embedding_data));
        let embeddings = FixedSizeListArray::try_new(field, 3, values, None).unwrap();

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(StringArray::from(vec![
                    "Alice", "Bob", "Carol", "David", "Eve",
                ])),
                Arc::new(embeddings),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn test_vector_search_basic() {
        let batch = create_test_batch();

        let results = VectorSearch::new("embedding")
            .query_vector(vec![1.0, 0.0, 0.0])
            .metric(DistanceMetric::L2)
            .top_k(3)
            .search(&batch)
            .await
            .unwrap();

        assert_eq!(results.num_rows(), 3);

        // Check that Alice is first (closest to [1,0,0])
        let names = results
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(names.value(0), "Alice");
        assert_eq!(names.value(1), "Bob");
    }

    #[tokio::test]
    async fn test_vector_search_cosine() {
        let batch = create_test_batch();

        let results = VectorSearch::new("embedding")
            .query_vector(vec![1.0, 0.0, 0.0])
            .metric(DistanceMetric::Cosine)
            .top_k(2)
            .search(&batch)
            .await
            .unwrap();

        assert_eq!(results.num_rows(), 2);

        let names = results
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(names.value(0), "Alice");
    }

    #[tokio::test]
    async fn test_vector_search_with_distance() {
        let batch = create_test_batch();

        let results = VectorSearch::new("embedding")
            .query_vector(vec![1.0, 0.0, 0.0])
            .metric(DistanceMetric::L2)
            .top_k(2)
            .include_distance(true)
            .search(&batch)
            .await
            .unwrap();

        // Should have original columns + distance column
        assert_eq!(results.num_columns(), 4);

        // Check distance column exists and has correct name
        let schema = results.schema();
        assert!(schema.field_with_name("_distance").is_ok());

        // First result should have distance 0 (identical vector)
        let distances = results
            .column(3)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert_eq!(distances.value(0), 0.0);
    }

    #[tokio::test]
    async fn test_vector_search_without_distance() {
        let batch = create_test_batch();

        let results = VectorSearch::new("embedding")
            .query_vector(vec![1.0, 0.0, 0.0])
            .metric(DistanceMetric::L2)
            .top_k(2)
            .include_distance(false)
            .search(&batch)
            .await
            .unwrap();

        // Should have only original columns
        assert_eq!(results.num_columns(), 3);
    }

    #[tokio::test]
    async fn test_vector_search_custom_distance_column() {
        let batch = create_test_batch();

        let results = VectorSearch::new("embedding")
            .query_vector(vec![1.0, 0.0, 0.0])
            .metric(DistanceMetric::L2)
            .top_k(2)
            .distance_column_name("similarity_score")
            .search(&batch)
            .await
            .unwrap();

        let schema = results.schema();
        assert!(schema.field_with_name("similarity_score").is_ok());
    }

    #[tokio::test]
    async fn test_vector_search_missing_query() {
        let batch = create_test_batch();

        let result = VectorSearch::new("embedding")
            .metric(DistanceMetric::L2)
            .top_k(2)
            .search(&batch)
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_vector_search_missing_column() {
        let batch = create_test_batch();

        let result = VectorSearch::new("nonexistent")
            .query_vector(vec![1.0, 0.0, 0.0])
            .top_k(2)
            .search(&batch)
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_vector_search_top_k_larger_than_data() {
        let batch = create_test_batch();

        let results = VectorSearch::new("embedding")
            .query_vector(vec![1.0, 0.0, 0.0])
            .metric(DistanceMetric::L2)
            .top_k(100) // More than 5 rows
            .search(&batch)
            .await
            .unwrap();

        // Should return all 5 rows
        assert_eq!(results.num_rows(), 5);
    }
}
