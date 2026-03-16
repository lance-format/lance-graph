// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vector Operations
//!
//! Helpers for vector similarity search and distance computation

use crate::ast::DistanceMetric;
use crate::error::{GraphError, Result};
use arrow::array::{Array, ArrayRef, FixedSizeListArray, Float32Array, ListArray};

/// Extract vectors from Arrow ListArray or FixedSizeListArray
///
/// Accepts both types for user convenience:
/// - FixedSizeListArray: from Lance datasets or explicit construction
/// - ListArray: from natural table construction with nested lists
pub fn extract_vectors(array: &ArrayRef) -> Result<Vec<Vec<f32>>> {
    // Try FixedSizeListArray first (more common in Lance)
    if let Some(list_array) = array.as_any().downcast_ref::<FixedSizeListArray>() {
        let mut vectors = Vec::with_capacity(list_array.len());
        for i in 0..list_array.len() {
            if list_array.is_null(i) {
                return Err(GraphError::ExecutionError {
                    message: "Null vector in FixedSizeListArray".to_string(),
                    location: snafu::Location::new(file!(), line!(), column!()),
                });
            }
            let value_array = list_array.value(i);
            let float_array = value_array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| GraphError::ExecutionError {
                    message: "Expected Float32Array in vector".to_string(),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

            let vec: Vec<f32> = (0..float_array.len())
                .map(|j| float_array.value(j))
                .collect();
            vectors.push(vec);
        }
        return Ok(vectors);
    }

    // Try ListArray (from nested list construction)
    if let Some(list_array) = array.as_any().downcast_ref::<ListArray>() {
        let mut vectors = Vec::with_capacity(list_array.len());
        for i in 0..list_array.len() {
            if list_array.is_null(i) {
                return Err(GraphError::ExecutionError {
                    message: "Null vector in ListArray".to_string(),
                    location: snafu::Location::new(file!(), line!(), column!()),
                });
            }
            let value_array = list_array.value(i);
            let float_array = value_array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| GraphError::ExecutionError {
                    message: "Expected Float32Array in vector".to_string(),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

            let vec: Vec<f32> = (0..float_array.len())
                .map(|j| float_array.value(j))
                .collect();
            vectors.push(vec);
        }
        return Ok(vectors);
    }

    Err(GraphError::ExecutionError {
        message: "Expected ListArray or FixedSizeListArray for vector column".to_string(),
        location: snafu::Location::new(file!(), line!(), column!()),
    })
}

/// Extract a single vector from a ScalarValue
/// This avoids allocating a full array when we just need one vector
pub fn extract_single_vector_from_scalar(
    scalar: &datafusion::scalar::ScalarValue,
) -> Result<Vec<f32>> {
    // Convert scalar to a single-element array, then extract
    let array = scalar.to_array().map_err(|e| GraphError::ExecutionError {
        message: format!("Failed to convert scalar to array: {}", e),
        location: snafu::Location::new(file!(), line!(), column!()),
    })?;

    let list_array = array
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| GraphError::ExecutionError {
            message: "Expected FixedSizeListArray for vector scalar".to_string(),
            location: snafu::Location::new(file!(), line!(), column!()),
        })?;

    if list_array.is_empty() {
        return Err(GraphError::ExecutionError {
            message: "Empty vector array".to_string(),
            location: snafu::Location::new(file!(), line!(), column!()),
        });
    }

    let value_array = list_array.value(0);
    let float_array = value_array
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| GraphError::ExecutionError {
            message: "Expected Float32Array in vector".to_string(),
            location: snafu::Location::new(file!(), line!(), column!()),
        })?;

    Ok((0..float_array.len())
        .map(|j| float_array.value(j))
        .collect())
}

/// Compute L2 (Euclidean) distance between two vectors
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        // Dimension mismatch - return max distance
        return f32::MAX;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Compute cosine distance (1 - cosine_similarity) between two vectors
/// Returns a value in [0, 2] where 0 means identical and 2 means opposite
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        // Dimension mismatch - return max distance
        return 2.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 2.0; // Maximum distance for zero vectors
    }

    let similarity = dot / (norm_a * norm_b);
    1.0 - similarity
}

/// Compute cosine similarity (for vector_similarity function)
/// Returns a value in [-1, 1] where 1 means identical and -1 means opposite
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        // Dimension mismatch - return minimum similarity
        return -1.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return -1.0; // Minimum similarity for zero vectors
    }

    dot / (norm_a * norm_b)
}

/// Compute dot product between two vectors
/// For similarity search, we return the negative (so lower is better for sorting)
pub fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        // Dimension mismatch - return worst distance to exclude from results
        return f32::MAX;
    }

    -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

/// Compute dot product similarity (for vector_similarity function)
pub fn dot_product_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        // Dimension mismatch - return worst similarity to exclude from results
        return f32::MIN;
    }

    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

/// Compute vector distance for an array of vectors against a single query vector
pub fn compute_vector_distances(
    vectors: &[Vec<f32>],
    query_vector: &[f32],
    metric: &DistanceMetric,
) -> Vec<f32> {
    vectors
        .iter()
        .map(|v| match metric {
            DistanceMetric::L2 => l2_distance(v, query_vector),
            DistanceMetric::Cosine => cosine_distance(v, query_vector),
            DistanceMetric::Dot => dot_product_distance(v, query_vector),
        })
        .collect()
}

/// Compute vector similarities for an array of vectors against a single query vector
pub fn compute_vector_similarities(
    vectors: &[Vec<f32>],
    query_vector: &[f32],
    metric: &DistanceMetric,
) -> Vec<f32> {
    vectors
        .iter()
        .map(|v| match metric {
            DistanceMetric::L2 => {
                // For L2, convert distance to similarity (inverse)
                let dist = l2_distance(v, query_vector);
                if dist == 0.0 {
                    1.0 // Perfect match
                } else {
                    1.0 / (1.0 + dist) // Similarity decreases as distance increases
                }
            }
            DistanceMetric::Cosine => cosine_similarity(v, query_vector),
            DistanceMetric::Dot => dot_product_similarity(v, query_vector),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_l2_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let dist = l2_distance(&a, &b);
        assert!((dist - 1.414).abs() < 0.01); // sqrt(2)
    }

    #[test]
    fn test_l2_distance_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let dist = l2_distance(&a, &b);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let dist = cosine_distance(&a, &b);
        assert_eq!(dist, 0.0); // Identical vectors
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let dist = cosine_distance(&a, &b);
        assert_eq!(dist, 1.0); // Orthogonal vectors
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 1.0); // Identical
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let sim = dot_product_similarity(&a, &b);
        assert_eq!(sim, 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];

        let dist = l2_distance(&a, &b);
        assert_eq!(dist, f32::MAX);

        let dist = cosine_distance(&a, &b);
        assert_eq!(dist, 2.0);

        let dist = dot_product_distance(&a, &b);
        assert_eq!(dist, f32::MAX);

        let sim = dot_product_similarity(&a, &b);
        assert_eq!(sim, f32::MIN);
    }

    #[test]
    fn test_extract_single_vector_from_scalar() {
        use arrow::array::FixedSizeListArray;
        use arrow::datatypes::{DataType, Field};
        use datafusion::scalar::ScalarValue;

        // Create a FixedSizeList scalar value with a 3D vector [1.0, 2.0, 3.0]
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let values = Arc::new(Float32Array::from(vec![1.0, 2.0, 3.0]));
        let list_array = FixedSizeListArray::try_new(field.clone(), 3, values, None).unwrap();

        // Create a scalar from the first element
        let scalar = ScalarValue::try_from_array(&list_array, 0).unwrap();

        // Extract the vector
        let result = extract_single_vector_from_scalar(&scalar);
        assert!(result.is_ok());

        let vec = result.unwrap();
        assert_eq!(vec.len(), 3);
        assert_eq!(vec[0], 1.0);
        assert_eq!(vec[1], 2.0);
        assert_eq!(vec[2], 3.0);
    }

    #[test]
    fn test_extract_single_vector_from_scalar_different_dimensions() {
        use arrow::array::FixedSizeListArray;
        use arrow::datatypes::{DataType, Field};
        use datafusion::scalar::ScalarValue;

        // Create a 5D vector [0.1, 0.2, 0.3, 0.4, 0.5]
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let values = Arc::new(Float32Array::from(vec![0.1, 0.2, 0.3, 0.4, 0.5]));
        let list_array = FixedSizeListArray::try_new(field.clone(), 5, values, None).unwrap();

        let scalar = ScalarValue::try_from_array(&list_array, 0).unwrap();
        let result = extract_single_vector_from_scalar(&scalar);
        assert!(result.is_ok());

        let vec = result.unwrap();
        assert_eq!(vec.len(), 5);
        assert!((vec[0] - 0.1).abs() < 0.001);
        assert!((vec[4] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_compute_vector_distances_broadcast() {
        // Test that compute_vector_distances properly broadcasts a single query vector
        let data_vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let query_vector = vec![1.0, 0.0, 0.0];

        let distances = compute_vector_distances(&data_vectors, &query_vector, &DistanceMetric::L2);

        assert_eq!(distances.len(), 3);
        assert_eq!(distances[0], 0.0); // Same as query
        assert!((distances[1] - 1.414).abs() < 0.01); // Orthogonal
        assert!((distances[2] - 1.414).abs() < 0.01); // Orthogonal
    }

    #[test]
    fn test_compute_vector_similarities_broadcast() {
        // Test that compute_vector_similarities properly broadcasts a single query vector
        let data_vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.5, 0.5, 0.0], // 45 degrees from x-axis
        ];
        let query_vector = vec![1.0, 0.0, 0.0];

        let similarities =
            compute_vector_similarities(&data_vectors, &query_vector, &DistanceMetric::Cosine);

        assert_eq!(similarities.len(), 3);
        assert_eq!(similarities[0], 1.0); // Same as query
        assert_eq!(similarities[1], 0.0); // Orthogonal
        assert!((similarities[2] - 0.707).abs() < 0.01); // cos(45°) ≈ 0.707
    }

    #[test]
    fn test_extract_vectors_from_fixed_size_list() {
        use arrow::datatypes::{DataType, Field};

        // Create FixedSizeListArray with 3D vectors
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let values = Arc::new(Float32Array::from(vec![
            1.0, 0.0, 0.0, // Vector 1
            0.0, 1.0, 0.0, // Vector 2
            0.0, 0.0, 1.0, // Vector 3
        ]));
        let list_array = FixedSizeListArray::try_new(field, 3, values, None).unwrap();
        let array_ref: ArrayRef = Arc::new(list_array);

        let result = extract_vectors(&array_ref);
        assert!(result.is_ok());

        let vectors = result.unwrap();
        assert_eq!(vectors.len(), 3);
        assert_eq!(vectors[0], vec![1.0, 0.0, 0.0]);
        assert_eq!(vectors[1], vec![0.0, 1.0, 0.0]);
        assert_eq!(vectors[2], vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_extract_vectors_from_list_array() {
        use arrow::array::ListBuilder;

        // Create ListArray with variable-length vectors (though we use same length)
        let values_builder = Float32Array::builder(9);
        let mut list_builder = ListBuilder::new(values_builder);

        // Add first vector [1.0, 0.0, 0.0]
        list_builder.values().append_value(1.0);
        list_builder.values().append_value(0.0);
        list_builder.values().append_value(0.0);
        list_builder.append(true);

        // Add second vector [0.0, 1.0, 0.0]
        list_builder.values().append_value(0.0);
        list_builder.values().append_value(1.0);
        list_builder.values().append_value(0.0);
        list_builder.append(true);

        // Add third vector [0.5, 0.5, 0.0]
        list_builder.values().append_value(0.5);
        list_builder.values().append_value(0.5);
        list_builder.values().append_value(0.0);
        list_builder.append(true);

        let list_array = list_builder.finish();
        let array_ref: ArrayRef = Arc::new(list_array);

        let result = extract_vectors(&array_ref);
        assert!(result.is_ok());

        let vectors = result.unwrap();
        assert_eq!(vectors.len(), 3);
        assert_eq!(vectors[0], vec![1.0, 0.0, 0.0]);
        assert_eq!(vectors[1], vec![0.0, 1.0, 0.0]);
        assert_eq!(vectors[2], vec![0.5, 0.5, 0.0]);
    }

    #[test]
    fn test_extract_vectors_rejects_invalid_type() {
        // Test that extract_vectors rejects non-list arrays
        let float_array = Float32Array::from(vec![1.0, 2.0, 3.0]);
        let array_ref: ArrayRef = Arc::new(float_array);

        let result = extract_vectors(&array_ref);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Expected ListArray or FixedSizeListArray"));
    }

    #[test]
    fn test_extract_vectors_rejects_null_in_fixed_size_list() {
        use arrow::datatypes::{DataType, Field};

        // Create FixedSizeListArray with a null vector
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let values = Arc::new(Float32Array::from(vec![
            1.0, 0.0, 0.0, // Vector 1
            0.0, 1.0, 0.0, // Vector 2 (will be null)
            0.0, 0.0, 1.0, // Vector 3
        ]));
        let null_buffer = arrow::buffer::NullBuffer::from(vec![true, false, true]);
        let list_array = FixedSizeListArray::try_new(field, 3, values, Some(null_buffer)).unwrap();
        let array_ref: ArrayRef = Arc::new(list_array);

        let result = extract_vectors(&array_ref);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Null vector in FixedSizeListArray"));
    }
}
