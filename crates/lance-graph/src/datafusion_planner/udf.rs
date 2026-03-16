// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! User-Defined Functions (UDFs) for DataFusion
//!
//! This module contains UDF implementations for vector operations used in graph queries.

use crate::ast::DistanceMetric;
use crate::datafusion_planner::vector_ops;
use arrow::array::ArrayRef;
use arrow::datatypes::DataType;
use datafusion::logical_expr::{ScalarUDF, Signature, Volatility};
use datafusion::physical_plan::ColumnarValue;
use std::sync::{Arc, LazyLock};

/// Type alias for UDF function closures
type UdfFunc =
    Arc<dyn Fn(&[ColumnarValue]) -> datafusion::error::Result<ColumnarValue> + Send + Sync>;

/// Vector distance computation function (used by UDF)
///
/// This function handles four cases efficiently:
/// 1. Array vs Array: Pairwise distance computation
/// 2. Array vs Scalar: Broadcast single query vector against all data vectors (memory efficient)
/// 3. Scalar vs Array: Broadcast single query vector against all data vectors (memory efficient)
/// 4. Scalar vs Scalar: Single distance computation
fn vector_distance_func(
    args: &[ColumnarValue],
    metric: &DistanceMetric,
) -> datafusion::error::Result<ColumnarValue> {
    if args.len() != 2 {
        return Err(datafusion::error::DataFusionError::Execution(
            "vector_distance requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        // Case 1: Both are arrays - pairwise or broadcast based on lengths
        (ColumnarValue::Array(left_arr), ColumnarValue::Array(right_arr)) => {
            let left_vectors = vector_ops::extract_vectors(left_arr)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
            let right_vectors = vector_ops::extract_vectors(right_arr)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;

            let distances: Vec<f32> = if right_vectors.len() == 1 {
                // Broadcast right against all left vectors
                vector_ops::compute_vector_distances(&left_vectors, &right_vectors[0], metric)
            } else if left_vectors.len() == 1 {
                // Broadcast left against all right vectors
                vector_ops::compute_vector_distances(&right_vectors, &left_vectors[0], metric)
            } else if left_vectors.len() == right_vectors.len() {
                // Pairwise distance computation
                left_vectors
                    .iter()
                    .zip(right_vectors.iter())
                    .map(|(l, r)| match metric {
                        DistanceMetric::L2 => vector_ops::l2_distance(l, r),
                        DistanceMetric::Cosine => vector_ops::cosine_distance(l, r),
                        DistanceMetric::Dot => vector_ops::dot_product_distance(l, r),
                        DistanceMetric::Hamming => f32::MAX,
                    })
                    .collect()
            } else {
                return Err(datafusion::error::DataFusionError::Execution(format!(
                    "Vector count mismatch: left has {} vectors, right has {}",
                    left_vectors.len(),
                    right_vectors.len()
                )));
            };

            let result = Arc::new(arrow::array::Float32Array::from(distances)) as ArrayRef;
            Ok(ColumnarValue::Array(result))
        }

        // Case 2: Left is array, right is scalar - broadcast scalar against all left vectors
        // This is the common case for similarity search: comparing many vectors to one query
        (ColumnarValue::Array(left_arr), ColumnarValue::Scalar(right_scalar)) => {
            let left_vectors = vector_ops::extract_vectors(left_arr)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;

            // Extract single query vector from scalar WITHOUT allocating a full array
            let query_vector = vector_ops::extract_single_vector_from_scalar(right_scalar)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;

            let distances =
                vector_ops::compute_vector_distances(&left_vectors, &query_vector, metric);
            let result = Arc::new(arrow::array::Float32Array::from(distances)) as ArrayRef;
            Ok(ColumnarValue::Array(result))
        }

        // Case 3: Left is scalar, right is array - broadcast scalar against all right vectors
        (ColumnarValue::Scalar(left_scalar), ColumnarValue::Array(right_arr)) => {
            let right_vectors = vector_ops::extract_vectors(right_arr)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;

            // Extract single query vector from scalar WITHOUT allocating a full array
            let query_vector = vector_ops::extract_single_vector_from_scalar(left_scalar)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;

            let distances =
                vector_ops::compute_vector_distances(&right_vectors, &query_vector, metric);
            let result = Arc::new(arrow::array::Float32Array::from(distances)) as ArrayRef;
            Ok(ColumnarValue::Array(result))
        }

        // Case 4: Both are scalars - single distance computation
        (ColumnarValue::Scalar(left_scalar), ColumnarValue::Scalar(right_scalar)) => {
            let left_vec = vector_ops::extract_single_vector_from_scalar(left_scalar)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
            let right_vec = vector_ops::extract_single_vector_from_scalar(right_scalar)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;

            let distance = match metric {
                DistanceMetric::L2 => vector_ops::l2_distance(&left_vec, &right_vec),
                DistanceMetric::Cosine => vector_ops::cosine_distance(&left_vec, &right_vec),
                DistanceMetric::Dot => vector_ops::dot_product_distance(&left_vec, &right_vec),
                DistanceMetric::Hamming => f32::MAX,
            };

            // Return as scalar since both inputs were scalars
            Ok(ColumnarValue::Scalar(
                datafusion::scalar::ScalarValue::Float32(Some(distance)),
            ))
        }
    }
}

/// Cached vector distance UDFs (one per metric type)
/// These are initialized once and reused across all queries for better performance
static VECTOR_DISTANCE_L2_UDF: LazyLock<Arc<ScalarUDF>> = LazyLock::new(|| {
    let func = move |args: &[ColumnarValue]| -> datafusion::error::Result<ColumnarValue> {
        vector_distance_func(args, &DistanceMetric::L2)
    };

    Arc::new(ScalarUDF::new_from_impl(VectorDistanceUDF {
        name: "vector_distance_l2".to_string(),
        func: Arc::new(func),
        metric: DistanceMetric::L2,
        signature: Signature::any(2, Volatility::Immutable),
    }))
});

static VECTOR_DISTANCE_COSINE_UDF: LazyLock<Arc<ScalarUDF>> = LazyLock::new(|| {
    let func = move |args: &[ColumnarValue]| -> datafusion::error::Result<ColumnarValue> {
        vector_distance_func(args, &DistanceMetric::Cosine)
    };

    Arc::new(ScalarUDF::new_from_impl(VectorDistanceUDF {
        name: "vector_distance_cosine".to_string(),
        func: Arc::new(func),
        metric: DistanceMetric::Cosine,
        signature: Signature::any(2, Volatility::Immutable),
    }))
});

static VECTOR_DISTANCE_DOT_UDF: LazyLock<Arc<ScalarUDF>> = LazyLock::new(|| {
    let func = move |args: &[ColumnarValue]| -> datafusion::error::Result<ColumnarValue> {
        vector_distance_func(args, &DistanceMetric::Dot)
    };

    Arc::new(ScalarUDF::new_from_impl(VectorDistanceUDF {
        name: "vector_distance_dot".to_string(),
        func: Arc::new(func),
        metric: DistanceMetric::Dot,
        signature: Signature::any(2, Volatility::Immutable),
    }))
});

/// Get a cached vector distance UDF for the given distance metric
pub(crate) fn create_vector_distance_udf(metric: &DistanceMetric) -> Arc<ScalarUDF> {
    match metric {
        DistanceMetric::L2 => VECTOR_DISTANCE_L2_UDF.clone(),
        DistanceMetric::Cosine => VECTOR_DISTANCE_COSINE_UDF.clone(),
        DistanceMetric::Dot => VECTOR_DISTANCE_DOT_UDF.clone(),
        DistanceMetric::Hamming => HAMMING_DISTANCE_UDF.clone(),
    }
}

/// Vector similarity computation function (used by UDF)
///
/// This function handles four cases efficiently:
/// 1. Array vs Array: Pairwise similarity computation
/// 2. Array vs Scalar: Broadcast single query vector against all data vectors (memory efficient)
/// 3. Scalar vs Array: Broadcast single query vector against all data vectors (memory efficient)
/// 4. Scalar vs Scalar: Single similarity computation
fn vector_similarity_func(
    args: &[ColumnarValue],
    metric: &DistanceMetric,
) -> datafusion::error::Result<ColumnarValue> {
    if args.len() != 2 {
        return Err(datafusion::error::DataFusionError::Execution(
            "vector_similarity requires exactly 2 arguments".to_string(),
        ));
    }

    // Helper function to compute single similarity value
    let compute_single_similarity = |l: &[f32], r: &[f32]| match metric {
        DistanceMetric::L2 => {
            let dist = vector_ops::l2_distance(l, r);
            if dist == 0.0 {
                1.0
            } else {
                1.0 / (1.0 + dist)
            }
        }
        DistanceMetric::Cosine => vector_ops::cosine_similarity(l, r),
        DistanceMetric::Dot => vector_ops::dot_product_similarity(l, r),
        DistanceMetric::Hamming => 0.0, // Use hamming_similarity UDF for binary vectors
    };

    match (&args[0], &args[1]) {
        // Case 1: Both are arrays - pairwise or broadcast based on lengths
        (ColumnarValue::Array(left_arr), ColumnarValue::Array(right_arr)) => {
            let left_vectors = vector_ops::extract_vectors(left_arr)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
            let right_vectors = vector_ops::extract_vectors(right_arr)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;

            let similarities: Vec<f32> = if right_vectors.len() == 1 {
                // Broadcast right against all left vectors
                vector_ops::compute_vector_similarities(&left_vectors, &right_vectors[0], metric)
            } else if left_vectors.len() == 1 {
                // Broadcast left against all right vectors
                vector_ops::compute_vector_similarities(&right_vectors, &left_vectors[0], metric)
            } else if left_vectors.len() == right_vectors.len() {
                // Pairwise similarity computation
                left_vectors
                    .iter()
                    .zip(right_vectors.iter())
                    .map(|(l, r)| compute_single_similarity(l, r))
                    .collect()
            } else {
                return Err(datafusion::error::DataFusionError::Execution(format!(
                    "Vector count mismatch: left has {} vectors, right has {}",
                    left_vectors.len(),
                    right_vectors.len()
                )));
            };

            let result = Arc::new(arrow::array::Float32Array::from(similarities)) as ArrayRef;
            Ok(ColumnarValue::Array(result))
        }

        // Case 2: Left is array, right is scalar - broadcast scalar against all left vectors
        // This is the common case for similarity search: comparing many vectors to one query
        (ColumnarValue::Array(left_arr), ColumnarValue::Scalar(right_scalar)) => {
            let left_vectors = vector_ops::extract_vectors(left_arr)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;

            // Extract single query vector from scalar WITHOUT allocating a full array
            let query_vector = vector_ops::extract_single_vector_from_scalar(right_scalar)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;

            let similarities =
                vector_ops::compute_vector_similarities(&left_vectors, &query_vector, metric);
            let result = Arc::new(arrow::array::Float32Array::from(similarities)) as ArrayRef;
            Ok(ColumnarValue::Array(result))
        }

        // Case 3: Left is scalar, right is array - broadcast scalar against all right vectors
        (ColumnarValue::Scalar(left_scalar), ColumnarValue::Array(right_arr)) => {
            let right_vectors = vector_ops::extract_vectors(right_arr)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;

            // Extract single query vector from scalar WITHOUT allocating a full array
            let query_vector = vector_ops::extract_single_vector_from_scalar(left_scalar)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;

            let similarities =
                vector_ops::compute_vector_similarities(&right_vectors, &query_vector, metric);
            let result = Arc::new(arrow::array::Float32Array::from(similarities)) as ArrayRef;
            Ok(ColumnarValue::Array(result))
        }

        // Case 4: Both are scalars - single similarity computation
        (ColumnarValue::Scalar(left_scalar), ColumnarValue::Scalar(right_scalar)) => {
            let left_vec = vector_ops::extract_single_vector_from_scalar(left_scalar)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
            let right_vec = vector_ops::extract_single_vector_from_scalar(right_scalar)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;

            let similarity = compute_single_similarity(&left_vec, &right_vec);

            // Return as scalar since both inputs were scalars
            Ok(ColumnarValue::Scalar(
                datafusion::scalar::ScalarValue::Float32(Some(similarity)),
            ))
        }
    }
}

// UDFs are cached using `LazyLock` static variables to avoid recreating them for each query.
// Each distance metric (L2, Cosine, Dot) has its own cached UDF instance for both
// `vector_distance` and `vector_similarity` functions. This provides significant performance
// improvements when executing multiple queries, as UDF initialization only happens once.

static VECTOR_SIMILARITY_L2_UDF: LazyLock<Arc<ScalarUDF>> = LazyLock::new(|| {
    let func = move |args: &[ColumnarValue]| -> datafusion::error::Result<ColumnarValue> {
        vector_similarity_func(args, &DistanceMetric::L2)
    };

    Arc::new(ScalarUDF::new_from_impl(VectorSimilarityUDF {
        name: "vector_similarity_l2".to_string(),
        func: Arc::new(func),
        metric: DistanceMetric::L2,
        signature: Signature::any(2, Volatility::Immutable),
    }))
});

static VECTOR_SIMILARITY_COSINE_UDF: LazyLock<Arc<ScalarUDF>> = LazyLock::new(|| {
    let func = move |args: &[ColumnarValue]| -> datafusion::error::Result<ColumnarValue> {
        vector_similarity_func(args, &DistanceMetric::Cosine)
    };

    Arc::new(ScalarUDF::new_from_impl(VectorSimilarityUDF {
        name: "vector_similarity_cosine".to_string(),
        func: Arc::new(func),
        metric: DistanceMetric::Cosine,
        signature: Signature::any(2, Volatility::Immutable),
    }))
});

static VECTOR_SIMILARITY_DOT_UDF: LazyLock<Arc<ScalarUDF>> = LazyLock::new(|| {
    let func = move |args: &[ColumnarValue]| -> datafusion::error::Result<ColumnarValue> {
        vector_similarity_func(args, &DistanceMetric::Dot)
    };

    Arc::new(ScalarUDF::new_from_impl(VectorSimilarityUDF {
        name: "vector_similarity_dot".to_string(),
        func: Arc::new(func),
        metric: DistanceMetric::Dot,
        signature: Signature::any(2, Volatility::Immutable),
    }))
});

/// Get a cached vector similarity UDF for the given distance metric
pub(crate) fn create_vector_similarity_udf(metric: &DistanceMetric) -> Arc<ScalarUDF> {
    match metric {
        DistanceMetric::L2 => VECTOR_SIMILARITY_L2_UDF.clone(),
        DistanceMetric::Cosine => VECTOR_SIMILARITY_COSINE_UDF.clone(),
        DistanceMetric::Dot => VECTOR_SIMILARITY_DOT_UDF.clone(),
        DistanceMetric::Hamming => HAMMING_SIMILARITY_UDF.clone(),
    }
}

/// UDF implementation for vector distance
struct VectorDistanceUDF {
    name: String,
    func: UdfFunc,
    metric: DistanceMetric,
    signature: Signature,
}

impl std::fmt::Debug for VectorDistanceUDF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorDistanceUDF")
            .field("name", &self.name)
            .field("metric", &self.metric)
            .finish()
    }
}

impl datafusion::logical_expr::ScalarUDFImpl for VectorDistanceUDF {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> datafusion::error::Result<DataType> {
        Ok(DataType::Float32)
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> datafusion::error::Result<ColumnarValue> {
        (self.func)(&args.args)
    }
}

impl PartialEq for VectorDistanceUDF {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.metric == other.metric
        // Note: signature is structural so we don't compare it
    }
}

impl Eq for VectorDistanceUDF {}

impl std::hash::Hash for VectorDistanceUDF {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

/// UDF implementation for vector similarity
struct VectorSimilarityUDF {
    name: String,
    func: UdfFunc,
    metric: DistanceMetric,
    signature: Signature,
}

impl std::fmt::Debug for VectorSimilarityUDF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorSimilarityUDF")
            .field("name", &self.name)
            .field("metric", &self.metric)
            .finish()
    }
}

impl datafusion::logical_expr::ScalarUDFImpl for VectorSimilarityUDF {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> datafusion::error::Result<DataType> {
        Ok(DataType::Float32)
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> datafusion::error::Result<ColumnarValue> {
        (self.func)(&args.args)
    }
}

impl PartialEq for VectorSimilarityUDF {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.metric == other.metric
        // Note: signature is structural so we don't compare it
    }
}

impl Eq for VectorSimilarityUDF {}

impl std::hash::Hash for VectorSimilarityUDF {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

// ---------------------------------------------------------------------------
// Hamming distance / similarity UDFs for binary vectors (FixedSizeBinary)
// ---------------------------------------------------------------------------

/// Hamming distance UDF operating on `FixedSizeBinary(2048)` columns.
///
/// Returns the number of differing bits (as `Float32`) between two binary
/// vectors. This is the primary distance metric for 16384-bit fingerprints.
fn hamming_distance_func(
    args: &[ColumnarValue],
) -> datafusion::error::Result<ColumnarValue> {
    if args.len() != 2 {
        return Err(datafusion::error::DataFusionError::Execution(
            "hamming_distance requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (ColumnarValue::Array(left_arr), ColumnarValue::Array(right_arr)) => {
            let left = vector_ops::extract_binary_vectors(left_arr)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
            let right = vector_ops::extract_binary_vectors(right_arr)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;

            let distances: Vec<f32> = if right.len() == 1 {
                vector_ops::compute_hamming_distances(&left, &right[0])
            } else if left.len() == 1 {
                vector_ops::compute_hamming_distances(&right, &left[0])
            } else if left.len() == right.len() {
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| vector_ops::hamming_distance(l, r) as f32)
                    .collect()
            } else {
                return Err(datafusion::error::DataFusionError::Execution(format!(
                    "Binary vector count mismatch: left={}, right={}",
                    left.len(),
                    right.len()
                )));
            };

            let result = Arc::new(arrow::array::Float32Array::from(distances)) as ArrayRef;
            Ok(ColumnarValue::Array(result))
        }

        (ColumnarValue::Array(left_arr), ColumnarValue::Scalar(right_scalar)) => {
            let left = vector_ops::extract_binary_vectors(left_arr)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
            let query = vector_ops::extract_single_binary_from_scalar(right_scalar)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
            let distances = vector_ops::compute_hamming_distances(&left, &query);
            let result = Arc::new(arrow::array::Float32Array::from(distances)) as ArrayRef;
            Ok(ColumnarValue::Array(result))
        }

        (ColumnarValue::Scalar(left_scalar), ColumnarValue::Array(right_arr)) => {
            let right = vector_ops::extract_binary_vectors(right_arr)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
            let query = vector_ops::extract_single_binary_from_scalar(left_scalar)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
            let distances = vector_ops::compute_hamming_distances(&right, &query);
            let result = Arc::new(arrow::array::Float32Array::from(distances)) as ArrayRef;
            Ok(ColumnarValue::Array(result))
        }

        (ColumnarValue::Scalar(left_scalar), ColumnarValue::Scalar(right_scalar)) => {
            let left = vector_ops::extract_single_binary_from_scalar(left_scalar)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
            let right = vector_ops::extract_single_binary_from_scalar(right_scalar)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
            let dist = vector_ops::hamming_distance(&left, &right) as f32;
            Ok(ColumnarValue::Scalar(
                datafusion::scalar::ScalarValue::Float32(Some(dist)),
            ))
        }
    }
}

/// Hamming similarity UDF: returns `1.0 - hamming_distance / total_bits`.
fn hamming_similarity_func(
    args: &[ColumnarValue],
) -> datafusion::error::Result<ColumnarValue> {
    if args.len() != 2 {
        return Err(datafusion::error::DataFusionError::Execution(
            "hamming_similarity requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (ColumnarValue::Array(left_arr), ColumnarValue::Array(right_arr)) => {
            let left = vector_ops::extract_binary_vectors(left_arr)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
            let right = vector_ops::extract_binary_vectors(right_arr)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;

            let sims: Vec<f32> = if right.len() == 1 {
                vector_ops::compute_hamming_similarities(&left, &right[0])
            } else if left.len() == 1 {
                vector_ops::compute_hamming_similarities(&right, &left[0])
            } else if left.len() == right.len() {
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| vector_ops::hamming_similarity(l, r))
                    .collect()
            } else {
                return Err(datafusion::error::DataFusionError::Execution(format!(
                    "Binary vector count mismatch: left={}, right={}",
                    left.len(),
                    right.len()
                )));
            };

            let result = Arc::new(arrow::array::Float32Array::from(sims)) as ArrayRef;
            Ok(ColumnarValue::Array(result))
        }

        (ColumnarValue::Array(left_arr), ColumnarValue::Scalar(right_scalar)) => {
            let left = vector_ops::extract_binary_vectors(left_arr)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
            let query = vector_ops::extract_single_binary_from_scalar(right_scalar)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
            let sims = vector_ops::compute_hamming_similarities(&left, &query);
            let result = Arc::new(arrow::array::Float32Array::from(sims)) as ArrayRef;
            Ok(ColumnarValue::Array(result))
        }

        (ColumnarValue::Scalar(left_scalar), ColumnarValue::Array(right_arr)) => {
            let right = vector_ops::extract_binary_vectors(right_arr)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
            let query = vector_ops::extract_single_binary_from_scalar(left_scalar)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
            let sims = vector_ops::compute_hamming_similarities(&right, &query);
            let result = Arc::new(arrow::array::Float32Array::from(sims)) as ArrayRef;
            Ok(ColumnarValue::Array(result))
        }

        (ColumnarValue::Scalar(left_scalar), ColumnarValue::Scalar(right_scalar)) => {
            let left = vector_ops::extract_single_binary_from_scalar(left_scalar)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
            let right = vector_ops::extract_single_binary_from_scalar(right_scalar)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
            let sim = vector_ops::hamming_similarity(&left, &right);
            Ok(ColumnarValue::Scalar(
                datafusion::scalar::ScalarValue::Float32(Some(sim)),
            ))
        }
    }
}

/// Cached hamming distance UDF.
static HAMMING_DISTANCE_UDF: LazyLock<Arc<ScalarUDF>> = LazyLock::new(|| {
    let func = move |args: &[ColumnarValue]| -> datafusion::error::Result<ColumnarValue> {
        hamming_distance_func(args)
    };

    Arc::new(ScalarUDF::new_from_impl(HammingDistanceUDF {
        name: "hamming_distance".to_string(),
        func: Arc::new(func),
        signature: Signature::any(2, Volatility::Immutable),
    }))
});

/// Cached hamming similarity UDF.
static HAMMING_SIMILARITY_UDF: LazyLock<Arc<ScalarUDF>> = LazyLock::new(|| {
    let func = move |args: &[ColumnarValue]| -> datafusion::error::Result<ColumnarValue> {
        hamming_similarity_func(args)
    };

    Arc::new(ScalarUDF::new_from_impl(HammingDistanceUDF {
        name: "hamming_similarity".to_string(),
        func: Arc::new(func),
        signature: Signature::any(2, Volatility::Immutable),
    }))
});

/// Get the cached hamming distance UDF.
#[allow(dead_code)]
pub(crate) fn create_hamming_distance_udf() -> Arc<ScalarUDF> {
    HAMMING_DISTANCE_UDF.clone()
}

/// Get the cached hamming similarity UDF.
#[allow(dead_code)]
pub(crate) fn create_hamming_similarity_udf() -> Arc<ScalarUDF> {
    HAMMING_SIMILARITY_UDF.clone()
}

/// UDF implementation for hamming distance on binary vectors.
struct HammingDistanceUDF {
    name: String,
    func: UdfFunc,
    signature: Signature,
}

impl std::fmt::Debug for HammingDistanceUDF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HammingDistanceUDF")
            .field("name", &self.name)
            .finish()
    }
}

impl datafusion::logical_expr::ScalarUDFImpl for HammingDistanceUDF {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> datafusion::error::Result<DataType> {
        Ok(DataType::Float32)
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> datafusion::error::Result<ColumnarValue> {
        (self.func)(&args.args)
    }
}

impl PartialEq for HammingDistanceUDF {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for HammingDistanceUDF {}

impl std::hash::Hash for HammingDistanceUDF {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, FixedSizeListArray, Float32Array};
    use arrow::datatypes::{DataType, Field};
    use std::sync::Arc;

    /// Helper to create a FixedSizeListArray from vectors
    fn create_vector_array(vectors: Vec<Vec<f32>>) -> ArrayRef {
        let dim = vectors[0].len() as i32;
        let mut values = Vec::new();
        for vec in vectors {
            values.extend(vec);
        }

        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let value_array = Arc::new(Float32Array::from(values));
        Arc::new(FixedSizeListArray::try_new(field, dim, value_array, None).unwrap())
    }

    #[test]
    fn test_vector_distance_l2_udf() {
        // Create test vectors: [1,0,0] and [0,1,0]
        let left = create_vector_array(vec![vec![1.0, 0.0, 0.0]]);
        let right = create_vector_array(vec![vec![0.0, 1.0, 0.0]]);

        let args = vec![ColumnarValue::Array(left), ColumnarValue::Array(right)];

        // Call the distance function directly
        let result = super::vector_distance_func(&args, &DistanceMetric::L2).unwrap();

        if let ColumnarValue::Array(arr) = result {
            let float_arr = arr.as_any().downcast_ref::<Float32Array>().unwrap();
            assert_eq!(float_arr.len(), 1);
            // Distance should be sqrt(2) ≈ 1.414
            assert!((float_arr.value(0) - 1.414).abs() < 0.01);
        } else {
            panic!("Expected array result");
        }
    }

    #[test]
    fn test_vector_distance_cosine_udf() {
        // Create identical vectors: [1,0,0] and [1,0,0]
        let left = create_vector_array(vec![vec![1.0, 0.0, 0.0]]);
        let right = create_vector_array(vec![vec![1.0, 0.0, 0.0]]);

        let args = vec![ColumnarValue::Array(left), ColumnarValue::Array(right)];

        let result = super::vector_distance_func(&args, &DistanceMetric::Cosine).unwrap();

        if let ColumnarValue::Array(arr) = result {
            let float_arr = arr.as_any().downcast_ref::<Float32Array>().unwrap();
            assert_eq!(float_arr.len(), 1);
            // Cosine distance of identical vectors should be 0
            assert_eq!(float_arr.value(0), 0.0);
        } else {
            panic!("Expected array result");
        }
    }

    #[test]
    fn test_vector_similarity_cosine_udf() {
        // Create identical vectors: [1,0,0] and [1,0,0]
        let left = create_vector_array(vec![vec![1.0, 0.0, 0.0]]);
        let right = create_vector_array(vec![vec![1.0, 0.0, 0.0]]);

        let args = vec![ColumnarValue::Array(left), ColumnarValue::Array(right)];

        let result = super::vector_similarity_func(&args, &DistanceMetric::Cosine).unwrap();

        if let ColumnarValue::Array(arr) = result {
            let float_arr = arr.as_any().downcast_ref::<Float32Array>().unwrap();
            assert_eq!(float_arr.len(), 1);
            // Cosine similarity of identical vectors should be 1.0
            assert_eq!(float_arr.value(0), 1.0);
        } else {
            panic!("Expected array result");
        }
    }

    #[test]
    fn test_vector_distance_broadcast() {
        // Multiple left vectors, single right vector (broadcast)
        let left = create_vector_array(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        let right = create_vector_array(vec![vec![1.0, 0.0, 0.0]]);

        let args = vec![ColumnarValue::Array(left), ColumnarValue::Array(right)];

        let result = super::vector_distance_func(&args, &DistanceMetric::L2).unwrap();

        if let ColumnarValue::Array(arr) = result {
            let float_arr = arr.as_any().downcast_ref::<Float32Array>().unwrap();
            assert_eq!(float_arr.len(), 3);
            assert_eq!(float_arr.value(0), 0.0); // Identical
            assert!((float_arr.value(1) - 1.414).abs() < 0.01); // Orthogonal
            assert!((float_arr.value(2) - 1.414).abs() < 0.01); // Orthogonal
        } else {
            panic!("Expected array result");
        }
    }

    #[test]
    fn test_vector_distance_wrong_arg_count() {
        let left = create_vector_array(vec![vec![1.0, 0.0, 0.0]]);

        // Only 1 argument (should fail)
        let args = vec![ColumnarValue::Array(left)];

        let result = super::vector_distance_func(&args, &DistanceMetric::L2);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("requires exactly 2 arguments"));
    }

    /// Helper to create a scalar from a single vector
    fn create_vector_scalar(vec: Vec<f32>) -> datafusion::scalar::ScalarValue {
        use datafusion::scalar::ScalarValue;

        let dim = vec.len() as i32;
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let values = Arc::new(Float32Array::from(vec));
        let list_array = FixedSizeListArray::try_new(field, dim, values, None).unwrap();

        ScalarValue::try_from_array(&list_array, 0).unwrap()
    }

    #[test]
    fn test_vector_distance_array_vs_scalar() {
        // Test memory-efficient scalar broadcast: array of vectors vs single scalar query
        let left = create_vector_array(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        let right_scalar = create_vector_scalar(vec![1.0, 0.0, 0.0]);

        let args = vec![
            ColumnarValue::Array(left),
            ColumnarValue::Scalar(right_scalar),
        ];

        let result = super::vector_distance_func(&args, &DistanceMetric::L2).unwrap();

        if let ColumnarValue::Array(arr) = result {
            let float_arr = arr.as_any().downcast_ref::<Float32Array>().unwrap();
            assert_eq!(float_arr.len(), 3);
            assert_eq!(float_arr.value(0), 0.0); // Identical to query
            assert!((float_arr.value(1) - 1.414).abs() < 0.01); // Orthogonal
            assert!((float_arr.value(2) - 1.414).abs() < 0.01); // Orthogonal
        } else {
            panic!("Expected array result");
        }
    }

    #[test]
    fn test_vector_distance_scalar_vs_array() {
        // Test the reverse: scalar query vs array of vectors
        let left_scalar = create_vector_scalar(vec![1.0, 0.0, 0.0]);
        let right = create_vector_array(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);

        let args = vec![
            ColumnarValue::Scalar(left_scalar),
            ColumnarValue::Array(right),
        ];

        let result = super::vector_distance_func(&args, &DistanceMetric::Cosine).unwrap();

        if let ColumnarValue::Array(arr) = result {
            let float_arr = arr.as_any().downcast_ref::<Float32Array>().unwrap();
            assert_eq!(float_arr.len(), 2);
            assert_eq!(float_arr.value(0), 0.0); // Identical to query (cosine distance = 0)
            assert_eq!(float_arr.value(1), 1.0); // Orthogonal (cosine distance = 1)
        } else {
            panic!("Expected array result");
        }
    }

    #[test]
    fn test_vector_distance_scalar_vs_scalar() {
        // Test scalar vs scalar - should return a scalar result
        let left_scalar = create_vector_scalar(vec![1.0, 0.0, 0.0]);
        let right_scalar = create_vector_scalar(vec![0.0, 1.0, 0.0]);

        let args = vec![
            ColumnarValue::Scalar(left_scalar),
            ColumnarValue::Scalar(right_scalar),
        ];

        let result = super::vector_distance_func(&args, &DistanceMetric::L2).unwrap();

        // Should return a scalar, not an array
        if let ColumnarValue::Scalar(scalar) = result {
            if let datafusion::scalar::ScalarValue::Float32(Some(dist)) = scalar {
                assert!((dist - 1.414).abs() < 0.01); // sqrt(2) for orthogonal unit vectors
            } else {
                panic!("Expected Float32 scalar");
            }
        } else {
            panic!("Expected scalar result for scalar vs scalar");
        }
    }

    #[test]
    fn test_vector_similarity_array_vs_scalar() {
        // Test memory-efficient scalar broadcast for similarity
        let left = create_vector_array(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.707, 0.707, 0.0], // 45 degrees
        ]);
        let right_scalar = create_vector_scalar(vec![1.0, 0.0, 0.0]);

        let args = vec![
            ColumnarValue::Array(left),
            ColumnarValue::Scalar(right_scalar),
        ];

        let result = super::vector_similarity_func(&args, &DistanceMetric::Cosine).unwrap();

        if let ColumnarValue::Array(arr) = result {
            let float_arr = arr.as_any().downcast_ref::<Float32Array>().unwrap();
            assert_eq!(float_arr.len(), 3);
            assert_eq!(float_arr.value(0), 1.0); // Identical to query
            assert_eq!(float_arr.value(1), 0.0); // Orthogonal
            assert!((float_arr.value(2) - 0.707).abs() < 0.01); // cos(45°) ≈ 0.707
        } else {
            panic!("Expected array result");
        }
    }

    #[test]
    fn test_vector_similarity_scalar_vs_scalar() {
        // Test scalar vs scalar for similarity
        let left_scalar = create_vector_scalar(vec![1.0, 0.0, 0.0]);
        let right_scalar = create_vector_scalar(vec![1.0, 0.0, 0.0]);

        let args = vec![
            ColumnarValue::Scalar(left_scalar),
            ColumnarValue::Scalar(right_scalar),
        ];

        let result = super::vector_similarity_func(&args, &DistanceMetric::Cosine).unwrap();

        // Should return a scalar, not an array
        if let ColumnarValue::Scalar(scalar) = result {
            if let datafusion::scalar::ScalarValue::Float32(Some(sim)) = scalar {
                assert_eq!(sim, 1.0); // Identical vectors have similarity 1.0
            } else {
                panic!("Expected Float32 scalar");
            }
        } else {
            panic!("Expected scalar result for scalar vs scalar");
        }
    }

    #[test]
    fn test_vector_distance_dot_product_with_scalar() {
        // Test dot product metric with scalar broadcast
        let left = create_vector_array(vec![vec![1.0, 0.0, 0.0], vec![0.9, 0.1, 0.0]]);
        let right_scalar = create_vector_scalar(vec![1.0, 0.0, 0.0]);

        let args = vec![
            ColumnarValue::Array(left),
            ColumnarValue::Scalar(right_scalar),
        ];

        let result = super::vector_distance_func(&args, &DistanceMetric::Dot).unwrap();

        if let ColumnarValue::Array(arr) = result {
            let float_arr = arr.as_any().downcast_ref::<Float32Array>().unwrap();
            assert_eq!(float_arr.len(), 2);
            // Dot product distance = -dot_product
            assert_eq!(float_arr.value(0), -1.0); // -1.0 * 1.0 = -1.0
            assert!((float_arr.value(1) + 0.9).abs() < 0.01); // -(0.9 * 1.0) = -0.9
        } else {
            panic!("Expected array result");
        }
    }

    /// Helper to create a FixedSizeBinaryArray from byte vectors
    fn create_binary_array(vectors: Vec<Vec<u8>>) -> ArrayRef {
        use arrow::array::FixedSizeBinaryArray;
        let byte_len = vectors[0].len() as i32;
        let arr = FixedSizeBinaryArray::try_from_iter(vectors.iter().map(|v| v.as_slice()))
            .unwrap();
        Arc::new(arr) as ArrayRef
    }

    #[test]
    fn test_hamming_distance_identical() {
        let v = vec![0xFFu8; 8]; // 64 bits all set
        let left = create_binary_array(vec![v.clone()]);
        let right = create_binary_array(vec![v]);

        let args = vec![ColumnarValue::Array(left), ColumnarValue::Array(right)];
        let result = hamming_distance_func(&args).unwrap();

        if let ColumnarValue::Array(arr) = result {
            let float_arr = arr.as_any().downcast_ref::<Float32Array>().unwrap();
            assert_eq!(float_arr.value(0), 0.0);
        } else {
            panic!("Expected array result");
        }
    }

    #[test]
    fn test_hamming_distance_complement() {
        let a = vec![0xFFu8; 8]; // all 1s
        let b = vec![0x00u8; 8]; // all 0s
        let left = create_binary_array(vec![a]);
        let right = create_binary_array(vec![b]);

        let args = vec![ColumnarValue::Array(left), ColumnarValue::Array(right)];
        let result = hamming_distance_func(&args).unwrap();

        if let ColumnarValue::Array(arr) = result {
            let float_arr = arr.as_any().downcast_ref::<Float32Array>().unwrap();
            assert_eq!(float_arr.value(0), 64.0); // 8 bytes * 8 bits = 64 differing bits
        } else {
            panic!("Expected array result");
        }
    }

    #[test]
    fn test_hamming_similarity_identical() {
        let v = vec![0xAAu8; 16];
        let left = create_binary_array(vec![v.clone()]);
        let right = create_binary_array(vec![v]);

        let args = vec![ColumnarValue::Array(left), ColumnarValue::Array(right)];
        let result = hamming_similarity_func(&args).unwrap();

        if let ColumnarValue::Array(arr) = result {
            let float_arr = arr.as_any().downcast_ref::<Float32Array>().unwrap();
            assert_eq!(float_arr.value(0), 1.0);
        } else {
            panic!("Expected array result");
        }
    }

    #[test]
    fn test_hamming_distance_broadcast() {
        let a = vec![0xFFu8; 8];
        let b = vec![0x00u8; 8];
        let c = vec![0x0Fu8; 8]; // half bits differ from 0xFF
        let left = create_binary_array(vec![a, b, c]);
        let right = create_binary_array(vec![vec![0xFFu8; 8]]);

        let args = vec![ColumnarValue::Array(left), ColumnarValue::Array(right)];
        let result = hamming_distance_func(&args).unwrap();

        if let ColumnarValue::Array(arr) = result {
            let float_arr = arr.as_any().downcast_ref::<Float32Array>().unwrap();
            assert_eq!(float_arr.len(), 3);
            assert_eq!(float_arr.value(0), 0.0);  // identical
            assert_eq!(float_arr.value(1), 64.0); // complement
            assert_eq!(float_arr.value(2), 32.0); // 0x0F vs 0xFF = 4 bits per byte * 8 bytes
        } else {
            panic!("Expected array result");
        }
    }
}
