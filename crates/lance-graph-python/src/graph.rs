// Copyright 2024 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Graph query functionality for Lance datasets

use std::collections::HashMap;
use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::Schema;
use datafusion::datasource::{DefaultTableSource, MemTable};
use datafusion::execution::context::SessionContext;
use lance_graph::{
    ast::{DistanceMetric as RustDistanceMetric, GraphPattern, ReadingClause},
    CypherQuery as RustCypherQuery, ExecutionStrategy as RustExecutionStrategy,
    GraphConfig as RustGraphConfig, GraphError as RustGraphError, InMemoryCatalog,
    VectorSearch as RustVectorSearch,
};
use pyo3::{
    exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
    IntoPyObject,
};
use serde_json::Value as JsonValue;

use crate::namespace::PyDirNamespace;
use crate::RT;

/// Execution strategy for Cypher queries
#[pyclass(name = "ExecutionStrategy", module = "lance.graph")]
#[derive(Clone, Copy)]
pub enum ExecutionStrategy {
    /// Use DataFusion query planner (default, full feature support)
    DataFusion,
    /// Use simple single-table executor (legacy, limited features)
    Simple,
    /// Use Lance native executor (not yet implemented)
    LanceNative,
}

impl From<ExecutionStrategy> for RustExecutionStrategy {
    fn from(strategy: ExecutionStrategy) -> Self {
        match strategy {
            ExecutionStrategy::DataFusion => RustExecutionStrategy::DataFusion,
            ExecutionStrategy::Simple => RustExecutionStrategy::Simple,
            ExecutionStrategy::LanceNative => RustExecutionStrategy::LanceNative,
        }
    }
}

/// Distance metric for vector similarity search
#[pyclass(name = "DistanceMetric", module = "lance.graph")]
#[derive(Clone, Copy)]
pub enum DistanceMetric {
    /// L2 (Euclidean) distance - smaller is more similar
    L2,
    /// Cosine distance - smaller is more similar (1 - cosine_similarity)
    Cosine,
    /// Dot product distance - for normalized vectors, larger is more similar
    Dot,
}

impl From<DistanceMetric> for RustDistanceMetric {
    fn from(metric: DistanceMetric) -> Self {
        match metric {
            DistanceMetric::L2 => RustDistanceMetric::L2,
            DistanceMetric::Cosine => RustDistanceMetric::Cosine,
            DistanceMetric::Dot => RustDistanceMetric::Dot,
        }
    }
}

/// Vector similarity search builder
///
/// This class provides an explicit API for vector similarity search that can work with:
/// - In-memory PyArrow tables (brute-force search)
/// - Lance datasets (ANN search with indices)
///
/// This is distinct from the UDF-based vector search (`vector_distance()`, `vector_similarity()`)
/// which is integrated into Cypher queries.
///
/// Examples
/// --------
/// >>> from lance_graph import VectorSearch, DistanceMetric
/// >>>
/// >>> # Basic vector search on a PyArrow table
/// >>> results = VectorSearch("embedding") \
/// ...     .query_vector([0.1, 0.2, 0.3]) \
/// ...     .metric(DistanceMetric.Cosine) \
/// ...     .top_k(10) \
/// ...     .search(table)
#[pyclass(name = "VectorSearch", module = "lance.graph")]
#[derive(Clone)]
pub struct VectorSearch {
    inner: RustVectorSearch,
    /// Flag to enable vector-first Lance ANN execution path.
    /// This is stored separately because RustVectorSearch doesn't have this concept.
    use_lance_index: bool,
}

#[pymethods]
impl VectorSearch {
    /// Create a new VectorSearch builder
    ///
    /// Parameters
    /// ----------
    /// column : str
    ///     Name of the vector column to search
    ///
    /// Returns
    /// -------
    /// VectorSearch
    ///     A new VectorSearch builder
    #[new]
    fn new(column: &str) -> Self {
        Self {
            inner: RustVectorSearch::new(column),
            use_lance_index: false,
        }
    }

    /// Set the query vector
    ///
    /// Parameters
    /// ----------
    /// vector : list[float]
    ///     The query vector for similarity computation
    ///
    /// Returns
    /// -------
    /// VectorSearch
    ///     A new builder with the query vector set
    fn query_vector(&self, vector: Vec<f32>) -> Self {
        Self {
            inner: self.inner.clone().query_vector(vector),
            use_lance_index: self.use_lance_index,
        }
    }

    /// Set the distance metric
    ///
    /// Parameters
    /// ----------
    /// metric : DistanceMetric
    ///     The distance metric to use (L2, Cosine, or Dot)
    ///
    /// Returns
    /// -------
    /// VectorSearch
    ///     A new builder with the metric set
    fn metric(&self, metric: DistanceMetric) -> Self {
        Self {
            inner: self.inner.clone().metric(metric.into()),
            use_lance_index: self.use_lance_index,
        }
    }

    /// Set the number of results to return
    ///
    /// Parameters
    /// ----------
    /// k : int
    ///     Number of top results to return
    ///
    /// Returns
    /// -------
    /// VectorSearch
    ///     A new builder with top_k set
    fn top_k(&self, k: usize) -> Self {
        Self {
            inner: self.inner.clone().top_k(k),
            use_lance_index: self.use_lance_index,
        }
    }

    /// Whether to include the distance column in results
    ///
    /// Parameters
    /// ----------
    /// include : bool
    ///     If True, adds a distance column to results (default: True)
    ///
    /// Returns
    /// -------
    /// VectorSearch
    ///     A new builder with the setting applied
    fn include_distance(&self, include: bool) -> Self {
        Self {
            inner: self.inner.clone().include_distance(include),
            use_lance_index: self.use_lance_index,
        }
    }

    /// Set the name of the distance column
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Name for the distance column (default: "_distance")
    ///
    /// Returns
    /// -------
    /// VectorSearch
    ///     A new builder with the column name set
    fn distance_column_name(&self, name: &str) -> Self {
        Self {
            inner: self.inner.clone().distance_column_name(name),
            use_lance_index: self.use_lance_index,
        }
    }

    /// Use Lance ANN index when datasets are Lance datasets.
    ///
    /// This enables a vector-first execution path that queries the Lance index
    /// and then runs the Cypher query on the top-k results. This can be much faster
    /// for large datasets but may change semantics when the Cypher query includes
    /// filters or additional constraints.
    ///
    /// Parameters
    /// ----------
    /// enabled : bool
    ///     If True, use Lance ANN index for vector search when possible.
    ///
    /// Returns
    /// -------
    /// VectorSearch
    ///     A new builder with the setting applied
    fn use_lance_index(&self, enabled: bool) -> Self {
        Self {
            inner: self.inner.clone(),
            use_lance_index: enabled,
        }
    }

    /// Execute brute-force vector search on a PyArrow table
    ///
    /// Parameters
    /// ----------
    /// table : pyarrow.Table
    ///     The table containing vectors to search
    ///
    /// Returns
    /// -------
    /// pyarrow.Table
    ///     Results sorted by similarity with top_k rows
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If query vector is not set or column not found
    /// RuntimeError
    ///     If search execution fails
    fn search(&self, py: Python, table: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let batch = python_any_to_record_batch(table)?;
        let batch = normalize_record_batch(batch)?;

        let inner = self.inner.clone();
        let result = RT
            .block_on(Some(py), inner.search(&batch))?
            .map_err(graph_error_to_pyerr)?;

        record_batch_to_python_table(py, &result)
    }

    fn __repr__(&self) -> String {
        "VectorSearch(...)".to_string()
    }
}

/// Convert GraphError to PyErr
fn graph_error_to_pyerr(err: RustGraphError) -> PyErr {
    match &err {
        RustGraphError::ParseError { .. }
        | RustGraphError::ConfigError { .. }
        | RustGraphError::PlanError { .. }
        | RustGraphError::InvalidPattern { .. } => PyValueError::new_err(err.to_string()),
        RustGraphError::UnsupportedFeature { .. } => {
            PyNotImplementedError::new_err(err.to_string())
        }
        _ => PyRuntimeError::new_err(err.to_string()),
    }
}

/// Graph configuration for interpreting Lance datasets as property graphs
#[pyclass(name = "GraphConfig", module = "lance.graph")]
#[derive(Clone)]
pub struct GraphConfig {
    inner: RustGraphConfig,
}

#[pymethods]
impl GraphConfig {
    /// Create a new GraphConfig builder
    #[staticmethod]
    fn builder() -> GraphConfigBuilder {
        GraphConfigBuilder::new()
    }

    /// Get node labels
    fn node_labels(&self) -> Vec<String> {
        // Extract from the config's node mappings
        self.inner.node_mappings.keys().cloned().collect()
    }

    /// Get relationship types
    fn relationship_types(&self) -> Vec<String> {
        // Extract from the config's relationship mappings
        self.inner.relationship_mappings.keys().cloned().collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "GraphConfig(nodes={:?}, relationships={:?})",
            self.node_labels(),
            self.relationship_types()
        )
    }
}

/// Builder for GraphConfig
#[pyclass(name = "GraphConfigBuilder", module = "lance.graph")]
#[derive(Clone)]
pub struct GraphConfigBuilder {
    inner: lance_graph::config::GraphConfigBuilder,
}

#[pymethods]
impl GraphConfigBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: RustGraphConfig::builder(),
        }
    }

    /// Add a node label mapping
    ///
    /// Parameters
    /// ----------
    /// label : str
    ///     The node label (e.g., "Person")
    /// id_field : str
    ///     The field in the dataset that serves as the node ID
    ///
    /// Returns
    /// -------
    /// GraphConfigBuilder
    ///     A new builder with the node mapping applied
    fn with_node_label(&self, label: &str, id_field: &str) -> Self {
        Self {
            inner: self.inner.clone().with_node_label(label, id_field),
        }
    }

    /// Add a relationship mapping
    ///
    /// Parameters
    /// ----------
    /// rel_type : str
    ///     The relationship type (e.g., "KNOWS")
    /// source_field : str
    ///     The field containing source node IDs
    /// target_field : str
    ///     The field containing target node IDs
    ///
    /// Returns
    /// -------
    /// GraphConfigBuilder
    ///     A new builder with the relationship mapping applied
    fn with_relationship(&self, rel_type: &str, source_field: &str, target_field: &str) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .with_relationship(rel_type, source_field, target_field),
        }
    }

    /// Build the GraphConfig
    ///
    /// Returns
    /// -------
    /// GraphConfig
    ///     The configured graph config
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the configuration is invalid
    fn build(&self) -> PyResult<GraphConfig> {
        let config = self.inner.clone().build().map_err(graph_error_to_pyerr)?;
        Ok(GraphConfig { inner: config })
    }
}

/// Cypher query interface for Lance datasets
#[pyclass(name = "CypherQuery", module = "lance.graph")]
#[derive(Clone)]
pub struct CypherQuery {
    inner: RustCypherQuery,
}

#[pymethods]
impl CypherQuery {
    /// Create a new Cypher query
    ///
    /// Parameters
    /// ----------
    /// query_text : str
    ///     The Cypher query string
    ///
    /// Returns
    /// -------
    /// CypherQuery
    ///     The parsed query
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the query cannot be parsed
    #[new]
    fn new(query_text: &str) -> PyResult<Self> {
        let query = RustCypherQuery::new(query_text).map_err(graph_error_to_pyerr)?;
        Ok(Self { inner: query })
    }

    /// Set the graph configuration
    ///
    /// Parameters
    /// ----------
    /// config : GraphConfig
    ///     The graph configuration
    ///
    /// Returns
    /// -------
    /// CypherQuery
    ///     A new query instance with the config set
    fn with_config(&self, config: &GraphConfig) -> Self {
        Self {
            inner: self.inner.clone().with_config(config.inner.clone()),
        }
    }

    /// Add a query parameter
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///     Parameter name
    /// value : object
    ///     Parameter value (will be converted to JSON)
    ///
    /// Returns
    /// -------
    /// CypherQuery
    ///     A new query instance with the parameter added
    fn with_parameter(&self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Convert Python value to JSON
        let json_value = python_to_json(value)?;
        Ok(Self {
            inner: self.inner.clone().with_parameter(key, json_value),
        })
    }

    /// Get the query text
    fn query_text(&self) -> &str {
        self.inner.query_text()
    }

    /// Get query parameters
    fn parameters(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for (key, value) in self.inner.parameters() {
            let py_value = json_to_python(py, value)?;
            dict.set_item(key, py_value)?;
        }
        Ok(dict.unbind())
    }

    /// Convert query to SQL
    ///
    /// Parameters
    /// ----------
    /// datasets : dict
    ///     Dictionary mapping table names to Lance datasets
    ///
    /// Returns
    /// -------
    /// str
    ///     The generated SQL query
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If SQL generation fails
    fn to_sql(&self, py: Python, datasets: &Bound<'_, PyDict>) -> PyResult<String> {
        // Convert datasets to Arrow RecordBatch map
        let arrow_datasets = python_datasets_to_batches(datasets)?;

        // Clone for async move
        let inner_query = self.inner.clone();

        // Execute via runtime
        let sql = RT
            .block_on(Some(py), inner_query.to_sql(arrow_datasets))?
            .map_err(graph_error_to_pyerr)?;

        Ok(sql)
    }

    /// Execute query against Lance datasets
    ///
    /// Parameters
    /// ----------
    /// datasets : dict
    ///     Dictionary mapping table names to Lance datasets
    /// strategy : ExecutionStrategy, optional
    ///     Execution strategy to use (defaults to DataFusion)
    ///
    /// Returns
    /// -------
    /// pyarrow.Table
    ///     Query results as Arrow table
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If query execution fails
    ///
    /// Examples
    /// --------
    /// >>> # Default strategy (DataFusion)
    /// >>> result = query.execute(datasets)
    ///
    /// >>> # Explicit strategy
    /// >>> from lance.graph import ExecutionStrategy
    /// >>> result = query.execute(datasets, strategy=ExecutionStrategy.Simple)
    #[pyo3(signature = (datasets, strategy=None))]
    fn execute(
        &self,
        py: Python,
        datasets: &Bound<'_, PyDict>,
        strategy: Option<ExecutionStrategy>,
    ) -> PyResult<PyObject> {
        // Convert datasets to Arrow batches while holding the GIL
        let arrow_datasets = python_datasets_to_batches(datasets)?;

        // Convert Python strategy to Rust strategy
        let rust_strategy = strategy.map(|s| s.into());

        // Clone the inner query for use in the async block
        let inner_query = self.inner.clone();

        // Use RT.block_on with Some(py) like the scanner to_pyarrow method
        let result_batch = RT
            .block_on(Some(py), inner_query.execute(arrow_datasets, rust_strategy))?
            .map_err(graph_error_to_pyerr)?;

        record_batch_to_python_table(py, &result_batch)
    }

    /// Execute query using a namespace resolver.
    ///
    /// Parameters
    /// ----------
    /// namespace : DirNamespace
    ///     Directory-backed namespace that resolves table names to Lance datasets.
    /// strategy : ExecutionStrategy, optional
    ///     Execution strategy to use (defaults to DataFusion)
    ///
    /// Returns
    /// -------
    /// pyarrow.Table
    ///     Query results as Arrow table
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If query execution fails
    #[pyo3(signature = (namespace, strategy=None))]
    fn execute_with_namespace(
        &self,
        py: Python,
        namespace: &Bound<'_, PyDirNamespace>,
        strategy: Option<ExecutionStrategy>,
    ) -> PyResult<PyObject> {
        let rust_strategy = strategy.map(|s| s.into());
        let inner_query = self.inner.clone();
        let namespace_arc = namespace.borrow().inner.clone();

        let result_batch = RT
            .block_on(
                Some(py),
                inner_query.execute_with_namespace_arc(namespace_arc, rust_strategy),
            )?
            .map_err(graph_error_to_pyerr)?;

        record_batch_to_python_table(py, &result_batch)
    }

    /// Explain query using the DataFusion planner with in-memory datasets
    ///
    /// Parameters
    /// ----------
    /// datasets : dict
    ///     Dictionary mapping table names to in-memory tables (pyarrow.Table, LanceDataset, etc.)
    ///     Keys should match node labels and relationship types in the graph config.
    ///
    /// Returns
    /// -------
    /// str
    ///     Query graph logical plan, DataFusion logical plan, DataFusion physical plan as string
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the query is invalid or datasets are missing
    /// RuntimeError
    ///     If query explain fails
    fn explain(&self, py: Python, datasets: &Bound<'_, PyDict>) -> PyResult<String> {
        // Convert datasets to Arrow RecordBatch map
        let arrow_datasets = python_datasets_to_batches(datasets)?;

        // Clone for async move
        let inner_query = self.inner.clone();

        // Execute via runtime
        let plan = RT
            .block_on(Some(py), inner_query.explain(arrow_datasets))?
            .map_err(graph_error_to_pyerr)?;

        Ok(plan)
    }

    /// Get variables used in the query
    fn variables(&self) -> Vec<String> {
        self.inner.variables()
    }

    /// Get node labels referenced in the query
    fn node_labels(&self) -> Vec<String> {
        self.inner.ast().get_node_labels()
    }

    /// Get relationship types referenced in the query
    fn relationship_types(&self) -> Vec<String> {
        self.inner.ast().get_relationship_types()
    }

    /// Execute Cypher query, then apply vector search reranking on results
    ///
    /// This is a convenience method for the common GraphRAG pattern:
    /// 1. Run Cypher query to get candidate entities via graph traversal
    /// 2. Rerank candidates by vector similarity
    ///
    /// Parameters
    /// ----------
    /// datasets : dict
    ///     Dictionary mapping table names to Lance datasets or PyArrow tables
    /// vector_search : VectorSearch
    ///     VectorSearch configuration for reranking
    ///     (Use VectorSearch.use_lance_index(True) to enable a vector-first
    ///     execution path when datasets are Lance datasets.)
    ///
    /// Returns
    /// -------
    /// pyarrow.Table
    ///     Results sorted by vector similarity with top_k rows
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the query is invalid or vector search is misconfigured
    /// RuntimeError
    ///     If query execution fails
    ///
    /// Examples
    /// --------
    /// >>> from lance_graph import CypherQuery, VectorSearch, DistanceMetric, GraphConfig
    /// >>>
    /// >>> config = GraphConfig.builder().with_node_label("Document", "id").build()
    /// >>> query = CypherQuery("MATCH (d:Document) WHERE d.category = 'tech' RETURN d.id, d.name, d.embedding")
    /// >>> query = query.with_config(config)
    /// >>>
    /// >>> results = query.execute_with_vector_rerank(
    /// ...     datasets,
    /// ...     VectorSearch("d.embedding")
    /// ...         .query_vector([0.1, 0.2, 0.3])
    /// ...         .metric(DistanceMetric.Cosine)
    /// ...         .top_k(10)
    /// ... )
    fn execute_with_vector_rerank(
        &self,
        py: Python,
        datasets: &Bound<'_, PyDict>,
        vector_search: &VectorSearch,
    ) -> PyResult<PyObject> {
        if vector_search.use_lance_index {
            if let Some(result) = try_execute_with_lance_index(py, &self.inner, datasets, vector_search)? {
                return record_batch_to_python_table(py, &result);
            }
        }

        // Convert datasets to Arrow batches
        let arrow_datasets = python_datasets_to_batches(datasets)?;

        // Clone for async move
        let inner_query = self.inner.clone();
        let vs = vector_search.inner.clone();

        // Execute via runtime
        let result = RT
            .block_on(
                Some(py),
                inner_query.execute_with_vector_rerank(arrow_datasets, vs),
            )?
            .map_err(graph_error_to_pyerr)?;

        record_batch_to_python_table(py, &result)
    }

    fn __repr__(&self) -> String {
        format!("CypherQuery(\"{}\")", self.inner.query_text())
    }
}

// Helper functions to convert between Python and JSON values
fn python_to_json(value: &Bound<'_, PyAny>) -> PyResult<JsonValue> {
    if value.is_none() {
        Ok(JsonValue::Null)
    } else if let Ok(b) = value.extract::<bool>() {
        Ok(JsonValue::Bool(b))
    } else if let Ok(i) = value.extract::<i64>() {
        Ok(JsonValue::Number(i.into()))
    } else if let Ok(f) = value.extract::<f64>() {
        Ok(JsonValue::Number(
            serde_json::Number::from_f64(f)
                .ok_or_else(|| PyValueError::new_err("Invalid float value"))?,
        ))
    } else if let Ok(s) = value.extract::<String>() {
        Ok(JsonValue::String(s))
    } else {
        Err(PyValueError::new_err("Unsupported parameter type"))
    }
}

fn json_to_python(py: Python, value: &JsonValue) -> PyResult<PyObject> {
    match value {
        JsonValue::Null => Ok(py.None()),
        JsonValue::Bool(b) => {
            use pyo3::types::PyBool;
            Ok(PyBool::new(py, *b).to_owned().into())
        }
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.unbind().into())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.unbind().into())
            } else {
                Ok(n.to_string().into_pyobject(py)?.unbind().into())
            }
        }
        JsonValue::String(s) => Ok(s.as_str().into_pyobject(py)?.unbind().into()),
        JsonValue::Array(_) | JsonValue::Object(_) => {
            // For complex types, convert to string representation
            Ok(value.to_string().into_pyobject(py)?.unbind().into())
        }
    }
}

// Helper functions for Arrow conversion

/// Convert a single Python dataset value to a RecordBatch
fn python_dataset_to_batch(value: &Bound<'_, PyAny>) -> PyResult<RecordBatch> {
    let batch = if is_lance_dataset(value)? {
        lance_dataset_to_record_batch(value)?
    } else if value.hasattr("to_table")? {
        let table = value.call_method0("to_table")?;
        python_any_to_record_batch(&table)?
    } else {
        python_any_to_record_batch(value)?
    };
    normalize_record_batch(batch)
}

fn python_datasets_to_batches(
    datasets: &Bound<'_, PyDict>,
) -> PyResult<HashMap<String, RecordBatch>> {
    python_datasets_to_batches_impl(datasets, None)
}

fn python_datasets_to_batches_with_override(
    datasets: &Bound<'_, PyDict>,
    override_label: &str,
    override_batch: &RecordBatch,
) -> PyResult<HashMap<String, RecordBatch>> {
    python_datasets_to_batches_impl(datasets, Some((override_label, override_batch)))
}

fn python_datasets_to_batches_impl(
    datasets: &Bound<'_, PyDict>,
    override_entry: Option<(&str, &RecordBatch)>,
) -> PyResult<HashMap<String, RecordBatch>> {
    let mut arrow_datasets = HashMap::new();
    for (key, value) in datasets.iter() {
        let table_name: String = key.extract()?;

        // Check if this table should use the override batch
        if let Some((override_label, override_batch)) = override_entry {
            if table_name == override_label {
                arrow_datasets.insert(table_name, override_batch.clone());
                continue;
            }
        }

        let batch = python_dataset_to_batch(&value)?;
        arrow_datasets.insert(table_name, batch);
    }
    Ok(arrow_datasets)
}

fn try_execute_with_lance_index(
    py: Python,
    query: &RustCypherQuery,
    datasets: &Bound<'_, PyDict>,
    vector_search: &VectorSearch,
) -> PyResult<Option<RecordBatch>> {
    // Only use vector-first path for simple queries without filters.
    // Queries with WITH/WHERE clauses need the standard rerank path to ensure correct semantics.
    let ast = query.ast();
    if ast.with_clause.is_some()
        || ast.where_clause.is_some()
        || ast.post_with_where_clause.is_some()
    {
        return Ok(None);
    }

    let query_vector = match vector_search.inner.get_query_vector() {
        Some(vec) => vec.to_vec(),
        None => {
            return Err(PyValueError::new_err(
                "VectorSearch.query_vector is required when use_lance_index is enabled",
            ))
        }
    };

    let (alias, column) = split_vector_column(vector_search.inner.column());
    let label = resolve_vector_label(query, alias.as_deref())?;
    let label = match label {
        Some(label) => label,
        None => return Ok(None),
    };

    let dataset_value = match datasets.get_item(&label)? {
        Some(value) => value,
        None => return Ok(None),
    };

    if !is_lance_dataset(&dataset_value)? {
        return Ok(None);
    }

    let metric_str = match vector_search.inner.get_metric() {
        RustDistanceMetric::L2 => "l2",
        RustDistanceMetric::Cosine => "cosine",
        RustDistanceMetric::Dot => "dot",
    };

    // Build the `nearest` dict for Lance's to_table() ANN query.
    // Setting use_index=true tells Lance to use the ANN index if available,
    // otherwise it falls back to flat (brute-force) search.
    let nearest = PyDict::new(py);
    nearest.set_item("column", column)?;
    nearest.set_item("k", vector_search.inner.get_top_k())?;
    nearest.set_item("q", query_vector)?;
    nearest.set_item("metric", metric_str)?;
    nearest.set_item("use_index", true)?;

    let kwargs = PyDict::new(py);
    kwargs.set_item("nearest", nearest)?;

    let table = dataset_value.call_method("to_table", (), Some(&kwargs))?;
    let batch = python_any_to_record_batch(&table)?;
    let batch = normalize_record_batch(batch)?;

    let arrow_datasets = python_datasets_to_batches_with_override(datasets, &label, &batch)?;

    let inner_query = query.clone();
    let result = RT
        .block_on(Some(py), inner_query.execute(arrow_datasets, None))?
        .map_err(graph_error_to_pyerr)?;

    Ok(Some(result))
}

fn split_vector_column(column: &str) -> (Option<String>, &str) {
    let mut parts = column.splitn(2, '.');
    let first = parts.next().unwrap_or(column);
    if let Some(rest) = parts.next() {
        (Some(first.to_string()), rest)
    } else {
        (None, column)
    }
}

fn resolve_vector_label(
    query: &RustCypherQuery,
    alias: Option<&str>,
) -> PyResult<Option<String>> {
    let alias_map = alias_map_from_query(query);
    if let Some(alias) = alias {
        return Ok(alias_map.get(alias).cloned());
    }
    if alias_map.len() == 1 {
        return Ok(alias_map.values().next().cloned());
    }
    Ok(None)
}

fn alias_map_from_query(query: &RustCypherQuery) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let ast = query.ast();
    for clause in ast
        .reading_clauses
        .iter()
        .chain(ast.post_with_reading_clauses.iter())
    {
        if let ReadingClause::Match(match_clause) = clause {
            for pattern in &match_clause.patterns {
                collect_aliases_from_pattern(pattern, &mut map);
            }
        }
    }
    map
}

fn collect_aliases_from_pattern(pattern: &GraphPattern, map: &mut HashMap<String, String>) {
    match pattern {
        GraphPattern::Node(node) => {
            if let (Some(var), Some(label)) = (node.variable.as_ref(), node.labels.first()) {
                map.entry(var.clone()).or_insert_with(|| label.clone());
            }
        }
        GraphPattern::Path(path) => {
            if let (Some(var), Some(label)) = (path.start_node.variable.as_ref(), path.start_node.labels.first()) {
                map.entry(var.clone()).or_insert_with(|| label.clone());
            }
            for segment in &path.segments {
                if let (Some(var), Some(label)) = (segment.end_node.variable.as_ref(), segment.end_node.labels.first()) {
                    map.entry(var.clone()).or_insert_with(|| label.clone());
                }
            }
        }
    }
}

fn normalize_record_batch(batch: RecordBatch) -> PyResult<RecordBatch> {
    if batch.schema().metadata().is_empty() {
        return Ok(batch);
    }

    // DataFusion expects stable, metadata-free schemas across optimization passes.
    // Rebuild the schema without the PyArrow-specific metadata to avoid mismatches.
    let columns = batch.columns().to_vec();
    let fields = batch
        .schema()
        .fields()
        .iter()
        .map(|field| (**field).clone())
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, columns).map_err(|e| {
        PyRuntimeError::new_err(format!("Failed to strip metadata from Arrow batch: {}", e))
    })
}

// Check if a Python object is a Lance dataset
fn is_lance_dataset(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    // Check the type name directly
    if let Ok(type_name) = value.get_type().repr() {
        let type_str = type_name.to_string();
        let is_lance = type_str.contains("lance.dataset.LanceDataset");
        return Ok(is_lance);
    }

    // Fallback: check for uri (which we know Lance datasets have)
    value.hasattr("uri")
}

// Convert Lance dataset to RecordBatch using alternative methods that avoid GIL issues
fn lance_dataset_to_record_batch(dataset: &Bound<'_, PyAny>) -> PyResult<RecordBatch> {
    // Try the scanner() approach that's used elsewhere in the Lance codebase
    if let Ok(scanner) = dataset.call_method0("scanner") {
        if let Ok(py_reader) = scanner.call_method0("to_pyarrow") {
            return python_any_to_record_batch(&py_reader);
        }
    }

    // Method 2: Use the count_rows + take approach to get data without to_table()
    if dataset.hasattr("count_rows")? && dataset.hasattr("take")? {
        let count = dataset.call_method0("count_rows")?;
        let count_int: usize = count.extract()?;
        if count_int > 0 {
            // Take a range of rows (limit to 10000 for performance)
            let take_count = std::cmp::min(count_int, 10000);

            // Create a Python list of indices
            let py = dataset.py();
            let range_list = pyo3::types::PyList::empty(py);
            for i in 0..take_count {
                range_list.append(i)?;
            }

            if let Ok(table) = dataset.call_method1("take", (range_list,)) {
                return python_any_to_record_batch(&table);
            }
        }
    }

    // Fallback: Use to_table() (this might still cause GIL issues but is last resort)
    let table = dataset.call_method0("to_table")?;
    python_any_to_record_batch(&table)
}

fn python_any_to_record_batch(value: &Bound<'_, PyAny>) -> PyResult<RecordBatch> {
    use arrow::pyarrow::FromPyArrow;

    if let Ok(batch) = RecordBatch::from_pyarrow_bound(value) {
        return Ok(batch);
    }

    let mut reader = ArrowArrayStreamReader::from_pyarrow_bound(value)?;
    let schema = reader.schema();
    let mut batches = Vec::new();
    while let Some(batch) = reader
        .next()
        .transpose()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read Arrow batch: {}", e)))?
    {
        batches.push(batch);
    }

    if batches.is_empty() {
        return Err(PyRuntimeError::new_err("Table has no data"));
    }

    concat_batches(&schema, &batches)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to concatenate batches: {}", e)))
}

fn record_batch_to_python_table(
    py: Python,
    batch: &arrow_array::RecordBatch,
) -> PyResult<PyObject> {
    use arrow::pyarrow::ToPyArrow;
    use pyo3::types::PyList;

    // Convert RecordBatch -> PyArrow.RecordBatch
    let py_rb = batch.to_pyarrow(py)?;

    // Build pyarrow.Table from batches
    let pa = py.import("pyarrow")?;
    let table_cls = pa.getattr("Table")?;
    let batches = PyList::new(py, [py_rb])?;
    let table = table_cls.call_method1("from_batches", (batches,))?;
    Ok(table.unbind())
}

/// Cypher query engine with cached catalog for efficient multi-query execution
///
/// This class provides a high-performance query execution interface by building
/// the catalog and DataFusion context once, then reusing them across multiple queries.
/// This avoids the cold-start penalty of rebuilding metadata structures on every query.
///
/// Use this class when you need to execute multiple queries against the same datasets.
/// For single queries, the simpler `CypherQuery.execute()` API may be more convenient.
///
/// Implementation Details
/// ----------------------
/// - Each dataset is registered as both a node and relationship source in the catalog.
///   The GraphConfig and query planner determine at runtime which interpretation to use
///   based on the Cypher query pattern (e.g., (p:Person) vs -[:KNOWS]->).
/// - SessionContext is internally Arc-wrapped, so cloning for each query is cheap
///   (just incrementing refcounts, not copying state).
///
/// Examples
/// --------
/// >>> from lance_graph import CypherEngine, GraphConfig
/// >>> import pyarrow as pa
/// >>>
/// >>> # Setup
/// >>> config = GraphConfig.builder() \\
/// ...     .with_node_label("Person", "id") \\
/// ...     .with_relationship("KNOWS", "src_id", "dst_id") \\
/// ...     .build()
/// >>>
/// >>> datasets = {
/// ...     "Person": person_table,
/// ...     "KNOWS": knows_table
/// ... }
/// >>>
/// >>> # Create engine once
/// >>> engine = CypherEngine(config, datasets)
/// >>>
/// >>> # Execute multiple queries efficiently
/// >>> result1 = engine.execute("MATCH (p:Person) WHERE p.age > 30 RETURN p.name")
/// >>> result2 = engine.execute("MATCH (p:Person)-[:KNOWS]->(f) RETURN p.name, f.name")
/// >>> result3 = engine.execute("MATCH (p:Person) RETURN count(*)")
#[pyclass(name = "CypherEngine", module = "lance.graph")]
pub struct CypherEngine {
    config: RustGraphConfig,
    catalog: Arc<dyn lance_graph::GraphSourceCatalog>,
    context: Arc<datafusion::execution::context::SessionContext>,
}

#[pymethods]
impl CypherEngine {
    /// Create a new CypherEngine with cached catalog
    ///
    /// This builds the catalog and DataFusion context once during initialization.
    /// Subsequent queries will reuse these structures for better performance.
    ///
    /// Parameters
    /// ----------
    /// config : GraphConfig
    ///     The graph configuration defining node labels and relationships
    /// datasets : dict
    ///     Dictionary mapping table names to Lance datasets or PyArrow tables
    ///
    /// Returns
    /// -------
    /// CypherEngine
    ///     A new engine instance ready to execute queries
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the configuration or datasets are invalid
    /// RuntimeError
    ///     If catalog building fails
    #[new]
    fn new(config: &GraphConfig, datasets: &Bound<'_, PyDict>) -> PyResult<Self> {
        // Convert datasets to Arrow batches
        let arrow_datasets = python_datasets_to_batches(datasets)?;

        if arrow_datasets.is_empty() {
            return Err(PyValueError::new_err("No input datasets provided"));
        }

        // Create session context and catalog
        let ctx = SessionContext::new();
        let mut catalog = InMemoryCatalog::new();

        // Register all datasets as tables
        for (name, batch) in &arrow_datasets {
            let mem_table = Arc::new(
                MemTable::try_new(batch.schema(), vec![vec![batch.clone()]])
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create MemTable for {}: {}", name, e)))?,
            );

            // Register in session context for execution
            let normalized_name = name.to_lowercase();
            ctx.register_table(&normalized_name, mem_table.clone())
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to register table {}: {}", name, e)))?;

            let table_source = Arc::new(DefaultTableSource::new(mem_table));

            // Register as both node and relationship source with original name.
            //
            // This is intentional: lance-graph uses GraphConfig to determine at query-planning
            // time whether a dataset should be treated as a node table or relationship table
            // based on the Cypher query pattern (e.g., MATCH (p:Person) vs -[:KNOWS]->).
            //
            // By registering all datasets in both catalogs, we allow the planner to look up
            // the correct source based on query context. This pattern matches the Rust 
            // implementation in query.rs:build_catalog_and_context_from_datasets.
            catalog = catalog
                .with_node_source(name, table_source.clone())
                .with_relationship_source(name, table_source);
        }

        Ok(Self {
            config: config.inner.clone(),
            catalog: Arc::new(catalog),
            context: Arc::new(ctx),
        })
    }

    /// Execute a Cypher query using the cached catalog
    ///
    /// This method reuses the catalog and context built during initialization,
    /// avoiding the overhead of rebuilding metadata structures.
    ///
    /// Parameters
    /// ----------
    /// query : str
    ///     The Cypher query string to execute
    ///
    /// Returns
    /// -------
    /// pyarrow.Table
    ///     Query results as Arrow table
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the query is invalid
    /// RuntimeError
    ///     If query execution fails
    ///
    /// Examples
    /// --------
    /// >>> result = engine.execute("MATCH (p:Person) WHERE p.age > 30 RETURN p.name")
    /// >>> print(result.to_pandas())
    fn execute(
        &self,
        py: Python,
        query: &str,
    ) -> PyResult<PyObject> {
        // Parse the query
        let cypher_query = RustCypherQuery::new(query)
            .map_err(graph_error_to_pyerr)?
            .with_config(self.config.clone());

        // Execute using the cached catalog and context
        // Note: SessionContext is internally Arc-wrapped, so clone() is cheap.
        // We clone the SessionContext (not the Arc) because execute_with_catalog_and_context
        // takes ownership. DataFusion's SessionContext::clone() just increments Arc refcounts
        // for internal state (RuntimeEnv, SessionState), so this is a shallow clone.
        let catalog = self.catalog.clone();
        let context = self.context.as_ref().clone();

        let result_batch = RT
            .block_on(
                Some(py),
                cypher_query.execute_with_catalog_and_context(catalog, context),
            )?
            .map_err(graph_error_to_pyerr)?;

        record_batch_to_python_table(py, &result_batch)
    }

    /// Get the graph configuration
    fn config(&self) -> GraphConfig {
        GraphConfig {
            inner: self.config.clone(),
        }
    }

    /// Execute Cypher query with vector reranking
    ///
    /// Convenience method combining graph traversal and vector similarity ranking.
    /// See CypherQuery.execute_with_vector_rerank for detailed documentation.
    ///
    /// Parameters
    /// ----------
    /// query : str
    ///     Cypher query string
    /// vector_search : VectorSearch
    ///     Vector search configuration
    ///
    /// Returns
    /// -------
    /// pyarrow.Table
    ///     Results sorted by vector similarity
    fn execute_with_vector_rerank(
        &self,
        py: Python,
        query: &str,
        vector_search: &VectorSearch,
    ) -> PyResult<PyObject> {
        // Parse query and execute with cached catalog/context
        let cypher_query = RustCypherQuery::new(query)
            .map_err(graph_error_to_pyerr)?
            .with_config(self.config.clone());

        let catalog = self.catalog.clone();
        let context = self.context.as_ref().clone();
        let vs = vector_search.inner.clone();

        // Execute query to get candidates, then apply vector reranking
        let result_batch = RT
            .block_on(Some(py), async move {
                let candidates = cypher_query
                    .execute_with_catalog_and_context(catalog, context)
                    .await?;
                vs.search(&candidates).await
            })?
            .map_err(graph_error_to_pyerr)?;

        record_batch_to_python_table(py, &result_batch)
    }

    fn __repr__(&self) -> String {
        format!(
            "CypherEngine(nodes={}, relationships={})",
            self.config.node_mappings.len(),
            self.config.relationship_mappings.len()
        )
    }
}

/// Register graph functionality with the Python module
pub fn register_graph_module(py: Python, parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let graph_module = PyModule::new(py, "graph")?;

    graph_module.add_class::<ExecutionStrategy>()?;
    graph_module.add_class::<DistanceMetric>()?;
    graph_module.add_class::<GraphConfig>()?;
    graph_module.add_class::<GraphConfigBuilder>()?;
    graph_module.add_class::<CypherQuery>()?;
    graph_module.add_class::<CypherEngine>()?;
    graph_module.add_class::<VectorSearch>()?;
    graph_module.add_class::<PyDirNamespace>()?;

    parent_module.add_submodule(&graph_module)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_vector_column_with_alias() {
        let (alias, column) = split_vector_column("d.embedding");
        assert_eq!(alias, Some("d".to_string()));
        assert_eq!(column, "embedding");
    }

    #[test]
    fn test_split_vector_column_without_alias() {
        let (alias, column) = split_vector_column("embedding");
        assert_eq!(alias, None);
        assert_eq!(column, "embedding");
    }

    #[test]
    fn test_split_vector_column_with_multiple_dots() {
        // Should only split on the first dot
        let (alias, column) = split_vector_column("d.nested.embedding");
        assert_eq!(alias, Some("d".to_string()));
        assert_eq!(column, "nested.embedding");
    }

    #[test]
    fn test_split_vector_column_empty_string() {
        let (alias, column) = split_vector_column("");
        assert_eq!(alias, None);
        assert_eq!(column, "");
    }

    #[test]
    fn test_alias_map_from_simple_node_query() {
        let query = RustCypherQuery::new("MATCH (d:Document) RETURN d.name").unwrap();
        let map = alias_map_from_query(&query);
        assert_eq!(map.get("d"), Some(&"Document".to_string()));
    }

    #[test]
    fn test_alias_map_from_multiple_nodes() {
        let query =
            RustCypherQuery::new("MATCH (p:Person), (d:Document) RETURN p.name, d.title").unwrap();
        let map = alias_map_from_query(&query);
        assert_eq!(map.get("p"), Some(&"Person".to_string()));
        assert_eq!(map.get("d"), Some(&"Document".to_string()));
    }

    #[test]
    fn test_alias_map_from_path_query() {
        let query =
            RustCypherQuery::new("MATCH (p:Person)-[:KNOWS]->(f:Friend) RETURN p.name, f.name")
                .unwrap();
        let map = alias_map_from_query(&query);
        assert_eq!(map.get("p"), Some(&"Person".to_string()));
        assert_eq!(map.get("f"), Some(&"Friend".to_string()));
    }

    #[test]
    fn test_resolve_vector_label_with_alias() {
        let query = RustCypherQuery::new("MATCH (d:Document) RETURN d.name").unwrap();
        let result = resolve_vector_label(&query, Some("d")).unwrap();
        assert_eq!(result, Some("Document".to_string()));
    }

    #[test]
    fn test_resolve_vector_label_without_alias_single_node() {
        let query = RustCypherQuery::new("MATCH (d:Document) RETURN d.name").unwrap();
        // When no alias is provided and there's only one node, should return that label
        let result = resolve_vector_label(&query, None).unwrap();
        assert_eq!(result, Some("Document".to_string()));
    }

    #[test]
    fn test_resolve_vector_label_without_alias_multiple_nodes() {
        let query =
            RustCypherQuery::new("MATCH (p:Person), (d:Document) RETURN p.name, d.title").unwrap();
        // When no alias is provided and there are multiple nodes, should return None
        let result = resolve_vector_label(&query, None).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_resolve_vector_label_unknown_alias() {
        let query = RustCypherQuery::new("MATCH (d:Document) RETURN d.name").unwrap();
        // When alias doesn't exist in the query, should return None
        let result = resolve_vector_label(&query, Some("x")).unwrap();
        assert_eq!(result, None);
    }
}
