// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Python bindings for the catalog integration.

use std::sync::Arc;

use lance_graph::sql_catalog::build_context_from_connector;
use lance_graph::{
    CatalogInfo, Connector, SchemaInfo, TableInfo, UnityCatalogConfig, UnityCatalogProvider,
};
use lance_graph::table_readers::default_table_readers;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::graph::{graph_error_to_pyerr, SqlEngine};
use crate::RT;

// ---- Python wrapper for CatalogInfo ----

#[pyclass(name = "CatalogInfo", module = "lance.graph")]
#[derive(Clone)]
pub struct PyCatalogInfo {
    inner: CatalogInfo,
}

#[pymethods]
impl PyCatalogInfo {
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn comment(&self) -> Option<&str> {
        self.inner.comment.as_deref()
    }

    fn __repr__(&self) -> String {
        format!("CatalogInfo(name='{}')", self.inner.name)
    }
}

// ---- Python wrapper for SchemaInfo ----

#[pyclass(name = "SchemaInfo", module = "lance.graph")]
#[derive(Clone)]
pub struct PySchemaInfo {
    inner: SchemaInfo,
}

#[pymethods]
impl PySchemaInfo {
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn catalog_name(&self) -> &str {
        &self.inner.catalog_name
    }

    #[getter]
    fn comment(&self) -> Option<&str> {
        self.inner.comment.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "SchemaInfo(name='{}.{}')",
            self.inner.catalog_name, self.inner.name
        )
    }
}

// ---- Python wrapper for TableInfo ----

#[pyclass(name = "TableInfo", module = "lance.graph")]
#[derive(Clone)]
pub struct PyTableInfo {
    inner: TableInfo,
}

#[pymethods]
impl PyTableInfo {
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn catalog_name(&self) -> &str {
        &self.inner.catalog_name
    }

    #[getter]
    fn schema_name(&self) -> &str {
        &self.inner.schema_name
    }

    #[getter]
    fn table_type(&self) -> &str {
        match self.inner.table_type {
            lance_graph::TableType::Managed => "MANAGED",
            lance_graph::TableType::External => "EXTERNAL",
        }
    }

    #[getter]
    fn data_source_format(&self) -> String {
        format!("{:?}", self.inner.data_source_format)
    }

    #[getter]
    fn storage_location(&self) -> Option<&str> {
        self.inner.storage_location.as_deref()
    }

    #[getter]
    fn comment(&self) -> Option<&str> {
        self.inner.comment.as_deref()
    }

    #[getter]
    fn num_columns(&self) -> usize {
        self.inner.columns.len()
    }

    /// Return column info as a list of dicts.
    fn columns<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for col in &self.inner.columns {
            let d = PyDict::new(py);
            d.set_item("name", &col.name)?;
            d.set_item("type_text", &col.type_text)?;
            d.set_item("type_name", &col.type_name)?;
            d.set_item("position", col.position)?;
            d.set_item("nullable", col.nullable)?;
            d.set_item("comment", col.comment.as_deref())?;
            list.append(d)?;
        }
        Ok(list)
    }

    fn __repr__(&self) -> String {
        format!(
            "TableInfo(name='{}.{}.{}', format={:?}, columns={})",
            self.inner.catalog_name,
            self.inner.schema_name,
            self.inner.name,
            self.inner.data_source_format,
            self.inner.columns.len()
        )
    }
}

// ---- Python UnityCatalog client ----

/// Unity Catalog client for browsing catalog metadata and auto-registering
/// tables into SqlEngine.
///
/// Examples
/// --------
/// >>> from lance_graph import UnityCatalog
/// >>> uc = UnityCatalog("http://localhost:8080/api/2.1/unity-catalog")
/// >>> catalogs = uc.list_catalogs()
/// >>> engine = uc.create_sql_engine("unity", "default")
/// >>> result = engine.execute("SELECT * FROM my_table LIMIT 10")
#[pyclass(name = "UnityCatalog", module = "lance.graph")]
pub struct PyUnityCatalog {
    connector: Arc<Connector>,
}

#[pymethods]
impl PyUnityCatalog {
    /// Create a new UnityCatalog client.
    ///
    /// Parameters
    /// ----------
    /// base_url : str
    ///     Base URL of the Unity Catalog server
    ///     (e.g., "http://localhost:8080/api/2.1/unity-catalog")
    /// token : str, optional
    ///     Bearer token for authentication
    /// timeout : int, optional
    ///     Request timeout in seconds
    #[new]
    #[pyo3(signature = (base_url, token=None, timeout=None))]
    fn new(base_url: &str, token: Option<&str>, timeout: Option<u64>) -> PyResult<Self> {
        let mut config = UnityCatalogConfig::new(base_url);
        if let Some(t) = token {
            config = config.with_token(t);
        }
        if let Some(secs) = timeout {
            config = config.with_timeout(secs);
        }

        let provider = UnityCatalogProvider::new(config)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let readers = default_table_readers();
        let connector = Connector::new(Arc::new(provider), readers);

        Ok(Self {
            connector: Arc::new(connector),
        })
    }

    /// List all catalogs.
    fn list_catalogs(&self, py: Python) -> PyResult<Vec<PyCatalogInfo>> {
        let connector = self.connector.clone();
        let result = RT
            .block_on(Some(py), async move { connector.list_catalogs().await })?
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(result
            .into_iter()
            .map(|c| PyCatalogInfo { inner: c })
            .collect())
    }

    /// List schemas in a catalog.
    ///
    /// Parameters
    /// ----------
    /// catalog_name : str
    ///     Name of the catalog
    fn list_schemas(&self, py: Python, catalog_name: &str) -> PyResult<Vec<PySchemaInfo>> {
        let connector = self.connector.clone();
        let cat = catalog_name.to_string();
        let result = RT
            .block_on(Some(py), async move {
                connector.list_schemas(&cat).await
            })?
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(result
            .into_iter()
            .map(|s| PySchemaInfo { inner: s })
            .collect())
    }

    /// List tables in a schema.
    ///
    /// Parameters
    /// ----------
    /// catalog_name : str
    ///     Name of the catalog
    /// schema_name : str
    ///     Name of the schema
    fn list_tables(
        &self,
        py: Python,
        catalog_name: &str,
        schema_name: &str,
    ) -> PyResult<Vec<PyTableInfo>> {
        let connector = self.connector.clone();
        let cat = catalog_name.to_string();
        let sch = schema_name.to_string();
        let result = RT
            .block_on(Some(py), async move {
                connector.list_tables(&cat, &sch).await
            })?
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(result
            .into_iter()
            .map(|t| PyTableInfo { inner: t })
            .collect())
    }

    /// Get detailed table info including columns.
    ///
    /// Parameters
    /// ----------
    /// catalog_name : str
    ///     Name of the catalog
    /// schema_name : str
    ///     Name of the schema
    /// table_name : str
    ///     Name of the table
    fn get_table(
        &self,
        py: Python,
        catalog_name: &str,
        schema_name: &str,
        table_name: &str,
    ) -> PyResult<PyTableInfo> {
        let connector = self.connector.clone();
        let cat = catalog_name.to_string();
        let sch = schema_name.to_string();
        let tbl = table_name.to_string();
        let result = RT
            .block_on(Some(py), async move {
                connector.get_table(&cat, &sch, &tbl).await
            })?
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyTableInfo { inner: result })
    }

    /// Create an SqlEngine with all tables from a UC schema auto-registered.
    ///
    /// Discovers all tables in the specified catalog.schema, registers them
    /// using the appropriate format reader (Parquet, Delta, etc.), and returns
    /// an SqlEngine ready for SQL queries.
    ///
    /// Parameters
    /// ----------
    /// catalog_name : str
    ///     Name of the UC catalog
    /// schema_name : str
    ///     Name of the UC schema
    ///
    /// Returns
    /// -------
    /// SqlEngine
    ///     An SqlEngine with all UC tables registered, ready for SQL queries.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If connection to UC fails or table registration fails
    fn create_sql_engine(
        &self,
        py: Python,
        catalog_name: &str,
        schema_name: &str,
    ) -> PyResult<SqlEngine> {
        let connector = self.connector.clone();
        let cat = catalog_name.to_string();
        let sch = schema_name.to_string();

        let ctx = RT
            .block_on(Some(py), async move {
                build_context_from_connector(&connector, &cat, &sch).await
            })?
            .map_err(graph_error_to_pyerr)?;

        Ok(SqlEngine::from_context(ctx))
    }

    fn __repr__(&self) -> String {
        "UnityCatalog(...)".to_string()
    }
}

/// Register catalog classes with the Python module.
pub fn register_catalog_module(_py: Python, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyUnityCatalog>()?;
    module.add_class::<PyCatalogInfo>()?;
    module.add_class::<PySchemaInfo>()?;
    module.add_class::<PyTableInfo>()?;
    Ok(())
}
