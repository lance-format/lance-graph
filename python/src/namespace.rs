// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;

use async_trait::async_trait;
use lance_namespace::models::{DescribeTableRequest, DescribeTableResponse};
use lance_namespace::{Error as NamespaceError, LanceNamespace, Result as NamespaceResult};
use pyo3::prelude::*;
use pyo3::prelude::PyAnyMethods;
use pyo3::types::{PyAny, PyDict, PyList};
use snafu::Location;

#[derive(Clone, Debug)]
pub struct PyNamespaceAdapter {
    namespace: Py<PyAny>,
}

impl PyNamespaceAdapter {
    pub fn new(namespace: Py<PyAny>) -> Self {
        Self { namespace }
    }

    fn convert_response(
        py: Python,
        response: &Bound<'_, PyAny>,
    ) -> NamespaceResult<DescribeTableResponse> {
        let location = extract_field::<String>(py, response, "location")?;
        let storage_options = extract_field::<HashMap<String, String>>(
            py,
            response,
            "storage_options",
        )?;

        Ok(DescribeTableResponse {
            location,
            storage_options,
            ..DescribeTableResponse::new()
        })
    }
}

fn extract_field<T>(
    py: Python,
    obj: &Bound<'_, PyAny>,
    key: &str,
) -> NamespaceResult<Option<T>>
where
    T: for<'a> FromPyObject<'a>,
{
    if let Ok(attr) = obj.getattr(key) {
        if attr.is_none() {
            return Ok(None);
        }
        return attr
            .extract::<T>()
            .map(Some)
            .map_err(py_err_to_namespace_error);
    }

    if obj
        .hasattr("get")
        .map_err(py_err_to_namespace_error)?
    {
        let value = obj
            .call_method1("get", (key, py.None()))
            .map_err(py_err_to_namespace_error)?;
        if value.is_none() {
            return Ok(None);
        }
        return value
            .extract::<T>()
            .map(Some)
            .map_err(py_err_to_namespace_error);
    }

    Ok(None)
}

unsafe impl Send for PyNamespaceAdapter {}
unsafe impl Sync for PyNamespaceAdapter {}

#[derive(Debug)]
struct PyNamespaceError(String);

impl std::fmt::Display for PyNamespaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for PyNamespaceError {}

fn py_err_to_namespace_error(err: PyErr) -> NamespaceError {
    NamespaceError::Namespace {
        source: Box::new(PyNamespaceError(err.to_string())),
        location: Location::new(file!(), line!(), column!()),
    }
}

#[async_trait]
impl LanceNamespace for PyNamespaceAdapter {
    fn namespace_id(&self) -> String {
        Python::with_gil(|py| {
            let obj = self.namespace.bind(py);
            if let Ok(id_obj) = obj.call_method0("namespace_id") {
                if let Ok(id) = id_obj.extract::<String>() {
                    return id;
                }
            }

            obj.repr()
                .map(|repr| repr.to_string())
                .unwrap_or_else(|_| "PyNamespaceAdapter".to_string())
        })
    }

    async fn describe_table(
        &self,
        request: DescribeTableRequest,
    ) -> NamespaceResult<DescribeTableResponse> {
        let namespace = self.namespace.clone();
        let id = request.id.clone();
        let version = request.version;

        tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> NamespaceResult<DescribeTableResponse> {
                let obj = namespace.bind(py);
                let kwargs = PyDict::new(py);

                if let Some(id_components) = id {
                    let py_list =
                        PyList::new(py, &id_components).map_err(py_err_to_namespace_error)?;
                    kwargs
                        .set_item("id", py_list)
                        .map_err(py_err_to_namespace_error)?;
                }

                if let Some(version) = version {
                    kwargs
                        .set_item("version", version)
                        .map_err(py_err_to_namespace_error)?;
                }

                let result = obj
                    .call_method("describe_table", (), Some(&kwargs))
                    .map_err(py_err_to_namespace_error)?;

                Self::convert_response(py, &result)
            })
        })
        .await
        .map_err(|err| NamespaceError::Namespace {
            source: Box::new(PyNamespaceError(format!(
                "Python describe_table task failed: {}",
                err
            ))),
            location: Location::new(file!(), line!(), column!()),
        })?
    }
}
