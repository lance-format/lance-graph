use std::sync::Arc;

use lance_graph::namespace::DirNamespace;
use pyo3::prelude::*;

#[pyclass(name = "DirNamespace", module = "lance.graph")]
pub struct PyDirNamespace {
    pub(crate) inner: Arc<DirNamespace>,
}

#[pymethods]
impl PyDirNamespace {
    #[new]
    fn new(base_uri: String) -> Self {
        Self {
            inner: Arc::new(DirNamespace::new(base_uri)),
        }
    }

    #[getter]
    fn base_uri(&self) -> String {
        self.inner.base_uri().to_string()
    }
}
