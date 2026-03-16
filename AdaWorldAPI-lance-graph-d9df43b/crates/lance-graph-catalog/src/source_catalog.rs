// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Context-free source catalog for DataFusion logical planning.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::{Schema, SchemaRef};
use datafusion::logical_expr::TableSource;

/// A minimal catalog to resolve node labels and relationship types to logical table sources.
pub trait GraphSourceCatalog: Send + Sync {
    fn node_source(&self, label: &str) -> Option<Arc<dyn TableSource>>;
    fn relationship_source(&self, rel_type: &str) -> Option<Arc<dyn TableSource>>;
}

/// A simple in-memory catalog useful for tests and bootstrap wiring.
pub struct InMemoryCatalog {
    node_sources: HashMap<String, Arc<dyn TableSource>>,
    rel_sources: HashMap<String, Arc<dyn TableSource>>,
}

impl InMemoryCatalog {
    pub fn new() -> Self {
        Self {
            node_sources: HashMap::new(),
            rel_sources: HashMap::new(),
        }
    }

    pub fn with_node_source(
        mut self,
        label: impl Into<String>,
        source: Arc<dyn TableSource>,
    ) -> Self {
        // Normalize key to lowercase for case-insensitive lookup
        self.node_sources
            .insert(label.into().to_lowercase(), source);
        self
    }

    pub fn with_relationship_source(
        mut self,
        rel_type: impl Into<String>,
        source: Arc<dyn TableSource>,
    ) -> Self {
        // Normalize key to lowercase for case-insensitive lookup
        self.rel_sources
            .insert(rel_type.into().to_lowercase(), source);
        self
    }
}

impl Default for InMemoryCatalog {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSourceCatalog for InMemoryCatalog {
    /// Get node source with case-insensitive label lookup
    ///
    /// Note: Keys are stored as lowercase, so this is an O(1) operation.
    fn node_source(&self, label: &str) -> Option<Arc<dyn TableSource>> {
        self.node_sources.get(&label.to_lowercase()).cloned()
    }

    /// Get relationship source with case-insensitive type lookup
    ///
    /// Note: Keys are stored as lowercase, so this is an O(1) operation.
    fn relationship_source(&self, rel_type: &str) -> Option<Arc<dyn TableSource>> {
        self.rel_sources.get(&rel_type.to_lowercase()).cloned()
    }
}

/// A trivial logical table source with a fixed schema.
pub struct SimpleTableSource {
    schema: SchemaRef,
}

impl SimpleTableSource {
    pub fn new(schema: SchemaRef) -> Self {
        Self { schema }
    }
    pub fn empty() -> Self {
        Self {
            schema: Arc::new(Schema::empty()),
        }
    }
}

impl TableSource for SimpleTableSource {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
