// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Connector facade that bundles a [`CatalogProvider`] with [`TableReader`]s.
//!
//! Inspired by Presto's `Connector` interface which exposes `getMetadata()` +
//! `getPageSourceProvider()`, this struct provides a convenient entry point
//! for users who want to browse a catalog and register tables for querying.

use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;

use crate::catalog_provider::{CatalogError, CatalogInfo, CatalogProvider, CatalogResult, DataSourceFormat, SchemaInfo, TableInfo};
use crate::table_reader::TableReader;

/// Bundles a [`CatalogProvider`] with [`TableReader`]s for convenient use.
///
/// The `Connector` is the primary entry point for interacting with an external
/// catalog. It delegates metadata operations to the catalog provider and data
/// reading to the appropriate table reader based on the table's data format.
///
/// # Extensibility
///
/// - Swap the catalog: pass a different `CatalogProvider` (e.g., AWS Glue).
/// - Add formats: pass additional `TableReader`s (e.g., Iceberg).
///
/// # Example
///
/// ```no_run
/// # use lance_graph_catalog::connector::Connector;
/// # use lance_graph_catalog::{UnityCatalogConfig, UnityCatalogProvider};
/// # fn example() {
/// // let catalog = Arc::new(UnityCatalogProvider::new(config)?);
/// // let readers = default_table_readers(); // from lance-graph crate
/// // let connector = Connector::new(catalog, readers);
/// // let tables = connector.list_tables("unity", "default").await?;
/// # }
/// ```
pub struct Connector {
    catalog: Arc<dyn CatalogProvider>,
    readers: Vec<Arc<dyn TableReader>>,
}

impl Connector {
    /// Create a new connector with the given catalog provider and table readers.
    pub fn new(
        catalog: Arc<dyn CatalogProvider>,
        readers: Vec<Arc<dyn TableReader>>,
    ) -> Self {
        Self { catalog, readers }
    }

    /// Get a reference to the underlying catalog provider.
    pub fn catalog(&self) -> &dyn CatalogProvider {
        self.catalog.as_ref()
    }

    /// Find a table reader that supports the given data format.
    pub fn reader_for(&self, format: &DataSourceFormat) -> Option<&dyn TableReader> {
        self.readers
            .iter()
            .find(|r| r.supported_formats().contains(format))
            .map(|r| r.as_ref())
    }

    /// List all available table readers.
    pub fn readers(&self) -> &[Arc<dyn TableReader>] {
        &self.readers
    }

    // ---- Delegate catalog operations ----

    pub async fn list_catalogs(&self) -> CatalogResult<Vec<CatalogInfo>> {
        self.catalog.list_catalogs().await
    }

    pub async fn get_catalog(&self, name: &str) -> CatalogResult<CatalogInfo> {
        self.catalog.get_catalog(name).await
    }

    pub async fn list_schemas(&self, catalog_name: &str) -> CatalogResult<Vec<SchemaInfo>> {
        self.catalog.list_schemas(catalog_name).await
    }

    pub async fn get_schema(
        &self,
        catalog_name: &str,
        schema_name: &str,
    ) -> CatalogResult<SchemaInfo> {
        self.catalog.get_schema(catalog_name, schema_name).await
    }

    pub async fn list_tables(
        &self,
        catalog_name: &str,
        schema_name: &str,
    ) -> CatalogResult<Vec<TableInfo>> {
        self.catalog.list_tables(catalog_name, schema_name).await
    }

    pub async fn get_table(
        &self,
        catalog_name: &str,
        schema_name: &str,
        table_name: &str,
    ) -> CatalogResult<TableInfo> {
        self.catalog
            .get_table(catalog_name, schema_name, table_name)
            .await
    }

    /// Register all tables from a catalog schema into a DataFusion `SessionContext`.
    ///
    /// For each table:
    /// 1. Retrieves full table metadata (including columns) from the catalog.
    /// 2. Converts columns to an Arrow schema.
    /// 3. Finds an appropriate [`TableReader`] for the table's data format.
    /// 4. Registers the table in the session context.
    ///
    /// If no reader matches the table's format, falls back to registering an
    /// empty `MemTable` with the correct schema (schema-only, for planning).
    ///
    /// Individual table failures are logged as warnings but do not abort the
    /// registration of remaining tables.
    ///
    /// Returns a list of `(table_name, schema)` for successfully registered tables.
    pub async fn register_schema(
        &self,
        ctx: &SessionContext,
        catalog_name: &str,
        schema_name: &str,
    ) -> CatalogResult<Vec<(String, SchemaRef)>> {
        let tables = self.catalog.list_tables(catalog_name, schema_name).await?;
        let mut registered = Vec::new();

        for table_summary in &tables {
            match self
                .register_single_table(ctx, catalog_name, schema_name, &table_summary.name)
                .await
            {
                Ok((name, schema)) => {
                    registered.push((name, schema));
                }
                Err(e) => {
                    eprintln!(
                        "Warning: failed to register table {}.{}.{}: {}",
                        catalog_name, schema_name, table_summary.name, e
                    );
                }
            }
        }

        Ok(registered)
    }

    async fn register_single_table(
        &self,
        ctx: &SessionContext,
        catalog_name: &str,
        schema_name: &str,
        table_name: &str,
    ) -> CatalogResult<(String, SchemaRef)> {
        let table_info = self
            .catalog
            .get_table(catalog_name, schema_name, table_name)
            .await?;
        let arrow_schema = self.catalog.table_to_arrow_schema(&table_info)?;
        let normalized_name = table_info.name.to_lowercase();

        // Find a reader for this format
        let reader = self.reader_for(&table_info.data_source_format);

        match reader {
            Some(r) => {
                r.register_table(ctx, &normalized_name, &table_info, arrow_schema.clone())
                    .await?;
            }
            None => {
                // No reader — register schema-only (empty MemTable for planning)
                let mem_table = MemTable::try_new(arrow_schema.clone(), vec![]).map_err(|e| {
                    CatalogError::Other(format!(
                        "Failed to create empty table '{}': {}",
                        normalized_name, e
                    ))
                })?;
                ctx.register_table(&normalized_name, Arc::new(mem_table))
                    .map_err(|e| {
                        CatalogError::Other(format!(
                            "Failed to register table '{}': {}",
                            normalized_name, e
                        ))
                    })?;
            }
        }

        Ok((normalized_name, arrow_schema))
    }
}
