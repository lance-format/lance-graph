// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Table reader trait for format-specific data reading.
//!
//! Inspired by Presto's `ConnectorPageSourceProvider`, this trait decouples
//! data format reading from catalog metadata. Each implementation handles
//! one or more data formats and is reusable across any [`CatalogProvider`].

use std::collections::HashMap;

use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::execution::context::SessionContext;

use crate::catalog_provider::{CatalogResult, DataSourceFormat, TableInfo};

/// Reads table data in a specific format and registers it into a DataFusion
/// `SessionContext`.
///
/// Analogous to Presto's `ConnectorPageSourceProvider` — decoupled from
/// catalog metadata so that format readers are reusable across any catalog.
///
/// # Extensibility
///
/// Implement this trait to add support for new data formats:
/// - Parquet (provided)
/// - Delta Lake (provided, behind `delta` feature)
/// - CSV (future)
/// - Iceberg (future)
/// - ORC (future)
#[async_trait]
pub trait TableReader: Send + Sync {
    /// Human-readable name of this reader (e.g., "parquet", "delta").
    fn name(&self) -> &str;

    /// The data format(s) this reader can handle.
    fn supported_formats(&self) -> &[DataSourceFormat];

    /// Register a table into a DataFusion `SessionContext` using its storage
    /// location.
    ///
    /// The reader should read (or reference) the data at `table_info.storage_location`
    /// and register it as a DataFusion `TableProvider` so it can be queried via SQL.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The DataFusion session context to register the table in.
    /// * `table_name` - The name to register the table under (already lowercased).
    /// * `table_info` - Full table metadata from the catalog, including `storage_location`.
    /// * `schema` - Arrow schema derived from the table's column definitions.
    /// * `storage_options` - Key-value pairs for cloud storage credentials
    ///   (e.g., `azure_storage_account_name`, `aws_access_key_id`, etc.).
    async fn register_table(
        &self,
        ctx: &SessionContext,
        table_name: &str,
        table_info: &TableInfo,
        schema: SchemaRef,
        storage_options: &HashMap<String, String>,
    ) -> CatalogResult<()>;
}
