// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Built-in [`TableReader`] implementations for common data formats.
//!
//! - [`ParquetTableReader`] — reads Parquet tables using DataFusion's built-in support.
//! - [`DeltaTableReader`] — reads Delta Lake tables (behind `delta` feature flag).

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::execution::context::SessionContext;

use lance_graph_catalog::catalog_provider::{
    CatalogError, CatalogResult, DataSourceFormat, TableInfo,
};
use lance_graph_catalog::table_reader::TableReader;

/// Reads Parquet tables using DataFusion's built-in `register_parquet()`.
pub struct ParquetTableReader;

#[async_trait]
impl TableReader for ParquetTableReader {
    fn name(&self) -> &str {
        "parquet"
    }

    fn supported_formats(&self) -> &[DataSourceFormat] {
        &[DataSourceFormat::Parquet]
    }

    async fn register_table(
        &self,
        ctx: &SessionContext,
        table_name: &str,
        table_info: &TableInfo,
        _schema: arrow_schema::SchemaRef,
    ) -> CatalogResult<()> {
        let location = table_info.storage_location.as_deref().ok_or_else(|| {
            CatalogError::Other(format!("Table '{}' has no storage_location", table_name))
        })?;

        ctx.register_parquet(
            table_name,
            location,
            datafusion::datasource::file_format::options::ParquetReadOptions::default(),
        )
        .await
        .map_err(|e| {
            CatalogError::Other(format!(
                "Failed to register Parquet table '{}' at '{}': {}",
                table_name, location, e
            ))
        })
    }
}

/// Reads Delta Lake tables using the `deltalake` crate.
///
/// Opens the Delta table at the storage location and registers it as a
/// DataFusion `TableProvider`, enabling full SQL query support including
/// time travel, schema evolution, and partition pruning.
#[cfg(feature = "delta")]
pub struct DeltaTableReader;

#[cfg(feature = "delta")]
#[async_trait]
impl TableReader for DeltaTableReader {
    fn name(&self) -> &str {
        "delta"
    }

    fn supported_formats(&self) -> &[DataSourceFormat] {
        &[DataSourceFormat::Delta]
    }

    async fn register_table(
        &self,
        ctx: &SessionContext,
        table_name: &str,
        table_info: &TableInfo,
        _schema: arrow_schema::SchemaRef,
    ) -> CatalogResult<()> {
        let location = table_info.storage_location.as_deref().ok_or_else(|| {
            CatalogError::Other(format!("Table '{}' has no storage_location", table_name))
        })?;

        let table_url = url::Url::parse(location).map_err(|e| {
            CatalogError::Other(format!(
                "Invalid storage location URL '{}': {}",
                location, e
            ))
        })?;

        let delta_table = deltalake::open_table(table_url).await.map_err(|e| {
            CatalogError::Other(format!(
                "Failed to open Delta table '{}' at '{}': {}",
                table_name, location, e
            ))
        })?;

        ctx.register_table(table_name, Arc::new(delta_table))
            .map_err(|e| {
                CatalogError::Other(format!(
                    "Failed to register Delta table '{}': {}",
                    table_name, e
                ))
            })?;

        Ok(())
    }
}

/// Returns the default set of table readers.
///
/// Includes Parquet support, and Delta Lake support when the `delta` feature is enabled.
pub fn default_table_readers() -> Vec<Arc<dyn TableReader>> {
    let mut readers: Vec<Arc<dyn TableReader>> = vec![Arc::new(ParquetTableReader)];
    #[cfg(feature = "delta")]
    readers.push(Arc::new(DeltaTableReader));
    readers
}
