// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Catalog provider trait and data types for external catalog integration.
//!
//! Inspired by Presto's `ConnectorMetadata` SPI, this module defines the
//! abstract interface for browsing external catalogs (Unity Catalog, Hive
//! Metastore, AWS Glue, etc.).

use std::collections::HashMap;

use arrow_schema::SchemaRef;
use async_trait::async_trait;

/// Metadata about a catalog (top-level namespace).
#[derive(Debug, Clone)]
pub struct CatalogInfo {
    pub name: String,
    pub comment: Option<String>,
    pub properties: HashMap<String, String>,
    pub created_at: Option<i64>,
    pub updated_at: Option<i64>,
}

/// Metadata about a schema (second-level namespace within a catalog).
#[derive(Debug, Clone)]
pub struct SchemaInfo {
    pub name: String,
    pub catalog_name: String,
    pub comment: Option<String>,
    pub properties: HashMap<String, String>,
    pub created_at: Option<i64>,
    pub updated_at: Option<i64>,
}

/// Metadata about a column in a table.
#[derive(Debug, Clone)]
pub struct ColumnInfo {
    pub name: String,
    /// Human-readable type string (e.g., "INT", "VARCHAR(255)").
    pub type_text: String,
    /// Canonical type name from the catalog (e.g., "INT", "STRING").
    pub type_name: String,
    /// Column position (0-based).
    pub position: i32,
    pub nullable: bool,
    pub comment: Option<String>,
}

/// Data format of the underlying storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataSourceFormat {
    Delta,
    Parquet,
    Csv,
    Json,
    Avro,
    Orc,
    Text,
    Other(String),
}

/// Type of table (managed vs external).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TableType {
    Managed,
    External,
}

/// Full table metadata including columns and storage information.
#[derive(Debug, Clone)]
pub struct TableInfo {
    pub name: String,
    pub catalog_name: String,
    pub schema_name: String,
    pub table_type: TableType,
    pub data_source_format: DataSourceFormat,
    pub columns: Vec<ColumnInfo>,
    pub storage_location: Option<String>,
    pub comment: Option<String>,
    pub properties: HashMap<String, String>,
    pub created_at: Option<i64>,
    pub updated_at: Option<i64>,
}

/// Errors that can occur during catalog operations.
#[derive(Debug)]
pub enum CatalogError {
    /// Network or HTTP error.
    ConnectionError(String),
    /// Resource not found (catalog, schema, or table).
    NotFound(String),
    /// Authentication or authorization failure.
    AuthError(String),
    /// Invalid or unparsable response from the catalog server.
    InvalidResponse(String),
    /// Failed to map a catalog type to an Arrow type.
    TypeMappingError(String),
    /// Other errors.
    Other(String),
}

impl std::fmt::Display for CatalogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionError(msg) => write!(f, "Catalog connection error: {}", msg),
            Self::NotFound(msg) => write!(f, "Not found: {}", msg),
            Self::AuthError(msg) => write!(f, "Auth error: {}", msg),
            Self::InvalidResponse(msg) => write!(f, "Invalid response: {}", msg),
            Self::TypeMappingError(msg) => write!(f, "Type mapping error: {}", msg),
            Self::Other(msg) => write!(f, "Catalog error: {}", msg),
        }
    }
}

impl std::error::Error for CatalogError {}

pub type CatalogResult<T> = std::result::Result<T, CatalogError>;

/// Abstract trait for browsing an external catalog.
///
/// Analogous to Presto's `ConnectorMetadata`. Implementations provide access
/// to catalog metadata (catalogs, schemas, tables, columns) without being
/// coupled to any specific data format or storage backend.
///
/// # Extensibility
///
/// Implement this trait to add support for new catalog backends:
/// - Unity Catalog (provided)
/// - Hive Metastore (future)
/// - AWS Glue (future)
/// - Iceberg REST Catalog (future)
#[async_trait]
pub trait CatalogProvider: Send + Sync {
    /// Human-readable name of this catalog provider (e.g., "unity-catalog").
    fn name(&self) -> &str;

    /// List all catalogs available in this provider.
    async fn list_catalogs(&self) -> CatalogResult<Vec<CatalogInfo>>;

    /// Get information about a specific catalog.
    async fn get_catalog(&self, name: &str) -> CatalogResult<CatalogInfo>;

    /// List all schemas within a catalog.
    async fn list_schemas(&self, catalog_name: &str) -> CatalogResult<Vec<SchemaInfo>>;

    /// Get information about a specific schema.
    async fn get_schema(&self, catalog_name: &str, schema_name: &str) -> CatalogResult<SchemaInfo>;

    /// List all tables within a schema.
    async fn list_tables(
        &self,
        catalog_name: &str,
        schema_name: &str,
    ) -> CatalogResult<Vec<TableInfo>>;

    /// Get detailed information about a specific table, including columns.
    async fn get_table(
        &self,
        catalog_name: &str,
        schema_name: &str,
        table_name: &str,
    ) -> CatalogResult<TableInfo>;

    /// Convert a table's column definitions to an Arrow schema.
    ///
    /// The default implementation uses the standard type mapping from
    /// [`crate::type_mapping::columns_to_arrow_schema`].
    fn table_to_arrow_schema(&self, table: &TableInfo) -> CatalogResult<SchemaRef> {
        crate::type_mapping::columns_to_arrow_schema(&table.columns)
    }
}
