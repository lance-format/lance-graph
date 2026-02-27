// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Catalog and namespace utilities for Lance Graph.
//!
//! This crate provides the SPI (Service Provider Interface) layer for external
//! catalog integration, inspired by Presto's connector architecture:
//!
//! - [`CatalogProvider`] — browse catalog metadata (analogous to `ConnectorMetadata`)
//! - [`TableReader`] — read table data in specific formats (analogous to `ConnectorPageSourceProvider`)
//! - [`Connector`] — bundles catalog + readers (analogous to Presto's `Connector`)

pub mod catalog_provider;
pub mod connector;
pub mod namespace;
pub mod source_catalog;
pub mod table_reader;
pub mod type_mapping;
#[cfg(feature = "unity-catalog")]
pub mod unity_catalog;

// Existing exports
pub use namespace::DirNamespace;
pub use source_catalog::{GraphSourceCatalog, InMemoryCatalog, SimpleTableSource};

// Catalog provider exports
pub use catalog_provider::{
    CatalogError, CatalogInfo, CatalogProvider, CatalogResult, ColumnInfo, DataSourceFormat,
    SchemaInfo, TableInfo, TableType,
};
pub use connector::Connector;
pub use table_reader::TableReader;
pub use type_mapping::columns_to_arrow_schema;

#[cfg(feature = "unity-catalog")]
pub use unity_catalog::{UnityCatalogConfig, UnityCatalogProvider};
