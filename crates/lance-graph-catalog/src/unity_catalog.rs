// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Unity Catalog REST API client implementing the [`CatalogProvider`] trait.
//!
//! Connects to an OSS Unity Catalog server and provides catalog browsing
//! capabilities. This is the first `CatalogProvider` implementation.

use std::collections::HashMap;

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;

use crate::catalog_provider::*;

/// Configuration for connecting to a Unity Catalog server.
#[derive(Debug, Clone)]
pub struct UnityCatalogConfig {
    /// Base URL of the UC server (e.g., `http://localhost:8080/api/2.1/unity-catalog`).
    pub base_url: String,
    /// Optional bearer token for authenticated access.
    pub bearer_token: Option<String>,
    /// Optional request timeout in seconds (default: 30).
    pub timeout_secs: Option<u64>,
}

impl UnityCatalogConfig {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            bearer_token: None,
            timeout_secs: None,
        }
    }

    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.bearer_token = Some(token.into());
        self
    }

    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = Some(secs);
        self
    }
}

/// Unity Catalog REST API client.
pub struct UnityCatalogProvider {
    config: UnityCatalogConfig,
    client: Client,
}

impl UnityCatalogProvider {
    pub fn new(config: UnityCatalogConfig) -> CatalogResult<Self> {
        let mut builder = Client::builder();
        if let Some(timeout) = config.timeout_secs {
            builder = builder.timeout(std::time::Duration::from_secs(timeout));
        }
        let client = builder
            .build()
            .map_err(|e| CatalogError::ConnectionError(format!("Failed to build HTTP client: {}", e)))?;

        Ok(Self { config, client })
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.config.base_url, path);
        let mut req = self.client.request(method, &url);
        if let Some(ref token) = self.config.bearer_token {
            req = req.bearer_auth(token);
        }
        req
    }

    async fn handle_response<T: serde::de::DeserializeOwned>(
        &self,
        resp: reqwest::Response,
        resource_name: &str,
    ) -> CatalogResult<T> {
        let status = resp.status();

        if status == reqwest::StatusCode::NOT_FOUND {
            return Err(CatalogError::NotFound(format!(
                "{} not found",
                resource_name
            )));
        }
        if status == reqwest::StatusCode::UNAUTHORIZED
            || status == reqwest::StatusCode::FORBIDDEN
        {
            let body = resp.text().await.unwrap_or_default();
            return Err(CatalogError::AuthError(format!(
                "HTTP {}: {}",
                status, body
            )));
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(CatalogError::ConnectionError(format!(
                "HTTP {}: {}",
                status, body
            )));
        }

        resp.json::<T>()
            .await
            .map_err(|e| CatalogError::InvalidResponse(e.to_string()))
    }
}

// ---- Serde models for UC REST API JSON responses ----

#[derive(Deserialize)]
struct ListCatalogsResponse {
    #[serde(default)]
    catalogs: Vec<UcCatalog>,
}

#[derive(Deserialize)]
struct UcCatalog {
    name: String,
    comment: Option<String>,
    #[serde(default)]
    properties: HashMap<String, String>,
    created_at: Option<i64>,
    updated_at: Option<i64>,
}

#[derive(Deserialize)]
struct ListSchemasResponse {
    #[serde(default)]
    schemas: Vec<UcSchema>,
}

#[derive(Deserialize)]
struct UcSchema {
    name: String,
    catalog_name: String,
    comment: Option<String>,
    #[serde(default)]
    properties: HashMap<String, String>,
    created_at: Option<i64>,
    updated_at: Option<i64>,
}

#[derive(Deserialize)]
struct ListTablesResponse {
    #[serde(default)]
    tables: Vec<UcTable>,
}

#[derive(Deserialize)]
struct UcTable {
    name: String,
    catalog_name: String,
    schema_name: String,
    table_type: String,
    data_source_format: Option<String>,
    #[serde(default)]
    columns: Vec<UcColumn>,
    storage_location: Option<String>,
    comment: Option<String>,
    #[serde(default)]
    properties: HashMap<String, String>,
    created_at: Option<i64>,
    updated_at: Option<i64>,
}

#[derive(Deserialize)]
struct UcColumn {
    name: String,
    type_text: String,
    type_name: String,
    position: i32,
    #[serde(default = "default_nullable")]
    nullable: bool,
    comment: Option<String>,
}

fn default_nullable() -> bool {
    true
}

// ---- Conversion helpers ----

impl From<UcCatalog> for CatalogInfo {
    fn from(uc: UcCatalog) -> Self {
        CatalogInfo {
            name: uc.name,
            comment: uc.comment,
            properties: uc.properties,
            created_at: uc.created_at,
            updated_at: uc.updated_at,
        }
    }
}

impl From<UcSchema> for SchemaInfo {
    fn from(uc: UcSchema) -> Self {
        SchemaInfo {
            name: uc.name,
            catalog_name: uc.catalog_name,
            comment: uc.comment,
            properties: uc.properties,
            created_at: uc.created_at,
            updated_at: uc.updated_at,
        }
    }
}

impl From<UcTable> for TableInfo {
    fn from(uc: UcTable) -> Self {
        TableInfo {
            name: uc.name,
            catalog_name: uc.catalog_name,
            schema_name: uc.schema_name,
            table_type: match uc.table_type.as_str() {
                "EXTERNAL" => TableType::External,
                _ => TableType::Managed,
            },
            data_source_format: match uc.data_source_format.as_deref() {
                Some("DELTA") => DataSourceFormat::Delta,
                Some("PARQUET") => DataSourceFormat::Parquet,
                Some("CSV") => DataSourceFormat::Csv,
                Some("JSON") => DataSourceFormat::Json,
                Some("AVRO") => DataSourceFormat::Avro,
                Some("ORC") => DataSourceFormat::Orc,
                Some("TEXT") => DataSourceFormat::Text,
                Some(other) => DataSourceFormat::Other(other.to_string()),
                None => DataSourceFormat::Other("UNKNOWN".to_string()),
            },
            columns: uc.columns.into_iter().map(Into::into).collect(),
            storage_location: uc.storage_location,
            comment: uc.comment,
            properties: uc.properties,
            created_at: uc.created_at,
            updated_at: uc.updated_at,
        }
    }
}

impl From<UcColumn> for ColumnInfo {
    fn from(uc: UcColumn) -> Self {
        ColumnInfo {
            name: uc.name,
            type_text: uc.type_text,
            type_name: uc.type_name,
            position: uc.position,
            nullable: uc.nullable,
            comment: uc.comment,
        }
    }
}

// ---- CatalogProvider implementation ----

#[async_trait]
impl CatalogProvider for UnityCatalogProvider {
    fn name(&self) -> &str {
        "unity-catalog"
    }

    async fn list_catalogs(&self) -> CatalogResult<Vec<CatalogInfo>> {
        let resp = self
            .request(reqwest::Method::GET, "/catalogs")
            .send()
            .await
            .map_err(|e| CatalogError::ConnectionError(e.to_string()))?;

        let body: ListCatalogsResponse = self.handle_response(resp, "catalogs").await?;
        Ok(body.catalogs.into_iter().map(Into::into).collect())
    }

    async fn get_catalog(&self, name: &str) -> CatalogResult<CatalogInfo> {
        let resp = self
            .request(reqwest::Method::GET, &format!("/catalogs/{}", name))
            .send()
            .await
            .map_err(|e| CatalogError::ConnectionError(e.to_string()))?;

        let body: UcCatalog = self
            .handle_response(resp, &format!("catalog '{}'", name))
            .await?;
        Ok(body.into())
    }

    async fn list_schemas(&self, catalog_name: &str) -> CatalogResult<Vec<SchemaInfo>> {
        let resp = self
            .request(reqwest::Method::GET, "/schemas")
            .query(&[("catalog_name", catalog_name)])
            .send()
            .await
            .map_err(|e| CatalogError::ConnectionError(e.to_string()))?;

        let body: ListSchemasResponse = self
            .handle_response(resp, &format!("schemas in '{}'", catalog_name))
            .await?;
        Ok(body.schemas.into_iter().map(Into::into).collect())
    }

    async fn get_schema(
        &self,
        catalog_name: &str,
        schema_name: &str,
    ) -> CatalogResult<SchemaInfo> {
        let full_name = format!("{}.{}", catalog_name, schema_name);
        let resp = self
            .request(reqwest::Method::GET, &format!("/schemas/{}", full_name))
            .send()
            .await
            .map_err(|e| CatalogError::ConnectionError(e.to_string()))?;

        let body: UcSchema = self
            .handle_response(resp, &format!("schema '{}'", full_name))
            .await?;
        Ok(body.into())
    }

    async fn list_tables(
        &self,
        catalog_name: &str,
        schema_name: &str,
    ) -> CatalogResult<Vec<TableInfo>> {
        let resp = self
            .request(reqwest::Method::GET, "/tables")
            .query(&[
                ("catalog_name", catalog_name),
                ("schema_name", schema_name),
            ])
            .send()
            .await
            .map_err(|e| CatalogError::ConnectionError(e.to_string()))?;

        let body: ListTablesResponse = self
            .handle_response(
                resp,
                &format!("tables in '{}.{}'", catalog_name, schema_name),
            )
            .await?;
        Ok(body.tables.into_iter().map(Into::into).collect())
    }

    async fn get_table(
        &self,
        catalog_name: &str,
        schema_name: &str,
        table_name: &str,
    ) -> CatalogResult<TableInfo> {
        let full_name = format!("{}.{}.{}", catalog_name, schema_name, table_name);
        let resp = self
            .request(reqwest::Method::GET, &format!("/tables/{}", full_name))
            .send()
            .await
            .map_err(|e| CatalogError::ConnectionError(e.to_string()))?;

        let body: UcTable = self
            .handle_response(resp, &format!("table '{}'", full_name))
            .await?;
        Ok(body.into())
    }
}
