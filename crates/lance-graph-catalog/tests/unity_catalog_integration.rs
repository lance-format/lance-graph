// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Integration tests for UnityCatalogProvider using wiremock to mock the REST API.

use std::collections::HashMap;

use lance_graph_catalog::{
    CatalogProvider, DataSourceFormat, TableType, UnityCatalogConfig, UnityCatalogProvider,
};
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn setup_provider(server: &MockServer) -> UnityCatalogProvider {
    let config = UnityCatalogConfig::new(server.uri());
    UnityCatalogProvider::new(config).unwrap()
}

// ---- list_catalogs ----

#[tokio::test]
async fn test_list_catalogs() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/catalogs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "catalogs": [
                { "name": "unity", "comment": "Main catalog" },
                { "name": "staging", "comment": null }
            ]
        })))
        .mount(&server)
        .await;

    let provider = setup_provider(&server).await;
    let catalogs = provider.list_catalogs().await.unwrap();

    assert_eq!(catalogs.len(), 2);
    assert_eq!(catalogs[0].name, "unity");
    assert_eq!(catalogs[0].comment.as_deref(), Some("Main catalog"));
    assert_eq!(catalogs[1].name, "staging");
    assert_eq!(catalogs[1].comment, None);
}

#[tokio::test]
async fn test_list_catalogs_empty() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/catalogs"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({ "catalogs": [] })),
        )
        .mount(&server)
        .await;

    let provider = setup_provider(&server).await;
    let catalogs = provider.list_catalogs().await.unwrap();
    assert!(catalogs.is_empty());
}

// ---- get_catalog ----

#[tokio::test]
async fn test_get_catalog() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/catalogs/unity"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "name": "unity",
            "comment": "Main catalog",
            "properties": { "owner": "admin" },
            "created_at": 1700000000000_i64,
            "updated_at": 1700000001000_i64
        })))
        .mount(&server)
        .await;

    let provider = setup_provider(&server).await;
    let catalog = provider.get_catalog("unity").await.unwrap();

    assert_eq!(catalog.name, "unity");
    assert_eq!(catalog.comment.as_deref(), Some("Main catalog"));
    assert_eq!(catalog.properties.get("owner").unwrap(), "admin");
    assert_eq!(catalog.created_at, Some(1700000000000));
}

#[tokio::test]
async fn test_get_catalog_not_found() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/catalogs/nonexistent"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let provider = setup_provider(&server).await;
    let err = provider.get_catalog("nonexistent").await.unwrap_err();
    assert!(err.to_string().contains("not found"));
}

// ---- list_schemas ----

#[tokio::test]
async fn test_list_schemas() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/schemas"))
        .and(query_param("catalog_name", "unity"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "schemas": [
                {
                    "name": "default",
                    "catalog_name": "unity",
                    "comment": "Default schema"
                },
                {
                    "name": "staging",
                    "catalog_name": "unity",
                    "comment": null
                }
            ]
        })))
        .mount(&server)
        .await;

    let provider = setup_provider(&server).await;
    let schemas = provider.list_schemas("unity").await.unwrap();

    assert_eq!(schemas.len(), 2);
    assert_eq!(schemas[0].name, "default");
    assert_eq!(schemas[0].catalog_name, "unity");
    assert_eq!(schemas[0].comment.as_deref(), Some("Default schema"));
}

// ---- get_schema ----

#[tokio::test]
async fn test_get_schema() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/schemas/unity.default"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "name": "default",
            "catalog_name": "unity",
            "comment": "Default schema"
        })))
        .mount(&server)
        .await;

    let provider = setup_provider(&server).await;
    let schema = provider.get_schema("unity", "default").await.unwrap();

    assert_eq!(schema.name, "default");
    assert_eq!(schema.catalog_name, "unity");
}

#[tokio::test]
async fn test_get_schema_not_found() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/schemas/unity.nonexistent"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let provider = setup_provider(&server).await;
    let err = provider.get_schema("unity", "nonexistent").await.unwrap_err();
    assert!(err.to_string().contains("not found"));
}

// ---- list_tables ----

#[tokio::test]
async fn test_list_tables() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/tables"))
        .and(query_param("catalog_name", "unity"))
        .and(query_param("schema_name", "default"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "tables": [
                {
                    "name": "marksheet",
                    "catalog_name": "unity",
                    "schema_name": "default",
                    "table_type": "MANAGED",
                    "data_source_format": "DELTA",
                    "columns": [],
                    "storage_location": "s3://bucket/marksheet",
                    "comment": "Student marks"
                },
                {
                    "name": "users",
                    "catalog_name": "unity",
                    "schema_name": "default",
                    "table_type": "EXTERNAL",
                    "data_source_format": "PARQUET",
                    "columns": [],
                    "storage_location": "/data/users.parquet"
                }
            ]
        })))
        .mount(&server)
        .await;

    let provider = setup_provider(&server).await;
    let tables = provider.list_tables("unity", "default").await.unwrap();

    assert_eq!(tables.len(), 2);
    assert_eq!(tables[0].name, "marksheet");
    assert_eq!(tables[0].table_type, TableType::Managed);
    assert_eq!(tables[0].data_source_format, DataSourceFormat::Delta);
    assert_eq!(
        tables[0].storage_location.as_deref(),
        Some("s3://bucket/marksheet")
    );
    assert_eq!(tables[0].comment.as_deref(), Some("Student marks"));

    assert_eq!(tables[1].name, "users");
    assert_eq!(tables[1].table_type, TableType::External);
    assert_eq!(tables[1].data_source_format, DataSourceFormat::Parquet);
}

// ---- get_table with columns ----

#[tokio::test]
async fn test_get_table_with_columns() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/tables/unity.default.marksheet"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "name": "marksheet",
            "catalog_name": "unity",
            "schema_name": "default",
            "table_type": "MANAGED",
            "data_source_format": "DELTA",
            "columns": [
                {
                    "name": "id",
                    "type_text": "INT",
                    "type_name": "INT",
                    "position": 0,
                    "nullable": false,
                    "comment": "Primary key"
                },
                {
                    "name": "name",
                    "type_text": "STRING",
                    "type_name": "STRING",
                    "position": 1,
                    "nullable": true,
                    "comment": "Student name"
                },
                {
                    "name": "mark",
                    "type_text": "DOUBLE",
                    "type_name": "DOUBLE",
                    "position": 2,
                    "nullable": true
                }
            ],
            "storage_location": "s3://bucket/marksheet",
            "comment": "Student marks",
            "properties": { "delta.minReaderVersion": "1" }
        })))
        .mount(&server)
        .await;

    let provider = setup_provider(&server).await;
    let table = provider
        .get_table("unity", "default", "marksheet")
        .await
        .unwrap();

    assert_eq!(table.name, "marksheet");
    assert_eq!(table.catalog_name, "unity");
    assert_eq!(table.schema_name, "default");
    assert_eq!(table.table_type, TableType::Managed);
    assert_eq!(table.data_source_format, DataSourceFormat::Delta);
    assert_eq!(table.columns.len(), 3);

    assert_eq!(table.columns[0].name, "id");
    assert_eq!(table.columns[0].type_name, "INT");
    assert_eq!(table.columns[0].position, 0);
    assert!(!table.columns[0].nullable);
    assert_eq!(table.columns[0].comment.as_deref(), Some("Primary key"));

    assert_eq!(table.columns[1].name, "name");
    assert_eq!(table.columns[1].type_name, "STRING");
    assert!(table.columns[1].nullable);

    assert_eq!(table.columns[2].name, "mark");
    assert_eq!(table.columns[2].type_name, "DOUBLE");
    assert_eq!(table.columns[2].comment, None);

    assert_eq!(
        table.properties.get("delta.minReaderVersion").unwrap(),
        "1"
    );
}

#[tokio::test]
async fn test_get_table_not_found() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/tables/unity.default.nonexistent"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let provider = setup_provider(&server).await;
    let err = provider
        .get_table("unity", "default", "nonexistent")
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not found"));
}

// ---- table_to_arrow_schema ----

#[tokio::test]
async fn test_table_to_arrow_schema() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/tables/unity.default.test_table"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "name": "test_table",
            "catalog_name": "unity",
            "schema_name": "default",
            "table_type": "MANAGED",
            "data_source_format": "PARQUET",
            "columns": [
                { "name": "id", "type_text": "LONG", "type_name": "LONG", "position": 0, "nullable": false },
                { "name": "value", "type_text": "DOUBLE", "type_name": "DOUBLE", "position": 1, "nullable": true },
                { "name": "label", "type_text": "STRING", "type_name": "STRING", "position": 2, "nullable": true }
            ],
            "storage_location": "/data/test_table"
        })))
        .mount(&server)
        .await;

    let provider = setup_provider(&server).await;
    let table = provider
        .get_table("unity", "default", "test_table")
        .await
        .unwrap();
    let schema = provider.table_to_arrow_schema(&table).unwrap();

    assert_eq!(schema.fields().len(), 3);
    assert_eq!(schema.field(0).name(), "id");
    assert_eq!(
        *schema.field(0).data_type(),
        arrow_schema::DataType::Int64
    );
    assert!(!schema.field(0).is_nullable());
    assert_eq!(schema.field(1).name(), "value");
    assert_eq!(
        *schema.field(1).data_type(),
        arrow_schema::DataType::Float64
    );
    assert_eq!(schema.field(2).name(), "label");
    assert_eq!(*schema.field(2).data_type(), arrow_schema::DataType::Utf8);
}

// ---- auth error ----

#[tokio::test]
async fn test_auth_error() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/catalogs"))
        .respond_with(ResponseTemplate::new(401).set_body_string("Unauthorized"))
        .mount(&server)
        .await;

    let provider = setup_provider(&server).await;
    let err = provider.list_catalogs().await.unwrap_err();
    assert!(err.to_string().contains("Auth error"));
}

// ---- bearer token is sent ----

#[tokio::test]
async fn test_bearer_token_sent() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/catalogs"))
        .and(wiremock::matchers::header("Authorization", "Bearer my-token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "catalogs": [{ "name": "unity" }]
        })))
        .mount(&server)
        .await;

    let config = UnityCatalogConfig::new(server.uri()).with_token("my-token");
    let provider = UnityCatalogProvider::new(config).unwrap();
    let catalogs = provider.list_catalogs().await.unwrap();

    assert_eq!(catalogs.len(), 1);
    assert_eq!(catalogs[0].name, "unity");
}

// ---- data source format parsing ----

#[tokio::test]
async fn test_data_source_format_parsing() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/tables"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "tables": [
                { "name": "t1", "catalog_name": "c", "schema_name": "s", "table_type": "MANAGED", "data_source_format": "DELTA" },
                { "name": "t2", "catalog_name": "c", "schema_name": "s", "table_type": "EXTERNAL", "data_source_format": "PARQUET" },
                { "name": "t3", "catalog_name": "c", "schema_name": "s", "table_type": "MANAGED", "data_source_format": "CSV" },
                { "name": "t4", "catalog_name": "c", "schema_name": "s", "table_type": "MANAGED", "data_source_format": "JSON" },
                { "name": "t5", "catalog_name": "c", "schema_name": "s", "table_type": "MANAGED", "data_source_format": "AVRO" },
                { "name": "t6", "catalog_name": "c", "schema_name": "s", "table_type": "MANAGED", "data_source_format": "ORC" },
                { "name": "t7", "catalog_name": "c", "schema_name": "s", "table_type": "MANAGED", "data_source_format": "CUSTOM_FORMAT" },
                { "name": "t8", "catalog_name": "c", "schema_name": "s", "table_type": "MANAGED" }
            ]
        })))
        .mount(&server)
        .await;

    let provider = setup_provider(&server).await;
    let tables = provider.list_tables("c", "s").await.unwrap();

    assert_eq!(tables[0].data_source_format, DataSourceFormat::Delta);
    assert_eq!(tables[1].data_source_format, DataSourceFormat::Parquet);
    assert_eq!(tables[2].data_source_format, DataSourceFormat::Csv);
    assert_eq!(tables[3].data_source_format, DataSourceFormat::Json);
    assert_eq!(tables[4].data_source_format, DataSourceFormat::Avro);
    assert_eq!(tables[5].data_source_format, DataSourceFormat::Orc);
    assert_eq!(
        tables[6].data_source_format,
        DataSourceFormat::Other("CUSTOM_FORMAT".to_string())
    );
    assert_eq!(
        tables[7].data_source_format,
        DataSourceFormat::Other("UNKNOWN".to_string())
    );
}

// ---- connection error ----

#[tokio::test]
async fn test_connection_error_on_bad_url() {
    let config = UnityCatalogConfig::new("http://localhost:1")
        .with_timeout(1);
    let provider = UnityCatalogProvider::new(config).unwrap();
    let err = provider.list_catalogs().await.unwrap_err();
    assert!(err.to_string().contains("connection error"));
}
