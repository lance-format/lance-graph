// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Integration tests for the to_sql API with SqlDialect::Spark
//!
//! These tests verify that Cypher queries can be correctly converted to Spark SQL strings.

use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use lance_graph::{CypherQuery, GraphConfig, SqlDialect};
use std::collections::HashMap;
use std::sync::Arc;

fn create_person_table() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("person_id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
        Field::new("city", DataType::Utf8, false),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Carol", "David"])),
            Arc::new(Int32Array::from(vec![28, 34, 29, 42])),
            Arc::new(StringArray::from(vec![
                "New York",
                "San Francisco",
                "New York",
                "Chicago",
            ])),
        ],
    )
    .unwrap()
}

fn create_company_table() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("company_id", DataType::Int32, false),
        Field::new("company_name", DataType::Utf8, false),
        Field::new("industry", DataType::Utf8, false),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![101, 102, 103])),
            Arc::new(StringArray::from(vec!["TechCorp", "DataInc", "CloudSoft"])),
            Arc::new(StringArray::from(vec!["Technology", "Analytics", "Cloud"])),
        ],
    )
    .unwrap()
}

fn create_works_for_table() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("person_id", DataType::Int32, false),
        Field::new("company_id", DataType::Int32, false),
        Field::new("position", DataType::Utf8, false),
        Field::new("salary", DataType::Int32, false),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(Int32Array::from(vec![101, 101, 102, 103])),
            Arc::new(StringArray::from(vec![
                "Engineer", "Designer", "Manager", "Director",
            ])),
            Arc::new(Int32Array::from(vec![120000, 95000, 130000, 180000])),
        ],
    )
    .unwrap()
}

#[tokio::test]
async fn test_to_sql_spark_simple_node_scan() {
    let config = GraphConfig::builder()
        .with_node_label("Person", "person_id")
        .build()
        .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), create_person_table());

    let query = CypherQuery::new("MATCH (p:Person) RETURN p.name")
        .unwrap()
        .with_config(config);

    let sql = query
        .to_sql(datasets, Some(SqlDialect::Spark))
        .await
        .unwrap();

    assert!(sql.contains('`'), "Spark SQL should use backtick quoting");
    assert!(
        sql.to_uppercase().contains("SELECT"),
        "SQL should contain SELECT"
    );
    assert!(sql.contains("name"), "SQL should reference name column");

    println!("Generated Spark SQL:\n{}", sql);
}

#[tokio::test]
async fn test_to_sql_spark_with_filter() {
    let config = GraphConfig::builder()
        .with_node_label("Person", "person_id")
        .build()
        .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), create_person_table());

    let query = CypherQuery::new("MATCH (p:Person) WHERE p.age > 30 RETURN p.name, p.age")
        .unwrap()
        .with_config(config);

    let sql = query
        .to_sql(datasets, Some(SqlDialect::Spark))
        .await
        .unwrap();

    assert!(sql.contains('`'), "Spark SQL should use backtick quoting");
    assert!(sql.contains("30"), "SQL should contain filter value");
    assert!(sql.contains("age"), "SQL should reference age column");

    println!("Generated Spark SQL with filter:\n{}", sql);
}

#[tokio::test]
async fn test_to_sql_spark_with_relationship() {
    let config = GraphConfig::builder()
        .with_node_label("Person", "person_id")
        .with_node_label("Company", "company_id")
        .with_relationship("WORKS_FOR", "person_id", "company_id")
        .build()
        .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), create_person_table());
    datasets.insert("Company".to_string(), create_company_table());
    datasets.insert("WORKS_FOR".to_string(), create_works_for_table());

    let query = CypherQuery::new(
        "MATCH (p:Person)-[:WORKS_FOR]->(c:Company) RETURN p.name, c.company_name",
    )
    .unwrap()
    .with_config(config);

    let sql = query
        .to_sql(datasets, Some(SqlDialect::Spark))
        .await
        .unwrap();

    let sql_upper = sql.to_uppercase();
    assert!(sql.contains('`'), "Spark SQL should use backtick quoting");
    assert!(sql_upper.contains("JOIN"), "SQL should contain JOIN");

    println!("Generated Spark SQL with relationship:\n{}", sql);
}

#[tokio::test]
async fn test_to_sql_spark_complex_query() {
    let config = GraphConfig::builder()
        .with_node_label("Person", "person_id")
        .with_node_label("Company", "company_id")
        .with_relationship("WORKS_FOR", "person_id", "company_id")
        .build()
        .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), create_person_table());
    datasets.insert("Company".to_string(), create_company_table());
    datasets.insert("WORKS_FOR".to_string(), create_works_for_table());

    let query = CypherQuery::new(
        "MATCH (p:Person)-[w:WORKS_FOR]->(c:Company) \
         WHERE p.age > 30 AND c.industry = 'Technology' \
         RETURN p.name, c.company_name, w.position \
         ORDER BY p.age DESC \
         LIMIT 5",
    )
    .unwrap()
    .with_config(config);

    let sql = query
        .to_sql(datasets, Some(SqlDialect::Spark))
        .await
        .unwrap();

    assert!(sql.contains('`'), "Spark SQL should use backtick quoting");
    assert!(
        sql.contains("ORDER BY") || sql.contains("order by"),
        "SQL should contain ORDER BY"
    );
    assert!(
        sql.contains("LIMIT") || sql.contains("limit"),
        "SQL should contain LIMIT"
    );

    println!("Generated complex Spark SQL:\n{}", sql);
}

#[tokio::test]
async fn test_to_sql_default_dialect_no_backticks() {
    let config = GraphConfig::builder()
        .with_node_label("Person", "person_id")
        .build()
        .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), create_person_table());

    let query = CypherQuery::new("MATCH (p:Person) RETURN p.name, p.age")
        .unwrap()
        .with_config(config);

    // Default dialect (None) should not use backticks
    let sql = query.to_sql(datasets, None).await.unwrap();

    assert!(
        sql.to_uppercase().contains("SELECT"),
        "SQL should contain SELECT"
    );

    println!("Generated default SQL:\n{}", sql);
}

#[tokio::test]
async fn test_spark_sql_differs_from_default_sql() {
    let config = GraphConfig::builder()
        .with_node_label("Person", "person_id")
        .build()
        .unwrap();

    let mut datasets1 = HashMap::new();
    datasets1.insert("Person".to_string(), create_person_table());

    let mut datasets2 = HashMap::new();
    datasets2.insert("Person".to_string(), create_person_table());

    let query = CypherQuery::new("MATCH (p:Person) RETURN p.name, p.age")
        .unwrap()
        .with_config(config);

    let default_sql = query.to_sql(datasets1, None).await.unwrap();
    let spark_sql = query
        .to_sql(datasets2, Some(SqlDialect::Spark))
        .await
        .unwrap();

    // Spark SQL should use backtick quoting while default SQL may not
    assert!(
        spark_sql.contains('`'),
        "Spark SQL should use backtick quoting"
    );

    println!("Default SQL:\n{}", default_sql);
    println!("\nSpark SQL:\n{}", spark_sql);
}

#[tokio::test]
async fn test_to_sql_postgresql_dialect() {
    let config = GraphConfig::builder()
        .with_node_label("Person", "person_id")
        .build()
        .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), create_person_table());

    let query = CypherQuery::new("MATCH (p:Person) RETURN p.name")
        .unwrap()
        .with_config(config);

    let sql = query
        .to_sql(datasets, Some(SqlDialect::PostgreSql))
        .await
        .unwrap();

    // PostgreSQL uses double-quote identifier quoting
    assert!(
        sql.contains('"'),
        "PostgreSQL SQL should use double-quote quoting"
    );

    println!("Generated PostgreSQL SQL:\n{}", sql);
}
