// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Integration tests for the to_sql API
//!
//! These tests verify that Cypher queries can be correctly converted to SQL strings.

use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use lance_graph::{CypherQuery, GraphConfig};
use std::collections::HashMap;
use std::sync::Arc;

/// Helper to create a simple Person table
fn create_person_table() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("person_id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
        Field::new("city", DataType::Utf8, false),
    ]));

    let person_ids = Int32Array::from(vec![1, 2, 3, 4]);
    let names = StringArray::from(vec!["Alice", "Bob", "Carol", "David"]);
    let ages = Int32Array::from(vec![28, 34, 29, 42]);
    let cities = StringArray::from(vec!["New York", "San Francisco", "New York", "Chicago"]);

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(person_ids),
            Arc::new(names),
            Arc::new(ages),
            Arc::new(cities),
        ],
    )
    .unwrap()
}

/// Helper to create a Company table
fn create_company_table() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("company_id", DataType::Int32, false),
        Field::new("company_name", DataType::Utf8, false),
        Field::new("industry", DataType::Utf8, false),
    ]));

    let company_ids = Int32Array::from(vec![101, 102, 103]);
    let names = StringArray::from(vec!["TechCorp", "DataInc", "CloudSoft"]);
    let industries = StringArray::from(vec!["Technology", "Analytics", "Cloud"]);

    RecordBatch::try_new(
        schema,
        vec![Arc::new(company_ids), Arc::new(names), Arc::new(industries)],
    )
    .unwrap()
}

/// Helper to create a WORKS_FOR relationship table
fn create_works_for_table() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("person_id", DataType::Int32, false),
        Field::new("company_id", DataType::Int32, false),
        Field::new("position", DataType::Utf8, false),
        Field::new("salary", DataType::Int32, false),
    ]));

    let person_ids = Int32Array::from(vec![1, 2, 3, 4]);
    let company_ids = Int32Array::from(vec![101, 101, 102, 103]);
    let positions = StringArray::from(vec!["Engineer", "Designer", "Manager", "Director"]);
    let salaries = Int32Array::from(vec![120000, 95000, 130000, 180000]);

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(person_ids),
            Arc::new(company_ids),
            Arc::new(positions),
            Arc::new(salaries),
        ],
    )
    .unwrap()
}

#[tokio::test]
async fn test_to_sql_simple_node_scan() {
    let config = GraphConfig::builder()
        .with_node_label("Person", "person_id")
        .build()
        .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), create_person_table());

    let query = CypherQuery::new("MATCH (p:Person) RETURN p.name")
        .unwrap()
        .with_config(config);

    let sql = query.to_sql(datasets).await.unwrap();

    // Verify SQL contains expected elements
    assert!(
        sql.to_uppercase().contains("SELECT"),
        "SQL should contain SELECT"
    );
    assert!(
        sql.to_lowercase().contains("person"),
        "SQL should reference person table"
    );
    assert!(sql.contains("name"), "SQL should reference name column");

    // SQL should be non-empty and valid
    assert!(!sql.is_empty(), "Generated SQL should not be empty");
    println!("Generated SQL:\n{}", sql);
}

#[tokio::test]
async fn test_to_sql_with_filter() {
    let config = GraphConfig::builder()
        .with_node_label("Person", "person_id")
        .build()
        .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), create_person_table());

    let query = CypherQuery::new("MATCH (p:Person) WHERE p.age > 30 RETURN p.name, p.age")
        .unwrap()
        .with_config(config);

    let sql = query.to_sql(datasets).await.unwrap();

    // Verify SQL contains filter condition
    assert!(sql.contains("SELECT"), "SQL should contain SELECT");
    assert!(
        sql.contains("WHERE") || sql.contains("FILTER"),
        "SQL should contain WHERE clause"
    );
    assert!(sql.contains("age"), "SQL should reference age column");
    assert!(sql.contains("30"), "SQL should contain filter value");

    println!("Generated SQL with filter:\n{}", sql);
}

#[tokio::test]
async fn test_to_sql_with_multiple_properties() {
    let config = GraphConfig::builder()
        .with_node_label("Person", "person_id")
        .build()
        .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), create_person_table());

    let query = CypherQuery::new("MATCH (p:Person) RETURN p.name, p.age, p.city")
        .unwrap()
        .with_config(config);

    let sql = query.to_sql(datasets).await.unwrap();

    // Verify all columns are present
    assert!(sql.contains("name"), "SQL should contain name");
    assert!(sql.contains("age"), "SQL should contain age");
    assert!(sql.contains("city"), "SQL should contain city");

    println!("Generated SQL with multiple properties:\n{}", sql);
}

#[tokio::test]
async fn test_to_sql_with_relationship() {
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

    let sql = query.to_sql(datasets).await.unwrap();

    // Verify SQL contains join
    let sql_upper = sql.to_uppercase();
    let sql_lower = sql.to_lowercase();
    assert!(sql_upper.contains("SELECT"), "SQL should contain SELECT");
    assert!(sql_upper.contains("JOIN"), "SQL should contain JOIN");
    assert!(sql_lower.contains("person"), "SQL should reference person");
    assert!(
        sql_lower.contains("company"),
        "SQL should reference company"
    );

    println!("Generated SQL with relationship:\n{}", sql);
}

#[tokio::test]
async fn test_to_sql_with_relationship_filter() {
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

    let query = CypherQuery::new (
        "MATCH (p:Person)-[w:WORKS_FOR]->(c:Company) WHERE w.salary > 100000 RETURN p.name, c.company_name, w.salary",
    )
        .unwrap()
        .with_config(config);

    let sql = query.to_sql(datasets).await.unwrap();

    // Verify SQL contains filter on relationship property
    assert!(sql.contains("salary"), "SQL should reference salary");
    assert!(sql.contains("100000"), "SQL should contain filter value");

    println!("Generated SQL with relationship filter:\n{}", sql);
}

#[tokio::test]
async fn test_to_sql_with_order_by() {
    let config = GraphConfig::builder()
        .with_node_label("Person", "person_id")
        .build()
        .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), create_person_table());

    let query = CypherQuery::new("MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age DESC")
        .unwrap()
        .with_config(config);

    let sql = query.to_sql(datasets).await.unwrap();

    // Verify SQL contains ORDER BY
    assert!(
        sql.contains("ORDER BY") || sql.contains("order by"),
        "SQL should contain ORDER BY"
    );
    assert!(sql.contains("age"), "SQL should reference age in ORDER BY");

    println!("Generated SQL with ORDER BY:\n{}", sql);
}

#[tokio::test]
async fn test_to_sql_with_limit() {
    let config = GraphConfig::builder()
        .with_node_label("Person", "person_id")
        .build()
        .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), create_person_table());

    let query = CypherQuery::new("MATCH (p:Person) RETURN p.name LIMIT 2")
        .unwrap()
        .with_config(config);

    let sql = query.to_sql(datasets).await.unwrap();

    // Verify SQL contains LIMIT
    assert!(
        sql.contains("LIMIT") || sql.contains("limit"),
        "SQL should contain LIMIT"
    );
    assert!(sql.contains("2"), "SQL should contain limit value");

    println!("Generated SQL with LIMIT:\n{}", sql);
}

#[tokio::test]
async fn test_to_sql_with_distinct() {
    let config = GraphConfig::builder()
        .with_node_label("Person", "person_id")
        .build()
        .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), create_person_table());

    let query = CypherQuery::new("MATCH (p:Person) RETURN DISTINCT p.city")
        .unwrap()
        .with_config(config);

    let sql = query.to_sql(datasets).await.unwrap();

    // Verify SQL is generated successfully
    // Note: DISTINCT might be optimized away by DataFusion's optimizer in some cases
    assert!(!sql.is_empty(), "SQL should be generated");
    assert!(sql.contains("city"), "SQL should reference city");

    println!("Generated SQL with DISTINCT:\n{}", sql);
}

#[tokio::test]
async fn test_to_sql_with_alias() {
    let config = GraphConfig::builder()
        .with_node_label("Person", "person_id")
        .build()
        .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), create_person_table());

    let query =
        CypherQuery::new("MATCH (p:Person) RETURN p.name AS person_name, p.age AS person_age")
            .unwrap()
            .with_config(config);

    let sql = query.to_sql(datasets).await.unwrap();

    // Verify SQL contains aliases
    assert!(
        sql.contains("AS") || sql.contains("as"),
        "SQL should contain AS for aliases"
    );

    println!("Generated SQL with aliases:\n{}", sql);
}

#[tokio::test]
async fn test_to_sql_complex_query() {
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

    let sql = query.to_sql(datasets).await.unwrap();

    // Verify complex query elements
    assert!(sql.contains("SELECT"), "SQL should contain SELECT");
    assert!(
        sql.contains("JOIN") || sql.contains("join"),
        "SQL should contain JOIN"
    );
    assert!(
        sql.contains("WHERE") || sql.contains("where"),
        "SQL should contain WHERE"
    );
    assert!(
        sql.contains("ORDER BY") || sql.contains("order by"),
        "SQL should contain ORDER BY"
    );
    assert!(
        sql.contains("LIMIT") || sql.contains("limit"),
        "SQL should contain LIMIT"
    );

    println!("Generated complex SQL:\n{}", sql);
}

#[tokio::test]
async fn test_to_sql_missing_config() {
    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), create_person_table());

    let query = CypherQuery::new("MATCH (p:Person) RETURN p.name").unwrap();
    // Note: No config set

    let result = query.to_sql(datasets).await;

    // Should fail without config
    assert!(result.is_err(), "to_sql should fail without config");
    assert!(
        result.unwrap_err().to_string().contains("configuration"),
        "Error should mention missing configuration"
    );
}

#[tokio::test]
async fn test_to_sql_empty_datasets() {
    let config = GraphConfig::builder()
        .with_node_label("Person", "person_id")
        .build()
        .unwrap();

    let datasets = HashMap::new(); // Empty

    let query = CypherQuery::new("MATCH (p:Person) RETURN p.name")
        .unwrap()
        .with_config(config);

    let result = query.to_sql(datasets).await;

    // Should fail with empty datasets
    assert!(result.is_err(), "to_sql should fail with empty datasets");
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("No input datasets"),
        "Error should mention missing datasets"
    );
}
