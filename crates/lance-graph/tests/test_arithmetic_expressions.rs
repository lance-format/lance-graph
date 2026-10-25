// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! End-to-end tests for arithmetic expressions (`+`, `-`, `*`, `/`, `%`) in RETURN and WHERE.

use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance_graph::config::GraphConfig;
use lance_graph::{CypherQuery, ExecutionStrategy, NodeMapping};
use std::collections::HashMap;
use std::sync::Arc;

// | id | name    | age |
// |----|---------|-----|
// | 1  | Bob     | 25  |
// | 2  | Alice   | 30  |
// | 3  | Charlie | 35  |
fn person_datasets() -> HashMap<String, RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["Bob", "Alice", "Charlie"])),
            Arc::new(Int64Array::from(vec![25, 30, 35])),
        ],
    )
    .unwrap();
    HashMap::from([("Person".to_string(), batch)])
}

fn config() -> GraphConfig {
    GraphConfig::builder()
        .with_node_mapping(NodeMapping {
            label: "Person".to_string(),
            id_field: "id".to_string(),
            property_fields: vec!["name".to_string(), "age".to_string()],
            filter_conditions: None,
        })
        .build()
        .unwrap()
}

async fn run(query: &str) -> RecordBatch {
    CypherQuery::new(query)
        .unwrap()
        .with_config(config())
        .execute(person_datasets(), Some(ExecutionStrategy::DataFusion))
        .await
        .unwrap()
}

fn int_column(batch: &RecordBatch, name: &str) -> Vec<i64> {
    batch
        .column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec()
}

#[tokio::test]
async fn test_return_arithmetic_with_precedence() {
    let result =
        run("MATCH (p:Person) RETURN p.age + p.id * 10 AS score, (p.age - 5) % 7 AS bucket ORDER BY p.id").await;
    assert_eq!(int_column(&result, "score"), vec![35, 50, 65]);
    assert_eq!(int_column(&result, "bucket"), vec![6, 4, 2]);
}

#[tokio::test]
async fn test_where_arithmetic_comparison() {
    let result =
        run("MATCH (p:Person) WHERE p.age / 5 - 1 >= 5 RETURN p.name ORDER BY p.name").await;
    let names = result
        .column_by_name("p.name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let names: Vec<&str> = names.iter().map(|n| n.unwrap()).collect();
    assert_eq!(names, vec!["Alice", "Charlie"]);
}
