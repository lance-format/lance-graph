// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Integration tests for SqlQuery

use arrow_array::{Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance_graph::SqlQuery;
use std::collections::HashMap;
use std::sync::Arc;

fn person_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int64, false),
        Field::new("city", DataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Carol", "David"])),
            Arc::new(Int64Array::from(vec![28, 34, 29, 42])),
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

fn knows_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src_id", DataType::Int64, false),
        Field::new("dst_id", DataType::Int64, false),
        Field::new("since_year", DataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 1, 2, 3])),
            Arc::new(Int64Array::from(vec![2, 3, 4, 4])),
            Arc::new(Int64Array::from(vec![2015, 2018, 2020, 2021])),
        ],
    )
    .unwrap()
}

fn make_datasets() -> HashMap<String, RecordBatch> {
    let mut datasets = HashMap::new();
    datasets.insert("person".to_string(), person_batch());
    datasets.insert("knows".to_string(), knows_batch());
    datasets
}

// ============================================================================
// Basic SELECT with WHERE, ORDER BY, LIMIT
// ============================================================================

#[tokio::test]
async fn test_select_with_where_order_by_limit() {
    let query = SqlQuery::new("SELECT name, age FROM person WHERE age > 30 ORDER BY age LIMIT 10");
    let result = query.execute(make_datasets()).await.unwrap();

    let names: Vec<&str> = result
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|v| v.unwrap())
        .collect();

    assert_eq!(names, vec!["Bob", "David"]);

    let ages: Vec<i64> = result
        .column_by_name("age")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .iter()
        .map(|v| v.unwrap())
        .collect();

    assert_eq!(ages, vec![34, 42]);
}

#[tokio::test]
async fn test_select_star() {
    let query = SqlQuery::new("SELECT * FROM person ORDER BY id");
    let result = query.execute(make_datasets()).await.unwrap();
    assert_eq!(result.num_rows(), 4);
    assert_eq!(result.num_columns(), 4);
}

#[tokio::test]
async fn test_select_limit() {
    let query = SqlQuery::new("SELECT name FROM person ORDER BY name LIMIT 2");
    let result = query.execute(make_datasets()).await.unwrap();
    assert_eq!(result.num_rows(), 2);
}

// ============================================================================
// JOINs between node and relationship tables
// ============================================================================

#[tokio::test]
async fn test_inner_join() {
    let query = SqlQuery::new(
        "SELECT p.name, k.dst_id, k.since_year \
         FROM person p \
         JOIN knows k ON p.id = k.src_id \
         ORDER BY p.name, k.dst_id",
    );
    let result = query.execute(make_datasets()).await.unwrap();

    let names: Vec<&str> = result
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|v| v.unwrap())
        .collect();

    // Alice->2, Alice->3, Bob->4, Carol->4
    assert_eq!(names, vec!["Alice", "Alice", "Bob", "Carol"]);
}

#[tokio::test]
async fn test_self_join_friends() {
    let query = SqlQuery::new(
        "SELECT p1.name AS person, p2.name AS friend \
         FROM person p1 \
         JOIN knows k ON p1.id = k.src_id \
         JOIN person p2 ON p2.id = k.dst_id \
         ORDER BY p1.name, p2.name",
    );
    let result = query.execute(make_datasets()).await.unwrap();

    let persons: Vec<&str> = result
        .column_by_name("person")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|v| v.unwrap())
        .collect();
    let friends: Vec<&str> = result
        .column_by_name("friend")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|v| v.unwrap())
        .collect();

    assert_eq!(persons, vec!["Alice", "Alice", "Bob", "Carol"]);
    assert_eq!(friends, vec!["Bob", "Carol", "David", "David"]);
}

// ============================================================================
// Aggregations (COUNT, SUM, AVG)
// ============================================================================

#[tokio::test]
async fn test_count() {
    let query = SqlQuery::new("SELECT COUNT(*) AS cnt FROM person");
    let result = query.execute(make_datasets()).await.unwrap();

    let cnt = result
        .column_by_name("cnt")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(cnt, 4);
}

#[tokio::test]
async fn test_sum() {
    let query = SqlQuery::new("SELECT SUM(age) AS total_age FROM person");
    let result = query.execute(make_datasets()).await.unwrap();

    let total = result
        .column_by_name("total_age")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(total, 28 + 34 + 29 + 42);
}

#[tokio::test]
async fn test_avg() {
    let query = SqlQuery::new("SELECT AVG(age) AS avg_age FROM person");
    let result = query.execute(make_datasets()).await.unwrap();

    let avg = result
        .column_by_name("avg_age")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    assert!((avg - 33.25).abs() < 0.01);
}

#[tokio::test]
async fn test_group_by_with_count() {
    let query = SqlQuery::new(
        "SELECT city, COUNT(*) AS cnt FROM person GROUP BY city ORDER BY cnt DESC, city",
    );
    let result = query.execute(make_datasets()).await.unwrap();

    let cities: Vec<&str> = result
        .column_by_name("city")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|v| v.unwrap())
        .collect();
    let counts: Vec<i64> = result
        .column_by_name("cnt")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .iter()
        .map(|v| v.unwrap())
        .collect();

    assert_eq!(cities[0], "New York");
    assert_eq!(counts[0], 2);
}

// ============================================================================
// Execute with SessionContext (pre-registered tables)
// ============================================================================

#[tokio::test]
async fn test_execute_with_session_context() {
    use datafusion::datasource::MemTable;
    use datafusion::execution::context::SessionContext;

    let ctx = SessionContext::new();

    // Register person table
    let batch = person_batch();
    let schema = batch.schema();
    let mem_table = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table("people", mem_table).unwrap();

    // Register knows table
    let batch = knows_batch();
    let schema = batch.schema();
    let mem_table = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table("relationships", mem_table).unwrap();

    let query = SqlQuery::new(
        "SELECT p.name, r.since_year \
         FROM people p \
         JOIN relationships r ON p.id = r.src_id \
         ORDER BY p.name, r.since_year",
    );
    let result = query.execute_with_context(ctx).await.unwrap();

    assert_eq!(result.num_rows(), 4);

    let names: Vec<&str> = result
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|v| v.unwrap())
        .collect();
    assert_eq!(names[0], "Alice");
}

// ============================================================================
// Explain
// ============================================================================

#[tokio::test]
async fn test_explain_output() {
    let query = SqlQuery::new("SELECT p.name FROM person p JOIN knows k ON p.id = k.src_id");
    let plan = query.explain(make_datasets()).await.unwrap();
    assert!(plan.contains("Logical Plan"));
    assert!(plan.contains("Physical Plan"));
}

// ============================================================================
// Error handling
// ============================================================================

#[tokio::test]
async fn test_invalid_sql() {
    let query = SqlQuery::new("NOT VALID SQL");
    let result = query.execute(make_datasets()).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_missing_table() {
    let query = SqlQuery::new("SELECT * FROM nonexistent_table");
    let result = query.execute(make_datasets()).await;
    assert!(result.is_err());
}

// ============================================================================
// Case insensitivity (table names are lowercased)
// ============================================================================

#[tokio::test]
async fn test_case_insensitive_table_names() {
    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch());

    // Table registered as lowercase "person", so SQL should use lowercase
    let query = SqlQuery::new("SELECT name FROM person ORDER BY name LIMIT 1");
    let result = query.execute(datasets).await.unwrap();
    assert_eq!(result.num_rows(), 1);
}
