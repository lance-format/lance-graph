// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! End-to-end parity tests: native CSR expand vs DataFusion join path.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use lance_graph::config::GraphConfig;
use lance_graph::query::{CypherQuery, ExecutionStrategy};

fn person_batch() -> RecordBatch {
    // Dense ids 0..4 (row id == id_field value).
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from(vec![0u64, 1, 2, 3])),
            Arc::new(StringArray::from(vec!["alice", "bob", "carol", "dave"])),
            Arc::new(Int64Array::from(vec![30i64, 40, 25, 50])),
        ],
    )
    .unwrap()
}

fn knows_batch() -> RecordBatch {
    // 0->1, 0->2, 1->3, 2->3
    let schema = Arc::new(Schema::new(vec![
        Field::new("src_id", DataType::UInt64, false),
        Field::new("dst_id", DataType::UInt64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from(vec![0u64, 0, 1, 2])),
            Arc::new(UInt64Array::from(vec![1u64, 2, 3, 3])),
        ],
    )
    .unwrap()
}

fn config() -> GraphConfig {
    GraphConfig::builder()
        .with_node_label("Person", "id")
        .with_relationship("KNOWS", "src_id", "dst_id")
        .build()
        .unwrap()
}

fn datasets() -> HashMap<String, RecordBatch> {
    let mut d = HashMap::new();
    d.insert("Person".to_string(), person_batch());
    d.insert("KNOWS".to_string(), knows_batch());
    d
}

/// Collect (a.name, b.name) rows as a sorted Vec for order-independent compare.
fn name_pairs(batch: &RecordBatch) -> Vec<(String, String)> {
    let cols: Vec<&StringArray> = (0..batch.num_columns())
        .map(|i| {
            batch
                .column(i)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
        })
        .collect();
    let mut rows: Vec<(String, String)> = (0..batch.num_rows())
        .map(|r| (cols[0].value(r).to_string(), cols[1].value(r).to_string()))
        .collect();
    rows.sort();
    rows
}

#[tokio::test]
async fn test_native_expand_matches_datafusion_names() {
    let q = "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name";
    let query = CypherQuery::new(q).unwrap().with_config(config());

    let native = query
        .execute(datasets(), Some(ExecutionStrategy::LanceNative))
        .await
        .unwrap();
    let df = query
        .execute(datasets(), Some(ExecutionStrategy::DataFusion))
        .await
        .unwrap();

    let expected = vec![
        ("alice".to_string(), "bob".to_string()),
        ("alice".to_string(), "carol".to_string()),
        ("bob".to_string(), "dave".to_string()),
        ("carol".to_string(), "dave".to_string()),
    ];
    assert_eq!(name_pairs(&native), expected);
    assert_eq!(name_pairs(&native), name_pairs(&df));
}

#[tokio::test]
async fn test_native_expand_with_target_filter() {
    let q = "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE b.age > 30 RETURN a.name, b.name";
    let query = CypherQuery::new(q).unwrap().with_config(config());

    let native = query
        .execute(datasets(), Some(ExecutionStrategy::LanceNative))
        .await
        .unwrap();
    let df = query
        .execute(datasets(), Some(ExecutionStrategy::DataFusion))
        .await
        .unwrap();
    assert_eq!(name_pairs(&native), name_pairs(&df));
    // bob(40) and dave(50) qualify as targets: (alice,bob),(bob,dave),(carol,dave)
    assert_eq!(
        name_pairs(&native),
        vec![
            ("alice".to_string(), "bob".to_string()),
            ("bob".to_string(), "dave".to_string()),
            ("carol".to_string(), "dave".to_string()),
        ]
    );
}

#[tokio::test]
async fn test_native_expand_incoming_matches_datafusion() {
    let q = "MATCH (a:Person)<-[:KNOWS]-(b:Person) RETURN a.name, b.name";
    let query = CypherQuery::new(q).unwrap().with_config(config());
    let native = query
        .execute(datasets(), Some(ExecutionStrategy::LanceNative))
        .await
        .unwrap();
    let df = query
        .execute(datasets(), Some(ExecutionStrategy::DataFusion))
        .await
        .unwrap();
    assert_eq!(name_pairs(&native), name_pairs(&df));
}

#[tokio::test]
async fn test_native_varlength_falls_back_and_matches() {
    // Variable-length path is unsupported natively; LanceNative must fall back
    // and produce the same result as DataFusion.
    let q = "MATCH (a:Person)-[:KNOWS*1..2]->(b:Person) RETURN a.name, b.name";
    let query = CypherQuery::new(q).unwrap().with_config(config());
    let native = query
        .execute(datasets(), Some(ExecutionStrategy::LanceNative))
        .await
        .unwrap();
    let df = query
        .execute(datasets(), Some(ExecutionStrategy::DataFusion))
        .await
        .unwrap();
    assert_eq!(name_pairs(&native), name_pairs(&df));
}
