// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Relationships without a type, such as `-[]->` and `-[*1..2]->`
//! (https://github.com/lance-format/lance-graph/issues/104).

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use lance_graph::config::GraphConfig;
use lance_graph::{CypherQuery, ExecutionStrategy};
use std::collections::HashMap;
use std::sync::Arc;

fn int_batch(columns: &[(&str, Vec<i64>)]) -> RecordBatch {
    let schema = Arc::new(Schema::new(
        columns
            .iter()
            .map(|(name, _)| Field::new(*name, DataType::Int64, false))
            .collect::<Vec<_>>(),
    ));
    let arrays = columns
        .iter()
        .map(|(_, values)| Arc::new(Int64Array::from(values.clone())) as _)
        .collect();
    RecordBatch::try_new(schema, arrays).unwrap()
}

/// Person 1 FOLLOWS 2, and 2 FOLLOWS 3. With `with_blocks`, the graph also has a
/// BLOCKS relationship type (1 BLOCKS 3).
async fn run(query: &str, with_blocks: bool) -> lance_graph::Result<Vec<i64>> {
    let mut builder = GraphConfig::builder()
        .with_node_label("Person", "id")
        .with_relationship("FOLLOWS", "src", "dst");
    let mut datasets = HashMap::from([
        ("Person".to_string(), int_batch(&[("id", vec![1, 2, 3])])),
        (
            "FOLLOWS".to_string(),
            int_batch(&[("src", vec![1, 2]), ("dst", vec![2, 3])]),
        ),
    ]);
    if with_blocks {
        builder = builder.with_relationship("BLOCKS", "src", "dst");
        datasets.insert(
            "BLOCKS".to_string(),
            int_batch(&[("src", vec![1]), ("dst", vec![3])]),
        );
    }
    let result = CypherQuery::new(query)?
        .with_config(builder.build()?)
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await?;
    let ids = result
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let mut ids = ids.values().to_vec();
    ids.sort();
    Ok(ids)
}

#[tokio::test]
async fn test_untyped_relationship_uses_the_only_relationship_type() {
    let single_hop = run(
        "MATCH (a:Person {id: 1})-[]->(b:Person) RETURN b.id AS id",
        false,
    )
    .await
    .unwrap();
    assert_eq!(single_hop, vec![2]);

    let variable_length = run(
        "MATCH (a:Person {id: 1})-[*1..2]->(b:Person) RETURN b.id AS id",
        false,
    )
    .await
    .unwrap();
    assert_eq!(variable_length, vec![2, 3]);
}

#[tokio::test]
async fn test_untyped_relationship_with_several_relationship_types_is_rejected() {
    for query in [
        "MATCH (a:Person {id: 1})-[]->(b:Person) RETURN b.id AS id",
        "MATCH (a:Person {id: 1})-[*1..2]->(b:Person) RETURN b.id AS id",
    ] {
        let error = run(query, true).await.unwrap_err().to_string();
        assert!(
            error.contains("Relationships without a type") && error.contains("BLOCKS, FOLLOWS"),
            "{query}: {error}"
        );
    }

    // A typed relationship still works in the same graph.
    let typed = run(
        "MATCH (a:Person {id: 1})-[:BLOCKS]->(b:Person) RETURN b.id AS id",
        true,
    )
    .await
    .unwrap();
    assert_eq!(typed, vec![3]);
}
