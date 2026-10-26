// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! WHERE predicates on aliases that are not returned, in queries with several
//! comma-separated path patterns (https://github.com/lance-format/lance-graph/issues/117).

use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance_graph::config::GraphConfig;
use lance_graph::{CypherQuery, ExecutionStrategy};
use std::collections::HashMap;
use std::sync::Arc;

fn ids(name: &str, values: Vec<i64>) -> (Field, Arc<Int64Array>) {
    (
        Field::new(name, DataType::Int64, false),
        Arc::new(Int64Array::from(values)),
    )
}

fn nodes(ids_: Vec<i64>, names: Option<Vec<&str>>) -> RecordBatch {
    let (id_field, id_col) = ids("id", ids_);
    match names {
        Some(names) => RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                id_field,
                Field::new("name", DataType::Utf8, false),
            ])),
            vec![id_col, Arc::new(StringArray::from(names))],
        )
        .unwrap(),
        None => RecordBatch::try_new(Arc::new(Schema::new(vec![id_field])), vec![id_col]).unwrap(),
    }
}

fn edges(src: Vec<i64>, dst: Vec<i64>) -> RecordBatch {
    let (src_field, src_col) = ids("src", src);
    let (dst_field, dst_col) = ids("dst", dst);
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![src_field, dst_field])),
        vec![src_col, dst_col],
    )
    .unwrap()
}

fn config() -> GraphConfig {
    GraphConfig::builder()
        .with_node_label("Person", "id")
        .with_node_label("City", "id")
        .with_node_label("Country", "id")
        .with_node_label("Hobby", "id")
        .with_relationship("livesIn", "src", "dst")
        .with_relationship("hasHobby", "src", "dst")
        .with_relationship("inCountry", "src", "dst")
        .build()
        .unwrap()
}

// Person 1 lives in Paris (France) and plays chess; person 2 lives in Berlin (Germany)
// and plays chess; person 3 lives in Paris and plays go.
fn datasets() -> HashMap<String, RecordBatch> {
    HashMap::from([
        ("Person".to_string(), nodes(vec![1, 2, 3], None)),
        (
            "City".to_string(),
            nodes(vec![10, 11], Some(vec!["Paris", "Berlin"])),
        ),
        (
            "Country".to_string(),
            nodes(vec![100, 101], Some(vec!["France", "Germany"])),
        ),
        (
            "Hobby".to_string(),
            nodes(vec![20, 21], Some(vec!["Chess", "Go"])),
        ),
        (
            "livesIn".to_string(),
            edges(vec![1, 2, 3], vec![10, 11, 10]),
        ),
        (
            "hasHobby".to_string(),
            edges(vec![1, 2, 3], vec![20, 20, 21]),
        ),
        ("inCountry".to_string(), edges(vec![10, 11], vec![100, 101])),
    ])
}

async fn returned_ids(query: &str) -> Vec<i64> {
    let result = CypherQuery::new(query)
        .unwrap()
        .with_config(config())
        .execute(datasets(), Some(ExecutionStrategy::DataFusion))
        .await
        .unwrap();
    let column = result
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let mut ids: Vec<i64> = column.values().to_vec();
    ids.sort();
    ids
}

#[tokio::test]
async fn test_where_on_unreturned_aliases_across_patterns() {
    let ids = returned_ids(
        "MATCH (c:City)-[:inCountry]->(co:Country), (p:Person)-[:livesIn]->(c), (p)-[:hasHobby]->(h:Hobby)
         WHERE co.name = 'France' AND h.name = 'Chess'
         RETURN p.id AS id",
    )
    .await;
    assert_eq!(ids, vec![1]);
}

#[tokio::test]
async fn test_second_pattern_starting_at_a_new_variable() {
    // `p` is introduced by the second pattern and must be scanned, not assumed bound.
    let ids = returned_ids(
        "MATCH (c:City)-[:inCountry]->(co:Country), (p:Person)-[:livesIn]->(c)
         WHERE co.name = 'France'
         RETURN p.id AS id",
    )
    .await;
    assert_eq!(ids, vec![1, 3]);
}

#[tokio::test]
async fn test_filter_on_alias_from_each_pattern() {
    let ids = returned_ids(
        "MATCH (p:Person)-[:livesIn]->(c:City), (p)-[:hasHobby]->(h:Hobby)
         WHERE c.name = 'Paris' AND h.name = 'Go'
         RETURN p.id AS id",
    )
    .await;
    assert_eq!(ids, vec![3]);
}
