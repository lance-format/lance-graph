// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, sync::Arc};

use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance_graph::{CypherQuery, ExecutionStrategy, GraphConfig};

fn create_person_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let ids = Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5]));
    let names = Arc::new(StringArray::from(vec![
        "Alice", "Bob", "Charlie", "David", "Eve",
    ]));

    RecordBatch::try_new(schema, vec![ids, names]).unwrap()
}

#[tokio::test]
async fn test_tolower_works_in_simple_executor() {
    let person_batch = create_person_batch();
    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .build()
        .unwrap();

    let query = CypherQuery::new(
        "MATCH (p:Person) RETURN p.name AS name, tolower(p.name) AS lowered ORDER BY name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let result = query
        .execute(datasets, Some(ExecutionStrategy::Simple))
        .await
        .unwrap();

    let names = result
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let lowered = result
        .column_by_name("lowered")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let got: Vec<(String, String)> = (0..result.num_rows())
        .map(|i| (names.value(i).to_string(), lowered.value(i).to_string()))
        .collect();

    assert_eq!(
        got,
        vec![
            ("Alice".to_string(), "alice".to_string()),
            ("Bob".to_string(), "bob".to_string()),
            ("Charlie".to_string(), "charlie".to_string()),
            ("David".to_string(), "david".to_string()),
            ("Eve".to_string(), "eve".to_string()),
        ]
    );
}
