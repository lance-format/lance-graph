use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance_arrow::SchemaExt;
use lance_graph::config::GraphConfig;
use lance_graph::{CypherQuery, ExecutionStrategy, NodeMapping};
use std::collections::HashMap;
use std::sync::Arc;

// This test suite validates complex RETURN clause scenarios
//
// Datasets used:
//
// Person Dataset:
// | id | name    |
// |----|---------|
// | 1  | Alice   |
// | 2  | Bob     |
// | 3  | Charlie |
//
// Scenarios Tested:
// 1. RETURN node_variable; should expand to all properties in RETURN clause

/// Helper to create Person dataset
fn create_person_dataset() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])),
        ],
    )
    .unwrap()
}

fn create_graph_config() -> GraphConfig {
    GraphConfig::builder()
        .with_node_mapping(NodeMapping {
            label: "Person".to_string(),
            id_field: "id".to_string(),
            property_fields: vec!["name".to_string()],
            filter_conditions: None,
        })
        .build()
        .unwrap()
}

#[tokio::test]
async fn test_return_node_variable_expands_properties() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();

    let query = CypherQuery::new("MATCH (p:Person) RETURN p")
        .unwrap()
        .with_config(config);
    let datasets = HashMap::from([("Person".to_string(), person_batch)]);
    let result = query
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await
        .unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.schema().field_names(), vec!["p.id", "p.name"]);
}
