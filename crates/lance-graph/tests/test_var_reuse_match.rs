use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance_graph::config::GraphConfig;
use lance_graph::{CypherQuery, ExecutionStrategy};
use std::collections::HashMap;
use std::sync::Arc;

// This test suite validates variable reuse scenarios in MATCH clauses
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
// Company Dataset:
// | id | name   |
// |----|--------|
// | 10 | Acme   |
// | 11 | Globex |
//
// WORKS_AT Relationship:
// | src | dst |
// |-----|-----|
// | 1   | 1   | Alice -> Alice (Self-employment/Loop)
// | 1   | 2   | Alice -> Bob
// | 2   | 10  | Bob -> Acme
//
// Scenarios Tested:
// 1. Valid Reuse: Matching a node that connects to itself (cyclic/self-loop)
// 2. Invalid Reuse (Unlabelled): Reusing a variable without defining its label in its first usage
// 3. Label Mismatch: Reusing a variable with a contradicting label (e.g., Person vs Company)
// 4. Type Mismatch: Reusing a node variable as a relationship variable

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

/// Helper to create Company dataset
fn create_company_dataset() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![10, 11])),
            Arc::new(StringArray::from(vec!["Acme", "Globex"])),
        ],
    )
    .unwrap()
}

/// Helper to create WORKS_AT relationship
fn create_works_at_dataset() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src_id", DataType::Int64, false),
        Field::new("dst_id", DataType::Int64, false),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 1, 2])),
            Arc::new(Int64Array::from(vec![1, 2, 10])), // Alice works at Alice, Alice works at Bob, Bob works at Acme
        ],
    )
    .unwrap()
}

fn create_graph_config() -> GraphConfig {
    GraphConfig::builder()
        .with_node_label("Person", "id")
        .with_node_label("Company", "id")
        .with_relationship("WORKS_AT", "src_id", "dst_id")
        .build()
        .unwrap()
}

#[tokio::test]
async fn test_var_reuse() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let works_at_batch = create_works_at_dataset();
    
    // MATCH (a:Person)-[:WORKS_AT]->(a) RETURN a.name
    // Should find Alice (works at herself)
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:WORKS_AT]->(a) RETURN a.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("WORKS_AT".to_string(), works_at_batch);

    let out = query
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await
        .unwrap();

    assert_eq!(out.num_rows(), 1);
    let names = out.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(names.value(0), "Alice");
}

#[tokio::test]
async fn test_var_reuse_unlabelled_existing_var_and_new_var() {
     let config = create_graph_config();
    
    let query = CypherQuery::new(
        "MATCH (a)-[:WORKS_AT]->(a) RETURN a.name",
    )
    .unwrap()
    .with_config(config);
    
    let person_batch = create_person_dataset();
    let works_at_batch = create_works_at_dataset();
    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("WORKS_AT".to_string(), works_at_batch);

    let result = query
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await;
        
    assert!(result.is_err());
    let err_msg = format!("{}", result.err().unwrap());
    assert!(err_msg.contains("Variable 'a' is not assigned a node label but re-used"));
}

#[tokio::test]
async fn test_var_reuse_unlabelled_existing_var_labelled_new_var() {
    let config = create_graph_config();
    let query = CypherQuery::new(
        "MATCH (a)-[:WORKS_AT]->(a:Company) RETURN a.name",
    ).unwrap().with_config(config);

    let person_batch = create_person_dataset();
    let works_at_batch = create_works_at_dataset();
    let company_batch = create_company_dataset();
    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("WORKS_AT".to_string(), works_at_batch);
    datasets.insert("Company".to_string(), company_batch);

    let result = query.execute(datasets, Some(ExecutionStrategy::DataFusion)).await;
    assert!(result.is_err());
    let err_msg = format!("{}", result.err().unwrap());
    assert!(err_msg.contains("Variable 'a' is not assigned a node label but re-used"));
}

#[tokio::test]
async fn test_var_reuse_label_mismatch() {
    let config = create_graph_config();
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:WORKS_AT]->(a:Company) RETURN a.name",
    ).unwrap().with_config(config);

    let person_batch = create_person_dataset();
    let works_at_batch = create_works_at_dataset();
    let company_batch = create_company_dataset();
    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("WORKS_AT".to_string(), works_at_batch);
    datasets.insert("Company".to_string(), company_batch);

    let result = query.execute(datasets, Some(ExecutionStrategy::DataFusion)).await;
    assert!(result.is_err());
    let err_msg = format!("{}", result.err().unwrap());
    assert!(err_msg.contains("Variable 'a' has conflicting labels: 'Person' and 'Company'"));
}

#[tokio::test]
async fn test_var_reuse_rel() {
    let config = create_graph_config();
    let query = CypherQuery::new(
        "MATCH (a:Person)-[a:WORKS_AT]->(b:Company) RETURN a.name",
    ).unwrap().with_config(config);
    
    let person_batch = create_person_dataset();
    let works_at_batch = create_works_at_dataset();
    let company_batch = create_company_dataset();
    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("WORKS_AT".to_string(), works_at_batch);
    datasets.insert("Company".to_string(), company_batch);

    let result = query.execute(datasets, Some(ExecutionStrategy::DataFusion)).await;
    assert!(result.is_err());
    let err_msg = format!("{}", result.err().unwrap());
    assert!(err_msg.contains("Variable cannot be re-used on a rel: 'a'") || err_msg.contains("Variable 'a' redefined with different type"));
}
