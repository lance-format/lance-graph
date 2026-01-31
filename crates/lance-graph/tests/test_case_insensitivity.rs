//! Case-insensitivity tests
//!
//! Verifies that all identifier types (labels, properties, relationships, variables)
//! are case-insensitive throughout the system.

use arrow_array::{BooleanArray, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance_graph::{CypherQuery, ExecutionStrategy, GraphConfig};
use std::collections::HashMap;
use std::sync::Arc;

/// Helper to create a test person dataset with mixed-case properties
fn create_test_person_dataset() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("fullName", DataType::Utf8, false),
        Field::new("isActive", DataType::Boolean, false),
        Field::new("numFollowers", DataType::Int64, false),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])),
            Arc::new(StringArray::from(vec![
                "Alice Smith",
                "Bob Jones",
                "Charlie Brown",
            ])),
            Arc::new(BooleanArray::from(vec![true, false, true])),
            Arc::new(Int64Array::from(vec![100, 200, 150])),
        ],
    )
    .unwrap()
}

#[tokio::test]
async fn test_identifiers_case_insensitive() {
    // Test labels, properties, and table names with various case combinations
    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .build()
        .unwrap();
    let person_batch = create_test_person_dataset();

    let test_cases = vec![
        // Label case variations
        ("MATCH (p:Person) RETURN p.name", 3),
        ("MATCH (p:PERSON) RETURN p.name", 3),
        ("MATCH (p:person) RETURN p.name", 3),
        // Property case variations
        ("MATCH (p:Person) RETURN p.fullName", 3),
        ("MATCH (p:Person) RETURN p.FULLNAME", 3),
        ("MATCH (p:Person) RETURN p.fullname", 3),
        ("MATCH (p:Person) RETURN p.isActive", 3),
        ("MATCH (p:Person) RETURN p.ISACTIVE", 3),
        ("MATCH (p:Person) RETURN p.numFollowers", 3),
        ("MATCH (p:Person) RETURN p.NUMFOLLOWERS", 3),
    ];

    for (query_str, expected_rows) in test_cases {
        let mut datasets = HashMap::new();
        datasets.insert("Person".to_string(), person_batch.clone());

        let result = CypherQuery::new(query_str)
            .unwrap()
            .with_config(config.clone())
            .execute(datasets, Some(ExecutionStrategy::DataFusion))
            .await;

        assert!(
            result.is_ok(),
            "Query failed: {} with error: {:?}",
            query_str,
            result.err()
        );
        assert_eq!(
            result.unwrap().num_rows(),
            expected_rows,
            "Query: {}",
            query_str
        );
    }

    // Test with lowercase table registration
    let mut datasets = HashMap::new();
    datasets.insert("person".to_string(), person_batch.clone());
    let result = CypherQuery::new("MATCH (p:Person) RETURN p.name")
        .unwrap()
        .with_config(config.clone())
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap().num_rows(), 3);

    // Test with uppercase table registration
    let mut datasets = HashMap::new();
    datasets.insert("PERSON".to_string(), person_batch);
    let result = CypherQuery::new("MATCH (p:person) RETURN p.name")
        .unwrap()
        .with_config(config)
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap().num_rows(), 3);
}

#[tokio::test]
async fn test_clauses_case_insensitive() {
    // Test WHERE, ORDER BY, and variable references with case variations
    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .build()
        .unwrap();
    let person_batch = create_test_person_dataset();

    let test_cases = vec![
        // WHERE clause with different property cases
        ("MATCH (p:Person) WHERE p.isActive = true RETURN p.name", 2),
        ("MATCH (p:Person) WHERE p.ISACTIVE = true RETURN p.name", 2),
        (
            "MATCH (p:Person) WHERE p.numFollowers > 100 RETURN p.name",
            2,
        ),
        (
            "MATCH (p:Person) WHERE p.NUMFOLLOWERS > 100 RETURN p.name",
            2,
        ),
        // Variable case: lowercase p in MATCH, uppercase P in WHERE/RETURN
        ("MATCH (p:Person) WHERE P.isActive = true RETURN p.name", 2),
        ("MATCH (p:Person) RETURN P.name", 3),
        (
            "MATCH (P:Person) WHERE p.numFollowers > 100 RETURN P.name",
            2,
        ),
        // ORDER BY with different property cases
        (
            "MATCH (p:Person) RETURN p.name ORDER BY p.numFollowers DESC",
            3,
        ),
        (
            "MATCH (p:Person) RETURN p.name ORDER BY P.NUMFOLLOWERS DESC",
            3,
        ),
        // Alias case: different case in ORDER BY
        (
            "MATCH (p:Person) RETURN p.numFollowers AS Score ORDER BY score DESC",
            3,
        ),
        (
            "MATCH (p:Person) RETURN p.numFollowers AS score ORDER BY SCORE DESC",
            3,
        ),
        // Aggregate functions with variable case
        ("MATCH (p:Person) RETURN count(P)", 1),
        ("MATCH (P:Person) RETURN count(p)", 1),
        ("MATCH (p:Person) RETURN sum(P.numFollowers)", 1),
    ];

    for (query_str, expected_rows) in test_cases {
        let mut datasets = HashMap::new();
        datasets.insert("Person".to_string(), person_batch.clone());

        let result = CypherQuery::new(query_str)
            .unwrap()
            .with_config(config.clone())
            .execute(datasets, Some(ExecutionStrategy::DataFusion))
            .await;

        assert!(
            result.is_ok(),
            "Query failed: {} with error: {:?}",
            query_str,
            result.err()
        );
        assert_eq!(
            result.unwrap().num_rows(),
            expected_rows,
            "Wrong row count for query: {}",
            query_str
        );
    }

    // Verify ORDER BY actually orders correctly
    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    let result = CypherQuery::new("MATCH (p:Person) RETURN p.name ORDER BY p.NUMFOLLOWERS DESC")
        .unwrap()
        .with_config(config)
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await
        .unwrap();

    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(names.value(0), "Bob"); // 200 followers
    assert_eq!(names.value(1), "Charlie"); // 150 followers
    assert_eq!(names.value(2), "Alice"); // 100 followers
}

#[tokio::test]
async fn test_relationships_case_insensitive() {
    // Test relationship types and relationship variables with case variations
    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .with_relationship("KNOWS", "src_id", "dst_id")
        .build()
        .unwrap();

    let person_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let knows_schema = Arc::new(Schema::new(vec![
        Field::new("src_id", DataType::Int64, false),
        Field::new("dst_id", DataType::Int64, false),
        Field::new("since", DataType::Int64, false),
    ]));

    let person_batch = RecordBatch::try_new(
        person_schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["Alice", "Bob"])),
        ],
    )
    .unwrap();

    let knows_batch = RecordBatch::try_new(
        knows_schema,
        vec![
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(Int64Array::from(vec![2])),
            Arc::new(Int64Array::from(vec![2020])),
        ],
    )
    .unwrap();

    let test_cases = vec![
        // Relationship type case variations
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name",
        "MATCH (a:Person)-[:knows]->(b:Person) RETURN a.name",
        "MATCH (a:Person)-[:Knows]->(b:Person) RETURN a.name",
        // Relationship variable case: lowercase r in pattern, uppercase R in RETURN/WHERE
        "MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN R.since",
        "MATCH (a:Person)-[R:KNOWS]->(b:Person) RETURN r.since",
        "MATCH (a:Person)-[r:KNOWS]->(b:Person) WHERE R.since > 2019 RETURN r.since",
    ];

    for query_str in test_cases {
        let mut datasets = HashMap::new();
        datasets.insert("Person".to_string(), person_batch.clone());
        datasets.insert("KNOWS".to_string(), knows_batch.clone());

        let result = CypherQuery::new(query_str)
            .unwrap()
            .with_config(config.clone())
            .execute(datasets, Some(ExecutionStrategy::DataFusion))
            .await;

        assert!(
            result.is_ok(),
            "Query failed: {} with error: {:?}",
            query_str,
            result.err()
        );
        assert_eq!(result.unwrap().num_rows(), 1, "Query: {}", query_str);
    }
}

#[tokio::test]
async fn test_complex_mixed_case_query() {
    // Comprehensive end-to-end test with mixed case everywhere
    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .build()
        .unwrap();
    let person_batch = create_test_person_dataset();

    let mut datasets = HashMap::new();
    datasets.insert("PERSON".to_string(), person_batch);

    // Query with mixed case in labels, properties, variables, and aliases
    let query = "
        MATCH (P:person)
        WHERE P.IsActive = true AND P.NumFollowers > 50
        RETURN P.FullName AS PersonName, P.NumFollowers AS Followers
        ORDER BY Followers DESC
    ";

    let result = CypherQuery::new(query)
        .unwrap()
        .with_config(config)
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await;

    assert!(
        result.is_ok(),
        "Complex mixed-case query failed: {:?}",
        result.err()
    );
    let result = result.unwrap();
    assert_eq!(result.num_rows(), 2); // Alice and Charlie

    // Verify correct ordering and values
    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let followers = result
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();

    // Should be ordered by Followers DESC: Charlie (150), Alice (100)
    assert_eq!(names.value(0), "Charlie Brown");
    assert_eq!(followers.value(0), 150);
    assert_eq!(names.value(1), "Alice Smith");
    assert_eq!(followers.value(1), 100);
}
