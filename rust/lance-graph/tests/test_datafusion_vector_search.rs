// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! End-to-end integration tests for vector similarity search

use arrow_array::{Array, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, FieldRef, Schema};
use lance_graph::config::GraphConfig;
use lance_graph::{CypherQuery, ExecutionStrategy, Result};
use std::collections::HashMap;
use std::sync::Arc;

/// Helper function to create a test graph with vector embeddings
fn create_person_graph_with_embeddings() -> (GraphConfig, HashMap<String, RecordBatch>) {
    // Create schema with 3D embeddings for simplicity
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int64, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                3, // 3-dimensional vectors for testing
            ),
            false,
        ),
    ]));

    // Create test data with embeddings
    // Person vectors are chosen to have clear similarity relationships:
    // - Alice [1, 0, 0] and Bob [0.9, 0.1, 0] are very similar
    // - Carol [0, 1, 0] is orthogonal to Alice
    // - David [0, 0, 1] is orthogonal to both Alice and Carol
    // - Eve [0.5, 0.5, 0] is in between Alice and Carol
    let embedding_data = vec![
        1.0, 0.0, 0.0, // Alice
        0.9, 0.1, 0.0, // Bob
        0.0, 1.0, 0.0, // Carol
        0.0, 0.0, 1.0, // David
        0.5, 0.5, 0.0, // Eve
    ];

    // Create FixedSizeListArray using standard Arrow API
    let field = Arc::new(Field::new("item", DataType::Float32, true)) as FieldRef;
    let values = Arc::new(Float32Array::from(embedding_data));
    let embeddings = FixedSizeListArray::try_new(field, 3, values, None).unwrap();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec![
                "Alice", "Bob", "Carol", "David", "Eve",
            ])),
            Arc::new(Int64Array::from(vec![30, 25, 35, 28, 32])),
            Arc::new(embeddings),
        ],
    )
    .unwrap();

    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .build()
        .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), batch);

    (config, datasets)
}

#[tokio::test]
async fn test_vector_distance_l2_simple() -> Result<()> {
    let (config, datasets) = create_person_graph_with_embeddings();

    // Simpler test: just return vector distance in SELECT (not in WHERE)
    // Compare each person's embedding to Alice's
    let query = CypherQuery::new(
        "MATCH (p:Person), (alice:Person {name: 'Alice'}) \
         RETURN p.name, vector_distance(p.embedding, alice.embedding, l2) AS dist \
         ORDER BY dist",
    )?
    .with_config(config);

    let result = query
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await?;

    // Should return all 5 people ordered by distance to Alice
    assert_eq!(result.num_rows(), 5);
    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Alice should be first (distance=0)
    assert_eq!(names.value(0), "Alice");

    Ok(())
}

#[tokio::test]
async fn test_vector_distance_where_no_cross_product() -> Result<()> {
    let (config, datasets) = create_person_graph_with_embeddings();

    // Test WHERE clause without cross product - self-comparison
    let query = CypherQuery::new(
        "MATCH (p:Person) \
         WHERE vector_distance(p.embedding, p.embedding, l2) < 0.1 \
         RETURN p.name",
    )?
    .with_config(config);

    let result = query
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await?;

    // Self-distance is always 0, so all should match
    assert_eq!(result.num_rows(), 5);

    Ok(())
}

#[tokio::test]
async fn test_vector_distance_l2() -> Result<()> {
    let (config, datasets) = create_person_graph_with_embeddings();

    // Find people with L2 distance < 0.5 from Alice (should find Bob and Alice herself)
    let query = CypherQuery::new(
        "MATCH (p:Person), (alice:Person {name: 'Alice'}) \
         WHERE vector_distance(p.embedding, alice.embedding, l2) < 0.5 \
         RETURN p.name \
         ORDER BY p.name",
    )?
    .with_config(config);

    let result = query
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await?;

    // Should return Alice (distance=0) and Bob (distance≈0.14)
    assert_eq!(result.num_rows(), 2);
    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    assert_eq!(names.value(0), "Alice");
    assert_eq!(names.value(1), "Bob");

    Ok(())
}

#[tokio::test]
async fn test_vector_distance_l2_with_literal() -> Result<()> {
    let (config, datasets) = create_person_graph_with_embeddings();

    // Same test as above but using vector literal instead of cross product
    // Find people similar to [1, 0, 0] (Alice's embedding)
    let query = CypherQuery::new(
        "MATCH (p:Person) \
         WHERE vector_distance(p.embedding, [1.0, 0.0, 0.0], l2) < 0.2 \
         RETURN p.name ORDER BY p.name",
    )?
    .with_config(config);

    let result = query
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await?;

    // Should return Alice (exact match) and Bob (very similar)
    assert_eq!(result.num_rows(), 2);
    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(names.value(0), "Alice");
    assert_eq!(names.value(1), "Bob");

    Ok(())
}

#[tokio::test]
async fn test_vector_distance_cosine() -> Result<()> {
    let (config, datasets) = create_person_graph_with_embeddings();

    // Find people with cosine distance < 0.1 from Alice
    // Cosine distance between Alice [1,0,0] and Bob [0.9,0.1,0] ≈ 0.005
    let query = CypherQuery::new(
        "MATCH (p:Person), (alice:Person {name: 'Alice'}) \
         WHERE vector_distance(p.embedding, alice.embedding, cosine) < 0.1 \
         RETURN p.name \
         ORDER BY p.name",
    )?
    .with_config(config);

    let result = query
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await?;

    assert_eq!(result.num_rows(), 2);
    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    assert_eq!(names.value(0), "Alice");
    assert_eq!(names.value(1), "Bob");

    Ok(())
}

#[tokio::test]
async fn test_vector_similarity_cosine() -> Result<()> {
    let (config, datasets) = create_person_graph_with_embeddings();

    // Find people with cosine similarity > 0.9 to Alice
    let query = CypherQuery::new(
        "MATCH (p:Person), (alice:Person {name: 'Alice'}) \
         WHERE vector_similarity(p.embedding, alice.embedding, cosine) > 0.9 \
         RETURN p.name \
         ORDER BY p.name",
    )?
    .with_config(config);

    let result = query
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await?;

    // Alice has similarity 1.0, Bob has similarity ≈ 0.995
    assert_eq!(result.num_rows(), 2);
    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    assert_eq!(names.value(0), "Alice");
    assert_eq!(names.value(1), "Bob");

    Ok(())
}

#[tokio::test]
async fn test_vector_distance_order_by() -> Result<()> {
    let (config, datasets) = create_person_graph_with_embeddings();

    // Order all people by distance to Alice
    let query = CypherQuery::new(
        "MATCH (p:Person), (alice:Person {name: 'Alice'}) \
         RETURN p.name, vector_distance(p.embedding, alice.embedding, l2) AS dist \
         ORDER BY dist ASC",
    )?
    .with_config(config);

    let result = query
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await?;

    assert_eq!(result.num_rows(), 5);
    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let distances = result
        .column(1)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();

    // Alice should be first (distance=0)
    assert_eq!(names.value(0), "Alice");
    assert_eq!(distances.value(0), 0.0);

    // Bob should be second (closest to Alice)
    assert_eq!(names.value(1), "Bob");
    assert!(distances.value(1) < 0.2);

    // Distances should be in ascending order
    for i in 1..distances.len() {
        assert!(distances.value(i) >= distances.value(i - 1));
    }

    Ok(())
}

#[tokio::test]
async fn test_vector_distance_order_by_with_literal() -> Result<()> {
    let (config, datasets) = create_person_graph_with_embeddings();

    // Same as above but using vector literal - order by distance to [1,0,0] (Alice's vector)
    let query = CypherQuery::new(
        "MATCH (p:Person) \
         RETURN p.name \
         ORDER BY vector_distance(p.embedding, [1.0, 0.0, 0.0], cosine) ASC \
         LIMIT 3",
    )?
    .with_config(config);

    let result = query
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await?;

    // Should return Alice (closest), Bob (second), Eve (third)
    assert_eq!(result.num_rows(), 3);
    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(names.value(0), "Alice");
    assert_eq!(names.value(1), "Bob");
    assert_eq!(names.value(2), "Eve");

    Ok(())
}

#[tokio::test]
async fn test_vector_similarity_order_by() -> Result<()> {
    let (config, datasets) = create_person_graph_with_embeddings();

    // Order all people by similarity to Carol [0,1,0]
    let query = CypherQuery::new(
        "MATCH (p:Person), (carol:Person {name: 'Carol'}) \
         RETURN p.name, vector_similarity(p.embedding, carol.embedding, cosine) AS sim \
         ORDER BY sim DESC \
         LIMIT 3",
    )?
    .with_config(config);

    let result = query
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await?;

    assert_eq!(result.num_rows(), 3);
    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let similarities = result
        .column(1)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();

    // Carol should be first (similarity=1.0)
    assert_eq!(names.value(0), "Carol");
    assert!((similarities.value(0) - 1.0).abs() < 0.01);

    // Eve [0.5, 0.5, 0] should be second (has y component)
    assert_eq!(names.value(1), "Eve");

    // Similarities should be in descending order
    for i in 1..similarities.len() {
        assert!(similarities.value(i) <= similarities.value(i - 1));
    }

    Ok(())
}

#[tokio::test]
async fn test_hybrid_query_property_and_vector() -> Result<()> {
    let (config, datasets) = create_person_graph_with_embeddings();

    // Find people over 26 who are similar to Alice
    let query = CypherQuery::new(
        "MATCH (p:Person), (alice:Person {name: 'Alice'}) \
         WHERE p.age > 26 \
           AND vector_distance(p.embedding, alice.embedding, l2) < 1.0 \
         RETURN p.name, p.age \
         ORDER BY p.name",
    )?
    .with_config(config);

    let result = query
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await?;

    // Should find Alice (age=30) and Eve (age=32)
    // Bob is excluded (age=25), Carol and David are too far
    assert!(result.num_rows() >= 1); // At least Alice
    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Alice should be in results
    let alice_found = (0..names.len()).any(|i| names.value(i) == "Alice");
    assert!(alice_found);

    // Verify all results meet age criteria
    let ages = result
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    for i in 0..ages.len() {
        assert!(ages.value(i) > 26);
    }

    Ok(())
}

#[tokio::test]
async fn test_hybrid_query_with_vector_literal() -> Result<()> {
    let (config, datasets) = create_person_graph_with_embeddings();

    // Combine property filter with vector literal search
    let query = CypherQuery::new(
        "MATCH (p:Person) \
         WHERE p.age > 25 \
           AND vector_distance(p.embedding, [1.0, 0.0, 0.0], l2) < 0.3 \
         RETURN p.name ORDER BY p.name",
    )?
    .with_config(config);

    let result = query
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await?;

    // Should return only Alice (age 30 > 25, close to [1,0,0])
    // Bob is age 25, not > 25
    assert_eq!(result.num_rows(), 1);
    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(names.value(0), "Alice");

    Ok(())
}

#[tokio::test]
async fn test_vector_distance_dot_product() -> Result<()> {
    let (config, datasets) = create_person_graph_with_embeddings();

    // Test dot product metric
    // Dot product between Alice [1,0,0] and Bob [0.9,0.1,0] = 0.9
    // We negate it for distance, so distance = -0.9
    let query = CypherQuery::new(
        "MATCH (alice:Person {name: 'Alice'}), (bob:Person {name: 'Bob'}) \
         RETURN vector_distance(alice.embedding, bob.embedding, dot) AS dist",
    )?
    .with_config(config);

    let result = query
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await?;

    assert_eq!(result.num_rows(), 1);
    let distance = result
        .column(0)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap()
        .value(0);

    // Distance should be negative of dot product: -0.9
    assert!((distance + 0.9).abs() < 0.01);

    Ok(())
}

#[tokio::test]
async fn test_vector_similarity_dot_product() -> Result<()> {
    let (config, datasets) = create_person_graph_with_embeddings();

    // Test dot product similarity
    let query = CypherQuery::new(
        "MATCH (alice:Person {name: 'Alice'}), (bob:Person {name: 'Bob'}) \
         RETURN vector_similarity(alice.embedding, bob.embedding, dot) AS sim",
    )?
    .with_config(config);

    let result = query
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await?;

    assert_eq!(result.num_rows(), 1);
    let similarity = result
        .column(0)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap()
        .value(0);

    // Similarity should be positive dot product: 0.9
    assert!((similarity - 0.9).abs() < 0.01);

    Ok(())
}

#[tokio::test]
async fn test_vector_search_with_limit() -> Result<()> {
    let (config, datasets) = create_person_graph_with_embeddings();

    // Find top 2 most similar people to Alice (excluding herself)
    let query = CypherQuery::new(
        "MATCH (p:Person), (alice:Person {name: 'Alice'}) \
         WHERE p.name <> 'Alice' \
         RETURN p.name, vector_distance(p.embedding, alice.embedding, cosine) AS dist \
         ORDER BY dist ASC \
         LIMIT 2",
    )?
    .with_config(config);

    let result = query
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await?;

    assert_eq!(result.num_rows(), 2);
    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Bob should be first (most similar to Alice)
    assert_eq!(names.value(0), "Bob");

    Ok(())
}

#[tokio::test]
async fn test_vector_distance_between_different_people() -> Result<()> {
    let (config, datasets) = create_person_graph_with_embeddings();

    // Compute distance between Carol and David (orthogonal vectors)
    let query = CypherQuery::new(
        "MATCH (carol:Person {name: 'Carol'}), (david:Person {name: 'David'}) \
         RETURN vector_distance(carol.embedding, david.embedding, cosine) AS dist",
    )?
    .with_config(config);

    let result = query
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await?;

    assert_eq!(result.num_rows(), 1);
    let distance = result
        .column(0)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap()
        .value(0);

    // Carol [0,1,0] and David [0,0,1] are orthogonal
    // Cosine distance should be 1.0
    assert!((distance - 1.0).abs() < 0.01);

    Ok(())
}

#[tokio::test]
async fn test_multiple_vector_comparisons() -> Result<()> {
    let (config, datasets) = create_person_graph_with_embeddings();

    // Find people similar to either Alice or Carol
    let query = CypherQuery::new(
        "MATCH (p:Person), (alice:Person {name: 'Alice'}), (carol:Person {name: 'Carol'}) \
         WHERE vector_distance(p.embedding, alice.embedding, l2) < 0.3 \
            OR vector_distance(p.embedding, carol.embedding, l2) < 0.3 \
         RETURN DISTINCT p.name \
         ORDER BY p.name",
    )?
    .with_config(config);

    let result = query
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await?;

    // Should find at least Alice, Bob, and Carol
    assert!(result.num_rows() >= 3);
    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let name_vec: Vec<&str> = (0..names.len()).map(|i| names.value(i)).collect();

    assert!(name_vec.contains(&"Alice"));
    assert!(name_vec.contains(&"Bob"));
    assert!(name_vec.contains(&"Carol"));

    Ok(())
}

#[tokio::test]
async fn test_vector_similarity_l2_conversion() -> Result<()> {
    let (config, datasets) = create_person_graph_with_embeddings();

    // Test L2 similarity (converted from distance: 1/(1+dist))
    let query = CypherQuery::new(
        "MATCH (alice:Person {name: 'Alice'}), (bob:Person {name: 'Bob'}) \
         RETURN vector_similarity(alice.embedding, bob.embedding, l2) AS sim",
    )?
    .with_config(config);

    let result = query
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await?;

    assert_eq!(result.num_rows(), 1);
    let similarity = result
        .column(0)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap()
        .value(0);

    // L2 distance between Alice and Bob ≈ 0.14
    // Similarity = 1/(1+0.14) ≈ 0.877
    assert!(similarity > 0.8 && similarity < 1.0);

    Ok(())
}

#[tokio::test]
async fn test_vector_search_with_aggregation() -> Result<()> {
    let (config, datasets) = create_person_graph_with_embeddings();

    // Count how many people are similar to Alice (distance < 0.5)
    let query = CypherQuery::new(
        "MATCH (p:Person), (alice:Person {name: 'Alice'}) \
         WHERE vector_distance(p.embedding, alice.embedding, l2) < 0.5 \
         RETURN count(p) AS similar_count",
    )?
    .with_config(config);

    let result = query
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await?;

    assert_eq!(result.num_rows(), 1);
    let count = result
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);

    // Should find Alice and Bob
    assert_eq!(count, 2);

    Ok(())
}

#[tokio::test]
async fn test_vector_distance_self_comparison() -> Result<()> {
    let (config, datasets) = create_person_graph_with_embeddings();

    // Vector distance from a person to themselves should be 0
    let query = CypherQuery::new(
        "MATCH (p:Person {name: 'Carol'}) \
         RETURN vector_distance(p.embedding, p.embedding, l2) AS dist",
    )?
    .with_config(config);

    let result = query
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await?;

    assert_eq!(result.num_rows(), 1);
    let distance = result
        .column(0)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap()
        .value(0);

    assert_eq!(distance, 0.0);

    Ok(())
}

#[tokio::test]
async fn test_vector_similarity_self_comparison() -> Result<()> {
    let (config, datasets) = create_person_graph_with_embeddings();

    // Vector similarity from a person to themselves should be 1.0 (for cosine)
    let query = CypherQuery::new(
        "MATCH (p:Person {name: 'David'}) \
         RETURN vector_similarity(p.embedding, p.embedding, cosine) AS sim",
    )?
    .with_config(config);

    let result = query
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await?;

    assert_eq!(result.num_rows(), 1);
    let similarity = result
        .column(0)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap()
        .value(0);

    assert!((similarity - 1.0).abs() < 0.001);

    Ok(())
}

#[tokio::test]
async fn test_vector_literal_in_return_clause() -> Result<()> {
    let (config, datasets) = create_person_graph_with_embeddings();

    // Use vector literal in RETURN to compute distances
    let query = CypherQuery::new(
        "MATCH (p:Person) \
         RETURN p.name, vector_distance(p.embedding, [0.5, 0.5, 0.0], l2) AS dist \
         ORDER BY dist ASC \
         LIMIT 1",
    )?
    .with_config(config);

    let result = query
        .execute(datasets, Some(ExecutionStrategy::DataFusion))
        .await?;

    // Should return Eve (closest to [0.5, 0.5, 0.0])
    assert_eq!(result.num_rows(), 1);
    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(names.value(0), "Eve");

    // Distance should be 0 (exact match)
    let distances = result
        .column(1)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    assert!(distances.value(0) < 0.001);

    Ok(())
}
