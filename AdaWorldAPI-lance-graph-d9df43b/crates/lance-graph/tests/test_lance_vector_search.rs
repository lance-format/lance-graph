// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Integration tests for Lance Vector Search API

use arrow_array::{Array, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, FieldRef, Schema};
use lance_graph::ast::DistanceMetric;
use lance_graph::config::GraphConfig;
use lance_graph::{CypherQuery, Result, VectorSearch};
use std::collections::HashMap;
use std::sync::Arc;

/// Helper function to create a test graph with vector embeddings
fn create_test_graph_with_embeddings() -> (GraphConfig, HashMap<String, RecordBatch>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("category", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 3),
            false,
        ),
    ]));

    // Create test data with embeddings
    // Vectors are chosen to have clear similarity relationships:
    // - Doc1 [1, 0, 0] and Doc2 [0.9, 0.1, 0] are very similar (category: tech)
    // - Doc3 [0, 1, 0] is orthogonal to Doc1 (category: science)
    // - Doc4 [0, 0, 1] is orthogonal to both (category: tech)
    // - Doc5 [0.5, 0.5, 0] is in between Doc1 and Doc3 (category: science)
    let embedding_data = vec![
        1.0, 0.0, 0.0, // Doc1
        0.9, 0.1, 0.0, // Doc2
        0.0, 1.0, 0.0, // Doc3
        0.0, 0.0, 1.0, // Doc4
        0.5, 0.5, 0.0, // Doc5
    ];

    let field = Arc::new(Field::new("item", DataType::Float32, true)) as FieldRef;
    let values = Arc::new(Float32Array::from(embedding_data));
    let embeddings = FixedSizeListArray::try_new(field, 3, values, None).unwrap();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec![
                "Doc1", "Doc2", "Doc3", "Doc4", "Doc5",
            ])),
            Arc::new(StringArray::from(vec![
                "tech", "tech", "science", "tech", "science",
            ])),
            Arc::new(embeddings),
        ],
    )
    .unwrap();

    let config = GraphConfig::builder()
        .with_node_label("Document", "id")
        .build()
        .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Document".to_string(), batch);

    (config, datasets)
}

#[tokio::test]
async fn test_execute_with_vector_rerank_basic() -> Result<()> {
    let (config, datasets) = create_test_graph_with_embeddings();

    // Use the convenience method that combines Cypher + vector rerank
    let query = CypherQuery::new(
        "MATCH (d:Document) \
         RETURN d.id, d.name, d.embedding",
    )?
    .with_config(config);

    let results = query
        .execute_with_vector_rerank(
            datasets,
            VectorSearch::new("d.embedding")
                .query_vector(vec![1.0, 0.0, 0.0])
                .metric(DistanceMetric::L2)
                .top_k(3),
        )
        .await?;

    assert_eq!(results.num_rows(), 3);

    let names = results
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Doc1 should be first (closest to [1,0,0])
    assert_eq!(names.value(0), "Doc1");
    assert_eq!(names.value(1), "Doc2");

    Ok(())
}

#[tokio::test]
async fn test_execute_with_vector_rerank_filtered() -> Result<()> {
    let (config, datasets) = create_test_graph_with_embeddings();

    // Filter by category first, then rerank
    let query = CypherQuery::new(
        "MATCH (d:Document) \
         WHERE d.category = 'science' \
         RETURN d.id, d.name, d.embedding",
    )?
    .with_config(config);

    let results = query
        .execute_with_vector_rerank(
            datasets,
            VectorSearch::new("d.embedding")
                .query_vector(vec![0.0, 1.0, 0.0]) // Query similar to Doc3
                .metric(DistanceMetric::Cosine)
                .top_k(2),
        )
        .await?;

    assert_eq!(results.num_rows(), 2);

    let names = results
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Doc3 should be first (closest to [0,1,0])
    assert_eq!(names.value(0), "Doc3");

    Ok(())
}

#[tokio::test]
async fn test_execute_with_vector_rerank_with_distance() -> Result<()> {
    let (config, datasets) = create_test_graph_with_embeddings();

    let query = CypherQuery::new(
        "MATCH (d:Document) \
         WHERE d.category = 'tech' \
         RETURN d.id, d.name, d.embedding",
    )?
    .with_config(config);

    let results = query
        .execute_with_vector_rerank(
            datasets,
            VectorSearch::new("d.embedding")
                .query_vector(vec![1.0, 0.0, 0.0])
                .metric(DistanceMetric::L2)
                .top_k(2)
                .include_distance(true),
        )
        .await?;

    assert_eq!(results.num_rows(), 2);

    // Should have distance column
    let schema = results.schema();
    assert!(schema.field_with_name("_distance").is_ok());

    // First result should have distance 0 (Doc1 is [1,0,0])
    let distances = results
        .column(results.num_columns() - 1)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    assert_eq!(distances.value(0), 0.0);

    Ok(())
}

#[tokio::test]
async fn test_execute_with_vector_rerank_different_metrics() -> Result<()> {
    let (config, datasets) = create_test_graph_with_embeddings();

    // Test with Cosine metric
    let query = CypherQuery::new(
        "MATCH (d:Document) \
         RETURN d.id, d.name, d.embedding",
    )?
    .with_config(config.clone());

    let results_cosine = query
        .execute_with_vector_rerank(
            datasets.clone(),
            VectorSearch::new("d.embedding")
                .query_vector(vec![1.0, 0.0, 0.0])
                .metric(DistanceMetric::Cosine)
                .top_k(1),
        )
        .await?;

    let names_cosine = results_cosine
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(names_cosine.value(0), "Doc1");

    // Test with Dot metric
    let query = CypherQuery::new(
        "MATCH (d:Document) \
         RETURN d.id, d.name, d.embedding",
    )?
    .with_config(config);

    let results_dot = query
        .execute_with_vector_rerank(
            datasets,
            VectorSearch::new("d.embedding")
                .query_vector(vec![1.0, 0.0, 0.0])
                .metric(DistanceMetric::Dot)
                .top_k(1),
        )
        .await?;

    let names_dot = results_dot
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(names_dot.value(0), "Doc1");

    Ok(())
}

#[tokio::test]
async fn test_graphrag_workflow_cypher_filter_then_vector_rerank() -> Result<()> {
    // This test demonstrates a typical GraphRAG workflow using the convenience API:
    // 1. Use Cypher to filter by graph structure/properties
    // 2. Rerank by semantic similarity

    let (config, datasets) = create_test_graph_with_embeddings();

    // Scenario: Find tech documents, rank by similarity to a query about "machine learning"
    // (represented by vector [0.8, 0.2, 0.0])
    let query = CypherQuery::new(
        "MATCH (d:Document) \
         WHERE d.category = 'tech' \
         RETURN d.id, d.name, d.category, d.embedding",
    )?
    .with_config(config);

    let query_embedding = vec![0.8, 0.2, 0.0]; // Similar to Doc1 and Doc2

    let ranked_results = query
        .execute_with_vector_rerank(
            datasets,
            VectorSearch::new("d.embedding")
                .query_vector(query_embedding)
                .metric(DistanceMetric::Cosine)
                .top_k(2)
                .include_distance(true),
        )
        .await?;

    assert_eq!(ranked_results.num_rows(), 2);

    let names = ranked_results
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Doc1 and Doc2 should be the top results (closest to query)
    let top_names: Vec<&str> = (0..names.len()).map(|i| names.value(i)).collect();
    assert!(top_names.contains(&"Doc1"));
    assert!(top_names.contains(&"Doc2"));

    // Verify category filter was applied (all results should be "tech")
    let categories = ranked_results
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    for i in 0..categories.len() {
        assert_eq!(categories.value(i), "tech");
    }

    Ok(())
}

#[tokio::test]
async fn test_execute_with_vector_rerank_preserves_columns() -> Result<()> {
    let (config, datasets) = create_test_graph_with_embeddings();

    let query = CypherQuery::new(
        "MATCH (d:Document) \
         RETURN d.id, d.name, d.category, d.embedding",
    )?
    .with_config(config);

    let results = query
        .execute_with_vector_rerank(
            datasets,
            VectorSearch::new("d.embedding")
                .query_vector(vec![1.0, 0.0, 0.0])
                .metric(DistanceMetric::L2)
                .top_k(1)
                .include_distance(true),
        )
        .await?;

    // Verify all columns from Cypher query are preserved
    let schema = results.schema();
    assert!(schema.field_with_name("d.id").is_ok());
    assert!(schema.field_with_name("d.name").is_ok());
    assert!(schema.field_with_name("d.category").is_ok());
    assert!(schema.field_with_name("d.embedding").is_ok());
    assert!(schema.field_with_name("_distance").is_ok());

    // Verify the data is correct
    let ids = results
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(ids.value(0), 1); // Doc1 has id=1

    let names = results
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(names.value(0), "Doc1");

    Ok(())
}

#[tokio::test]
async fn test_execute_with_vector_rerank_empty_result() -> Result<()> {
    let (config, datasets) = create_test_graph_with_embeddings();

    // Query that returns no results
    let query = CypherQuery::new(
        "MATCH (d:Document) \
         WHERE d.category = 'nonexistent' \
         RETURN d.id, d.name, d.embedding",
    )?
    .with_config(config);

    let results = query
        .execute_with_vector_rerank(
            datasets,
            VectorSearch::new("d.embedding")
                .query_vector(vec![1.0, 0.0, 0.0])
                .metric(DistanceMetric::L2)
                .top_k(10),
        )
        .await?;

    // Should return empty result
    assert_eq!(results.num_rows(), 0);

    Ok(())
}
