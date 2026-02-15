# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Tests for the VectorSearch API.

This tests the explicit two-step vector search workflow:
1. Cypher query for graph traversal/filtering
2. VectorSearch for similarity ranking
"""

import pyarrow as pa
import pytest
from lance_graph import CypherQuery, DistanceMetric, GraphConfig, VectorSearch


@pytest.fixture
def vector_env():
    """Create test data with vector embeddings."""
    # Create documents with 3D embeddings
    # Create embedding column with explicit float32 type
    # Vectors are chosen to have clear similarity relationships:
    # - Doc1 [1, 0, 0] and Doc2 [0.9, 0.1, 0] are very similar (category: tech)
    # - Doc3 [0, 1, 0] is orthogonal to Doc1 (category: science)
    # - Doc4 [0, 0, 1] is orthogonal to both (category: tech)
    # - Doc5 [0.5, 0.5, 0] is in between Doc1 and Doc3 (category: science)
    embedding_values = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
    ]

    documents_table = pa.table(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Doc1", "Doc2", "Doc3", "Doc4", "Doc5"],
            "category": ["tech", "tech", "science", "tech", "science"],
            "embedding": pa.array(embedding_values, type=pa.list_(pa.float32())),
        }
    )

    config = GraphConfig.builder().with_node_label("Document", "id").build()

    datasets = {"Document": documents_table}

    return config, datasets, documents_table


def test_vector_search_basic(vector_env):
    """Test basic vector search on a PyArrow table."""
    _, _, table = vector_env

    results = (
        VectorSearch("embedding")
        .query_vector([1.0, 0.0, 0.0])
        .metric(DistanceMetric.L2)
        .top_k(3)
        .search(table)
    )

    data = results.to_pydict()
    assert len(data["name"]) == 3
    # Doc1 should be first (closest to [1,0,0])
    assert data["name"][0] == "Doc1"
    assert data["name"][1] == "Doc2"


def test_vector_search_with_distance(vector_env):
    """Test vector search with distance column included."""
    _, _, table = vector_env

    results = (
        VectorSearch("embedding")
        .query_vector([1.0, 0.0, 0.0])
        .metric(DistanceMetric.L2)
        .top_k(2)
        .include_distance(True)
        .search(table)
    )

    data = results.to_pydict()
    assert "_distance" in data
    # First result should have distance 0 (identical vector)
    assert data["_distance"][0] == pytest.approx(0.0, abs=1e-6)


def test_vector_search_cosine_metric(vector_env):
    """Test vector search with cosine distance metric."""
    _, _, table = vector_env

    results = (
        VectorSearch("embedding")
        .query_vector([1.0, 0.0, 0.0])
        .metric(DistanceMetric.Cosine)
        .top_k(3)
        .search(table)
    )

    data = results.to_pydict()
    assert len(data["name"]) == 3
    # Doc1 and Doc2 should be closest (cosine similarity)
    assert data["name"][0] == "Doc1"
    assert data["name"][1] == "Doc2"


def test_vector_search_dot_metric(vector_env):
    """Test vector search with dot product metric."""
    _, _, table = vector_env

    results = (
        VectorSearch("embedding")
        .query_vector([1.0, 0.0, 0.0])
        .metric(DistanceMetric.Dot)
        .top_k(2)
        .search(table)
    )

    data = results.to_pydict()
    assert len(data["name"]) == 2
    # Doc1 should be first (highest dot product with [1,0,0])
    assert data["name"][0] == "Doc1"


def test_vector_search_custom_distance_column(vector_env):
    """Test vector search with custom distance column name."""
    _, _, table = vector_env

    results = (
        VectorSearch("embedding")
        .query_vector([1.0, 0.0, 0.0])
        .metric(DistanceMetric.L2)
        .top_k(2)
        .include_distance(True)
        .distance_column_name("similarity_score")
        .search(table)
    )

    data = results.to_pydict()
    assert "similarity_score" in data
    assert "_distance" not in data


def test_vector_search_without_distance(vector_env):
    """Test vector search without distance column."""
    _, _, table = vector_env

    results = (
        VectorSearch("embedding")
        .query_vector([1.0, 0.0, 0.0])
        .metric(DistanceMetric.L2)
        .top_k(2)
        .include_distance(False)
        .search(table)
    )

    data = results.to_pydict()
    assert "_distance" not in data


def test_execute_with_vector_rerank_basic(vector_env):
    """Test the convenience method that combines Cypher + vector rerank."""
    config, datasets, _ = vector_env

    query = CypherQuery(
        "MATCH (d:Document) RETURN d.id, d.name, d.embedding"
    ).with_config(config)

    results = query.execute_with_vector_rerank(
        datasets,
        VectorSearch("d.embedding")
        .query_vector([1.0, 0.0, 0.0])
        .metric(DistanceMetric.L2)
        .top_k(3),
    )

    data = results.to_pydict()
    assert len(data["d.name"]) == 3
    # Doc1 should be first (closest to [1,0,0])
    assert data["d.name"][0] == "Doc1"
    assert data["d.name"][1] == "Doc2"


def test_execute_with_vector_rerank_filtered(vector_env):
    """Test Cypher filter + vector rerank."""
    config, datasets, _ = vector_env

    # Filter by category first, then rerank
    query = CypherQuery(
        "MATCH (d:Document) WHERE d.category = 'science' "
        "RETURN d.id, d.name, d.embedding"
    ).with_config(config)

    results = query.execute_with_vector_rerank(
        datasets,
        VectorSearch("d.embedding")
        .query_vector([0.0, 1.0, 0.0])  # Query similar to Doc3
        .metric(DistanceMetric.Cosine)
        .top_k(2),
    )

    data = results.to_pydict()
    assert len(data["d.name"]) == 2
    # Doc3 should be first (closest to [0,1,0])
    assert data["d.name"][0] == "Doc3"


def test_execute_with_vector_rerank_with_distance(vector_env):
    """Test Cypher + vector rerank with distance column."""
    config, datasets, _ = vector_env

    query = CypherQuery(
        "MATCH (d:Document) WHERE d.category = 'tech' RETURN d.id, d.name, d.embedding"
    ).with_config(config)

    results = query.execute_with_vector_rerank(
        datasets,
        VectorSearch("d.embedding")
        .query_vector([1.0, 0.0, 0.0])
        .metric(DistanceMetric.L2)
        .top_k(2)
        .include_distance(True),
    )

    data = results.to_pydict()
    assert len(data["d.name"]) == 2
    assert "_distance" in data
    # First result should have distance 0 (Doc1 is [1,0,0])
    assert data["_distance"][0] == pytest.approx(0.0, abs=1e-6)


def test_graphrag_workflow(vector_env):
    """Test a typical GraphRAG workflow: graph filter + vector rerank."""
    config, datasets, _ = vector_env

    # Scenario: Find tech documents, rank by similarity to a query
    query = CypherQuery(
        "MATCH (d:Document) WHERE d.category = 'tech' "
        "RETURN d.id, d.name, d.category, d.embedding"
    ).with_config(config)

    # Query vector similar to Doc1 and Doc2
    query_embedding = [0.8, 0.2, 0.0]

    results = query.execute_with_vector_rerank(
        datasets,
        VectorSearch("d.embedding")
        .query_vector(query_embedding)
        .metric(DistanceMetric.Cosine)
        .top_k(2)
        .include_distance(True),
    )

    data = results.to_pydict()
    assert len(data["d.name"]) == 2

    # Doc1 and Doc2 should be the top results
    top_names = set(data["d.name"])
    assert "Doc1" in top_names
    assert "Doc2" in top_names

    # All results should be "tech" category
    assert all(cat == "tech" for cat in data["d.category"])


def test_vector_search_missing_query_vector(vector_env):
    """Test error when query vector is not set."""
    _, _, table = vector_env

    with pytest.raises(ValueError, match="Query vector is required"):
        VectorSearch("embedding").metric(DistanceMetric.L2).top_k(2).search(table)


def test_vector_search_missing_column(vector_env):
    """Test error when column doesn't exist."""
    _, _, table = vector_env

    with pytest.raises(ValueError, match="not found"):
        (
            VectorSearch("nonexistent_column")
            .query_vector([1.0, 0.0, 0.0])
            .top_k(2)
            .search(table)
        )


def test_vector_search_different_query_vectors(vector_env):
    """Test that different query vectors return different results."""
    _, _, table = vector_env

    # Query 1: Similar to Doc1 [1,0,0]
    results1 = (
        VectorSearch("embedding")
        .query_vector([1.0, 0.0, 0.0])
        .metric(DistanceMetric.L2)
        .top_k(1)
        .search(table)
    )
    assert results1.to_pydict()["name"][0] == "Doc1"

    # Query 2: Similar to Doc3 [0,1,0]
    results2 = (
        VectorSearch("embedding")
        .query_vector([0.0, 1.0, 0.0])
        .metric(DistanceMetric.L2)
        .top_k(1)
        .search(table)
    )
    assert results2.to_pydict()["name"][0] == "Doc3"

    # Query 3: Similar to Doc4 [0,0,1]
    results3 = (
        VectorSearch("embedding")
        .query_vector([0.0, 0.0, 1.0])
        .metric(DistanceMetric.L2)
        .top_k(1)
        .search(table)
    )
    assert results3.to_pydict()["name"][0] == "Doc4"


def test_cypher_engine_execute_with_vector_rerank(vector_env):
    """Test CypherEngine.execute_with_vector_rerank basic functionality."""
    from lance_graph import CypherEngine

    config, datasets, _ = vector_env
    engine = CypherEngine(config, datasets)

    results = engine.execute_with_vector_rerank(
        "MATCH (d:Document) WHERE d.category = 'tech' RETURN d.id, d.name, d.embedding",
        VectorSearch("d.embedding")
        .query_vector([1.0, 0.0, 0.0])
        .metric(DistanceMetric.L2)
        .top_k(2),
    )

    data = results.to_pydict()
    assert len(data["d.name"]) == 2
    assert data["d.name"][0] == "Doc1"


def test_cypher_engine_vs_cypher_query_vector_rerank_equivalence(vector_env):
    """Test that CypherEngine produces same results as CypherQuery for vector rerank."""
    from lance_graph import CypherEngine

    config, datasets, _ = vector_env

    query_text = (
        "MATCH (d:Document) WHERE d.category = 'tech' RETURN d.id, d.name, d.embedding"
    )
    vector_search = (
        VectorSearch("d.embedding")
        .query_vector([1.0, 0.0, 0.0])
        .metric(DistanceMetric.L2)
        .top_k(2)
    )

    # Execute with CypherQuery
    query = CypherQuery(query_text).with_config(config)
    result_query = query.execute_with_vector_rerank(datasets, vector_search)

    # Execute with CypherEngine
    engine = CypherEngine(config, datasets)
    result_engine = engine.execute_with_vector_rerank(query_text, vector_search)

    # Results should be identical
    assert result_query.to_pydict() == result_engine.to_pydict()


def test_cypher_engine_vector_rerank_multiple_queries(vector_env):
    """Test that CypherEngine efficiently handles multiple vector rerank queries."""
    from lance_graph import CypherEngine

    config, datasets, _ = vector_env
    engine = CypherEngine(config, datasets)

    # Execute multiple different queries using the same cached engine
    results1 = engine.execute_with_vector_rerank(
        "MATCH (d:Document) RETURN d.id, d.name, d.embedding",
        VectorSearch("d.embedding")
        .query_vector([1.0, 0.0, 0.0])
        .metric(DistanceMetric.L2)
        .top_k(2),
    )

    results2 = engine.execute_with_vector_rerank(
        "MATCH (d:Document) WHERE d.category = 'science' "
        "RETURN d.id, d.name, d.embedding",
        VectorSearch("d.embedding")
        .query_vector([0.0, 1.0, 0.0])
        .metric(DistanceMetric.Cosine)
        .top_k(1),
    )

    data1 = results1.to_pydict()
    data2 = results2.to_pydict()

    assert len(data1["d.name"]) == 2
    assert data1["d.name"][0] == "Doc1"

    assert len(data2["d.name"]) == 1
    assert data2["d.name"][0] == "Doc3"
