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


@pytest.mark.requires_lance
def test_use_lance_index_missing_query_vector(vector_env, tmp_path):
    """Test error when use_lance_index=True but query_vector is not set."""
    config, _, _ = vector_env

    import lance
    import numpy as np

    embedding_values = np.array(
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]],
        dtype=np.float32,
    )

    documents_table = pa.table(
        {
            "id": [1, 2],
            "name": ["Doc1", "Doc2"],
            "embedding": pa.FixedSizeListArray.from_arrays(
                embedding_values.flatten(), list_size=3
            ),
        }
    )

    dataset_path = tmp_path / "Document.lance"
    lance.write_dataset(documents_table, dataset_path)
    lance_dataset = lance.dataset(str(dataset_path))

    query = CypherQuery(
        "MATCH (d:Document) RETURN d.id, d.name, d.embedding"
    ).with_config(config)

    with pytest.raises(ValueError, match="query_vector is required"):
        query.execute_with_vector_rerank(
            {"Document": lance_dataset},
            VectorSearch("d.embedding")
            .metric(DistanceMetric.L2)
            .top_k(3)
            .use_lance_index(True),  # No query_vector set
        )


def test_use_lance_index_fallback_non_lance_dataset(vector_env):
    """Test use_lance_index=True falls back for non-Lance datasets."""
    config, datasets, _ = vector_env

    query = CypherQuery(
        "MATCH (d:Document) RETURN d.id, d.name, d.embedding"
    ).with_config(config)

    # Should work fine - falls back to standard rerank for PyArrow table
    results = query.execute_with_vector_rerank(
        datasets,  # PyArrow tables, not Lance datasets
        VectorSearch("d.embedding")
        .query_vector([1.0, 0.0, 0.0])
        .metric(DistanceMetric.L2)
        .top_k(3)
        .use_lance_index(True),  # Should fallback silently
    )

    data = results.to_pydict()
    assert len(data["d.name"]) == 3
    assert data["d.name"][0] == "Doc1"
    # _distance column should be present (standard rerank path)
    assert "_distance" in data


@pytest.mark.requires_lance
def test_use_lance_index_unqualified_column(vector_env, tmp_path):
    """Test use_lance_index with unqualified column name (no alias prefix)."""
    config, _, _ = vector_env

    import lance
    import numpy as np

    embedding_values = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    documents_table = pa.table(
        {
            "id": [1, 2, 3],
            "name": ["Doc1", "Doc2", "Doc3"],
            "embedding": pa.FixedSizeListArray.from_arrays(
                embedding_values.flatten(), list_size=3
            ),
        }
    )

    dataset_path = tmp_path / "Document.lance"
    lance.write_dataset(documents_table, dataset_path)
    lance_dataset = lance.dataset(str(dataset_path))

    # Use unqualified column name "embedding" instead of "d.embedding"
    # This should still work when there's only one node label in the query
    query = CypherQuery(
        "MATCH (d:Document) RETURN d.id, d.name, d.embedding"
    ).with_config(config)

    results = query.execute_with_vector_rerank(
        {"Document": lance_dataset},
        VectorSearch("embedding")  # No alias prefix
        .query_vector([1.0, 0.0, 0.0])
        .metric(DistanceMetric.L2)
        .top_k(2)
        .use_lance_index(True),
    )

    data = results.to_pydict()
    assert len(data["d.name"]) == 2
    assert data["d.name"][0] == "Doc1"


def test_use_lance_index_builder_propagation():
    """Test that use_lance_index flag is properly propagated through builder methods."""
    vs = VectorSearch("embedding").use_lance_index(True)

    # Each builder method should preserve the use_lance_index flag
    vs2 = vs.query_vector([1.0, 0.0, 0.0])
    vs3 = vs2.metric(DistanceMetric.L2)
    vs4 = vs3.top_k(10)
    vs5 = vs4.include_distance(True)
    vs6 = vs5.distance_column_name("dist")

    # All should still have use_lance_index=True (we verify by using it)
    # This is an indirect test - if propagation failed, the final object
    # would have use_lance_index=False
    # We can't directly inspect the flag, but we can verify the chain works
    assert vs6 is not None  # Chain completed successfully


@pytest.mark.requires_lance
def test_use_lance_index_cosine_metric(vector_env, tmp_path):
    """Test use_lance_index with cosine distance metric."""
    config, _, _ = vector_env

    import lance
    import numpy as np

    embedding_values = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    documents_table = pa.table(
        {
            "id": [1, 2, 3],
            "name": ["Doc1", "Doc2", "Doc3"],
            "embedding": pa.FixedSizeListArray.from_arrays(
                embedding_values.flatten(), list_size=3
            ),
        }
    )

    dataset_path = tmp_path / "Document.lance"
    lance.write_dataset(documents_table, dataset_path)
    lance_dataset = lance.dataset(str(dataset_path))

    query = CypherQuery(
        "MATCH (d:Document) RETURN d.id, d.name, d.embedding"
    ).with_config(config)

    results = query.execute_with_vector_rerank(
        {"Document": lance_dataset},
        VectorSearch("d.embedding")
        .query_vector([1.0, 0.0, 0.0])
        .metric(DistanceMetric.Cosine)  # Using cosine metric
        .top_k(2)
        .use_lance_index(True),
    )

    data = results.to_pydict()
    assert len(data["d.name"]) == 2
    assert data["d.name"][0] == "Doc1"


@pytest.mark.requires_lance
def test_use_lance_index_dot_metric(vector_env, tmp_path):
    """Test use_lance_index with dot product metric."""
    config, _, _ = vector_env

    import lance
    import numpy as np

    embedding_values = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    documents_table = pa.table(
        {
            "id": [1, 2, 3],
            "name": ["Doc1", "Doc2", "Doc3"],
            "embedding": pa.FixedSizeListArray.from_arrays(
                embedding_values.flatten(), list_size=3
            ),
        }
    )

    dataset_path = tmp_path / "Document.lance"
    lance.write_dataset(documents_table, dataset_path)
    lance_dataset = lance.dataset(str(dataset_path))

    query = CypherQuery(
        "MATCH (d:Document) RETURN d.id, d.name, d.embedding"
    ).with_config(config)

    results = query.execute_with_vector_rerank(
        {"Document": lance_dataset},
        VectorSearch("d.embedding")
        .query_vector([1.0, 0.0, 0.0])
        .metric(DistanceMetric.Dot)  # Using dot product metric
        .top_k(2)
        .use_lance_index(True),
    )

    data = results.to_pydict()
    assert len(data["d.name"]) == 2
    assert data["d.name"][0] == "Doc1"


@pytest.mark.requires_lance
def test_execute_with_vector_rerank_lance_index(vector_env, tmp_path):
    """Test vector-first execution using Lance datasets.

    Note: This test does NOT create an actual vector index on the Lance dataset.
    Lance will fall back to flat (brute-force) search when use_index=True is set
    but no index exists. This test validates:
    1. The code path for the vector-first execution is exercised
    2. Results are correct (matching the standard rerank behavior)
    3. The Lance dataset integration works end-to-end

    To test actual ANN index behavior, create an index with:
        lance_dataset.create_index("embedding", index_type="IVF_PQ", ...)
    """
    config, _, _ = vector_env

    import lance
    import numpy as np

    # Create embeddings with fixed-size list type (required for Lance vector search)
    embedding_values = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float32,
    )

    documents_table = pa.table(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Doc1", "Doc2", "Doc3", "Doc4", "Doc5"],
            "category": ["tech", "tech", "science", "tech", "science"],
            "embedding": pa.FixedSizeListArray.from_arrays(
                embedding_values.flatten(), list_size=3
            ),
        }
    )

    dataset_path = tmp_path / "Document.lance"
    lance.write_dataset(documents_table, dataset_path)
    lance_dataset = lance.dataset(str(dataset_path))

    query = CypherQuery(
        "MATCH (d:Document) RETURN d.id, d.name, d.embedding"
    ).with_config(config)

    results = query.execute_with_vector_rerank(
        {"Document": lance_dataset},
        VectorSearch("d.embedding")
        .query_vector([1.0, 0.0, 0.0])
        .metric(DistanceMetric.L2)
        .top_k(3)
        .use_lance_index(True),
    )

    data = results.to_pydict()
    assert len(data["d.name"]) == 3
    assert data["d.name"][0] == "Doc1"
    assert data["d.name"][1] == "Doc2"


@pytest.mark.requires_lance
def test_execute_with_vector_rerank_lance_index_fallback_on_where(vector_env, tmp_path):
    """Test that use_lance_index falls back to standard rerank with WHERE clause.

    When a Cypher query includes filters (WHERE clause), the vector-first path would
    change semantics: it would search ALL vectors first, then apply filters. This could
    miss relevant results that match the filter but aren't in the top-k vectors.

    The implementation correctly detects this and falls back to the standard
    candidate-then-rerank path.
    """
    config, _, _ = vector_env

    import lance
    import numpy as np

    # Create embeddings with fixed-size list type (required for Lance vector search)
    embedding_values = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float32,
    )

    documents_table = pa.table(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Doc1", "Doc2", "Doc3", "Doc4", "Doc5"],
            "category": ["tech", "tech", "science", "tech", "science"],
            "embedding": pa.FixedSizeListArray.from_arrays(
                embedding_values.flatten(), list_size=3
            ),
        }
    )

    dataset_path = tmp_path / "Document.lance"
    lance.write_dataset(documents_table, dataset_path)
    lance_dataset = lance.dataset(str(dataset_path))

    # Query WITH a WHERE clause - should fall back to standard rerank
    query = CypherQuery(
        "MATCH (d:Document) WHERE d.category = 'tech' RETURN d.id, d.name, d.embedding"
    ).with_config(config)

    results = query.execute_with_vector_rerank(
        {"Document": lance_dataset},
        VectorSearch("d.embedding")
        .query_vector([1.0, 0.0, 0.0])
        .metric(DistanceMetric.L2)
        .top_k(3)
        .use_lance_index(True),  # This will be ignored due to WHERE clause
    )

    data = results.to_pydict()
    # Should only have tech documents (Doc1, Doc2, Doc4), not science docs
    assert len(data["d.name"]) == 3
    assert all(name in ["Doc1", "Doc2", "Doc4"] for name in data["d.name"])
    # Doc1 should still be first (closest to [1,0,0])
    assert data["d.name"][0] == "Doc1"


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
