# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import pyarrow as pa
import pytest
from lance_graph import CypherEngine, CypherQuery, GraphConfig


@pytest.fixture
def graph_env():
    """Create sample graph data for testing."""
    people_table = pa.table(
        {
            "person_id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Carol", "David"],
            "age": [28, 34, 29, 42],
            "city": ["New York", "San Francisco", "New York", "Chicago"],
        }
    )

    companies_table = pa.table(
        {
            "company_id": [101, 102, 103],
            "company_name": ["TechCorp", "DataInc", "CloudSoft"],
            "industry": ["Technology", "Analytics", "Cloud"],
        }
    )

    employment_table = pa.table(
        {
            "person_id": [1, 2, 3, 4],
            "company_id": [101, 101, 102, 103],
            "position": ["Engineer", "Designer", "Manager", "Director"],
            "salary": [120000, 95000, 130000, 180000],
        }
    )

    friendship_table = pa.table(
        {
            "person1_id": [1, 1, 2, 3],
            "person2_id": [2, 3, 4, 4],
            "friendship_type": ["close", "casual", "close", "casual"],
            "years_known": [5, 2, 3, 1],
        }
    )

    config = (
        GraphConfig.builder()
        .with_node_label("Person", "person_id")
        .with_node_label("Company", "company_id")
        .with_relationship("WORKS_FOR", "person_id", "company_id")
        .with_relationship("FRIEND_OF", "person1_id", "person2_id")
        .build()
    )

    datasets = {
        "Person": people_table,
        "Company": companies_table,
        "WORKS_FOR": employment_table,
        "FRIEND_OF": friendship_table,
    }

    return config, datasets


def test_cypher_engine_basic_query(graph_env):
    """Test basic query execution with CypherEngine."""
    config, datasets = graph_env
    engine = CypherEngine(config, datasets)

    result = engine.execute("MATCH (p:Person) RETURN p.name, p.age")
    data = result.to_pydict()

    assert set(data.keys()) == {"p.name", "p.age"}
    assert len(data["p.name"]) == 4
    assert "Alice" in set(data["p.name"])


def test_cypher_engine_filtered_query(graph_env):
    """Test filtered query with WHERE clause."""
    config, datasets = graph_env
    engine = CypherEngine(config, datasets)

    result = engine.execute("MATCH (p:Person) WHERE p.age > 30 RETURN p.name, p.age")
    data = result.to_pydict()

    assert len(data["p.name"]) == 2
    assert set(data["p.name"]) == {"Bob", "David"}
    assert all(age > 30 for age in data["p.age"])


def test_cypher_engine_relationship_query(graph_env):
    """Test query with relationship traversal."""
    config, datasets = graph_env
    engine = CypherEngine(config, datasets)

    result = engine.execute(
        "MATCH (p:Person)-[:WORKS_FOR]->(c:Company) "
        "RETURN p.name AS person_name, c.company_name AS company_name"
    )
    data = result.to_pydict()

    assert len(data["person_name"]) == 4
    assert "Alice" in data["person_name"]
    assert "TechCorp" in data["company_name"]


def test_cypher_engine_multiple_queries(graph_env):
    """Test that catalog is reused across multiple queries."""
    config, datasets = graph_env
    engine = CypherEngine(config, datasets)

    # Execute multiple different queries
    result1 = engine.execute("MATCH (p:Person) WHERE p.age > 30 RETURN p.name")
    result2 = engine.execute("MATCH (p:Person) WHERE p.city = 'New York' RETURN p.name")
    result3 = engine.execute("MATCH (p:Person) RETURN count(*) as total")

    data1 = result1.to_pydict()
    data2 = result2.to_pydict()
    data3 = result3.to_pydict()

    assert len(data1["p.name"]) == 2
    assert len(data2["p.name"]) == 2
    assert data3["total"][0] == 4


def test_cypher_engine_aggregation(graph_env):
    """Test aggregation queries."""
    config, datasets = graph_env
    engine = CypherEngine(config, datasets)

    result = engine.execute(
        "MATCH (p:Person) RETURN count(*) as total, avg(p.age) as avg_age"
    )
    data = result.to_pydict()

    assert data["total"][0] == 4
    # Average of [28, 34, 29, 42] = 33.25
    assert abs(data["avg_age"][0] - 33.25) < 0.01


def test_cypher_engine_vs_cypher_query_equivalence(graph_env):
    """Test that CypherEngine produces same results as CypherQuery."""
    config, datasets = graph_env

    query_text = "MATCH (p:Person) WHERE p.age > 30 RETURN p.name, p.age ORDER BY p.age"

    # Execute with CypherQuery
    query = CypherQuery(query_text).with_config(config)
    result_query = query.execute(datasets)

    # Execute with CypherEngine
    engine = CypherEngine(config, datasets)
    result_engine = engine.execute(query_text)

    # Results should be identical
    assert result_query.to_pydict() == result_engine.to_pydict()


def test_cypher_engine_config_access(graph_env):
    """Test that we can access the engine's config."""
    config, datasets = graph_env
    engine = CypherEngine(config, datasets)

    engine_config = engine.config()

    assert "person" in engine_config.node_labels()  # case-insensitive
    assert "company" in engine_config.node_labels()
