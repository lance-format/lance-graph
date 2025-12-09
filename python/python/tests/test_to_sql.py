# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Tests for the to_sql API that converts Cypher queries to SQL."""

import pyarrow as pa
import pytest
from lance_graph import CypherQuery, GraphConfig


@pytest.fixture
def knowledge_graph_env():
    """Create a complex knowledge graph with multiple entity types and relationships."""
    # Authors and their publications
    authors_table = pa.table(
        {
            "author_id": [1, 2, 3, 4, 5],
            "name": [
                "Alice Chen",
                "Bob Smith",
                "Carol Wang",
                "David Lee",
                "Eve Martinez",
            ],
            "institution": ["MIT", "Stanford", "CMU", "Berkeley", "MIT"],
            "h_index": [45, 38, 52, 41, 29],
            "country": ["USA", "USA", "USA", "USA", "Spain"],
        }
    )

    papers_table = pa.table(
        {
            "paper_id": [101, 102, 103, 104, 105, 106],
            "title": [
                "Deep Learning Advances",
                "Graph Neural Networks",
                "Transformer Architecture",
                "Reinforcement Learning",
                "Computer Vision Methods",
                "Natural Language Processing",
            ],
            "year": [2020, 2021, 2019, 2022, 2021, 2020],
            "citations": [450, 320, 890, 210, 380, 520],
            "venue": ["NeurIPS", "ICML", "NeurIPS", "ICLR", "CVPR", "ACL"],
        }
    )

    authorship_table = pa.table(
        {
            "author_id": [1, 1, 2, 2, 3, 3, 4, 5, 5],
            "paper_id": [101, 102, 102, 103, 103, 104, 105, 105, 106],
            "author_position": [1, 1, 2, 1, 2, 1, 1, 2, 1],
        }
    )

    citations_table = pa.table(
        {
            "citing_paper_id": [102, 103, 104, 104, 105, 106],
            "cited_paper_id": [101, 101, 102, 103, 103, 101],
        }
    )

    config = (
        GraphConfig.builder()
        .with_node_label("Author", "author_id")
        .with_node_label("Paper", "paper_id")
        .with_relationship("AUTHORED", "author_id", "paper_id")
        .with_relationship("CITES", "citing_paper_id", "cited_paper_id")
        .build()
    )

    datasets = {
        "Author": authors_table,
        "Paper": papers_table,
        "AUTHORED": authorship_table,
        "CITES": citations_table,
    }

    return config, datasets


def test_multi_hop_relationship_with_aggregation(knowledge_graph_env):
    """Test complex multi-hop query with aggregation and filtering.

    Find authors who have written highly cited papers (>400 citations) and count
    how many such papers they have, filtering for prolific authors.
    """
    config, datasets = knowledge_graph_env
    query = CypherQuery(
        """
        MATCH (a:Author)-[:AUTHORED]->(p:Paper)
        WHERE p.citations > 400
        RETURN a.name, a.institution, COUNT(*) AS high_impact_papers
        ORDER BY high_impact_papers DESC
        """
    ).with_config(config)

    sql = query.to_sql(datasets)

    assert isinstance(sql, str)
    sql_upper = sql.upper()
    assert "SELECT" in sql_upper
    assert "JOIN" in sql_upper
    assert "WHERE" in sql_upper
    assert "COUNT" in sql_upper
    assert "GROUP BY" in sql_upper
    assert "ORDER BY" in sql_upper


def test_citation_network_analysis(knowledge_graph_env):
    """Test citation network traversal with multiple joins.

    Find papers that cite other papers, along with author information,
    filtered by venue and year range.
    """
    config, datasets = knowledge_graph_env
    query = CypherQuery(
        """
        MATCH (citing:Paper)-[:CITES]->(cited:Paper)
        WHERE citing.year >= 2020 AND citing.venue = 'NeurIPS'
        RETURN citing.title, cited.title, citing.year, cited.citations
        ORDER BY cited.citations DESC
        LIMIT 10
        """
    ).with_config(config)

    sql = query.to_sql(datasets)

    assert isinstance(sql, str)
    sql_upper = sql.upper()
    assert "SELECT" in sql_upper
    assert "JOIN" in sql_upper
    assert "WHERE" in sql_upper
    assert "ORDER BY" in sql_upper
    assert "LIMIT" in sql_upper


def test_collaborative_network_query(knowledge_graph_env):
    """Test finding collaboration patterns through shared papers.

    Find pairs of authors who have co-authored papers, with filtering
    on institution and h-index.
    """
    config, datasets = knowledge_graph_env
    query = CypherQuery(
        """
        MATCH (a1:Author)-[:AUTHORED]->(p:Paper)<-[:AUTHORED]-(a2:Author)
        WHERE a1.author_id < a2.author_id
          AND a1.institution = 'MIT'
          AND a2.h_index > 30
        RETURN DISTINCT a1.name, a2.name, p.title, p.year
        ORDER BY p.year DESC
        """
    ).with_config(config)

    sql = query.to_sql(datasets)

    assert isinstance(sql, str)
    sql_upper = sql.upper()
    assert "SELECT" in sql_upper
    # DISTINCT may be converted to GROUP BY by the SQL unparser
    assert "DISTINCT" in sql_upper or "GROUP BY" in sql_upper
    assert "JOIN" in sql_upper
    assert "WHERE" in sql_upper
    assert "ORDER BY" in sql_upper


def test_parameterized_complex_query(knowledge_graph_env):
    """Test complex query with multiple parameters.

    Find authors from a specific country with papers above a citation threshold,
    published in recent years.
    """
    config, datasets = knowledge_graph_env
    query = (
        CypherQuery(
            """
            MATCH (a:Author)-[:AUTHORED]->(p:Paper)
            WHERE a.country = $country
              AND p.citations > $min_citations
              AND p.year >= $min_year
            RETURN a.name, a.h_index, p.title, p.citations
            ORDER BY p.citations DESC, a.h_index DESC
            """
        )
        .with_config(config)
        .with_parameter("country", "USA")
        .with_parameter("min_citations", 300)
        .with_parameter("min_year", 2020)
    )

    sql = query.to_sql(datasets)

    assert isinstance(sql, str)
    sql_upper = sql.upper()
    assert "SELECT" in sql_upper
    assert "JOIN" in sql_upper
    assert "WHERE" in sql_upper
    assert "ORDER BY" in sql_upper


def test_to_sql_without_config_raises_error(knowledge_graph_env):
    """Test that to_sql fails gracefully without config."""
    _, datasets = knowledge_graph_env
    query = CypherQuery("MATCH (a:Author) RETURN a.name")

    with pytest.raises(Exception):
        query.to_sql(datasets)
