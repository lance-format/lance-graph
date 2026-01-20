# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Tests for Lance-namespace integration (automatic table resolution)."""

import pyarrow as pa
import pytest
from knowledge_graph.config import KnowledgeGraphConfig
from knowledge_graph.store import LanceGraphStore
from lance_graph import CypherQuery, execute_with_store


@pytest.fixture
def config(tmp_path):
    """Create a test configuration with temporary storage path."""
    return KnowledgeGraphConfig(
        storage_path=tmp_path / "test_storage",
        schema_path=tmp_path / "graph.yaml",
    )


@pytest.fixture
def store(config):
    """Create a LanceGraphStore instance."""
    return LanceGraphStore(config)


class TestInferGraphConfig:
    """Tests for infer_graph_config method."""

    def test_infer_graph_config_single_id_field_creates_node(self, store):
        """Test that tables with single *_id field are inferred as nodes."""
        person_table = pa.table({"person_id": [1, 2], "name": ["Alice", "Bob"]})
        store.write_tables({"Person": person_table})

        graph_config = store.infer_graph_config()

        node_labels = graph_config.node_labels()
        assert "Person" in node_labels

    def test_infer_graph_config_two_id_fields_creates_relationship(self, store):
        """Test that tables with 2+ *_id fields are inferred as relationships."""
        knows_table = pa.table({"person_id": [1], "friend_id": [2]})
        store.write_tables({"KNOWS": knows_table})

        graph_config = store.infer_graph_config()

        rel_types = graph_config.relationship_types()
        assert "KNOWS" in rel_types

    def test_infer_graph_config_handles_multiple_tables(self, store):
        """Test inference with both node and relationship tables."""
        person_table = pa.table({"person_id": [1, 2], "name": ["Alice", "Bob"]})
        company_table = pa.table({"company_id": [101, 102], "name": ["TechCorp", "DataInc"]})
        works_for_table = pa.table({"person_id": [1, 2], "company_id": [101, 102]})

        store.write_tables({
            "Person": person_table,
            "Company": company_table,
            "WORKS_FOR": works_for_table,
        })

        graph_config = store.infer_graph_config()

        node_labels = graph_config.node_labels()
        rel_types = graph_config.relationship_types()

        assert "Person" in node_labels
        assert "Company" in node_labels
        assert "WORKS_FOR" in rel_types

    def test_infer_graph_config_empty_store(self, store):
        """Test that empty store returns valid but empty config."""
        store.ensure_layout()

        graph_config = store.infer_graph_config()

        assert graph_config.node_labels() == []
        assert graph_config.relationship_types() == []

    def test_infer_graph_config_ignores_tables_without_id_fields(self, store):
        """Test that tables without *_id fields are ignored."""
        metadata_table = pa.table({"timestamp": [1, 2], "value": ["a", "b"]})
        person_table = pa.table({"person_id": [1, 2], "name": ["Alice", "Bob"]})

        store.write_tables({
            "Metadata": metadata_table,
            "Person": person_table,
        })

        graph_config = store.infer_graph_config()

        node_labels = graph_config.node_labels()
        assert "Person" in node_labels
        assert "Metadata" not in node_labels

    def test_infer_graph_config_uses_first_two_id_fields_for_relationships(self, store):
        """Test that relationships use first two *_id fields for source/target."""
        # Table with 3 id fields - should use first two for source/target
        rel_table = pa.table({
            "person_id": [1],
            "company_id": [101],
            "department_id": [5]  # Third ID field should be ignored for relationship
        })
        store.write_tables({"WORKS_FOR": rel_table})

        graph_config = store.infer_graph_config()

        # Just verify it doesn't crash and creates a relationship
        rel_types = graph_config.relationship_types()
        assert "WORKS_FOR" in rel_types


class TestExecuteWithStore:
    """Tests for execute_with_store function."""

    def test_execute_with_store_basic_query(self, store):
        """Test execute_with_store with a simple node query."""
        person_table = pa.table({"person_id": [1, 2], "name": ["Alice", "Bob"]})
        store.write_tables({"Person": person_table})

        query = CypherQuery("MATCH (p:Person) RETURN p.name")
        result = execute_with_store(query, store)

        assert result.num_rows == 2
        assert "p.name" in result.column_names
        names = result.column("p.name").to_pylist()
        assert set(names) == {"Alice", "Bob"}

    def test_execute_with_store_with_relationship_query(self, store):
        """Test execute_with_store with a relationship query."""
        person_table = pa.table({"person_id": [1, 2], "name": ["Alice", "Bob"]})
        knows_table = pa.table({"person_id": [1], "friend_id": [2]})

        store.write_tables({
            "Person": person_table,
            "KNOWS": knows_table,
        })

        query = CypherQuery("MATCH (p:Person)-[:KNOWS]->(f:Person) RETURN p.name, f.name")
        result = execute_with_store(query, store)

        assert result.num_rows == 1
        assert result.column("p.name").to_pylist() == ["Alice"]
        assert result.column("f.name").to_pylist() == ["Bob"]

    def test_execute_with_store_uses_inferred_config(self, store):
        """Test that execute_with_store uses inferred config when no YAML exists."""
        person_table = pa.table({"person_id": [1, 2], "name": ["Alice", "Bob"]})
        store.write_tables({"Person": person_table})

        # No YAML schema exists, should fall back to inference
        query = CypherQuery("MATCH (p:Person) RETURN p.name")
        result = execute_with_store(query, store)

        assert result.num_rows == 2

    def test_execute_with_store_with_explicit_config(self, store):
        """Test execute_with_store with explicitly provided config."""
        from lance_graph import GraphConfig

        person_table = pa.table({"person_id": [1, 2], "name": ["Alice", "Bob"]})
        store.write_tables({"Person": person_table})

        # Create explicit config
        config = GraphConfig.builder().with_node_label("Person", "person_id").build()

        query = CypherQuery("MATCH (p:Person) RETURN p.name")
        result = execute_with_store(query, store, config=config)

        assert result.num_rows == 2

    def test_execute_with_store_loads_only_required_tables(self, store):
        """Test that execute_with_store only loads tables needed for the query."""
        from unittest.mock import patch

        person_table = pa.table({"person_id": [1, 2], "name": ["Alice", "Bob"]})
        company_table = pa.table({"company_id": [101, 102], "name": ["TechCorp", "DataInc"]})

        store.write_tables({
            "Person": person_table,
            "Company": company_table,
        })

        # Patch load_tables to verify which tables are loaded
        original_load_tables = store.load_tables
        loaded_tables = []

        def track_load_tables(names=None):
            loaded_tables.append(names)
            return original_load_tables(names)

        with patch.object(store, "load_tables", side_effect=track_load_tables):
            query = CypherQuery("MATCH (p:Person) RETURN p.name")
            result = execute_with_store(query, store)

            assert result.num_rows == 2

            # Verify load_tables was called
            assert len(loaded_tables) == 1
            loaded_names = set(loaded_tables[0])  # First call's argument
            assert "Person" in loaded_names
            # Company should not be loaded since it's not in the query
            assert "Company" not in loaded_names

    def test_execute_with_store_multiple_node_types(self, store):
        """Test execute_with_store with query using multiple node types."""
        person_table = pa.table({"person_id": [1, 2], "name": ["Alice", "Bob"]})
        company_table = pa.table({"company_id": [101, 102], "name": ["TechCorp", "DataInc"]})
        works_for_table = pa.table({"person_id": [1, 2], "company_id": [101, 102]})

        store.write_tables({
            "Person": person_table,
            "Company": company_table,
            "WORKS_FOR": works_for_table,
        })

        query = CypherQuery(
            "MATCH (p:Person)-[:WORKS_FOR]->(c:Company) RETURN p.name, c.name"
        )
        result = execute_with_store(query, store)

        assert result.num_rows == 2


class TestIntegration:
    """Integration tests combining infer_graph_config and execute_with_store."""

    def test_end_to_end_workflow(self, store):
        """Test complete workflow: write tables, infer config, execute query."""
        # Setup tables
        person_table = pa.table({
            "person_id": [1, 2, 3],
            "name": ["Alice", "Bob", "Carol"],
        })
        knows_table = pa.table({
            "person_id": [1, 1, 2],
            "friend_id": [2, 3, 3],
        })

        store.write_tables({
            "Person": person_table,
            "KNOWS": knows_table,
        })

        # Execute query without explicit config
        query = CypherQuery("MATCH (p:Person)-[:KNOWS]->(f:Person) RETURN p.name, f.name")
        result = execute_with_store(query, store)

        assert result.num_rows == 3
        assert set(result.column("p.name").to_pylist()) == {"Alice", "Bob"}
        assert set(result.column("f.name").to_pylist()) == {"Bob", "Carol"}

    def test_empty_store_query(self, store):
        """Test querying an empty store returns empty results."""
        store.ensure_layout()

        query = CypherQuery("MATCH (p:Person) RETURN p.name")

        # Should not crash, but may return empty results or raise appropriate error
        # depending on query engine behavior
        try:
            result = execute_with_store(query, store)
            # If it doesn't raise, verify it's empty
            assert result.num_rows == 0
        except FileNotFoundError:
            # It's also acceptable to raise FileNotFoundError for missing tables
            pass
