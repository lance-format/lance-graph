from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest
from knowledge_graph.config import KnowledgeGraphConfig, build_graph_config_from_mapping
from lance_graph import CypherQuery


def test_build_graph_config_from_mapping_supports_simple_nodes():
    mapping = {"nodes": {"Person": "person_id"}}
    config = build_graph_config_from_mapping(mapping)

    table = pa.table(
        {
            "person_id": [1, 2],
            "name": ["Alice", "Bob"],
        }
    )

    data = (
        CypherQuery("MATCH (p:Person) RETURN p.person_id AS id")
        .with_config(config)
        .execute({"Person": table})
        .to_pydict()
    )
    assert data["id"] == [1, 2]


def test_build_graph_config_from_mapping_requires_id_field():
    with pytest.raises(ValueError):
        build_graph_config_from_mapping({"nodes": {"Person": {}}})


class TestKnowledgeGraphConfigPathSupport:
    def test_from_root_with_path(self, tmp_path):
        """Test from_root with a Path object."""
        config = KnowledgeGraphConfig.from_root(tmp_path)
        assert isinstance(config.storage_path, Path)
        assert config.storage_path == tmp_path
        assert config.resolved_schema_path() == tmp_path / "graph.yaml"

    def test_from_root_with_local_string(self):
        """Test from_root with a local path string."""
        path_str = "/tmp/graph"
        config = KnowledgeGraphConfig.from_root(path_str)
        assert isinstance(config.storage_path, Path)
        assert str(config.storage_path) == path_str
        assert config.resolved_schema_path() == Path(path_str) / "graph.yaml"

    def test_from_root_with_s3_uri(self):
        """Test from_root with an S3 URI string."""
        uri = "s3://bucket/graph"
        config = KnowledgeGraphConfig.from_root(uri)
        assert isinstance(config.storage_path, str)
        assert config.storage_path == uri
        assert config.resolved_schema_path() == "s3://bucket/graph/graph.yaml"

    def test_ensure_directories_s3_noop(self):
        """Test ensure_directories is a no-op for S3 URIs."""
        uri = "s3://bucket/graph"
        config = KnowledgeGraphConfig.from_root(uri)
        # Should not raise exception or try to make directories
        config.ensure_directories()

    def test_resolved_schema_path_logic(self, tmp_path):
        """Test resolved_schema_path logic for mixed types."""
        # Case 1: Path object
        config = KnowledgeGraphConfig(storage_path=tmp_path)
        assert config.resolved_schema_path() == tmp_path / "graph.yaml"

        # Case 2: String local path
        config = KnowledgeGraphConfig(storage_path=str(tmp_path))
        assert config.resolved_schema_path() == tmp_path / "graph.yaml"

        # Case 3: S3 URI
        config = KnowledgeGraphConfig(storage_path="s3://bucket/prefix")
        assert config.resolved_schema_path() == "s3://bucket/prefix/graph.yaml"

        # Case 4: Explicit schema path
        explicit = tmp_path / "custom.yaml"
        config = KnowledgeGraphConfig(storage_path=tmp_path, schema_path=explicit)
        assert config.resolved_schema_path() == explicit

    @patch("knowledge_graph.config.pyarrow")
    def test_load_schema_payload_s3(self, mock_pyarrow):
        """Test loading schema from S3 URI."""
        uri = "s3://bucket/graph"
        config = KnowledgeGraphConfig.from_root(uri)

        # Mock filesystem
        mock_fs = MagicMock()
        mock_pyarrow.fs.FileSystem.from_uri.return_value = (
            mock_fs,
            "bucket/graph/graph.yaml",
        )

        # Mock file content
        mock_input_stream = MagicMock()
        mock_input_stream.__enter__.return_value = mock_input_stream
        mock_input_stream.read.return_value = b"nodes:\n  Person: id"
        mock_fs.open_input_stream.return_value = mock_input_stream

        payload = config._load_schema_payload()

        assert payload == {"nodes": {"Person": "id"}}
        mock_pyarrow.fs.FileSystem.from_uri.assert_called_with(
            "s3://bucket/graph/graph.yaml"
        )
        mock_fs.open_input_stream.assert_called_with("bucket/graph/graph.yaml")
