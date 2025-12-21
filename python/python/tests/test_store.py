# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Unit tests for LanceGraphStore."""

from unittest.mock import Mock, patch

import lance
import pyarrow as pa
import pytest
from knowledge_graph.config import KnowledgeGraphConfig
from knowledge_graph.store import LanceGraphStore


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


class TestLanceModuleImport:
    """Tests for Lance module import and validation."""

    def test_get_lance_raises_import_error_when_module_not_found(self, store):
        """Test that missing lance module raises ImportError with helpful message."""
        with patch("knowledge_graph.store.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module named 'lance'")

            with pytest.raises(ImportError) as exc_info:
                store._get_lance()

            assert "Lance module is required but not installed" in str(exc_info.value)
            assert "pip install pylance" in str(exc_info.value)

    def test_get_lance_raises_import_error_when_missing_dataset_method(self, store):
        """Test that lance module without dataset() raises ImportError."""
        with patch("knowledge_graph.store.import_module") as mock_import:
            mock_lance = Mock()
            mock_lance.write_dataset = Mock()
            # Missing dataset method
            delattr(mock_lance, "dataset")
            mock_import.return_value = mock_lance

            with pytest.raises(ImportError) as exc_info:
                store._get_lance()

            assert "missing required dataset APIs" in str(exc_info.value)

    def test_get_lance_only_checks_dataset_method(self, store):
        """Test that lance module only requires dataset() method."""
        with patch("knowledge_graph.store.import_module") as mock_import:
            mock_lance = Mock()
            mock_lance.dataset = Mock()
            mock_lance.write_dataset = Mock()
            mock_import.return_value = mock_lance

            result = store._get_lance()

            assert result == mock_lance

    def test_get_lance_succeeds_with_valid_module(self, store):
        """Test that valid lance module loads successfully."""
        with patch("knowledge_graph.store.import_module") as mock_import:
            mock_lance = Mock()
            mock_lance.write_dataset = Mock()
            mock_lance.dataset = Mock()
            mock_import.return_value = mock_lance

            result = store._get_lance()

            assert result == mock_lance
            mock_import.assert_called_once_with("lance")

    def test_get_lance_caches_result(self, store):
        """Test that _get_lance() caches the module and doesn't re-import."""
        with patch("knowledge_graph.store.import_module") as mock_import:
            mock_lance = Mock()
            mock_lance.write_dataset = Mock()
            mock_lance.dataset = Mock()
            mock_import.return_value = mock_lance

            # First call
            result1 = store._get_lance()
            # Second call
            result2 = store._get_lance()

            assert result1 == result2 == mock_lance
            # import_module should only be called once due to caching
            mock_import.assert_called_once_with("lance")


class TestBasicOperations:
    """Tests for basic store operations."""

    def test_config_property_returns_config(self, store, config):
        """Test that config property returns the stored configuration."""
        assert store.config == config

    def test_root_property_returns_storage_path(self, store, config):
        """Test that root property returns the storage path."""
        assert store.root == config.storage_path

    def test_ensure_layout_creates_directory(self, store, tmp_path):
        """Test that ensure_layout creates the storage directory."""
        storage_path = tmp_path / "test_storage"
        assert not storage_path.exists()

        store.ensure_layout()

        assert storage_path.exists()
        assert storage_path.is_dir()

    def test_ensure_layout_succeeds_when_directory_exists(self, store):
        """Test that ensure_layout is idempotent."""
        store.ensure_layout()
        store.ensure_layout()  # Should not raise

        assert store.root.exists()

    def test_dataset_path_generates_correct_path(self, store):
        """Test that _dataset_path generates paths with .lance extension."""
        path = store._dataset_path("Person")

        assert path == store.root / "Person.lance"
        assert path.suffix == ".lance"

    def test_dataset_path_sanitizes_slashes(self, store):
        """Test that _dataset_path replaces slashes with underscores."""
        path = store._dataset_path("some/nested/name")

        assert path == store.root / "some_nested_name.lance"
        assert "/" not in path.name

    def test_list_datasets_returns_empty_when_root_not_exists(self, store):
        """Test that list_datasets returns empty dict when root doesn't exist."""
        datasets = store.list_datasets()

        assert datasets == {}

    def test_list_datasets_finds_lance_directories(self, store):
        """Test that list_datasets finds .lance directories."""
        store.ensure_layout()
        (store.root / "Person.lance").mkdir()
        (store.root / "Company.lance").mkdir()
        (store.root / "other.txt").touch()  # Should be ignored

        datasets = store.list_datasets()

        assert len(datasets) == 2
        assert "Person" in datasets
        assert "Company" in datasets
        assert datasets["Person"] == store.root / "Person.lance"
        assert datasets["Company"] == store.root / "Company.lance"


class TestWriteTables:
    """Tests for write_tables method."""

    def test_write_tables_creates_storage_directory(self, store):
        """Test that write_tables creates storage directory if it doesn't exist."""
        table = pa.table({"id": [1, 2], "name": ["Alice", "Bob"]})

        store.write_tables({"Person": table})

        assert store.root.exists()
        assert (store.root / "Person.lance").exists()

    def test_write_tables_writes_real_lance_dataset(self, store):
        """Test that write_tables writes a real Lance dataset."""
        table = pa.table({"id": [1, 2], "name": ["Alice", "Bob"]})
        store.ensure_layout()

        store.write_tables({"Person": table})

        # Verify the dataset was written
        dataset_path = store.root / "Person.lance"
        assert dataset_path.exists()
        assert dataset_path.is_dir()

    def test_write_tables_uses_overwrite_mode_when_path_exists(self, store):
        """Test that write_tables uses 'overwrite' mode for existing datasets."""
        table1 = pa.table({"id": [1, 2], "name": ["Alice", "Bob"]})
        table2 = pa.table({"id": [3, 4, 5], "name": ["Carol", "David", "Eve"]})
        store.ensure_layout()

        # First write
        store.write_tables({"Person": table1})
        # Second write should overwrite
        store.write_tables({"Person": table2})

        # Verify overwrite worked - should have 3 rows, not 2
        dataset = lance.dataset(str(store.root / "Person.lance"))
        result = dataset.scanner().to_table()
        assert len(result) == 3

    def test_write_tables_handles_multiple_tables(self, store):
        """Test that write_tables can write multiple tables."""
        person_table = pa.table({"id": [1, 2], "name": ["Alice", "Bob"]})
        company_table = pa.table({"id": [101, 102], "name": ["TechCorp", "DataInc"]})

        store.write_tables({"Person": person_table, "Company": company_table})

        assert (store.root / "Person.lance").exists()
        assert (store.root / "Company.lance").exists()

    def test_write_tables_raises_type_error_for_non_table(self, store):
        """Test that write_tables raises TypeError for non-PyArrow tables."""
        with pytest.raises(TypeError) as exc_info:
            store.write_tables({"Person": {"not": "a table"}})

        assert "must be a pyarrow.Table" in str(exc_info.value)


class TestLoadTables:
    """Tests for load_tables method."""

    def test_load_tables_reads_real_lance_dataset(self, store):
        """Test that load_tables reads a real Lance dataset."""
        # First write a dataset
        table = pa.table({"id": [1, 2], "name": ["Alice", "Bob"]})
        store.ensure_layout()
        store.write_tables({"Person": table})

        # Now load it
        result = store.load_tables(["Person"])

        assert "Person" in result
        loaded_table = result["Person"]
        assert len(loaded_table) == 2
        assert loaded_table.column_names == ["id", "name"]
        assert loaded_table.to_pydict() == {"id": [1, 2], "name": ["Alice", "Bob"]}

    def test_load_tables_returns_pyarrow_table(self, store):
        """Test that load_tables returns PyArrow table instances."""
        table = pa.table({"id": [1, 2], "name": ["Alice", "Bob"]})
        store.ensure_layout()
        store.write_tables({"Person": table})

        result = store.load_tables(["Person"])

        assert isinstance(result["Person"], pa.Table)

    def test_load_tables_loads_all_datasets_when_names_not_provided(self, store):
        """Test that load_tables loads all datasets when names is None."""
        person_table = pa.table({"id": [1, 2], "name": ["Alice", "Bob"]})
        company_table = pa.table({"id": [101, 102], "name": ["TechCorp", "DataInc"]})
        store.ensure_layout()
        store.write_tables({"Person": person_table, "Company": company_table})

        result = store.load_tables()  # No names provided

        assert len(result) == 2
        assert "Person" in result
        assert "Company" in result
        assert len(result["Person"]) == 2
        assert len(result["Company"]) == 2

    def test_load_tables_raises_file_not_found_for_missing_dataset(self, store):
        """Test that load_tables raises FileNotFoundError for non-existent dataset."""
        store.ensure_layout()

        with pytest.raises(FileNotFoundError) as exc_info:
            store.load_tables(["NonExistent"])

        assert "Dataset 'NonExistent' not found" in str(exc_info.value)

    def test_load_tables_loads_multiple_datasets(self, store):
        """Test that load_tables can load multiple datasets."""
        person_table = pa.table({"id": [1, 2], "name": ["Alice", "Bob"]})
        company_table = pa.table({"id": [101, 102], "name": ["TechCorp", "DataInc"]})
        store.ensure_layout()
        store.write_tables({"Person": person_table, "Company": company_table})

        result = store.load_tables(["Person", "Company"])

        assert len(result) == 2
        assert "Person" in result
        assert "Company" in result
        assert result["Person"].to_pydict() == person_table.to_pydict()
        assert result["Company"].to_pydict() == company_table.to_pydict()


class TestIntegration:
    """Integration-style tests combining multiple operations."""

    def test_write_and_list_workflow(self, store):
        """Test write_tables followed by list_datasets."""
        table = pa.table({"id": [1, 2], "name": ["Alice", "Bob"]})
        store.write_tables({"Person": table})

        datasets = store.list_datasets()

        assert "Person" in datasets
        assert datasets["Person"].exists()

    def test_write_then_load_roundtrip(self, store):
        """Test full write->load roundtrip with real Lance datasets."""
        original_person = pa.table(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Carol"],
                "age": [28, 34, 29],
            }
        )
        original_company = pa.table(
            {
                "id": [101, 102],
                "name": ["TechCorp", "DataInc"],
            }
        )

        # Write
        store.write_tables({"Person": original_person, "Company": original_company})

        # Load back
        loaded = store.load_tables()

        assert len(loaded) == 2
        assert loaded["Person"].to_pydict() == original_person.to_pydict()
        assert loaded["Company"].to_pydict() == original_company.to_pydict()

    def test_lance_module_failure_propagates_to_write_tables(self, store):
        """Test that lance import failure in write_tables raises ImportError."""
        with patch("knowledge_graph.store.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module named 'lance'")

            table = pa.table({"id": [1, 2], "name": ["Alice", "Bob"]})

            with pytest.raises(ImportError) as exc_info:
                store.write_tables({"Person": table})

            assert "Lance module is required" in str(exc_info.value)

    def test_lance_module_failure_propagates_to_load_tables(self, store):
        """Test that lance import failure in load_tables raises ImportError."""
        with patch("knowledge_graph.store.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module named 'lance'")

            store.ensure_layout()
            (store.root / "Person.lance").mkdir()

            with pytest.raises(ImportError) as exc_info:
                store.load_tables(["Person"])

            assert "Lance module is required" in str(exc_info.value)


class TestS3Support:
    """Tests for S3/Cloud storage support."""

    @pytest.fixture
    def s3_config(self):
        return KnowledgeGraphConfig(
            storage_path="s3://bucket/graph", schema_path="s3://bucket/graph/graph.yaml"
        )

    @pytest.fixture
    def s3_store(self, s3_config):
        with patch("knowledge_graph.store.fsspec") as mock_fsspec:
            # Setup default mock behavior for init
            mock_fs = Mock()
            mock_fsspec.core.url_to_fs.return_value = (mock_fs, "bucket/graph")
            store = LanceGraphStore(s3_config)
            # Attach the mock to the store instance for tests to configure further
            store._mock_fsspec = mock_fsspec
            store._mock_fs = mock_fs
            return store

    def test_init_with_s3_path(self, s3_store):
        """Test initializing store with S3 path."""
        assert s3_store.root == "s3://bucket/graph"
        assert isinstance(s3_store.root, str)
        # Verify fsspec was called
        s3_store._mock_fsspec.core.url_to_fs.assert_called_with("s3://bucket/graph")

    def test_ensure_layout_s3_noop(self, s3_store):
        """Test ensure_layout calls makedirs on fs."""
        s3_store.ensure_layout()
        s3_store._mock_fs.makedirs.assert_called_with("bucket/graph", exist_ok=True)

    def test_dataset_path_s3(self, s3_store):
        """Test dataset path generation for S3."""
        path = s3_store._dataset_path("Person")
        assert path == "s3://bucket/graph/Person.lance"

    def test_dataset_path_s3_nested(self, s3_store):
        """Test dataset path generation with nested names for S3."""
        path = s3_store._dataset_path("foo/bar")
        assert path == "s3://bucket/graph/foo_bar.lance"

    def test_list_datasets_s3(self, s3_store):
        """Test listing datasets from S3."""
        mock_fs = s3_store._mock_fs
        mock_fs.exists.return_value = True

        # Mock fs.ls results
        info1 = {"name": "bucket/graph/Person.lance", "type": "directory"}
        info2 = {"name": "bucket/graph/Company.lance", "type": "directory"}
        info3 = {"name": "bucket/graph/other.txt", "type": "file"}

        mock_fs.ls.return_value = [info1, info2, info3]

        datasets = s3_store.list_datasets()

        assert len(datasets) == 2
        assert "Person" in datasets
        assert "Company" in datasets
        assert datasets["Person"] == "s3://bucket/graph/Person.lance"
        assert datasets["Company"] == "s3://bucket/graph/Company.lance"

    def test_list_datasets_s3_handles_error(self, s3_store):
        """Test list_datasets handles S3 errors gracefully."""
        mock_fs = s3_store._mock_fs
        # Mocking exists raising exception or returning False
        # If exists raises, it should be caught if it's "not found"
        mock_fs.exists.side_effect = Exception("File not found")

        datasets = s3_store.list_datasets()
        assert datasets == {}
