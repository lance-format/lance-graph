# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Tests for Unity Catalog integration.

Unit tests verify the Python binding classes work correctly.
Integration tests (marked with @pytest.mark.integration) require a running
Unity Catalog server and are skipped by default.
"""

import pytest
from lance_graph import CatalogInfo, SchemaInfo, SqlEngine, TableInfo, UnityCatalog


# ==========================================================================
# Unit tests — verify Python class construction and repr
# ==========================================================================


class TestUnityCatalogConstruction:
    def test_create_client(self):
        """UnityCatalog client can be constructed."""
        uc = UnityCatalog("http://localhost:8080/api/2.1/unity-catalog")
        assert repr(uc) == "UnityCatalog(...)"

    def test_create_client_with_token(self):
        uc = UnityCatalog(
            "http://localhost:8080/api/2.1/unity-catalog",
            token="my-secret-token",
        )
        assert repr(uc) == "UnityCatalog(...)"

    def test_create_client_with_timeout(self):
        uc = UnityCatalog(
            "http://localhost:8080/api/2.1/unity-catalog",
            timeout=60,
        )
        assert repr(uc) == "UnityCatalog(...)"

    def test_create_client_with_all_options(self):
        uc = UnityCatalog(
            "http://localhost:8080/api/2.1/unity-catalog",
            token="tok",
            timeout=30,
        )
        assert repr(uc) == "UnityCatalog(...)"


class TestConnectionErrors:
    def test_list_catalogs_connection_refused(self):
        """Connecting to a non-existent server raises RuntimeError."""
        uc = UnityCatalog("http://localhost:1/api/2.1/unity-catalog", timeout=1)
        with pytest.raises(RuntimeError, match="connection error|Connection refused|error"):
            uc.list_catalogs()

    def test_list_schemas_connection_refused(self):
        uc = UnityCatalog("http://localhost:1/api/2.1/unity-catalog", timeout=1)
        with pytest.raises(RuntimeError):
            uc.list_schemas("unity")

    def test_list_tables_connection_refused(self):
        uc = UnityCatalog("http://localhost:1/api/2.1/unity-catalog", timeout=1)
        with pytest.raises(RuntimeError):
            uc.list_tables("unity", "default")

    def test_get_table_connection_refused(self):
        uc = UnityCatalog("http://localhost:1/api/2.1/unity-catalog", timeout=1)
        with pytest.raises(RuntimeError):
            uc.get_table("unity", "default", "marksheet")

    def test_create_sql_engine_connection_refused(self):
        uc = UnityCatalog("http://localhost:1/api/2.1/unity-catalog", timeout=1)
        with pytest.raises((RuntimeError, ValueError)):
            uc.create_sql_engine("unity", "default")


# ==========================================================================
# Integration tests — require a running UC server
# ==========================================================================


def _uc_available():
    """Check if a UC server is running on localhost:8080."""
    try:
        uc = UnityCatalog(
            "http://localhost:8080/api/2.1/unity-catalog", timeout=2
        )
        uc.list_catalogs()
        return True
    except Exception:
        return False


@pytest.fixture
def uc():
    """Connect to local OSS Unity Catalog."""
    if not _uc_available():
        pytest.skip("Unity Catalog server not available on localhost:8080")
    return UnityCatalog("http://localhost:8080/api/2.1/unity-catalog")


@pytest.mark.integration
class TestUnityCatalogBrowsing:
    def test_list_catalogs(self, uc):
        catalogs = uc.list_catalogs()
        assert len(catalogs) > 0
        assert isinstance(catalogs[0], CatalogInfo)
        assert catalogs[0].name
        assert "CatalogInfo" in repr(catalogs[0])

    def test_list_schemas(self, uc):
        catalogs = uc.list_catalogs()
        schemas = uc.list_schemas(catalogs[0].name)
        assert len(schemas) > 0
        assert isinstance(schemas[0], SchemaInfo)
        assert schemas[0].name
        assert schemas[0].catalog_name == catalogs[0].name
        assert "SchemaInfo" in repr(schemas[0])

    def test_list_tables(self, uc):
        catalogs = uc.list_catalogs()
        schemas = uc.list_schemas(catalogs[0].name)
        tables = uc.list_tables(catalogs[0].name, schemas[0].name)
        assert len(tables) > 0
        assert isinstance(tables[0], TableInfo)
        assert tables[0].name
        assert tables[0].catalog_name == catalogs[0].name
        assert tables[0].schema_name == schemas[0].name
        assert tables[0].num_columns >= 0
        assert "TableInfo" in repr(tables[0])

    def test_get_table(self, uc):
        catalogs = uc.list_catalogs()
        schemas = uc.list_schemas(catalogs[0].name)
        tables = uc.list_tables(catalogs[0].name, schemas[0].name)
        if not tables:
            pytest.skip("No tables in the first schema")

        table = uc.get_table(
            catalogs[0].name, schemas[0].name, tables[0].name
        )
        assert table.name == tables[0].name
        assert table.num_columns > 0
        cols = table.columns()
        assert len(cols) == table.num_columns
        assert all("name" in c and "type_name" in c for c in cols)

    def test_table_info_properties(self, uc):
        catalogs = uc.list_catalogs()
        schemas = uc.list_schemas(catalogs[0].name)
        tables = uc.list_tables(catalogs[0].name, schemas[0].name)
        if not tables:
            pytest.skip("No tables")

        t = tables[0]
        assert t.table_type in ("MANAGED", "EXTERNAL")
        assert isinstance(t.data_source_format, str)


@pytest.mark.integration
class TestUnityCatalogSqlEngine:
    def test_create_sql_engine(self, uc):
        catalogs = uc.list_catalogs()
        schemas = uc.list_schemas(catalogs[0].name)
        engine = uc.create_sql_engine(catalogs[0].name, schemas[0].name)
        assert isinstance(engine, SqlEngine)
        assert "SqlEngine" in repr(engine)
