# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import pyarrow as pa
import pytest
from lance_graph import SqlEngine, SqlQuery


@pytest.fixture
def person_table():
    return pa.table(
        {
            "id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Carol", "David"],
            "age": [28, 34, 29, 42],
            "city": ["New York", "San Francisco", "New York", "Chicago"],
        }
    )


@pytest.fixture
def knows_table():
    return pa.table(
        {
            "src_id": [1, 1, 2, 3],
            "dst_id": [2, 3, 4, 4],
            "since_year": [2015, 2018, 2020, 2021],
        }
    )


@pytest.fixture
def datasets(person_table, knows_table):
    return {"person": person_table, "knows": knows_table}


# ==========================================================================
# SqlQuery tests
# ==========================================================================


class TestSqlQuery:
    def test_basic_select(self, datasets):
        query = SqlQuery("SELECT name, age FROM person WHERE age > 30 ORDER BY age")
        result = query.execute(datasets)
        data = result.to_pydict()

        assert data["name"] == ["Bob", "David"]
        assert data["age"] == [34, 42]

    def test_select_star(self, datasets):
        query = SqlQuery("SELECT * FROM person")
        result = query.execute(datasets)
        assert result.num_rows == 4
        assert result.num_columns == 4

    def test_limit(self, datasets):
        query = SqlQuery("SELECT name FROM person ORDER BY name LIMIT 2")
        result = query.execute(datasets)
        assert result.num_rows == 2

    def test_join(self, datasets):
        query = SqlQuery(
            "SELECT p.name, k.dst_id "
            "FROM person p "
            "JOIN knows k ON p.id = k.src_id "
            "ORDER BY p.name, k.dst_id"
        )
        result = query.execute(datasets)
        data = result.to_pydict()

        assert data["name"] == ["Alice", "Alice", "Bob", "Carol"]

    def test_self_join(self, datasets):
        query = SqlQuery(
            "SELECT p1.name AS person, p2.name AS friend "
            "FROM person p1 "
            "JOIN knows k ON p1.id = k.src_id "
            "JOIN person p2 ON p2.id = k.dst_id "
            "ORDER BY p1.name, p2.name"
        )
        result = query.execute(datasets)
        data = result.to_pydict()

        assert data["person"] == ["Alice", "Alice", "Bob", "Carol"]
        assert data["friend"] == ["Bob", "Carol", "David", "David"]

    def test_count(self, datasets):
        query = SqlQuery("SELECT COUNT(*) AS cnt FROM person")
        result = query.execute(datasets)
        assert result.to_pydict()["cnt"] == [4]

    def test_sum(self, datasets):
        query = SqlQuery("SELECT SUM(age) AS total FROM person")
        result = query.execute(datasets)
        assert result.to_pydict()["total"] == [28 + 34 + 29 + 42]

    def test_avg(self, datasets):
        query = SqlQuery("SELECT AVG(age) AS avg_age FROM person")
        result = query.execute(datasets)
        avg = result.to_pydict()["avg_age"][0]
        assert abs(avg - 33.25) < 0.01

    def test_group_by(self, datasets):
        query = SqlQuery(
            "SELECT city, COUNT(*) AS cnt "
            "FROM person GROUP BY city ORDER BY cnt DESC, city"
        )
        result = query.execute(datasets)
        data = result.to_pydict()
        assert data["city"][0] == "New York"
        assert data["cnt"][0] == 2

    def test_explain(self, datasets):
        query = SqlQuery("SELECT name FROM person WHERE age > 30")
        plan = query.explain(datasets)
        assert "Logical Plan" in plan
        assert "Physical Plan" in plan

    def test_sql_accessor(self):
        query = SqlQuery("SELECT 1")
        assert query.sql() == "SELECT 1"

    def test_repr(self):
        query = SqlQuery("SELECT 1")
        assert "SqlQuery" in repr(query)

    def test_invalid_sql(self, datasets):
        query = SqlQuery("INVALID SQL")
        with pytest.raises((RuntimeError, ValueError)):
            query.execute(datasets)

    def test_case_insensitive_table_names(self, person_table):
        """Table name 'Person' should be lowercased to 'person'."""
        query = SqlQuery("SELECT name FROM person LIMIT 1")
        result = query.execute({"Person": person_table})
        assert result.num_rows == 1


# ==========================================================================
# SqlEngine tests
# ==========================================================================


class TestSqlEngine:
    def test_basic_query(self, datasets):
        engine = SqlEngine(datasets)
        result = engine.execute(
            "SELECT name, age FROM person WHERE age > 30 ORDER BY age"
        )
        data = result.to_pydict()
        assert data["name"] == ["Bob", "David"]

    def test_multiple_queries(self, datasets):
        engine = SqlEngine(datasets)

        r1 = engine.execute("SELECT COUNT(*) AS cnt FROM person")
        r2 = engine.execute("SELECT name FROM person WHERE age > 30 ORDER BY name")
        r3 = engine.execute(
            "SELECT p.name, k.dst_id "
            "FROM person p JOIN knows k ON p.id = k.src_id "
            "ORDER BY p.name LIMIT 2"
        )

        assert r1.to_pydict()["cnt"] == [4]
        assert r2.to_pydict()["name"] == ["Bob", "David"]
        assert r3.num_rows == 2

    def test_repr(self, datasets):
        engine = SqlEngine(datasets)
        assert "SqlEngine" in repr(engine)

    def test_empty_datasets_raises(self):
        with pytest.raises(ValueError, match="No input datasets"):
            SqlEngine({})
