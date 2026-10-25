# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Exercise the installed binary wheel outside the source tree, without Rust."""

import importlib.metadata
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
from lance import write_dataset
from lance_graph import CypherQuery, DirNamespace, GraphConfig, SqlQuery, _internal


def main():
    assert sys.prefix != sys.base_prefix, "Use a clean virtual environment"
    assert shutil.which("cargo") is None, "Remove Cargo from the smoke-test PATH"
    assert shutil.which("rustc") is None, "Remove rustc from the smoke-test PATH"
    extension = Path(_internal.__file__).resolve()
    assert extension.is_relative_to(Path(sys.prefix).resolve()), extension
    assert extension.suffix == ".so", "Expected the installed native extension"

    people = pa.table({"id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"]})
    knows = pa.table({"source": [1, 2], "target": [2, 3]})
    config = (
        GraphConfig.builder()
        .with_node_label("Person", "id")
        .with_relationship("KNOWS", "source", "target")
        .build()
    )
    query = CypherQuery(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) "
        "WHERE a.id = 1 RETURN b.name AS friend, c.name AS friend_of_friend"
    ).with_config(config)
    expected = [{"friend": "Bob", "friend_of_friend": "Carol"}]
    assert query.execute({"Person": people, "KNOWS": knows}).to_pylist() == expected
    with TemporaryDirectory() as root:
        write_dataset(people, Path(root) / "Person.lance")
        write_dataset(knows, Path(root) / "KNOWS.lance")
        assert query.execute_with_namespace(DirNamespace(root)).to_pylist() == expected
    assert SqlQuery("SELECT COUNT(*) AS count FROM people").execute(
        {"people": people}
    ).to_pylist() == [{"count": 3}]
    print(
        f"Installed lance-graph {importlib.metadata.version('lance-graph')} wheel "
        f"passed Cypher and SQL checks on Python {sys.version.split()[0]} without Rust"
    )


if __name__ == "__main__":
    main()
