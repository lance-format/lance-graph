# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import pyarrow as pa
import pytest
from lance_graph import CypherQuery, GraphConfig


def test_unimplemented_scalar_function_errors() -> None:
    cfg = GraphConfig.builder().with_node_label("Person", "name").build()
    datasets = {"Person": pa.table({"name": ["Alice", "BOB", "CaSeY"]})}

    query = CypherQuery(
        "MATCH (p:Person) "
        "RETURN p.name AS name, replace(p.name, 'A', 'a') AS replaced "
        "ORDER BY name",
    ).with_config(cfg)

    with pytest.raises(Exception) as excinfo:
        query.execute(datasets)

    assert "replace" in str(excinfo.value).lower()
