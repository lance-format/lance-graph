# Copyright 2024 Lance Developers.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests to verify GIL release during graph operations.

This module ensures that lance-graph properly releases the Python Global
Interpreter Lock (GIL) during blocking I/O and compute-intensive operations,
allowing other Python threads to run concurrently.
"""

import threading
import time
from typing import Dict

import pyarrow as pa
import pytest

from lance_graph import CypherQuery, GraphConfig


def create_test_datasets() -> Dict[str, pa.RecordBatch]:
    """Create simple test datasets for query execution."""
    # Create a Person node dataset
    person_data = {
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age": [30, 25, 35, 28, 32],
    }
    person_batch = pa.RecordBatch.from_pydict(person_data)

    # Create a KNOWS relationship dataset
    knows_data = {
        "src_person_id": [1, 1, 2, 3, 4],
        "dst_person_id": [2, 3, 3, 4, 5],
    }
    knows_batch = pa.RecordBatch.from_pydict(knows_data)

    return {"Person": person_batch, "KNOWS": knows_batch}


def test_gil_release_during_query_execution():
    """Verify that GIL is released during query execution.

    This test runs a Cypher query in the main thread while a background
    thread increments a counter. If the GIL is properly released, the
    background thread should be able to make progress while the query
    executes.
    """
    # Setup
    config = (
        GraphConfig.builder()
        .with_node_label("Person", "id")
        .with_relationship("KNOWS", "src_person_id", "dst_person_id")
        .build()
    )

    datasets = create_test_datasets()
    query = CypherQuery("MATCH (p:Person) WHERE p.age > 25 RETURN p.name").with_config(
        config
    )

    # Background thread state
    heartbeats = [0]  # Use list for mutable capture
    stop_event = threading.Event()

    def heartbeat():
        """Background thread that increments a counter."""
        while not stop_event.is_set():
            heartbeats[0] += 1
            time.sleep(0.001)  # 1ms sleep

    # Start background thread
    t = threading.Thread(target=heartbeat, daemon=True)
    t.start()

    try:
        # Execute query (this should release GIL)
        result = query.execute(datasets)

        # Verify query succeeded
        assert result is not None
        table = result.to_pydict()
        assert "p.name" in table
        assert len(table["p.name"]) > 0

        # Verify background thread made progress
        # If GIL was held, heartbeats would be 0 or very small
        assert heartbeats[0] >= 5, (
            f"Background thread was starved! GIL was not released. "
            f"Only {heartbeats[0]} heartbeats occurred."
        )

    finally:
        stop_event.set()
        t.join(timeout=1.0)


def test_gil_release_during_to_sql():
    """Verify that GIL is released during SQL translation."""
    config = (
        GraphConfig.builder()
        .with_node_label("Person", "id")
        .with_relationship("KNOWS", "src_person_id", "dst_person_id")
        .build()
    )

    datasets = create_test_datasets()
    query = CypherQuery("MATCH (p:Person)-[:KNOWS]->(f:Person) RETURN p.name, f.name").with_config(
        config
    )

    heartbeats = [0]
    stop_event = threading.Event()

    def heartbeat():
        while not stop_event.is_set():
            heartbeats[0] += 1
            time.sleep(0.001)

    t = threading.Thread(target=heartbeat, daemon=True)
    t.start()

    try:
        # to_sql is typically fast, but should still release GIL
        sql = query.to_sql(datasets)
        assert sql is not None
        assert len(sql) > 0

        # With GIL released, background thread should make some progress
        assert heartbeats[0] > 0, "Background thread was starved during to_sql"

    finally:
        stop_event.set()
        t.join(timeout=1.0)


def test_gil_release_during_explain():
    """Verify that GIL is released during query explanation."""
    config = (
        GraphConfig.builder()
        .with_node_label("Person", "id")
        .with_relationship("KNOWS", "src_person_id", "dst_person_id")
        .build()
    )

    datasets = create_test_datasets()
    query = CypherQuery("MATCH (p:Person) RETURN p.name").with_config(config)

    heartbeats = [0]
    stop_event = threading.Event()

    def heartbeat():
        while not stop_event.is_set():
            heartbeats[0] += 1
            time.sleep(0.001)

    t = threading.Thread(target=heartbeat, daemon=True)
    t.start()

    try:
        explanation = query.explain(datasets)
        assert explanation is not None

        assert heartbeats[0] > 0, "Background thread was starved during explain"

    finally:
        stop_event.set()
        t.join(timeout=1.0)


def test_concurrent_query_execution():
    """Verify that multiple queries can execute concurrently.

    This test runs multiple queries in parallel threads to ensure that
    GIL release allows true concurrent execution.
    """
    config = (
        GraphConfig.builder()
        .with_node_label("Person", "id")
        .with_relationship("KNOWS", "src_person_id", "dst_person_id")
        .build()
    )

    datasets = create_test_datasets()

    queries = [
        "MATCH (p:Person) WHERE p.age > 25 RETURN p.name",
        "MATCH (p:Person) WHERE p.age < 30 RETURN p.name",
        "MATCH (p:Person)-[:KNOWS]->(f:Person) RETURN p.name, f.name",
    ]

    results = [None] * len(queries)
    errors = [None] * len(queries)

    def run_query(idx, query_str):
        try:
            query = CypherQuery(query_str).with_config(config)
            results[idx] = query.execute(datasets)
        except Exception as e:
            errors[idx] = e

    # Run queries in parallel
    threads = []
    for i, q in enumerate(queries):
        t = threading.Thread(target=run_query, args=(i, q))
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join(timeout=5.0)

    # Verify all queries succeeded
    for i, error in enumerate(errors):
        assert error is None, f"Query {i} failed: {error}"

    for i, result in enumerate(results):
        assert result is not None, f"Query {i} returned None"


@pytest.mark.slow
def test_gil_release_with_heavy_query():
    """Test GIL release with a heavier query that takes measurable time.

    This test creates a larger dataset and runs a more complex query
    to ensure GIL release works correctly with longer-running operations.
    """
    # Create larger dataset
    num_nodes = 1000
    person_data = {
        "id": list(range(num_nodes)),
        "name": [f"Person{i}" for i in range(num_nodes)],
        "age": [(20 + i % 50) for i in range(num_nodes)],
    }
    person_batch = pa.RecordBatch.from_pydict(person_data)

    # Create many relationships
    num_edges = 5000
    knows_data = {
        "src_person_id": [i % num_nodes for i in range(num_edges)],
        "dst_person_id": [(i + 1) % num_nodes for i in range(num_edges)],
    }
    knows_batch = pa.RecordBatch.from_pydict(knows_data)

    datasets = {"Person": person_batch, "KNOWS": knows_batch}

    config = (
        GraphConfig.builder()
        .with_node_label("Person", "id")
        .with_relationship("KNOWS", "src_person_id", "dst_person_id")
        .build()
    )

    query = CypherQuery(
        "MATCH (p:Person)-[:KNOWS]->(f:Person) WHERE p.age > 30 AND f.age < 40 RETURN p.name, f.name"
    ).with_config(config)

    heartbeats = [0]
    stop_event = threading.Event()
    start_time = [None]

    def heartbeat():
        start_time[0] = time.time()
        while not stop_event.is_set():
            heartbeats[0] += 1
            time.sleep(0.001)

    t = threading.Thread(target=heartbeat, daemon=True)
    t.start()

    try:
        result = query.execute(datasets)
        assert result is not None

        elapsed = time.time() - start_time[0]

        # With a larger query taking measurable time, we should see
        # many heartbeats (roughly elapsed_seconds * 1000)
        # Allow for some variance, but expect at least 50% of theoretical max
        expected_min_heartbeats = int(elapsed * 1000 * 0.5)
        assert heartbeats[0] > expected_min_heartbeats, (
            f"Background thread severely starved. "
            f"Expected ~{int(elapsed * 1000)} heartbeats in {elapsed:.3f}s, "
            f"got {heartbeats[0]}. GIL may not be properly released."
        )

    finally:
        stop_event.set()
        t.join(timeout=1.0)
