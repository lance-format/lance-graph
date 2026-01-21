import threading
import time

import pyarrow as pa

from lance_graph import CypherQuery, DistanceMetric, GraphConfig, VectorSearch


def _make_large_embedding_table(num_rows: int = 20_000, dim: int = 64) -> pa.Table:
    """Build a table large enough to exercise a noticeable compute path."""
    ids = pa.array(range(num_rows), type=pa.int32())
    # Generate deterministic float32 data to avoid relying on numpy.
    values = [
        float((i % dim) + (i // dim) % 7) / float(dim)
        for i in range(num_rows * dim)
    ]
    flat = pa.array(values, type=pa.float32())
    embedding = pa.FixedSizeListArray.from_arrays(flat, dim)
    return pa.table({"id": ids, "embedding": embedding})


def test_cypher_query_releases_gil_during_execution():
    table = _make_large_embedding_table()
    config = GraphConfig.builder().with_node_label("Document", "id").build()
    datasets = {"Document": table}

    # Warm up the execution path before measuring concurrency.
    query = CypherQuery("MATCH (d:Document) RETURN d.id, d.embedding").with_config(config)
    warmup = query.execute(datasets)
    assert warmup.num_rows == table.num_rows

    dim = table.schema.field("embedding").type.list_size
    vector_search = (
        VectorSearch("embedding")
        .query_vector([0.1] * dim)
        .metric(DistanceMetric.L2)
        .top_k(10)
    )
    rerank = vector_search.search(table)
    assert rerank.num_rows == 10

    heartbeats = 0
    stop_event = threading.Event()

    def heartbeat() -> None:
        nonlocal heartbeats
        while not stop_event.is_set():
            heartbeats += 1
            time.sleep(0.01)

    monitor = threading.Thread(target=heartbeat)
    monitor.start()

    try:
        start_wait = time.perf_counter()
        # Wait until the heartbeat thread has run at least once so the baseline
        # reflects activity outside the critical section we want to observe.
        while heartbeats == 0 and time.perf_counter() - start_wait < 1.0:
            time.sleep(0.01)

        baseline = heartbeats
        start = time.perf_counter()
        # Run query/vector search pairs to create a noticeable critical section.
        for _ in range(3):
            result = query.execute(datasets)
            assert result.num_rows == table.num_rows
            rerank = vector_search.search(table)
            assert rerank.num_rows == 10
        elapsed = time.perf_counter() - start
    finally:
        stop_event.set()
        monitor.join()

    # Without releasing the GIL, the heartbeat thread would be unable to make progress.
    assert elapsed > 0.0
    assert heartbeats - baseline >= 1
