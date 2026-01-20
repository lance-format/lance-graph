"""Python bindings for the ``lance-graph`` crate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def _load_bindings() -> ModuleType:
    try:
        from . import _internal as bindings  # type: ignore[attr-defined]
    except ImportError:
        try:
            import _internal as bindings  # type: ignore[import-not-found]
        except ImportError:
            bindings = _load_dev_build()
    return bindings


def _load_dev_build() -> ModuleType:
    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parent.parent  # .../python
    repo_root = project_root.parent

    search_paths = [
        project_root / "target" / "debug",
        project_root / "target" / "debug" / "deps",
        project_root / "target" / "release",
        project_root / "target" / "release" / "deps",
        project_root / "target" / "maturin",
        repo_root / "target" / "debug",
        repo_root / "target" / "debug" / "deps",
        repo_root / "target" / "release",
        repo_root / "target" / "release" / "deps",
        repo_root / "target" / "maturin",
    ]

    candidates = []
    for base in search_paths:
        candidates.extend(base.glob("_internal*.so"))
        candidates.extend(base.glob("lib_internal*.so"))
        candidates.extend(base.glob("_internal*.pyd"))
        candidates.extend(base.glob("lib_internal*.pyd"))

    for candidate in candidates:
        if not candidate.exists():
            continue
        spec = importlib.util.spec_from_file_location(
            f"{__name__}._internal", candidate
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[call-arg]
        sys.modules[spec.name] = module
        return module

    raise ImportError(
        "Unable to locate the compiled lance-graph extension. "
        "Run `maturin develop` (or install the package) before importing."
    )


_bindings = _load_bindings()

GraphConfig = _bindings.graph.GraphConfig
GraphConfigBuilder = _bindings.graph.GraphConfigBuilder
CypherQuery = _bindings.graph.CypherQuery
VectorSearch = _bindings.graph.VectorSearch
DistanceMetric = _bindings.graph.DistanceMetric


def execute_with_store(query, store, config=None):
    """Execute a Cypher query using tables from a LanceGraphStore.

    Parameters
    ----------
    query : CypherQuery
        The parsed Cypher query
    store : LanceGraphStore
        The store containing Lance datasets
    config : GraphConfig, optional
        Graph configuration. If not provided:
        - Tries to load from store's YAML schema
        - Falls back to convention-based inference

    Returns
    -------
    pyarrow.Table
        Query results

    Examples
    --------
    >>> from lance_graph import CypherQuery, execute_with_store
    >>> from knowledge_graph import LanceGraphStore, KnowledgeGraphConfig
    >>> config = KnowledgeGraphConfig.from_root("s3://my-bucket/graph-data")
    >>> store = LanceGraphStore(config)
    >>> query = CypherQuery("MATCH (p:Person) RETURN p.name")
    >>> result = execute_with_store(query, store)
    """
    # 1. Resolve config
    if config is None:
        try:
            config = store.config.load_graph_config()
        except FileNotFoundError:
            config = store.infer_graph_config()

    query = query.with_config(config)

    # 2. Load only required tables (avoids full enumeration)
    required = set(query.node_labels() + query.relationship_types())
    tables = store.load_tables(required)

    # 3. Execute
    return query.execute(tables)


__all__ = [
    "GraphConfig",
    "GraphConfigBuilder",
    "CypherQuery",
    "VectorSearch",
    "DistanceMetric",
    "execute_with_store",
]

__version__ = _bindings.__version__
