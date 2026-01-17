"""Graph visualization helpers using streamlit-agraph."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from streamlit_agraph import Config, Edge, Node

# Color palette for different entity types
ENTITY_COLORS = {
    "Person": "#4CAF50",      # Green
    "Company": "#2196F3",     # Blue
    "Location": "#FF9800",    # Orange
    "Technology": "#9C27B0",  # Purple
    "Document": "#F44336",    # Red
    "Event": "#00BCD4",       # Cyan
    "UNKNOWN": "#9E9E9E",     # Grey
}


def get_entity_color(entity_type: str) -> str:
    """Return a color for the given entity type."""
    return ENTITY_COLORS.get(entity_type, ENTITY_COLORS["UNKNOWN"])


def entities_to_agraph(
    entities: pa.Table,
    relationships: pa.Table,
    *,
    max_nodes: int = 100,
    entity_type_filter: str | None = None,
) -> tuple[list[Node], list[Edge]]:
    """Convert Entity and Relationship tables to agraph nodes and edges.

    Args:
        entities: PyArrow table with entity_id, name, entity_type columns
        relationships: PyArrow table with source_entity_id, target_entity_id, relationship_type
        max_nodes: Maximum number of nodes to display
        entity_type_filter: Filter to only show entities of this type (None = all)

    Returns:
        Tuple of (nodes, edges) for agraph
    """
    entity_rows = entities.to_pylist()

    # Apply entity type filter if specified
    if entity_type_filter and entity_type_filter != "All":
        entity_rows = [
            e for e in entity_rows
            if e.get("entity_type") == entity_type_filter
        ]

    # Limit nodes
    entity_rows = entity_rows[:max_nodes]

    # Build a set of visible entity IDs
    visible_ids = {e["entity_id"] for e in entity_rows}

    # Create nodes
    nodes = []
    for entity in entity_rows:
        entity_type = entity.get("entity_type", "UNKNOWN")
        nodes.append(Node(
            id=entity["entity_id"],
            label=entity.get("name", entity["entity_id"]),
            size=25,
            color=get_entity_color(entity_type),
            title=f"{entity.get('name', '')}\nType: {entity_type}\n{entity.get('context', '')}",
        ))

    # Create edges (only for visible nodes)
    edges = []
    for rel in relationships.to_pylist():
        source = rel.get("source_entity_id")
        target = rel.get("target_entity_id")
        if source in visible_ids and target in visible_ids:
            edges.append(Edge(
                source=source,
                target=target,
                label=rel.get("relationship_type", ""),
                color="#888888",
            ))

    return nodes, edges


def query_result_to_agraph(
    result: pa.Table,
    *,
    node_id_col: str | None = None,
    node_label_col: str | None = None,
) -> tuple[list[Node], list[Edge]]:
    """Convert a Cypher query result to agraph nodes.

    This is a simpler visualization for query results that may not have
    the full Entity/Relationship structure.

    Args:
        result: Query result as PyArrow table
        node_id_col: Column to use as node ID (auto-detected if None)
        node_label_col: Column to use as node label (auto-detected if None)

    Returns:
        Tuple of (nodes, edges) - edges are empty for simple results
    """
    rows = result.to_pylist()
    if not rows:
        return [], []

    # Auto-detect columns
    columns = result.column_names
    if node_id_col is None:
        # Look for common ID columns
        for col in ["entity_id", "id", "name"]:
            if col in columns:
                node_id_col = col
                break
        if node_id_col is None and columns:
            node_id_col = columns[0]

    if node_label_col is None:
        # Look for common label columns
        for col in ["name", "label", "title"]:
            if col in columns:
                node_label_col = col
                break
        if node_label_col is None:
            node_label_col = node_id_col

    # Create nodes
    nodes = []
    seen_ids = set()
    for row in rows:
        node_id = str(row.get(node_id_col, ""))
        if not node_id or node_id in seen_ids:
            continue
        seen_ids.add(node_id)

        label = str(row.get(node_label_col, node_id))
        entity_type = row.get("entity_type", "UNKNOWN")

        # Build tooltip from all row data
        tooltip_lines = [f"{k}: {v}" for k, v in row.items() if v is not None]
        tooltip = "\n".join(tooltip_lines)

        nodes.append(Node(
            id=node_id,
            label=label,
            size=25,
            color=get_entity_color(entity_type),
            title=tooltip,
        ))

    return nodes, []


def get_agraph_config(
    *,
    height: int = 500,
    width: int = 800,
    directed: bool = True,
    physics: bool = True,
) -> Config:
    """Return a streamlit-agraph Config with sensible defaults."""
    return Config(
        height=height,
        width=width,
        directed=directed,
        physics=physics,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
    )


def get_entity_types(entities: pa.Table) -> list[str]:
    """Extract unique entity types from an Entity table."""
    if "entity_type" not in entities.column_names:
        return []
    types = entities.column("entity_type").to_pylist()
    return sorted(set(t for t in types if t))
