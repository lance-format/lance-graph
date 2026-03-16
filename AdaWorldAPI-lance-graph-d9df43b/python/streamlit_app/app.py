"""Lance Graph Knowledge Graph UI.

A Streamlit application for visualizing and querying lance-graph knowledge graphs.

Run with:
    cd python
    uv sync --extra ui
    uv run streamlit run streamlit_app/app.py
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st
from streamlit_agraph import agraph

from graph_viz import (
    entities_to_agraph,
    get_agraph_config,
    get_entity_types,
    query_result_to_agraph,
)

# Page configuration
st.set_page_config(
    page_title="Lance Graph Explorer",
    page_icon=":spider_web:",
    layout="wide",
)


def init_session_state() -> None:
    """Initialize session state variables."""
    if "kg" not in st.session_state:
        st.session_state.kg = None
    if "tables" not in st.session_state:
        st.session_state.tables = None
    if "config" not in st.session_state:
        st.session_state.config = None
    if "data_source" not in st.session_state:
        st.session_state.data_source = None
    if "last_query_result" not in st.session_state:
        st.session_state.last_query_result = None


def load_from_directory(directory: str) -> None:
    """Load knowledge graph from a Lance directory."""
    from knowledge_graph import KnowledgeGraphConfig, LanceGraphStore, LanceKnowledgeGraph
    from knowledge_graph.config import build_default_graph_config

    path = Path(directory)
    if not path.exists():
        st.error(f"Directory does not exist: {directory}")
        return

    try:
        config = KnowledgeGraphConfig.from_root(path)
        store = LanceGraphStore(config)
        # Try to load schema, fall back to default if not found
        try:
            graph_config = config.load_graph_config()
        except FileNotFoundError:
            graph_config = build_default_graph_config()
        service = LanceKnowledgeGraph(graph_config, storage=store)
        service.ensure_initialized()

        # Load tables for visualization
        tables = service.load_tables()
        st.session_state.kg = service
        st.session_state.tables = dict(tables)
        st.session_state.config = graph_config
        st.session_state.data_source = directory
        st.success(f"Loaded knowledge graph from {directory}")
    except Exception as e:
        st.error(f"Failed to load: {e}")


def run_query(query: str) -> None:
    """Execute a Cypher query and store the result."""
    if st.session_state.kg is None:
        st.error("No knowledge graph loaded. Please load data first.")
        return

    try:
        start = time.time()
        result = st.session_state.kg.run(query)
        elapsed = time.time() - start
        st.session_state.last_query_result = {
            "result": result,
            "query": query,
            "elapsed": elapsed,
        }
    except Exception as e:
        st.error(f"Query error: {e}")
        st.session_state.last_query_result = None


# Example queries for the UI
EXAMPLE_QUERIES = {
    "List all entities": "MATCH (e:Entity) RETURN e.name, e.entity_type LIMIT 50",
    "Search by name (contains 'matrix')": "MATCH (e:Entity) WHERE e.name_lower CONTAINS 'matrix' RETURN e.name, e.entity_type, e.context",
    "Find all relationships": "MATCH (a:Entity)-[r:RELATIONSHIP]->(b:Entity) RETURN a.name, r.relationship_type, b.name LIMIT 100",
    "Find connected entities": "MATCH (e:Entity)-[r:RELATIONSHIP]-(connected:Entity) RETURN e.name, r.relationship_type, connected.name LIMIT 50",
    "Count by entity type": "MATCH (e:Entity) RETURN e.entity_type, count(*) as count",
    "People only": "MATCH (e:Entity) WHERE e.entity_type = 'Person' RETURN e.name, e.context",
}


def main() -> None:
    """Main Streamlit application."""
    init_session_state()

    st.title("Lance Graph Explorer")

    # Sidebar for data source configuration
    with st.sidebar:
        st.header("Data Source")

        directory = st.text_input(
            "Knowledge Graph Directory",
            value="./examples/test_kg",
            help="Path to the Lance knowledge graph storage directory",
        )
        if st.button("Load", type="primary"):
            expanded_path = str(Path(directory).expanduser())
            load_from_directory(expanded_path)

        # Show current data source status
        st.divider()
        if st.session_state.kg is not None:
            st.success(f"Connected: {st.session_state.data_source}")
            if st.session_state.tables:
                st.write("**Available datasets:**")
                for name in st.session_state.tables.keys():
                    table = st.session_state.tables[name]
                    st.write(f"- {name}: {table.num_rows} rows")
        else:
            st.warning("No data loaded")

    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["Entity Search", "Cypher Query", "Graph Visualization"])

    with tab1:
        st.header("Entity Search")

        if st.session_state.tables is None or "Entity" not in st.session_state.tables:
            st.info("Load a knowledge graph to search entities.")
        else:
            entities_table = st.session_state.tables["Entity"]
            entity_types = ["All"] + get_entity_types(entities_table)

            col1, col2 = st.columns([3, 1])
            with col1:
                search_term = st.text_input(
                    "Search by name",
                    placeholder="Enter entity name...",
                )
            with col2:
                type_filter = st.selectbox(
                    "Filter by type",
                    options=entity_types,
                )

            # Filter entities
            entities = entities_table.to_pylist()
            if search_term:
                search_lower = search_term.lower()
                entities = [
                    e for e in entities
                    if search_lower in e.get("name_lower", e.get("name", "").lower())
                ]
            if type_filter and type_filter != "All":
                entities = [e for e in entities if e.get("entity_type") == type_filter]

            st.write(f"Found **{len(entities)}** entities")
            if entities:
                st.dataframe(
                    entities,
                    use_container_width=True,
                    hide_index=True,
                )

    with tab2:
        st.header("Cypher Query")

        if st.session_state.kg is None:
            st.info("Load a knowledge graph to run queries.")
        else:
            # Example query selector
            example = st.selectbox(
                "Example queries",
                options=[""] + list(EXAMPLE_QUERIES.keys()),
                format_func=lambda x: x if x else "Select an example...",
            )

            # Query input
            default_query = EXAMPLE_QUERIES.get(example, "")
            query = st.text_area(
                "Cypher Query",
                value=default_query,
                height=100,
                placeholder="MATCH (n:Entity) RETURN n.name LIMIT 10",
            )

            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Execute", type="primary", disabled=not query.strip()):
                    run_query(query)

            # Display results
            if st.session_state.last_query_result:
                result_data = st.session_state.last_query_result
                st.write(f"Query executed in **{result_data['elapsed']:.3f}s**")

                result_table = result_data["result"]
                rows = result_table.to_pylist()
                st.write(f"**{len(rows)}** rows returned")

                if rows:
                    st.dataframe(rows, use_container_width=True, hide_index=True)

                    # Option to visualize results
                    if st.checkbox("Visualize as graph"):
                        nodes, edges = query_result_to_agraph(result_table)
                        if nodes:
                            agraph(nodes=nodes, edges=edges, config=get_agraph_config())
                        else:
                            st.info("No nodes to visualize in query result")

    with tab3:
        st.header("Graph Visualization")

        if st.session_state.tables is None:
            st.info("Load a knowledge graph to visualize.")
        else:
            entities_table = st.session_state.tables.get("Entity")
            relationships_table = st.session_state.tables.get("RELATIONSHIP")

            if entities_table is None:
                st.warning("No Entity table found in the loaded data.")
            else:
                # Visualization controls
                col1, col2, col3 = st.columns(3)
                with col1:
                    entity_types = ["All"] + get_entity_types(entities_table)
                    type_filter = st.selectbox(
                        "Filter by entity type",
                        options=entity_types,
                        key="viz_type_filter",
                    )
                with col2:
                    max_nodes = st.slider(
                        "Max nodes to display",
                        min_value=10,
                        max_value=200,
                        value=50,
                        step=10,
                    )
                with col3:
                    physics = st.checkbox("Enable physics", value=True)

                # Build and render graph
                if relationships_table is None:
                    import pyarrow as pa
                    relationships_table = pa.table({
                        "source_entity_id": [],
                        "target_entity_id": [],
                        "relationship_type": [],
                    })

                filter_val = type_filter if type_filter != "All" else None
                nodes, edges = entities_to_agraph(
                    entities_table,
                    relationships_table,
                    max_nodes=max_nodes,
                    entity_type_filter=filter_val,
                )

                if nodes:
                    st.write(f"Showing **{len(nodes)}** nodes and **{len(edges)}** edges")
                    agraph(
                        nodes=nodes,
                        edges=edges,
                        config=get_agraph_config(height=600, physics=physics),
                    )
                else:
                    st.info("No entities to display with current filters.")


if __name__ == "__main__":
    main()
