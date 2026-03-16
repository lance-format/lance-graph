# Lance Graph Explorer

A Streamlit UI for visualizing and querying lance-graph knowledge graphs.

## Features

- **Entity Search**: Search entities by name, filter by type
- **Cypher Query**: Execute Cypher queries with pre-built examples
- **Graph Visualization**: Interactive graph with node coloring by entity type

## Setup

```bash
cd python
uv sync --extra ui
```

## Run

```bash
uv run streamlit run streamlit_app/app.py
```

The app opens at `http://localhost:8501`.

## Usage

1. Enter the path to your knowledge graph directory in the sidebar (default: `./examples/test_kg`)
2. Click **Load** to connect
3. Use the tabs to search entities, run Cypher queries, or visualize the graph

## Example Queries

```cypher
-- List all entities
MATCH (e:Entity) RETURN e.name, e.entity_type LIMIT 50

-- Find relationships
MATCH (a:Entity)-[r:RELATIONSHIP]->(b:Entity)
RETURN a.name, r.relationship_type, b.name

-- Search by name
MATCH (e:Entity) WHERE e.name_lower CONTAINS 'matrix'
RETURN e.name, e.entity_type
```
