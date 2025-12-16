# Lance Graph

**A high-performance Cypher-capable graph query engine with Python bindings for building scalable, serverless knowledge graphs.**

Lance Graph combines a Rust-powered Cypher query engine with Python APIs for:
- Fast graph queries using Cypher query language
- AI-powered knowledge extraction from text (via LLM)
- Lance-backed storage for efficient graph data management
- Natural language Q&A over your knowledge graphs
- FastAPI web service for graph queries

## Installation

```bash
pip install lance-graph
```

## Quick Start

### 1. Simple Cypher Query

```python
import pyarrow as pa
from lance_graph import CypherQuery, GraphConfig

# Create sample data
people = pa.table({
    "person_id": [1, 2, 3, 4],
    "name": ["Alice", "Bob", "Carol", "David"],
    "age": [28, 34, 29, 42],
})

# Configure graph schema
config = (
    GraphConfig.builder()
    .with_node_label("Person", "person_id")
    .build()
)

# Execute Cypher query
query = CypherQuery("MATCH (p:Person) WHERE p.age > 30 RETURN p.name, p.age")
result = query.with_config(config).execute({"Person": people})

print(result.to_pydict())
# Output: {'name': ['Bob', 'David'], 'age': [34, 42]}
```

### 2. Build a Knowledge Graph from Text

```python
from pathlib import Path
from knowledge_graph import (
    KnowledgeGraphConfig,
    LanceKnowledgeGraph,
    LanceGraphStore,
    get_extractor,
)
from knowledge_graph.cli.ingest import extract_and_add

# Initialize knowledge graph
config = KnowledgeGraphConfig.from_root(Path("./my_graph"))
config.ensure_directories()

# Create schema
schema_path = config.resolved_schema_path()
if not schema_path.exists():
    schema_content = """
nodes:
  Entity:
    id_field: entity_id

relationships:
  RELATIONSHIP:
    source: source_entity_id
    target: target_entity_id
"""
    schema_path.write_text(schema_content, encoding="utf-8")

store = LanceGraphStore(config)
store.ensure_layout()

graph_config = config.load_graph_config()
kg = LanceKnowledgeGraph(graph_config, storage=store)
kg.ensure_initialized()

# Extract and add entities/relationships from text
# Using heuristic extractor for testing without API key
extractor = get_extractor("heuristic")
# or using LLM extractor (requires API key)
# extractor = get_extractor("llm", llm_model="gpt-4o-mini")
text = """
Albert Einstein developed the theory of relativity at Princeton.
Marie Curie discovered radioactivity in Paris.
"""

extract_and_add(text, kg, extractor, embedding_generator=None)

# Query the graph
result = kg.query("""
    MATCH (e:Entity)
    RETURN e.name, e.entity_type
    LIMIT 10
""")
print(result.to_pylist())
```

### 3. Natural Language Q&A

```python
from knowledge_graph.llm.qa import ask_question

# Ask questions in natural language
answer = ask_question(
    "Who discovered radioactivity?",
    kg,
    llm_model="gpt-4o-mini"
)
print(answer)
# Output: Marie Curie discovered radioactivity.
```

## Command-Line Interface

Lance Graph includes a CLI for building and querying knowledge graphs:

```bash
# Initialize and extract
knowledge_graph --root ./my_graph --init
knowledge_graph --root ./my_graph --extract-and-add notes.txt

# Query with Cypher
knowledge_graph --root ./my_graph "MATCH (e:Entity) RETURN e.name LIMIT 10"

# Natural language Q&A
knowledge_graph --root ./my_graph --ask "Who discovered DNA?"
```

For complete CLI documentation and examples, see the [main README](../README.md#cli-usage).

## Requirements

- Python 3.11+
- Optional: OpenAI API key for LLM extraction

## Contributing

Lance Graph is open source! Contributions are welcome.

### Quick start

```bash
cd python
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install maturin[patchelf]
uv pip install -e '.[tests]'
maturin develop
pytest python/tests/ -v
```

### Development workflow

For linting and type checks:

```bash
# Install dev dependencies
uv pip install -e '.[dev]'

# Run linters and type checker
ruff format python/              # format code
ruff check python/               # lint code
pyright                          # type check

# Run specific tests
pytest python/tests/test_graph.py::test_basic_node_selection -v

# Rebuild extension after Rust changes
maturin develop
```

> If another virtual environment is already active, run `deactivate` (or
> `unset VIRTUAL_ENV`) before invoking `uv run` so uv binds to `.venv`.

## Repository layout

- `python/src/` – PyO3 bridge that exposes graph APIs to Python
- `python/python/lance_graph/` – pure-Python wrapper and `__init__`
- `python/python/knowledge_graph/` – CLI, FastAPI, and extractor utilities built on Lance
- `python/python/tests/` – graph-centric functional tests

For more information on development setup, building from source, running tests, and code quality guidelines, see [DEVELOPMENT.md](./DEVELOPMENT.md).

## License

Apache 2.0

## Links

- [GitHub](https://github.com/lancedb/lance-graph)
- [Documentation](https://deepwiki.com/lancedb/lance-graph)
- [PyPI](https://pypi.org/project/lance-graph/)
- [LanceDB](https://lancedb.com/)
