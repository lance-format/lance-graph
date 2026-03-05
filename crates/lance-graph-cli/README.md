# lgraph CLI

Command-line interface for the lance-graph query engine. Run Cypher and SQL queries against Lance, Parquet, and Delta Lake datasets.

## Installation

```bash
cargo install --path crates/lance-graph-cli
```

Or build from the workspace:

```bash
cargo build -p lance-graph-cli --release
```

The binary is named `lgraph`.

## Quick Start

```bash
# Generate a config file
lgraph init

# Edit lgraph.toml to point to your data and define graph schema
# Then run queries:
lgraph cypher "MATCH (p:Person)-[:KNOWS]->(f:Person) RETURN p.name, f.name"
lgraph sql "SELECT name, age FROM person WHERE age > 30"
```

## Configuration

`lgraph` uses a TOML config file (default: `./lgraph.toml`). Use `-c <path>` to specify a different location.

### Namespace (local Lance/Parquet tables)

```toml
namespace = "/path/to/tables"

[graph.nodes.Person]
id_field = "person_id"

[graph.nodes.Company]
id_field = "company_id"

[graph.relationships.WORKS_AT]
source_field = "person_id"
target_field = "company_id"
```

The `namespace` path should point to a directory containing `.lance` datasets. Table names in the `[graph]` section must match the dataset directory names.

### Unity Catalog

```toml
[catalog]
url = "http://localhost:8080/api/2.1/unity-catalog"
catalog_name = "main"
schema_name = "default"

[catalog.storage_options]
aws_access_key_id = "..."
aws_secret_access_key = "..."
```

### Graph schema

The `[graph]` section maps your tabular data to a property graph model. It is required for Cypher queries but optional for SQL.

- **Nodes**: each entry names a node label and its ID column.
- **Relationships**: each entry names a relationship type and its source/target ID columns.

## Commands

### `cypher` — Run a Cypher query

```bash
lgraph cypher "MATCH (p:Person) WHERE p.age > 30 RETURN p.name, p.age"
```

Requires a `[graph]` section in the config file.

### `sql` — Run a SQL query

```bash
lgraph sql "SELECT name, age FROM person ORDER BY age DESC LIMIT 10"
```

Tables defined in the `[graph]` section are registered and available by name (lowercased).

### `tables` — List available tables

```bash
lgraph tables
```

### `schema` — Show table schema

```bash
lgraph schema Person
```

### `init` — Create a template config

```bash
lgraph init                    # creates ./lgraph.toml
lgraph init -c my-config.toml  # creates at custom path
```

## Output Formats

Use `--format` (or `-f`) to control output. When omitted, the format is auto-detected:

- **TTY** (interactive terminal): `table`
- **Piped** (scripts, Claude skills): `jsonl`

| Format | Flag | Description |
|--------|------|-------------|
| Table | `-f table` | Human-readable ASCII table |
| JSONL | `-f jsonl` | One JSON object per row (best for agents) |
| JSON | `-f json` | Single JSON object with metadata envelope |
| CSV | `-f csv` | Comma-separated values with header |

### Examples

```bash
# Pretty table in terminal
lgraph cypher "MATCH (p:Person) RETURN p.name" -f table

# JSONL for piping to jq
lgraph cypher "MATCH (p:Person) RETURN p.name" -f jsonl | jq '.name'

# JSON envelope with metadata
lgraph sql "SELECT count(*) as cnt FROM person" -f json

# CSV for data export
lgraph sql "SELECT * FROM person" -f csv > people.csv
```

### JSONL output (default when piped)

```json
{"name": "Alice", "age": 28}
{"name": "Bob", "age": 34}
```

### JSON envelope output

```json
{
  "columns": ["name", "age"],
  "row_count": 2,
  "rows": [
    {"name": "Alice", "age": 28},
    {"name": "Bob", "age": 34}
  ]
}
```

## Usage with Claude Skills

`lgraph` is designed to be called from Claude Code skills via bash. When piped, it defaults to JSONL output which is optimal for LLM consumption.

```bash
# In a Claude skill, simply call lgraph:
lgraph cypher "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.name, c.name"

# The JSONL output is self-describing and truncation-safe
lgraph sql "SELECT * FROM person WHERE city = 'New York'" -f json
```
