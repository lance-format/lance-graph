// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `lgraph` — CLI tool for translating Cypher queries to SQL.
//!
//! # Usage
//!
//! ```sh
//! # Translate a Cypher query using a graph config file
//! lgraph -c graph.json "MATCH (p:Person) WHERE p.age > 30 RETURN p.name"
//!
//! # Specify a dialect
//! lgraph -c graph.json -d spark "MATCH (p:Person) RETURN p.name"
//!
//! # Read query from stdin
//! echo "MATCH (p:Person) RETURN p.name" | lgraph -c graph.json
//! ```

use std::collections::HashMap;
use std::io::{self, Read};
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use clap::Parser;
use lance_graph::{CypherQuery, GraphConfig, SqlDialect};
use serde::Deserialize;

/// Translate Cypher queries to SQL for various dialects.
#[derive(Parser)]
#[command(name = "lgraph", version, about)]
struct Cli {
    /// Cypher query string. If omitted, reads from stdin.
    query: Option<String>,

    /// Path to graph config JSON file describing the schema.
    #[arg(short, long)]
    config: PathBuf,

    /// SQL dialect for the output.
    #[arg(short, long, value_enum, default_value_t = DialectArg::Default)]
    dialect: DialectArg,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum DialectArg {
    Default,
    Spark,
    #[value(alias = "postgres")]
    Postgresql,
    Mysql,
    Sqlite,
}

impl From<DialectArg> for SqlDialect {
    fn from(d: DialectArg) -> Self {
        match d {
            DialectArg::Default => SqlDialect::Default,
            DialectArg::Spark => SqlDialect::Spark,
            DialectArg::Postgresql => SqlDialect::PostgreSql,
            DialectArg::Mysql => SqlDialect::MySql,
            DialectArg::Sqlite => SqlDialect::Sqlite,
        }
    }
}

// ── Config file schema ──────────────────────────────────────────────

/// Top-level graph config file format.
///
/// Example `graph.json`:
/// ```json
/// {
///   "nodes": {
///     "Person": {
///       "id_field": "person_id",
///       "fields": { "name": "Utf8", "age": "Int32" }
///     }
///   },
///   "relationships": {
///     "KNOWS": {
///       "source_field": "src_id",
///       "target_field": "dst_id",
///       "fields": { "since": "Int32" }
///     }
///   }
/// }
/// ```
#[derive(Deserialize)]
struct ConfigFile {
    nodes: HashMap<String, NodeDef>,
    #[serde(default)]
    relationships: HashMap<String, RelDef>,
}

#[derive(Deserialize)]
struct NodeDef {
    id_field: String,
    #[serde(default)]
    fields: HashMap<String, String>,
}

#[derive(Deserialize)]
struct RelDef {
    source_field: String,
    target_field: String,
    #[serde(default)]
    fields: HashMap<String, String>,
}

/// Parse a type name string into an Arrow DataType.
fn parse_data_type(s: &str) -> DataType {
    match s.to_lowercase().as_str() {
        "bool" | "boolean" => DataType::Boolean,
        "int8" => DataType::Int8,
        "int16" => DataType::Int16,
        "int32" | "int" => DataType::Int32,
        "int64" | "bigint" | "long" => DataType::Int64,
        "uint8" => DataType::UInt8,
        "uint16" => DataType::UInt16,
        "uint32" => DataType::UInt32,
        "uint64" => DataType::UInt64,
        "float16" | "half" => DataType::Float16,
        "float32" | "float" => DataType::Float32,
        "float64" | "double" => DataType::Float64,
        "date32" | "date" => DataType::Date32,
        "date64" => DataType::Date64,
        _ => DataType::Utf8, // default to string
    }
}

/// Build a [`GraphConfig`] and empty-schema datasets from the config file.
fn build_from_config(
    cfg: &ConfigFile,
) -> Result<(GraphConfig, HashMap<String, RecordBatch>), Box<dyn std::error::Error>> {
    let mut builder = GraphConfig::builder();
    let mut datasets: HashMap<String, RecordBatch> = HashMap::new();

    for (label, node_def) in &cfg.nodes {
        builder = builder.with_node_label(label, &node_def.id_field);

        // Build schema: id field + declared fields
        let mut fields = vec![Field::new(&node_def.id_field, DataType::Utf8, true)];
        for (fname, ftype) in &node_def.fields {
            fields.push(Field::new(fname, parse_data_type(ftype), true));
        }
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::new_empty(schema);
        datasets.insert(label.clone(), batch);
    }

    for (rel_type, rel_def) in &cfg.relationships {
        builder =
            builder.with_relationship(rel_type, &rel_def.source_field, &rel_def.target_field);

        // Build schema: source + target fields + declared fields
        let mut fields = vec![
            Field::new(&rel_def.source_field, DataType::Utf8, true),
            Field::new(&rel_def.target_field, DataType::Utf8, true),
        ];
        for (fname, ftype) in &rel_def.fields {
            fields.push(Field::new(fname, parse_data_type(ftype), true));
        }
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::new_empty(schema);
        datasets.insert(rel_type.clone(), batch);
    }

    let config = builder.build()?;
    Ok((config, datasets))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Read query from argument or stdin
    let query_str = match cli.query {
        Some(q) => q,
        None => {
            let mut buf = String::new();
            io::stdin().read_to_string(&mut buf)?;
            buf.trim().to_string()
        }
    };

    if query_str.is_empty() {
        eprintln!("Error: no Cypher query provided");
        std::process::exit(1);
    }

    // Load config
    let config_text = std::fs::read_to_string(&cli.config)
        .map_err(|e| format!("Failed to read config file {:?}: {}", cli.config, e))?;
    let config_file: ConfigFile = serde_json::from_str(&config_text)
        .map_err(|e| format!("Failed to parse config file: {}", e))?;

    let (graph_config, datasets) = build_from_config(&config_file)?;

    // Build CypherQuery, translate to SQL
    let cypher = CypherQuery::new(&query_str)?.with_config(graph_config);
    let dialect: SqlDialect = cli.dialect.into();
    let sql = cypher.to_sql(datasets, Some(dialect)).await?;

    println!("{sql}");
    Ok(())
}
