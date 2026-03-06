// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Configuration file loading for lgraph CLI.

use anyhow::{Context, Result};
use lance_graph::{GraphConfig, NodeMapping, RelationshipMapping};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Top-level config file structure (`lgraph.toml`).
#[derive(Debug, Deserialize)]
pub struct LGraphConfig {
    /// Path to a directory of Lance tables.
    pub namespace: Option<String>,

    /// Unity Catalog connection settings.
    pub catalog: Option<CatalogConfig>,

    /// Graph schema mappings (needed for Cypher queries).
    pub graph: Option<GraphSection>,
}

/// Unity Catalog connection config.
#[derive(Debug, Deserialize)]
pub struct CatalogConfig {
    pub url: String,
    pub catalog_name: String,
    pub schema_name: String,
    /// Cloud storage options (S3/Azure/GCS credentials).
    #[serde(default)]
    pub storage_options: HashMap<String, String>,
}

/// Graph schema section of the config.
#[derive(Debug, Deserialize)]
pub struct GraphSection {
    #[serde(default)]
    pub nodes: HashMap<String, NodeConfig>,
    #[serde(default)]
    pub relationships: HashMap<String, RelationshipConfig>,
}

#[derive(Debug, Deserialize)]
pub struct NodeConfig {
    pub id_field: String,
    /// Optional: actual table name if different from the node label.
    /// E.g., label "Person" can read from table "person_entity".
    pub table: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RelationshipConfig {
    pub source_field: String,
    pub target_field: String,
    /// Optional: actual table name if different from the relationship type.
    /// E.g., type "WORKS_AT" can read from table "employment_info".
    pub table: Option<String>,
}

impl LGraphConfig {
    /// Load config from a TOML file.
    pub fn load(path: &Path) -> Result<Self> {
        let content =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let config: LGraphConfig =
            toml::from_str(&content).with_context(|| format!("parsing {}", path.display()))?;
        Ok(config)
    }

    /// Build a `GraphConfig` from the `[graph]` section.
    pub fn build_graph_config(&self) -> Result<Option<GraphConfig>> {
        let section = match &self.graph {
            Some(g) => g,
            None => return Ok(None),
        };

        let mut builder = GraphConfig::builder();
        for (label, node) in &section.nodes {
            let mut mapping = NodeMapping::new(label, &node.id_field);
            if let Some(table) = &node.table {
                mapping = mapping.with_table_name(table);
            }
            builder = builder.with_node_mapping(mapping);
        }
        for (rel_type, rel) in &section.relationships {
            let mut mapping =
                RelationshipMapping::new(rel_type, &rel.source_field, &rel.target_field);
            if let Some(table) = &rel.table {
                mapping = mapping.with_table_name(table);
            }
            builder = builder.with_relationship_mapping(mapping);
        }
        let config = builder.build().context("building GraphConfig")?;
        Ok(Some(config))
    }
}

/// Generate a template config file.
pub fn template_config() -> &'static str {
    r#"# lgraph configuration file
# Uncomment and configure the data source you want to use.

# Option 1: Local directory of Lance / Parquet tables
# namespace = "/path/to/tables"

# Option 2: Unity Catalog
# [catalog]
# url = "http://localhost:8080/api/2.1/unity-catalog"
# catalog_name = "main"
# schema_name = "default"
# [catalog.storage_options]
# aws_access_key_id = "..."
# aws_secret_access_key = "..."

# Graph schema mappings (required for Cypher queries)
# Each section key is the graph label used in Cypher queries.
# Use "table" to map a label to a different physical table name.
#
# [graph.nodes.Person]
# id_field = "person_id"
# table = "person_entity"  # optional: reads from "person_entity" table
#
# [graph.nodes.Company]
# id_field = "company_id"
#
# [graph.relationships.WORKS_AT]
# source_field = "person_id"
# target_field = "company_id"
# table = "employment_info"  # optional: reads from "employment_info" table
"#
}
