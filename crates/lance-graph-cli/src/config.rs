// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Configuration file loading for lgraph CLI.

use anyhow::{Context, Result};
use lance_graph::GraphConfig;
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
}

#[derive(Debug, Deserialize)]
pub struct RelationshipConfig {
    pub source_field: String,
    pub target_field: String,
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
            builder = builder.with_node_label(label, &node.id_field);
        }
        for (rel_type, rel) in &section.relationships {
            builder = builder.with_relationship(rel_type, &rel.source_field, &rel.target_field);
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
# [graph.nodes.Person]
# id_field = "person_id"
#
# [graph.nodes.Company]
# id_field = "company_id"
#
# [graph.relationships.WORKS_AT]
# source_field = "person_id"
# target_field = "company_id"
"#
}
