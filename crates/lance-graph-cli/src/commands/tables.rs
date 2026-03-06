// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use anyhow::{bail, Context, Result};

use crate::config::LGraphConfig;
use crate::output::OutputFormat;

struct TableEntry {
    label: String,
    table: String,
    kind: &'static str, // "node" or "relationship"
}

pub async fn run(config: &LGraphConfig, format: OutputFormat) -> Result<()> {
    if let Some(ns_path) = &config.namespace {
        let graph = config.build_graph_config()?;
        match graph {
            Some(gc) => {
                let mut entries: Vec<TableEntry> = Vec::new();
                for nm in gc.node_mappings.values() {
                    entries.push(TableEntry {
                        label: nm.label.clone(),
                        table: nm.resolved_table_name().to_string(),
                        kind: "node",
                    });
                }
                for rm in gc.relationship_mappings.values() {
                    entries.push(TableEntry {
                        label: rm.relationship_type.clone(),
                        table: rm.resolved_table_name().to_string(),
                        kind: "relationship",
                    });
                }
                entries.sort_by(|a, b| a.label.cmp(&b.label));
                print_entries(&entries, &format!("namespace: {ns_path}"), format);
            }
            None => {
                eprintln!("No [graph] section in config. Cannot determine tables in namespace.");
                eprintln!("Add node and relationship mappings to your lgraph.toml.");
            }
        }
    } else if let Some(catalog_cfg) = &config.catalog {
        let ctx = crate::catalog_helpers::build_catalog_context(catalog_cfg)
            .await
            .context("setting up catalog")?;
        let catalog_list = ctx.catalog_names();
        let mut entries: Vec<TableEntry> = Vec::new();
        for catalog_name in &catalog_list {
            if let Some(catalog) = ctx.catalog(catalog_name) {
                for schema_name in catalog.schema_names() {
                    if let Some(schema) = catalog.schema(&schema_name) {
                        for name in schema.table_names() {
                            entries.push(TableEntry {
                                label: name.clone(),
                                table: name,
                                kind: "table",
                            });
                        }
                    }
                }
            }
        }
        entries.sort_by(|a, b| a.label.cmp(&b.label));
        let source = format!(
            "catalog: {}.{}",
            catalog_cfg.catalog_name, catalog_cfg.schema_name
        );
        print_entries(&entries, &source, format);
    } else {
        bail!("No data source configured. Set 'namespace' or '[catalog]' in your config file.");
    }

    Ok(())
}

fn print_entries(entries: &[TableEntry], source: &str, format: OutputFormat) {
    match format {
        OutputFormat::Json | OutputFormat::Jsonl => {
            for e in entries {
                let mut obj = serde_json::json!({
                    "label": e.label,
                    "kind": e.kind,
                });
                if e.table != e.label {
                    obj["table"] = serde_json::Value::String(e.table.clone());
                }
                println!("{}", serde_json::to_string(&obj).unwrap());
            }
        }
        OutputFormat::Table => {
            println!("Tables ({source}):");
            for e in entries {
                if e.table != e.label {
                    println!("  {} -> {} ({})", e.label, e.table, e.kind);
                } else {
                    println!("  {} ({})", e.label, e.kind);
                }
            }
        }
        OutputFormat::Csv => {
            println!("label,table,kind");
            for e in entries {
                println!("{},{},{}", e.label, e.table, e.kind);
            }
        }
    }
}
