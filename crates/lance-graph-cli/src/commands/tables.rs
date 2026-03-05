// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use anyhow::{bail, Context, Result};

use crate::config::LGraphConfig;
use crate::output::OutputFormat;

pub async fn run(config: &LGraphConfig, format: OutputFormat) -> Result<()> {
    if let Some(ns_path) = &config.namespace {
        // List tables based on graph config node/relationship labels.
        let graph = config.build_graph_config()?;
        match graph {
            Some(gc) => {
                let mut table_names: Vec<String> = Vec::new();
                for nm in gc.node_mappings.values() {
                    table_names.push(nm.label.clone());
                }
                for rm in gc.relationship_mappings.values() {
                    table_names.push(rm.relationship_type.clone());
                }
                table_names.sort();
                print_table_list(&table_names, &format!("namespace: {ns_path}"), format);
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
        let mut table_names = Vec::new();
        for catalog_name in &catalog_list {
            if let Some(catalog) = ctx.catalog(catalog_name) {
                for schema_name in catalog.schema_names() {
                    if let Some(schema) = catalog.schema(&schema_name) {
                        for name in schema.table_names() {
                            table_names.push(name);
                        }
                    }
                }
            }
        }
        table_names.sort();
        let source = format!(
            "catalog: {}.{}",
            catalog_cfg.catalog_name, catalog_cfg.schema_name
        );
        print_table_list(&table_names, &source, format);
    } else {
        bail!("No data source configured. Set 'namespace' or '[catalog]' in your config file.");
    }

    Ok(())
}

fn print_table_list(tables: &[String], source: &str, format: OutputFormat) {
    match format {
        OutputFormat::Json | OutputFormat::Jsonl => {
            for name in tables {
                let obj = serde_json::json!({ "table": name });
                println!("{}", serde_json::to_string(&obj).unwrap());
            }
        }
        OutputFormat::Table => {
            println!("Tables ({source}):");
            for name in tables {
                println!("  {name}");
            }
        }
        OutputFormat::Csv => {
            println!("table");
            for name in tables {
                println!("{name}");
            }
        }
    }
}
