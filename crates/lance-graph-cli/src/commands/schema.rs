// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use anyhow::{bail, Context, Result};

use crate::config::LGraphConfig;
use crate::output::OutputFormat;

pub async fn run(config: &LGraphConfig, table_name: &str, format: OutputFormat) -> Result<()> {
    let ctx = if let Some(ns_path) = &config.namespace {
        crate::namespace_helpers::build_namespace_context(config, ns_path)
            .await
            .context("setting up namespace tables")?
    } else if let Some(catalog_cfg) = &config.catalog {
        crate::catalog_helpers::build_catalog_context(catalog_cfg)
            .await
            .context("setting up catalog")?
    } else {
        bail!("No data source configured.");
    };

    let table_provider = ctx
        .table_provider(&table_name.to_lowercase())
        .await
        .context(format!("table '{}' not found", table_name))?;
    let schema = table_provider.schema();

    match format {
        OutputFormat::Json | OutputFormat::Jsonl => {
            for field in schema.fields() {
                let obj = serde_json::json!({
                    "name": field.name(),
                    "type": format!("{}", field.data_type()),
                    "nullable": field.is_nullable(),
                });
                println!("{}", serde_json::to_string(&obj).unwrap());
            }
        }
        OutputFormat::Table => {
            println!("Schema for '{table_name}':");
            println!("{:<30} {:<20} Nullable", "Name", "Type");
            println!("{}", "-".repeat(60));
            for field in schema.fields() {
                println!(
                    "{:<30} {:<20} {}",
                    field.name(),
                    field.data_type(),
                    field.is_nullable()
                );
            }
        }
        OutputFormat::Csv => {
            println!("name,type,nullable");
            for field in schema.fields() {
                println!(
                    "{},{},{}",
                    field.name(),
                    field.data_type(),
                    field.is_nullable()
                );
            }
        }
    }

    Ok(())
}
