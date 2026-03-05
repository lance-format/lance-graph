// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use anyhow::{bail, Context, Result};
use lance_graph::SqlQuery;

use crate::config::LGraphConfig;
use crate::output::{print_batch, OutputFormat};

pub async fn run(config: &LGraphConfig, query: &str, format: OutputFormat) -> Result<()> {
    let sql = SqlQuery::new(query);

    let ctx = if let Some(ns_path) = &config.namespace {
        crate::namespace_helpers::build_namespace_context(config, ns_path)
            .await
            .context("setting up namespace tables")?
    } else if let Some(catalog_cfg) = &config.catalog {
        crate::catalog_helpers::build_catalog_context(catalog_cfg)
            .await
            .context("setting up catalog")?
    } else {
        bail!("No data source configured. Set 'namespace' or '[catalog]' in your config file.");
    };

    let result = sql
        .execute_with_context(ctx)
        .await
        .context("executing SQL query")?;

    print_batch(&result, format)?;
    Ok(())
}
