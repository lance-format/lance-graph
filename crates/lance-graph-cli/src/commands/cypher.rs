// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use anyhow::{bail, Context, Result};
use lance_graph::{CypherQuery, DirNamespace};

use crate::config::LGraphConfig;
use crate::output::{print_batch, OutputFormat};

pub async fn run(config: &LGraphConfig, query: &str, format: OutputFormat) -> Result<()> {
    let graph_config = config
        .build_graph_config()?
        .context("Cypher queries require a [graph] section in the config file")?;

    let cypher = CypherQuery::new(query)
        .context("parsing Cypher query")?
        .with_config(graph_config);

    let result = if let Some(ns_path) = &config.namespace {
        let namespace = DirNamespace::new(ns_path);
        cypher
            .execute_with_namespace(namespace, None)
            .await
            .context("executing Cypher query")?
    } else if let Some(catalog_cfg) = &config.catalog {
        let ctx = crate::catalog_helpers::build_catalog_context(catalog_cfg)
            .await
            .context("setting up catalog")?;
        cypher
            .execute_with_context(ctx)
            .await
            .context("executing Cypher query")?
    } else {
        bail!("No data source configured. Set 'namespace' or '[catalog]' in your config file.");
    };

    print_batch(&result, format)?;
    Ok(())
}
