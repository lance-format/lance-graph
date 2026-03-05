// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Helper for building a DataFusion SessionContext from a namespace directory.

use anyhow::{Context, Result};
use datafusion::execution::context::SessionContext;
use lance::datafusion::LanceTableProvider;
use lance_graph::DirNamespace;
use lance_namespace::models::DescribeTableRequest;
use lance_namespace::LanceNamespace;
use std::sync::Arc;

use crate::config::LGraphConfig;

/// Build a SessionContext with tables from a namespace directory registered.
///
/// Uses the `[graph]` config section to determine which tables to register.
pub async fn build_namespace_context(config: &LGraphConfig, ns_path: &str) -> Result<SessionContext> {
    let graph_config = config
        .build_graph_config()?
        .context("A [graph] section is required to resolve tables from a namespace")?;

    let namespace = DirNamespace::new(ns_path);
    let ctx = SessionContext::new();

    // Collect all table names from graph config.
    let mut table_names: Vec<String> = Vec::new();
    for nm in graph_config.node_mappings.values() {
        table_names.push(nm.label.clone());
    }
    for rm in graph_config.relationship_mappings.values() {
        table_names.push(rm.relationship_type.clone());
    }

    for table_name in &table_names {
        let mut request = DescribeTableRequest::new();
        request.id = Some(vec![table_name.clone()]);

        let response = namespace
            .describe_table(request)
            .await
            .with_context(|| format!("resolving table '{table_name}' in namespace"))?;

        let location = response
            .location
            .with_context(|| format!("no location for table '{table_name}'"))?;

        let dataset = lance::dataset::Dataset::open(&location)
            .await
            .with_context(|| format!("opening dataset for table '{table_name}'"))?;

        let provider: Arc<dyn datafusion::datasource::TableProvider> =
            Arc::new(LanceTableProvider::new(Arc::new(dataset), true, true));

        let normalized_name = table_name.to_lowercase();
        ctx.register_table(&normalized_name, provider)
            .with_context(|| format!("registering table '{table_name}'"))?;
    }

    Ok(ctx)
}
