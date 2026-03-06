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
/// When a node/relationship has a `table_name` override, the physical table is
/// resolved using that name but registered under the graph label so Cypher
/// queries can reference it by label.
pub async fn build_namespace_context(
    config: &LGraphConfig,
    ns_path: &str,
) -> Result<SessionContext> {
    let graph_config = config
        .build_graph_config()?
        .context("A [graph] section is required to resolve tables from a namespace")?;

    let namespace = DirNamespace::new(ns_path);
    let ctx = SessionContext::new();

    // Collect (physical_table_name, register_as_label) pairs.
    let mut table_entries: Vec<(String, String)> = Vec::new();
    for nm in graph_config.node_mappings.values() {
        table_entries.push((
            nm.resolved_table_name().to_string(),
            nm.label.to_lowercase(),
        ));
    }
    for rm in graph_config.relationship_mappings.values() {
        table_entries.push((
            rm.resolved_table_name().to_string(),
            rm.relationship_type.to_lowercase(),
        ));
    }

    for (physical_name, register_as) in &table_entries {
        let mut request = DescribeTableRequest::new();
        request.id = Some(vec![physical_name.clone()]);

        let response = namespace
            .describe_table(request)
            .await
            .with_context(|| format!("resolving table '{physical_name}' in namespace"))?;

        let location = response
            .location
            .with_context(|| format!("no location for table '{physical_name}'"))?;

        let dataset = lance::dataset::Dataset::open(&location)
            .await
            .with_context(|| format!("opening dataset for table '{physical_name}'"))?;

        let provider: Arc<dyn datafusion::datasource::TableProvider> =
            Arc::new(LanceTableProvider::new(Arc::new(dataset), true, true));

        ctx.register_table(register_as, provider)
            .with_context(|| {
                format!("registering table '{physical_name}' as '{register_as}'")
            })?;
    }

    Ok(ctx)
}
