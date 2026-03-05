// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Helper for building a DataFusion SessionContext from a Unity Catalog.

use anyhow::{Context, Result};
use datafusion::execution::context::SessionContext;
use lance_graph::{default_table_readers, Connector, UnityCatalogConfig, UnityCatalogProvider};
use std::sync::Arc;

use crate::config::CatalogConfig;

/// Build a SessionContext with tables from Unity Catalog registered.
pub async fn build_catalog_context(catalog_cfg: &CatalogConfig) -> Result<SessionContext> {
    let uc_config = UnityCatalogConfig::new(&catalog_cfg.url);
    let provider =
        UnityCatalogProvider::new(uc_config).context("connecting to Unity Catalog")?;
    let readers = default_table_readers();
    let connector = Connector::new(Arc::new(provider), readers)
        .with_storage_options(catalog_cfg.storage_options.clone());
    let ctx = SessionContext::new();
    connector
        .register_schema(&ctx, &catalog_cfg.catalog_name, &catalog_cfg.schema_name)
        .await
        .context("registering catalog schema")?;
    Ok(ctx)
}
