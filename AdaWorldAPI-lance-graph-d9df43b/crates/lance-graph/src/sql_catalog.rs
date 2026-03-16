// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bridge between catalog connectors and the SQL query engine.
//!
//! Provides utilities for building a DataFusion `SessionContext` from a
//! [`Connector`], enabling users to auto-register tables from an external
//! catalog and query them via [`SqlQuery`] or [`SqlEngine`].

use crate::error::{GraphError, Result};
use datafusion::execution::context::SessionContext;
use lance_graph_catalog::connector::Connector;

/// Build a DataFusion `SessionContext` with all tables from a catalog schema
/// auto-registered.
///
/// This discovers tables in the given catalog/schema via the connector's
/// [`CatalogProvider`] and registers them using the appropriate [`TableReader`]
/// based on each table's data format.
///
/// # Example
///
/// ```no_run
/// # use lance_graph::sql_catalog::build_context_from_connector;
/// # use lance_graph_catalog::Connector;
/// # async fn example(connector: &Connector) {
/// let ctx = build_context_from_connector(connector, "unity", "default")
///     .await
///     .unwrap();
/// // ctx now has all tables from unity.default registered
/// # }
/// ```
pub async fn build_context_from_connector(
    connector: &Connector,
    catalog_name: &str,
    schema_name: &str,
) -> Result<SessionContext> {
    let ctx = SessionContext::new();
    connector
        .register_schema(&ctx, catalog_name, schema_name)
        .await
        .map_err(|e| GraphError::PlanError {
            message: format!("Failed to register catalog tables: {}", e),
            location: snafu::Location::new(file!(), line!(), column!()),
        })?;
    Ok(ctx)
}
