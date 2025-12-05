// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Simple single-table query executor with limited Cypher feature support
//!
//! This module provides a lightweight execution strategy for basic Cypher queries
//! that don't require the full DataFusion planner. It supports:
//! - Single-table scans with property filters
//! - Multi-hop path patterns via join chains
//! - Basic projections, DISTINCT, ORDER BY, SKIP, and LIMIT

mod aliases;
mod clauses;
mod expr;
mod path_executor;

pub(crate) use expr::{
    to_df_boolean_expr_simple, to_df_order_by_expr_simple, to_df_value_expr_simple,
};
pub(crate) use path_executor::PathExecutor;
