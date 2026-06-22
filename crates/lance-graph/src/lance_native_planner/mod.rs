// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Native physical planner (placeholder)
//!
//! Rewritten in a later task to lower single-hop `Expand` onto CSR-backed
//! extension nodes. For now it keeps the original placeholder behavior so the
//! crate compiles between tasks.

mod direction;
mod csr_expand;
mod take;
mod extension_planner;

pub use direction::NativeDirection;
pub use extension_planner::CsrQueryPlanner;

use crate::config::GraphConfig;
use crate::datafusion_planner::GraphPhysicalPlanner;
use crate::error::Result;
use crate::logical_plan::LogicalOperator;
use datafusion::common::DFSchema;
use datafusion::logical_expr::{EmptyRelation, LogicalPlan};
use std::sync::Arc;

/// Placeholder Lance-native planner
pub struct LanceNativePlanner {
    #[allow(dead_code)]
    config: GraphConfig,
}

impl LanceNativePlanner {
    pub fn new(config: GraphConfig) -> Self {
        Self { config }
    }
}

impl GraphPhysicalPlanner for LanceNativePlanner {
    fn plan(&self, _logical_plan: &LogicalOperator) -> Result<LogicalPlan> {
        let schema = Arc::new(DFSchema::empty());
        Ok(LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema,
        }))
    }
}
