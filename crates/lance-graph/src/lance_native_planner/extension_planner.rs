// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Physical planning for the native CSR extension nodes.
//!
//! `CsrQueryPlanner` is registered on the execution `SessionContext`. It runs
//! the `DefaultPhysicalPlanner` with `CsrExtensionPlanner`, which builds the
//! `CsrIndex` (from the edge table) and the `InMemoryMaterializer` (from the
//! target node table) at physical-planning time.

use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow_schema::Field;
use async_trait::async_trait;
use datafusion::common::Result as DFResult;
use datafusion::error::DataFusionError;
use datafusion::execution::context::{QueryPlanner, SessionContext, SessionState};
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNode};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};

use super::csr_expand::{CsrExpandExec, CsrExpandNode};
use super::direction::NativeDirection;
use super::take::{InMemoryMaterializer, LanceTakeExec, LanceTakeNode};
use crate::csr_index::CsrIndexBuilder;

/// Collect a registered table to a single `RecordBatch`.
///
/// Phase 2 builds the CSR / materializer eagerly at physical-planning time, so
/// the whole edge / target table is read into memory here. An empty table yields
/// an empty batch (correct schema), producing an empty CSR — i.e. no traversals.
async fn collect_table(
    session_state: &SessionState,
    table: &str,
) -> DFResult<arrow_array::RecordBatch> {
    let ctx = SessionContext::new_with_state(session_state.clone());
    let df = ctx.table(table).await?;
    let schema = df.schema().inner().clone();
    let batches = df.collect().await?;
    concat_batches(&schema, &batches).map_err(|e| DataFusionError::Execution(e.to_string()))
}

/// Extension planner that lowers `CsrExpandNode` and `LanceTakeNode`.
#[derive(Debug)]
pub struct CsrExtensionPlanner;

#[async_trait]
impl ExtensionPlanner for CsrExtensionPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        session_state: &SessionState,
    ) -> DFResult<Option<Arc<dyn ExecutionPlan>>> {
        if let Some(expand) = node.as_any().downcast_ref::<CsrExpandNode>() {
            let input = physical_inputs
                .first()
                .ok_or_else(|| DataFusionError::Internal(
                    "CsrExpandNode: expected 1 physical input, got 0".to_string(),
                ))?
                .clone();

            // Build CSR from the edge table (reverse columns for Incoming).
            let edges = collect_table(session_state, &expand.rel_type).await?;
            let (src_col, dst_col) = match expand.direction {
                NativeDirection::Outgoing => (&expand.src_field, &expand.dst_field),
                NativeDirection::Incoming => (&expand.dst_field, &expand.src_field),
            };
            let csr = CsrIndexBuilder::new()
                .add_edges_from_batch_with_columns(&edges, src_col, dst_col)
                .map_err(|e| DataFusionError::Execution(e.to_string()))?
                .build();

            let in_schema = input.schema();
            let source_id_idx = in_schema.index_of(&expand.source_id_column).map_err(|e| {
                DataFusionError::Execution(format!(
                    "CsrExpand: source id column '{}' not found in input: {}",
                    expand.source_id_column, e
                ))
            })?;
            let neighbor_field =
                Field::new(&expand.neighbor_column, expand.neighbor_data_type.clone(), true);
            let out_schema = expand.schema.inner().clone();

            return Ok(Some(Arc::new(CsrExpandExec::new(
                input,
                Arc::new(csr),
                source_id_idx,
                neighbor_field,
                out_schema,
            ))));
        }

        if let Some(take) = node.as_any().downcast_ref::<LanceTakeNode>() {
            let input = physical_inputs
                .first()
                .ok_or_else(|| DataFusionError::Internal(
                    "LanceTakeNode: expected 1 physical input, got 0".to_string(),
                ))?
                .clone();

            let target = collect_table(session_state, &take.target_table).await?;
            let materializer = Arc::new(InMemoryMaterializer::new(target));

            let in_schema = input.schema();
            let row_id_idx = in_schema.index_of(&take.row_id_column).map_err(|e| {
                DataFusionError::Execution(format!(
                    "LanceTake: row id column '{}' not found in input: {}",
                    take.row_id_column, e
                ))
            })?;
            let out_schema = take.schema.inner().clone();

            return Ok(Some(Arc::new(LanceTakeExec::new(
                input,
                materializer,
                row_id_idx,
                take.take_cols.clone(),
                out_schema,
            ))));
        }

        Ok(None)
    }
}

/// Query planner that installs `CsrExtensionPlanner`.
#[derive(Debug, Default)]
pub struct CsrQueryPlanner;

impl CsrQueryPlanner {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl QueryPlanner for CsrQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let planner =
            DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(CsrExtensionPlanner)]);
        planner
            .create_physical_plan(logical_plan, session_state)
            .await
    }
}
