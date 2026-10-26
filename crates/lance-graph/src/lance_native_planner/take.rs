// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Native materialization: take target node columns by row id.
//!
//! `CsrExpandExec` produces target *row ids*; `LanceTakeExec` turns those into
//! target *properties* via a `RowMaterializer`. Under the dense-ROWID model the
//! in-memory materializer is a direct `arrow::compute::take` by offset — the
//! concrete reason CSR beats a hash join.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::compute::{cast, take};
use arrow_array::{ArrayRef, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, SchemaRef};
use datafusion::common::{DFSchemaRef, Result as DFResult};
use datafusion::execution::TaskContext;
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use futures::StreamExt;

use crate::error::{GraphError, Result};

/// Materializes rows of a target node table by row id.
pub trait RowMaterializer: Send + Sync + fmt::Debug {
    /// Take `columns` (raw, unqualified names) for the given `row_ids`.
    /// The returned batch has one row per element of `row_ids`, columns in the
    /// requested order, named by their raw names.
    fn take(&self, row_ids: &UInt64Array, columns: &[String]) -> Result<RecordBatch>;
}

/// In-memory materializer over a fully-collected target node batch. Row id ==
/// offset into the batch (dense-ROWID model).
#[derive(Debug)]
pub struct InMemoryMaterializer {
    batch: RecordBatch,
}

impl InMemoryMaterializer {
    pub fn new(batch: RecordBatch) -> Self {
        Self { batch }
    }
}

impl RowMaterializer for InMemoryMaterializer {
    fn take(&self, row_ids: &UInt64Array, columns: &[String]) -> Result<RecordBatch> {
        let mut fields: Vec<Field> = Vec::with_capacity(columns.len());
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(columns.len());
        for name in columns {
            let col = self
                .batch
                .column_by_name(name)
                .ok_or_else(|| GraphError::ExecutionError {
                    message: format!("take: target column '{}' not found", name),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
            let taken = take(col, row_ids, None).map_err(|e| GraphError::ExecutionError {
                message: format!("take: failed on column '{}': {}", name, e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
            let nullable = self
                .batch
                .schema()
                .field_with_name(name)
                .map(|f| f.is_nullable())
                .unwrap_or(true);
            fields.push(Field::new(name, col.data_type().clone(), nullable));
            arrays.push(taken);
        }
        RecordBatch::try_new(Arc::new(arrow_schema::Schema::new(fields)), arrays).map_err(|e| {
            GraphError::ExecutionError {
                message: format!("take: build batch: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            }
        })
    }
}

/// Append materialized target columns to one input batch.
///
/// `row_id_idx` is the index of the row-id column in `input`. `take_cols` are
/// the raw target column names to materialize, in the same order as the
/// appended fields of `out_schema`. `out_schema` = `input.schema()` followed by
/// the qualified materialized columns.
pub(crate) fn take_batch(
    input: &RecordBatch,
    row_id_idx: usize,
    materializer: &dyn RowMaterializer,
    take_cols: &[String],
    out_schema: &SchemaRef,
) -> Result<RecordBatch> {
    // Nothing to materialize: output is the input re-stamped with out_schema.
    // (When take_cols is empty, out_schema == input.schema().)
    if take_cols.is_empty() {
        return RecordBatch::try_new(out_schema.clone(), input.columns().to_vec()).map_err(|e| {
            GraphError::ExecutionError {
                message: format!("take: build output batch: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            }
        });
    }

    let ids_u64 = cast(input.column(row_id_idx), &DataType::UInt64).map_err(|e| {
        GraphError::ExecutionError {
            message: format!("take: cast row id to u64: {}", e),
            location: snafu::Location::new(file!(), line!(), column!()),
        }
    })?;
    // Arrow 56.x: cast(_, UInt64) always yields a plain UInt64Array.
    let ids = ids_u64
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("cast to UInt64 yields UInt64Array");

    let materialized = materializer.take(ids, take_cols)?;

    let mut cols: Vec<ArrayRef> = input.columns().to_vec();
    cols.extend(materialized.columns().iter().cloned());

    RecordBatch::try_new(out_schema.clone(), cols).map_err(|e| GraphError::ExecutionError {
        message: format!("take: build output batch: {}", e),
        location: snafu::Location::new(file!(), line!(), column!()),
    })
}

/// Logical extension node for materializing target columns via take().
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LanceTakeNode {
    /// Input subplan (a `CsrExpandNode`).
    pub input: LogicalPlan,
    /// Lowercased target node table name (to collect rows from).
    pub target_table: String,
    /// Qualified column in `input` holding the row ids (e.g. `b__id`).
    pub row_id_column: String,
    /// Raw (unqualified, lowercased) target columns to materialize, in output order.
    pub take_cols: Vec<String>,
    /// Output schema = input schema + qualified materialized columns.
    pub schema: DFSchemaRef,
}

impl PartialOrd for LanceTakeNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (&self.target_table, &self.row_id_column, &self.take_cols).partial_cmp(&(
            &other.target_table,
            &other.row_id_column,
            &other.take_cols,
        ))
    }
}

impl UserDefinedLogicalNodeCore for LanceTakeNode {
    fn name(&self) -> &str {
        "LanceTake"
    }
    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }
    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }
    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }
    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "LanceTake: table={}, row_id={}, cols={:?}",
            self.target_table, self.row_id_column, self.take_cols
        )
    }
    fn with_exprs_and_inputs(
        &self,
        _exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> DFResult<Self> {
        Ok(Self {
            input: inputs.remove(0),
            ..self.clone()
        })
    }
}

/// Physical operator for `LanceTakeNode`.
#[derive(Debug)]
pub struct LanceTakeExec {
    input: Arc<dyn ExecutionPlan>,
    materializer: Arc<dyn RowMaterializer>,
    row_id_idx: usize,
    take_cols: Vec<String>,
    out_schema: SchemaRef,
    props: PlanProperties,
}

impl LanceTakeExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        materializer: Arc<dyn RowMaterializer>,
        row_id_idx: usize,
        take_cols: Vec<String>,
        out_schema: SchemaRef,
    ) -> Self {
        let props = PlanProperties::new(
            EquivalenceProperties::new(out_schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self {
            input,
            materializer,
            row_id_idx,
            take_cols,
            out_schema,
            props,
        }
    }
}

impl DisplayAs for LanceTakeExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LanceTakeExec: cols={:?}", self.take_cols)
    }
}

impl ExecutionPlan for LanceTakeExec {
    fn name(&self) -> &str {
        "LanceTakeExec"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn properties(&self) -> &PlanProperties {
        &self.props
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(LanceTakeExec::new(
            children[0].clone(),
            self.materializer.clone(),
            self.row_id_idx,
            self.take_cols.clone(),
            self.out_schema.clone(),
        )))
    }
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let materializer = self.materializer.clone();
        let row_id_idx = self.row_id_idx;
        let take_cols = self.take_cols.clone();
        let out_schema = self.out_schema.clone();
        let out_schema_for_stream = out_schema.clone();
        let stream = input.map(move |rb| {
            let rb = rb?;
            take_batch(&rb, row_id_idx, materializer.as_ref(), &take_cols, &out_schema)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            out_schema_for_stream,
            stream,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StringArray;
    use arrow_schema::Schema;

    fn target_batch() -> RecordBatch {
        // person table: id, name, age (raw, lowercased column names)
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::Int64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vec![0u64, 1, 2])),
                Arc::new(StringArray::from(vec!["alice", "bob", "carol"])),
                Arc::new(arrow_array::Int64Array::from(vec![30i64, 40, 50])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_in_memory_materializer_take_subset() {
        let m = InMemoryMaterializer::new(target_batch());
        let ids = UInt64Array::from(vec![2u64, 0]);
        let out = m.take(&ids, &["name".to_string()]).unwrap();
        assert_eq!(out.num_columns(), 1);
        let names = out.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(names.value(0), "carol");
        assert_eq!(names.value(1), "alice");
    }

    #[test]
    fn test_in_memory_materializer_missing_column_errors() {
        let m = InMemoryMaterializer::new(target_batch());
        let ids = UInt64Array::from(vec![0u64]);
        assert!(m.take(&ids, &["nonexistent".to_string()]).is_err());
    }

    #[test]
    fn test_take_batch_appends_qualified_columns() {
        // input: a__name, b__id  (b__id is the neighbor row id)
        let in_schema = Arc::new(Schema::new(vec![
            Field::new("a__name", DataType::Utf8, false),
            Field::new("b__id", DataType::UInt64, true),
        ]));
        let input = RecordBatch::try_new(
            in_schema,
            vec![
                Arc::new(StringArray::from(vec!["x", "y"])),
                Arc::new(UInt64Array::from(vec![1u64, 2])),
            ],
        )
        .unwrap();
        // out: a__name, b__id, b__name, b__age
        let out_schema = Arc::new(Schema::new(vec![
            Field::new("a__name", DataType::Utf8, false),
            Field::new("b__id", DataType::UInt64, true),
            Field::new("b__name", DataType::Utf8, true),
            Field::new("b__age", DataType::Int64, true),
        ]));
        let m = InMemoryMaterializer::new(target_batch());
        let out = take_batch(
            &input,
            1,
            &m,
            &["name".to_string(), "age".to_string()],
            &out_schema,
        )
        .unwrap();
        assert_eq!(out.num_rows(), 2);
        let b_name = out.column(2).as_any().downcast_ref::<StringArray>().unwrap();
        let b_age = out
            .column(3)
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .unwrap();
        assert_eq!(b_name.value(0), "bob"); // row id 1
        assert_eq!(b_name.value(1), "carol"); // row id 2
        assert_eq!(b_age.values(), &[40, 50]);
    }

    #[test]
    fn test_take_batch_empty_take_cols_passthrough() {
        // With no target columns to materialize, output == input (same rows/cols).
        let in_schema = Arc::new(Schema::new(vec![
            Field::new("a__name", DataType::Utf8, false),
            Field::new("b__id", DataType::UInt64, true),
        ]));
        let input = RecordBatch::try_new(
            in_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["x", "y"])),
                Arc::new(UInt64Array::from(vec![1u64, 2])),
            ],
        )
        .unwrap();
        let m = InMemoryMaterializer::new(target_batch());
        let out = take_batch(&input, 1, &m, &[], &in_schema).unwrap();
        assert_eq!(out.num_rows(), 2);
        assert_eq!(out.num_columns(), 2);
    }
}
