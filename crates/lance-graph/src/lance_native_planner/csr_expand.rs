// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Native single-hop expand: logical node + physical operator + core function.
//!
//! `CsrExpandExec` does topology only — for each input row it looks up the
//! source vertex's neighbors in the CSR index and emits one output row per
//! neighbor, carrying through all input columns and appending the neighbor row
//! id as a new column. Target property materialization is handled separately by
//! `LanceTakeExec`.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::compute::{cast, take};
use arrow_array::{Array, ArrayRef, RecordBatch, UInt32Array, UInt64Array};
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

use super::direction::NativeDirection;
use crate::csr_index::CsrIndex;
use crate::error::{GraphError, Result};

/// Expand one input batch: for every input row, append one output row per
/// neighbor of that row's source vertex.
///
/// `source_id_idx` is the column index of the source vertex id within `input`.
/// `neighbor_field` is the appended column (its data type is the target id
/// field's type; neighbor ids are cast into it). `out_schema` must equal
/// `input.schema()` fields followed by `neighbor_field`.
pub(crate) fn expand_batch(
    input: &RecordBatch,
    source_id_idx: usize,
    csr: &CsrIndex,
    neighbor_field: &Field,
    out_schema: &SchemaRef,
) -> Result<RecordBatch> {
    let map_err = |e: arrow_schema::ArrowError, what: &str| GraphError::ExecutionError {
        message: format!("CsrExpand {}: {}", what, e),
        location: snafu::Location::new(file!(), line!(), column!()),
    };

    // Source ids may be any integer type; normalize to u64.
    let src_u64 = cast(input.column(source_id_idx), &DataType::UInt64)
        .map_err(|e| map_err(e, "cast source id to u64"))?;
    // Arrow 56.x: cast(_, UInt64) always yields a plain UInt64Array.
    let src = src_u64
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("cast to UInt64 yields UInt64Array");

    let mut parent_idx: Vec<u32> = Vec::with_capacity(input.num_rows());
    let mut neighbors: Vec<u64> = Vec::with_capacity(input.num_rows());
    for row in 0..input.num_rows() {
        if src.is_null(row) {
            continue;
        }
        let row_u32 = u32::try_from(row).map_err(|_| GraphError::ExecutionError {
            message: "CsrExpand: input batch row index exceeds u32::MAX".to_string(),
            location: snafu::Location::new(file!(), line!(), column!()),
        })?;
        for &n in csr.neighbors(src.value(row)) {
            parent_idx.push(row_u32);
            neighbors.push(n);
        }
    }

    let take_idx = UInt32Array::from(parent_idx);
    let mut cols: Vec<ArrayRef> = Vec::with_capacity(input.num_columns() + 1);
    for c in input.columns() {
        cols.push(take(c, &take_idx, None).map_err(|e| map_err(e, "take carried column"))?);
    }
    let neigh_u64 = Arc::new(UInt64Array::from(neighbors)) as ArrayRef;
    let neigh_col = cast(&neigh_u64, neighbor_field.data_type())
        .map_err(|e| map_err(e, "cast neighbor id"))?;
    cols.push(neigh_col);

    RecordBatch::try_new(out_schema.clone(), cols).map_err(|e| GraphError::ExecutionError {
        message: format!("CsrExpand build output batch: {}", e),
        location: snafu::Location::new(file!(), line!(), column!()),
    })
}

/// Logical extension node for a single-hop CSR expand.
///
/// Holds only hashable metadata; the physical operator (and its `CsrIndex`) is
/// constructed by the extension planner at physical-planning time.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CsrExpandNode {
    /// Source subplan (a node scan, optionally with a source-only filter).
    pub input: LogicalPlan,
    /// Relationship type (lowercased table name to look up the edge table).
    pub rel_type: String,
    /// Edge table column holding source vertex ids.
    pub src_field: String,
    /// Edge table column holding destination vertex ids.
    pub dst_field: String,
    /// Traversal direction.
    pub direction: NativeDirection,
    /// Qualified column in `input` carrying the source vertex id (e.g. `a__id`).
    pub source_id_column: String,
    /// Qualified output column for the neighbor row id (e.g. `b__id`).
    pub neighbor_column: String,
    /// Arrow data type of the neighbor column (target id field's type).
    pub neighbor_data_type: DataType,
    /// Output schema = input schema + neighbor column.
    pub schema: DFSchemaRef,
}

impl PartialOrd for CsrExpandNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (
            &self.rel_type,
            &self.src_field,
            &self.dst_field,
            &self.source_id_column,
            &self.neighbor_column,
        )
            .partial_cmp(&(
                &other.rel_type,
                &other.src_field,
                &other.dst_field,
                &other.source_id_column,
                &other.neighbor_column,
            ))
    }
}

impl UserDefinedLogicalNodeCore for CsrExpandNode {
    fn name(&self) -> &str {
        "CsrExpand"
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
            "CsrExpand: rel={}, dir={:?}, src={}, neighbor={}",
            self.rel_type, self.direction, self.source_id_column, self.neighbor_column
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

/// Physical operator for `CsrExpandNode`.
#[derive(Debug)]
pub struct CsrExpandExec {
    input: Arc<dyn ExecutionPlan>,
    csr: Arc<CsrIndex>,
    source_id_idx: usize,
    neighbor_field: Field,
    out_schema: SchemaRef,
    props: PlanProperties,
}

impl CsrExpandExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        csr: Arc<CsrIndex>,
        source_id_idx: usize,
        neighbor_field: Field,
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
            csr,
            source_id_idx,
            neighbor_field,
            out_schema,
            props,
        }
    }
}

impl DisplayAs for CsrExpandExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "CsrExpandExec: neighbor={}", self.neighbor_field.name())
    }
}

impl ExecutionPlan for CsrExpandExec {
    fn name(&self) -> &str {
        "CsrExpandExec"
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
        Ok(Arc::new(CsrExpandExec::new(
            children[0].clone(),
            self.csr.clone(),
            self.source_id_idx,
            self.neighbor_field.clone(),
            self.out_schema.clone(),
        )))
    }
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let csr = self.csr.clone();
        let idx = self.source_id_idx;
        let field = self.neighbor_field.clone();
        let out_schema = self.out_schema.clone();
        let out_schema_for_stream = out_schema.clone();
        let stream = input.map(move |rb| {
            let rb = rb?;
            expand_batch(&rb, idx, &csr, &field, &out_schema)
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
    use arrow_schema::Schema;
    use crate::csr_index::CsrIndexBuilder;

    fn input_batch() -> RecordBatch {
        // a__id = [0,1,2,3], a__name = ["n0","n1","n2","n3"]
        let schema = Arc::new(Schema::new(vec![
            Field::new("a__id", DataType::UInt64, false),
            Field::new("a__name", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vec![0u64, 1, 2, 3])),
                Arc::new(arrow_array::StringArray::from(vec!["n0", "n1", "n2", "n3"])),
            ],
        )
        .unwrap()
    }

    fn out_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a__id", DataType::UInt64, false),
            Field::new("a__name", DataType::Utf8, false),
            Field::new("b__id", DataType::UInt64, true),
        ]))
    }

    #[test]
    fn test_expand_batch_outgoing() {
        // 0->1, 0->2, 1->2, 3-> (none)
        let csr = CsrIndexBuilder::new()
            .with_num_vertices(4)
            .add_edge(0, 1)
            .add_edge(0, 2)
            .add_edge(1, 2)
            .build();
        let neighbor_field = Field::new("b__id", DataType::UInt64, true);
        let out = expand_batch(&input_batch(), 0, &csr, &neighbor_field, &out_schema()).unwrap();

        assert_eq!(out.num_rows(), 3);
        let a_id = out.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
        let b_id = out.column(2).as_any().downcast_ref::<UInt64Array>().unwrap();
        let a_name = out
            .column(1)
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .unwrap();
        // Rows: (0,n0,1) (0,n0,2) (1,n1,2)
        assert_eq!(a_id.values(), &[0, 0, 1]);
        assert_eq!(b_id.values(), &[1, 2, 2]);
        assert_eq!(a_name.value(0), "n0");
        assert_eq!(a_name.value(1), "n0");
        assert_eq!(a_name.value(2), "n1");
    }

    #[test]
    fn test_expand_batch_no_neighbors_and_out_of_range() {
        let csr = CsrIndexBuilder::new().with_num_vertices(2).build(); // no edges
        let neighbor_field = Field::new("b__id", DataType::UInt64, true);
        let out = expand_batch(&input_batch(), 0, &csr, &neighbor_field, &out_schema()).unwrap();
        assert_eq!(out.num_rows(), 0);
    }

    #[test]
    fn test_expand_batch_casts_source_id_from_int64() {
        // Source id column is Int64 (not UInt64): must still work.
        let schema = Arc::new(Schema::new(vec![Field::new("a__id", DataType::Int64, false)]));
        let input = RecordBatch::try_new(
            schema,
            vec![Arc::new(arrow_array::Int64Array::from(vec![0i64, 1]))],
        )
        .unwrap();
        let out_schema = Arc::new(Schema::new(vec![
            Field::new("a__id", DataType::Int64, false),
            Field::new("b__id", DataType::UInt64, true),
        ]));
        let csr = CsrIndexBuilder::new()
            .with_num_vertices(2)
            .add_edge(0, 1)
            .build();
        let neighbor_field = Field::new("b__id", DataType::UInt64, true);
        let out = expand_batch(&input, 0, &csr, &neighbor_field, &out_schema).unwrap();
        assert_eq!(out.num_rows(), 1);
        let b_id = out.column(1).as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(b_id.values(), &[1]);
    }
}
