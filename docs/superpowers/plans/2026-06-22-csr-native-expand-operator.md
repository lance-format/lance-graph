# Phase 2: CSR-backed Native Single-Hop Expand — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute single-hop Cypher `Expand` natively via the Phase 1 `CsrIndex` — a custom DataFusion `ExecutionPlan` does neighbor lookups (`CsrExpandExec`) and a `take()` materializes target columns (`LanceTakeExec`), instead of relationship-scan + two joins.

**Architecture:** `LanceNativePlanner` wraps `DataFusionPlanner` and overrides only `Expand` lowering, emitting two logical extension nodes (`CsrExpandNode`, `LanceTakeNode`) over the existing source scan; anything it can't serve delegates to the join path. A `CsrQueryPlanner` registered on the `SessionContext` turns those nodes into physical operators, building the CSR and an in-memory row materializer at physical-planning time. Dense-ROWID id model: CSR vertex id == node row id; target props via `take()`.

**Tech Stack:** Rust, DataFusion 50.3 (`UserDefinedLogicalNodeCore`, `ExtensionPlanner`, `QueryPlanner`), Arrow 56.2 (`arrow::compute::{take, cast}`), async-trait 0.1, tokio (tests).

**Spec:** `docs/superpowers/specs/2026-06-22-csr-native-expand-operator-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `crates/lance-graph/src/csr_index.rs` | Modify | Add `add_edges_from_batch_with_columns` |
| `crates/lance-graph/src/datafusion_planner/mod.rs` | Modify | `mod expression;` → `pub(crate) mod expression;` |
| `crates/lance-graph/src/lance_native_planner.rs` | Delete | Replaced by directory module |
| `crates/lance-graph/src/lance_native_planner/mod.rs` | Create | `LanceNativePlanner`: native-vs-fallback + lowering; re-exports |
| `crates/lance-graph/src/lance_native_planner/direction.rs` | Create | `NativeDirection` enum |
| `crates/lance-graph/src/lance_native_planner/take.rs` | Create | `RowMaterializer`, `InMemoryMaterializer`, `take_batch`, `LanceTakeNode`, `LanceTakeExec` |
| `crates/lance-graph/src/lance_native_planner/csr_expand.rs` | Create | `expand_batch`, `CsrExpandNode`, `CsrExpandExec` |
| `crates/lance-graph/src/lance_native_planner/extension_planner.rs` | Create | `CsrExtensionPlanner`, `CsrQueryPlanner` |
| `crates/lance-graph/src/query.rs` | Modify | Wire `ExecutionStrategy::LanceNative` (in-memory datasets path) |
| `crates/lance-graph/tests/test_lance_native_expand.rs` | Create | End-to-end parity tests |

Run all crate tests with: `cargo test -p lance-graph`
Run a single test with: `cargo test -p lance-graph <test_name> -- --exact` (or `cargo test -p lance-graph --test test_lance_native_expand <name>` for integration tests).

---

## Task 1: Generalize CSR builder with custom edge column names

**Files:**
- Modify: `crates/lance-graph/src/csr_index.rs:240-274` (the `add_edges_from_batch` method) and its test module.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `crates/lance-graph/src/csr_index.rs` (before the closing `}`):

```rust
    #[test]
    fn test_build_from_record_batch_custom_columns() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("src_person_id", DataType::UInt64, false),
            Field::new("dst_person_id", DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vec![0, 0, 1])),
                Arc::new(UInt64Array::from(vec![1, 2, 2])),
            ],
        )
        .unwrap();

        // Forward (outgoing): src -> dst
        let idx = CsrIndexBuilder::new()
            .add_edges_from_batch_with_columns(&batch, "src_person_id", "dst_person_id")
            .unwrap()
            .build();
        assert_eq!(idx.neighbors(0), &[1, 2]);
        assert_eq!(idx.neighbors(1), &[2]);

        // Reversed (incoming): swap the column args -> dst -> src
        let rev = CsrIndexBuilder::new()
            .add_edges_from_batch_with_columns(&batch, "dst_person_id", "src_person_id")
            .unwrap()
            .build();
        assert_eq!(rev.neighbors(2), &[0, 1]);
        assert_eq!(rev.neighbors(1), &[0]);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p lance-graph test_build_from_record_batch_custom_columns`
Expected: FAIL — `no method named add_edges_from_batch_with_columns`.

- [ ] **Step 3: Implement**

In `crates/lance-graph/src/csr_index.rs`, replace the existing `add_edges_from_batch` method (lines ~239-274) with a thin wrapper plus the generalized method:

```rust
    /// Add edges from an Arrow RecordBatch with `src_id` and `dst_id` columns.
    pub fn add_edges_from_batch(self, batch: &RecordBatch) -> Result<Self> {
        self.add_edges_from_batch_with_columns(batch, "src_id", "dst_id")
    }

    /// Add edges from an Arrow RecordBatch, reading source vertex ids from
    /// `src_col` and destination vertex ids from `dst_col`.
    ///
    /// Both columns must be `UInt64`. To build a reversed (incoming/CSC) index,
    /// pass the destination column name as `src_col` and vice versa.
    pub fn add_edges_from_batch_with_columns(
        mut self,
        batch: &RecordBatch,
        src_col: &str,
        dst_col: &str,
    ) -> Result<Self> {
        let src_array = batch
            .column_by_name(src_col)
            .ok_or_else(|| GraphError::PlanError {
                message: format!("Edge batch missing '{}' column", src_col),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| GraphError::PlanError {
                message: format!("'{}' column must be UInt64", src_col),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        let dst_array = batch
            .column_by_name(dst_col)
            .ok_or_else(|| GraphError::PlanError {
                message: format!("Edge batch missing '{}' column", dst_col),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| GraphError::PlanError {
                message: format!("'{}' column must be UInt64", dst_col),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        for i in 0..batch.num_rows() {
            self.edges.push((src_array.value(i), dst_array.value(i)));
        }

        Ok(self)
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p lance-graph csr_index`
Expected: PASS (including the existing `test_build_from_record_batch`, which now routes through the wrapper).

- [ ] **Step 5: Commit**

```bash
git add crates/lance-graph/src/csr_index.rs
git commit -m "feat(csr): add_edges_from_batch_with_columns for custom edge column names"
```

---

## Task 2: Module skeleton + `NativeDirection`

Converts the placeholder file into a directory module so later tasks have a home. Keeps the existing `LanceNativePlanner` placeholder behavior compiling (it is rewritten in Task 6).

**Files:**
- Delete: `crates/lance-graph/src/lance_native_planner.rs`
- Create: `crates/lance-graph/src/lance_native_planner/mod.rs`
- Create: `crates/lance-graph/src/lance_native_planner/direction.rs`

- [ ] **Step 1: Create the direction enum**

Create `crates/lance-graph/src/lance_native_planner/direction.rs`:

```rust
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Traversal direction for the native CSR expand operators.

/// Direction a single-hop expand traverses. `Undirected` is intentionally
/// absent — undirected expands fall back to the DataFusion join planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativeDirection {
    /// Follow edges source -> destination (CSR).
    Outgoing,
    /// Follow edges destination -> source (CSC / reversed).
    Incoming,
}
```

- [ ] **Step 2: Move the placeholder into the new mod.rs**

Delete `crates/lance-graph/src/lance_native_planner.rs` and create
`crates/lance-graph/src/lance_native_planner/mod.rs` with the **exact previous
contents** of the placeholder file, with one added line at the top of the module
body to declare the submodule. The full file:

```rust
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Native physical planner (placeholder)
//!
//! Rewritten in Task 6 to lower single-hop `Expand` onto CSR-backed
//! extension nodes. For now it keeps the original placeholder behavior so the
//! crate compiles between tasks.

mod direction;

pub use direction::NativeDirection;

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
```

(The placeholder's `#[cfg(test)] mod tests { ... }` is dropped; Task 6 adds real tests.)

- [ ] **Step 3: Build to verify the module restructure compiles**

Run: `cargo build -p lance-graph`
Expected: builds (a warning about unused `NativeDirection` re-export is acceptable; it is consumed in Task 3).

- [ ] **Step 4: Commit**

```bash
git add -A crates/lance-graph/src/lance_native_planner.rs crates/lance-graph/src/lance_native_planner/
git commit -m "refactor(native): promote lance_native_planner to module dir; add NativeDirection"
```

---

## Task 3: `expand_batch` + `CsrExpandNode` + `CsrExpandExec`

**Files:**
- Create: `crates/lance-graph/src/lance_native_planner/csr_expand.rs`
- Modify: `crates/lance-graph/src/lance_native_planner/mod.rs` (add `mod csr_expand;`)

- [ ] **Step 1: Write the failing test for the pure expansion function**

Create `crates/lance-graph/src/lance_native_planner/csr_expand.rs` with the test module first (the rest is added in Step 3):

```rust
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
use arrow_schema::{DataType, Field, Schema, SchemaRef};
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

use crate::csr_index::CsrIndex;
use crate::error::{GraphError, Result};
use super::direction::NativeDirection;

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
    let src = src_u64
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("cast to UInt64 yields UInt64Array");

    let mut parent_idx: Vec<u32> = Vec::new();
    let mut neighbors: Vec<u64> = Vec::new();
    for row in 0..input.num_rows() {
        if src.is_null(row) {
            continue;
        }
        for &n in csr.neighbors(src.value(row)) {
            parent_idx.push(row as u32);
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

#[cfg(test)]
mod tests {
    use super::*;
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
```

Add `mod csr_expand;` to `crates/lance-graph/src/lance_native_planner/mod.rs` (after `mod direction;`).

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p lance-graph expand_batch`
Expected: FAIL to compile if `expand_batch` body absent — but here it is present, so this step verifies the tests pass for the pure function. If they pass, proceed; the node/exec below add no new tests.

Run: `cargo test -p lance-graph expand_batch`
Expected: PASS (3 tests).

- [ ] **Step 3: Add the logical node and physical operator**

Append to `crates/lance-graph/src/lance_native_planner/csr_expand.rs` (after `expand_batch`, before `#[cfg(test)]`):

```rust
/// Logical extension node for a single-hop CSR expand.
///
/// Holds only hashable metadata; the physical operator (and its `CsrIndex`) is
/// constructed by `CsrExtensionPlanner` at physical-planning time.
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
        // Order by stable, comparable fields only.
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
```

- [ ] **Step 4: Build and run tests**

Run: `cargo test -p lance-graph csr_expand`
Expected: PASS (3 tests; node/exec compile).

- [ ] **Step 5: Commit**

```bash
git add crates/lance-graph/src/lance_native_planner/
git commit -m "feat(native): CsrExpandNode/Exec and expand_batch core"
```

---

## Task 4: `RowMaterializer` + `take_batch` + `LanceTakeNode` + `LanceTakeExec`

**Files:**
- Create: `crates/lance-graph/src/lance_native_planner/take.rs`
- Modify: `crates/lance-graph/src/lance_native_planner/mod.rs` (add `mod take;`)

- [ ] **Step 1: Write the failing test for the materializer + take function**

Create `crates/lance-graph/src/lance_native_planner/take.rs`:

```rust
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
use arrow_schema::{DataType, Field, Schema, SchemaRef};
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
            fields.push(Field::new(name, col.data_type().clone(), true));
            arrays.push(taken);
        }
        RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).map_err(|e| {
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
    let ids_u64 = cast(input.column(row_id_idx), &DataType::UInt64).map_err(|e| {
        GraphError::ExecutionError {
            message: format!("take: cast row id to u64: {}", e),
            location: snafu::Location::new(file!(), line!(), column!()),
        }
    })?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StringArray;

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
}
```

Add `mod take;` to `crates/lance-graph/src/lance_native_planner/mod.rs`.

- [ ] **Step 2: Run tests to verify they pass**

Run: `cargo test -p lance-graph -- take::tests`
Expected: PASS (3 tests).

- [ ] **Step 3: Add the logical node and physical operator**

Append to `crates/lance-graph/src/lance_native_planner/take.rs` (after `take_batch`, before `#[cfg(test)]`):

```rust
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
```

- [ ] **Step 4: Build and run tests**

Run: `cargo test -p lance-graph -- take::tests`
Expected: PASS (3 tests; node/exec compile).

- [ ] **Step 5: Commit**

```bash
git add crates/lance-graph/src/lance_native_planner/
git commit -m "feat(native): RowMaterializer, LanceTakeNode/Exec, take_batch core"
```

---

## Task 5: `CsrExtensionPlanner` + `CsrQueryPlanner`

Turns the two logical nodes into physical operators, building the `CsrIndex` and `InMemoryMaterializer` from tables registered on the session.

**Files:**
- Create: `crates/lance-graph/src/lance_native_planner/extension_planner.rs`
- Modify: `crates/lance-graph/src/lance_native_planner/mod.rs` (add `mod extension_planner; pub use extension_planner::CsrQueryPlanner;`)

- [ ] **Step 1: Create the extension planner and query planner**

Create `crates/lance-graph/src/lance_native_planner/extension_planner.rs`:

```rust
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
use datafusion::execution::TaskContext;
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNode};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{
    DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner,
};

use super::csr_expand::{CsrExpandExec, CsrExpandNode};
use super::direction::NativeDirection;
use super::take::{CsrExtensionMaterializer, InMemoryMaterializer, LanceTakeExec, LanceTakeNode};
use crate::csr_index::CsrIndexBuilder;

/// Collect a registered table to a single `RecordBatch`.
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
            let input = physical_inputs[0].clone();

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
            let source_id_idx = in_schema
                .index_of(&expand.source_id_column)
                .map_err(|e| DataFusionError::Execution(format!(
                    "CsrExpand: source id column '{}' not found in input: {}",
                    expand.source_id_column, e
                )))?;
            let neighbor_field = Field::new(
                &expand.neighbor_column,
                expand.neighbor_data_type.clone(),
                true,
            );
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
            let input = physical_inputs[0].clone();

            let target = collect_table(session_state, &take.target_table).await?;
            let materializer = Arc::new(InMemoryMaterializer::new(target));

            let in_schema = input.schema();
            let row_id_idx = in_schema
                .index_of(&take.row_id_column)
                .map_err(|e| DataFusionError::Execution(format!(
                    "LanceTake: row id column '{}' not found in input: {}",
                    take.row_id_column, e
                )))?;
            let out_schema = take.schema.inner().clone();

            return Ok(Some(Arc::new(LanceTakeExec::new(
                input,
                materializer as Arc<dyn CsrExtensionMaterializer>,
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

// Silence unused import in some build configs.
#[allow(unused_imports)]
use TaskContext as _TaskContext;
```

> Note: `CsrExtensionMaterializer` referenced above is a type alias for
> `dyn RowMaterializer` used to keep the trait-object cast explicit. Add to
> `take.rs` (top level, after the `RowMaterializer` trait):
> ```rust
> /// Convenience alias for the boxed materializer trait object.
> pub type CsrExtensionMaterializer = dyn RowMaterializer;
> ```
> and change `LanceTakeExec::new` / field type from `Arc<dyn RowMaterializer>` to
> accept `Arc<dyn RowMaterializer>` (the alias is `dyn RowMaterializer`, so
> `Arc<CsrExtensionMaterializer>` == `Arc<dyn RowMaterializer>`; no signature
> change needed — pass `materializer` directly without the `as` cast if simpler).

- [ ] **Step 2: Update mod.rs**

Add to `crates/lance-graph/src/lance_native_planner/mod.rs`:

```rust
mod extension_planner;

pub use extension_planner::CsrQueryPlanner;
```

- [ ] **Step 3: Build**

Run: `cargo build -p lance-graph`
Expected: builds. If the `CsrExtensionMaterializer` alias causes friction, delete the alias and the `as Arc<dyn CsrExtensionMaterializer>` cast and pass `materializer` directly (it is already `Arc<InMemoryMaterializer>`; coercion to `Arc<dyn RowMaterializer>` is automatic at the call site because `LanceTakeExec::new` takes `Arc<dyn RowMaterializer>`).

- [ ] **Step 4: Commit**

```bash
git add crates/lance-graph/src/lance_native_planner/
git commit -m "feat(native): CsrExtensionPlanner + CsrQueryPlanner physical planning"
```

---

## Task 6: `LanceNativePlanner` lowering + fallback

Rewrites `mod.rs` to lower a supported single-hop `Expand` onto the extension nodes and delegate everything else to `DataFusionPlanner`.

**Files:**
- Modify: `crates/lance-graph/src/lance_native_planner/mod.rs`
- Modify: `crates/lance-graph/src/datafusion_planner/mod.rs:20` (`mod expression;` → `pub(crate) mod expression;`)

- [ ] **Step 1: Expose the expression helpers**

In `crates/lance-graph/src/datafusion_planner/mod.rs`, change:

```rust
mod expression;
```
to:
```rust
pub(crate) mod expression;
```

- [ ] **Step 2: Write the failing planner tests**

Replace the body of `crates/lance-graph/src/lance_native_planner/mod.rs` (keep the
`mod`/`pub use` declarations from Tasks 2–5) so the file is:

```rust
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Native physical planner.
//!
//! Lowers a supported single-hop `Expand` onto CSR-backed extension nodes
//! (`CsrExpandNode` + `LanceTakeNode`). Any plan it cannot serve natively is
//! delegated wholesale to `DataFusionPlanner`, so `LanceNative` execution is
//! always correct on valid Cypher — it simply uses joins when CSR cannot help.

mod csr_expand;
mod direction;
mod extension_planner;
mod take;

pub use direction::NativeDirection;
pub use extension_planner::CsrQueryPlanner;

use std::sync::Arc;

use datafusion::common::DFSchema;
use datafusion::logical_expr::{Expr, Extension, LogicalPlan, LogicalPlanBuilder};
use arrow_schema::{Field, Schema};

use crate::ast::RelationshipDirection;
use crate::case_insensitive::qualify_column;
use crate::config::GraphConfig;
use crate::datafusion_planner::expression::{
    to_df_boolean_expr, to_df_value_expr,
};
use crate::datafusion_planner::{
    analysis, DataFusionPlanner, GraphPhysicalPlanner, PlanningContext,
};
use crate::error::{GraphError, Result};
use crate::logical_plan::{LogicalOperator, ProjectionItem, SortItem};

use csr_expand::CsrExpandNode;
use direction::NativeDirection;
use take::LanceTakeNode;

/// Lance-native planner: CSR single-hop expand with DataFusion fallback.
pub struct LanceNativePlanner {
    config: GraphConfig,
    df: DataFusionPlanner,
}

impl LanceNativePlanner {
    pub fn new(config: GraphConfig) -> Self {
        Self {
            df: DataFusionPlanner::new(config.clone()),
            config,
        }
    }

    pub fn with_catalog(
        config: GraphConfig,
        catalog: Arc<dyn lance_graph_catalog::GraphSourceCatalog>,
    ) -> Self {
        Self {
            df: DataFusionPlanner::with_catalog(config.clone(), catalog),
            config,
        }
    }
}

impl GraphPhysicalPlanner for LanceNativePlanner {
    fn plan(&self, logical_plan: &LogicalOperator) -> Result<LogicalPlan> {
        if !can_plan_natively(logical_plan) {
            return self.df.plan(logical_plan);
        }
        let analysis = analysis::analyze(logical_plan)?;
        let mut ctx = PlanningContext::new(&analysis);
        self.build_native(&mut ctx, logical_plan)
    }
}

impl LanceNativePlanner {
    /// Build the native plan for a supported tree. Unary operators above the
    /// single expand are rebuilt on the native child; the expand itself lowers
    /// to `CsrExpandNode` + `LanceTakeNode`.
    fn build_native(
        &self,
        ctx: &mut PlanningContext,
        op: &LogicalOperator,
    ) -> Result<LogicalPlan> {
        match op {
            LogicalOperator::ScanByLabel {
                variable,
                label,
                properties,
            } => self.df.build_scan(ctx, variable, label, properties),

            LogicalOperator::Filter { input, predicate } => {
                let child = self.build_native(ctx, input)?;
                let expr = to_df_boolean_expr(predicate);
                LogicalPlanBuilder::from(child)
                    .filter(expr)
                    .map_err(|e| self.plan_err("filter", e))?
                    .build()
                    .map_err(|e| self.plan_err("filter build", e))
            }

            LogicalOperator::Project { input, projections } => {
                let child = self.build_native(ctx, input)?;
                self.build_project_on(child, projections)
            }

            LogicalOperator::Sort { input, sort_items } => {
                let child = self.build_native(ctx, input)?;
                self.build_sort_on(child, sort_items)
            }

            LogicalOperator::Limit { input, count } => {
                let child = self.build_native(ctx, input)?;
                LogicalPlanBuilder::from(child)
                    .limit(0, Some(*count as usize))
                    .map_err(|e| self.plan_err("limit", e))?
                    .build()
                    .map_err(|e| self.plan_err("limit build", e))
            }

            LogicalOperator::Offset { input, offset } => {
                let child = self.build_native(ctx, input)?;
                LogicalPlanBuilder::from(child)
                    .limit(*offset as usize, None)
                    .map_err(|e| self.plan_err("offset", e))?
                    .build()
                    .map_err(|e| self.plan_err("offset build", e))
            }

            LogicalOperator::Distinct { input } => {
                let child = self.build_native(ctx, input)?;
                LogicalPlanBuilder::from(child)
                    .distinct()
                    .map_err(|e| self.plan_err("distinct", e))?
                    .build()
                    .map_err(|e| self.plan_err("distinct build", e))
            }

            LogicalOperator::Expand {
                input,
                source_variable,
                target_variable,
                target_label,
                relationship_types,
                direction,
                ..
            } => self.build_expand_native(
                ctx,
                input,
                source_variable,
                target_variable,
                target_label,
                relationship_types,
                direction,
            ),

            // Unsupported here would have been rejected by can_plan_natively.
            other => Err(GraphError::PlanError {
                message: format!("native planner reached unsupported operator: {:?}", other),
                location: snafu::Location::new(file!(), line!(), column!()),
            }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_expand_native(
        &self,
        ctx: &mut PlanningContext,
        input: &LogicalOperator,
        source_variable: &str,
        target_variable: &str,
        target_label: &str,
        relationship_types: &[String],
        direction: &RelationshipDirection,
    ) -> Result<LogicalPlan> {
        let source_plan = self.build_native(ctx, input)?;

        let rel_type = &relationship_types[0];
        let rel_map = self.config.get_relationship_mapping(rel_type).ok_or_else(|| {
            GraphError::ConfigError {
                message: format!("No relationship mapping for '{}'", rel_type),
                location: snafu::Location::new(file!(), line!(), column!()),
            }
        })?;

        let src_label = ctx
            .analysis
            .var_to_label
            .get(source_variable)
            .ok_or_else(|| GraphError::PlanError {
                message: format!("No label for source variable '{}'", source_variable),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        let src_node = self.config.get_node_mapping(src_label).ok_or_else(|| {
            GraphError::ConfigError {
                message: format!("No node mapping for label '{}'", src_label),
                location: snafu::Location::new(file!(), line!(), column!()),
            }
        })?;
        let tgt_node = self.config.get_node_mapping(target_label).ok_or_else(|| {
            GraphError::ConfigError {
                message: format!("No node mapping for label '{}'", target_label),
                location: snafu::Location::new(file!(), line!(), column!()),
            }
        })?;

        let catalog = self.df.catalog_ref().ok_or_else(|| GraphError::ConfigError {
            message: "LanceNativePlanner requires a catalog for native expand".to_string(),
            location: snafu::Location::new(file!(), line!(), column!()),
        })?;
        let tgt_source = catalog.node_source(target_label).ok_or_else(|| {
            GraphError::ConfigError {
                message: format!("No table source for target label '{}'", target_label),
                location: snafu::Location::new(file!(), line!(), column!()),
            }
        })?;
        let tgt_arrow = tgt_source.schema();

        let direction = match direction {
            RelationshipDirection::Outgoing => NativeDirection::Outgoing,
            RelationshipDirection::Incoming => NativeDirection::Incoming,
            RelationshipDirection::Undirected => {
                return Err(GraphError::PlanError {
                    message: "undirected expand is not natively supported".to_string(),
                    location: snafu::Location::new(file!(), line!(), column!()),
                });
            }
        };

        let source_id_column = qualify_column(source_variable, &src_node.id_field);
        let neighbor_column = qualify_column(target_variable, &tgt_node.id_field);
        let neighbor_data_type = tgt_arrow
            .field_with_name(&tgt_node.id_field.to_lowercase())
            .map_err(|e| GraphError::ConfigError {
                message: format!(
                    "target id field '{}' not found in '{}': {}",
                    tgt_node.id_field, target_label, e
                ),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?
            .data_type()
            .clone();

        // CsrExpandNode output schema = source schema + neighbor column.
        let src_arrow = source_plan.schema().inner();
        let mut expand_fields: Vec<Field> =
            src_arrow.fields().iter().map(|f| f.as_ref().clone()).collect();
        expand_fields.push(Field::new(
            &neighbor_column,
            neighbor_data_type.clone(),
            true,
        ));
        let expand_arrow = Schema::new(expand_fields);
        let expand_schema = Arc::new(
            DFSchema::try_from(expand_arrow).map_err(|e| self.plan_err("expand schema", e))?,
        );

        let expand_node = CsrExpandNode {
            input: source_plan,
            rel_type: rel_type.to_lowercase(),
            src_field: rel_map.source_id_field.to_lowercase(),
            dst_field: rel_map.target_id_field.to_lowercase(),
            direction,
            source_id_column,
            neighbor_column: neighbor_column.clone(),
            neighbor_data_type,
            schema: expand_schema.clone(),
        };
        let expand_plan = LogicalPlan::Extension(Extension {
            node: Arc::new(expand_node),
        });

        // LanceTakeNode: materialize all target columns except the id field.
        let id_lower = tgt_node.id_field.to_lowercase();
        let take_cols: Vec<String> = tgt_arrow
            .fields()
            .iter()
            .map(|f| f.name().to_lowercase())
            .filter(|n| n != &id_lower)
            .collect();

        let mut take_fields: Vec<Field> = expand_arrow_fields(&expand_plan);
        for raw in &take_cols {
            let f = tgt_arrow
                .field_with_name(raw)
                .map_err(|e| self.plan_err("target field", e))?;
            take_fields.push(Field::new(
                qualify_column(target_variable, raw),
                f.data_type().clone(),
                true,
            ));
        }
        let take_arrow = Schema::new(take_fields);
        let take_schema =
            Arc::new(DFSchema::try_from(take_arrow).map_err(|e| self.plan_err("take schema", e))?);

        let take_node = LanceTakeNode {
            input: expand_plan,
            target_table: target_label.to_lowercase(),
            row_id_column: neighbor_column,
            take_cols,
            schema: take_schema,
        };
        Ok(LogicalPlan::Extension(Extension {
            node: Arc::new(take_node),
        }))
    }

    fn build_project_on(
        &self,
        input: LogicalPlan,
        projections: &[ProjectionItem],
    ) -> Result<LogicalPlan> {
        // Delegate to DataFusionPlanner's project-on-plan helpers (handle
        // aggregates + Cypher dot-notation aliasing identically to the join path).
        let has_agg = projections.iter().any(|p| {
            crate::datafusion_planner::expression::contains_aggregate(&p.expression)
        });
        if has_agg {
            self.df.build_project_with_aggregates(input, projections)
        } else {
            self.df.build_simple_project(input, projections)
        }
    }

    fn build_sort_on(&self, input: LogicalPlan, sort_items: &[SortItem]) -> Result<LogicalPlan> {
        use datafusion::logical_expr::SortExpr;
        let sort_exprs: Vec<SortExpr> = sort_items
            .iter()
            .map(|item| {
                let expr = to_df_value_expr(&item.expression);
                let asc = matches!(item.direction, crate::ast::SortDirection::Ascending);
                SortExpr {
                    expr,
                    asc,
                    nulls_first: true,
                }
            })
            .collect();
        LogicalPlanBuilder::from(input)
            .sort(sort_exprs)
            .map_err(|e| self.plan_err("sort", e))?
            .build()
            .map_err(|e| self.plan_err("sort build", e))
    }

    fn plan_err<E: std::fmt::Display>(&self, what: &str, e: E) -> GraphError {
        GraphError::PlanError {
            message: format!("native {}: {}", what, e),
            location: snafu::Location::new(file!(), line!(), column!()),
        }
    }
}

/// Extract the arrow fields of a `CsrExpandNode` extension plan as owned `Field`s.
fn expand_arrow_fields(plan: &LogicalPlan) -> Vec<Field> {
    plan.schema()
        .inner()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect()
}

/// True iff the plan is a single-hop expand the native planner can serve.
fn can_plan_natively(op: &LogicalOperator) -> bool {
    let mut expands = 0usize;
    if !walk_supported(op, &mut expands) {
        return false;
    }
    expands == 1
}

fn walk_supported(op: &LogicalOperator, expands: &mut usize) -> bool {
    match op {
        LogicalOperator::ScanByLabel { .. } => true,
        LogicalOperator::Filter { input, .. }
        | LogicalOperator::Project { input, .. }
        | LogicalOperator::Sort { input, .. }
        | LogicalOperator::Limit { input, .. }
        | LogicalOperator::Offset { input, .. }
        | LogicalOperator::Distinct { input } => walk_supported(input, expands),
        LogicalOperator::Expand {
            input,
            relationship_types,
            direction,
            ..
        } => {
            *expands += 1;
            if relationship_types.len() != 1 {
                return false;
            }
            if matches!(direction, RelationshipDirection::Undirected) {
                return false;
            }
            walk_supported(input, expands)
        }
        // Not supported natively in Phase 2.
        LogicalOperator::VariableLengthExpand { .. }
        | LogicalOperator::Join { .. }
        | LogicalOperator::Unwind { .. } => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datafusion_planner::test_fixtures::{make_catalog, person_knows_config, person_scan};
    use crate::logical_plan::{LogicalOperator, ProjectionItem};
    use crate::ast::{PropertyRef, ValueExpression};

    fn knows_expand(direction: RelationshipDirection) -> LogicalOperator {
        LogicalOperator::Expand {
            input: Box::new(person_scan("a")),
            source_variable: "a".to_string(),
            target_variable: "b".to_string(),
            target_label: "Person".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction,
            relationship_variable: None,
            properties: Default::default(),
            target_properties: Default::default(),
        }
    }

    #[test]
    fn test_can_plan_natively_single_hop() {
        let plan = LogicalOperator::Project {
            input: Box::new(knows_expand(RelationshipDirection::Outgoing)),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef::new("b", "name")),
                alias: None,
            }],
        };
        assert!(can_plan_natively(&plan));
    }

    #[test]
    fn test_cannot_plan_undirected_or_multitype() {
        assert!(!can_plan_natively(&knows_expand(RelationshipDirection::Undirected)));
        let mut multi = knows_expand(RelationshipDirection::Outgoing);
        if let LogicalOperator::Expand {
            relationship_types, ..
        } = &mut multi
        {
            relationship_types.push("LIKES".to_string());
        }
        assert!(!can_plan_natively(&multi));
    }

    #[test]
    fn test_cannot_plan_zero_expands() {
        assert!(!can_plan_natively(&person_scan("a")));
    }

    #[test]
    fn test_native_plan_contains_extension_nodes() {
        let plan = LogicalOperator::Project {
            input: Box::new(knows_expand(RelationshipDirection::Outgoing)),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef::new("b", "name")),
                alias: None,
            }],
        };
        let planner = LanceNativePlanner::with_catalog(person_knows_config(), make_catalog());
        let df_plan = planner.plan(&plan).unwrap();
        let s = format!("{:?}", df_plan);
        assert!(s.contains("CsrExpand"), "missing CsrExpand: {}", s);
        assert!(s.contains("LanceTake"), "missing LanceTake: {}", s);
    }

    #[test]
    fn test_unsupported_falls_back_to_join() {
        // Variable-length expand must fall back to the DataFusion join path.
        let vlexpand = LogicalOperator::VariableLengthExpand {
            input: Box::new(person_scan("a")),
            source_variable: "a".into(),
            target_variable: "b".into(),
            relationship_types: vec!["KNOWS".into()],
            direction: RelationshipDirection::Outgoing,
            relationship_variable: None,
            min_length: Some(1),
            max_length: Some(2),
            target_properties: Default::default(),
        };
        let planner = LanceNativePlanner::with_catalog(person_knows_config(), make_catalog());
        let df_plan = planner.plan(&vlexpand).unwrap();
        let s = format!("{:?}", df_plan);
        assert!(!s.contains("CsrExpand"), "should not be native: {}", s);
    }
}
```

This task also requires two `pub(crate)` accessors on `DataFusionPlanner`. Add them in `crates/lance-graph/src/datafusion_planner/mod.rs` inside `impl DataFusionPlanner`:

```rust
    /// Access the catalog, if any (used by the native planner).
    pub(crate) fn catalog_ref(
        &self,
    ) -> Option<&Arc<dyn GraphSourceCatalog>> {
        self.catalog.as_ref()
    }
```

(`build_simple_project`, `build_project_with_aggregates`, and `build_scan` are already `pub(crate)`. `contains_aggregate` lives in `expression`, now `pub(crate)`.)

- [ ] **Step 3: Run tests to verify they fail, then pass**

Run: `cargo test -p lance-graph -- lance_native_planner::tests`
Expected: compiles and PASSES (5 tests). If `build_project_with_aggregates` is not `pub(crate)`, change its visibility in `crates/lance-graph/src/datafusion_planner/builder/aggregate_ops.rs` from `fn` to `pub(crate) fn` (verify; `build_simple_project` is already `pub(crate)`).

- [ ] **Step 4: Commit**

```bash
git add crates/lance-graph/src/lance_native_planner/ crates/lance-graph/src/datafusion_planner/
git commit -m "feat(native): LanceNativePlanner single-hop lowering with DataFusion fallback"
```

---

## Task 7: Wire `ExecutionStrategy::LanceNative` and end-to-end parity tests

**Files:**
- Modify: `crates/lance-graph/src/query.rs` (the two `LanceNative` arms; native context + planner)
- Create: `crates/lance-graph/tests/test_lance_native_expand.rs`

- [ ] **Step 1: Write the failing end-to-end test**

Create `crates/lance-graph/tests/test_lance_native_expand.rs`:

```rust
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! End-to-end parity tests: native CSR expand vs DataFusion join path.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use lance_graph::config::GraphConfig;
use lance_graph::query::{CypherQuery, ExecutionStrategy};

fn person_batch() -> RecordBatch {
    // Dense ids 0..4 (row id == id_field value).
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from(vec![0u64, 1, 2, 3])),
            Arc::new(StringArray::from(vec!["alice", "bob", "carol", "dave"])),
            Arc::new(Int64Array::from(vec![30i64, 40, 25, 50])),
        ],
    )
    .unwrap()
}

fn knows_batch() -> RecordBatch {
    // 0->1, 0->2, 1->3, 2->3
    let schema = Arc::new(Schema::new(vec![
        Field::new("src_id", DataType::UInt64, false),
        Field::new("dst_id", DataType::UInt64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from(vec![0u64, 0, 1, 2])),
            Arc::new(UInt64Array::from(vec![1u64, 2, 3, 3])),
        ],
    )
    .unwrap()
}

fn config() -> GraphConfig {
    GraphConfig::builder()
        .with_node_label("Person", "id")
        .with_relationship("KNOWS", "src_id", "dst_id")
        .build()
        .unwrap()
}

fn datasets() -> HashMap<String, RecordBatch> {
    let mut d = HashMap::new();
    d.insert("Person".to_string(), person_batch());
    d.insert("KNOWS".to_string(), knows_batch());
    d
}

/// Collect (a.name, b.name) rows as a sorted Vec for order-independent compare.
fn name_pairs(batch: &RecordBatch) -> Vec<(String, String)> {
    let cols: Vec<&StringArray> = (0..batch.num_columns())
        .map(|i| batch.column(i).as_any().downcast_ref::<StringArray>().unwrap())
        .collect();
    let mut rows: Vec<(String, String)> = (0..batch.num_rows())
        .map(|r| (cols[0].value(r).to_string(), cols[1].value(r).to_string()))
        .collect();
    rows.sort();
    rows
}

#[tokio::test]
async fn test_native_expand_matches_datafusion_names() {
    let q = "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name";
    let query = CypherQuery::new(q).unwrap().with_config(config());

    let native = query
        .execute(datasets(), Some(ExecutionStrategy::LanceNative))
        .await
        .unwrap();
    let df = query
        .execute(datasets(), Some(ExecutionStrategy::DataFusion))
        .await
        .unwrap();

    let expected = vec![
        ("alice".to_string(), "bob".to_string()),
        ("alice".to_string(), "carol".to_string()),
        ("bob".to_string(), "dave".to_string()),
        ("carol".to_string(), "dave".to_string()),
    ];
    assert_eq!(name_pairs(&native), expected);
    assert_eq!(name_pairs(&native), name_pairs(&df));
}

#[tokio::test]
async fn test_native_expand_with_target_filter() {
    let q = "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE b.age > 30 RETURN a.name, b.name";
    let query = CypherQuery::new(q).unwrap().with_config(config());

    let native = query
        .execute(datasets(), Some(ExecutionStrategy::LanceNative))
        .await
        .unwrap();
    let df = query
        .execute(datasets(), Some(ExecutionStrategy::DataFusion))
        .await
        .unwrap();
    assert_eq!(name_pairs(&native), name_pairs(&df));
    // bob(40) and dave(50) qualify as targets: (alice,bob),(bob,dave),(carol,dave)
    assert_eq!(
        name_pairs(&native),
        vec![
            ("alice".to_string(), "bob".to_string()),
            ("bob".to_string(), "dave".to_string()),
            ("carol".to_string(), "dave".to_string()),
        ]
    );
}

#[tokio::test]
async fn test_native_expand_incoming_matches_datafusion() {
    let q = "MATCH (a:Person)<-[:KNOWS]-(b:Person) RETURN a.name, b.name";
    let query = CypherQuery::new(q).unwrap().with_config(config());
    let native = query
        .execute(datasets(), Some(ExecutionStrategy::LanceNative))
        .await
        .unwrap();
    let df = query
        .execute(datasets(), Some(ExecutionStrategy::DataFusion))
        .await
        .unwrap();
    assert_eq!(name_pairs(&native), name_pairs(&df));
}

#[tokio::test]
async fn test_native_varlength_falls_back_and_matches() {
    // Variable-length path is unsupported natively; LanceNative must fall back
    // and produce the same result as DataFusion.
    let q = "MATCH (a:Person)-[:KNOWS*1..2]->(b:Person) RETURN a.name, b.name";
    let query = CypherQuery::new(q).unwrap().with_config(config());
    let native = query
        .execute(datasets(), Some(ExecutionStrategy::LanceNative))
        .await
        .unwrap();
    let df = query
        .execute(datasets(), Some(ExecutionStrategy::DataFusion))
        .await
        .unwrap();
    assert_eq!(name_pairs(&native), name_pairs(&df));
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p lance-graph --test test_lance_native_expand`
Expected: FAIL — `LanceNative` currently returns `UnsupportedFeature`.

- [ ] **Step 3: Wire the native execution path in query.rs**

In `crates/lance-graph/src/query.rs`:

(a) Change the private context builder to optionally install the query planner.
Replace the signature and `SessionContext::new()` line of
`build_catalog_and_context_from_datasets` (around line 608 and 627):

```rust
    async fn build_catalog_and_context_from_datasets(
        &self,
        datasets: HashMap<String, arrow::record_batch::RecordBatch>,
        native: bool,
    ) -> Result<(
        lance_graph_catalog::InMemoryCatalog,
        datafusion::execution::context::SessionContext,
    )> {
        use datafusion::datasource::{DefaultTableSource, MemTable};
        use datafusion::execution::context::SessionContext;
        use lance_graph_catalog::InMemoryCatalog;
        use std::sync::Arc;

        if datasets.is_empty() {
            return Err(GraphError::ConfigError {
                message: "No input datasets provided".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            });
        }

        // Create session context (with the CSR query planner when native).
        let ctx = if native {
            use datafusion::execution::session_state::SessionStateBuilder;
            let state = SessionStateBuilder::new()
                .with_default_features()
                .with_query_planner(Arc::new(crate::lance_native_planner::CsrQueryPlanner::new()))
                .build();
            SessionContext::new_with_state(state)
        } else {
            SessionContext::new()
        };
        let mut catalog = InMemoryCatalog::new();
```

(Leave the rest of the method body — the dataset registration loop and the
`Ok((catalog, ctx))` return — unchanged.)

(b) Update the three existing callers to pass `false`:
- `execute_datafusion` (around line 599): `self.build_catalog_and_context_from_datasets(datasets, false)`
- `explain` (around line 329): `self.build_catalog_and_context_from_datasets(datasets, false)`
- `to_sql` (around line 369): `self.build_catalog_and_context_from_datasets(datasets, false)`

(c) Add the `create_logical_plans_native` helper next to `create_logical_plans`
(after line 841):

```rust
    fn create_logical_plans_native(
        &self,
        catalog: std::sync::Arc<dyn lance_graph_catalog::GraphSourceCatalog>,
    ) -> Result<datafusion::logical_expr::LogicalPlan> {
        use crate::datafusion_planner::GraphPhysicalPlanner;
        use crate::lance_native_planner::LanceNativePlanner;
        use crate::semantic::SemanticAnalyzer;

        let config = self.require_config()?;

        let mut analyzer = SemanticAnalyzer::new(config.clone());
        let semantic = analyzer.analyze(&self.ast, &self.parameters)?;
        if !semantic.errors.is_empty() {
            return Err(GraphError::PlanError {
                message: format!("Semantic analysis failed:\n{}", semantic.errors.join("\n")),
                location: snafu::Location::new(file!(), line!(), column!()),
            });
        }

        let mut logical_planner = crate::logical_plan::LogicalPlanner::new(config);
        let logical_plan = logical_planner.plan(&semantic.ast)?;

        let native = LanceNativePlanner::with_catalog(config.clone(), catalog);
        native.plan(&logical_plan)
    }
```

Make sure `LanceNativePlanner` is re-exported: in
`crates/lance-graph/src/lance_native_planner/mod.rs` it is already `pub struct`,
and `crate::lance_native_planner::LanceNativePlanner` is accessible since
`lib.rs` has `pub mod lance_native_planner;`.

(d) Add the native execute method (after `execute_datafusion`, ~line 604):

```rust
    async fn execute_lance_native(
        &self,
        datasets: HashMap<String, arrow::record_batch::RecordBatch>,
    ) -> Result<arrow::record_batch::RecordBatch> {
        use arrow::compute::concat_batches;
        use std::sync::Arc;

        let (catalog, ctx) = self
            .build_catalog_and_context_from_datasets(datasets, true)
            .await?;

        let df_logical_plan = self.create_logical_plans_native(Arc::new(catalog))?;

        let df = ctx
            .execute_logical_plan(df_logical_plan)
            .await
            .map_err(|e| GraphError::ExecutionError {
                message: format!("Failed to execute native plan: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        let result_schema = df.schema().inner().clone();
        let batches = df.collect().await.map_err(|e| GraphError::ExecutionError {
            message: format!("Failed to collect native results: {}", e),
            location: snafu::Location::new(file!(), line!(), column!()),
        })?;
        concat_batches(&result_schema, &batches).map_err(|e| GraphError::ExecutionError {
            message: format!("Failed to concat native results: {}", e),
            location: snafu::Location::new(file!(), line!(), column!()),
        })
    }
```

(e) Route the in-memory `execute` arm (around line 234-240): replace the
`ExecutionStrategy::LanceNative => Err(...)` arm with:

```rust
            ExecutionStrategy::LanceNative => self.execute_lance_native(datasets).await,
```

(Leave the `execute_with_namespace_internal` `LanceNative` arm returning
`UnsupportedFeature` — namespace native execution is Phase 4.)

- [ ] **Step 4: Run the end-to-end tests**

Run: `cargo test -p lance-graph --test test_lance_native_expand`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the full crate test suite**

Run: `cargo test -p lance-graph`
Expected: PASS (all prior tests + new ones). Then `cargo clippy -p lance-graph --all-targets` — fix any warnings (e.g. unused imports).

- [ ] **Step 6: Commit**

```bash
git add crates/lance-graph/src/query.rs crates/lance-graph/tests/test_lance_native_expand.rs
git commit -m "feat(native): wire LanceNative execution strategy + e2e parity tests"
```

---

## Self-Review

**Spec coverage:**
- Custom DataFusion `ExecutionPlan` execution model → Tasks 3, 4, 5 (nodes, execs, planner). ✓
- Dense-ROWID id model (vertex id == row id; source `a__<id_field>`, neighbor `b__<id_field>`) → Task 6 `build_expand_native`. ✓
- Output via `take()`, materialize all target columns except id → Task 4 + Task 6 `take_cols`. ✓
- Separate `LanceTakeExec` + `RowMaterializer` (arrow take now, Lance later) → Task 4. ✓
- Planner reuse + override only `Expand`; fallback for var-length/multi-hop/multi-type/undirected/Join/Unwind → Task 6 `can_plan_natively`/`build_native`. ✓
- CSR built at physical-planning time with real `rel_map` columns (reversed for incoming) → Task 1 + Task 5. ✓
- query.rs wiring: in-memory `execute` native path wired; namespace native stays `UnsupportedFeature` → Task 7. ✓
- Tests: operator units (Tasks 3,4), planner native/fallback (Task 6), e2e parity incl. filter, incoming, fallback (Task 7). ✓

**Placeholder scan:** No TBD/TODO; all steps contain concrete code and commands.

**Type consistency:** `expand_batch`/`take_batch` signatures match their `execute` call sites; `CsrExpandNode`/`LanceTakeNode` field names match construction in Task 6 and downcast use in Task 5; `add_edges_from_batch_with_columns` signature matches Task 5 usage; `CsrQueryPlanner::new` / `with_catalog` / `build_simple_project` / `build_project_with_aggregates` / `catalog_ref` names consistent across tasks.

**Risk notes for the implementer:**
- DataFusion 50.3 API drift: if `PlanProperties::new`, `EmissionType`/`Boundedness` import paths, or `ExtensionPlanner`/`QueryPlanner` async signatures differ, consult `cargo doc -p datafusion --open` for the exact paths; the field/method *names* used here are stable across 49–50.
- If `build_project_with_aggregates` is private, widen it to `pub(crate)` (Task 6, Step 3).
- `with_default_features()` on `SessionStateBuilder` is required so standard scalar/aggregate functions resolve in the native context.
