# Phase 2: CSR-backed native single-hop Expand operator

Issue: [lance-format/lance-graph#159](https://github.com/lance-format/lance-graph/issues/159)
Status: Approved design (2026-06-22)
Builds on: Phase 1 — `CsrIndex` (PR #160, commit `c7e4f18`)

## Problem

Today every Cypher traversal is lowered to DataFusion SQL joins. A single-hop
`(a)-[:KNOWS]->(b)` becomes a relationship scan plus two inner joins. Phase 1
added an in-memory `CsrIndex` (dense `u64` adjacency with `neighbors()`, `bfs()`,
`shortest_path()`, Arrow (de)serialization) but nothing consumes it —
`LanceNativePlanner` is a placeholder that returns `EmptyRelation`, and
`ExecutionStrategy::LanceNative` errors with "not yet implemented".

Phase 2 wires CSR into a real native execution path for **single-hop `Expand`**,
replacing the join with a direct neighbor lookup followed by a `take()`.

## Foundational decisions (locked)

1. **Execution model — custom DataFusion `ExecutionPlan` (DuckPGQ-style).**
   A logical extension node lowers to a streaming physical operator that does
   neighbor lookups at execution time and composes with the rest of the
   DataFusion pipeline. (Alternatives considered: DataFusion-materialized
   MemTable; pure-native bypassing DataFusion. Rejected — less faithful / less
   composable.)

2. **ID mapping — dense ROWID model.** The CSR vertex id *is* the node's row id.
   `csr.neighbors(src_rowid) -> dst_rowid`s; target properties come from
   `take(dst_rowids)`. This mirrors how every Lance index works (key → row ids →
   `take()` to materialize) and reuses Lance's addressing instead of inventing a
   dictionary. For Phase 2 (in-memory, single fragment) "row id == dense row
   offset"; this generalizes to **Lance stable row ids** in Phase 4.

3. **Output contract — properties materialized via `take()`.** `Expand` does not
   fall back when target properties are projected. The neighbor row ids feed a
   `take()` that materializes the referenced target columns, so `RETURN b.name`
   runs fully native.

4. **`take()` placement — a separate `LanceTakeExec` operator.**
   `CsrExpandExec` does *topology only* (row id → neighbor row ids);
   `LanceTakeExec` does *materialization only* via a `RowMaterializer`
   abstraction. Single-purpose operators, reused by Phase 3 (multi-hop) and
   Phase 5 (hybrid vector). Mirrors Lance's own scan+take shape.

## Architecture & modules

Promote `crates/lance-graph/src/lance_native_planner.rs` to a module directory:

| File | Responsibility |
|---|---|
| `lance_native_planner/mod.rs` | `LanceNativePlanner` — native-vs-fallback decision; lowering that overrides only `Expand` |
| `lance_native_planner/csr_expand.rs` | `CsrExpandNode` (logical extension) + `CsrExpandExec` (physical): row id → neighbor row ids |
| `lance_native_planner/take.rs` | `LanceTakeNode` + `LanceTakeExec` + `RowMaterializer` trait + `InMemoryMaterializer` |
| `lance_native_planner/extension_planner.rs` | `ExtensionPlanner` mapping logical nodes → physical execs (builds CSR + materializer at plan time); `CsrQueryPlanner` wrapping `DefaultPhysicalPlanner` |

## Data flow

For `MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE b.age > 30 RETURN a.name, b.name`:

```
TableScan(person AS a)        a__id(=rowid), a__name, a__age        [reuse existing scan_ops]
 └─ CsrExpandExec             + b__id  (one row per neighbor; a__* carried through, b__id = neighbor row id)
     └─ LanceTakeExec         + b__name, b__age  (take all other target cols by offset)
         └─ Filter(b__age>30) [DataFusion native physical op]
             └─ Project(...)  [DataFusion native physical op]
```

`CsrExpandExec` and `LanceTakeExec` produce a normal `RecordBatch` stream with a
correct qualified schema (`{var}__{col}`), so all operators above (Filter,
Project, Sort, Limit, Distinct, Offset) are ordinary DataFusion physical
operators.

## The planner: reuse + override

`LanceNativePlanner` holds a `DataFusionPlanner` and **overrides only `Expand`
lowering**; every other `LogicalOperator` delegates to the existing crate-internal
builders (`pub(crate)`), so scans/projects/filters/limits behave identically to
the DataFusion path.

- **`Expand` →** `CsrExpandNode` (appends `b__<target_id_field>` — the neighbor
  row id, which under the dense model equals the target's id value) wrapped by
  `LanceTakeNode` (appends **all remaining target node columns**, qualified
  `b__<col>`).
- **Materialize all target columns, not just referenced ones.** This is exactly
  what the DataFusion target scan does (`build_qualified_target_scan` projects
  every target field), so the native output schema matches the join path's column
  set and every `b__col` is available to downstream Project/Filter/Sort. It also
  removes any need to walk expressions collecting `b.*` references. Projection
  pushdown (materializing only needed columns) is a later optimization.
- Source row-id column = `a__<source_id_field>` (dense-ROWID assumption; becomes
  `a___rowid` in Phase 4).
- The neighbor column emitted by `CsrExpandExec` is named `b__<target_id_field>`
  (e.g. `b__id`), so `RETURN b.id` is served directly and `LanceTakeExec` reuses
  that column as the row-id input. `LanceTakeExec` therefore materializes the
  target columns *other than* the id field.
- Output column naming stays `{var}__{col}` to match the DataFusion path exactly.

### Native-vs-fallback rule

A query is served natively **iff** its plan contains exactly one single-hop
`Expand` with a single relationship type and direction Outgoing or Incoming.
**Otherwise the entire query delegates to `DataFusionPlanner`.** Falls back for:
`VariableLengthExpand`, more than one `Expand` (multi-hop), `Expand` with more
than one relationship type, `Undirected` direction, `Join`, `Unwind`.

Consequence: `ExecutionStrategy::LanceNative` is always correct on valid Cypher —
it never errors, it just uses joins when CSR cannot serve the shape.

## CSR construction & RowMaterializer

- `ExtensionPlanner::plan_extension` (async) looks up the edge table and target
  node table from `SessionState`, collects them, and builds:
  - `Arc<CsrIndex>` via the Phase 1 builder. **Add
    `CsrIndexBuilder::add_edges_from_batch_with_columns(batch, src_col, dst_col)`**
    so it uses the real `RelationshipMapping` field names
    (`source_id_field`/`target_id_field`) instead of the hardcoded
    `src_id`/`dst_id`. Outgoing builds `(src→dst)`; Incoming reverses to `(dst→src)`.
  - `InMemoryMaterializer { batch }` over the collected target node table.
- `RowMaterializer` trait:
  ```rust
  trait RowMaterializer: Send + Sync {
      fn schema(&self) -> SchemaRef;
      fn take(&self, row_ids: &UInt64Array, columns: &[String]) -> Result<RecordBatch>;
  }
  ```
  In-memory impl = `arrow::compute::take` by offset — O(1) random access, the
  concrete reason CSR beats the hash join. The take node materializes **all**
  target columns except the id field. Phase 4 adds a `LanceDatasetMaterializer`
  backed by `LanceDataset::take`.
- CSR is built once per physical planning (rebuild per query is acceptable for
  Phase 2; Phase 4 persists CSR as a Lance dataset).
- `num_vertices` is inferred from the edge data (Phase 1 default = max id + 1);
  source ids beyond range yield empty neighbors (no error), matching Phase 1.

## query.rs wiring

- Build the native `SessionContext` via
  `SessionStateBuilder::new().with_default_features().with_query_planner(Arc::new(CsrQueryPlanner)).build()`.
  `CsrQueryPlanner` wraps `DefaultPhysicalPlanner` with the `ExtensionPlanner`.
- `create_logical_plans` gains a planner choice (DataFusion vs LanceNative).
- The **in-memory `execute(datasets, Some(LanceNative))` path is fully wired and
  tested.** It builds the native context, plans with `LanceNativePlanner`, and
  executes.
- The **namespace native path stays `UnsupportedFeature` for Phase 2** — it needs
  the Lance-dataset materializer, which is Phase 4 (persistence).

## Error handling

Reuse `GraphError` with `snafu::Location`. Missing or wrong-typed edge columns at
CSR-build time and take failures surface as `PlanError` / `ExecutionError`. Source
row ids beyond CSR range produce empty neighbors rather than an error.

## Testing

- **Operator unit tests**
  - `CsrExpandExec`: outgoing, incoming, vertex with no neighbors, source id out
    of range, input split across multiple batches, carry-through of source columns.
  - `LanceTakeExec`: take correctness by offset, column subset selection, empty
    input, row-id column dropped from output.
  - `CsrIndexBuilder::add_edges_from_batch_with_columns`: custom column names,
    reversed (incoming) build.
- **Planner tests**
  - Supported shape lowers to a plan containing `CsrExpandExec` + `LanceTakeExec`.
  - Each unsupported shape (var-length, two-hop, undirected, multi-type) falls
    back to a join plan (`Join` present, no extension nodes).
- **End-to-end parity** (`execute(datasets, …)` LanceNative vs DataFusion return
  identical results)
  - `MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name`
  - same with `WHERE b.age > 30`
  - same with `LIMIT`
  - incoming direction `(a)<-[:KNOWS]-(b)`

## Out of scope (later phases)

- Multi-hop / `VariableLengthExpand`, BFS/DFS/shortest-path operators (Phase 3).
- Persisting CSR as Lance datasets, incremental updates, Lance stable row ids,
  `LanceDatasetMaterializer`, namespace native path (Phase 4).
- Hybrid CSR + vector search (Phase 5).
