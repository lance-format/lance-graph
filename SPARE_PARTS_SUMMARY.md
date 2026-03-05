# Lance-Graph Spare Parts: Summary, Open Ends & Vision

## Context

Lance-graph was built as an architecture review repo for graph query processing
over Lance columnar storage. During review, we determined that lance-graph's
**engine** violates ladybug-rs ground truths (rustynum mandatory, BindSpace
zero-copy mandatory, existing lancedb+datafusion deps). The engine stays in
ladybug-rs. What we take from lance-graph are the **bumpers and rims** —
hardened entry and exit points.

---

## What We Take (Spare Parts)

### Bumpers (Hardened Input Protection)

| File | Lines | Role |
|------|-------|------|
| `parser.rs` | ~1,800 | nom-combinator Cypher parser. Rejects malformed queries before they touch SPO. Handles MATCH, WHERE, RETURN, WITH, ORDER BY, LIMIT, SKIP, variable-length paths, property filters, vector distance/similarity, aggregates, UNWIND. |
| `ast.rs` | 543 | Pure data types (serde-serializable). CypherQuery, NodePattern, RelationshipPattern, PathPattern, BooleanExpression, ValueExpression. Zero external deps beyond serde. |
| `error.rs` | 234 | `#[track_caller]` zero-cost error macros (`plan_err!`, `config_err!`, `exec_err!`). snafu-based with Location tracking. Compile-time call-site capture, 0 runtime cycles. |

**Adaptation required**: Strip `DataFusion`, `LanceCore`, `Arrow` error variants.
Keep `ParseError`, `PlanError`, `ConfigError`, `ExecutionError`, `UnsupportedFeature`,
`InvalidPattern`. The parser becomes a standalone hardened gate.

### Rims (Output Alignment)

The row/column join logic from lance-graph's DataFusion planner provides ground
truth verification AFTER SPO thinking completes. Key patterns:

- **Qualified column naming**: `variable__property` internally, `variable.property` at output
- **Join key construction**: direction-aware (Outgoing/Incoming/Undirected)
- **Variable reuse detection**: filter instead of redundant join
- **Schema preservation**: empty results still carry correct column schema

**Adaptation required**: ladybug-rs and rustynum already depend on datafusion.
Use their existing datafusion dep, don't duplicate. The n8n-rs `n8n-arrow` crate
already has `RecordBatch` <-> row conversion (`convert.rs`, `schema.rs`). Reuse that
pattern for neo4j-rs compatibility.

---

## What Stays in Ladybug-rs (NOT from lance-graph)

| Module | Owner | Role |
|--------|-------|------|
| `sparse.rs` | ladybug-rs | BITMAP_WORDS=4, SparseContainer, dense<->sparse, AxisDescriptors |
| `builder.rs` | ladybug-rs | SpoBuilder with BUNDLE/BIND, verb permutation, ContainerGeometry::Spo=6 |
| `store.rs` | ladybug-rs | Three-axis content-addressable graph, scent-pruned projections (SxP2O, PxO2S, SxO2P) |
| `scent.rs` | ladybug-rs | NibbleScent 48-byte histogram, L1 prefilter before Hamming |
| `truth.rs` | ladybug-rs | Full NARS inference: revision, deduction, induction, abduction, analogy |
| `semiring.rs` | ladybug-rs | 7 semiring variants (BFS, HdrPathBind, HammingMinPlus, PageRank, Resonance, ...) |
| `clam_path.rs` | ladybug-rs | 24-bit MSB-first tree encoding + 40-bit MerkleRoot in word[0] |
| `bind_space.rs` | ladybug-rs | 8+8 addressing, 65,536 slots, zero-copy container system |

---

## The Pipeline

```
Query string (Cypher / GQL / NARS)
    |
    v
[BUMPERS] parser.rs + ast.rs + error.rs     <-- lance-graph spare part
    |       validates, rejects malformed input
    v
AST decomposes into SPO triples
    |
    v
[ENGINE] ladybug-rs SPO engine               <-- ladybug-rs native
    |       BindSpace zero-copy, rustynum arrays
    |       NibbleScent prefilter -> Hamming ANN
    |       NARS truth gating
    |       semiring chain traversal
    v
SPO results (thinking complete)
    |
    v
[RIMS] DataFusion row/column joins            <-- existing datafusion dep
    |       ground truth verification
    |       qualified column naming
    v
Row/column output
    |
    v
neo4j-rs compatible format                   <-- n8n-arrow conversion pattern
```

---

## Open Ends

### 1. Parser Extraction
- `parser.rs` still imports `crate::ast::*` and `crate::error::*` — needs to be
  packaged as a standalone crate or module that ladybug-rs can depend on
- Strip lance-specific error variants (DataFusion, LanceCore, Arrow)
- Decide: separate crate (`lance-graph-parser`) or inline into ladybug-rs?

### 2. GQL and NARS Syntax
- Parser currently handles Cypher only
- GQL (ISO/IEC 39075) is ~90% Cypher syntax but has divergences:
  graph patterns, OPTIONAL keyword placement, GRAPH prefix
- NARS syntax (`<S --> P>. %f;c%`) is fundamentally different — needs its own
  parser arm or a separate nom combinator module
- Decision: extend parser.rs with `alt()` branches, or separate parsers per syntax?

### 3. Semantic Analysis Dependency
- `semantic.rs` (~1,800 lines) depends on `GraphConfig` from lance-graph
- Needs adaptation to validate against ladybug-rs BindSpace schema instead
- Variable binding and scope validation is generic and portable
- Type checking needs to know what labels/properties exist in BindSpace

### 4. Neo4j-rs Result Bridge
- aiwar-neo4j-harvest is currently write-only (Cypher generation, no result reading)
- n8n-rs `n8n-arrow` has the `RecordBatch` <-> row conversion pattern
- Need to wire: SPO results -> DataFusion RecordBatch -> neo4j-rs Row format
- Open: should this be a trait in rustynum or a standalone adapter crate?

### 5. Ground Truth Test Portability
- lance-graph has 7 SPO ground truth tests (spo_ground_truth.rs)
- Test patterns (round-trip, projection verbs, gate filtering, prefilter rejection,
  chain traversal, merkle integrity, cypher convergence) are valuable
- Need to rewrite against ladybug-rs types (SparseContainer, CogRecord, etc.)
- The verify_lineage gap (doesn't re-hash content) is documented — verify_integrity
  is the correct path

### 6. Outage Recovery
- PRs 168-171 on ladybug-rs are pending during infrastructure outages
- Merging spare parts should wait until the storm passes
- Risk: force-pushing during outages can lose work

### 7. lance-graph Engine Disposal
- Once spare parts are extracted, lance-graph's SPO engine code
  (builder.rs, store.rs, truth.rs, semiring.rs, merkle.rs, sparse.rs, fingerprint.rs)
  can be archived or removed
- The engine was a prototype — ladybug-rs has the production implementation
- Keep the ground truth test patterns as reference

---

## Vision

Lance-graph becomes a **thin hardened shell** — a parser crate that validates
Cypher/GQL/NARS input and produces a clean AST. No engine, no storage, no
traversal. Just bumpers and rims.

The AST feeds into ladybug-rs's SPO engine, which does the actual graph
thinking using BindSpace zero-copy containers, rustynum arrays, NibbleScent
prefiltering, NARS truth inference, and semiring algebra. All native, all
zero-copy.

After thinking, the results flow through the existing DataFusion dep in
ladybug-rs/rustynum for row/column ground truth verification, then out
through n8n-arrow's conversion pattern into neo4j-rs compatible format.

```
lance-graph-parser (bumpers)
         |
         v
   ladybug-rs SPO engine (rustynum + BindSpace)
         |
         v
   datafusion (already in ladybug-rs) -> row/column rims
         |
         v
   neo4j-rs / n8n-arrow (existing pattern)
```

Three repos, one pipeline, zero duplication. The parser protects the entry,
the engine does the thinking, the rims format the output. Each part owned
by the repo that knows it best.
