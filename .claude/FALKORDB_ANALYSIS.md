# FalkorDB Architecture Analysis

Comprehensive competitive analysis of FalkorDB (formerly RedisGraph) for informing
lance-graph architectural decisions. Based on source code review of the FalkorDB
repository at `/tmp/falkordb-ref`.

---

## 1. FalkorDB Architecture Overview

FalkorDB is a graph database implemented as a Redis module. It stores property graphs
using SuiteSparse GraphBLAS as the core linear algebra engine for adjacency and
traversal operations. The system translates Cypher queries into algebraic expressions
over sparse boolean matrices, evaluates them via GraphBLAS `mxm` (matrix-matrix
multiply), and resolves entity properties through separate side tables.

### Key Components

- **Graph Core** (`src/graph/`): Graph struct containing adjacency matrix,
  per-label matrices, per-relation tensors, node/edge DataBlocks
- **Delta Matrix** (`src/graph/delta_matrix/`): MVCC-style wrapper around GrB_Matrix
  with `M` (committed), `delta_plus` (pending adds), `delta_minus` (pending deletes)
- **Tensor** (`src/graph/tensor/`): 3D matrix (Delta_Matrix of UINT64) where each
  entry is either a scalar edge ID or a pointer to a GrB_Vector of edge IDs
  (handling multi-edges between same node pair)
- **Algebraic Expression** (`src/arithmetic/algebraic_expression/`): Tree-structured
  representation of matrix operations (MUL, ADD, TRANSPOSE, POW)
- **Execution Plan** (`src/execution_plan/`): Volcano-style pull-based iterator model
  with ~30 operator types
- **Index** (`src/index/`): Delegates to RediSearch for full-text, range, and HNSW
  vector indexes

### Licensing

FalkorDB uses SSPL (Server Side Public License v1) -- a copyleft license that
restricts offering the software as a service. Our Apache-2.0 license is a significant
competitive advantage for cloud deployment and embedding.

---

## 2. GraphBLAS Integration Details

### 2.1 Semiring Usage

FalkorDB uses exactly **one semiring** for all traversal operations:

```c
GrB_Semiring semiring = GxB_ANY_PAIR_BOOL;
```

This is the `ANY_PAIR` semiring over booleans:
- **Multiply operator (`PAIR`)**: Always returns `true` regardless of inputs. It only
  cares about structural non-zeros (i.e., "does an edge exist?").
- **Add operator (`ANY`)**: Returns any one of its inputs (non-deterministic, but
  since all values are `true`, this is fine).

This means FalkorDB's GraphBLAS traversals are purely **structural/topological** --
they determine reachability, not weighted paths. All matrix entries are boolean: an
entry exists or it does not.

For weighted operations (shortest paths), FalkorDB builds a separate weighted matrix
in procedure code (`src/procedures/utility/build_weighted_matrix.c`) by extracting
edge properties into a double-typed GrB_Matrix, then uses standard graph algorithms
outside the core algebraic expression pipeline.

A secondary semiring is used for edge collection:
```c
GxB_ANY_SECOND_BOOL  // used in graph_collect_node_edges.c
```

### 2.2 Matrix Operations

The core algebraic expression evaluator (`algebraic_expression_eval.c`) dispatches:

- **`_Eval_Mul`**: Chains `Delta_mxm(res, GxB_ANY_PAIR_BOOL, A, M)` for each
  operand pair. Early-exits when result becomes empty (`nvals == 0`).
- **`_Eval_Add`**: Chains `Delta_eWiseAdd(res, GxB_ANY_PAIR_BOOL, A, B)` for
  element-wise union of matrices.

### 2.3 Delta_mxm Implementation

The `Delta_mxm` function (`delta_mxm.c`) handles MVCC-aware multiplication:

```
C = (A * (M + delta_plus)) <!delta_minus>
```

Where:
1. If `delta_minus` has entries: compute `mask = A * delta_minus` (entries to exclude)
2. If `delta_plus` has entries: compute `accum = A * delta_plus` (entries to add)
3. Compute `C = (A * B) <!mask>` using complement mask descriptor `GrB_DESC_RSC`
4. If additions exist: `C = C + accum`

This is their most sophisticated GraphBLAS pattern -- MVCC at the matrix level.

### 2.4 Filter Matrix Pattern (Key Innovation)

The **Conditional Traverse** operator creates a "filter matrix" `F` of dimensions
`[batch_size x graph_dim]` where `F[i, srcId] = true` maps batch record `i` to
source node `srcId`. This is then prepended to the algebraic expression:

```
Result = F * (Label_A * Rel_R * Label_B)
```

The result matrix `M[i, destId]` directly maps batch record index `i` to destination
nodes. This batched traversal pattern (default batch = 16 records) avoids per-record
traversal overhead.

---

## 3. Cypher to Execution Pipeline

### 3.1 MATCH (a)-[r]->(b) --> Which semiring/mxv?

**Answer**: `GxB_ANY_PAIR_BOOL` via `GrB_mxm` (matrix-matrix multiply, NOT mxv).

The path `(a)-[r]->(b)` becomes an algebraic expression:
```
Label_a * Rel_r * Label_b
```
Where:
- `Label_a` is a diagonal matrix (node label filter for `a`)
- `Rel_r` is the relation matrix for edge type `r`
- `Label_b` is a diagonal matrix (node label filter for `b`)

For multi-type edges like `-[:A|:B]->`, FalkorDB uses matrix addition:
```
(Rel_A + Rel_B)
```

For bidirectional edges `()-[]-()`:
```
Rel + Transpose(Rel)
```

Construction happens in `algebraic_expression_construction.c`, which:
1. Extracts longest paths from the query graph using DFS
2. Normalizes edge directions (minimizing transposes)
3. Builds algebraic expressions from paths
4. Handles variable-length edges by isolating them

### 3.2 WHERE r.weight > 0.5 --> How is this a matrix mask?

**Answer**: It is NOT a matrix mask. Property filters are evaluated as post-hoc
row-by-row filters.

FalkorDB's approach to property filtering:
1. The `OpFilter` operator (`op_filter.c`) sits above traversal operators
2. For each record produced by traversal, it calls `FilterTree_applyFilters`
3. If the filter fails, the record is discarded and the next is requested

There is one optimization: the `utilizeIndices` optimization (`utilize_indices.c`)
can convert some property filters into RediSearch index scans, replacing
`LabelScan + Filter` with `IndexScan`. But this only works for indexed properties.

Property filters NEVER become GraphBLAS masks. The matrices are purely boolean
(structural), and properties live in side tables (AttributeSet on each entity).

### 3.3 RETURN count(*) --> Which reduction?

**Answer**: Hash-based aggregation, NOT a GraphBLAS reduction.

The `OpAggregate` operator (`op_aggregate.c`):
1. Eagerly consumes all child records
2. Groups records by key expressions using XXH64 hash into a hash table
3. For each group, maintains cloned aggregate expression trees
4. Calls `AR_EXP_Aggregate(exp, r)` for each record
5. On handoff, calls `AR_EXP_FinalizeAggregations` to produce final values

However, there IS a `reduceCount` optimization (`reduce_count.c`) that recognizes
the pattern `Scan -> Aggregate(count) -> Results` and replaces it with a direct
`Graph_NodeCount()` or `Graph_LabeledNodeCount()` call, bypassing both scan and
aggregation entirely. Similarly for edge counts via `Graph_RelationEdgeCount()`.

For `OPTIONAL MATCH`, the conditional traverse uses a GraphBLAS reduction:
```c
GrB_Matrix_reduce_Monoid(op->w, NULL, NULL, GxB_LOR_BOOL_MONOID, M, NULL)
```
This reduces rows of the result matrix to identify which source nodes had matches.

---

## 4. Property Graph Storage Model

### 4.1 Node Labels --> Separate Matrices Per Label

**Yes.** Each label gets its own `Delta_Matrix` (boolean, diagonal):

```c
Delta_Matrix *labels;        // array of label matrices, one per label
Delta_Matrix node_labels;    // mapping of all node IDs to all labels
```

- `labels[i]` is an NxN diagonal boolean matrix where `L[j,j] = true` means node `j`
  has label `i`
- `node_labels` is an NxL matrix mapping each node to all its labels
- Nodes can have multiple labels (multi-label support)

### 4.2 Edge Types --> Separate Matrices Per Type (Tensors)

**Yes, using Tensors.** Each relation type gets a `Tensor`:

```c
Tensor *relations;  // array of relation tensors, one per type
```

A Tensor is a `Delta_Matrix` of type `GrB_UINT64`. Each entry `T[src, dest]` is
either:
- A **scalar** edge ID (when there is exactly one edge of this type between src/dest)
- A **pointer to a GrB_Vector** (MSB set) containing multiple edge IDs (multi-edges)

This is an elegant solution to the multi-edge problem that standard GraphBLAS
boolean matrices cannot represent.

Additionally, a global `adjacency_matrix` (boolean Delta_Matrix) tracks whether ANY
edge exists between two nodes, regardless of type.

### 4.3 Properties --> Side Tables (Not Inline)

Properties are stored in **side tables**, completely separate from the matrix
structure:

```c
typedef struct {
    AttributeSet *attributes;  // pointer to attribute set
    EntityID id;
} GraphEntity;
```

- Nodes and edges are stored in `DataBlock` structures (memory pools with block
  allocation)
- Each entity has an `AttributeSet` -- a compact sorted array of (AttributeID, SIValue)
  pairs
- AttributeIDs are `uint16_t` (max 65535 distinct attribute names)
- Properties are accessed by entity ID lookup into the DataBlock, then linear scan
  of the AttributeSet

This means property access during filtering is O(1) for entity lookup + O(k) for
attribute scan where k is the number of attributes on that entity.

---

## 5. Indexing Strategy

FalkorDB delegates ALL indexing to **RediSearch** (the Redis search module):

### 5.1 Range Index

- Created via `Index_RangeCreate`
- Supports numeric, string (tag), geo, and array fields
- Field names prefixed with `range:` internally
- Uses RediSearch's NUMERIC and TAG field types
- Enables `WHERE n.age > 30` to bypass label scan + filter

### 5.2 Full-Text Index

- Created via `Index_FulltextCreate`
- Supports stemming, phonetic search, custom weights, stopwords, language selection
- Uses RediSearch's FULLTEXT field type
- Enables `CALL db.idx.fulltext.queryNodes('idx', 'search term')`

### 5.3 Vector Index

- Created via `Index_VectorCreate`
- Uses HNSW algorithm (via RediSearch's vector similarity)
- Configurable parameters: dimension, M (max outgoing edges), efConstruction,
  efRuntime, similarity function (VecSimMetric)
- Supports `float32` vectors only (via `vecf32()` function)
- Uses `Index_BuildVectorQueryTree` to construct KNN queries

### 5.4 Index Integration with Query Planning

The `utilizeIndices` optimizer can:
- Replace `NodeByLabelScan + Filter` with `NodeByIndexScan`
- Replace `CondTraverse + Filter` with `EdgeByIndexScan`
- Convert equality, range, IN, contains, starts-with, ends-with predicates
- Handle composite filters (AND conditions across indexed fields)

---

## 6. Execution Plan Patterns

### 6.1 Operator Taxonomy

FalkorDB has approximately 30 operator types in a Volcano-model (pull-based):

**Source operators** (data producers):
- `AllNodeScan` -- scan all nodes
- `NodeByLabelScan` -- scan nodes of specific label (iterates label matrix diagonal)
- `NodeByLabelAndIDScan` -- label scan with ID range filter (uses Roaring bitmaps)
- `NodeByIndexScan` -- use RediSearch index
- `EdgeByIndexScan` -- use RediSearch index for edges
- `NodeByIdSeek` -- direct node lookup by ID
- `ArgumentList` -- receives input from parent scope

**Traversal operators**:
- `ConditionalTraverse` -- the core traversal: batched mxm with filter matrix
- `ExpandInto` -- traversal when both endpoints already bound (checks connectivity)
- `CondVarLenTraverse` -- variable-length path traversal (BFS/DFS-based)

**Transform operators**:
- `Filter` -- row-by-row predicate evaluation
- `Project` -- expression evaluation and column selection
- `Aggregate` -- hash-based grouping with aggregate functions
- `Sort` -- in-memory quicksort
- `Distinct` -- hash-based deduplication
- `Limit`, `Skip` -- pagination
- `Unwind` -- list expansion
- `CartesianProduct` -- cross join
- `ValueHashJoin` -- hash join on expression equality

**Mutation operators**:
- `Create`, `Delete`, `Update`, `Merge`

### 6.2 Standard Pattern: Scan -> Filter -> Expand -> Aggregate

A typical MATCH query:
```
MATCH (a:Person)-[r:KNOWS]->(b:Person) WHERE a.age > 30 RETURN b.name, count(*)
```

Produces (bottom-up):
```
Results
  Aggregate [b.name, count(*)]
    Conditional Traverse (a)-[r:KNOWS]->(b)
      Filter [a.age > 30]     // or NodeByIndexScan if indexed
        Node By Label Scan [a:Person]
```

With index optimization, this becomes:
```
Results
  Aggregate [b.name, count(*)]
    Conditional Traverse (a)-[r:KNOWS]->(b)
      Node By Index Scan [a:Person WHERE a.age > 30]
```

### 6.3 Compile-Time Optimizations

Applied in order (`optimizer.c`):
1. `reduceScans` -- remove redundant scan operations
2. `filterVariableLengthEdges` -- push filters into var-len traversals
3. `reduceCartesianProductStreamCount` -- optimize cross joins
4. `applyJoin` -- convert cartesian products to hash joins
5. `reduceTraversal` -- convert traverse to expand-into when both nodes bound
6. `reduceDistinct` -- remove distinct after aggregation
7. `batchOptionalMatch` -- batch optional match operations

### 6.4 Runtime Optimizations

1. `compactFilters` -- merge filter trees
2. `utilizeIndices` -- replace scans with index lookups
3. `costBaseLabelScan` -- choose label with fewest entities for scan
4. `seekByID` -- replace scan + ID filter with direct seek
5. `reduceFilters` -- combine multiple filters into one
6. `reduceCount` -- replace scan + count aggregate with precomputed value

---

## 7. What to Steal (Adapt for Our Architecture)

### 7.1 Filter Matrix Batching Pattern (HIGH PRIORITY)

FalkorDB's filter matrix pattern is highly adaptable:

```
F[batch_idx, src_id] = 1    // map batch records to source nodes
Result = F * RelationMatrix   // batched traversal
```

**For lance-graph**: We could implement this using our ndarray/sparse matrix types.
Instead of boolean filter matrices, we could use our BitVec filter matrices, enabling
batch traversal with our custom semirings. This would allow:
```
F_hamming[batch_idx, src_id] = query_bitvec
Result = F_hamming (*) RelationMatrix   // using HammingMin semiring
```

This batches similarity queries across multiple source nodes simultaneously.

### 7.2 Delta Matrix / MVCC Pattern (MEDIUM PRIORITY)

The three-matrix MVCC approach (M + delta_plus - delta_minus) is worth studying:
- Enables concurrent reads during writes
- Deferred flush means writes don't block reads
- The `<!mask>` complement mask pattern for applying deletions during mxm is clean

**For lance-graph**: We could implement a similar pattern for our sparse matrices
to support concurrent read/write access. Our pure-Rust GraphBLAS makes this easier
since we control the implementation.

### 7.3 Tensor for Multi-Edge Storage (LOW PRIORITY)

The Tensor approach (scalar vs vector entries using MSB flag) is clever for
multi-edges. If two nodes have exactly one edge, it's a scalar (just the edge ID).
If they have multiple edges, it transparently becomes a vector.

**For lance-graph**: Our edge weights (BF16 truth values) are always present, so
we may not need this exact pattern. But the concept of promoting scalar entries to
vectors on demand could be useful for multi-relation edges.

### 7.4 Algebraic Expression Tree from Query Graph (HIGH PRIORITY)

The construction algorithm (`algebraic_expression_construction.c`) is valuable:
1. Find longest path through query graph (DFS)
2. Normalize edge directions (minimize transposes)
3. Build expression tree with MUL for path chains, ADD for type unions
4. Handle variable-length edges by isolation

**For lance-graph**: Our DataFusion integration should build similar algebraic
expression trees. The longest-path-first strategy for expression ordering is
directly applicable.

### 7.5 reduceCount Optimization (MEDIUM PRIORITY)

Detecting `Scan -> Aggregate(count) -> Results` and replacing it with a precomputed
count is a simple but effective optimization. We should implement similar pattern
detection in our DataFusion optimizer rules.

### 7.6 Cost-Based Label Selection (MEDIUM PRIORITY)

The `costBaseLabelScan` optimization picks the label with fewest entities when
scanning. This cardinality-based optimization is easy to implement and high impact.

---

## 8. Our Advantages (What They Cannot Do)

### 8.1 Binary Hamming Search

FalkorDB has NO binary vector support. Their vector index uses `float32` exclusively
(via `vecf32()` function) with HNSW and cosine/L2/IP similarity metrics.

**Our advantage**: 16384-bit BitVec (2KB) binary vectors with native Hamming distance.
This enables:
- Hardware-accelerated `VPOPCNTDQ` for popcount on AVX-512
- Exact binary search (deterministic, reproducible)
- 128x storage reduction vs float32 at equivalent dimensions
- No approximation error from ANN indexes

### 8.2 Cascade Multi-Resolution Rejection

FalkorDB has a flat search model: query the HNSW index, get top-K results.
No multi-resolution filtering exists.

**Our advantage**: Cascade search evaluates at multiple resolutions, rejecting
non-matches early at coarse resolution before investing compute at fine resolution.
This provides sub-linear search time that HNSW cannot match for binary vectors.

### 8.3 BF16 Truth Encoding

FalkorDB's edge weights are not first-class citizens in the matrix. Properties are
stored in side tables, and weights must be extracted into separate matrices for
weighted algorithms.

**Our advantage**: BF16-encoded NARS truth values (frequency + confidence) ARE the
matrix elements. Edge weights participate directly in semiring operations:
- `Resonance` semiring computes truth-value revision
- `SimilarityMax` semiring propagates confidence-weighted similarity
- No side-table lookup required during traversal

### 8.4 Deterministic Integer Operations

FalkorDB uses `GxB_ANY_PAIR_BOOL` which is explicitly non-deterministic (the `ANY`
monoid can return any input). While this is fine for boolean reachability, it means
their algebraic operations are not reproducible across runs.

**Our advantage**: All 7 of our semirings produce deterministic results:
- `XorBundle`: deterministic XOR composition
- `BindFirst`: deterministic binding
- `HammingMin`: deterministic minimum Hamming distance
- `Boolean`: standard deterministic boolean algebra
- `XorField`: deterministic field operations over GF(2)

### 8.5 VPOPCNTDQ Acceleration

FalkorDB relies on SuiteSparse GraphBLAS for SIMD acceleration, which uses
general-purpose sparse matrix kernels. They have NO specialized hardware support
for binary vector operations.

**Our advantage**: Direct AVX-512 VPOPCNTDQ instruction support for population
count operations on our 16384-bit vectors. A single 512-bit register processes
64 bytes at once, and our 2KB vectors fit in exactly 32 AVX-512 registers.

### 8.6 Custom Semiring Algebra

FalkorDB uses exactly one semiring (`ANY_PAIR_BOOL`) for all traversals. Their
algebraic expression system is hard-coded to boolean reachability.

**Our advantage**: 7 distinct semirings enabling fundamentally different traversal
semantics:
- Pattern matching (XorBundle, XorField)
- Similarity search (HammingMin, SimilarityMax)
- Logical inference (Boolean, Resonance)
- Variable binding (BindFirst)

### 8.7 Pure Rust GraphBLAS (No FFI)

FalkorDB vendors SuiteSparse GraphBLAS (C library, ~200K+ lines) and uses it via
direct C linkage.

**Our advantage**: Pure Rust implementation means:
- No unsafe FFI boundaries
- Rust ownership/borrowing for memory safety
- Easier to customize semirings and matrix types
- Cross-compilation without C toolchain
- No SSPL contamination risk from SuiteSparse

### 8.8 DataFusion Query Planning

FalkorDB has a custom Cypher parser and hand-written optimizer with ~12 optimization
rules.

**Our advantage**: DataFusion provides:
- Mature cost-based optimizer with dozens of rules
- SQL and custom language support
- Arrow-native columnar execution
- Distributed query planning
- Community-maintained optimization passes

---

## 9. Competitive Position Summary

### Where FalkorDB is Stronger

| Capability | FalkorDB | lance-graph |
|---|---|---|
| Production maturity | Years in production | Pre-production |
| Cypher completeness | Full Cypher support | DataFusion SQL + custom |
| Ecosystem | Redis module ecosystem | Standalone |
| Full-text search | RediSearch integration | Not yet |
| ACID transactions | Delta Matrix MVCC | TBD |
| Multi-edge support | Tensor (scalar/vector) | TBD |
| Graph algorithms | BFS, DFS, shortest paths, cycle detection | TBD |

### Where lance-graph is Stronger

| Capability | FalkorDB | lance-graph |
|---|---|---|
| Binary vector search | None (float32 only) | 16384-bit BitVec native |
| Semiring variety | 1 (ANY_PAIR_BOOL) | 7 custom semirings |
| Edge weight semantics | Side table properties | BF16 truth values in matrix |
| Hardware acceleration | Generic SuiteSparse | VPOPCNTDQ, AVX-512 |
| Search algorithm | Flat HNSW | Cascade multi-resolution |
| Determinism | Non-deterministic ANY | Fully deterministic |
| License | SSPL (restrictive) | Apache-2.0 (permissive) |
| Memory safety | C (manual) | Rust (guaranteed) |
| Query planning | Custom (~12 rules) | DataFusion (mature) |

### Fundamental Architectural Difference

FalkorDB is a **boolean reachability engine** with properties on the side.
Its matrices answer "can you get from A to B?" but not "how similar is A to B?"
or "what is the truth value of the path from A to B?"

lance-graph is a **semantic algebra engine** where the matrix elements themselves
carry meaning (binary patterns, truth values, similarity scores) and the semiring
operations define the semantics of traversal.

This is not a minor difference -- it represents fundamentally different computational
models. FalkorDB cannot add our capabilities without replacing their entire GraphBLAS
usage pattern.

---

## 10. Priority Actions for lance-graph

### P0 (Immediate)

1. **Implement filter matrix batching**: Adapt FalkorDB's `F * Expression` pattern
   for our semirings. This is the single most impactful optimization to steal.
   Batch 16-64 source nodes into a filter matrix, multiply through the expression
   tree in one pass.

2. **Algebraic expression builder from query graph**: Port the longest-path DFS +
   normalize-transposes + build-expression-tree algorithm as a DataFusion physical
   plan rule.

### P1 (Next Sprint)

3. **reduceCount and similar pattern optimizations**: Detect common aggregation
   patterns and short-circuit them. Easy wins with high user-visible impact.

4. **Cost-based scan selection**: When multiple labels are specified, start with
   the smallest. Use graph statistics (our equivalent of `GraphStatistics`).

5. **Delta matrix for MVCC**: Design our concurrent access pattern. The three-matrix
   approach (M + DP - DM with complement masking) is well-tested.

### P2 (Future)

6. **Index integration**: Define our index API for range, full-text, and (our
   native) binary vector indexes. Unlike FalkorDB, our vector index should use
   Cascade search natively.

7. **Variable-length path traversal**: Implement our equivalent of
   `CondVarLenTraverse`, using semiring-aware BFS (not just boolean reachability).

8. **Bulk import**: FalkorDB's binary bulk import protocol is simple (typed
   binary stream with header). We should define a similar protocol, potentially
   using Arrow IPC format for better interop.

### P3 (Differentiation)

9. **Semiring-aware shortest paths**: FalkorDB's shortest path procedures extract
   weights into separate matrices. We can do weighted shortest paths DIRECTLY in
   the semiring (e.g., MinPlus over BF16 truth values).

10. **Multi-semiring expressions**: Enable algebraic expressions that use different
    semirings at different stages. E.g., `HammingMin` for the first hop,
    `Resonance` for aggregation. FalkorDB cannot do this -- they are locked to
    `ANY_PAIR_BOOL`.

---

*Analysis performed: 2026-03-16*
*Source: FalkorDB repository at /tmp/falkordb-ref*
*Target: lance-graph architecture at /home/user/lance-graph*
