# SESSION_FALKORDB_CROSSCHECK.md

## Mission: Reverse-Engineer FalkorDB-rs-next-gen → Connect Dots to Lance-Graph

**Context:** FalkorDB is the only Rust graph database with GraphBLAS semiring algebra.
They are rewriting their C engine in Rust (`falkordb-rs-next-gen`). Their architecture
descends from RedisGraph, which we also transcoded into lance-graph via HoloGraph.
We share DNA. We need to know what they solved, what they got wrong, and what we
can learn to make lance-graph the permissively-licensed successor that captures
Kuzu's orphaned users AND FalkorDB's SSPL-blocked users.

**License note:** FalkorDB core = SSPL v1 (toxic for cloud/service deployment).
Their Rust CLIENT (falkordb-rs) = MIT. The next-gen ENGINE license: check `LICENSE` file.
Our lance-graph = needs to be Apache 2.0 or MIT to win the market.

---

## STEP 1: Clone and Inventory

```bash
cd /home/claude
git clone https://github.com/FalkorDB/falkordb-rs-next-gen.git falkordb-ng
cd falkordb-ng
find . -name "*.rs" -not -path "*/target/*" | head -100
find . -name "*.rs" -not -path "*/target/*" | wc -l
cat Cargo.toml
```

Map their crate structure. What workspace members? What dependencies?
Specifically look for: GraphBLAS bindings, sparse matrix types, Cypher parser,
query planner, execution engine, storage layer.

---

## STEP 2: GraphBLAS Integration — How Do They Bind?

```
QUESTIONS:
  1. Do they use SuiteSparse:GraphBLAS via FFI (C library dependency)?
     Or did they rewrite GraphBLAS operations in pure Rust?
  2. What semirings do they support? Just the standard ones or custom?
  3. How do they dispatch semiring operations? FactoryKernels? JIT? Enum match?
  4. What sparse matrix format? CSR? CSC? Hypersparse? Multiple?
  5. Do they support user-defined semirings or is it fixed?
```

Search for:
```bash
grep -rn "GraphBLAS\|GrB_\|graphblas\|semiring\|Semiring" --include="*.rs" | head -50
grep -rn "extern.*\"C\"\|#\[link\|ffi\|bindgen" --include="*.rs" | head -20
grep -rn "sparse\|CSR\|CSC\|adjacency" --include="*.rs" | head -30
```

**Why this matters:** If they FFI to SuiteSparse C library, they carry a MASSIVE
dependency (SuiteSparse is ~500K LOC of C). We have pure Rust semirings in
lance-graph. If their Rust rewrite is pure Rust GraphBLAS, study HOW they
implemented it — sparse matrix multiply, semiring dispatch, element-wise ops.

---

## STEP 3: Cypher Parser — What Do They Support?

```bash
grep -rn "MATCH\|RETURN\|WHERE\|CREATE\|MERGE\|DELETE\|SET\|cypher\|parser\|ast\|AST" --include="*.rs" | head -40
find . -name "*parser*" -o -name "*cypher*" -o -name "*ast*" -not -path "*/target/*"
```

**QUESTIONS:**
  1. Hand-written parser or parser generator (pest, nom, lalrpop)?
  2. What subset of openCypher? Full spec or limited?
  3. How does the AST map to GraphBLAS operations?
     (This is the Cypher → semiring compilation step)
  4. Do they have a query optimizer? Cost-based? Rule-based?
  5. How do they handle MATCH patterns with variable-length paths?
     (This is where Bellman-Ford / BFS kicks in)

**Why this matters:** Our DataFusion planner already does Cypher → plan.
But their compilation to GraphBLAS semiring ops is what we need to study.
The mapping from `MATCH (a)-[r:KNOWS]->(b)` to `grb_mxv` with the right
semiring is the critical path.

---

## STEP 4: Sparse Matrix Representation — Memory Layout

```bash
grep -rn "struct.*Matrix\|struct.*Sparse\|struct.*Graph\|adjacency" --include="*.rs" | head -30
# Look for the core graph data structure
find . -name "*graph*" -o -name "*matrix*" -o -name "*storage*" -not -path "*/target/*"
```

**QUESTIONS:**
  1. How do they store the adjacency matrix in memory?
     (FalkorDB C version uses SuiteSparse GrB_Matrix internally)
  2. How do they store node properties? Separate columns? Inline?
  3. How do they store relationship properties (edge weights)?
  4. How do they handle typed relationships (multiple adjacency matrices)?
  5. What's their memory layout for SIMD? 64-byte aligned?
  6. Do they use memory-mapped files? Or pure heap?

**Why this matters:** Their memory layout determines whether we can replace
their scalar edge weights with our binary Planes. If their adjacency matrix
stores `f64` per edge, we need to understand how to substitute `BitVec` (2KB)
or `BF16` (2 bytes) without breaking the semiring dispatch.

**COMPARISON TABLE TO BUILD:**
```
ASPECT              FALKORDB-RS-NG          OUR LANCE-GRAPH
────────────────────────────────────────────────────────────
Edge weight type    f64? i64? property?     BitVec (16K-bit) + BF16 cache
Node representation Node ID + properties    Node (3 × Plane = 6KB)
Adjacency format    Sparse matrix (CSR?)    Sparse matrix + Plane distance
Semiring dispatch   FFI? enum match?        enum match on HdrSemiring
Storage backend     Redis? Memory? File?    Lance format (NVMe columnar)
Query language      openCypher              openCypher (DataFusion)
Query optimizer     ?                       DataFusion cost-based
SIMD                SuiteSparse auto-vec?   Hand-written AVX-512
```

---

## STEP 5: Query Execution Pipeline

```bash
# Trace the execution path from query string to result
grep -rn "fn.*execute\|fn.*query\|fn.*eval\|fn.*run" --include="*.rs" | head -30
# Look for the execution engine
find . -name "*exec*" -o -name "*engine*" -o -name "*eval*" -not -path "*/target/*"
```

**QUESTIONS:**
  1. What's the execution model? Volcano (pull)? Push-based? Vectorized?
     (Kuzu was vectorized, FalkorDB C was pull-based)
  2. How do they handle multi-hop patterns? Iterative mxv? Recursive?
  3. Do they support aggregation? GROUP BY? COLLECT?
  4. How do they handle OPTIONAL MATCH (left outer join in graph)?
  5. Do they pipeline results or materialize intermediate matrices?

**Why this matters:** DataFusion gives us vectorized push-based execution
for free. If FalkorDB uses pull-based, we're already faster for analytics.
But their GraphBLAS integration might give them advantages for iterative
algorithms (PageRank, connected components) where mxv is called repeatedly.

---

## STEP 6: What They DON'T Have (Our Advantages)

After full inventory, explicitly document what FalkorDB-rs-next-gen LACKS
that we have:

```
EXPECTED MISSING:
  ✗ Binary vector representations (Plane, BitVec, 16K-bit)
  ✗ Bundle operation (majority vote → concept formation)
  ✗ Cascade search (HDR multi-resolution early rejection)
  ✗ BF16 truth values on edges
  ✗ NARS truth (frequency, confidence)
  ✗ Seal/Staunen (Merkle integrity verification)
  ✗ encounter() (INSERT that also trains)
  ✗ Deterministic RL (integer-only learning loop)
  ✗ Fibonacci/Prime encoding
  ✗ Tropical attention (cascade = multi-head attention)
  ✗ Lance format persistence (NVMe columnar, ACID versioning)
  ✗ Permissive license (they are SSPL)
```

Verify each. If they DO have any of these, document HOW they implemented it.

---

## STEP 7: What They Have That We Should Steal

After inventory, document what they solved that we haven't:

```
EXPECTED LEARNINGS:
  ? How they compile Cypher patterns to semiring mxv calls
  ? How they handle variable-length paths (Kleene star)
  ? How they index for fast node/edge lookup by property
  ? How they handle graph mutations (CREATE, DELETE, SET)
  ? How they manage memory for large graphs
  ? How they handle concurrency (multiple readers/writers)
  ? How they expose the Rust API to Python
  ? How they do TCK (Technology Compatibility Kit) testing for Cypher compliance
  ? How they handle schema-free properties on nodes and edges
```

---

## STEP 8: Architecture Bridge Document

Produce a document mapping FalkorDB-rs-next-gen concepts to lance-graph:

```
FALKORDB CONCEPT          → LANCE-GRAPH EQUIVALENT         STATUS
──────────────────────────────────────────────────────────────────
GrB_Matrix                → GrBMatrix (blasgraph/matrix.rs) EXISTS
GrB_Vector                → GrBVector (blasgraph/vector.rs) EXISTS
GrB_Semiring              → HdrSemiring (blasgraph/semiring.rs) EXISTS
Cypher parser             → ast.rs + DataFusion planner     EXISTS
Node properties           → Plane fields (acc, bits, alpha) DIFFERENT
Edge properties           → BF16 truth + projection byte    DIFFERENT
GRAPH.QUERY command       → CypherQuery::execute()          EXISTS
Multi-graph               → select_graph()                  EXISTS?
Redis protocol            → NOT NEEDED (embedded)           N/A
SuiteSparse FFI           → NOT NEEDED (pure Rust semirings) N/A
```

---

## STEP 9: Competitive Positioning Document

Write a document for potential Kuzu/FalkorDB users explaining:

```
IF YOU'RE COMING FROM KUZU:
  ✓ Same: Embedded, Cypher, fast, Rust
  ✓ Better: binary representations, learning, deterministic
  ✓ Different: Lance storage (versioned, NVMe-optimized)
  ✗ Missing: [list anything Kuzu had that we don't yet]

IF YOU'RE COMING FROM FALKORDB:
  ✓ Same: GraphBLAS semirings, Cypher, Rust
  ✓ Better: permissive license, embedded (no Redis), binary planes, learning
  ✓ Different: Lance storage instead of Redis
  ✗ Missing: [list anything FalkorDB has that we don't yet]

IF YOU'RE USING SEMANTIC KERNEL / CREWAI:
  ✓ We replace: the vector DB + knowledge graph they connect to
  ✓ We add: learning (the DB gets smarter with use)
  ✓ Integration: Python bindings, Lance format ↔ Arrow ↔ Pandas
```

---

## STEP 10: Priority Action Items

Based on the crosscheck, produce a prioritized list:

```
CRITICAL (blocks adoption):
  ? [whatever Cypher features they support that we don't]
  ? [whatever graph operations they have that we lack]

HIGH (competitive parity):
  ? [optimizations they have that improve our performance]
  ? [API patterns that their users expect]

MEDIUM (differentiation):
  ? [things we should steal from their architecture]
  ? [patterns we should adapt for our binary substrate]

LOW (nice to have):
  ? [features unique to Redis that we don't need]
```

---

## FILES TO PRODUCE

```
1. FALKORDB_INVENTORY.md      — complete file/type/method map of their codebase
2. FALKORDB_VS_LANCE.md       — side-by-side comparison table
3. FALKORDB_STEAL_LIST.md     — what to adapt for lance-graph
4. COMPETITIVE_POSITION.md    — user-facing migration guide
5. CYPHER_COVERAGE_GAP.md     — which Cypher features we need to add
```

Push all to `.claude/` in lance-graph repo.

---

## KEY CONTEXT FILES TO READ FIRST

Before starting the crosscheck, read these files from our repos for context:

```
OUR ARCHITECTURE:
  rustynum/.claude/ARCHITECTURE_INDEX.md     — master index
  rustynum/.claude/INVENTORY_MAP.md          — our complete type inventory
  rustynum/.claude/BF16_SEMIRING_EPIPHANIES.md — the 5:2 semiring split
  rustynum/.claude/RESEARCH_THREADS.md       — DreamerV3, BitGNN connections
  rustynum/.claude/DEEP_ADJACENT_EXPLORATION.md — RDF-3X RISC design

OUR BLASGRAPH:
  lance-graph/crates/lance-graph/src/graph/blasgraph/semiring.rs
  lance-graph/crates/lance-graph/src/graph/blasgraph/ops.rs
  lance-graph/crates/lance-graph/src/graph/blasgraph/types.rs
  lance-graph/crates/lance-graph/src/graph/blasgraph/matrix.rs
  lance-graph/crates/lance-graph/src/graph/blasgraph/hdr.rs

OUR DATAFUSION PLANNER:
  lance-graph/crates/lance-graph/src/datafusion_planner/
  lance-graph/crates/lance-graph/src/ast.rs
```

The goal is NOT to copy FalkorDB. The goal is to understand their architectural
decisions so we can build the system they WOULD have built if they had binary
planes, BF16 truth values, encounter-based learning, cascade search, and a
permissive license. We are not competing with their code. We are competing
with their POSITION in the market — and we have better technology.
