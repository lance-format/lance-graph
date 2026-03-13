# feat: Arrow 57 / DataFusion 51 / Lance 2 + BlasGraph semiring algebra + SPO triple store

## Summary

This PR adds two new graph computation modules beneath the existing Cypher query engine:

1. **BlasGraph** — A GraphBLAS-inspired sparse matrix algebra over 16,384-bit hyperdimensional vectors (3,173 lines, 81 unit tests)
2. **SPO triple store** — A Subject-Predicate-Object graph store using 512-bit fingerprint bitmaps, NARS truth values, and Merkle integrity stamping (1,443 lines + 355 lines integration tests, 38 unit + 7 integration tests)

Both are accompanied by a dependency upgrade (arrow 56→57, datafusion 50→51, lance 1→2, deltalake 0.29→0.30, pyo3 0.25→0.26) and CI coverage for `lance-graph-python`.

**29 files changed, 7,451 insertions, 877 deletions. 126 tests, all passing. Clippy clean across all crates.**

---

## 1. Dependency Upgrades

### Version matrix

| Dependency | Before | After | Crates |
|---|---|---|---|
| arrow / arrow-array / arrow-schema | 56.2 | 57 | lance-graph, lance-graph-catalog, lance-graph-python |
| datafusion (+ common, expr, sql, functions-aggregate) | 50.3 | 51 | lance-graph, lance-graph-catalog, lance-graph-python |
| lance / lance-linalg / lance-namespace | 1.x | 2 | lance-graph |
| deltalake | 0.29 | 0.30 | lance-graph (optional, `delta` feature) |
| pyo3 | 0.25 | 0.26 | lance-graph-python |
| lance-arrow / lance-index | — | 2 | lance-graph (dev-dependencies) |

### Why deltalake 0.29 → 0.30

`deltalake 0.29` depends on `datafusion 50.3`. With our bump to `datafusion 51`, the `delta` feature (enabled by default) would fail resolution. `deltalake 0.30` uses `datafusion ^51.0`, resolving the conflict.

### Why pyo3 0.25 → 0.26

`arrow-pyarrow 57` bumped its pyo3 dependency from 0.25 to 0.26. Since both `arrow` (via pyarrow feature) and `lance-graph-python` (direct dep) pull in `pyo3-ffi`, and `pyo3-ffi` declares `links = "python"`, having two different pyo3-ffi versions in the same workspace causes a Cargo resolver conflict. Bumping to 0.26 ensures a single `pyo3-ffi` version.

The pyarrow feature (`arrow = { version = "57", features = ["pyarrow"] }`) is preserved — zero-copy C Data Interface bridging between Python RecordBatches and Rust RecordBatches remains intact.

### API adaptations

**`sql_query.rs:175`** — DataFusion 51 schema accessor change:
```rust
// Before (datafusion 50.3):
let arrow_schema = Arc::new(arrow_schema::Schema::from(df.schema()));
// After (datafusion 51):
let arrow_schema = Arc::new(df.schema().as_arrow().clone());
```

**`table_readers.rs:128`** — Suppress `unused_mut` warning when `delta` feature is disabled:
```rust
#[allow(unused_mut)]
let mut readers: Vec<Arc<dyn TableReader>> = vec![Arc::new(ParquetTableReader)];
#[cfg(feature = "delta")]
readers.push(Arc::new(DeltaTableReader));
```

**`executor.rs`** — pyo3 0.26 renames (mechanical, no behaviour change):
- `Python::with_gil(|py| ...)` → `Python::attach(|py| ...)`
- `py.allow_threads(|| ...)` → `py.detach(|| ...)`

**`graph.rs`** — pyo3 0.26 type alias deprecation (mechanical):
- `PyObject` → `Py<PyAny>` (10 occurrences)

---

## 2. BlasGraph — GraphBLAS Semiring Algebra

**Location:** `crates/lance-graph/src/graph/blasgraph/`
**8 files, 3,173 lines, 81 unit tests**

### What is it

A pure-Rust implementation of the [GraphBLAS](https://graphblas.org/) sparse matrix algebra interface, adapted for hyperdimensional computing (HDC). Instead of operating on scalars (f64, bool), all operations work on **16,384-bit binary vectors** (256 × u64 words). This enables algebraic graph algorithms where "vertex labels" and "edge weights" are high-dimensional representations that compose via XOR binding and bundle via majority vote.

### Why

Graph algorithms expressed as sparse linear algebra over semirings are composable, parallelisable, and can be accelerated on GPUs. By choosing the right semiring, the same `mxm` (matrix-matrix multiply) kernel computes BFS, shortest path, PageRank, or reachability — no algorithm-specific code needed.

### Architecture

```
blasgraph/
├── types.rs       # BitVec (16384-bit), HdrScalar, operators (Unary/Binary/Monoid/Select)
├── semiring.rs    # Semiring trait + 7 built-in semirings
├── sparse.rs      # SparseVec, CooStorage, CsrStorage
├── matrix.rs      # GrBMatrix — CSR-backed sparse matrix
├── vector.rs      # GrBVector — sorted sparse vector
├── descriptor.rs  # Descriptor — operation control (transpose, mask, replace)
├── ops.rs         # Free-function API + graph algorithms (BFS, SSSP, PageRank)
└── mod.rs         # Exports, GrBInfo status codes
```

### Type system (`types.rs`, 557 lines, 23 tests)

The fundamental type is `BitVec` — a 16,384-bit binary vector stored as `[u64; 256]`:

```rust
pub const VECTOR_WORDS: usize = 256;
pub const VECTOR_BITS: usize = VECTOR_WORDS * 64; // 16,384

pub struct BitVec {
    words: [u64; VECTOR_WORDS],
}
```

**BitVec operations:**
- Bitwise: `xor()`, `and()`, `or()`, `not()`
- HDC: `bundle(vecs)` — majority vote across vectors (consensus); `permute(n)` — cyclic left-shift by n bits (role binding)
- Metrics: `popcount()`, `density()` (fraction of 1-bits), `hamming_distance(other)`
- Construction: `zero()`, `ones()`, `random(seed)` — uses `splitmix64` PRNG to prevent XOR collapse

`HdrScalar` is the tagged value union used throughout matrix/vector operations:

```rust
pub enum HdrScalar {
    Vector(BitVec),  // 16384-bit HD vector
    Float(f32),      // distances, similarities
    Bool(bool),      // reachability
    Empty,           // structural zero
}
```

Operator enums (`UnaryOp`, `BinaryOp`, `MonoidOp`, `SelectOp`) parameterise all operations, enabling the same kernel code to dispatch different algebraic behaviours at runtime.

### 7 built-in semirings (`semiring.rs`, 476 lines, 8 tests)

A semiring `(⊕, ⊗, 0)` defines how inner products accumulate. The `Semiring` trait:

```rust
pub trait Semiring: Send + Sync {
    fn multiply(&self, a: &HdrScalar, b: &HdrScalar) -> HdrScalar; // ⊗
    fn add(&self, a: &HdrScalar, b: &HdrScalar) -> HdrScalar;      // ⊕
    fn zero(&self) -> HdrScalar;                                     // identity for ⊕
    fn name(&self) -> &str;
}
```

| `HdrSemiring` variant | ⊗ Multiply | ⊕ Add | Use case |
|---|---|---|---|
| `XorBundle` | XOR | Bundle (majority vote) | Path composition — XOR binds edge to path, bundle accumulates multiple paths |
| `BindFirst` | XOR | First non-empty | BFS — propagate first-discovered path, ignore later arrivals |
| `HammingMin` | Hamming distance | Min | Shortest path — distance = Hamming between vectors, accumulate = min |
| `SimilarityMax` | 1 − hamming/16384 | Max | Best match — find most similar vector by cosine-like metric |
| `Resonance` | XOR | Best density (closest to 0.5) | Query expansion — keep most "balanced" (maximally entropic) vector |
| `Boolean` | AND | OR | Reachability — standard Boolean algebra |
| `XorField` | XOR | XOR | GF(2) algebra — finite field for linear independence checks |

### Sparse storage (`sparse.rs`, 389 lines, 9 tests)

Two formats, mirroring GraphBLAS:

- **`CooStorage`** (coordinate/triplet) — `(row, col, value)` triples, used for incremental construction. `push(row, col, val)` appends; `to_csr()` converts for computation.
- **`CsrStorage`** (compressed sparse row) — `row_ptrs[nrows+1]` + `col_indices[nnz]` + `values[nnz]`. Efficient row iteration for `mxv`. `O(log nnz_per_row)` random access via binary search on `col_indices`.
- **`SparseVec`** — sorted `(index, BitVec)` pairs. `O(log n)` get, insert maintains sort invariant.

### Matrix (`matrix.rs`, 561 lines, 11 tests)

`GrBMatrix` wraps `CsrStorage`. Core operations:

```rust
// Semiring-parameterised matrix multiply: C = A ⊕.⊗ B
pub fn mxm(&self, other: &GrBMatrix, semiring: &dyn Semiring, desc: &Descriptor) -> GrBMatrix;
pub fn mxv(&self, vec: &GrBVector, semiring: &dyn Semiring, desc: &Descriptor) -> GrBVector;
pub fn vxm(vec: &GrBVector, mat: &GrBMatrix, semiring: &dyn Semiring, desc: &Descriptor) -> GrBVector;

// Element-wise (Hadamard-like)
pub fn ewise_add(&self, other: &GrBMatrix, op: BinaryOp, desc: &Descriptor) -> GrBMatrix;  // union
pub fn ewise_mult(&self, other: &GrBMatrix, op: BinaryOp, desc: &Descriptor) -> GrBMatrix; // intersection

// Reductions
pub fn reduce_rows(&self, monoid: MonoidOp) -> GrBVector;   // each row → scalar
pub fn reduce_cols(&self, monoid: MonoidOp) -> GrBVector;   // each col → scalar
pub fn reduce(&self, monoid: MonoidOp) -> HdrScalar;        // entire matrix → scalar

// Structural
pub fn extract(&self, rows: &[usize], cols: &[usize]) -> GrBMatrix;
pub fn apply(&self, op: UnaryOp) -> GrBMatrix;
pub fn transpose(&self) -> GrBMatrix;
```

The `Descriptor` controls transposition of inputs, mask complement, and output replacement — matching the GraphBLAS C API semantics. Pre-built descriptors in `GrBDesc`: `default()`, `t0()`, `t1()`, `t0t1()`, `comp()`, `replace()`, `structure()`.

### Vector (`vector.rs`, 370 lines, 11 tests)

`GrBVector` wraps `SparseVec`. In addition to the standard algebraic ops (`ewise_add`, `ewise_mult`, `apply`, `reduce`), provides nearest-neighbour search:

```rust
pub fn find_nearest(&self, query: &BitVec, k: usize) -> Vec<(usize, u32)>;       // k-NN by Hamming
pub fn find_within(&self, query: &BitVec, max_distance: u32) -> Vec<(usize, u32)>; // range search
pub fn find_most_similar(&self, query: &BitVec) -> Option<(usize, f32)>;          // best match
```

### Graph algorithms (`ops.rs`, 478 lines, 14 tests)

Three standard algorithms expressed as semiring iterations:

**`hdr_bfs(adj, source, max_depth)`** — Breadth-first search using `BindFirst` semiring. Level-synchronous frontier expansion. Returns vector where `v[i]` = XOR-bound path representation from source to vertex i. Stops when no new vertices discovered.

**`hdr_sssp(adj, source, max_iters)`** — Single-source shortest path using `HammingMin` semiring. Bellman-Ford-style relaxation. Returns vector where `v[i]` = minimum total Hamming distance from source to vertex i. Converges when no distances improve.

**`hdr_pagerank(adj, max_iters, damping)`** — Importance ranking using `XorBundle` semiring. Each node's rank = bundle of neighbours' ranks. Initialises each vertex with `BitVec::random(seed)`, iterates `max_iters` times.

---

## 3. SPO Triple Store

**Location:** `crates/lance-graph/src/graph/spo/` + `crates/lance-graph/src/graph/fingerprint.rs` + `crates/lance-graph/src/graph/sparse.rs`
**Integration tests:** `crates/lance-graph/tests/spo_ground_truth.rs`
**8 files, 1,798 lines total, 38 unit + 7 integration tests**

### What is it

An in-memory Subject-Predicate-Object triple store that represents graph edges as fingerprint-packed bitmaps. Instead of exact key lookup, queries use approximate nearest-neighbour (ANN) search over Hamming distance — enabling fuzzy graph traversal where "find edges near this pattern" replaces "find edges matching this exact key."

### Why

The Cypher engine operates on property graphs via DataFusion (tabular). The SPO store provides a complementary path for direct graph operations:
- Fingerprint-based addressing avoids schema constraints
- ANN queries enable similarity-based graph exploration
- Truth values gate query results by confidence (relevant for knowledge graphs with uncertain edges)
- Semiring chain traversal composes multi-hop paths algebraically

### Architecture

```
graph/
├── fingerprint.rs  # Fingerprint = [u64; 8] (512-bit), label_fp(), dn_hash(), hamming_distance()
├── sparse.rs       # Bitmap = [u64; 8], pack_axes(), bitwise ops
├── spo/
│   ├── truth.rs    # TruthValue (frequency, confidence), TruthGate thresholds
│   ├── builder.rs  # SpoBuilder — constructs SpoRecord from (s, p, o, truth)
│   ├── store.rs    # SpoStore — HashMap<u64, SpoRecord> with bitmap ANN queries
│   ├── semiring.rs # HammingMin semiring for chain traversal
│   └── merkle.rs   # MerkleRoot, ClamPath, BindSpace — integrity verification
└── mod.rs          # ContainerGeometry enum (Spo = 6)
```

### Fingerprints (`fingerprint.rs`, 144 lines, 6 tests)

```rust
pub const FINGERPRINT_WORDS: usize = 8;  // 512 bits total
pub type Fingerprint = [u64; FINGERPRINT_WORDS];
```

- **`label_fp(label: &str) → Fingerprint`** — FNV-1a inspired hash across all 8 words. Includes an 11% density guard: if the resulting fingerprint has > 11% bits set, it's thinned via XOR-fold to prevent bitmap saturation when fingerprints are OR-packed.
- **`dn_hash(dn: &str) → u64`** — FNV-1a hash for keying records in `SpoStore`. Deterministic: same DN always yields same key.
- **`hamming_distance(a, b) → u32`** — XOR all words, popcount. Satisfies metric axioms (self-distance = 0, symmetry, triangle inequality).

### Bitmap packing (`sparse.rs`, 128 lines, 6 tests)

```rust
pub const BITMAP_WORDS: usize = 8;  // matches FINGERPRINT_WORDS
pub type Bitmap = [u64; BITMAP_WORDS];
```

The key function is `pack_axes(s, p, o) → Bitmap` which computes `s | p | o` — the OR of all three fingerprints. This creates a combined search vector where a query bitmap (e.g., `s | p` for forward queries) can be compared against stored bitmaps via Hamming distance.

**Note:** `BITMAP_WORDS` was previously hardcoded as 2, which silently truncated fingerprints to 128 bits. Now matches `FINGERPRINT_WORDS = 8` for full 512-bit coverage.

### Truth values (`truth.rs`, 175 lines, 6 tests)

NARS-inspired two-valued truth representation:

```rust
pub struct TruthValue {
    pub frequency: f32,   // how often the statement is true [0, 1]
    pub confidence: f32,  // how much evidence supports this [0, 1]
}
```

- **`expectation()`** = `confidence * (frequency − 0.5) + 0.5` — the decision-theoretic expected truth
- **`strength()`** = `frequency * confidence` — simple product
- **`revision(other)`** — combines independent evidence via weighted averaging (confidence-weighted)

**`TruthGate`** — 5 preset thresholds for filtering query results:

| Gate | Min expectation | Use |
|---|---|---|
| OPEN | 0.0 | Accept everything |
| WEAK | 0.4 | Loose filter |
| NORMAL | 0.6 | Default |
| STRONG | 0.75 | High confidence only |
| CERTAIN | 0.9 | Near-certain only |

### Builder (`builder.rs`, 119 lines, 2 tests)

```rust
pub struct SpoRecord {
    pub subject: Fingerprint,
    pub predicate: Fingerprint,
    pub object: Fingerprint,
    pub packed: Bitmap,         // subject | predicate | object
    pub truth: TruthValue,
}
```

`SpoBuilder` constructs records and query vectors:
- `build_edge(s, p, o, truth) → SpoRecord` — packs via `s | p | o`
- `build_forward_query(s, p) → Bitmap` — returns `s | p` (for SxP→O queries)
- `build_reverse_query(p, o) → Bitmap` — returns `p | o` (for PxO→S queries)
- `build_relation_query(s, o) → Bitmap` — returns `s | o` (for SxO→P queries)

### Store (`store.rs`, 308 lines, 3 tests)

```rust
pub struct SpoStore {
    records: HashMap<u64, SpoRecord>,
}
```

**2³ projection verbs** — the three canonical query patterns for a triple store:

```rust
pub fn query_forward(&self, subject: &Fingerprint, predicate: &Fingerprint, radius: u32) -> Vec<SpoHit>;
pub fn query_reverse(&self, predicate: &Fingerprint, object: &Fingerprint, radius: u32) -> Vec<SpoHit>;
pub fn query_relation(&self, subject: &Fingerprint, object: &Fingerprint, radius: u32) -> Vec<SpoHit>;
```

Each constructs the appropriate query bitmap, scans all records, and returns hits within `radius` Hamming distance. A post-filter ("Belichtung") verifies individual axis distances against the component fingerprints to reject false positives from the OR-packed bitmap.

**Gated queries** add truth filtering:
```rust
pub fn query_forward_gated(&self, subject: &Fingerprint, predicate: &Fingerprint, radius: u32, gate: TruthGate) -> Vec<SpoHit>;
```

**Chain traversal** using HammingMin semiring:
```rust
pub fn walk_chain_forward(&self, start_subject: &Fingerprint, radius: u32, max_hops: usize) -> Vec<TraversalHop>;
```
Greedy walk: at each hop, find the nearest forward match, use its object as the next subject. Accumulates cumulative Hamming distance via `saturating_add`. Returns when no match found within radius or `max_hops` reached.

### Semiring (`semiring.rs`, 99 lines, 4 tests)

```rust
pub trait SpoSemiring {
    type Cost: Copy + Ord + Default;
    fn one(&self) -> Self::Cost;                           // identity for ⊗ extend
    fn zero(&self) -> Self::Cost;                          // identity for ⊕ combine (= infinity)
    fn combine(&self, a: Self::Cost, b: Self::Cost) -> Self::Cost;  // ⊕ (min)
    fn extend(&self, path: Self::Cost, hop: Self::Cost) -> Self::Cost; // ⊗ (saturating add)
}
```

`HammingMin` implements min-plus tropical semiring: ⊕ = min, ⊗ = saturating_add. This is the standard shortest-path semiring.

`TraversalHop` records each hop: `{ target_key, distance, truth, cumulative_distance }`.

### Merkle integrity (`merkle.rs`, 246 lines, 5 tests)

```rust
pub struct MerkleRoot(u64);  // XOR-fold hash of fingerprint
pub struct ClamPath { path: String, depth: u32 }  // colon-separated DN path
pub struct BindSpace { nodes: Vec<BindNode> }  // write-addressed node store
```

- `MerkleRoot::from_fingerprint(fp)` — XOR-folds all 8 fingerprint words into a single u64, then applies a secondary mix
- `BindSpace::write_dn_path(path, fp, depth)` — writes a node, stamps its MerkleRoot at write time, returns its address
- `BindSpace::verify_integrity(addr)` — re-computes hash from stored fingerprint, compares to stamped root

**Documented gap:** `verify_lineage(addr)` performs structural checks only (non-zero root, non-zero fingerprint) — it does NOT re-hash. This is intentional for now and tested explicitly (test 6 validates that `verify_lineage` misses corruption while `verify_integrity` catches it).

### Integration tests (`spo_ground_truth.rs`, 354 lines)

7 end-to-end tests modelling a RedisGraph-style social graph:

| # | Test | What it validates |
|---|---|---|
| 1 | `spo_hydration_round_trip` | Insert Jan-KNOWS-Ada → forward query finds Ada (distance < 50), reverse finds Jan |
| 2 | `projection_verbs_consistency` | SxP2O, SxO2P, PxO2S all find the same Jan-CREATES-Ada triple |
| 3 | `truth_gate_filters_low_confidence` | OPEN → 2 hits, STRONG → 1 hit (rejects low-confidence), CERTAIN → 0 hits |
| 4 | `belichtung_rejection_rate` | 100 random edges, tight radius → <10 hits (Belichtung post-filter works) |
| 5 | `semiring_walk_chain` | A→B→C→D chain, 3 hops, cumulative_distance non-decreasing |
| 6 | `clam_merkle_integrity` | verify_lineage misses corruption (documented gap), verify_integrity catches it |
| 7 | `cypher_vs_projection_convergence` | SPO projection path validated; Cypher convergence noted as future work |

---

## 4. CI Changes

**`style.yml`** — Added `Clippy lance-graph-python` step:
```yaml
- name: Clippy lance-graph-python
  run: cargo clippy --manifest-path crates/lance-graph-python/Cargo.toml --all-targets -- -D warnings
```

**`build.yml`** — Added `Build lance-graph-python` step:
```yaml
- name: Build lance-graph-python
  run: cargo check --manifest-path crates/lance-graph-python/Cargo.toml
```

Both crates now have CI coverage for compilation and lint.

---

## Module dependency graph

```
lance-graph (lib)
├── cypher/          # existing Cypher query engine (unchanged)
├── sql_query.rs     # DataFusion SQL execution (1 line changed: as_arrow())
├── table_readers.rs # Table readers (1 line changed: allow(unused_mut))
└── graph/           # NEW — graph primitives
    ├── mod.rs           # ContainerGeometry enum (Spo = 6)
    ├── fingerprint.rs   # Fingerprint = [u64; 8], label_fp, dn_hash, hamming_distance
    ├── sparse.rs        # Bitmap = [u64; 8], pack_axes, bitwise ops
    ├── blasgraph/       # GraphBLAS semiring algebra over 16384-bit vectors
    │   ├── types.rs     # BitVec, HdrScalar, operator enums
    │   ├── semiring.rs  # 7 semirings (XorBundle, HammingMin, Boolean, ...)
    │   ├── sparse.rs    # CooStorage, CsrStorage, SparseVec
    │   ├── matrix.rs    # GrBMatrix (mxm, mxv, ewise_add, reduce, ...)
    │   ├── vector.rs    # GrBVector (k-NN, range search, ewise ops)
    │   ├── descriptor.rs# Operation descriptors (transpose, mask, replace)
    │   └── ops.rs       # BFS, SSSP, PageRank
    └── spo/             # SPO triple store with bitmap ANN
        ├── truth.rs     # TruthValue, TruthGate (NARS-inspired)
        ├── builder.rs   # SpoRecord, query vector construction
        ├── store.rs     # SpoStore (HashMap + bitmap scan + chain walk)
        ├── semiring.rs  # HammingMin (min-plus tropical)
        └── merkle.rs    # MerkleRoot, ClamPath, BindSpace
```

## Test plan

- [x] `cargo test --manifest-path crates/lance-graph/Cargo.toml --lib` — 423 tests pass
- [x] `cargo test --manifest-path crates/lance-graph/Cargo.toml --test spo_ground_truth` — 7 integration tests pass
- [x] `cargo clippy --manifest-path crates/lance-graph/Cargo.toml --all-targets -- -D warnings` — clean
- [x] `cargo clippy --manifest-path crates/lance-graph-python/Cargo.toml --all-targets -- -D warnings` — clean
- [x] `cargo check --manifest-path crates/lance-graph-python/Cargo.toml` — compiles (pyarrow zero-copy intact)
- [x] `cargo fmt --manifest-path crates/lance-graph/Cargo.toml -- --check` — clean
- [ ] CI: build.yml, style.yml, rust-test.yml (needs CI run)
