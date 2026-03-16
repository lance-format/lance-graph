# INVENTORY_MAP.md

## Complete Inventory: What Exists, What's Missing, What Connects

**Date:** March 15, 2026
**Repos:** rustynum (AdaWorldAPI/rustynum), lance-graph (AdaWorldAPI/lance-graph)

---

## 1. RUSTYNUM-CORE: The Muscle Layer

### 1.1 Binary Plane Substrate (IMPLEMENTED)

```
FILE: rustynum-core/src/plane.rs
─────────────────────────────────
TYPE: Plane
  acc: Box<Acc16K>           # i8[16384] — the ONLY stored state
  bits: Fingerprint<256>     # cached sign(acc) — 2KB
  alpha: Fingerprint<256>    # cached |acc| > threshold — 2KB
  dirty: bool                # lazy cache invalidation
  encounters: u32            # how many observations shaped this

METHODS:
  new()                      → empty Plane, maximum uncertainty
  encounters() → u32         → observation count
  bits() → &Fingerprint<256> → data bits (lazy from acc)
  alpha() → &Fingerprint<256>→ defined mask (lazy from acc)
  encounter_bits(&mut, evidence: &Fingerprint<256>)
                             → INTEGER accumulate evidence into acc
                             → marks dirty, increments encounters
  encounter(&mut, text: &str)→ hash text to fingerprint, then encounter_bits
  distance(&mut, other: &mut) → Distance enum
  truth(&mut) → Truth        → {defined, agreed, total, frequency_f32, confidence_f32}
  merkle(&mut) → MerkleRoot  → blake3 of (bits & alpha), 48-bit truncated
  verify(&mut, stored: &MerkleRoot) → Seal (Wisdom | Staunen)

CONSTANTS:
  PLANE_BITS = 16384
  PLANE_BYTES = 2048

CONNECTION POINTS:
  → encounter_bits IS the DreamerV3 STE gradient (integer form)
  → distance IS the BNN forward pass (XOR + popcount)
  → truth IS the NARS truth value (frequency + confidence)
  → merkle/verify IS the Seal integrity system
  → acc IS the BNN weight vector (i8 saturating arithmetic)

WHAT'S MISSING:
  ✗ encounter_toward(&mut, other: &Plane)  — encounter toward another plane's bits
  ✗ encounter_away(&mut, other: &Plane)    — encounter AGAINST another plane's bits
  ✗ reward_encounter(&mut, evidence, reward_sign: i8) — DreamerV3-style RL gradient
  ✗ BF16 cache of last computed distance
```

### 1.2 Node: Three Planes as SPO (IMPLEMENTED)

```
FILE: rustynum-core/src/node.rs
────────────────────────────────
TYPE: Node { s: Plane, p: Plane, o: Plane }

TYPE: Mask { s: bool, p: bool, o: bool }
  Constants: SPO, SP_, S_O, _PO, S__, _P_, __O, ___

METHODS:
  new() → empty Node
  random(seed: u64) → deterministic random Node
  distance(&mut, other: &mut, mask: Mask) → Distance
  truth(&mut, mask: Mask) → Truth

CONNECTION POINTS:
  → Mask constants ARE the 2³ decomposition (8 projections)
  → distance(mask=S__) IS projection "WHO matches?"
  → distance(mask=_P_) IS projection "WHAT matches?"
  → All 7 non-null masks → 7 distances → BF16 exponent bits

WHAT'S MISSING:
  ✗ project_all(&mut, other: &mut) → [Distance; 7]  — all 7 projections at once
  ✗ bf16_truth(&mut, other: &mut) → u16             — pack projections into BF16
  ✗ encounter_toward(&mut, other: &Node, mask: Mask) — RL credit assignment per projection
  ✗ edges: Vec<Edge>  — adjacency list (Neo4j-style warm path)
```

### 1.3 Seal: Integrity Verification (IMPLEMENTED)

```
FILE: rustynum-core/src/seal.rs
────────────────────────────────
TYPE: Seal { Wisdom, Staunen }
TYPE: MerkleRoot([u8; 6])

METHODS:
  Plane::merkle() → MerkleRoot (blake3, alpha-masked)
  Plane::verify(stored) → Seal

CONNECTION POINTS:
  → Staunen event = BF16 sign bit flip (causing → caused)
  → Wisdom = BF16 sign bit stable (no causality reversal)
  → verify IS the convergence check for RL loop

WHAT'S MISSING:
  ✗ Node-level seal (composite over S/P/O plane seals)
  ✗ Seal history (sequence of Wisdom/Staunen events = tree path bits)
```

### 1.4 HDR Cascade: Multi-Resolution Search (IMPLEMENTED)

```
FILE: rustynum-core/src/hdr.rs
───────────────────────────────
TYPE: Cascade { bands, reservoir, shift_history, ... }
TYPE: Band { Foveal, Near, Good, Weak, Reject }
TYPE: RankedHit { index, distance, band }
TYPE: ShiftAlert { old_bands, new_bands, n_observations }

METHODS:
  Cascade::calibrate(distances, vec_bytes) → calibrated Cascade
  Cascade::expose(distance) → Band classification
  Cascade::observe(distance) → Option<ShiftAlert>  (distribution drift detection)
  Cascade::recalibrate(alert)
  Cascade::query(query, database, k, threshold) → Vec<RankedHit>

CONNECTION POINTS:
  → Band classification → BF16 exponent bit (Foveal/Near = 1, else = 0)
  → observe/recalibrate IS the self-organizing boundary fold (QUANTILE_HEALING)
  → query IS the hot path entry point

WHAT'S MISSING:
  ✗ Tetris strokes (non-overlapping incremental slices)
  ✗ Prefetch interleaving
  ✗ BF16 similarity cache (write on query, read on cold path)
  ✗ PackedDatabase (stroke-aligned layout for streaming)
  ✗ typed array_chunks strokes (Rust 1.94 array_windows)
```

### 1.5 BF16 Hamming (IMPLEMENTED)

```
FILE: rustynum-core/src/bf16_hamming.rs
────────────────────────────────────────
TYPES:
  BF16Weights              — per-byte weights for weighted hamming
  BF16StructuralDiff       — exponent/mantissa/sign decomposition of diff
  AwarenessState           — Crystallized/Tensioned/Uncertain/Noise
  SuperpositionState       — decomposition into awareness states
  PackedQualia             — 16 resonance dimensions + BF16 scalar
  
FUNCTIONS:
  fp32_to_bf16_bytes(floats) → Vec<u8>
  bf16_bytes_to_fp32(bytes) → Vec<f32>
  structural_diff(a, b) → BF16StructuralDiff
  pack_awareness_states / unpack_awareness_states
  superposition_decompose
  compress_to_qualia / hydrate_qualia_f32 / hydrate_qualia_bf16
  qualia_dot(a, b) → f32
  bundle_qualia(items) → PackedQualia
  invert_qualia_polarity

CONNECTION POINTS:
  → structural_diff IS the BF16 exponent extraction (integer)
  → AwarenessState maps to NARS truth (Crystallized% = frequency, 1-Noise% = confidence)
  → PackedQualia IS the qualia vector for Fibonacci encoding
  → compress_to_qualia / hydrate IS the φ-fold / φ-unfold

WHAT'S MISSING:
  ✗ bf16_from_projections(projections: [Band; 7], finest_distance: u32) → u16
  ✗ bf16_extract_exponent(bf16: u16) → u8  (integer extraction for graph ops)
  ✗ NarsTruth packed as BF16 pair (32 bits)
  ✗ VDPBF16PS-accelerated qualia_dot (currently scalar)
```

### 1.6 Causality (IMPLEMENTED)

```
FILE: rustynum-core/src/causality.rs
─────────────────────────────────────
TYPES:
  CausalityDirection { Causing, Experiencing }
  NarsTruthValue { frequency: f32, confidence: f32 }
  CausalityDecomposition { direction, strength, ... }

FUNCTIONS:
  causality_decompose(qualia_a, qualia_b) → CausalityDecomposition
  spo_encode_causal(node, direction) → SPO encoding
  spatial_nars_truth(crystal) → NarsTruthValue

CONNECTION POINTS:
  → CausalityDirection = BF16 sign bit
  → NarsTruthValue = what we pack into BF16 pair (32 bits)
  → causality_decompose uses WARMTH/SOCIAL/SACREDNESS dims = the 3 causality axes

WHAT'S MISSING:
  ✗ NarsTruth as BF16 pair type (u32 = 2×BF16)
  ✗ NARS revision using VDPBF16PS for multiply-accumulate terms
  ✗ Connection to Node's 2³ projection system
```

### 1.7 SIMD Dispatch (IMPLEMENTED but bloated)

```
FILE: rustynum-core/src/simd.rs (2435 lines, needs simd_clean.rs refactor)
FILE: rustynum-core/src/simd_avx512.rs (stable AVX-512 wrapper types)
FILE: rustynum-core/src/simd_avx2.rs (AVX2 fallback implementations)
FILE: rustynum-core/src/simd_isa.rs (Isa trait for portable_simd)
FILE: rustynum-core/src/scalar_fns.rs (scalar fallbacks)

KEY FUNCTIONS:
  simd::hamming_distance(a, b) → u64
  simd::popcount(a) → u64
  simd::dot_f32(a, b) → f32
  simd::dot_i8(a, b) → i64  (VNNI)
  simd::hamming_batch / hamming_top_k
  + 16 element-wise ops (add/sub/mul/div × f32/f64 × scalar/vec)
  + BLAS-1 (axpy, scal, asum, nrm2, iamax)

WHAT NEEDS TO CHANGE:
  → Replace simd.rs with simd_clean.rs (234 lines, LazyLock + dispatch! macro)
  → Fix simd_ops + array_struct to use simd:: not simd_avx512:: types
  → Add safe intrinsics (Rust 1.94) to remove unsafe inlining barriers
```

---

## 2. LANCE-GRAPH: The Brain Layer

### 2.1 BlasGraph Semiring Algebra (IMPLEMENTED)

```
FILE: crates/lance-graph/src/graph/blasgraph/semiring.rs
────────────────────────────────────────────────────────
TRAIT: Semiring { add, multiply, zero, one }
TYPE: HdrSemiring enum (the 7 semirings)

FILE: crates/lance-graph/src/graph/blasgraph/ops.rs
───────────────────────────────────────────────────
FUNCTIONS:
  grb_mxm(A, B, semiring)              → matrix × matrix
  grb_mxv(A, v, semiring)              → matrix × vector
  grb_vxm(v, A, semiring)              → vector × matrix
  grb_ewise_add/mult_matrix/vector     → element-wise operations
  grb_reduce_matrix/vector             → reduction (fold)
  grb_apply/extract/assign/transpose   → utility ops
  hdr_bfs(adj, source, depth)          → BFS traversal
  hdr_sssp(adj, source, iters)         → shortest path (tropical!)
  hdr_pagerank(adj, iters, damping)    → PageRank

THE 5:2 SPLIT:
  BITWISE (port 0, integer):
    XorBundle     → VPXORD
    BindFirst     → VPSHUFB/VPERMD
    HammingMin    → VPXORD + VPOPCNTDQ + VPMINSD
    Boolean       → VPORD + VPANDD
    XorField      → VPXORD + VPCLMULQDQ
  
  FLOAT (port 1, BF16):
    SimilarityMax → VDPBF16PS + VPMAXPS
    Resonance     → VDPBF16PS

WHAT'S MISSING:
  ✗ BF16 edge weight type (currently HdrScalar = integer)
  ✗ Dual pipeline dispatch (integer vs BF16 per semiring)
  ✗ Tropical pathfinding on Hamming distances (hdr_sssp uses HdrScalar)
  ✗ NarsTruth as edge attribute
  ✗ VPCLMULQDQ acceleration for XorField
```

### 2.2 BitVec: 16K-bit Binary Vector (IMPLEMENTED in lance-graph)

```
FILE: crates/lance-graph/src/graph/blasgraph/types.rs
─────────────────────────────────────────────────────
TYPE: BitVec { words: [u64; 256] }  — 16384 bits = 2KB

CONSTANTS: VECTOR_WORDS=256, VECTOR_BITS=16384

METHODS: zero(), ones(), random(seed), xor, and, or, popcount, density,
         hamming_distance, bind, bundle, threshold, ...

CONNECTION TO RUSTYNUM:
  BitVec.words ↔ Plane.bits().words()   — SAME DATA, different wrappers
  BitVec.hamming_distance ↔ simd::hamming_distance
  BitVec.xor ↔ VPXORD
  BitVec.popcount ↔ VPOPCNTDQ

WHAT'S MISSING:
  ✗ Zero-copy bridge between BitVec and Fingerprint<256>
  ✗ Plane integration (BitVec as the bits() view, alpha() as mask)
```

### 2.3 Cascade in lance-graph (IMPLEMENTED, separate from rustynum)

```
FILE: crates/lance-graph/src/graph/blasgraph/hdr.rs
───────────────────────────────────────────────────
TYPES: Band, RankedHit, ShiftAlert, ReservoirSample, Cascade

NOTE: This is a SEPARATE implementation from rustynum-core/src/hdr.rs.
      Session C (cross-pollination) was designed to merge them.
      
WHAT'S MISSING:
  ✗ Cross-pollination (Session C): port 5 algorithms from lance-graph to rustynum
  ✗ Single source of truth for Cascade (rustynum-core is the canonical)
```

### 2.4 Graph Execution Engine (IMPLEMENTED)

```
FILE: crates/lance-graph/src/graph/blasgraph/matrix.rs
FILE: crates/lance-graph/src/graph/blasgraph/sparse.rs
FILE: crates/lance-graph/src/graph/blasgraph/vector.rs
─────────────────────────────────────────────────────
GrBMatrix: sparse matrix with HdrScalar entries
GrBVector: sparse vector with HdrScalar entries
Sparse format: CSR-like with BitVec values

The full GraphBLAS-style API: mxm, mxv, vxm, reduce, apply, extract, assign
```

### 2.5 DataFusion Planner: Cypher-to-Semiring (IMPLEMENTED)

```
FILES: crates/lance-graph/src/datafusion_planner/*.rs
──────────────────────────────────────────────────────
ast.rs         → Cypher-like AST
expression.rs  → Expression evaluation
scan_ops.rs    → Table scans
join_ops.rs    → Join operations (MATCH patterns)
vector_ops.rs  → Vector similarity operations
udf.rs         → User-defined functions

THIS IS THE COLD PATH. Cypher queries → DataFusion plan → semiring operations.

WHAT'S MISSING:
  ✗ BF16 edge weight predicates (WHERE similarity > 0.8 → scan BF16 cache)
  ✗ Hot path integration (MATCH triggers cascade for discovery)
  ✗ Cost estimation using BF16 logarithmic values
```

---

## 3. WHAT'S NOT IN EITHER REPO YET

### 3.1 Wiring Between Repos

```
RUSTYNUM has: Plane, Node, Cascade, BF16, PackedQualia, NarsTruthValue
LANCE-GRAPH has: BitVec, Semirings, GrBMatrix, DataFusion planner

MISSING BRIDGE:
  ✗ lance-graph using rustynum-core's Plane instead of its own BitVec
  ✗ lance-graph using rustynum-core's Cascade instead of its own hdr.rs
  ✗ GrBMatrix entries as Plane distances (not just HdrScalar integers)
  ✗ Semiring operations calling simd:: dispatch layer
  ✗ DataFusion planner calling Cascade for similarity queries
```

### 3.2 The Deterministic f32 Pipeline

```
WHAT EXISTS:
  ✓ Plane.encounter_bits() — integer evidence accumulation
  ✓ Node.distance(mask) — per-projection hamming
  ✓ Cascade.expose() — band classification
  ✓ bf16_hamming::structural_diff() — BF16 decomposition
  ✓ causality::CausalityDirection — sign bit meaning
  ✓ causality::NarsTruthValue — frequency + confidence

WHAT'S MISSING (the wiring):
  ✗ Node.project_all() → 7 distances
  ✗ 7 distances → 7 band classifications → 8 exponent bits
  ✗ 8 exponent bits + 7 mantissa bits + 1 sign bit → BF16 value
  ✗ BF16 value → tree leaf insertion → 16 path bits
  ✗ BF16 + path bits → f32 hydration
  ✗ RL credit assignment: read exponent bits → encounter per projection
  ✗ Convergence detection: seal stable across iterations
```

### 3.3 Database Layer

```
WHAT EXISTS:
  ✓ Cascade.query() returns RankedHits from a database slice
  
WHAT'S MISSING:
  ✗ PackedDatabase (stroke-aligned for streaming, GEMM panel packing analogy)
  ✗ Edge storage (BF16 truth per edge, compact adjacency)
  ✗ Index structure (hot edges via pointers, cold discovery via cascade)
```

### 3.4 GNN / Message Passing

```
WHAT EXISTS:
  ✓ Plane.encounter_bits() — can accumulate evidence from neighbors
  ✓ Node.distance() — can compare with neighbors

WHAT'S MISSING:
  ✗ encounter_toward / encounter_away — directional encounter for RL reward
  ✗ Message passing loop (K rounds of neighbor aggregation)
  ✗ Edge-weighted encounters (BF16 confidence as encounter weight)
```

---

## 4. DEPENDENCY MAP: WHAT MUST BE BUILT IN WHAT ORDER

```
LAYER 0 (foundation, no dependencies):
  [0a] simd_clean.rs refactor (234 lines, replaces 2435)
  [0b] Plane: add encounter_toward(), encounter_away(), reward_encounter()
  [0c] Node: add project_all() → [Distance; 7]

LAYER 1 (needs Layer 0):
  [1a] BF16 truth assembly: bf16_from_projections([Band; 7], finest: u32) → u16
  [1b] BF16 exponent extraction: bf16_extract_exponent(u16) → u8
  [1c] NarsTruth as BF16 pair: struct NarsTruth(u32) with frequency/confidence as BF16
  [1d] PackedDatabase: stroke-aligned layout for streaming cascade

LAYER 2 (needs Layer 1):
  [2a] Edge type: { target: u32, truth: NarsTruth, projection: u8 }
  [2b] Node with adjacency: first_edge, edge_count
  [2c] Cascade with BF16 cache: write on query, read on cold path
  [2d] Message passing: K-round neighbor encounter aggregation

LAYER 3 (needs Layer 2):
  [3a] RL credit assignment: read BF16 exponent → selective encounter
  [3b] Tropical pathfinding: HammingMin Floyd-Warshall on semiring ops
  [3c] Cold path integration: DataFusion planner reads BF16 edge cache
  [3d] Convergence detection: seal stability across RL iterations

LAYER 4 (the unicorn, needs Layer 3):
  [4a] Tree leaf insertion with path extraction
  [4b] f32 hydration: BF16 bits OR path bits
  [4c] Full RL loop: observe → project → classify → pack → insert → hydrate → learn
  [4d] Benchmark: 1M candidates, measure end-to-end RL epoch time
```
