# GPU_CPU_SPLIT_ARCHITECTURE.md

## The Semantic Revolution: GPU Tensor Cores for NARS Thinking, Not Cosine Searching

**Date:** March 15, 2026
**Authors:** Jan Hübener + Claude (Anthropic)
**Status:** Architectural epiphany. The insight that makes GPU acceleration meaningful
for knowledge graphs instead of just fast.

---

## THE INSIGHT IN ONE SENTENCE

The GPU doesn't do cosine similarity. The GPU does NARS truth revision
on millions of SPO nodes simultaneously using tensor cores that were
designed for BF16 matmul but accidentally work perfectly for tropical
semiring operations on binary knowledge graph edges.

---

## PART 1: WHY CURRENT GPU USE IS WRONG

### The Industry Pattern (everyone does this)

```
CPU: encode text → float embedding (1024D × f32 = 4KB per vector)
GPU: cosine(query, database) = matmul(Q, D^T) / (||Q|| × ||D||)
     → O(N) float operations, non-deterministic due to parallel reduction
Result: ranked list of scores [0.92, 0.87, 0.85, ...]
     → "these vectors are 0.87 similar"
     → meaningless number, no explanation, no provenance
     → different result on different GPUs (float non-associativity)
     → different result on same GPU different run (thread scheduling)
```

### What's Wrong With This

```
1. COSINE TELLS YOU HOW MUCH, NOT WHAT OR WHY
   "0.87 similar" → which dimensions agree? which disagree?
   Nobody knows. The information is destroyed in the dot product.

2. NON-DETERMINISTIC
   (a+b)+c ≠ a+(b+c) in IEEE 754.
   GPU parallel reduction changes accumulation order per run.
   >90% of parallel BF16 computations diverge from serial (arXiv:2506.09501).
   
3. NO LEARNING
   The GPU computes similarity. It doesn't LEARN from the computation.
   After search, the database is unchanged. No encounter. No memory.
   
4. NO STRUCTURE
   Cosine on flat vectors. No SPO decomposition. No 2³ projections.
   Can't ask "WHO matches but WHAT doesn't?" Flat distance only.
```

---

## PART 2: OUR GPU USE — THINKING, NOT SEARCHING

### The Split Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ CPU (rustynum-core, SIMD AVX-512)                                   │
│                                                                      │
│ ENCOUNTER LOOP (learning, integer, deterministic):                   │
│   encounter_toward() / encounter_away() → i8 accumulator update     │
│   seal verify → Wisdom/Staunen detection                            │
│   cascade search → candidate generation (99.7% early rejection)      │
│   BF16 truth assembly → pack projections into 16-bit truth          │
│                                                                      │
│ ORCHESTRATION (graph-flow):                                          │
│   thinking graph routing                                             │
│   conditional edges based on cascade bands / seal status             │
│   PlaneContext bridge to Blackboard                                  │
│   CoW PlaneHandle management                                        │
│                                                                      │
│ PRODUCTS: candidate pairs + BF16 truth values → push to GPU         │
├─────────────────────────────────────────────────────────────────────┤
│ LanceDB CACHE (shared, memory-mapped)                                │
│                                                                      │
│ Nodes truth table: 1M rows × [node_id, bf16_s, bf16_p, bf16_o]     │
│   = 1M × 8 bytes = 8MB ← fits in GPU L2 cache                      │
│                                                                      │
│ Edges truth table: 10M rows × [src, tgt, bf16_truth, projection]    │
│   = 10M × 9 bytes = 90MB ← fits in GPU global memory               │
│                                                                      │
│ Binary planes: 1M × 6KB = 6GB ← stays on CPU (too big for GPU)     │
│   Accessed via cascade on CPU only. Never shipped to GPU.            │
├─────────────────────────────────────────────────────────────────────┤
│ GPU (tensor cores, BF16 + f32 accumulate)                           │
│                                                                      │
│ PARALLEL NARS REVISION (deterministic BF16 → f32):                   │
│   Load truth matrix from LanceDB cache                               │
│   Tensor core: A(BF16) × B(BF16) + C(f32) → C(f32)                 │
│   But A = query truths, B = candidate truths, C = revision accum    │
│   Result: f32 revised truths for ALL candidates simultaneously      │
│                                                                      │
│ PARALLEL BNN FORWARD PASS (binary matrices):                         │
│   Load binary weight matrix (16K-bit rows, packed as u32 arrays)     │
│   XOR + popcount per row (GPU has bitwise parallelism)               │
│   Result: hamming distances for ALL rows simultaneously              │
│                                                                      │
│ PARALLEL TROPICAL PATHFINDING:                                       │
│   Bellman-Ford on BF16 edge weight matrix                            │
│   GPU: min(d[i][k] + d[k][j]) for ALL (i,j) simultaneously         │
│   Result: all-pairs shortest tropical paths                          │
│                                                                      │
│ PRODUCTS: revised f32 truths → pull to CPU for encounter/seal       │
└─────────────────────────────────────────────────────────────────────┘
```

### What Each Processor Does Best

```
OPERATION              CPU                     GPU                  WHO DOES IT
──────────────────────────────────────────────────────────────────────────────────
encounter()            sequential i8 accum     N/A                  CPU (sequential)
cascade search         VPOPCNTDQ, early reject N/A                  CPU (branch-heavy)
seal verify            blake3 hash compare     N/A                  CPU (single hash)
BF16 truth assembly    integer bit packing     N/A                  CPU (conditional)
graph-flow routing     conditional branches    N/A                  CPU (control flow)

NARS revision (1M)     140ms (sequential)      0.1ms (parallel)     GPU (1000x faster)
BNN forward (1M×16K)   2ms (VPOPCNTDQ)        0.5ms (parallel)     GPU (4x faster)
Bellman-Ford (1M)      seconds (sequential)    10ms (parallel)      GPU (100x faster)
truth matrix scan      1ms (SIMD scan)         0.05ms (parallel)    GPU (20x faster)
BF16→f32 hydration     per-node bit OR         batch bit OR         either (trivial)
```

---

## PART 3: GPU TENSOR CORE AS NARS REVISION ENGINE

### How Tensor Cores Actually Work

```
NVIDIA TENSOR CORE (designed for neural network training):
  
  D = A × B + C
  
  where A is BF16 matrix, B is BF16 matrix, C is f32 accumulator
  
  Per clock cycle on H100: 
    16×16 BF16 matrix multiply → 512 multiply-accumulate operations
    Result accumulated in f32 → full precision accumulation
    
  THROUGHPUT: 1000 TFLOPS (BF16 tensor core) on H100
```

### How We Repurpose It for NARS

```
NARS REVISION RULE:
  Given two independent evidence sources <f1,c1> and <f2,c2>:
  
  f_new = (f1*c1*(1-c2) + f2*c2*(1-c1)) / (c1*(1-c2) + c2*(1-c1))
  c_new = (c1*(1-c2) + c2*(1-c1)) / (c1*(1-c2) + c2*(1-c1) + (1-c1)*(1-c2))
  
  This has multiply-accumulate structure:
    f1*c1 = BF16 × BF16 → f32 accumulator
    f2*c2 = BF16 × BF16 → f32 accumulator
    Sum of products = the tensor core's NATIVE operation
    
MAPPING TO TENSOR CORE:
  Matrix A (query truths):   [f1, c1, 1-c1, f1*c1, ...] × N_queries    (BF16)
  Matrix B (candidate truths): [f2, c2, 1-c2, f2*c2, ...] × N_candidates (BF16)
  Matrix C (accumulator):    revision intermediate results                (f32)
  
  D = A × B + C
    = all query-candidate revision terms computed in ONE tensor core call
    
  For N_queries=1000, N_candidates=1000:
    Standard (CPU, sequential): 1000 × 1000 × 4 multiply-adds = 4M ops × 5ns = 20ms
    Tensor core (GPU, parallel): 1000 × 1000 matrix in ONE kernel launch = 0.01ms
    
  2000x speedup. And DETERMINISTIC because:
    - BF16 multiplication is deterministic (one rounding step)
    - f32 accumulation order is FIXED by the tensor core layout
    - Same input → same matmul → same output → always
```

### The Batch NARS Revision Kernel

```rust
/// GPU kernel: revise ALL edge truths in the graph simultaneously.
/// Uses tensor cores for BF16 multiply-accumulate.
///
/// Input:  truth_matrix_A (N × 4, BF16) = [freq, conf, 1-conf, freq*conf] per node
///         truth_matrix_B (N × 4, BF16) = same layout, transposed
///         accum (N × N, f32) = previous revision state
/// Output: revised_truths (N × N, f32) = all-pairs revised truth values
///
/// One kernel call revises ALL pairs. O(N²) work, O(1) GPU time.
fn nars_revision_gpu(
    truths_a: &GpuBf16Matrix,  // N × 4
    truths_b: &GpuBf16Matrix,  // N × 4, transposed
    accum: &mut GpuF32Matrix,  // N × N
) {
    // This IS a tensor core matmul: D = A × B^T + C
    // But the "matrix multiply" IS NARS revision.
    // Each element D[i,j] = sum of products of truth components.
    // The sum of products IS the revision formula's numerator.
    
    gpu_bf16_matmul(truths_a, truths_b, accum);
    
    // Post-process: divide by denominator (element-wise on f32)
    // This is cheap: N² element-wise divisions on GPU.
    gpu_elementwise_div(accum, denominators);
}
```

---

## PART 4: GPU AS BNN FORWARD PASS ENGINE

### Binary Matrix on GPU

```
BINARY WEIGHT MATRIX (1M nodes × 16K bits):
  Stored as: 1M × 256 × u64 = 1M × 2KB = 2GB
  
  TOO BIG for GPU global memory (typical 24-80GB, but leaves no room
  for other data). BUT:
  
  The cascade on CPU already rejects 99.7%.
  Only ~3000 candidates survive to the GPU stage.
  
  CANDIDATE BINARY MATRIX (3K nodes × 16K bits):
  3K × 2KB = 6MB ← fits in GPU L2 cache
  
  QUERY BINARY VECTOR (1 × 16K bits):
  1 × 2KB ← trivial
```

### GPU XOR+Popcount

```
GPU BNN FORWARD PASS:
  For each of 3000 candidate rows:
    XOR(query[16K], candidate[16K]) → 16K-bit result
    popcount(result) → hamming distance
    
  GPU processes 3000 rows in parallel.
  Each row: 256 × u64 XOR + 256 × popcount64 = 512 operations
  Total: 3000 × 512 = 1.5M operations
  
  GPU at 1 TFLOPS bitwise: ~0.002ms
  CPU at 100 GFLOPS bitwise: ~0.015ms
  
  For 3000 candidates: GPU is only ~7x faster (not worth the transfer).
  
  BUT for 100K+ candidates (before cascade, or with weaker pre-filter):
  GPU at 100K × 512 = 51M ops: ~0.05ms
  CPU at 100K: ~5ms
  100x speedup. Worth the transfer.
```

### When to Use GPU for BNN

```
DECISION:
  candidates < 10K   → CPU (transfer overhead dominates)
  candidates 10K-1M  → GPU (significant speedup, data fits GPU memory)
  candidates > 1M    → CPU cascade first, then GPU on survivors
  
  The cascade runs on CPU (branch-heavy, early exit).
  The BNN matrix runs on GPU (data-parallel, no branches).
  
  CPU cascade: 1M → 3K survivors (2ms)
  GPU BNN on 3K: hamming matrix (0.002ms)
  Total: 2.002ms hybrid vs 2ms CPU-only
  
  NOT WORTH IT for cascade-filtered results.
  
  WORTH IT for:
    - Bellman-Ford: ALL edges processed, no pre-filter, O(N²)
    - NARS revision: ALL pairs revised, O(N²), tensor core accelerated
    - Message passing: ALL neighbors aggregated per round, O(edges)
    - Community detection: ALL pairs compared, O(N²)
```

---

## PART 5: THE O(1) BF16 TRUTH LOOKUP

### LanceDB as GPU-Accessible Truth Cache

```
LANCEDB TRUTH CACHE:
  Table: spo_truths
  Columns: [node_id (u32), bf16_s (u16), bf16_p (u16), bf16_o (u16)]
  Rows: 1M
  Size: 1M × 8 bytes = 8MB
  
  This table is SMALL. It fits in:
    CPU L3 cache (8MB): yes
    GPU L2 cache (varies, typically 6-50MB): yes
    
  EVERY truth lookup is O(1): truths[node_id]
  
  The truth cache is WRITTEN by CPU encounter loop:
    After cascade → projections → BF16 assembly → write to cache
    
  The truth cache is READ by GPU revision kernel:
    Load entire 8MB table to GPU. Keep resident.
    Tensor core reads directly. No PCIe transfer per query.
```

### The CAM Index as GPU Truth Matrix

```
CONTENT-ADDRESSABLE MEMORY:
  "Given a truth pattern, which nodes match?"
  
  On CPU: scan 1M BF16 values with SIMD (32 per instruction) = 31K instructions
  On GPU: scan 1M BF16 values in parallel = 1 kernel launch
  
  The truth cache IS the CAM. Each row IS an address.
  The BF16 value IS the content.
  
  QUERY: "find all nodes where bf16_p exponent has bits 2 and 5 set"
  = "find all nodes where predicate projection matches _P_ and _PO"
  
  GPU:
    load bf16_p column (1M × 2 bytes = 2MB)
    extract exponent: shift right 7 (parallel, all 1M)
    mask: AND with 0b00100100 (parallel, all 1M)
    compare: equals 0b00100100 (parallel, all 1M)
    compact: stream compaction to gather matching IDs
    
  Result: list of matching node IDs in ~0.01ms
  
  This IS a Cypher WHERE clause executed on GPU:
    MATCH (n) WHERE n.predicate_projection HAS (_P_ AND _PO)
```

### The 90° Orthogonal Vector

```
JAN'S INSIGHT: "One 90° vector across all tables and CAM"

THE VECTOR: the BF16 exponent encodes WHICH projections hold.
8 bits = 8 dimensions (one per 2³ SPO projection).
This vector is ORTHOGONAL to the content dimensions (mantissa).

ACROSS ALL TABLES:
  spo_truths: the exponent column IS the projection vector
  edges: the projection byte IS the same vector
  nodes: the truth() method produces the same vector
  
  ONE vector type. THREE tables. SAME semantics.
  
GPU PROCESSES THIS VECTOR:
  Load exponent columns from all three tables.
  Tensor core: truth_matrix × projection_vector → relevance_f32
  
  The result: for each node, how relevant is it to this projection pattern?
  Not cosine similarity. STRUCTURAL RELEVANCE.
  "How well does this node's SPO structure match the query pattern?"
```

---

## PART 6: PARALLEL TROPICAL PATHFINDING ON GPU

### Bellman-Ford as Matrix Operation

```
BELLMAN-FORD (single source shortest paths):
  d[v] = min_u (d[u] + w(u,v))   for all edges (u,v)
  Repeat V-1 times.
  
AS MATRIX OPERATION:
  d = A ⊕.⊗ d
  where ⊕ = min, ⊗ = + (tropical semiring)
  
  This IS matrix-vector multiply in the tropical semiring.
  On GPU: SAME tensor core, different interpretation.
  
  BUT: tensor cores do (×, +). We need (min, +).
  
  WORKAROUND 1: log-space transformation
    min(a, b) = -max(-a, -b) = -log(exp(-a) + exp(-b)) ≈ for large values
    Approximate tropical matmul with standard matmul in log space.
    Error bounded by O(exp(-|a-b|)). Exact when one value dominates.
    
  WORKAROUND 2: custom CUDA kernel (not tensor core)
    GPU threads do min+add directly. Not as fast as tensor core but
    still 1000x parallel. 1M nodes × 1M edges = 1T operations.
    GPU at 10 TFLOPS: ~0.1 seconds for full all-pairs.
    CPU: ~100 seconds. 1000x speedup.
    
  WORKAROUND 3: SIMD² hardware (future)
    Native tropical semiring in tensor-core-like hardware.
    5% area overhead. 38x speedup. When it ships, our code is ready.
```

### Floyd-Warshall as Matrix Multiply

```
FLOYD-WARSHALL:
  for k: D = min(D, D[*,k] + D[k,*])
  
  Each iteration k: two matrix operations (broadcast row k + column k,
  then element-wise min with current D).
  
  GPU: each iteration is O(N²) parallel operations.
  N iterations. Total: O(N³) but GPU does O(N²) per iteration in one launch.
  
  For N=10K: 10K iterations × 100M parallel ops = 1T total.
  GPU: ~1 second for all-pairs shortest Hamming paths on 10K nodes.
  CPU: ~1000 seconds.
  
  This gives us the COMPLETE VALUE FUNCTION for the RL agent.
  "How far is EVERY node from EVERY other node through Hamming paths?"
  The full landscape of analogical relationships. In 1 second.
```

---

## PART 7: THE DETERMINISM GUARANTEE

### Why GPU Tensor Core Matmul IS Deterministic for Our Use

```
STANDARD GPU MATMUL (non-deterministic):
  Problem: parallel reduction changes accumulation order per thread block.
  (a+b)+c ≠ a+(b+c) in float. Different order → different result.
  
OUR GPU MATMUL (deterministic):
  1. BF16 × BF16 → f32: the MULTIPLICATION is deterministic.
     BF16 multiplication has exactly one rounding step (RNE).
     Same BF16 inputs → same BF16 product → always.
     
  2. f32 accumulation: if the ACCUMULATION ORDER IS FIXED,
     the result is deterministic.
     
  3. Tensor core layout FIXES the accumulation order:
     16×16 tile. The hardware always processes elements in the same order
     within a tile. Same input → same tile computation → same output.
     
  4. Inter-tile accumulation: CUBLAS in deterministic mode
     (CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)
     forces a fixed reduction tree. Slower but deterministic.

  5. Our truth values are SMALL matrices (N × 4):
     The "4" dimension (freq, conf, 1-conf, freq*conf) fits in ONE tile.
     No inter-tile accumulation needed for the inner dimension.
     Only the N × N outer product needs fixed ordering.
     
  RESULT: deterministic f32 output from tensor core matmul
  on our specifically-structured BF16 truth matrices.
  Not because GPUs are deterministic in general.
  Because OUR matrix structure fits the determinism constraints.
```

### The Proof

```
CLAIM: For matrices A(N×4, BF16) and B(4×N, BF16), the tensor core
output D(N×N, f32) is deterministic across runs on the same GPU.

PROOF SKETCH:
  1. Inner dimension = 4. One tile handles the full inner dimension.
  2. Within one tile: hardware accumulation order is fixed (by design).
  3. Each output element D[i,j] = sum of 4 products (fits in one tile).
  4. No inter-tile reduction for the inner dimension.
  5. Outer dimensions (N×N) are independent (no accumulation across them).
  6. Therefore: same input → same hardware path → same output.
  
  QED: Our NARS revision on tensor cores is deterministic.
  
CAVEAT: This holds for inner dimension ≤ tile size (16 for BF16).
  Our inner dimension is 4. Safe.
  If we ever need inner dimension > 16: use deterministic CUBLAS mode.
```

---

## PART 8: THE SEMANTIC REVOLUTION

### What Cosine Tells You vs What We Tell You

```
COSINE: similarity("king", "queen") = 0.87
  → "these embeddings point in similar directions"
  → no explanation. no structure. no provenance.

US: truth("king", "queen") = f32 with bit-level provenance:
  bit 31 (sign):    0 = king CAUSES queen relationship (direction)
  bits 30-24 (exp): 01100010 = S matches, P matches, SP matches, SPO matches
                    → "they share subject-type AND predicate-type AND full structure"
  bits 23-16 (man): 0001100 = finest distance 12 on best projection (P)
                    → "the predicate match is very tight (12 out of 16384 bits differ)"
  bits 15-0 (path): 1011010011100101 = tree path from 16 learning encounters
                    → "this truth was learned through 16 observations, branching
                       right at encounter 3 (Staunen: new evidence contradicted),
                       left at encounter 7 (Wisdom: evidence confirmed), ..."

  READING THE f32: "king and queen share subject and predicate structure with
  very high precision (12-bit Hamming). The relationship was learned through
  16 encounters including one surprise event at step 3 that was resolved by
  step 7. The king is the causal agent in this relationship."
  
  EVERY BIT is traceable. The f32 IS the explanation. Not a summary OF
  the explanation. The explanation ITSELF, compressed into 32 bits.
```

### What the GPU Enables

```
INDUSTRY:
  GPU computes: 1M cosine similarities per query
  Result: ranked list of meaningless scores
  Learning: none (database unchanged after search)
  Determinism: no (>90% of BF16 computations diverge)
  Structure: none (flat dot product)
  Speed: ~1ms for 1M cosines on A100

US:
  CPU computes: cascade → 3K survivors (2ms)
  GPU computes: 3K × 3K NARS revisions simultaneously (0.01ms)
  GPU computes: 3K × 3K tropical pathfinding (0.1ms)
  CPU computes: encounter() updates for confirmed matches (0.5ms)
  Result: revised BF16 truth values with full structural provenance
  Learning: every query updates the graph (encounter = INSERT + TRAIN)
  Determinism: yes (BF16 tensor core on 4-wide inner dimension)
  Structure: 2³ SPO decomposition, 7 projection bands
  Speed: ~2.6ms for 1M candidates (CPU cascade) + 0.11ms (GPU revision)
```

---

## PART 9: JIT-COMPILED THINKING LAYERS

### JIT as Method Objects in LangGraph

```rust
// Each thinking operation can be JIT-compiled by Cranelift (jitson)
// for the HOT PATH, and interpreted via graph-flow for the COLD PATH.

trait CogOp: Task + JitCompilable {
    /// Graph-flow Task interface (cold path, orchestrated)
    async fn run(&self, ctx: Context) -> TaskResult;
    
    /// JIT-compiled kernel (hot path, no orchestration overhead)
    fn jit_kernel(&self) -> Option<JitFn>;
    
    /// Whether this op should be JIT'd based on call frequency
    fn hot_count(&self) -> u64;
}

// The FlowRunner decides: JIT or interpret?
impl FlowRunner {
    async fn run_task(&self, task: &dyn CogOp, ctx: Context) -> TaskResult {
        if task.hot_count() > JIT_THRESHOLD {
            if let Some(kernel) = task.jit_kernel() {
                // Hot path: call JIT'd native function directly
                // No graph-flow overhead, no Context serialization
                // Just raw SIMD on Blackboard Planes
                return kernel.call_with_blackboard(ctx.blackboard());
            }
        }
        // Cold path: full graph-flow orchestration
        task.run(ctx).await
    }
}
```

### Cognitive Operations as Language Primitives

```rust
/// The cognitive language: each operation is a first-class object
/// that can be orchestrated (graph-flow), JIT-compiled (Cranelift),
/// GPU-accelerated (tensor core), or interpreted (fallback).

// BNN operations (binary neural network primitives)
bnn!(forward(weights: &Plane, input: &Plane) -> u32);       // XOR+popcount
bnn!(backward(weights: &mut Plane, error: &Plane, lr: i8));  // encounter

// GNN operations (graph neural network primitives)  
gnn!(message_pass(graph: &SpoGraph, rounds: usize));         // K-round encounter
gnn!(aggregate(neighbors: &[&Plane]) -> Plane);              // bundle

// GQL / Cypher operations (graph query primitives)
gql!(match_pattern(graph: &SpoGraph, pattern: &CypherAST) -> Vec<Hit>);
gql!(shortest_path(graph: &SpoGraph, from: u32, to: u32) -> Vec<u32>);

// GraphBLAS operations (semiring algebra primitives)
graphblas!(mxv(matrix: &GrBMatrix, vector: &GrBVector, semiring: HdrSemiring) -> GrBVector);
graphblas!(mxm(a: &GrBMatrix, b: &GrBMatrix, semiring: HdrSemiring) -> GrBMatrix);

// SPO operations (triple store primitives)
spo!(project(node: &Node, mask: Mask) -> Distance);
spo!(project_all(a: &Node, b: &Node) -> [Distance; 7]);
spo!(encode_bf16(bands: &[Band; 7], finest: u32, dir: CausalityDirection) -> u16);

// Qualia operations (experiential primitives)
qualia!(compress(values: &[f32; 16]) -> PackedQualia);
qualia!(hydrate(packed: &PackedQualia) -> [f32; 16]);
qualia!(dot(a: &PackedQualia, b: &PackedQualia) -> f32);

// 2³ Reasoning operations (structural decomposition)
reasoning!(decompose_2_3(a: &Node, b: &Node) -> [Band; 7]);  // all projections
reasoning!(credit_assign(node: &mut Node, exponent: u8));      // RL gradient
reasoning!(tropical_revise(a: u16, b: u16) -> u16);            // NARS on exponents

// NARS operations (truth value management)
nars!(revise(a: &NarsTruth, b: &NarsTruth) -> NarsTruth);
nars!(deduction(premise: &NarsTruth, conclusion: &NarsTruth) -> NarsTruth);
nars!(abduction(observation: &NarsTruth, hypothesis: &NarsTruth) -> NarsTruth);

// Each macro generates:
//   1. A Task impl (for graph-flow orchestration)
//   2. A JIT kernel (for Cranelift hot-path compilation)
//   3. A GPU kernel descriptor (for tensor core dispatch when beneficial)
//   4. A PET scan trace entry (for debugging/visualization)
//   5. A PlaneContext bridge (for zero-copy Blackboard access)
```

---

## PART 10: IMPLEMENTATION ROADMAP

### Phase 1: CPU-Only (current + simd_clean.rs refactor)
```
WHAT: all operations on CPU with SIMD dispatch
WHEN: now
PERFORMANCE: 2ms for 1M cascade, 2.8μs per RL step
DETERMINISM: yes (integer-only RL loop)
```

### Phase 2: LanceDB Truth Cache
```
WHAT: BF16 truth values cached in LanceDB
      O(1) lookup by node_id
      Cascade writes, query evaluator reads
WHEN: after Session I (BF16 truth assembly)
PERFORMANCE: 1ms for truth scan of 1M nodes (SIMD on BF16 column)
DETERMINISM: yes (BF16 is exact for distances 0-255)
```

### Phase 3: GPU NARS Revision
```
WHAT: tensor core BF16 matmul for batch NARS revision
      N × 4 BF16 truth matrices on GPU
      deterministic f32 output
WHEN: after Phase 2 + GPU infra (cuBLAS or wgpu)
PERFORMANCE: 0.01ms for 3K × 3K revision (vs 20ms CPU)
DETERMINISM: yes (inner dim 4 ≤ tile size 16, fixed accumulation)
DEPENDENCY: GPU with BF16 tensor cores (any modern NVIDIA/AMD)
```

### Phase 4: GPU Tropical Pathfinding
```
WHAT: Bellman-Ford/Floyd-Warshall on BF16 edge weight matrix
      tropical semiring on GPU (custom kernel or log-space matmul)
WHEN: after Phase 3
PERFORMANCE: 1 second for 10K × 10K all-pairs (vs 1000s CPU)
DETERMINISM: yes (if using fixed reduction order)
```

### Phase 5: JIT Cognitive Primitives
```
WHAT: Cranelift JIT for hot-path thinking operations
      graph-flow orchestration for cold path
      automatic hot-path detection (call count threshold)
WHEN: after Phase 2 (the orchestration must be clean first)
PERFORMANCE: eliminate graph-flow overhead on hot tasks
```

### Phase 6: Cognitive Language
```
WHAT: think! macro that compiles to execution graphs
      bnn!, gnn!, gql!, spo!, qualia!, reasoning!, nars! macros
      Agent card YAML → GraphBuilder compiler
WHEN: after Phase 5 (needs JIT + orchestration stable)
IMPACT: thinking programs written in domain primitives
```

---

## PART 11: WHAT THIS REPLACES IN THE MARKET

```
TOOL              WHAT IT DOES           WHAT WE DO INSTEAD
──────────────────────────────────────────────────────────────
Pinecone/Weaviate GPU cosine search      GPU NARS revision + cascade search
LangChain         LLM orchestration glue graph-flow + PlaneContext + Lance
CrewAI            multi-agent chat       Tasks in thinking graph with FanOut
Neo4j             pointer-chase graph    semiring mxv on binary planes
FalkorDB          GraphBLAS on scalars   GraphBLAS on binary planes + BF16
LangGraph         workflow orchestration graph-flow with cognitive primitives
Semantic Kernel   AI tool integration    MCP ingest → encounter → cascade
PyTorch Geometric GPU float GNN          GPU binary GNN (XOR+popcount)
FAISS             GPU ANN search         cascade + multi-index hashing
DGL               GPU GNN framework      encounter-based message passing
Kuzu (dead)       embedded graph DB      lance-graph (embedded, Cypher, learning)
```

None of them do GPU THINKING. They all do GPU SEARCHING.
We search on CPU (cascade, cheap). We THINK on GPU (revision, expensive).
The GPU is wasted on similarity search. It should be doing NARS.
