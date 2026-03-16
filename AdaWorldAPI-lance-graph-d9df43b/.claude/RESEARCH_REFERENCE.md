# RESEARCH_REFERENCE.md

## Prior Art: What Exists, What to Steal, What to Investigate

---

## 1. DreamerV3 — Binary RL That WORKS (Nature 2025)

**Paper:** Hafner et al., "Mastering Diverse Domains through World Models"
**URL:** https://danijar.com/project/dreamerv3/
**Key finding:** Discrete categorical latent representations OUTPERFORM continuous.
**RLC 2024 follow-up:** The advantage comes from SPARSE BINARY nature specifically.

### What They Do
```
State representation: 32 categories × 32 classes = 1024 multi-one-hot dims
Gradient trick: STE (Straight-Through Estimator)
  Forward: binary = sign(latent)
  Backward: ∂L/∂latent ≈ ∂L/∂binary × 𝟙{|latent| ≤ 1}
Training: GPU, float matmul on sparse binary codes
Hardware: NVIDIA GPU with float tensor cores
```

### What We Can Steal
```
STE gradient → encounter_toward / encounter_away (our integer equivalent)
  STE clips at |latent| ≤ 1 → our acc[k] clips at i8 saturation (-128, 127)
  STE passes gradient through sign() → our acc += ±1 through encounter
  SAME FUNCTION. Different substrate.

World model → cascade as predictor
  DreamerV3 predicts next observation from state + action
  Our cascade predicts band classification from partial strokes
  Both: early termination when confident. Both: uncertainty-aware.

Actor-critic split → encounter_toward (actor) / truth() (critic)
  Actor: choose action (encounter toward good patterns)
  Critic: evaluate state (truth = frequency/confidence)
```

### What to Investigate
```
1. Does our 16K-dim binary match or beat their 1024-dim multi-one-hot?
   Test: same environment, our Plane as state, encounter as gradient.
   
2. Does integer accumulation converge as fast as STE float?
   Test: convergence curves for identical observations.
   
3. Can our cascade serve as a world model?
   Test: given partial observation (stroke 1), predict band of full observation.

4. RLC 2024 arXiv:2312.01203 — "Harnessing Discrete Representations
   for Continual Reinforcement Learning" — specifically studies why
   binary beats continuous. Read in detail.
```

---

## 2. BitGNN — Binary Graph Neural Networks (ICS 2023)

**Paper:** "BitGNN: Unleashing the Performance Potential of Binary GNNs on GPUs"
**URL:** https://dl.acm.org/doi/10.1145/3577193.3593725
**Key finding:** 8-22x speedup using XNOR+popcount for graph convolution.

### What They Do
```
Binary weights: W ∈ {-1, +1}^(N×D) → packed as bits
Binary features: X ∈ {-1, +1}^(N×D) → packed as bits
Convolution: XNOR(W, X) + popcount = dot product approximation
Aggregation: GPU scatter_add (NON-DETERMINISTIC)
Training: STE for binary weights, float for aggregation
```

### What We Can Steal
```
Their convolution IS our hamming_distance. Same instruction. Same operation.
Their aggregation IS our encounter(). But theirs is GPU scatter_add (races).
Ours is sequential integer accumulation (deterministic).

Their binary feature matrix IS our Plane.bits() fingerprint.
Their binary weight matrix IS the pattern we encounter toward/away.

Architecture translation:
  BitGNN layer = for each node: encounter_toward(neighbor.bits) for each neighbor
  K BitGNN layers = K rounds of message passing
  Output = node.truth() = classification/regression target
```

### What to Investigate
```
1. Bi-GCN (Wang et al., 2021) — earlier binary GCN, simpler.
   Binarizes both weights AND features. May be closer to our architecture.
   
2. BitGNN's GPU optimizations — they use "bit-level optimizations"
   and warp-level parallelism. What can we learn for VPOPCNTDQ batching?
   See: https://www.sciencedirect.com/science/article/abs/pii/S0743731523000357

3. ReActNet (ECCV 2020) — 69.4% ImageNet top-1 with binary.
   Uses learned activation shifts (RSign, RPReLU).
   Our threshold in Plane IS a learned activation. Can we adapt RSign?
   URL: https://ar5iv.labs.arxiv.org/html/2003.03488
```

---

## 3. GEMM Microkernel Architecture (siboehm, 2022)

**URL:** https://siboehm.com/articles/22/Fast-MMM-on-CPU
**Key finding:** 2x from panel packing (sequential memory access).

### What They Do
```
Optimization chain:
  Naive:           4481ms
  Compiler flags:  1621ms (3x — -O3 -march=native -ffast-math)
  Loop reorder:     89ms (18x — cache-aware inner loop)
  L1 tiling:        70ms (1.3x — tile to L1 size)
  Multithreading:   16ms (4.4x — OpenMP on rows/cols)
  MKL:               8ms (2x — panel packing + register blocking + prefetch)

Panel packing: repack matrix B into column panels contiguous in memory.
Instead of strided access (jump 1024 floats per row), sequential access.
The hardware prefetcher handles sequential patterns automatically.
```

### What We Can Steal
```
EXACT PARALLEL:
  Their matrix B → our candidate database
  Their column panel → our stroke region
  Their panel packing → our PackedDatabase
  Their tiling to L1 → our 2KB plane constraint
  Their register accumulation → our ZMM hamming accumulator
  Their prefetch schedule → our _mm_prefetch per candidate
  Their thread partitioning → rayon par_chunks on candidates

The missing 2x:
  MKL packs B into contiguous column panels BEFORE the kernel runs.
  We should pack candidates into contiguous stroke regions BEFORE cascade runs.
  
  Current: candidate[i].plane_S[0..128] scattered at 6KB intervals
  Packed: stroke1_all[i*128..(i+1)*128] contiguous

  Sequential access: hardware prefetcher free. 2-3x improvement.
```

### What to Investigate
```
1. https://github.com/flame/how-to-optimize-gemm — goes further than siboehm.
   Register blocking details for AVX-512. Directly applicable to our 
   hamming inner loop: keep partial popcount in ZMM, never spill.

2. OpenBLAS kernel: github.com/xianyi/OpenBLAS/blob/develop/kernel/x86_64/sgemm_kernel_16x4_haswell.S
   7K LOC of handwritten assembly. The microkernel structure is what
   level3.rs should aspire to.

3. Apple AMX instructions: undocumented matrix-matrix ops.
   Our Isa trait could abstract over AMX when Apple documents them.
```

---

## 4. GraphBLAS + SuiteSparse (Tim Davis)

**URL:** https://github.com/DrTimothyAldenDavis/GraphBLAS
**Paper:** ACM TOMS 2019/2023
**Key finding:** ~2000 pre-compiled semiring kernels + runtime JIT for custom ones.

### What They Do
```
C API for graph algorithms as sparse matrix operations.
Semiring: (S, ⊕, ⊗, 0⊕, 1⊗) replaces (+, ×).
Three-tier dispatch:
  1. FactoryKernels: pre-compiled for common type/semiring combos
  2. JIT: compile custom semiring via system C compiler, cache result
  3. Generic: function pointer fallback

SIMD: compiler auto-vectorization (-O3 -march=native), not hand-written.
Powers: FalkorDB (RedisGraph successor), MATLAB sparse multiply since R2021a.
```

### What We Can Steal
```
Their semiring trait → our HdrSemiring enum. Already similar.
Their FactoryKernels → our hand-written SIMD per semiring.
Their JIT → not needed yet (our 7 semirings are fixed).
Their mxm/mxv/vxm → our grb_mxm/grb_mxv/grb_vxm. Already implemented.

THE INSIGHT: SuiteSparse gets performance from sparse matrix FORMAT
(CSR/CSC/COO/hypersparse) not from custom SIMD. The format determines
whether operations are sequential (cache-friendly) or scattered (slow).

For us: the PackedDatabase IS the sparse format optimization.
Stroke-aligned = CSR for cascades.
```

### What to Investigate
```
1. cuASR (CUDA Algebra for Semirings): GPU semiring GEMM at 95% of standard GEMM.
   URL: https://github.com/hpcgarage/cuASR
   Their templates for tropical/bottleneck semirings on GPU.
   Can inform our SIMD templates for bitwise semirings on CPU.

2. SuiteSparse JITPackage: embeds source code in library binary.
   Interesting for deployment: one binary, custom semirings compiled on demand.

3. pygraphblas: Python wrapper for SuiteSparse.
   URL: https://github.com/Graphegon/pygraphblas
   Their Python API design for our future lance-graph-python.
```

---

## 5. Tropical Geometry + Neural Networks (Zhang, Naitzat, Lim 2018)

**Paper:** "Tropical Geometry of Deep Neural Networks" (ICML 2018)
**URL:** https://proceedings.mlr.press/v80/zhang18i.html
**Key finding:** ReLU networks are equivalent to tropical rational maps.

### What They Prove
```
ReLU(x) = max(x, 0) = tropical max operation.
Linear layer + ReLU = tropical polynomial evaluation.
Decision boundaries = tropical hypersurfaces.

Feedforward ReLU network with L layers:
  Output = tropical rational function of input.
  The network computes a piecewise-linear function
  whose pieces are determined by the tropical geometry.

TROPICAL SEMIRING: (ℝ ∪ {∞}, min, +, ∞, 0)
  "addition" = min (take the shorter path)
  "multiplication" = + (add edge weights)
  This is Bellman-Ford / Floyd-Warshall / Dijkstra.
```

### What This Means for Us
```
Our HammingMin semiring IS the tropical semiring in Hamming space:
  ⊕ = min (take the nearest neighbor)
  ⊗ = hamming (cost of traversing an edge)

Therefore:
  hdr_sssp() IS tropical pathfinding
  hdr_bfs() IS tropical reachability (thresholded)
  hdr_pagerank() can be reformulated tropically

The Plane accumulator IS a tropical computation:
  acc[k] += evidence IS tropical addition (accumulate path lengths)
  sign(acc[k]) IS the tropical decision boundary
  |acc[k]| > threshold IS the tropical activation (ReLU equivalent)

Our ENTIRE architecture is tropical. We just didn't name it that way.
```

### What to Investigate
```
1. "Tropical Attention" (arXiv:2505.17190, 2025) — attention in max-plus semiring.
   Combinatorial algorithms with superior OOD generalization.
   Could our cascade be reformulated as tropical attention?

2. Connection to NARS: NARS revision rule uses min/max operations.
   Is NARS revision a tropical computation?
   If so: revision = tropical matrix multiply on truth values.
```

---

## 6. Neo4j Internals: Index-Free Adjacency

**Source:** "Graph Databases" (Robinson, Webber, Eifrem) + internal docs
**Key insight:** O(1) per hop via pointer chasing. But cache-hostile.

### What They Do
```
Node record (15 bytes): [4B id] [4B first_rel_ptr] [4B first_prop_ptr] [3B flags]
Relationship record: doubly-linked list per endpoint node.
Storage: fixed-size records in store files, paged into memory.

Traversal: follow first_rel_ptr → walk linked list → filter by type + direction.
Each hop: one pointer dereference. O(1). But random memory access.

For 10 hops: 10 cache misses × ~40ns DRAM = ~400ns total.
For scan of 1M edges: 1M × 40ns = 40ms. Cache-hostile.
```

### What We Can Steal
```
Fixed-size records: our Node is fixed-size (6KB + 32B header).
  BUT our Node is SIMD-scannable, not just pointer-chaseable.

Adjacency pointer: add first_edge + edge_count to Node.
  For KNOWN relationships: O(1) pointer + walk.
  For DISCOVERY: cascade scan (O(N) but with 99.7% early rejection).

The HYBRID: Neo4j for warm path, cascade for hot path.
  warm_edges: Vec<Edge>  — confirmed relationships, Neo4j-style
  cold_discovery: cascade.query() — find new similar nodes
  
  BF16 truth on edges bridges both paths.
```

### What to Investigate
```
1. Hexastore (VLDB 2008) — SPO sextuple indexing.
   6 permutation indices: SPO, SOP, PSO, POS, OSP, OPS.
   For our Mask-based queries: each mask maps to 1-2 permutation indices.
   Could inform how we index Node collections.

2. Neo4j's page cache: custom off-heap memory management.
   Our PackedDatabase is similar: pre-allocated, aligned, streaming.
   
3. FalkorDB (formerly RedisGraph): uses SuiteSparse:GraphBLAS internally.
   Cypher queries → GraphBLAS semiring operations.
   This is EXACTLY our architecture: DataFusion planner → BlasGraph semirings.
```

---

## 7. SIMD² — Future Hardware for Semiring GEMM (ISCA 2022)

**Paper:** Zhang et al., "SIMD²: A Generalized Matrix Instruction Set"
**URL:** https://arxiv.org/abs/2205.01252
**Key finding:** 5% chip area overhead, 38.59× peak speedup for semiring GEMM.

### What They Propose
```
Configurable ALUs in Tensor Core-like units:
  ⊗ALU: supports {×, min, max, AND, XOR, popcount, fuzzy-AND}
  ⊕ALU: supports {+, min, max, OR, XOR, threshold}
  
Combinations cover: tropical, bottleneck, Boolean, Hamming, fuzzy.
One instruction does semiring mxv with arbitrary ⊕/⊗.

Hardware cost: ~5% additional chip area.
Performance: 38.59× over optimized CUDA for tropical semiring.
```

### What This Means for Us
```
Our 7 semirings would ALL be hardware-accelerated on SIMD² processors.
No need for dual pipeline (integer + BF16). One unified instruction.

Keep the semiring abstraction generic.
When SIMD² arrives: one new tier in the dispatch macro.

static TIER: LazyLock<Tier> = LazyLock::new(|| {
    if has_simd2() { return Tier::Simd2; }
    if avx512 { return Tier::Avx512; }
    ...
});
```

---

## 8. AVX-512 Instruction Reference for Our Pipeline

```
INSTRUCTION         OPERATION                    USE IN PIPELINE
──────────────────────────────────────────────────────────────────
VPXORD              512-bit XOR                  XOR two planes (binary diff)
VPOPCNTDQ           popcount per u64, 8 parallel Hamming distance
VPANDD/VPORD        512-bit AND/OR               Boolean semiring, alpha masking
VPMINSD/VPMAXSD     packed min/max i32            HammingMin ⊕, SimilarityMax ⊕
VDPBF16PS           32 BF16 dot → f32 accumulate  Similarity/Resonance semirings
VCVTNEPS2BF16       16 f32 → 16 BF16             Cache distance as BF16
VCVTBF162PS*        BF16 → f32 (Rust 1.94)        Hydrate BF16 truth to f32
VPCLMULQDQ          carry-less multiply 512-bit   XorField GF(2) polynomial multiply
VPSRLW              packed shift right word        Extract BF16 exponent (>>7)
VPCMPEQB            packed byte compare            Match exponent to structural pattern
VPTERNLOGD          ternary logic on 3 vectors     Complex Boolean semiring ops
VPSHUFB             shuffle bytes                  BindFirst permutation semiring
_mm_prefetch T0     prefetch to L1                 Next candidate pre-load
```

All stable in Rust 1.94 under `#[target_feature(enable = "avx512f,avx512bw,...")]`.
Safe calls (no pointer args) don't need `unsafe` blocks.
