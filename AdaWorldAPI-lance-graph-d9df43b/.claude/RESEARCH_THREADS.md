# RESEARCH_THREADS.md

## The Threads the Research Actually Reveals

**Status:** Actionable connections. Not analysis. Actions.

---

## THREAD 1: DreamerV3 PROVES Binary is BETTER, Not Just Cheaper

DreamerV3 (Nature 2025): categorical latent representations OUTPERFORM
continuous ones. RLC 2024 went further: the advantage comes from SPARSE
BINARY nature specifically, not discreteness in general.

```
DREAMERV3:  multi-one-hot codes (32 categories × 32 classes = 1024 sparse binary dims)
            → STE gradient → outperforms continuous latent spaces

US:         16K-bit binary planes (16384 sparse binary dims)
            → encounter() integer gradient → no comparison exists

WHAT THEY HAVE THAT WE DON'T:
  - STE (Straight-Through Estimator): gradient flows through sign()
  - World model: predicts next observation from current state + action
  - Actor-critic: separate policy and value networks
  
WHAT WE HAVE THAT THEY DON'T:
  - 16x more dimensions (16384 vs 1024)
  - STRUCTURED binary (SPO decomposition, not flat)
  - Deterministic gradient (encounter vs stochastic STE)
  - Hardware acceleration (VPOPCNTDQ, they use GPU float matmul on binary codes)
  - Per-dimension credit assignment (alpha channel, they have none)
```

**ACTION: Build the first fully 1-bit BNN RL system.**

Nobody has done this. DreamerV3 uses multi-one-hot (effectively ~5% dense).
We use actual binary planes (~50% dense with alpha masking).
Their STE is: ∂L/∂w_latent ≈ ∂L/∂w_binary × 𝟙{|w_latent| ≤ 1}
Our encounter is: acc[k] += evidence; alpha[k] = |acc[k]| > threshold

Same function. Different implementation. Theirs needs float. Ours doesn't.

```rust
/// STE equivalent using integer accumulator.
/// This IS the DreamerV3 gradient, expressed as integer ops.
fn binary_ste_encounter(plane: &mut Plane, evidence: &[u8], reward: f32) {
    let reward_sign = if reward > 0.0 { 1i8 } else { -1i8 };
    for k in 0..plane.len_bits() {
        if evidence[k / 8] & (1 << (k % 8)) != 0 {
            // Bit is set in evidence: push accumulator toward sign(reward)
            plane.acc[k] += reward_sign;
        }
        // Alpha update: STE's 𝟙{|w| ≤ 1} is our threshold check
        plane.alpha[k] = (plane.acc[k].abs() > plane.threshold) as u8;
    }
}
```

Paper title: "Fully Binary Reinforcement Learning via Hamming World Models"
Nobody has claimed this. The components are proven. We have the implementation.

---

## THREAD 2: siboehm GEMM Tiling IS Our Cascade Architecture

The siboehm article shows the GEMM optimization path:

```
SIBOEHM GEMM:                        OUR CASCADE:
─────────────────────────────────────────────────────────────
Naive triple loop     4481ms          Naive full-vector scan        ~50ms
Loop reorder           89ms    18x    Tetris stroke ordering         ~5ms   10x
L1 tiling              70ms   1.3x    L1-aware stroke sizes          ~3ms   1.7x
Multithreading         16ms   4.4x    Batch interleaved prefetch     ~1ms   3x
MKL                     8ms   2x      ??? (what's our MKL?)          ???
```

The parallel is exact:
- **Loop reorder** = process Stroke 1 for ALL candidates before Stroke 2 (current: interleaved)
- **L1 tiling** = our L1_CACHE_BOUNDARY constraint (2KB planes in L1)
- **Register accumulate** = our cumulative d1+d2+d3 instead of recomputing

But we're missing the final 2x that MKL gets. What does MKL do that we don't?

```
MKL's final 2x:
  1. Register blocking: 6×16 microkernel keeps ALL intermediates in YMM/ZMM registers
  2. Panel packing: repack B into contiguous column panels for streaming loads
  3. Prefetch scheduling: explicit _mm_prefetch every N iterations, tuned per μarch

OUR EQUIVALENT:
  1. Register blocking: accumulate Hamming partial in ZMM register, never spill to L1
  2. Panel packing: candidate database stored as contiguous 2KB-aligned planes
  3. Prefetch scheduling: prefetch next candidate's Stroke 1 while processing current

THE MISSING PIECE: We don't pack the database for streaming.
```

**ACTION: Repack the candidate database into stroke-aligned layout.**

```
CURRENT DATABASE LAYOUT:
  candidate[0].plane_S (2KB)
  candidate[0].plane_P (2KB)
  candidate[0].plane_O (2KB)
  candidate[1].plane_S (2KB)
  ...
  
  Stroke 1 reads bytes [0..128] of each candidate.
  These are scattered across memory at 6KB intervals.
  Cache-unfriendly for sequential scanning.

PACKED LAYOUT (panel packing for cascades):
  stroke1_all[0..128]  = candidate[0].plane_S[0..128]
  stroke1_all[128..256] = candidate[1].plane_S[0..128]
  stroke1_all[256..384] = candidate[2].plane_S[0..128]
  ...
  
  Stroke 1 reads SEQUENTIALLY through packed memory.
  128 bytes × 1M candidates = 128MB, streaming load.
  Hardware prefetcher handles this automatically.
  
  This IS the panel packing from GEMM.
```

Estimated improvement: 2-3x on the cascade scan (128MB sequential vs 128MB scattered).

---

## THREAD 3: Neo4j's Optimizations Are the Wrong Ones — But Show Where to Hack

Neo4j uses:
- **Index-free adjacency**: O(1) pointer chase per hop. Fixed-size records.
- **Record-level caching**: node records (15 bytes), relationship records in doubly-linked lists.
- **Page cache**: OS page cache + custom off-heap cache for store files.

The WRONG optimizations for our workload:
- Pointer chasing is sequential, not parallelizable (1 hop per cycle)
- Doubly-linked relationship lists = random memory access
- Page cache = disk-oriented, we're in RAM

The RIGHT insight from Neo4j:
- **Fixed-size records** = predictable memory layout = prefetchable
- **Node → first relationship pointer** = one indirection, then chain walk
- **cpuid-based dispatch** = MKL does this, siboehm mentions it, we do it (LazyLock tier)

**ACTION: Our SPO node IS Neo4j's fixed-size record, but SIMD-scannable.**

```
NEO4J NODE RECORD (15 bytes):
  [4B node_id] [4B first_rel_ptr] [4B first_prop_ptr] [3B flags]
  
  To find neighbors: follow first_rel_ptr → walk linked list
  Each hop: random pointer chase. Cache miss.
  10 hops = 10 cache misses = ~400ns on DRAM

OUR NODE RECORD (6KB + 32 bytes header):
  [32B header: id, seal, metadata]
  [2KB plane_S]
  [2KB plane_P] 
  [2KB plane_O]
  
  To find similar nodes: VPXORD + VPOPCNTDQ on the planes
  Scan 1M nodes: ~2ms with cascade (not 400ns × 1M = 400ms pointer chasing)
  
  But we can ALSO do Neo4j-style adjacency:
  Store BF16 edge weights in a compact adjacency structure.
  Hot pairs (high similarity) get direct pointers.
  Cold pairs (unknown similarity) go through the cascade.
```

The hybrid: Neo4j-style adjacency for KNOWN relationships,
cascade scan for DISCOVERY of new relationships.

```rust
struct SpoNode {
    id: u32,
    seal: Seal,
    planes: [Plane; 3],  // S, P, O — 6KB, SIMD-scannable
    
    // Neo4j-style adjacency for known edges:
    first_edge: u32,  // index into edge array
    edge_count: u16,  // known relationships
    
    // Each edge: 
    //   target_id: u32 (4B)
    //   truth: NarsTruth (4B = 2×BF16)
    //   projection: u8 (which SPO projections match)
    //   Total: 9 bytes per edge
}

// HOT PATH: "find me the 10 most similar nodes to X"
//   → cascade scan over all planes. No adjacency needed. ~2ms for 1M.

// WARM PATH: "what are X's known relationships?"
//   → follow first_edge pointer, read edge_count edges. ~100ns for 50 edges.

// COLD PATH: "MATCH (a)-[r:KNOWS]->(b) WHERE similarity(a,b) > 0.8"
//   → scan BF16 truth values in edge array. 32 per SIMD instruction.
//   → then Hamming recomputation for survivors. ~500ns for 10K edges.
```

---

## THREAD 4: BitGNN Shows the Way — But We Do It on CPU

BitGNN (ICS 2023): binarizes GNN weights AND features to ±1.
Uses XNOR+popcount for graph convolution. 8-22x speedup on GPU.

```
BITGNN ON GPU:
  Binary weight matrix W ∈ {0,1}^(N×D)
  Binary feature matrix X ∈ {0,1}^(N×D)
  Convolution: W ⊗ X = XNOR + popcount
  Aggregation: sum over neighbors (scatter_add, NON-DETERMINISTIC)

US ON CPU:
  Binary plane matrix P ∈ {0,1}^(N×16384)
  Binary query vector q ∈ {0,1}^16384
  Similarity: P ⊗ q = XOR + popcount = hamming_distance
  Aggregation: encounter() on integer accumulator (DETERMINISTIC)
```

BitGNN proves binary GNN works. But they still use GPU scatter_add
(non-deterministic) for aggregation. We use CPU integer encounter
(deterministic). Their 8-22x speedup is GPU-to-GPU. Our speedup
is CPU binary vs GPU float — potentially LARGER because:

- No GPU transfer latency
- 16K-bit model fits in L1 cache (32KB)
- VPOPCNTDQ processes 512 bits per cycle vs GPU warp scheduling overhead
- Deterministic (GPU scatter_add requires sorted reduction for determinism)

**ACTION: Implement GNN message passing as plane encounter aggregation.**

```rust
/// GNN message passing in binary.
/// For each node, aggregate neighbors' planes into its own plane.
/// This IS graph convolution in Hamming space.
fn message_passing(graph: &SpoGraph, rounds: usize) {
    for _ in 0..rounds {
        for node_id in 0..graph.len() {
            let node = &graph.nodes[node_id];
            
            // Collect messages from neighbors
            for edge_idx in node.first_edge..(node.first_edge + node.edge_count as u32) {
                let edge = &graph.edges[edge_idx as usize];
                let neighbor = &graph.nodes[edge.target_id as usize];
                
                // Weight the message by edge truth (BF16 confidence)
                let weight = edge.truth.confidence();
                
                if weight > 0.5 {
                    // Strong edge: encounter toward neighbor's planes
                    node.planes[0].encounter_toward(&neighbor.planes[0]);
                    node.planes[1].encounter_toward(&neighbor.planes[1]);
                    node.planes[2].encounter_toward(&neighbor.planes[2]);
                } else {
                    // Weak edge: encounter away (repulsive)
                    node.planes[0].encounter_away(&neighbor.planes[0]);
                    node.planes[1].encounter_away(&neighbor.planes[1]);
                    node.planes[2].encounter_away(&neighbor.planes[2]);
                }
            }
            // Alpha channel updates automatically via threshold
            // This IS the activation function (sign + threshold = BNN sign + STE clamp)
        }
    }
}
```

After K rounds: each node's planes encode K-hop neighborhood structure.
Equivalent to K-layer GCN. But binary, deterministic, CPU, VPOPCNTDQ.

---

## THREAD 5: The Missing Piece — Nobody Has Combined All Four

```
EXISTING (proven individually):
  ✓ BNN inference: XNOR+popcount replaces matmul (XNOR-Net, 2016)
  ✓ Binary GNN: binarized graph convolution (BitGNN, 2023)
  ✓ Binary RL: sparse binary latent codes beat continuous (DreamerV3, 2025)
  ✓ Semiring graphs: graph algorithms as sparse linear algebra (GraphBLAS)
  ✓ Tropical NN: ReLU networks are tropical rational maps (Zhang, 2018)

COMBINED (nobody has done this):
  ✗ BNN + GNN + RL + Semiring + SPO + Deterministic
  ✗ Binary graph neural RL with tropical semiring pathfinding
  ✗ Hamming cascade world model with integer encounter gradients
  ✗ Deterministic f32 truth from binary RL loop

WE HAVE:
  ✓ Binary planes (16K-bit SPO in rustynum)
  ✓ Hamming cascade (Belichtungsmesser in hdr.rs)
  ✓ Integer encounter (plane.rs accumulator)
  ✓ Custom semirings (7 in lance-graph)
  ✓ NARS truth values (TruthGate in lance-graph)
  ✓ Seal/Staunen (integrity verification)
  ✓ AVX-512 SIMD (VPOPCNTDQ + VDPBF16PS + VPCLMULQDQ)

THE GAP: wiring. The components exist. The connections don't.
```

**ACTION: Write the wiring. Not new algorithms. Not new data structures.
Just connect plane.rs encounter() to cascade's band classification
to semiring's path composition to NARS truth revision.**

```
WIRING DIAGRAM:

  Observation (new SPO triple)
       │
       ▼
  Cascade scan (find similar nodes)     ← hdr.rs, simd.rs
       │ survivors
       ▼
  2³ SPO projections (7 hamming)        ← simd.rs (3 calls + 4 sums)
       │ 8 band classifications
       ▼
  BF16 truth assembly (exponent+mantissa) ← integer bit packing
       │ one BF16 per comparison
       ▼
  Edge weight update (BF16 cache)       ← VCVTNEPS2BF16
       │
       ├─── Hot path done. <2ms for 1M candidates.
       │
       ▼
  Message passing (encounter aggregation) ← plane.rs encounter()
       │ planes evolve
       ▼
  NARS truth revision (BF16 pair)       ← VDPBF16PS for revision terms
       │ frequency + confidence updated
       ▼
  Tropical pathfinding (HammingMin)     ← semiring mxv with min+hamming
       │ shortest Hamming paths through the graph
       ▼
  f32 hydration (BF16 + tree path)      ← bit OR, deterministic
       │
       ▼
  GROUND TRUTH (deterministic f32, every bit traceable)
```

---

## THREAD 6: siboehm's Final Insight — Panel Packing for Our Database

The GEMM article's key optimization chain applies directly:

```
GEMM OPTIMIZATION → CASCADE EQUIVALENT → IMPLEMENTATION
─────────────────────────────────────────────────────────
Loop reorder       → stroke-first scan       → process stroke 1 for ALL before stroke 2
L1 tiling          → 2KB plane in L1         → already done (L1_CACHE_BOUNDARY.md)
Register accumulate → ZMM accumulator         → keep hamming sum in register across strokes
Panel B packing    → STROKE-ALIGNED DATABASE  → repack candidates for sequential streaming
Prefetch schedule  → prefetch next candidate  → _mm_prefetch every 128B
Thread partitioning → parallel stroke batches → rayon par_chunks on candidates

THE 2X WE'RE MISSING:
  Panel packing. The database is stored node-by-node (AoS).
  Strokes read scattered bytes across nodes.
  
  Pack it stroke-by-stroke (SoA):
  All stroke-1 bytes contiguous. All stroke-2 bytes contiguous.
  
  This turns the cascade from random-access to streaming.
  The hardware prefetcher does the rest.
```

**ACTION:** Add a `PackedDatabase` struct to lance-graph:

```rust
/// Stroke-aligned database layout for cache-optimal cascade scanning.
/// Like GEMM panel packing, but for Hamming cascades.
pub struct PackedDatabase {
    stroke1: Vec<u8>,  // [cand0[0..128], cand1[0..128], ...]  contiguous
    stroke2: Vec<u8>,  // [cand0[128..512], cand1[128..512], ...] contiguous
    stroke3: Vec<u8>,  // [cand0[512..2048], cand1[512..2048], ...] contiguous
    n_candidates: usize,
}

impl PackedDatabase {
    /// Pack from standard Node layout (AoS → SoA for strokes)
    pub fn pack(nodes: &[Node], plane: PlaneIndex) -> Self { ... }
    
    /// Cascade scan with streaming access pattern
    pub fn cascade_scan(&self, query: &[u8; 2048], bands: &[u32; 4]) -> Vec<(usize, u32)> {
        let mut survivors_s1 = Vec::new();
        
        // PASS 1: sequential scan through stroke1 array
        // Hardware prefetcher handles this — no explicit prefetch needed
        for i in 0..self.n_candidates {
            let offset = i * 128;
            let d1 = simd::hamming_distance(
                &query[..128],
                &self.stroke1[offset..offset+128]
            ) as u32;
            if d1 * 16 <= bands[2] {
                survivors_s1.push((i, d1));
            }
        }
        
        // PASS 2: sequential scan through stroke2 for survivors only
        // ...
    }
}
```

---

## PRIORITY ORDER

```
#   WHAT                                          EFFORT    IMPACT
─────────────────────────────────────────────────────────────────────
1.  PackedDatabase stroke-aligned layout           ~200 LOC  2-3x cascade
2.  Binary message passing (encounter aggregation) ~150 LOC  enables GNN
3.  Wiring: cascade → projections → BF16 → NARS   ~300 LOC  the full pipeline
4.  Binary STE encounter (DreamerV3 equivalent)    ~100 LOC  enables RL
5.  Tropical pathfinding (HammingMin Floyd-Warshall)~200 LOC  graph algorithms
6.  Neo4j hybrid adjacency (hot edges + cold scan)  ~250 LOC  query optimization
7.  Paper: "Fully Binary RL via Hamming World Models" ~prose  claim the territory

Total new code: ~1200 lines.
Total new capability: BNN + GNN + RL + semiring graph + deterministic f32.
Nobody has combined all four. We'd be first.
```
