# DEEP_ADJACENT_EXPLORATION.md

## Dropped Algorithms, RISC Design, and Adjacent Genius

Everything the research surfaced that wasn't about semirings or BF16 but
connects to our architecture in ways I didn't follow.

---

## 1. RDF-3X "RISC-STYLE" ENGINE: The Design We Should Steal Wholesale

Neumann & Weikum (VLDB 2008, 482 citations) built a triple store that
outperforms alternatives by 1-2 ORDERS OF MAGNITUDE. Their key insight
has NOTHING to do with fancy data structures. It's architectural minimalism:

```
RDF-3X RISC PRINCIPLES:
  1. ONE physical design for ALL workloads (no tuning knobs)
  2. EXHAUSTIVE indexes (all 6 SPO permutations + 3 binary + 3 unary = 15 indexes)
  3. ONE query operator: merge join (no hash joins, no nested loops)
  4. Compression everywhere (triples sorted lexicographically, delta-coded)
  5. Dictionary encoding (all literals → integer IDs)
  
  TOTAL: 15 compressed indexes + merge join + dictionary. That's the whole system.
  No configuration. No tuning. No knobs. RISC.
```

**Why this is us:**

```
OUR EQUIVALENT:
  1. ONE physical design: Plane = i8[16384] accumulator. Period.
  2. EXHAUSTIVE: 7 Mask projections (all SPO combinations). Always available.
  3. ONE query operator: hamming_distance (XOR + popcount). That's it.
  4. Compression: binary planes ARE maximally compressed (1 bit per dimension)
  5. Dictionary: Plane.encounter(text) hashes text → fingerprint bits. Same thing.
  
  TOTAL: Plane + 7 masks + hamming + encounter. That's the whole system.
  No configuration. No tuning. No knobs. RISC.
```

**What RDF-3X does that we don't yet:**

```
THEIR AGGREGATED PROJECTIONS:
  6 triple indexes: SPO, SOP, PSO, POS, OSP, OPS  ← we have 7 Mask constants
  3 binary projections: SP, SO, PO                  ← we DON'T have materialized pair projections
  3 unary projections: S, P, O                      ← we DON'T have materialized single projections
  
  The binary and unary projections are COUNT AGGREGATES:
    SP projection: for each (subject, predicate) pair, how many objects?
    S projection: for each subject, how many triples?
    
  These enable OPTIMAL JOIN ORDERING via selectivity estimation.
  "How selective is this pattern?" → look up the aggregated count.
```

**ACTION: Materialize count aggregates per Mask projection.**

```rust
/// RDF-3X style selectivity statistics for each Mask projection.
/// For each mask, store the distribution of distances (or alpha densities).
struct ProjectionStats {
    /// For mask S__: histogram of S-plane alpha densities across all nodes.
    /// Tells the optimizer: "how many nodes have well-defined S planes?"
    s_density_histogram: [u32; 256],   // 256 buckets
    p_density_histogram: [u32; 256],
    o_density_histogram: [u32; 256],
    
    /// For mask SP_: histogram of SP cross-distances.
    /// Tells the optimizer: "how similar are S and P across the graph?"
    sp_distance_histogram: [u32; 256],
    so_distance_histogram: [u32; 256],
    po_distance_histogram: [u32; 256],
    
    /// Total node count and active (alpha > threshold) count per plane.
    total_nodes: u32,
    active_s: u32,
    active_p: u32,
    active_o: u32,
}
```

These statistics cost ~3KB total. They enable the DataFusion planner
to do RDF-3X style cost-based join ordering WITHOUT scanning the graph.

**CRITICAL INSIGHT:** RDF-3X's "RISC" means NOT "reduced instruction set CPU."
It means "reduced instruction set DATABASE." One operation (merge join)
applied uniformly to sorted indexes. Our one operation (hamming distance)
applied uniformly to binary planes. Same philosophy. Same performance gains.
We just need the statistics layer on top.

---

## 2. HEXASTORE 3-LEVEL NESTED SORTED VECTORS: The Actual Data Structure

The Hexastore (Weiss, Karras, Bernstein VLDB 2008) stores SPO triples
in a specific 3-level tree structure PER PERMUTATION:

```
SPO INDEX:
  Level 1: sorted list of all subjects (S)
    For each S → Level 2: sorted list of predicates (P) for this subject
      For each P → Level 3: sorted list of objects (O) for this (S,P) pair

QUERY: (S=Alice, P=knows, O=?)
  Level 1: binary search for "Alice"    → O(log N_subjects)
  Level 2: binary search for "knows"    → O(log N_predicates_of_Alice)
  Level 3: read all objects              → O(N_results)
  
QUERY: (S=?, P=knows, O=Bob)
  WRONG index for SPO. Use PSO or OPS instead.
  PSO Level 1: find "knows"
  PSO Level 2: scan subjects
  PSO Level 3: check each for "Bob"
```

**The key insight I missed:** Hexastore shares terminal lists between
index pairs. SPO and PSO share the O-lists. SOP and OSP share the P-lists.
This means 5x storage, not 6x (advertised worst case).

**How this maps to our architecture:**

```
HEXASTORE LEVEL 1: sorted subjects     → our Node IDs sorted by S-plane alpha density
HEXASTORE LEVEL 2: sorted predicates   → our Mask projections (which planes match)  
HEXASTORE LEVEL 3: sorted objects      → our BF16 truth values sorted by confidence

THE MAPPING:
  "Alice knows Bob" (traditional triple)
  
  → Node_Alice.distance(Node_Bob, mask=_P_) = hamming on P planes
  → if distance < threshold → edge exists with BF16 truth value
  
  Hexastore lookup by subject → cascade scan filtered by S-plane fingerprint
  Hexastore lookup by predicate → cascade scan with mask=_P_
  Hexastore lookup by object → cascade scan filtered by O-plane fingerprint
```

**MERGE JOINS ON SORTED VECTORS:** Hexastore's killer feature is that
SPARQL joins become merge joins on sorted vectors:

```
QUERY: ?x foaf:knows ?y . ?y rdf:type foaf:Person
  
  Pattern 1: (?x, knows, ?y) → use PSO index, P="knows" → sorted list of (S,O) pairs
  Pattern 2: (?y, type, Person) → use PSO index, P="type" → sorted list of (S,O) pairs
  
  Join on ?y: merge the two sorted lists on the O column of pattern 1
              and the S column of pattern 2.
  
  Merge join = O(N+M) where N,M are list lengths. Not O(N×M).
```

**Our equivalent: sorted BF16 edge lists enable merge joins.**

```rust
/// Hexastore-style sorted edge list for one Mask projection.
/// Sorted by BF16 truth value (descending confidence).
/// Enables merge join between two edge lists on shared node IDs.
struct SortedEdgeList {
    /// Sorted by (source_node, bf16_truth descending)
    entries: Vec<(u32, u32, u16)>,  // (source_id, target_id, bf16_truth)
}

impl SortedEdgeList {
    /// Merge join: find all (a,b) where a→b in self AND b→c in other.
    /// Returns (a, b, c) triples. O(N+M) not O(N×M).
    fn merge_join(&self, other: &SortedEdgeList) -> Vec<(u32, u32, u32)> {
        // self sorted by target_id, other sorted by source_id
        // Walk both lists simultaneously
    }
}
```

---

## 3. BELLMAN-FORD AND FLOYD-WARSHALL IN HAMMING SPACE

These aren't just "graph algorithms we could implement." They're the
CORE OPERATIONS of knowledge graph reasoning.

```
BELLMAN-FORD: single-source shortest paths
  for each edge (u,v,w): relax d[v] = min(d[v], d[u] + w)
  Repeat V-1 times.
  
IN HAMMING SPACE:
  w = hamming_distance(node_u.plane, node_v.plane)
  d[v] = min(d[v], d[u] + w)
  
  "What is the shortest chain of similar nodes from A to B?"
  This IS analogical reasoning:
    A is similar to X (hamming 50)
    X is similar to Y (hamming 30)
    Y is similar to B (hamming 40)
    → A reaches B through X,Y with total distance 120
    → Direct A-B hamming might be 500
    → The INDIRECT path through similar nodes is SHORTER
    → This is HOW analogies work: A relates to B through intermediaries

FLOYD-WARSHALL: all-pairs shortest paths
  for k: for i: for j: d[i][j] = min(d[i][j], d[i][k] + d[k][j])
  
  With N nodes and hamming as edge weight:
  The full distance matrix tells you ALL analogical relationships.
  d[i][j] = shortest analogical chain between any two nodes.
  
  For N=1000 nodes: 1000³ = 1B hamming operations.
  At 140ns per hamming: 140 seconds.
  With cascade pre-filtering (reject 99.7%): ~0.4 seconds.
  With multi-index hashing for candidate generation: ~0.04 seconds.
```

**What Bellman-Ford gives us for RL:**

```
CURRENT RL: encounter one pair at a time, update locally.
  
BELLMAN-FORD RL: propagate reward through the graph.
  
  Node A gets reward. Bellman-Ford relaxation propagates:
    A's neighbors get discounted reward (γ × reward)
    Their neighbors get γ² × reward
    ...
  
  This IS temporal difference learning (TD(λ)).
  Bellman-Ford on the Hamming graph = TD learning on the SPO graph.
  The discount factor γ maps to hamming distance:
    close neighbors (low hamming) get more reward.
    distant nodes (high hamming) get less.
  
  γ = exp(-hamming / temperature)
  
  This is VALUE FUNCTION propagation through the graph.
  Bellman-Ford computes it in V-1 iterations.
  Each iteration: one round of encounter() on all edges.
```

**ACTION:** Implement Bellman-Ford as value propagation:

```rust
/// Bellman-Ford value propagation on SPO graph.
/// Propagates reward from source node through all reachable nodes.
/// Discount by hamming distance (close nodes get more reward).
fn bellman_ford_reward(
    graph: &mut SpoGraph,
    source: usize,
    reward: i8,  // +1 or -1
    temperature: u32,  // hamming distance for γ=1/e decay
    max_iterations: usize,
) {
    let mut values = vec![0i32; graph.nodes.len()];
    values[source] = reward as i32 * 1000;  // fixed point, scale by 1000
    
    for _ in 0..max_iterations {
        let mut changed = false;
        for edge in &graph.edges {
            let s = edge.source as usize;
            let t = edge.target as usize;
            // Discount = exp(-hamming/temperature) ≈ 1000 * (1 - hamming/temperature)
            let discount = 1000i32 - (edge.hamming as i32 * 1000 / temperature as i32);
            if discount <= 0 { continue; }
            
            let propagated = values[s] * discount / 1000;
            if propagated > values[t] {
                values[t] = propagated;
                changed = true;
            }
        }
        if !changed { break; }
    }
    
    // Apply propagated values as encounters
    for (i, &v) in values.iter().enumerate() {
        if v > 0 {
            graph.nodes[source].reward_encounter(&mut graph.nodes[i], 1);
        } else if v < 0 {
            graph.nodes[source].reward_encounter(&mut graph.nodes[i], -1);
        }
    }
}
```

---

## 4. ReLU = max(x, 0) = TROPICAL MAX = OUR ALPHA THRESHOLD

Zhang et al. (ICML 2018) proved ReLU networks are tropical rational maps.
I stated this but didn't follow the operational consequence:

```
ReLU(x) = max(x, 0)

OUR ALPHA: alpha[k] = (|acc[k]| > threshold) ? 1 : 0

REWRITE OUR ALPHA AS TROPICAL ReLU:
  alpha[k] = max(|acc[k]| - threshold, 0) > 0 ? 1 : 0
  
  But if we DON'T binarize — if we keep the continuous value:
  alpha_continuous[k] = max(|acc[k]| - threshold, 0)
  
  This IS a ReLU activation on the accumulator magnitude.
  The threshold IS the bias term in ReLU(x - b).
  
  α_k = ReLU(|acc_k| - b_k) where b_k is the per-bit threshold.
```

**What this means:** Our binary alpha channel is a BINARIZED ReLU activation.
If we keep the continuous version, we get a MULTI-LEVEL alpha:

```
acc[k] = 0   → alpha = 0.0 (undefined, maximum uncertainty)
acc[k] = 5   → alpha = 0.0 (below threshold, still uncertain)
acc[k] = 10  → alpha = 0.0 (just below threshold)
acc[k] = 11  → alpha = 1.0 (above threshold, BINARY: defined)

WITH CONTINUOUS ALPHA (ReLU):
acc[k] = 0   → alpha = 0.0 (undefined)
acc[k] = 5   → alpha = 0.0 (below threshold)
acc[k] = 10  → alpha = 0.0 (at threshold)
acc[k] = 11  → alpha = 0.008 (barely above threshold, LOW confidence)
acc[k] = 50  → alpha = 0.313 (moderate confidence)
acc[k] = 127 → alpha = 0.914 (high confidence, nearly saturated)

Normalized: alpha = ReLU(|acc| - threshold) / (127 - threshold)
```

**The continuous alpha gives us SOFT attention over bit positions.**
The distance computation becomes:

```
CURRENT (hard alpha):
  distance = popcount(XOR(a.bits, b.bits) & a.alpha & b.alpha)
  Every defined bit contributes equally.

WITH SOFT ALPHA:
  distance = Σ_k (a.bits[k] XOR b.bits[k]) × a.alpha[k] × b.alpha[k]
  Highly confident bits contribute MORE to distance.
  Recently defined bits contribute LESS.
  
  This IS attention-weighted hamming distance.
  The alpha channel IS the attention mask.
  ReLU gives us the gradient for learning the mask.
```

**HARDWARE:** The soft alpha distance needs VPMADDUBSW (multiply-add unsigned
bytes) instead of just VPANDD + VPOPCNTDQ. Slightly more expensive but
gives us learned, continuous attention for free.

---

## 5. DICTIONARY ENCODING: The Missing Compression Layer

Both RDF-3X and Hexastore use dictionary encoding: replace string literals
with integer IDs. All internal operations use the compact IDs.

```
RDF-3X DICTIONARY:
  "Alice" → 42
  "knows" → 7
  "Bob"   → 108
  
  Triple ("Alice", "knows", "Bob") → (42, 7, 108)
  Storage: 12 bytes (3 × u32) instead of ~17 bytes (strings)
  
  Join: integer comparison instead of string comparison.
  Sort: integer sort instead of string sort.
```

**We already have this, but don't call it that:**

```
OUR "DICTIONARY":
  "Alice" → encounter("Alice") → Fingerprint<256> (2KB hash)
  "knows" → encounter("knows") → Fingerprint<256> (2KB hash)
  
  We hash to 16K bits instead of assigning sequential IDs.
  The hash IS the ID. Collision probability: 2^(-16384). Zero in practice.
  
  ADVANTAGE over RDF-3X: no dictionary maintenance, no ID allocation,
  no lookup table. The fingerprint IS the representation.
  
  DISADVANTAGE: 2KB per "ID" instead of 4 bytes.
  But: the 2KB carries SEMANTIC DISTANCE information.
  RDF-3X's integer IDs carry NO distance information.
  "Alice" = 42 and "Bob" = 108 → distance = |42-108| = 66. MEANINGLESS.
  Our fingerprints: hamming(Alice, Bob) = MEANINGFUL semantic distance.
```

**The RISC insight applied:** RDF-3X eliminates string handling entirely.
All query processing operates on integers. We should ensure our DataFusion
planner NEVER handles raw strings during query execution — only Plane
references and hamming distances. The dictionary encoding (encounter → fingerprint)
happens ONCE at ingestion time. Everything after is integer/binary.

---

## 6. MATLAB + GRAPHBLAS: WHAT IF MATLAB USED OUR ARCHITECTURE?

MATLAB uses SuiteSparse:GraphBLAS internally since R2021a for sparse
matrix operations. The connection:

```
MATLAB/GraphBLAS:
  A = GrB(1000, 1000, 'double')  — sparse matrix, float64 entries
  C = A * B                       — semiring mxm (default: +.× over doubles)
  C = GrB.mxm('+.min', A, B)     — tropical semiring
  
  User writes: C = A * B
  MATLAB calls: GrB_mxm(C, NULL, NULL, GrB_PLUS_TIMES_FP64, A, B, NULL)
  SuiteSparse dispatches: FactoryKernel for (FP64, PLUS, TIMES)

IF MATLAB USED OUR ARCHITECTURE:
  A = BinaryMatrix(1000, 1000, 16384)  — sparse matrix, 16K-bit entries
  C = hamming_mxm(A, B)                — HammingMin semiring
  
  User writes: C = A * B
  Backend calls: grb_mxm(C, HammingMin, A, B)
  Dispatch: VPXORD + VPOPCNTDQ + VPMINSD
  
  For 1000×1000 with 16K-bit entries:
    Standard MATLAB (double): 2 × 1000³ × 8B = 16GB of float ops
    Our architecture: 1000³ × XOR(2KB) + popcount = 2TB of bitwise ops
    
    BUT: cascade pre-filtering rejects 99.7% of pairs.
    Effective: 1000³ × 0.003 = 27M hamming ops × 140ns = 3.8 seconds
    vs MATLAB double: 1000³ × 2 FLOP / 35 GFLOPS = 57 seconds
    
    15x faster. On CPU. No GPU.
```

**The genius solution:** A MATLAB toolbox that wraps our rustynum-core
as a MEX binary. MATLAB users write:

```matlab
% Standard MATLAB:
A = sparse_graph(nodes, edges);
D = shortest_paths(A);  % Floyd-Warshall, O(N³) double arithmetic

% Our toolbox:
A = binary_graph(nodes_16k, edges_hamming);
D = tropical_shortest(A);  % Floyd-Warshall, O(N³×0.003) with cascade rejection
                            % 15x faster, deterministic, explainable
```

**MATLAB's user base:** 4 million engineers and scientists. Giving them
access to binary SPO graph algorithms through a familiar interface
is the distribution channel for the whole architecture.

---

## 7. SUITESPARSE FACTORYKERNELS: Runtime JIT for Custom Semirings

SuiteSparse:GraphBLAS has ~2000 pre-compiled FactoryKernels for common
type/semiring combinations. For custom semirings, it JIT-compiles:

```
USER DEFINES:
  GrB_Semiring my_semiring;
  GrB_Semiring_new(&my_semiring, my_add_monoid, my_multiply_op);
  
SUITESPARSE:
  1. Check FactoryKernels — not found (custom)
  2. Generate C source code for this semiring's mxm kernel
  3. Compile with system cc -O3 -march=native
  4. dlopen() the .so, cache in ~/.SuiteSparse/
  5. Next call: use cached .so (performance = FactoryKernel)
```

**For us:** We have 7 fixed semirings. No JIT needed yet.
But if someone wants a CUSTOM semiring (e.g., fuzzy min-max):

```rust
// Our equivalent of FactoryKernels:
// 7 hand-written SIMD implementations, one per semiring.
// Dispatch via match on HdrSemiring enum.

// Future: JIT for custom semirings using Cranelift (already in rustynum via jitson)
// jitson already JIT-compiles scan kernels.
// Extend to JIT-compile custom semiring (⊕, ⊗) pairs.

fn jit_semiring(add_op: BinaryOp, mul_op: BinaryOp) -> CompiledKernel {
    let mut builder = jitson::Builder::new();
    // Generate: for each element, apply mul_op then accumulate with add_op
    // Compile via Cranelift
    // Cache the compiled kernel
    builder.build()
}
```

**The Cranelift connection:** We already have jitson (Cranelift-based JIT
for scan kernels). Extending it to JIT custom semiring kernels gives
us SuiteSparse-equivalent JIT capability without system cc dependency.

---

## 8. INDEX-FREE ADJACENCY → SIMD-SCANNABLE ADJACENCY

Neo4j's index-free adjacency: O(1) pointer per hop. Cache-hostile.
Our alternative: what if adjacency IS a binary plane?

```
NEO4J: node.first_rel_ptr → linked list of relationship records
  Each hop: follow pointer, random memory access, cache miss.

OUR ALTERNATIVE: node.neighbor_mask: Fingerprint<256>
  The neighbor mask has bit k SET if node k is a neighbor.
  
  "Who are Alice's neighbors?" = popcount(Alice.neighbor_mask)
  "Is Bob Alice's neighbor?" = Alice.neighbor_mask[Bob.id] (1 bit check)
  "Shared neighbors of Alice and Bob?" = popcount(AND(Alice.mask, Bob.mask))
  
  ALL of these are SIMD operations:
    popcount: VPOPCNTDQ
    bit check: VPTEST
    shared neighbors: VPANDD + VPOPCNTDQ
    
  O(1) for all of them. No pointer chasing. No cache misses.
  The adjacency IS a binary vector. SIMD-scannable.
```

**Limitation:** 16K bits = 16384 max nodes in the neighbor mask.
For larger graphs: hierarchical masking. 16K bits per cluster.
Cluster-level mask → node-level mask within cluster.

**For our SPO graph:** The P plane already encodes "what relationships
this node participates in." A separate neighbor_mask per plane:

```rust
struct SpoNode {
    s: Plane,          // WHO
    p: Plane,          // WHAT relationship
    o: Plane,          // TO WHOM
    
    // SIMD-scannable adjacency:
    s_neighbors: Fingerprint<256>,  // which nodes share S-plane similarity
    p_neighbors: Fingerprint<256>,  // which nodes share P-plane similarity
    o_neighbors: Fingerprint<256>,  // which nodes share O-plane similarity
    
    // "Alice knows Bob AND Bob knows Carol"
    // = popcount(AND(Alice.p_neighbors, Carol.p_neighbors))
    // = number of shared P-neighbors (transitive KNOWS chain)
}
```

---

## 9. SPARQL MERGE JOIN → HAMMING MERGE JOIN

The SPARQL insight from Hexastore: queries reduce to merge joins
on sorted vectors. The join variable provides the merge key.

```
SPARQL: ?x foaf:knows ?y . ?y rdf:type foaf:Person
  
  Pattern 1 result: sorted list of (?x, ?y) from "knows" index
  Pattern 2 result: sorted list of (?y) from "type=Person" index
  
  Merge join on ?y: walk both sorted lists, emit matches.
  O(N+M) time. Cache-friendly (sequential access on sorted arrays).
```

**Our version: hamming-sorted edge lists enable merge joins.**

```
QUERY: find all (a, b, c) where a~b on S-plane AND b~c on P-plane

  Step 1: cascade scan for S-similar pairs → sorted by BF16 truth → list_1
  Step 2: cascade scan for P-similar pairs → sorted by BF16 truth → list_2
  
  Step 3: merge join list_1 and list_2 on shared node ID (b).
  
  The BF16 truth values are already sorted (cascade returns ranked hits).
  The merge join walks both lists in O(N+M).
  
  TOTAL: 2 cascade scans + 1 merge join.
  NOT: N × M hamming comparisons (quadratic).
```

**This is the query optimizer strategy:**
Every multi-pattern query decomposes into:
1. Independent cascade scans (one per pattern)
2. Merge joins on shared variables (sorted by BF16 truth)

The DataFusion planner already does join ordering.
We just need to ensure cascade results are sorted by node ID
(or sortable in O(N log N) after cascade).

---

## 10. WHAT THE "RISC" PHILOSOPHY MEANS FOR THE WHOLE STACK

RDF-3X's RISC means: a few operations, applied uniformly, with no tuning.
Our RISC means:

```
OPERATIONS (the "reduced instruction set"):
  1. encounter(evidence)          — learn from observation (integer accumulate)
  2. hamming_distance(a, b)       — measure similarity (XOR + popcount)
  3. band_classify(distance)      — categorize similarity (integer compare)
  4. merge_join(sorted_a, sorted_b) — combine query results (sequential scan)
  
  FOUR operations. Everything else composes from these:
  
  RL gradient     = encounter with sign(reward)
  GNN convolution = encounter with neighbor evidence
  BF16 truth      = band_classify on 7 projections, pack bits
  NARS revision   = tropical arithmetic on BF16 exponents
  Graph traversal = Bellman-Ford using hamming as edge weight
  Query execution = cascade scan + merge join on sorted results
  Value function  = Floyd-Warshall shortest paths in Hamming space
  World model     = cascade predicts band from partial observation
  Attention       = cascade strokes = multi-head tropical attention
```

Everything is the same four operations at different scales.
That's RISC. Not reduced instruction set computer.
Reduced instruction set COGNITION.

---

## INVESTIGATION QUEUE

```
PAPER/TOPIC                              READ FOR                     PRIORITY
──────────────────────────────────────────────────────────────────────────────────
RDF-3X (Neumann, Weikum 2008)           RISC design + stats layer     CRITICAL
Hexastore (Weiss, Karras, Bernstein)    3-level nested vectors         HIGH
Bellman-Ford as TD learning             RL reward propagation          HIGH
Tropical Attention (arXiv:2505.17190)   Cascade = attention heads      HIGH
Multi-index hashing (Norouzi 2014)      Sub-linear search              HIGH
Mohri semiring transducers              Query optimization             MEDIUM
ReActNet per-bit learned thresholds     Continuous alpha / soft ReLU   MEDIUM
Zhang tropical geometry (ICML 2018)     ReLU = tropical max proof      MEDIUM
SIMD² (ISCA 2022)                       Future hardware for semirings  LOW
FalkorDB (GraphBLAS-powered graph DB)   Architecture reference         LOW
SuiteSparse JIT                         Custom semiring compilation    LOW
```
