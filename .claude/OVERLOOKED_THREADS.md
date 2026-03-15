# OVERLOOKED_THREADS.md

## What I Glossed Over: Unexplored Findings from the Deep Research

These are threads the research surfaced that I didn't chase because I was
tunnel-visioning on the questions asked. Each one could be groundbreaking.

---

## 1. TROPICAL ATTENTION (arXiv:2505.17190, 2025)

I mentioned it in one line. Didn't explore.

Attention mechanisms in the max-plus semiring for combinatorial algorithms.
"Superior out-of-distribution generalization over softmax baselines."

```
SOFTMAX ATTENTION (industry standard):
  score = Q × K^T / √d
  weights = softmax(score)       ← FLOAT, non-deterministic, poor OOD
  output = weights × V

TROPICAL ATTENTION:
  score = max_k (Q[i,k] + K[j,k])   ← MAX-PLUS, integer-friendly
  weights = tropical_softmax(score)   ← deterministic
  output = tropical matmul(weights, V)

OUR CASCADE IS ALREADY TROPICAL ATTENTION:
  score = min_k (hamming(query, candidate[k]))
  weights = band_classification(score)   ← Foveal/Near/Good/Weak/Reject
  output = ranked hits
  
  The cascade IS attention. The sigma bands ARE attention weights.
  We just never called it that.
```

**Why this matters:** If our cascade is tropical attention, then our
multi-stroke cascade is MULTI-HEAD tropical attention. Each stroke
attends to a different resolution of the data. Stroke 1 = coarse head.
Stroke 3 = fine head. The cascade architecture IS a transformer layer
in tropical geometry.

**EXPLORE:** Read the full paper. Map their formal definitions to our
Cascade struct. Can we reformulate hdr.rs as a tropical attention module?
If so: we have the first hardware-accelerated tropical transformer.
On CPU. Deterministic. Using VPOPCNTDQ.

---

## 2. MULTI-INDEX HASHING: Sub-Linear Search (Norouzi et al., TPAMI 2014)

I cited it. Didn't connect it.

```
CURRENT CASCADE: linear scan with early rejection
  Stroke 1: scan ALL 1M candidates × 128 bytes = 128MB
  Rejection: ~84% rejected → 160K survive to stroke 2
  Still: O(N) scan on stroke 1

MULTI-INDEX HASHING:
  Split 16K-bit code into m substrings of 16K/m bits each
  Build hash table for each substring
  Query: look up each substring → candidate set
  Pigeonhole: if hamming(a,b) ≤ r, at least one substring matches within ⌊r/m⌋
  
  Result: 100-1000x speedup over linear scan on 1B 64-bit codes
  
  For 16K-bit codes: split into 8 substrings of 2K bits each
  8 hash tables, each indexed by 2K-bit prefix
  Query: 8 lookups → intersect candidate sets → exact verify survivors
```

**Why this matters:** The cascade gives us early rejection (constant factor).
Multi-index hashing gives us SUB-LINEAR candidate generation (algorithmic).
They compose: multi-index hashing produces candidates, cascade verifies them.

```
COMBINED:
  Step 0: Multi-index lookup → ~1000 candidates (sub-linear, not 1M)
  Step 1: Cascade stroke 1 on 1000 candidates (not 1M)
  Step 2: Cascade stroke 2 on survivors
  Step 3: Exact on final survivors
  
  Cost: ~1000 × 140ns + cascade overhead ≈ 0.2ms (vs ~2ms for 1M linear cascade)
  10x improvement from algorithmic change, not SIMD tricks.
```

**EXPLORE:** Can our Fingerprint<256> be split into 8 Fingerprint<32> for
multi-index? The cascade already reads stroke 1 from bytes [0..128] — 
that IS a substring. Build a hash index on stroke 1 prefixes.

---

## 3. DEGREE-QUANT: High-Degree Nodes Break Quantization (ICLR 2021)

Buried in the GNN section. Didn't connect to our architecture.

Tailor et al. found that nodes with hundreds of neighbors create extreme
activation ranges that break standard quantization. This is the HUB PROBLEM.

```
OUR VERSION OF THE HUB PROBLEM:
  Node with 500 neighbors → 500 encounter_toward() calls
  i8 accumulator saturates at ±127 after ~127 encounters
  After saturation: ALL further encounters are ignored
  The most connected nodes LOSE information first
  
  This is the OPPOSITE of what we want.
  Hub nodes should be the MOST informed, not the FIRST saturated.
```

**Why this matters:** The i8 accumulator in Plane is 8 bits. It saturates.
Hub nodes in the graph hit saturation earliest. Their alpha becomes all-1s
(everything defined) but their PRECISION is frozen. They can't learn anymore.

**FIX IDEAS:**
```
1. Dynamic threshold: hub nodes get higher threshold before alpha=1.
   More evidence needed to commit a bit. Slower crystallization.
   
2. Accumulator width scaling: i8 for leaf nodes, i16 for hubs, i32 for super-hubs.
   The Plane type could be generic over accumulator width.
   
3. Exponential decay: acc[k] *= 0.99 per round (forgetting).
   Recent encounters matter more. Old evidence fades.
   But this introduces float. Can we do integer decay? acc[k] -= acc[k]/128?
   
4. ReActNet's learned shifts: per-bit threshold instead of global.
   Some bit positions need more evidence than others.
   The threshold IS a learned parameter (trained by encounter patterns).
```

**EXPLORE:** Run message passing on a scale-free graph (power law degree distribution).
Measure: do hub nodes saturate? Does accuracy degrade for high-degree nodes?

---

## 4. 90% OF PARALLEL BF16 COMPUTATIONS DIVERGE (arXiv:2506.09501, 2025)

I cited this as "our advantage." Didn't explore what it MEANS for the industry.

```
THEIR FINDING:
  >90% of parallel BF16 computations produce different results
  from serial BF16 computations. On the SAME inputs.
  
  Because: (a + b) + c ≠ a + (b + c) in IEEE 754.
  Thread scheduling changes accumulation order.
  Different order → different rounding → different result.
  90%+ of the time.

WHAT THIS MEANS:
  Every GNN trained on GPU with BF16 produces NON-REPRODUCIBLE results.
  Not "slightly different." DIFFERENT. 90% of the time.
  
  Every benchmark comparison between GNN models is UNRELIABLE.
  The "state of the art" leaderboard is comparing noise.
  Nobody talks about this because everyone uses the same non-deterministic stack.
  
  We don't have this problem. Our results are bit-identical across runs.
  This isn't a feature we should mention in a footnote.
  This is the ENTIRE VALUE PROPOSITION for applications that need trust:
    - Medical diagnosis graphs
    - Financial risk assessment
    - Legal reasoning chains
    - Safety-critical systems
    - Regulatory compliance (auditable, reproducible)
```

**EXPLORE:** Can we PROVE determinism formally? Not just "we don't use float"
but a formal proof that our RL loop is a deterministic function of its inputs.
This would be a significant contribution to the formal verification literature.

---

## 5. STOCHASTIC ROUNDING AS EXPLORATION MECHANISM

Everyone treats stochastic rounding as a PROBLEM. But in RL, exploration
is the GOAL. What if we WANT non-determinism at a specific point?

```
EXPLOITATION: integer encounter_toward (deterministic)
  → greedy: push toward known good patterns
  
EXPLORATION: stochastic BF16 quantization at the boundary
  → when a distance is NEAR a band boundary, randomly assign
    to higher or lower band with probability proportional to proximity
  → this IS stochastic rounding applied to band classification
  → deterministic WITHIN the RL loop, stochastic AT the boundary
  
  The RL agent exploits deterministically.
  The cascade explores stochastically at boundaries.
  The alpha channel records which bits are in the boundary zone.
```

**Why this matters:** RL needs exploration-exploitation tradeoff.
Everyone uses epsilon-greedy or entropy bonus (float). We could use
stochastic band classification — exploration proportional to uncertainty.
The PLANE ALPHA CHANNEL tells us exactly which bit positions are uncertain.
Explore there. Exploit everywhere else. Naturally. No hyperparameter.

**EXPLORE:** Implement stochastic_expose(distance, rng) in Cascade.
Only affects boundary-zone distances. Compare convergence with
deterministic_expose vs stochastic_expose on a known-optimal task.

---

## 6. VPCLMULQDQ → FASTER SEAL VERIFICATION

I noted VPCLMULQDQ for XorField semiring. Didn't connect to Seal.

VPCLMULQDQ is carry-less multiplication = polynomial multiplication in GF(2).
This is the CORE OPERATION of CRC and polynomial hashing.

```
CURRENT SEAL: blake3 hash of (bits & alpha). Full 256-byte input.
  blake3 is fast (~3 cycles/byte on AVX-512) but it's OVERKILL for a 48-bit hash.
  
ALTERNATIVE: GF(2) polynomial hash using VPCLMULQDQ
  Input: 256 bytes (Plane bits & alpha)
  Operation: VPCLMULQDQ fold — same as CRC-32C but wider
  Output: 48-bit or 64-bit hash
  
  VPCLMULQDQ processes 64 bytes per cycle (512-bit vectors).
  256 bytes = 4 cycles. Plus fold = ~8 cycles total.
  
  blake3 on 256 bytes ≈ 256 × 3 = ~768 cycles.
  
  96x faster seal verification.
```

**Why this matters:** Seal verification happens on every encounter
(to detect Staunen). If verification is 96x faster, we can verify
MORE OFTEN — potentially every message passing round instead of
only at convergence checks.

**CAVEAT:** blake3 has cryptographic properties (collision resistance).
VPCLMULQDQ polynomial hash does NOT. It's sufficient for integrity
verification (detecting accidental changes) but not for adversarial
resistance. For our use case (detecting Staunen from genuine observations,
not adversarial inputs), polynomial hash suffices.

**EXPLORE:** Implement `seal_fast()` using VPCLMULQDQ. Benchmark against
`blake3::hash()`. If 50x+ faster: use fast seal for per-round checks,
blake3 seal for persistent storage verification.

---

## 7. HEXASTORE SEXTUPLE INDEXING → MASK-AWARE INDEX

Hexastore (VLDB 2008): 6 permutation indices for SPO triples.
SPO, SOP, PSO, POS, OSP, OPS.

I noted it. Didn't connect to our Mask system.

```
HEXASTORE 6 INDICES:        OUR 7 MASKS:
  SPO (given S, find PO)     SPO (all three match)
  SOP (given S, find OP)     SP_ (S and P match)
  PSO (given P, find SO)     S_O (S and O match)
  POS (given P, find OS)     _PO (P and O match)
  OSP (given O, find SP)     S__ (only S matches)
  OPS (given O, find PS)     _P_ (only P matches)
                             __O (only O matches)
                             
THEIR 6 = permutations of "given X, find Y,Z"
OUR 7 = combinations of "which planes participate in distance"

THEY ARE DUAL. Their index lookup IS our masked distance query.
  Hexastore "given P=KNOWS, find all (S,O) pairs" 
  = our "cascade query with mask=_P_, query.P = KNOWS-fingerprint"
```

**Why this matters:** Hexastore MATERIALIZES all 6 indices (sorted vectors).
We don't materialize — we compute distances at query time.
But for KNOWN EDGES (warm path), we could materialize Hexastore-style indices
sorted by BF16 truth value. That gives O(log N) lookup for warm queries.

```
COMBINED:
  WARM (materialized, Hexastore-style):
    SPO_index: sorted by (S, P, O, BF16_truth)
    _P__index: sorted by (P, BF16_truth)  ← "all relationships of type P"
    etc.
    
  HOT (computed, cascade):
    "Find similar nodes" → cascade scan with mask

  The BF16 truth value IS the sort key for the materialized index.
  Logarithmic quantization means sorting by BF16 ≈ sorting by similarity.
```

**EXPLORE:** Can we build a hybrid index that materializes the top-K
edges per Hexastore permutation, sorted by BF16? Then cold Cypher queries
hit the materialized index first, only falling through to cascade for
novel queries.

---

## 8. NARS REVISION AS TROPICAL MATRIX MULTIPLY

I asked this question in the spec. Never answered it.

```
NARS REVISION RULE:
  Given two independent evidence sources with truths <f1,c1> and <f2,c2>:
  
  f_new = (f1*c1*(1-c2) + f2*c2*(1-c1)) / (c1*(1-c2) + c2*(1-c1))
  c_new = (c1*(1-c2) + c2*(1-c1)) / (c1*(1-c2) + c2*(1-c1) + (1-c1)*(1-c2))

This has multiply-accumulate structure: f*c terms accumulated.

IN TROPICAL FORM (taking log):
  log(f_new) = log(f1*c1*(1-c2) + f2*c2*(1-c1)) - log(c1*(1-c2) + c2*(1-c1))
  
  If we use max-plus (tropical) approximation:
  log(a + b) ≈ max(log(a), log(b))     ← tropical addition
  log(a * b) = log(a) + log(b)          ← tropical multiplication
  
  Then revision becomes:
  log(f_new) ≈ max(log(f1)+log(c1)+log(1-c2), log(f2)+log(c2)+log(1-c1))
                - max(log(c1)+log(1-c2), log(c2)+log(1-c1))
```

**Why this matters:** If NARS revision IS tropical arithmetic on
log-truth-values, then BF16 exponent (which IS a logarithmic representation)
gives us revision AS EXPONENT ARITHMETIC.

```
BF16 exponent of frequency ≈ log2(frequency)
BF16 exponent of confidence ≈ log2(confidence)

Tropical revision on exponents:
  max(exp_f1 + exp_c1, exp_f2 + exp_c2)  ← integer max and add
  
  This is ONE VPMINSD + ONE VPADDD instruction.
  Not VDPBF16PS (float dot product).
  INTEGER tropical arithmetic on BF16 exponents.
  
  NARS revision at 32 truth values per SIMD instruction.
  Deterministic. Integer. No float.
```

**EXPLORE:** Formalize the tropical approximation of NARS revision.
What's the error bound vs exact float revision? If < 1 BF16 mantissa
bit of error: the approximation is FREE (within BF16 precision anyway).

---

## 9. LEARNED PER-BIT THRESHOLDS (ReActNet, ECCV 2020)

ReActNet achieves 69.4% ImageNet top-1 with binary networks using
learned activation shifts: RSign(x) = sign(x + β) where β is learned.

```
OUR CURRENT:
  alpha[k] = |acc[k]| > THRESHOLD     ← global threshold, same for all bits
  
REACTNET:
  activation[k] = sign(acc[k] + shift[k])  ← per-bit learned shift
  
  shift[k] IS a threshold. It just moves the decision boundary per position.
  Some bits need MORE evidence to commit. Others less.
  
  THE SHIFT IS LEARNABLE FROM ENCOUNTER HISTORY:
  If bit k frequently flips between encounters → high uncertainty → high threshold
  If bit k always agrees with evidence → low uncertainty → low threshold
  
  shift[k] = some function of acc[k]'s variance over recent encounters
  
  This makes alpha ADAPTIVE PER BIT. Not just on/off.
  Uncertain bits stay undefined longer. Certain bits commit faster.
```

**Why this matters:** Currently our alpha channel is binary (0 or 1).
Per-bit thresholds make it a SPECTRUM. The alpha channel becomes a
confidence map, not just a defined/undefined mask. This directly
improves the BF16 mantissa precision — we can read CONFIDENCE
from the alpha channel, not just PRESENCE.

**EXPLORE:** Track variance of acc[k] over last N encounters.
High variance → high threshold. Low variance → low threshold.
Does this improve distance precision? Does convergence speed up?

---

## 10. MOHRI'S WEIGHTED TRANSDUCERS → CYPHER AS SEMIRING AUTOMATON

The NYU semiring paper (Mohri) studies semirings in weighted finite-state
transducers for NLP: speech recognition, parsing, translation.

```
WEIGHTED TRANSDUCER:
  States = graph nodes
  Transitions = edges with semiring weights
  Composition = semiring matrix multiply
  
  Query = input string (Cypher pattern)
  Traversal = transducer composition (semiring mxv chain)
  Result = output with accumulated semiring weight

OUR DATAFUSION PLANNER IS ALREADY A WEIGHTED TRANSDUCER:
  States = BlasGraph nodes
  Transitions = edges with HdrScalar / BF16 weights
  Cypher MATCH = transducer path
  WHERE clause = semiring threshold
  
  The Cypher-to-semiring compilation is transducer construction.
  Query execution is transducer composition.
```

**Why this matters:** Transducer theory gives us OPTIMAL composition
algorithms. Mohri's shortest-distance algorithm generalizes Dijkstra
to arbitrary semirings. We could import 30 years of automata theory
into our query optimizer.

**EXPLORE:** Read Mohri's "Semiring Frameworks and Algorithms for
Shortest-Distance Problems" (Journal of Automata, Languages and Combinatorics).
Map his algorithms to our HdrSemiring dispatch. Can his optimal
composition strategy improve our multi-hop query performance?

---

## PRIORITY RANKING

```
#   THREAD                          POTENTIAL IMPACT    EFFORT
──────────────────────────────────────────────────────────────────
1.  Multi-index hashing             10x search speed     medium
2.  NARS revision as tropical       deterministic NARS   low
3.  Tropical attention = cascade    reframe as transformer  low (conceptual)
4.  Hub saturation (Degree-Quant)   correctness fix       medium
5.  Learned per-bit thresholds      precision improvement  medium
6.  90% divergence → value prop     marketing/paper        low
7.  VPCLMULQDQ fast seal            96x faster verify      low
8.  Hexastore mask-aware index      O(log N) warm queries  medium
9.  Stochastic exploration          RL convergence         medium
10. Mohri transducers               query optimization     high (research)
```

Thread 1 (multi-index) and thread 2 (tropical NARS) have the highest
return per effort. Thread 3 (tropical attention) is a reframing that
costs zero code but opens a publication path.

Thread 4 (hub saturation) is a CORRECTNESS concern that needs investigation
before production deployment of message passing.
