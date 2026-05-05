# Mathematical folding confidence over full meaning space and causal history

Status: contextual architecture note. This document proposes the confidence-testing frame; it does not claim a finished theorem or implementation.

## One-sentence thesis

Given a fixed COCA/CAM codebook, deterministic projection grammar, replayable ThoughtCycleSoA lineage, and a 4096-cycle causal history, candidate confidence can be tested as a folding-distance problem: compare the current context projection against historical folded trajectories, then combine folding distance, NARS truth mass, sigma stability, theta aperture, and source entropy into a calibrated confidence certificate.

In shorthand:

```text
confidence(context x)
  ~= folding_distance(x, historical meaning field)
   x historical confidence
   x NARS witness mass
   x sigma stability
   x source/replay support
```

The goal is not mystical certainty. The goal is a replayable, testable confidence certificate over the compressed meaning manifold.

---

## Why folding confidence belongs here

The architecture already has the ingredients:

```text
AwarenessPlane16K
  compact CAM/COCA activation state

GrammarMarkovLens64
  parse/context interpretation lens

ReasoningWitness64
  NARS/style/source/projection transition angle

ThoughtCycleSoA
  cycle lineage and witness slices

AriGraph semantic/episodic memory
  historical fact and episode support

Sigma/theta
  uncertainty geometry and dynamic aperture

Chronos/perturbation
  forward implication forecast
```

A folding-confidence algorithm should not live outside this spine. It should consume the same thought-cycle rows and produce a replayable certificate.

---

## The folding problem

For a current context `x`, we have:

```text
x_plane       = AwarenessPlane16K
x_lens        = GrammarMarkovLens64
x_witnesses   = ReasoningWitness64[]
x_sigma       = sigma geometry
x_theta       = aperture state
x_projection  = TriadicProjection
```

Historical memory provides:

```text
H = {cycle rows over 4096 causal-history depth}
```

The question:

```text
How close is x to historically stable meaning trajectories under the same or compatible lenses?
```

The answer is not a plain Hamming distance only. It is a folded distance over:

```text
activation overlap
projection compatibility
NARS truth compatibility
style/perturbation compatibility
sigma-path stability
source/replay support
historical outcome confidence
```

---

## Candidate folding-distance components

### 1. Plane distance

The compact baseline:

```text
d_plane = normalized_hamming_or_xor_distance(x_plane, h_plane)
```

For bitplanes:

```text
similarity = 1 - popcount(xor(x, h)) / 16384
```

This gives a cheap first-stage survivor filter.

---

### 2. Projection distance

The current question lens must be compatible with the historical projection.

```text
d_projection = distance(x_projection, h_projection)
```

Examples:

```text
AB_ vs AB_     = close
AB_ vs ABC     = compatible if historical full triple answers the query
AB_ vs _BC     = farther, unless same triad completes through a known inverse
counterfactual vs evidence = not promotable, even if close
```

---

### 3. Witness distance

Compare `ReasoningWitness64` fields by semantic subfield, not raw integer distance.

Possible weighted components:

```text
NARS frequency delta
NARS confidence delta
polarity mismatch
relation_family mismatch
projection_kind mismatch
temporal_bucket mismatch
thinking_style mismatch
perturbation_class mismatch
source_lane compatibility
generation/replay distance
```

This can be summarized as:

```text
d_witness = weighted_distance(unpack(x_witness), unpack(h_witness))
```

---

### 4. Sigma folding distance

Use the sigma propagation kernel as geometry.

```text
sigma_path = ewa_sandwich(M, sigma_seed)
growth     = log_norm_growth(seed, sigma_path)
bound      = pillar_5plus_bound(n_hops)
```

Hard condition:

```text
growth <= bound
```

Soft confidence factor:

```text
sigma_stability = 1 - clamp(growth / bound, 0, 1)
```

The hard bound protects the field; the soft factor contributes confidence.

---

### 5. Historical confidence

Historical rows/candidates carry their own confidence:

```text
historical_confidence
  = NARS confidence mass
  x AriGraph support
  x source diversity
  x replay stability
  x outcome validation if available
```

This prevents a close match to a historically weak pattern from becoming overconfident.

---

## Folding confidence certificate

Proposed DTO:

```rust
pub struct FoldingConfidenceCertificate {
    pub thought_id: u64,
    pub cycle_id: u64,
    pub candidate_id: u64,

    pub plane_similarity_q8: u8,
    pub projection_compat_q8: u8,
    pub witness_compat_q8: u8,
    pub sigma_stability_q8: u8,
    pub historical_confidence_q8: u8,
    pub source_entropy_q8: u8,

    pub support_count: u16,
    pub contradiction_count: u16,
    pub historical_window: u16, // e.g. 4096

    pub final_confidence_q8: u8,
    pub replay_ref: u64,
}
```

Possible scalar composition:

```text
final_confidence
  = plane_similarity
  x projection_compat
  x witness_compat
  x sigma_stability
  x historical_confidence
  x source_entropy_adjustment
  x contradiction_penalty
```

This should be calibrated, not assumed.

---

## Historical 4096 causal-history depth

A 4096-cycle history is a natural companion to the 4096 codebook basis.

Use it as:

```text
history window = last 4096 thought cycles or causal hops
```

Possible access tiers:

```text
64 cycles:
  hot local confidence

512 cycles:
  recent episode confidence

4096 cycles:
  causal-history confidence

AriGraph semantic memory:
  stable fact confidence beyond window
```

The 4096 history does not need to be scanned naively. Use bitplane survivor filtering first, then witness/projection/sigma tests.

---

## Folding pipeline

```text
Current ThoughtCycleSoA row
  -> candidate projection
  -> XOR/popcount survivor search over historical AwarenessPlane16K
  -> projection compatibility filter
  -> ReasoningWitness64 compatibility scoring
  -> sigma propagation stability check
  -> historical confidence lookup
  -> contradiction/source entropy adjustment
  -> FoldingConfidenceCertificate
  -> NARS/AriGraph revision or ontology promotion gate
```

This makes confidence a replayable artifact rather than a naked float.

---

## Relation to NARS

NARS frequency/confidence should not be replaced by folding confidence.

Instead:

```text
NARS truth = local belief revision signal
folding confidence = historical/manifold compatibility signal
```

The two should cross-check each other.

Examples:

```text
high NARS confidence + high folding confidence:
  strong candidate

high NARS confidence + low folding confidence:
  possible novelty, contradiction, or parse/context error

low NARS confidence + high folding confidence:
  familiar but weakly supported candidate; request evidence

low NARS confidence + low folding confidence:
  exploratory/replay-only
```

---

## Relation to theta

Theta should modulate how aggressively candidates survive folding.

```text
theta_accept high:
  require tighter folding distance / higher historical confidence

theta_accept low:
  allow exploratory candidates but mark replay-only or scenario

theta_width high:
  broader Gaussian/folding neighborhood

theta_width low:
  sharper historical match requirement
```

So theta is the aperture over the folding confidence test.

---

## Relation to Chronos / future splats

Backward folding uses evidence replay.

Forward folding uses forecasted implication replay.

```text
backward folding:
  compare x against historical folded trajectories
  can support evidence confidence

forward folding:
  compare x against forecasted perturbation trajectories
  can guide prefetch/style/scenario
  must not promote facts directly
```

Possible future confidence artifact:

```rust
pub struct FutureFoldingImplication {
    pub candidate_id: u64,
    pub forecast_horizon: u16,
    pub folding_similarity_q8: u8,
    pub forecast_uncertainty_q8: u8,
    pub suggested_style: u8,
    pub suggested_theta_delta: i8,
    pub replay_ref: u64,
}
```

---

## Mathematical families to investigate

This section lists families likely relevant for a rigorous version of folding confidence. It is a research checklist, not a citation claim.

```text
metric learning on binary hypervectors
Hamming / Jaccard / angular distance on sparse bitplanes
random indexing / vector-symbolic architectures
concentration bounds for random projections
kernel density over binary manifolds
Gaussian/RBF kernels over Hamming distance
Mahalanobis-like distance using sigma codebook geometry
SPD manifold distance and log-Euclidean metrics
NARS truth revision and evidential confidence
empirical Bernstein / Berry-Esseen style finite-sample bounds
sequential confidence / martingale bounds
spectral gap and projector stability tests
Chronos/time-series quantile confidence for forward implication
```

The engineering rule:

```text
start with measurable, replayable distance components;
then prove/calibrate their composition.
```

---

## Minimum viable implementation

### PR 1: define certificate

Add `FoldingConfidenceCertificate` to the contract or cognitive-shader-driver boundary.

### PR 2: bitplane survivor distance

Implement:

```rust
fn plane_similarity(a: &AwarenessPlane16K, b: &AwarenessPlane16K) -> u8
```

using XOR + popcount.

### PR 3: witness compatibility

Implement field-aware distance over `ReasoningWitness64`, not raw `u64` subtraction.

### PR 4: projection compatibility

Implement compatibility table for `TriadicProjection`.

### PR 5: sigma stability factor

Use existing sigma propagation guardrail and expose soft stability score.

### PR 6: historical confidence aggregation

Aggregate over the last N cycles, default tiers:

```text
64, 512, 4096
```

### PR 7: certificate gate

Hydration candidates must carry a folding certificate before ontology promotion.

---

## Suggested architecture wording

Use this:

```text
Confidence is tested as a folding-distance certificate over the replayable meaning field. The current AwarenessPlane16K is compared against historical folded trajectories from the same 4096-codebook CAM space. Projection compatibility, ReasoningWitness64 truth/style/source compatibility, sigma-path stability, historical NARS confidence, source entropy, and contradiction mass are combined into a calibrated FoldingConfidenceCertificate. Backward folding can support evidence confidence; forward folding through Chronos-style perturbation can guide traversal and scenario planning but cannot promote facts directly.
```

Avoid this:

```text
The system just knows confidence over all meaning.
```

Better:

```text
The system computes replayable confidence certificates over its encoded meaning manifold.
```
