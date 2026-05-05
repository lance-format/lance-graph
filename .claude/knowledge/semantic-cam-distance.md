# Semantic CAM distance, not Hamming distance

Status: contextual architecture note. This document corrects the distance model for the deterministic COCA/CAM projection architecture.

## One-sentence thesis

Because the field uses deterministic content-addressable memory over a calibrated 4096-entry COCA codebook, the primary distance is not raw Hamming distance over bits. The primary distance is semantic CAM distance: distance between addressed codebook meanings, projection masks, witness angles, and historical resonance states.

In shorthand:

```text
not:
  confidence = Hamming distance over arbitrary bitstrings

but:
  confidence = semantic CAM distance over deterministic COCA addresses
             + projection compatibility
             + witness compatibility
             + entropy budget
             + historical support
```

---

## Why Hamming distance is insufficient

A 16,384-bit awareness plane may be stored as 256 `u64` words, but that does not make raw XOR/popcount the meaning metric.

Raw Hamming answers:

```text
how many bits differ?
```

Semantic CAM distance answers:

```text
which content-addressed meanings differ?
which projection masks differ?
which witness angles differ?
which historical resonance families differ?
```

The bits are not anonymous. They are addresses in a deterministic CAM/COCA codebook schema.

Therefore raw Hamming can be used only as a mechanical primitive or cheap sanity check, not as the core semantic distance.

---

## Deterministic content-addressed basis

The calibrated basis:

```text
4096 COCA codebook entries
  frequent English speaker vocabulary / meaning atoms

4096 x 4096 factor-pair field
  deterministic content-addressable pair space

SPO/ABC 2^3 projection grammar
  deterministic query/completion masks

ReasoningWitness64
  NARS truth + style + source + projection transition angle

AwarenessPlane16K
  materialized cycle resonance plane
```

The address of a bit/coordinate is meaningful because it is produced by the codebook and projection grammar.

So the distance must preserve that structure.

---

## Semantic CAM distance components

### 1. Codebook address distance

Compare the addressed COCA codebook coordinates, not anonymous bits.

```text
d_codebook(a, b)
```

Possible implementations:

```text
exact same codebook id        -> distance 0
same role family / cluster    -> small distance
same semantic neighborhood    -> medium distance
unrelated codebook address    -> large distance
```

This requires a codebook metric or table:

```rust
pub trait CodebookMetric {
    fn distance_q8(&self, lhs: u16, rhs: u16) -> u8;
    fn compatibility_q8(&self, lhs: u16, rhs: u16) -> u8;
}
```

The 4096-entry COCA basis should eventually carry cluster/family metadata so distance is content-aware.

---

### 2. Pairwise CAM address distance

The field address is a pair:

```text
(a, b) in 4096 x 4096
```

Distance should compare both axes under the codebook metric:

```text
d_pair((a,b), (a',b'))
  = combine(d_codebook(a,a'), d_codebook(b,b'))
```

For SPO/ABC, this is lens-dependent:

```text
Subject axis distance
Predicate/relation axis distance
Object/completion axis distance
```

---

### 3. Projection compatibility

Two addresses may be close only under compatible projection masks.

```text
ABC vs ABC     = direct comparison
AB_ vs ABC     = compatible if ABC answers the AB_ query
AB_ vs A_C     = compatible only via inverse/triadic closure
Counterfactual vs Evidence = scenario-compatible, not fact-compatible
```

So projection distance is not bit distance. It is query-role compatibility.

---

### 4. Witness-angle distance

`ReasoningWitness64` carries the transition angle.

Compare by unpacked semantic fields:

```text
NARS frequency/confidence delta
polarity mismatch
relation_family compatibility
projection_kind compatibility
temporal_bucket compatibility
thinking_style compatibility
perturbation_class compatibility
source_lane compatibility
generation/replay distance
```

This is semantic because each subfield has different meaning.

Raw `u64` numeric distance is wrong.

---

### 5. Resonance-history distance

A CAM coordinate should also be compared by its historical behavior.

```text
has this address family historically collapsed into the same candidate?
has it produced contradictions?
has it been stable across cycles?
has it required widened theta?
has sigma propagation stayed inside bounds?
```

This yields a historical semantic distance:

```text
d_history = distance(current resonance lineage, historical resonance lineage)
```

---

## Revised confidence model

Replace any raw Hamming-centered formula with:

```text
semantic_cam_confidence
  = codebook_address_compatibility
  x pairwise_cam_compatibility
  x projection_compatibility
  x witness_compatibility
  x entropy_budget_remaining
  x sigma_stability
  x historical_confidence
  x source_diversity_adjustment
  x contradiction_penalty
```

Hamming/popcount may still appear as an implementation detail:

```text
bitplane scan / cheap survivor prefilter / hardware acceleration
```

But it should be wrapped by semantic CAM interpretation before confidence or ontology promotion.

---

## SemanticCamDistanceCertificate

Proposed DTO:

```rust
pub struct SemanticCamDistanceCertificate {
    pub thought_id: u64,
    pub cycle_id: u64,
    pub candidate_id: u64,

    pub codebook_version: u32,
    pub projection_version: u32,
    pub witness_layout_version: u32,

    pub codebook_compat_q8: u8,
    pub pairwise_compat_q8: u8,
    pub projection_compat_q8: u8,
    pub witness_compat_q8: u8,
    pub history_compat_q8: u8,

    pub entropy_budget_remaining_q8: u8,
    pub sigma_stability_q8: u8,
    pub historical_confidence_q8: u8,
    pub contradiction_penalty_q8: u8,

    pub final_distance_q8: u8,
    pub final_confidence_q8: u8,
    pub replay_ref: u64,
}
```

This certificate supersedes any bare `hamming_distance` confidence artifact.

---

## Entropy budget relationship

The entropy-budget note says the main remaining problem is entropy accumulated over `4096 x 4096` deterministic superposition.

Semantic CAM distance is the metric used to measure whether that entropy still permits a stable collapse.

```text
EntropyBudgetCertificate:
  how much uncertainty accumulated?

SemanticCamDistanceCertificate:
  how far is this candidate from historically/semantically compatible addresses?
```

They should be siblings, or one may eventually embed the other.

---

## Relationship to JC scaffold

The JC math still matters, but not as raw Hamming-is-signal logic.

Suggested mapping:

```text
Jirak / Berry-Esseen:
  residual finite-sample and weak-dependence calibration after semantic CAM addressing

Weyl / phi:
  alias-resistant traversal of CAM address space

gamma / phi transform:
  coordinate regularization for folding pressure

Köstenberger-Stark:
  sigma aggregation concentration

Düker-Zoubouloglou:
  Hilbert-space / bundle convergence

EWA-Sandwich:
  multi-hop sigma path propagation

Pearl:
  SPO/ABC 2^3 projection grammar
```

So the raw-bit problem is not the center. The center is semantic distance over deterministic content addresses, with JC math calibrating residual entropy and stability.

---

## Practical algorithm sketch

```text
1. Current ThoughtCycleSoA row supplies AwarenessPlane16K, GrammarMarkovLens64, ReasoningWitness64[], sigma/theta, thought_id/cycle_id.
2. Decode active CAM addresses into codebook-addressed factor pairs.
3. Compare candidate address families using CodebookMetric, not anonymous bit positions.
4. Apply TriadicProjection compatibility.
5. Compare ReasoningWitness64 by semantic subfields.
6. Pull historical resonance behavior for the same/compatible CAM addresses.
7. Apply entropy budget and sigma/theta stability.
8. Emit SemanticCamDistanceCertificate.
9. Only NARS/AriGraph/ontology gate may promote after this certificate.
```

---

## Minimum viable implementation

### PR 1: CodebookMetric trait

```rust
pub trait CodebookMetric {
    fn distance_q8(&self, lhs: u16, rhs: u16) -> u8;
    fn compatibility_q8(&self, lhs: u16, rhs: u16) -> u8;
}
```

Start with identity/cluster distance, then upgrade to learned/calibrated COCA neighborhoods.

### PR 2: Projection compatibility table

Use `TriadicProjection` compatibility, not bit overlap.

### PR 3: Witness semantic distance

Unpack `ReasoningWitness64` and compare field-wise.

### PR 4: CAM address extraction

From `AwarenessPlane16K` and lens/witness metadata, recover active semantic CAM address families.

### PR 5: SemanticCamDistanceCertificate

Add certificate and tests.

### PR 6: Replace bare Hamming references in confidence docs

Any doc mentioning Hamming/popcount as confidence should clarify that it is a survivor prefilter only.

---

## Suggested wording

Use this:

```text
Because the awareness plane is a deterministic content-addressable projection over a calibrated 4096-entry COCA codebook, distance is semantic CAM distance, not raw Hamming distance. Popcount may accelerate survivor scans, but confidence is computed over codebook-address compatibility, triadic projection compatibility, ReasoningWitness64 semantic-field compatibility, historical resonance behavior, entropy budget, sigma stability, and NARS/AriGraph support.
```

Avoid this:

```text
We compare meanings by Hamming distance.
```

Better:

```text
We use deterministic CAM addresses and compute semantic distance over the COCA codebook schema.
```
