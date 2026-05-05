# Entropy budget after deterministic codebook x codebook superposition

Status: contextual architecture note. This document refines the folding-confidence model after the `4096 x 4096 -> 16,384` deterministic CAM/COCA projection claim.

## One-sentence thesis

Once the full `codebook x codebook` field is deconstructed into deterministic triadic superposition over a fixed 4096 COCA basis, the first-order access problem is no longer weak random overlap. The remaining problem is entropy accumulation across the `4096 x 4096` pairwise address field, its witnesses, temporal replay, and projection folds.

In shorthand:

```text
solved:
  local access ambiguity by deterministic codebook x codebook projection

remaining:
  entropy accumulated over 4096 x 4096 pairwise interactions
  + witness uncertainty
  + temporal/replay drift
  + projection/folding collisions
```

---

## What changed

Earlier confidence framing treated bitplane overlap as if the first gate were mostly a weak-dependence signal test:

```text
is this overlap real or random?
```

That remains useful for calibration, but it is no longer the central obstacle if the system has a fixed, deterministic codebook projection contract:

```text
4096 COCA atoms
x 4096 COCA atoms
x SPO/ABC 2^3 projection masks
x ReasoningWitness64 transition angle
-> AwarenessPlane16K
```

The field is not a random embedding soup. It is a deterministic resonance projection over a calibrated basis.

Therefore the main confidence problem becomes:

```text
how much entropy has accumulated while folding the full pairwise meaning field
into a compact, replayable 16,384-bit cycle state?
```

---

## New problem statement

For a current context `x`, confidence should be tested as:

```text
confidence(x)
  = deterministic_projection_validity(x)
  x entropy_budget_remaining(x)
  x historical_confidence(x)
  x NARS witness support(x)
  x sigma stability(x)
  x source/replay support(x)
```

The old question:

```text
is the bit overlap statistically meaningful?
```

becomes the narrower subquestion:

```text
after deterministic folding, is the remaining entropy low enough to trust this collapse?
```

---

## Sources of entropy

### 1. Pairwise field entropy

The base field has:

```text
4096 x 4096 = 16,777,216 pairwise addresses
```

Even with deterministic addressing, many factor pairs can project into the same 16K plane region. The question is not whether the map is arbitrary. The question is how much collision/alias pressure accumulated.

Track:

```text
active_pair_count
active_projection_count
collision_count
collision_mass
survivor_tile_count
```

---

### 2. Projection entropy

SPO/ABC `2^3` masks can express different question states:

```text
ABC
AB_
A_C
_BC
AOnly
BOnly
COnly
Background
```

Entropy increases when multiple projection kinds compete for the same plane/candidate.

Track:

```text
projection_distribution
projection_entropy
projection_conflict_mass
```

---

### 3. Witness entropy

`ReasoningWitness64` carries NARS truth, polarity, relation family, projection kind, temporal bucket, thinking style, perturbation class, source lane, and generation.

Entropy increases when witnesses disagree:

```text
support vs contradiction
high source concentration
style disagreement
temporal mismatch
projection mismatch
relation-family mismatch
```

Track:

```text
support_mass
contradiction_mass
witness_entropy
source_entropy
style_entropy
temporal_entropy
```

---

### 4. Temporal entropy

A 4096-cycle causal history helps confidence, but also adds drift.

Track:

```text
history_depth
replay_distance
cycle_age_decay
temporal_skew
Chronos forecast uncertainty
```

Backward replay can support evidence. Forward projection can guide traversal. They must not be mixed.

```text
past entropy = evidence uncertainty
future entropy = implication uncertainty
```

---

### 5. Sigma/theta entropy

Sigma measures uncertainty geometry. Theta controls aperture.

Entropy increases when:

```text
sigma path growth approaches bound
theta must widen to admit candidate
hydration width grows
candidate survives only under exploratory aperture
```

Track:

```text
sigma_log_norm_growth
sigma_bound_margin
theta_accept
theta_width
```

---

## EntropyBudgetCertificate

Proposed DTO:

```rust
pub struct EntropyBudgetCertificate {
    pub thought_id: u64,
    pub cycle_id: u64,
    pub candidate_id: u64,

    // deterministic projection validity
    pub codebook_version: u32,
    pub projection_version: u32,
    pub witness_layout_version: u32,
    pub deterministic_projection_ok: bool,

    // pairwise-field entropy
    pub active_pair_count: u32,
    pub survivor_tile_count: u16,
    pub collision_mass_q8: u8,
    pub pairwise_entropy_q8: u8,

    // projection/witness entropy
    pub projection_entropy_q8: u8,
    pub witness_entropy_q8: u8,
    pub source_entropy_q8: u8,
    pub style_entropy_q8: u8,
    pub temporal_entropy_q8: u8,

    // support/contradiction
    pub support_mass_q8: u8,
    pub contradiction_mass_q8: u8,

    // sigma/theta
    pub sigma_margin_q8: u8,
    pub theta_accept_q8: u8,
    pub theta_width_q8: u8,

    // history
    pub causal_history_depth: u16, // e.g. 4096
    pub historical_confidence_q8: u8,

    // result
    pub entropy_budget_remaining_q8: u8,
    pub final_confidence_q8: u8,
    pub replay_ref: u64,
}
```

---

## Confidence composition

The confidence path should be rewritten as:

```text
projection_ok
  AND sigma_guardrail_ok
  AND replay_lineage_ok
  AND entropy_budget_remaining >= threshold
```

Then compute:

```text
final_confidence
  = historical_confidence
  x NARS_support
  x sigma_margin
  x entropy_budget_remaining
  x source_diversity_adjustment
  x contradiction_penalty
```

Where:

```text
entropy_budget_remaining
  = 1 - combined_entropy(
        pairwise_entropy,
        projection_entropy,
        witness_entropy,
        temporal_entropy,
        sigma_theta_entropy
      )
```

The important change:

```text
Jirak/Berry-Esseen-style calibration is now a noise-floor component,
not the whole confidence model.
```

---

## Role of the JC scaffold after this correction

The JC math scaffold still matters, but its role shifts.

```text
Jirak / Berry-Esseen:
  finite-sample / weak-dependence noise-floor calibration for residual overlap and entropy estimates

Weyl / phi:
  alias-resistant traversal and sampling over the pairwise field

gamma / phi transform:
  coordinate regularization for folding pressure

Köstenberger-Stark:
  concentration behavior for sigma aggregation

Düker-Zoubouloglou:
  Hilbert-space bundle convergence / partial-sum behavior

EWA-Sandwich:
  multi-hop sigma propagation and geometric stability

Pearl:
  SPO/ABC 2^3 causal projection grammar
```

So the confidence certificate becomes:

```text
deterministic codebook projection
+ entropy budget
+ JC residual calibration
+ NARS/AriGraph historical support
```

---

## Entropy pressure algorithm sketch

```text
1. Project current context x into AwarenessPlane16K using fixed codebook/projection/witness rules.
2. Count active factor-pair contributions before folding.
3. Estimate collision/alias mass into the 16K plane.
4. Compute projection entropy over ABC/SPO 2^3 masks.
5. Compute witness entropy over ReasoningWitness64 subfields.
6. Compute temporal entropy over the 4096-cycle history window.
7. Compute sigma margin and theta aperture cost.
8. Pull historical confidence from matching folded trajectories.
9. Produce EntropyBudgetCertificate.
10. Allow NARS/AriGraph/ontology promotion only if entropy budget remains sufficient.
```

---

## Practical gates

### Strong candidate

```text
projection_ok = true
sigma_guardrail_ok = true
entropy_budget_remaining high
historical_confidence high
contradiction_mass low
```

### Novel candidate

```text
projection_ok = true
entropy_budget_remaining medium
historical_confidence low
NARS support emerging
```

Treat as candidate, not fact.

### Entropy-overloaded candidate

```text
projection_ok = true
collision/projection/witness entropy high
theta widened too much
sigma margin low
```

Treat as replay-only or request more context.

### Future implication

```text
forward Chronos projection
entropy may be high
not evidence
```

Use for prefetch/style/sweep, not ontology promotion.

---

## Relationship to folding confidence

This note refines `mathematical-folding-confidence.md`.

Old mental model:

```text
confidence = folding distance x historical confidence
```

Refined model:

```text
confidence = deterministic projection validity
           x entropy budget remaining after codebook folding
           x historical confidence
           x witness/NARS support
           x sigma stability
```

The folding distance is still useful, but the decisive question is now entropy budget over the full deterministic `4096 x 4096` pairwise field.

---

## Suggested architecture wording

Use this:

```text
After deconstructing the full 4096x4096 codebook-pair field into deterministic SPO/ABC superposition, the first-order random-overlap problem is no longer the main obstacle. The remaining confidence problem is entropy accumulation: how much collision, projection conflict, witness disagreement, temporal drift, and sigma/theta uncertainty was introduced while folding the pairwise meaning field into the 16,384-bit awareness plane. Confidence should therefore be certified by an EntropyBudgetCertificate that combines deterministic projection validity, residual JC-style noise calibration, NARS witness support, sigma margin, historical confidence, and replay/source diversity.
```

Avoid this:

```text
The field is automatically confident because the projection is deterministic.
```

Better:

```text
The projection is deterministic; confidence depends on the entropy budget left after folding.
```
