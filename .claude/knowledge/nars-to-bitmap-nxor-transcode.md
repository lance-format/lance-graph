# NARS -> bitmap NXOR transcode

Status: contextual architecture note. This document proposes the transcode needed for spatial math bundle superposition NARS. It does not claim a finished algebra or implementation.

## One-sentence thesis

To make spatial bundle superposition NARS work in the CAM/COCA field, NARS truth must be transcoded into bitmap operations: support and contradiction become semantic CAM bitplanes, compatibility becomes NXOR over addressed meaning/state bits, and `ReasoningWitness64` preserves the quantized NARS truth/style/source angle for replay and revision.

In shorthand:

```text
NARS truth/revision
  -> ReasoningWitness64
  -> support/contradiction CAM bitplanes
  -> NXOR semantic compatibility
  -> planar splat/bundle pressure
  -> certificate / replay / NARS revision
```

---

## Why NXOR, not XOR

XOR is useful for difference and survivor filtering:

```text
XOR(a, b) = disagreement mask
```

NXOR is useful for compatibility/resonance:

```text
NXOR(a, b) = agreement mask
```

For semantic CAM distance, agreement is often the useful pressure carrier:

```text
candidate plane agrees with historical support plane
candidate plane agrees with projection-compatible witness plane
candidate plane agrees with style/source-compatible history
```

So the spatial bundle needs both:

```text
XOR  -> where do planes disagree?
NXOR -> where do planes resonate / match?
```

NXOR is the natural bitmap operation for "same meaning pressure under this lens".

---

## What gets transcoded from NARS

NARS truth has at least:

```text
frequency
confidence
```

The hot field also needs:

```text
polarity: support vs contradiction
relation family
projection kind
temporal bucket
thinking style
perturbation class
source lane
generation
```

These are carried in `ReasoningWitness64`:

```rust
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReasoningWitness64(pub u64);

pub type CausalEdge64 = ReasoningWitness64;
pub type GrammarWitness64 = ReasoningWitness64;
pub type ThinkingWitness64 = ReasoningWitness64;
```

Proposed layout:

```text
bits  0..6   evidence_mantissa      7 bits
bit      7   polarity               1 bit

bits  8..15  nars_frequency         8 bits
bits 16..23  nars_confidence        8 bits

bits 24..29  relation_family        6 bits
bits 30..33  projection_kind        4 bits
bits 34..39  temporal_bucket        6 bits

bits 40..45  thinking_style         6 bits
bits 46..51  perturbation_class     6 bits
bits 52..57  source_lane            6 bits
bits 58..63  generation             6 bits
```

This is the scalar witness. The bitmap transcode turns it into plane pressure.

---

## NARS bitmap planes

Separate support and contradiction. Never let them silently cancel inside one anonymous vector.

```rust
pub struct NarsBitmapPlanes {
    pub support: AwarenessPlane16K,
    pub contradiction: AwarenessPlane16K,
    pub uncertain: AwarenessPlane16K,
    pub forecast: AwarenessPlane16K,
    pub counterfactual: AwarenessPlane16K,
}
```

Optional q8 accumulators can exist beside bitplanes:

```rust
pub struct NarsQ8Planes {
    pub support_q8: Box<[u8; 16_384]>,
    pub contradiction_q8: Box<[u8; 16_384]>,
    pub confidence_q8: Box<[u8; 16_384]>,
}
```

The bitplane path is the hot lane. The q8 path is a richer pressure lane.

---

## Transcode rule

A conceptual transcode:

```text
NARS statement + projection + CAM address
  -> ReasoningWitness64
  -> channel selection
  -> bit/q8 deposition into semantic CAM plane
```

Channel selection:

```text
positive polarity + evidence/current source:
  support plane

negative polarity / contradiction:
  contradiction plane

low confidence or ambiguous projection:
  uncertain plane

future/Chronos source:
  forecast plane

counterfactual style/projection:
  counterfactual plane
```

The exact statement remains in replay. The hot plane receives only the compact pressure.

---

## NXOR semantic compatibility

Given a candidate plane `x` and historical/support plane `h`:

```text
agree = NXOR(x, h)
disagree = XOR(x, h)
```

But raw bit agreement is not the whole semantic metric. NXOR should be applied over deterministic CAM addresses and then interpreted through:

```text
COCA codebook compatibility
TriadicProjection compatibility
ReasoningWitness64 compatibility
source/style compatibility
sigma/theta context
entropy budget
```

So:

```text
NXOR = hardware/bitmap compatibility primitive
semantic CAM distance = interpreted compatibility over addressed meaning
```

---

## Possible scoring sketch

```text
support_match
  = weighted_popcount(NXOR(candidate_plane, support_plane), semantic_weights)

contradiction_match
  = weighted_popcount(NXOR(candidate_plane, contradiction_plane), semantic_weights)

confidence
  = support_match
  x witness_compatibility
  x sigma_stability
  x entropy_budget_remaining
  x historical_confidence
  - contradiction_match penalty
```

For bit-only baseline:

```rust
fn nxor_words(a: &[u64; 256], b: &[u64; 256]) -> [u64; 256] {
    let mut out = [0u64; 256];
    for i in 0..256 {
        out[i] = !(a[i] ^ b[i]);
    }
    out
}
```

Production scoring must mask inactive/unaddressed bits; otherwise NXOR will count irrelevant zeros as agreement.

Important:

```text
NXOR must be masked by active CAM address set / survivor tile set.
```

Use:

```text
agree = NXOR(a,b) & active_mask
```

---

## Active mask requirement

Bare NXOR over sparse planes is dangerous because zero-zero agreement dominates.

Required masks:

```text
active candidate addresses
active historical addresses
projection-compatible addresses
survivor tile mask
```

Recommended:

```text
compat = NXOR(candidate, support) & active_union_or_projection_mask
```

Not:

```text
compat = NXOR(candidate, support) over all 16,384 bits
```

This keeps NXOR semantic rather than trivial.

---

## Relation to Gaussian/Fisher-Z splatting

Gaussian/Fisher-Z splatting and NARS bitmap NXOR can coexist.

```text
Fisher-Z splat:
  calibrated pressure amplitude / survivor field

NARS bitmap NXOR:
  compatibility/agreement over support/contradiction planes

compress+prune:
  collapse pressure into candidate set

NARS/AriGraph:
  validate truth and revise witness chain
```

The combined hot path:

```text
ReasoningWitness64
  -> NARS bitmap planes
  -> Fisher-Z/Gaussian splat pressure
  -> NXOR compatibility over active CAM masks
  -> semantic CAM distance certificate
  -> prune/collapse candidates
```

---

## Relationship to spatial NARS bundle algebra

A future `SpatialNarsBundle` would likely need these operations:

```text
bundle/support:
  OR or q8 accumulate into support plane

bundle/contradiction:
  OR or q8 accumulate into contradiction plane

compatibility:
  masked NXOR over projection-compatible active CAM addresses

difference:
  masked XOR for disagreement/novelty

revision:
  update ReasoningWitness64 frequency/confidence and re-deposit

collapse:
  select candidate addresses with high support NXOR, low contradiction NXOR, low entropy
```

This suggests a bitmap semiring-like structure:

```text
carrier:
  CAM bitplane + witness mass + sigma/theta context

addition:
  superpose support/contradiction pressure

multiplication/composition:
  projection/path-hop/causal transition

compatibility:
  masked NXOR

novelty/contradiction:
  masked XOR + contradiction plane
```

This is research-grade math. Keep production on the simpler splat/certificate path until the algebra is proven.

---

## Certificate

```rust
pub struct NarsBitmapNxorCertificate {
    pub thought_id: u64,
    pub cycle_id: u64,
    pub candidate_id: u64,

    pub active_mask_popcount: u16,
    pub support_nxor_q8: u8,
    pub contradiction_nxor_q8: u8,
    pub novelty_xor_q8: u8,

    pub witness_compat_q8: u8,
    pub projection_compat_q8: u8,
    pub semantic_cam_distance_q8: u8,
    pub entropy_budget_remaining_q8: u8,

    pub exact_replay_required: bool,
    pub replay_ref: u64,
}
```

Decision law:

```text
high support NXOR + low contradiction NXOR + sufficient entropy budget:
  candidate may proceed to NARS/AriGraph validation

high support NXOR + high contradiction NXOR:
  exact replay required

high novelty XOR:
  exploratory / replay-only unless evidence accumulates

future/counterfactual source:
  style/prefetch/scenario only, no fact promotion
```

---

## Minimum viable implementation

### PR 1: NarsBitmapPlanes

Add support/contradiction/uncertain/forecast/counterfactual planes.

### PR 2: witness -> bitmap deposition

Convert `ReasoningWitness64` into channel deposition under the current CAM address/projection.

### PR 3: masked NXOR primitive

Implement:

```rust
fn masked_nxor_popcount(a: &AwarenessPlane16K, b: &AwarenessPlane16K, mask: &AwarenessPlane16K) -> u16
```

### PR 4: certificate

Add `NarsBitmapNxorCertificate`.

### PR 5: integrate with semantic CAM distance

Use NXOR as compatibility primitive, not as final truth.

---

## Suggested wording

Use this:

```text
Spatial NARS bundle superposition requires a NARS -> bitmap transcode. NARS truth and revision are carried as ReasoningWitness64, then deposited into support, contradiction, uncertain, forecast, and counterfactual CAM planes. Compatibility is evaluated with masked NXOR over deterministic active CAM addresses; novelty/disagreement uses masked XOR. NXOR is a bitmap primitive for semantic agreement, not the whole truth metric. NARS/AriGraph and replay certificates remain the authority for belief promotion.
```

Avoid this:

```text
NXOR proves meaning.
```

Better:

```text
NXOR measures active-address agreement inside the semantic CAM schema.
```
