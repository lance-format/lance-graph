# Gaussian splat -> CAM plane splat workaround

Status: contextual architecture note. This document proposes a workaround for the cost of exact 4096-cycle replay versus planar bundling: use Gaussian splats as deterministic deposition kernels into CAM/COCA awareness planes.

## One-sentence thesis

Instead of choosing only between exact 4096-cycle replay and one blurred superposition bundle, use Gaussian splat -> CAM plane splat: project witness-weighted semantic pressure into a 16,384-bit or small multi-plane CAM surface, preserving exact replay in Lance while maintaining a cheap hot approximation for traversal, semantic CAM distance, and future implication planning.

In shorthand:

```text
ReasoningWitness64 + sigma + theta + projection lens
  -> Gaussian semantic splat
  -> CAM/COCA plane deposition
  -> planar bundle / survivor tiles
  -> exact replay only when certification requires it
```

---

## Why this is the workaround

The previous fork:

```text
A. 3D print all 4096 historical cycle planes
   exact, audit-clean, bandwidth-heavier

B. bundle last 4096 planes into one superposition
   cheap, fast, but risks blur/alias/temporal-order loss
```

The workaround:

```text
C. Gaussian splat -> CAM plane splat
   keep exact replay cold
   keep splatted CAM pressure hot
   use sigma/theta/witness semantics to control deposition
```

This preserves the thought-lineage guarantee while avoiding naive full-history scans for every question.

---

## What gets splatted

A splat is derived from hot semantic sidecars:

```text
AwarenessPlane16K
GrammarMarkovLens64
ReasoningWitness64[]
sigma index / SigmaCodebook
ThetaDecision
TriadicProjection
thought_id / cycle_id / replay_ref
```

The splat carries:

```text
center:
  CAM address or factor-pair tile

amplitude:
  NARS confidence/frequency x evidence mantissa x polarity

width:
  sigma geometry x theta_width

acceptance aperture:
  theta_accept

channel:
  support, contradiction, forecast, counterfactual, style, source

lineage:
  replay_ref and witness generation
```

---

## CAM plane splat, not raw image splat

This is not primarily a pixel/image splat.

It is a semantic deposition into a content-addressable codebook plane:

```text
factor pair / projection / witness
  -> semantic CAM address
  -> deposition into AwarenessPlane16K or support/contradiction planes
```

Possible hot planes:

```text
support_plane
contradiction_plane
forecast_plane
counterfactual_plane
style_plane
source_plane
```

Each plane is still a CAM/COCA plane. The splat controls how semantic pressure lands on it.

---

## Data structure sketch

```rust
pub enum SplatChannel {
    Support,
    Contradiction,
    Forecast,
    Counterfactual,
    Style,
    Source,
}

pub struct CamPlaneSplat {
    pub center_a: u16,
    pub center_b: u16,
    pub projection: TriadicProjection,
    pub channel: SplatChannel,

    pub amplitude_q8: u8,
    pub width_q8: u8,
    pub theta_accept_q8: u8,

    pub witness: ReasoningWitness64,
    pub replay_ref: u64,
}

pub struct SplatPlaneSet {
    pub support: AwarenessPlane16K,
    pub contradiction: AwarenessPlane16K,
    pub forecast: AwarenessPlane16K,
    pub counterfactual: AwarenessPlane16K,
    pub style: AwarenessPlane16K,
    pub source: AwarenessPlane16K,
}
```

Implementation may use bitplanes, q8 accumulation planes, or tile-local accumulators. The contract should not require float accumulation in the hot path.

---

## Deposition rule

A conceptual deposition rule:

```text
for each witness:
  unpack NARS truth, polarity, projection, style, source, temporal bucket
  derive semantic CAM center from factor pair + projection lens
  derive width from sigma + theta_width
  derive amplitude from evidence_mantissa, frequency, confidence, polarity
  deposit into one or more CAM planes
```

Pseudo-code:

```rust
fn witness_to_splat(
    factor_a: u16,
    factor_b: u16,
    projection: TriadicProjection,
    witness: ReasoningWitness64,
    sigma_idx: u8,
    theta: ThetaDecision,
    replay_ref: u64,
) -> CamPlaneSplat {
    // field-aware unpacking, not raw u64 math
    todo!()
}
```

The final deposition must be deterministic under fixed codebook, sigma codebook, theta policy, witness layout, and seeds.

---

## Relation to semantic CAM distance

Semantic CAM distance should compare against splatted pressure planes, not raw Hamming over anonymous bits.

```text
current candidate
  -> semantic CAM address
  -> compare to support_plane / contradiction_plane / forecast_plane
  -> projection compatibility
  -> witness compatibility
  -> entropy budget
  -> certificate
```

This lets the engine use cheap hot pressure maps while preserving semantic structure.

---

## Relationship to planar bundling

Planar bundling can be implemented as splatted planes instead of raw bundled vectors.

Old bundle idea:

```text
Bundle4096 = aggregate(last 4096 AwarenessPlane16K)
```

Better:

```text
PlanarSplatBundle4096 = splat(last 4096 cycle witnesses into support/contradiction/style/source planes)
```

This avoids blurring support and contradiction into the same surface.

Suggested structure:

```rust
pub struct PlanarSplatBundle4096 {
    pub local: SplatPlaneSet,
    pub short: SplatPlaneSet,
    pub medium: SplatPlaneSet,
    pub long: SplatPlaneSet,

    pub epoch_start_cycle: u64,
    pub epoch_end_cycle: u64,
    pub replay_ref: u64,
}
```

Bands:

```text
local:   8 or 16 cycles
short:   64 cycles
medium:  512 cycles
long:    4096 cycles
```

---

## Relationship to future projection

Future Gaussian splat projection becomes natural:

```text
Chronos/Perturbation forecast
  -> FutureImplication witness
  -> forecast splat
  -> forecast_plane
  -> prefetch / style accommodation / scenario planning
```

Future splats are not evidence.

```text
support/contradiction splats from replay/current cycles:
  may support evidence after NARS/AriGraph validation

forecast/counterfactual splats:
  guide traversal and scenario planning
  must not promote ontology facts
```

---

## Entropy advantage

A raw bundle hides where entropy accumulated.

A splatted multi-plane bundle can track:

```text
support entropy
contradiction entropy
projection entropy
style entropy
source entropy
temporal-band entropy
forecast uncertainty
```

This feeds the `EntropyBudgetCertificate` more cleanly than a single superposition vector.

---

## Certification path

A candidate using splatted CAM pressure should carry:

```rust
pub struct CamSplatCertificate {
    pub thought_id: u64,
    pub cycle_id: u64,
    pub candidate_id: u64,

    pub support_pressure_q8: u8,
    pub contradiction_pressure_q8: u8,
    pub forecast_pressure_q8: u8,
    pub counterfactual_pressure_q8: u8,

    pub projection_compat_q8: u8,
    pub witness_compat_q8: u8,
    pub sigma_margin_q8: u8,
    pub theta_accept_q8: u8,
    pub theta_width_q8: u8,

    pub entropy_budget_remaining_q8: u8,
    pub exact_replay_required: bool,
    pub replay_ref: u64,
}
```

Decision:

```text
if support high, contradiction low, entropy budget high:
  candidate may proceed to NARS/AriGraph validation

if support high but contradiction high:
  exact replay required

if forecast high but evidence low:
  use for prefetch/style only

if counterfactual high:
  scenario only
```

---

## Implementation sequence

### PR 1: splat contracts

Add:

```text
SplatChannel
CamPlaneSplat
SplatPlaneSet
CamSplatCertificate
```

### PR 2: witness -> splat

Implement deterministic conversion from `ReasoningWitness64 + sigma/theta + projection` into a `CamPlaneSplat`.

### PR 3: splat deposition

Implement deposition into support/contradiction/forecast/counterfactual planes.

Start with deterministic bit/q8 tile accumulation. Avoid mandatory floats.

### PR 4: PlanarSplatBundle4096

Maintain local/short/medium/long splat bands.

### PR 5: semantic CAM distance integration

Use splat planes for survivor tiles and distance certificates.

### PR 6: replay fallback

If the splat certificate is insufficient, load exact 4096-cycle replay slice.

---

## Suggested architecture wording

Use this:

```text
Gaussian splat -> CAM plane splat is the hot approximation layer between exact replay and raw bundling. Each ReasoningWitness64, under sigma geometry and theta aperture, deposits semantic pressure into content-addressed support, contradiction, forecast, and counterfactual CAM planes. These splatted planes provide cheap survivor selection and semantic CAM distance while exact ThoughtCycleSoA replay remains the authority for certification and ontology promotion.
```

Avoid this:

```text
The Gaussian splat replaces replay.
```

Better:

```text
The Gaussian splat creates certified hot pressure planes over replayable thought history.
```
