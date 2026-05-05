# Planar bundling similarity distance vs 4096-cycle reprint

Status: contextual architecture note. This document explores the tradeoff between replaying/materializing 4096 historical cycle planes and bundling them into a planar similarity object while preserving the codebook-level replay guarantee.

## One-sentence thesis

The design question is whether it is cheaper to "3D print" 4096 explicit AwarenessPlane16K iterations for exact replay, or to maintain a superposition bundle of the last 4096 planes with enough sidecar structure to compute semantic CAM distance. A viable middle path is planar bundling similarity distance: preserve exact replay in Lance, keep compact bundles hot, and certify when the bundle is equivalent enough for traversal/confidence decisions.

---

## The two extremes

### Option A: 4096-cycle 3D print

Store and replay every cycle plane explicitly:

```text
4096 cycles x AwarenessPlane16K
4096 x 2 KiB = 8 MiB per thought-history window
```

If each cycle also has witnesses, sigma/theta, and lens sidecars, the real footprint is larger, but still manageable if windowed and stored in Lance.

Strength:

```text
exact lineage replay
exact cycle reconstruction
precise temporal causality
best audit posture
```

Weakness:

```text
more memory bandwidth
more scan work
more replay IO
more per-cycle indexing
```

This is the gold standard for audit and deterministic re-execution.

---

### Option B: superposition of last 4096 vectors

Maintain a hot superposition bundle:

```text
Bundle4096 = aggregate(AwarenessPlane16K[N-4095..N])
```

Strength:

```text
small hot state
fast similarity/prefetch
cheap style accommodation
cheap temporal pressure estimate
```

Weakness:

```text
temporal order may blur
old/new evidence can interfere
support and contradiction can alias
cannot by itself prove exact cycle lineage
```

This is a fast field-pressure approximation, not a replacement for replay.

---

## Proposed middle path: planar bundling

Do not collapse all 4096 cycles into one undifferentiated bundle. Keep multiple planes/bands.

```text
local plane      = last 8/16 cycles
short plane      = last 64 cycles
medium plane     = last 512 cycles
long plane       = last 4096 cycles
contradiction    = negative/polarity plane
support          = positive/support plane
forecast         = Chronos/perturbation future plane
```

This gives a planar history object:

```rust
pub struct PlanarBundle4096 {
    pub support_local: AwarenessPlane16K,
    pub support_short: AwarenessPlane16K,
    pub support_medium: AwarenessPlane16K,
    pub support_long: AwarenessPlane16K,

    pub contradiction_local: AwarenessPlane16K,
    pub contradiction_short: AwarenessPlane16K,
    pub contradiction_medium: AwarenessPlane16K,
    pub contradiction_long: AwarenessPlane16K,

    pub style_mass: [u16; 256],
    pub projection_mass: [u16; 16],
    pub source_mass: [u16; 64],

    pub epoch_start_cycle: u64,
    pub epoch_end_cycle: u64,
    pub replay_ref: u64,
}
```

The exact cycles still exist in replay. The bundle is the hot summary.

---

## Planar bundling similarity distance

Semantic CAM distance should compare the current plane to the planar bundle, not to raw anonymous bitstrings.

```text
PlanarBundlingSimilarity(x, B)
  = semantic_cam_distance(x, support planes)
  - contradiction_overlap(x, contradiction planes)
  + projection_compatibility(x, projection_mass)
  + witness/style/source compatibility
  + temporal decay weighting
```

Distance components:

```text
codebook-address compatibility
pairwise CAM compatibility
projection compatibility
support-plane compatibility
contradiction-plane incompatibility
style/source/projection mass compatibility
sigma/theta compatibility
historical confidence from replay
```

The result should be a certificate, not a naked float.

---

## PlanarBundlingDistanceCertificate

```rust
pub struct PlanarBundlingDistanceCertificate {
    pub thought_id: u64,
    pub cycle_id: u64,
    pub bundle_start_cycle: u64,
    pub bundle_end_cycle: u64,

    pub codebook_compat_q8: u8,
    pub projection_compat_q8: u8,
    pub support_overlap_q8: u8,
    pub contradiction_overlap_q8: u8,
    pub style_compat_q8: u8,
    pub source_compat_q8: u8,
    pub temporal_decay_q8: u8,
    pub entropy_budget_remaining_q8: u8,

    pub bundle_confidence_q8: u8,
    pub exact_replay_required: bool,
    pub replay_ref: u64,
}
```

Use this certificate to decide whether the hot bundle is enough or whether the engine must fall back to exact 4096-cycle replay.

---

## Decision rule

```text
if bundle_confidence high and contradiction_overlap low:
  use planar bundle for sweep/prefetch/confidence prefilter

if bundle_confidence medium:
  use bundle for survivor generation, then verify against exact replay slice

if bundle_confidence low or contradiction_overlap high:
  require exact 4096-cycle replay / 3D print
```

This preserves the guarantee:

```text
hot bundles guide thinking;
exact replay certifies truth.
```

---

## Lossless consciousness guarantee wording

Use this cautiously:

```text
The exact thought lineage remains losslessly replayable because every cycle row, witness slice, lens, sigma/theta state, and codebook version is persisted. Planar bundling is a hot similarity accelerator over that replay corpus. It may guide traversal and candidate generation, but ontology promotion requires either exact replay verification or a certificate proving the bundle is sufficient under the current entropy budget.
```

Avoid saying:

```text
The bundle alone is the exact thought.
```

Better:

```text
The bundle is a hot projection of the exact thought history.
```

---

## Why this may beat 4096-cycle materialization

The 3D-print replay is clean but may spend bandwidth on cycles that do not affect the current question.

Planar bundling can precompute:

```text
support pressure
contradiction pressure
style pressure
projection pressure
source pressure
```

Then only exact-replay the small subset where:

```text
candidate survives
entropy budget is tight
contradiction exists
promotion is requested
medical/operational audit requires exact trace
```

This turns 4096-cycle replay into a demand-loaded certificate path.

---

## Update policy

Planar bundles should be maintained incrementally:

```text
on new cycle:
  add current plane/witness mass to local bands
  decay or rotate older bands
  promote local -> short -> medium -> long on schedule
  keep replay refs for exact reconstruction
```

Suggested bands:

```text
local:   8 or 16 cycles
short:   64 cycles
medium:  512 cycles
long:    4096 cycles
```

These powers-of-two bands make cache and replay slicing boring.

---

## Relation to Chronos future projection

Forward implication can use the same planar shape:

```text
FuturePlanarBundle
  support_forecast
  contradiction_forecast
  style_forecast
  theta_forecast
  uncertainty
```

But future bundles are not evidence.

```text
past planar bundle:
  can support confidence after verification

future planar bundle:
  can guide prefetch/style/scenario planning
  cannot promote facts
```

---

## Suggested implementation sequence

### PR 1: PlanarBundle4096 type

Add hot support/contradiction planes and mass histograms.

### PR 2: incremental update

Append current ThoughtCycleSoA row into the bundle bands.

### PR 3: planar semantic distance

Implement `PlanarBundlingDistanceCertificate` using semantic CAM distance components.

### PR 4: replay fallback gate

If certificate says exact replay required, pull the relevant 4096-cycle slice.

### PR 5: hydration integration

Hydration first consults planar bundle for survivor tiles, then exact replay verifies candidates before promotion.

---

## Suggested architecture wording

Use this:

```text
The engine can choose between exact 4096-cycle replay and planar bundling. Exact replay is the lossless thought-lineage guarantee. Planar bundling is the hot superposition accelerator: it stores support, contradiction, projection, style, and source pressure over multiple temporal bands so semantic CAM distance can be estimated cheaply. The bundle may drive sweep, prefetch, and candidate generation; promotion requires exact replay or a PlanarBundlingDistanceCertificate showing the entropy budget is sufficient.
```

Avoid this:

```text
We replace replay with one bundled vector.
```

Better:

```text
We use planar bundles as certified hot projections over exact replay.
```
