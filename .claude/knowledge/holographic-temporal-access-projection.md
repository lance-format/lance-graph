# Holographic temporal access projection: 4096x4096xQuestion x Time

Status: contextual architecture note. This document explains the projection model; it does not claim full runtime implementation.

## One-sentence thesis

The `4096 x 4096 x question x time` model is a holographic access projection over the CAM/COCA codebook: every factor-pair cell can be read as a question, every question has an implicit 4096-answer codebook, and every cycle can project backward through replay and forward through Chronos-style perturbation forecasts.

In shorthand:

```text
4096 x 4096          = factor-pair CAM address space
x question           = projection/query lens over that address
x implicit 4096      = answer/codebook completion space
x 4096 back          = replayable temporal memory depth
x Chronos forward    = forecasted implication / perturbation horizon
ReasoningWitness64   = NARS/style/source/time witness angle per transition
```

This solves the access problem by making every dot in the field both an address and a question-bearing projection surface.

---

## What "every dot is a question" means

A field coordinate is not merely a cell.

```text
cell(a, b)
```

is a pairwise address that can be viewed under a projection lens:

```text
A + B asks C
A + C asks B
B + C asks A
ABC validates full triad
```

For SPO:

```text
Subject + Predicate asks Object
Subject + Object asks Predicate
Predicate + Object asks Subject
```

For generic ABC:

```text
factor_a + factor_b asks factor_c
```

So a `4096 x 4096` coordinate is not dead storage. It is a deterministic query site into a 4096-entry implicit answer space.

---

## Holographic access projection

The access projection is holographic because the local coordinate carries enough structured sidecar state to reconstruct the relevant global meaning path.

Each active site can be interpreted through:

```text
factor_a
factor_b
projection_kind
GrammarMarkovLens64
ReasoningWitness64[]
sigma_idx
theta aperture
cycle_id
thought_id
replay_ref
```

That makes the site replayable as:

```text
what was active
which question it asked
which answer family it implied
which reasoning style produced it
which NARS truth texture it carried
which temporal direction it occupied
which replay/future horizon it belonged to
```

The local cell therefore behaves like a window into the whole thought lineage.

---

## Implicit 4096-answer space

The completion space is implicit, not densely materialized.

```text
cell(a,b) -> asks c in 0..4095
```

The system only needs to hydrate survivor candidates:

```text
active factor pair
  -> projection lens
  -> witness-weighted resonance
  -> candidate answer subset
  -> NARS/AriGraph validation
```

This gives each pairwise coordinate access to the calibrated 4096 COCA answer basis without allocating `4096 x 4096 x 4096` dense state.

---

## Time as the fourth access dimension

Time is another projection lens over the same thought spine.

Backward:

```text
cycle N
  -> cycle N-1 ... N-4096
  -> replayed witnesses
  -> AriGraph semantic/episodic support
  -> Markov Local5 / Section50 / Document500 context
```

Forward:

```text
cycle N
  -> Chronos/Perturbation forecast
  -> possible next style mass
  -> future sigma/theta aperture
  -> future implication candidates
```

The backward path is replay. The forward path is forecasted implication. Both are attached to the same field grammar and witness format.

---

## Chronos perturbation projection

Chronos should not decide truth. It forecasts cycle dynamics.

It can project likely future changes in:

```text
support mass
contradiction mass
collapse entropy
style distribution
perturbation class
sigma/theta behavior
source entropy
active-bit drift
witness-count drift
```

The forward projection then becomes a perturbation policy:

```text
if contradiction is forecast to rise:
  widen Markov replay
  activate abduction/counterfactual styles
  tighten promotion policy
  request AriGraph support

if support is forecast to stabilize:
  narrow theta aperture
  sharpen sigma hydration
  reduce fanout

if source entropy collapses:
  diversify source lanes
  avoid overfitting one witness family
```

Chronos is therefore a temporal prefetch and style-accommodation engine, not a fact engine.

---

## Spatial Gaussian splat projection into the future

The same Gaussian splat mechanism can be projected forward.

Backward splat:

```text
past witnesses + replay memory
  -> support/contradiction field
  -> current candidate validation
```

Forward splat:

```text
current witnesses + Chronos perturbation
  -> future implication field
  -> likely candidate trajectories
  -> style/sweep/theta prefetch plan
```

A future splat should be marked as forecast/scenario, never as fact.

Recommended distinction:

```text
past/replay splat:
  evidence-bearing, can support NARS revision

future/forecast splat:
  implication-bearing, can guide sweep/prefetch/style, cannot promote ontology facts
```

---

## ReasoningWitness64 temporal semantics

`ReasoningWitness64` should be able to distinguish temporal direction and evidence status.

Useful fields:

```text
temporal_bucket:
  past replay
  current cycle
  future forecast
  delayed effect
  counterfactual branch
  recurrent pattern

source_lane:
  grammar
  Markov
  AriGraph semantic memory
  AriGraph episodic memory
  Chronos forecast
  model seed
  doctor/operator correction

thinking_style:
  deduction
  abduction
  association
  counterfactual
  synthesis
  extrapolation
  introspection
  fanout
```

This makes future implications visible to the shader without confusing them with evidence.

---

## Access path

Canonical path:

```text
ThoughtCycleSoA row
  AwarenessPlane16K
  GrammarMarkovLens64
  ReasoningWitness64[]
  sigma
  theta
  thought_id / cycle_id
        ↓
Holographic access projection
        ↓
cell(a,b) under projection_kind asks implicit c in 0..4095
        ↓
backward replay support + forward Chronos perturbation
        ↓
Gaussian splat hydration over survivor tiles
        ↓
HydrationCandidate / FutureImplicationCandidate
        ↓
NARS/AriGraph validation or style-policy update
```

The same access site can therefore answer:

```text
what did this mean before?
what does it imply now?
what might it imply next?
which thinking style should explore it?
```

---

## Candidate types

Separate evidence candidates from forecast candidates.

```rust
pub enum TemporalCandidateKind {
    ReplayEvidence,
    CurrentHydration,
    FutureImplication,
    CounterfactualScenario,
}
```

```rust
pub struct TemporalHydrationCandidate {
    pub kind: TemporalCandidateKind,
    pub factor_a: u16,
    pub factor_b: u16,
    pub factor_c: Option<u16>,
    pub projection: TriadicProjection,
    pub cycle_origin: u64,
    pub cycle_target: i32, // negative = past, zero = current, positive = forecast horizon
    pub energy: f32,
    pub support_mass: f32,
    pub contradiction_mass: f32,
    pub forecast_uncertainty: f32,
    pub replay_ref: u64,
}
```

Promotion law:

```text
ReplayEvidence + CurrentHydration may support ontology candidates after NARS/AriGraph validation.
FutureImplication may update sweep/style/prefetch policy.
CounterfactualScenario may create scenario objects only.
FutureImplication and CounterfactualScenario may not promote facts directly.
```

---

## Spatial prefetch connection

Future Gaussian splats can drive spatial prefetch.

```text
Chronos predicts likely future support/contradiction ridge
  -> project ridge into factor-pair tiles
  -> prefetch those tiles
  -> run style-specific sweep
  -> keep only survivor trajectories
```

Thinking style chooses the gait:

```text
deduction       -> narrow stride prefetch
abduction       -> reverse causal sweep
association     -> similarity survivor sweep
counterfactual  -> dual factual/scenario lanes
synthesis       -> multi-source intersection
fanout          -> budgeted frontier expansion
introspection   -> replay/policy inspection
```

The future splat does not assert truth. It chooses where to look next.

---

## Losslessness and replay

The backward projection is replayable because the cycle stores:

```text
thought_id
cycle_id
AwarenessPlane16K
GrammarMarkovLens64
ReasoningWitness64[]
sigma/theta
style trace
replay refs
codebook versions
kernel/policy versions
```

The forward projection is not replay of something that already happened. It is a deterministic forecast artifact if the Chronos model version, inputs, seeds, policy, and codebooks are versioned.

Use this language:

```text
Backward access is evidence replay.
Forward access is versioned implication replay.
```

---

## Suggested architecture wording

Use this:

```text
The 4096x4096 CAM field is a holographic factor-pair access surface. Under the SPO/ABC 2^3 projection grammar, every active coordinate is a question site into an implicit 4096-answer codebook. The same ThoughtCycleSoA row binds that site to an AwarenessPlane16K, GrammarMarkovLens64, ReasoningWitness64 transition witnesses, sigma geometry, theta aperture, and replay identity. Backward access hydrates past support from replay and AriGraph memory. Forward access uses Chronos-style perturbation forecasts to splat likely future implications into survivor tiles for prefetch, style accommodation, and scenario planning. Only replay/current evidence can support ontology promotion; future splats guide thinking but do not become facts.
```

Avoid this:

```text
The system knows the future.
```

Better:

```text
The system projects versioned future implications as forecasted field perturbations for traversal and policy control.
```

---

## Relation to other docs

This document extends:

```text
.claude/knowledge/cam-codebook-resonance-projection.md
.claude/plans/thought-cycle-soa-awareness-integration-v1.md
```

It adds the temporal/holographic access interpretation:

```text
4096x4096 pairwise field
  -> implicit 4096 answer codebook
  -> 4096-cycle backward replay
  -> Chronos/Perturbation forward projection
  -> spatial Gaussian future splat for traversal guidance
```
