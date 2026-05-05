# ThoughtCycleSoA awareness integration plan v1

Status: integration plan. No implementation claimed here.

## Goal

Unify grammar parsing, Markov replay, thinking styles, NARS witnesses, sigma/theta field control, and ontology collapse on one replayable thought-cycle spine.

The central law:

```text
Bundles may be separate.
Thought identity may not be separate.

AwarenessPlane16K carries activation.
GrammarMarkovLens64 carries interpretation.
ReasoningWitness64 carries truth/style/causal transition.
ThoughtCycleSoA carries lineage.
```

The clean target is not another side pipeline. The clean target is one `ThoughtCycleSoA` where grammar, Markov, thinking, NARS, sigma/theta, hydration, and replay all attach to the same `thought_id` and `cycle_id`.

---

## Why this exists

The previous float carrier blurred responsibilities. `Vsa16kF32` was useful while it was unclear where grammar heuristics, Markov bundles, role bindings, thinking styles, and field state belonged. But once grammar/Markov and thinking styles are separated cleanly, the field should not carry all of that as opaque floats.

The corrected split:

```text
AwarenessPlane16K       = compact CAM/COCA field activation, 16,384 bits = 2 KiB
GrammarMarkovLens64     = parse/context interpretation lens
ReasoningWitness64      = NARS-compatible causal/thinking transition witness
ThoughtCycleSoA         = same thought, same cycle, same replay lineage
Outer ontology          = typed operational meaning after collapse
```

This preserves the `thought × 4096×4096 meaning replay` guarantee without paying full `-500..+500` hydration cost on every cycle.

---

## Existing repo anchors

Relevant landed pieces:

- `crates/cognitive-shader-driver/src/bindspace.rs` already has BindSpace/FingerprintColumns style SoA and a `sigma: u8` row index from PR #323.
- `crates/lance-graph-contract/src/sigma_propagation.rs` already has `Spd2`, `ewa_sandwich`, `log_norm_growth`, and `pillar_5plus_bound` from PR #322.
- `crates/deepnsm/src/disambiguator_glue.rs` introduced real sign-binarized `Binary16K` trajectory fingerprints from Markov bundles in PR #305.
- `crates/deepnsm/src/markov_bundle.rs` and `crates/lance-graph-contract/src/grammar/context_chain.rs` already provide the grammar/Markov context machinery.
- `crates/lance-graph-archetype/` already provides a scaffold for `Component`, `Processor`, `World`, and `CommandBroker`.

Missing bloodstream:

```text
GrammarTriangle / MarkovBundler
  -> AwarenessPlane16K + GrammarMarkovLens64 + ReasoningWitness64[]
  -> ThoughtCycleSoA
  -> CognitiveShader hydration
  -> NARS/AriGraph revision
  -> updated ReasoningWitness64[]
  -> ontology candidate
```

---

## Core types

### AwarenessPlane16K

```rust
#[repr(align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AwarenessPlane16K {
    pub words: [u64; 256], // 16,384 bits = 2 KiB
}
```

Purpose:

```text
what is active in the CAM/COCA-coded field
```

It must not store grammar details, full provenance, full NARS terms, or ontology objects.

---

### ReasoningWitness64

One shared witness carrier for grammar, thinking, and causal/NARS transitions.

```rust
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReasoningWitness64(pub u64);

pub type CausalEdge64 = ReasoningWitness64;
pub type GrammarWitness64 = ReasoningWitness64;
pub type ThinkingWitness64 = ReasoningWitness64;
```

Proposed bit layout:

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

Meaning:

```text
Grammar emits it.
Thinking consumes it.
NARS revises it.
CognitiveShader hydrates with it.
AriGraph/ontology collapse can trace it.
```

Do not create incompatible `GrammarWitness64`, `ThinkingWitness64`, and `CausalEdge64` layouts. Use aliases over one representation.

---

### GrammarMarkovLens64

The grammar/Markov sidecar lens. It explains how the plane should be read.

```rust
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GrammarMarkovLens64(pub u64);
```

Proposed layout:

```text
bits  0..7   grammar_style
bits  8..15  markov_radius_class
bits 16..23  replay_direction
bits 24..31  tekamolo_mask
bits 32..39  role_family
bits 40..47  projection_kind
bits 48..55  ambiguity_bucket
bits 56..63  lens_generation
```

Purpose:

```text
how this activation was parsed/contextualized
```

Full parse trees, sentence windows, and role bundles live in Lance replay, not in the hot lens.

---

### TriadicProjection

Agnostic replacement for hard-coding SPO in the hot layer.

```rust
pub enum TriadicProjection {
    Abc,
    AbAskC,
    AcAskB,
    BcAskA,
    AOnly,
    BOnly,
    COnly,
    Background,
}
```

SPO mapping:

```text
A = Subject
B = Predicate
C = Object

ABC = full observed triple
AB_ = Subject + Predicate asks Object
A_C = Subject + Object asks Predicate
_BC = Predicate + Object asks Subject
```

The same projection model can also represent sonography, clinical facts, persona/style actions, or generic graph triples.

---

### ThoughtCycleSoA

The spine.

```rust
pub struct ThoughtCycleSoA {
    pub thought_id: Vec<u64>,
    pub cycle_id: Vec<u64>,

    // core field state
    pub plane: Vec<AwarenessPlane16K>,

    // grammar/Markov lens sidecar
    pub grammar_lens: Vec<GrammarMarkovLens64>,
    pub markov_window_class: Vec<MarkovWindowClass>,

    // reasoning witness sidecar
    pub witness_offset: Vec<u32>,
    pub witness_len: Vec<u16>,
    pub witnesses: Vec<ReasoningWitness64>,

    // geometry / aperture
    pub sigma: Vec<u8>,
    pub theta_accept_q8: Vec<u8>,
    pub theta_width_q8: Vec<u8>,

    // thinking-policy trace
    pub dominant_style: Vec<u8>,
    pub dominant_perturbation: Vec<u8>,
    pub support_mass_q8: Vec<u8>,
    pub contradiction_mass_q8: Vec<u8>,

    // replay / provenance
    pub replay_ref: Vec<u64>,
}
```

Window tiers:

```rust
pub enum MarkovWindowClass {
    Local5,
    Section50,
    Document500,
}
```

All tiers for one cognitive act share the same `thought_id`.

---

## Parse-time reasoning witness

The key advantage of sharing `ReasoningWitness64` between grammar and thinking is that the parse object can recognize its reasoning mode at birth.

The parser does not only emit syntax or SPO slots. It emits:

```rust
pub struct ParsedMeaningObject {
    pub plane: AwarenessPlane16K,
    pub lens: GrammarMarkovLens64,
    pub witnesses: SmallVec<[ReasoningWitness64; 8]>,
    pub replay_ref: u64,
}
```

This allows parse-time classification:

```text
abduction
inference
deduction
association
counterfactual
synthesis
introspection / metacognition
```

Example:

```text
"The lesion may have increased because the therapy failed."
```

Parse-time witness:

```text
relation_family = causal_hypothesis
projection_kind = _BC asks A / observed effect asks cause
thinking_style = abduction
source_lane = grammar/text
NARS seed = medium frequency, low confidence
```

Markov then revises the same witness instead of reclassifying after bundling.

---

## Tiered context hydration

Hydrating `-500..+500` sentences against AriGraph and the `4096×4096` meaning field is not free. Therefore hydration must be staged.

```text
Stage 0: Local5
  ±5 Markov context
  default parse-time witness revision
  no full graph hydration unless ambiguity/contradiction survives

Stage 1: Section50
  ±50 context
  semantic-memory lookup
  survivor projections only

Stage 2: Document500
  ±500 context
  episodic-memory support / contradiction resolution
  delayed-reference and document-level argument resolution

Stage 3: Graph hydration
  only candidates surviving NARS, sigma, theta, and replay gates
```

Separate bundles are fine. Separate thought identity is not.

---

## Sigma and theta

Sigma:

```text
sigma: u8
  -> SigmaCodebook[256]
  -> Spd2
  -> ewa_sandwich()
  -> log_norm_growth()
  -> pillar_5plus_bound()
```

The sigma bound is the guardrail.

Theta is not a fixed constant. It is a dynamic aperture:

```text
theta_accept = candidate survival aperture
theta_width  = hydration/splat width aperture
```

Inputs:

```text
sigma log-norm growth
NARS confidence mass
contradiction mass
thinking style
perturbation class
AriGraph support
Chronos/anomaly hint
source entropy
```

Cycle-level hot storage:

```rust
pub struct ThetaDecision {
    pub theta_accept_q8: u8,
    pub theta_width_q8: u8,
    pub reason_mask: u64,
}
```

Law:

```text
Σ bound = guardrail.
θ = steering/aperture.
```

---

## Hydration input

The cognitive shader should consume one row shape:

```rust
pub struct HydrationInput<'a> {
    pub plane: AwarenessPlane16K,
    pub lens: GrammarMarkovLens64,
    pub sigma_idx: u8,
    pub theta_accept_q8: u8,
    pub theta_width_q8: u8,
    pub witnesses: &'a [ReasoningWitness64],
    pub replay_ref: u64,
}
```

It should not parse grammar and should not inspect full ontology objects.

Output:

```rust
pub struct HydrationCandidate {
    pub factor_a: u16,
    pub factor_b: u16,
    pub factor_c: Option<u16>,
    pub projection: TriadicProjection,
    pub energy: f32,
    pub support_mass: f32,
    pub contradiction_mass: f32,
    pub replay_ref: u64,
}
```

---

## Thinking style feedback

Witnesses produce style mass. Style mass drives the next cycle.

```rust
pub struct WitnessMassSummary {
    pub support_mass_by_style: [u16; 256],
    pub contradiction_mass_by_style: [u16; 256],
    pub source_mass: [u16; 64],
    pub projection_mass: [u16; 16],
}
```

Flow:

```text
ReasoningWitness64[]
  -> WitnessMassSummary
  -> SweepPolicy
  -> ThetaPolicy
  -> FanoutPolicy
  -> next ThoughtCycleSoA row
```

This lets grammar-time abduction/counterfactual/deduction influence prefetch and hydration immediately.

---

## Replay certificate

Every thought cycle should be reconstructable.

```rust
pub struct ThoughtReplayCertificate {
    pub thought_id: u64,
    pub cycle_id: u64,

    pub local_window_ref: Option<u64>,
    pub section_window_ref: Option<u64>,
    pub document_window_ref: Option<u64>,

    pub plane_hash: u64,
    pub grammar_lens: GrammarMarkovLens64,
    pub sigma_idx: u8,
    pub theta_accept_q8: u8,
    pub theta_width_q8: u8,

    pub witness_offset: u32,
    pub witness_len: u16,

    pub arigraph_semantic_refs: Vec<u64>,
    pub arigraph_episodic_refs: Vec<u64>,
}
```

The outer ontology should receive collapsed candidates with replay references, not raw hot-layer internals.

---

## End-to-end flow

```text
Input text / observation
    ↓
GrammarTriangle + MarkovBundler
    ↓
ParsedMeaningObject
  AwarenessPlane16K
  GrammarMarkovLens64
  ReasoningWitness64[]
    ↓
ThoughtCycleSoA row
    ↓
Local5 hydration
    ↓
if needed: Section50 replay
    ↓
if needed: Document500 / AriGraph replay
    ↓
Sigma propagation + dynamic theta
    ↓
Gaussian/CAM hydration over survivor projections
    ↓
ReasoningWitness64 append/revision
    ↓
ThinkingStyleSummary update
    ↓
candidate collapse
    ↓
AriGraph/NARS validation
    ↓
outer ontology proposal
```

---

## PR sequence

### PR 1: shared witness contract

Add in `lance-graph-contract`:

```text
AwarenessPlane16K
ReasoningWitness64
CausalEdge64 alias
GrammarWitness64 alias
ThinkingWitness64 alias
GrammarMarkovLens64
TriadicProjection
ThetaDecision
```

Tests:

```text
bit packing/unpacking
q8 conversion
projection mapping
witness alias equivalence
```

---

### PR 2: ThoughtCycleSoA scaffold

Add `ThoughtCycleSoA` with offset-based witness storage.

Tests:

```text
push/readback plane
push/readback lens
push/readback witness slice
sigma/theta columns align
thought_id/cycle_id lineage preserved
```

---

### PR 3: grammar/Markov parse emission

Add adapter:

```text
GrammarTriangle + MarkovBundler Trajectory
  -> ParsedMeaningObject
```

Use existing `Binary16K` sign-binarized trajectory bridge where possible.

Tests:

```text
same sentence/window -> stable plane/lens/witness
abductive sentence emits abductive witness
counterfactual sentence emits counterfactual witness
changed Markov radius changes lens, not necessarily plane
```

---

### PR 4: tiered Markov replay rows

Add `MarkovWindowClass::{Local5, Section50, Document500}` to cycle rows.

Tests:

```text
three window tiers share thought_id
window tiers retain separate lens/replay refs
Document500 only triggers on ambiguity/contradiction policy
```

---

### PR 5: SigmaCodebook B3

Implement:

```text
sigma: u8 -> SigmaCodebook[256] -> Spd2
```

No shader logic yet.

---

### PR 6: shader sigma propagation B4

Wire:

```text
edge emission
  -> sigma lookup
  -> ewa_sandwich
  -> log_norm_growth
  -> pillar_5plus_bound hard gate
  -> FreeEnergy gate
```

---

### PR 7: dynamic theta controller

Add:

```text
ThetaContext
ThetaPolicy
ThetaDecision
```

Tests:

```text
high contradiction widens/explores
high support tightens/confirms
counterfactual disables promotion
sigma bound remains hard gate
```

---

### PR 8: hydration adapter

Add:

```text
ThoughtCycleSoA row -> HydrationInput -> HydrationCandidate
```

First implementation may be tiled/stubbed. Boundary correctness first.

---

### PR 9: NARS/AriGraph witness revision loop

Hydration candidates become revised witnesses:

```text
HydrationCandidate
  -> NARS/AriGraph validation
  -> revised ReasoningWitness64
  -> append next cycle
```

---

### PR 10: outer ontology collapse

Map validated candidates to operational objects:

```text
HydrationCandidate / ThoughtStruct
  -> OntologyCandidate
  -> MedCare FindingCandidate / EvidenceEdge / CalibrationEvent
```

The outer layer must not receive raw witnesses, grammar lenses, or planes except through debug/replay tools.

---

## Promotion and safety laws

```text
Deduction + strong provenance:
  promotable after NARS/AriGraph/ontology gate

Abduction:
  candidate explanation only until evidence supports

Association:
  similar-case reference only

Counterfactual:
  scenario only, never fact

Introspection:
  policy update only, never ontology fact

Fanout:
  exploratory until re-scored
```

Model/persona/Chronos hints may bias style and theta. They may not promote facts.

---

## Final sentence

The parser does not output dead syntax. It emits a semantic object already carrying a NARS-compatible `ReasoningWitness64`. Markov bundling revises the same witness across local, section, and document tiers. CognitiveShader hydrates with that witness inside `ThoughtCycleSoA`. NARS/AriGraph validate and revise it. The outer ontology only receives collapsed, replay-backed candidates.
