# Oxigraph + AriGraph + Cognitive Shader SoA Merge v1

Status: architecture context / implementation prompt for Claude Code and future sessions. This document distills the session context around Oxigraph graph contexts, AriGraph/SPO semantic episodic facts, cognitive-shader SoA DTOs, Gaussian/Fisher-Z spatial splat consequence search, BLASGraph-style expansion, and 4096 Markov-history future prefetch.

This is not a claim that the implementation already exists. It is the intended integration shape.

---

## 0. One-sentence thesis

Compile Oxigraph RDF quads, AriGraph semantic episodes, and lance-graph SPO truth/replay into one context-aware SoA format under `cognitive-shader-driver`; then use 64x64 cause/effect, 256x256 palette/attention, and 4096x4096 SPO/Pearl hole fields to run Gaussian/Fisher-Z spatial splat consequence search over semantic holes, with Markov-history future prefetch and replay-certified promotion.

Short form:

```text
Oxigraph graph -> ontology_context_id
AriGraph episode -> semantic provenance
SPO/NARS -> truth/replay
cognitive-shader-driver -> hot SoA execution
Gaussian splat -> spatial consequence expansion over holes
Markov history -> prefetch future holes, not facts
```

---

## 1. The three graph worlds being merged

### 1.1 Oxigraph / RDF / OWL

Oxigraph speaks RDF datasets/quads:

```text
subject | predicate | object | graph
```

The important field is `graph`. In this architecture it becomes:

```text
ontology_context_id
```

Example graph contexts:

```text
graph:fma
graph:radlex
graph:snomed
graph:medcare-local
graph:ai-candidates
graph:doctor-corrections
graph:histology
graph:final-befund
graph:validated-outer-ontology
graph:billing
```

This keeps semantic domains separate. Billing does not merge with FMA. AI candidates stay speculative. Histology/final Befund are high-authority evidence contexts. Validated outer ontology is the committed-fact lane.

### 1.2 AriGraph

AriGraph is not merely RDF. It is semantic episodic memory:

```text
subject | predicate | object | episode | source | truth | witness | replay
```

It owns the lived context of a fact:

```text
language
retrieval
sensorium
orchestration
triplet graph
XAI/explanation
patient/case/episode meaning
```

AriGraph answers:

```text
What happened, in which episode, from which source, with which meaning?
```

### 1.3 lance-graph / SPO / cognitive-shader-driver

The hot runtime should use columnar SoA, not verbose RDF or SPARQL.

The shader sees compiled rows:

```text
subject_id
predicate_id
object_id
ontology_context_id
episode_id
source_id
frequency_q8
confidence_q8
contradiction_q8
witness_ref
replay_ref
embedding_ref
```

The shader answers:

```text
Which contexts support this SPO?
Which contexts contradict it?
Which witness lane is strongest?
Which candidates remain speculative?
Which claims can be promoted?
```

---

## 2. Core merge contract

### 2.1 RDF quad to SemanticSpoRow

Oxigraph quad:

```text
S | P | O | Graph
```

compiles to:

```rust
#[repr(C)]
pub struct SemanticSpoRow {
    pub subject_id: u64,
    pub predicate_id: u64,
    pub object_id: u64,

    // Oxigraph named graph compiled to integer.
    pub ontology_context_id: u32,

    // AriGraph semantic/episodic context.
    pub episode_id: u64,
    pub source_id: u32,

    // NARS / reasoning state.
    pub frequency_q8: u8,
    pub confidence_q8: u8,
    pub contradiction_q8: u8,

    // Runtime evidence and replay.
    pub witness_ref: u64,
    pub replay_ref: u64,
    pub embedding_ref: u64,
}
```

SoA form:

```rust
pub struct SemanticSpoSoA {
    pub subject_id: Vec<u64>,
    pub predicate_id: Vec<u64>,
    pub object_id: Vec<u64>,
    pub ontology_context_id: Vec<u32>,

    pub episode_id: Vec<u64>,
    pub source_id: Vec<u32>,

    pub frequency_q8: Vec<u8>,
    pub confidence_q8: Vec<u8>,
    pub contradiction_q8: Vec<u8>,

    pub witness_ref: Vec<u64>,
    pub replay_ref: Vec<u64>,
    pub embedding_ref: Vec<u64>,
}
```

### 2.2 Recommended placement

New crate:

```text
crates/lance-graph-rdf/
```

Owns:

```text
OntologyContextId
NamedGraphRegistry
RDF Term -> internal GraphId mapping
SemanticQuad
Oxigraph integration/wrapper
Quad -> SemanticSpoRow compilation
optional OWL/RDF import utilities
```

Shader boundary:

```text
crates/cognitive-shader-driver/src/wire.rs
```

Add or locate:

```text
SemanticSpoRow
SemanticSpoBatch
```

Hot storage:

```text
crates/cognitive-shader-driver/src/bindspace.rs
```

Add or locate:

```text
SemanticSpoSoA
append_semantic_spo_batch()
filter_by_context()
group_same_spo_across_contexts()
derive_context_witness()
```

---

## 3. Context separation law

The same SPO shape may be repeated across contexts:

```text
ROI_55 supports Tendinopathy graph:ai-candidates
ROI_55 supports Tendinopathy graph:doctor-review
Histology_9 contradicts Tendinopathy graph:histology
```

Those are not one muddy truth. They are contextual assertions.

The shader groups same or related SPO keys across contexts and derives witness pressure:

```text
support quorum
contradiction quorum
uncertainty
source authority
replay requirement
promotion decision
```

Rule:

```text
ontology_context_id = semantic namespace / context boundary
witness_ref = why/how/source this assertion is supported
replay_ref = exact audit/reconstruction path
```

Do not collapse context and witness into one field. Context says where the claim lives. Witness says why the claim is believed.

---

## 4. Cognitive shader idea

The cognitive shader is a context-aware SPO execution engine.

It compiles:

```text
SPO claim
+ ontology context
+ episode
+ witness
+ NARS truth
+ replay
```

into SoA lanes, then runs fast passes:

```text
filter by ontology_context_id
group same SPO across contexts
compare support vs contradiction
form witness quorum
update NARS truth
trigger replay if contradiction is high
rank candidates
keep AI candidates separate from validated facts
promote only replay-backed / validated claims
```

The shader should not call SPARQL in the hot path. Oxigraph is the formal ontology/RDF substrate. AriGraph is semantic episodic memory. `cognitive-shader-driver` is the hot SoA runtime.

Architecture sentence:

```text
Oxigraph supplies ontology context.
AriGraph supplies episodic meaning.
lance-graph/SPO supplies truth/replay.
cognitive-shader-driver compiles all of it into SoA lanes for fast witness formation.
```

---

## 5. 64x64 cause/effect reasoning lattice

The first reasoning plane is 64x64:

```text
64 cause  = GestaltCause64
64 effect = ThinkingEffect64
```

### 5.1 GestaltCause64

Classifies the shape of the current situation:

```text
missing object
missing predicate
missing cause
contradicted triad
timeline conflict
source disagreement
weak evidence
histology contradiction
unexplained anomaly
high confidence / low witness
high support / high contradiction
```

It asks:

```text
What kind of causal/semantic situation is this?
```

### 5.2 ThinkingEffect64

Selects the reasoning move:

```text
deduction
abduction
association
synthesis
counterfactual
replay validation
source diversification
fan-out
collapse candidate
ask for more context
future prefetch
introspection
revision
```

It asks:

```text
What thinking action should transform this situation?
```

### 5.3 Perturbationslernen

The learned mapping:

```text
GestaltCause64 -> ThinkingEffect64
```

conditioned by:

```text
meaning region
ontology context
attention state
witness quorum
entropy state
Markov history
```

Reward is not naked confidence. Reward is insight/epiphany pressure:

```text
hole closure
support gain
contradiction isolation
entropy reduction
attention compression
trajectory convergence
downstream question clarity
replay coherence
low compute cost
```

---

## 6. 256x256 palette/attention plane

The second reasoning plane is 256x256:

```text
256 palette   = semantic/sigma/Fisher-Z/CAM bucket
256 attention = read/focus/rank/aperture bucket
```

### 6.1 Palette256

Controls semantic exposure:

```text
cosine bucket
Fisher-Z bucket
semantic CAM distance
sigma width
theta aperture
texture bucket
uncertainty band
ontology-context filter
```

### 6.2 Attention256

Controls readout/focus:

```text
top-k priority
softmax-like rank
fanout budget
replay radius
contradiction sensitivity
collapse strictness
source diversity
prefetch budget
```

Meaning:

```text
Palette256 x Attention256 controls how the semantic field is exposed and read.
```

---

## 7. 4096x4096 SPO/Pearl 2^3 field

The third reasoning plane is 4096x4096:

```text
4096 = COCA / CAM meaning address basis
```

SPO:

```text
S = Subject
P = Predicate
O = Object
```

Pearl 2^3 projections:

```text
___ = prior / none
S__ = subject only
_P_ = predicate only
__O = object only
SP_ = subject + predicate asks object
S_O = subject + object asks predicate
_PO = predicate + object asks subject
SPO = full triad / validation / contradiction / counterfactual / witness query
```

Core hole queries:

```text
SP_ -> asks O
S_O -> asks P
_PO -> asks S
SPO -> asks witness / validation / contradiction
```

Expensive semantic query:

```text
[S{4096}, P{4096}] -> O candidates
[S{4096}, O{4096}] -> P candidates
[P{4096}, O{4096}] -> S candidates
```

Do not run this blindly per question. Batch active holes into L1 supercycles.

---

## 8. SPOW tetrahedral extension

Add W:

```text
W = witness / why / worldline / evidence / provenance
```

Tetrahedron:

```text
S / P / O / W
```

Hole forms:

```text
SP_ asks O
S_O asks P
_PO asks S
SPO asks W
SPW asks O
SOW asks P
POW asks S
```

This is the semantic case-board / spatial crossword. A candidate is good when it fills one hole and makes nearby holes cleaner.

---

## 9. Gaussian/Fisher-Z splat as spatial BLASGraph expansion

This is the missing idea that other sessions often lose.

Spatial Gaussian/Fisher-Z splat is not nearest-neighbor search. It is hypothesis consequence expansion over a semantic graph/field.

Think of it as a BLASGraph-style operation:

```text
candidate vector / semantic address
x local graph neighborhood / hole board
x Gaussian/Fisher-Z kernel
x context mask
x attention aperture
-> pressure field over neighboring holes
```

More operationally:

```text
Given candidate C for hole H:
  splat C across the local SPOW neighborhood
  under the selected cause/effect and palette/attention settings
  then score how the surrounding hole board changes.
```

Inputs:

```text
candidate answer
GestaltCause64
ThinkingEffect64
Palette256
Attention256
ReasoningWitness64 / NARS truth
ontology_context_id
sigma/theta
Markov history refs
replay refs
```

The splat produces pressure over nearby holes:

```text
current hole closure
neighbor support increase
contradiction localization
entropy decrease
downstream hole clarity
replay coherence
future/prefetch candidate holes
```

This is why it is a shader. It transforms candidate pressure spatially across the semantic lattice, much like a graphical splat spreads energy over pixels, except the target is a graph of semantic holes.

Good candidate:

```text
fills current hole
raises support
localizes contradiction
lowers entropy
makes next holes clearer
survives replay / NARS / AriGraph validation
```

Bad candidate:

```text
fills one hole but increases global disorder
spreads contradiction
requires too-wide attention
breaks replay coherence
```

---

## 10. TetraSplatCertificate and EpiphanySignal

Suggested certificate:

```rust
pub struct TetraSplatCertificate {
    pub thought_id: u64,
    pub cycle_id: u64,
    pub candidate_id: u64,

    pub filled_slot: TetraSlot,
    pub gestalt_cause64: u8,
    pub thinking_effect64: u8,
    pub palette256: u8,
    pub attention256: u8,
    pub ontology_context_id: u32,

    pub support_pressure_q8: u8,
    pub contradiction_pressure_q8: u8,
    pub entropy_after_q8: u8,
    pub downstream_clarity_q8: u8,
    pub replay_coherence_q8: u8,

    pub future_prefetch_count: u16,
    pub exact_replay_required: bool,
    pub replay_ref: u64,
}
```

Suggested reward:

```rust
pub struct EpiphanySignal {
    pub hole_closure_q8: u8,
    pub support_gain_q8: u8,
    pub contradiction_isolation_q8: u8,
    pub entropy_reduction_q8: u8,
    pub attention_compression_q8: u8,
    pub trajectory_convergence_q8: u8,
    pub downstream_question_clarity_q8: u8,
    pub replay_coherence_q8: u8,
    pub compute_cost_q8: u8,
    pub reward_q8: u8,
}
```

Epiphany is not high confidence. Epiphany is a useful phase transition:

```text
many possible cause/effect pairings
-> one high-resonance transition
-> hole fills
-> entropy drops
-> contradiction localizes
-> next holes become structured
-> replay coherence rises
```

---

## 11. Markov history and future prefetch

The system may have 4096 units of Markov history/replay.

Do not hydrate -500..+500 sentences against the graph per question. That is too expensive.

Use L1 supercycles:

```text
1. collect active holes from current frame/text/episode
2. include Markov history/replay refs
3. classify holes by cause/effect/projection/context
4. group by palette/attention and ontology_context_id
5. run expensive 4096x4096/CAM exposure once per group
6. splat candidates over current and neighboring holes
7. prefetch likely future holes from Markov history and temporal predictor
8. exact-replay only survivors
9. promote only validated facts
```

Future prefetch means:

```text
Use TemporalPredictor / Markov history to guess which holes are likely to matter next.
Store them as scenario/prefetch/candidate context.
Do not promote forecast facts directly.
```

Context names for prefetch:

```text
graph:future-prefetch
graph:scenario
graph:temporal-forecast
```

These are inner ontology contexts, not outer facts.

---

## 12. Existing repo signals to integrate

Before editing, inspect actual code. Do not invent types if nearby ones already exist.

Relevant areas:

```text
crates/lance-graph/src/graph/arigraph/
crates/lance-graph/src/graph/spo/
crates/cognitive-shader-driver/src/
crates/lance-graph-planner/src/cache/nars_engine.rs
crates/lance-graph-planner/src/cache/triple_model.rs
crates/lance-graph-planner/src/cache/candidate_pool.rs
crates/lance-graph-planner/src/cache/convergence.rs
crates/lance-graph-planner/src/prediction/temporal.rs
crates/lance-graph-planner/src/mul/dk.rs
crates/lance-graph-planner/src/mul/trust.rs
crates/lance-graph-planner/src/mul/compass.rs
.claude/plans/tetrahedral-epiphany-splat-integration-v1.md
```

Known conceptual anchors:

```text
nars_engine.rs:
  S/P/O palette indices
  NARS frequency/confidence
  Pearl 2^3 masks
  inference type
  temporal bucket
  style vectors over Pearl projections

candidate_pool.rs:
  composition phases
  Pointe = low surprise + high alignment, adjacent to epiphany

convergence.rs:
  AriGraph triplets -> p64/palette/cognitive shader
  8 predicate layers
  64x64 attention = 4096 heads
  CausalEdge64 forward/learn intent
  NARS revision

triple_model.rs:
  self/user/impact models
  adjacent to cause/effect learning

temporal.rs:
  replay horizon / future implication / prefetch

dk.rs:
  overconfidence / humility gate

trust.rs:
  source/witness weighting

compass.rs:
  attention direction:
    truth / exploration / safety / efficiency / balance
```

---

## 13. First implementation slice

### PR 1: RDF context crate

Create:

```text
crates/lance-graph-rdf/
```

Add:

```text
OntologyContextId
NamedGraphRegistry
SemanticQuad
TermId / GraphId mapping skeleton
optional feature flag for oxigraph
```

### PR 2: Shader wire DTOs

In:

```text
crates/cognitive-shader-driver/src/wire.rs
```

Add:

```text
SemanticSpoRow
SemanticSpoBatch
```

### PR 3: Bindspace SoA

In:

```text
crates/cognitive-shader-driver/src/bindspace.rs
```

Add:

```text
SemanticSpoSoA
append_semantic_spo_batch()
filter_by_context()
group_same_spo_across_contexts()
derive_context_witness()
```

### PR 4: tests

Test same or related SPO in multiple contexts:

```text
ROI_55 supports Tendinopathy graph:ai-candidates
ROI_55 supports Tendinopathy graph:doctor-review
Histology_9 contradicts Tendinopathy graph:histology
```

Verify:

```text
contexts remain separate
same SPO can be grouped
support/contradiction lanes derive witness summary
no context soup
```

### PR 5: splat docs / stubs

Add stubs or docs for:

```text
TetraSplatCertificate
EpiphanySignal
splat_consequence_search()
prefetch_future_holes()
```

Implementation can be heuristic first.

---

## 14. Constraints

Do not:

```text
turn hot path into SPARQL
replace AriGraph
replace existing SPO truth/semiring/Merkle bridge
merge billing/FMA/clinical domains semantically
promote future prefetch as fact
reward naked confidence alone
```

Do:

```text
use ontology_context_id to separate domains
treat AI candidates as speculative inner ontology
treat validated facts as outer ontology
use histology/final Befund as high-authority evidence contexts
require replay/NARS/AriGraph validation for promotion
let the shader propose, splat, prefetch, rank, and form witnesses
let the membrane commit only validated/replay-backed facts
```

---

## 15. Final distilled architecture sentence

The new shader compiles Oxigraph RDF quads, AriGraph semantic episodes, and lance-graph SPO truth into one context-aware SoA format. The `ontology_context_id` column preserves separation of concerns; the shader then runs fast support/contradiction/witness/replay passes over the rows. On top of this, 64x64 cause/effect, 256x256 palette/attention, and 4096x4096 SPO/Pearl holes form the reasoning lattice. Gaussian/Fisher-Z spatial splat search acts as a BLASGraph-style consequence expansion over the SPOW hole board, while 4096 Markov history and temporal prediction prefetch future holes without promoting them until validated.
