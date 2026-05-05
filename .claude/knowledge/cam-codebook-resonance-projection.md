# CAM codebook resonance projection: 4096x4096 SPO 2^3 + ReasoningWitness64 -> 16,384

Status: contextual architecture note. This document explains the model; it does not claim full runtime implementation.

## One-sentence thesis

`4096x4096 SPO 2^3 + ReasoningWitness64 -> 16,384` is a deterministic CAM/COCA codebook schema: pairwise factor interactions ask the missing third factor, `ReasoningWitness64` carries the NARS/style/causal transition angle, and the result is projected into a 16,384-bit awareness plane as a replayable superposition resonance state.

In plain language:

```text
4096x4096 = factor-pair address space
SPO 2^3   = triadic projection grammar
64u       = reasoning witness angle
16,384    = CAM/COCA awareness codebook plane
```

---

## Losslessness claim: what is guaranteed

The important claim is not merely that the 16,384-bit plane is small. The claim is that the projection can be treated as lossless **within the calibrated COCA/CAM codebook contract**.

The calibrated premise:

```text
4096 COCA codebook entries
  calibrated from frequent English speaker vocabulary / meaning atoms

4096 x 4096 pairwise factor field
  deterministic factor-pair address space

SPO / ABC 2^3 projection grammar
  deterministic observed/missing-factor masks

ReasoningWitness64 sidecar
  preserves NARS truth/style/source/projection transition angle

16,384-bit awareness plane
  materialized CAM/COCA resonance state
```

So the guarantee is not:

```text
all natural language meaning is magically lossless
```

The guarantee is:

```text
Given the fixed 4096 COCA codebook, fixed projection grammar, fixed witness layout,
fixed seeds/codebook versions, and deterministic kernels, the system can replay the
same thought-state projection without losing the encoded factors, projection mask,
reasoning angle, and cycle lineage.
```

That is a codebook-level losslessness guarantee.

Full sentence text, full parse details, full NARS terms, and ontology provenance still live in replay. The hot plane is lossless for the compressed codebook state, not for every raw source artifact.

---

## Why 4096x4096 -> 16,384 is not arbitrary compression

The arrow does not mean a dense 16,777,216-cell matrix is crushed blindly into 16,384 bits.

It means:

```text
4096x4096 defines the deterministic pairwise address space.
The active pair/projection/witness combinations are folded into a 16,384-bit
COCA/CAM awareness plane using the calibrated codebook mapping.
```

This is closer to a deterministic codebook projection than to lossy vector compression.

The `4096` basis matters because it is the calibrated vocabulary/meaning-atom codebook. The `16,384` plane is the replayable resonance surface over that codebook, with grammar and reasoning sidecars preserving interpretation and transition evidence.

In shorthand:

```text
4096-codebook atoms give the alphabet.
4096x4096 gives the pairwise grammar field.
SPO 2^3 gives the missing-factor query masks.
ReasoningWitness64 gives the truth/style angle.
16,384 bits gives the deterministic awareness projection.
```

---

## Why this is not a dense monster

The architecture should not be read as physically materializing a dense `16k x 16k` float matrix.

The materialized cycle state is:

```text
16,384 bits = 2,048 bytes = 2 KiB
```

So:

```text
2 KiB x 16,384 cycles = 33,554,432 bytes ~= 32 MiB
```

The large geometry is the implicit address/completion space. The hot state is a compact CAM/COCA-coded bitplane plus sidecar witnesses.

Use this language:

```text
The 4096x4096 field is the pairwise factor-interaction address space.
The 16,384-bit plane is the materialized cycle activation state.
The 20k/16k-scale completion surface is implicit, tiled, or survivor-hydrated.
ReasoningWitness64 stores the transition angle for replay and next-cycle policy.
```

Do not describe the hot path as a dense adjacency matrix unless a specific implementation really allocates one.

---

## Triadic factor model

The core is not hard-coded to RDF/SPO. It is a generic triadic completion substrate.

```text
A, B, C = three factors
any two factors can ask for the third
```

SPO is one lens:

```text
A = Subject
B = Predicate
C = Object
```

Other lenses are legal:

```text
sonography:
  A = ImageRegion
  B = Texture/Feature
  C = TissueClass/Finding

clinical:
  A = Patient/Context
  B = Measurement/Relation
  C = Finding/Outcome

text:
  A = Actor/Entity
  B = Action/Relation
  C = Target/Claim

causal/Pearl-like:
  A = Intervention/Condition
  B = Mechanism/Relation
  C = Outcome
```

This is why the inner field should use agnostic names such as `factor_a`, `factor_b`, `factor_c`, `projection_kind`, and `relation_family`. The outer ontology supplies domain names later.

---

## SPO 2^3 decomposition

For a triad `(A, B, C)`, there are 2^3 observation/query masks.

Recommended projection enum:

```rust
pub enum TriadicProjection {
    Abc,        // full observed triple
    AbAskC,    // A + B ask C
    AcAskB,    // A + C ask B
    BcAskA,    // B + C ask A
    AOnly,
    BOnly,
    COnly,
    Background,
}
```

SPO mapping:

```text
ABC = full SPO observed
AB_ = Subject + Predicate asks Object
A_C = Subject + Object asks Predicate
_BC = Predicate + Object asks Subject
```

The `2^3` structure is not decorative. It is the query grammar for the CAM field.

---

## 4096x4096 as CAM address space

`4096x4096` is the base pairwise interaction grid:

```text
factor_a in 0..4095
factor_b in 0..4095
cell(a,b) asks or accumulates evidence for factor_c
```

The grid can be interpreted as:

```text
Subject x Predicate -> Object pressure
Subject x Object    -> Predicate pressure
Predicate x Object  -> Subject pressure
```

or generically:

```text
A x B -> C
A x C -> B
B x C -> A
```

This gives a deterministic completion schema:

```text
factor pair
  -> projection kind
  -> CAM address
  -> codebook activation
  -> witness-weighted resonance
  -> candidate missing factor
```

The pairwise grid does not need to own the entire ontology. It only needs stable factor IDs and a projection lens.

---

## 16,384 as awareness codebook plane

The 16,384 target is the compact CAM/COCA codebook plane:

```rust
#[repr(align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AwarenessPlane16K {
    pub words: [u64; 256], // 16,384 bits = 2 KiB
}
```

Interpretation:

```text
bit i = activation/resonance presence for codebook coordinate i
```

The plane may encode:

```text
active factors
role-bound projections
SPO/triadic completion pressure
Markov context residue
style-conditioned survivor activations
AriGraph memory support traces
```

But the plane itself does not store full grammar, full text, full NARS terms, or full ontology objects. Those live in sidecars and replay.

---

## ReasoningWitness64 as transition angle

The 64-bit carrier is not a dead register. Inside the ThoughtCycleSoA it is the transition witness for the field.

Recommended single shared type:

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

Meaning:

```text
low byte        = BF16-ish local evidence charge
NARS bytes      = truth texture
middle fields   = relation/projection/time
upper fields    = thinking style, perturbation, source, generation
```

This makes the `64u` the NARS reasoning witness angle for the field.

---

## Deterministic superposition resonance projection

The full projection can be described as a deterministic map:

```text
factor_a
factor_b
projection_kind
GrammarMarkovLens64
ReasoningWitness64[]
sigma_idx
theta aperture
  -> AwarenessPlane16K
```

A simple mental model:

```text
address = hash_or_codebook(factor_a, factor_b, projection_kind, relation_family)
weight  = f(NARS frequency, NARS confidence, evidence mantissa, polarity)
style   = witness.thinking_style
lane    = witness.source_lane

plane[address] accumulates resonance under deterministic rules
```

This is a superposition because multiple witnesses and projections can contribute to the same 16K plane. It is deterministic because the codebook mapping, role keys, projection masks, and witness bit fields are fixed and replayable.

The result is not one symbolic assertion. It is a resonance state that can later collapse into candidates.

---

## How grammar and thinking merge

The same `ReasoningWitness64` must be used by grammar and thinking.

Grammar parse emits:

```text
AwarenessPlane16K
GrammarMarkovLens64
ReasoningWitness64[]
```

The witness already says whether the sentence object was parsed as:

```text
deduction
abduction
inference
association
counterfactual
synthesis
introspection/metacognition
```

Then Markov revises the same witness, not a different one.

```text
parse-time witness
  -> Local5 Markov revision
  -> Section50 replay revision
  -> Document500 replay revision
  -> NARS/AriGraph revision
```

This is the important side effect:

```text
text parsing is already semantic awareness, not a dead preprocessor
```

---

## GrammarMarkovLens64 is the interpretation lens

`GrammarMarkovLens64` should remain separate from the field and witness.

```rust
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GrammarMarkovLens64(pub u64);
```

Suggested layout:

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

Role:

```text
plane   = what is active
lens    = how it was parsed/contextualized
witness = why it mattered and how it transitions
```

---

## ThoughtCycleSoA is the owner

Do not let grammar, Markov, shader, and NARS become parallel side processes. They must attach to one cycle spine.

```rust
pub struct ThoughtCycleSoA {
    pub thought_id: Vec<u64>,
    pub cycle_id: Vec<u64>,

    pub plane: Vec<AwarenessPlane16K>,
    pub grammar_lens: Vec<GrammarMarkovLens64>,

    pub witness_offset: Vec<u32>,
    pub witness_len: Vec<u16>,
    pub witnesses: Vec<ReasoningWitness64>,

    pub sigma: Vec<u8>,
    pub theta_accept_q8: Vec<u8>,
    pub theta_width_q8: Vec<u8>,

    pub dominant_style: Vec<u8>,
    pub dominant_perturbation: Vec<u8>,
    pub replay_ref: Vec<u64>,
}
```

Separate bundles are allowed. Separate thought identity is not.

---

## Tiered context hydration

Hydrating `-500..+500` sentences against graph memory is not free. Use staged replay.

```text
Local5:
  parse-time witness revision and local ambiguity handling

Section50:
  section-level support/contradiction and role continuity

Document500:
  document-level delayed reference, argument continuity, episodic memory support

Graph hydration:
  only for candidates that survive witness, sigma, theta, and replay gates
```

All tiers share the same `thought_id` and produce/revise the same witness format.

---

## Sigma and theta role

Sigma is the uncertainty geometry index:

```text
sigma: u8 -> SigmaCodebook[256] -> Spd2 -> ewa_sandwich propagation
```

Theta is not a constant. It is a dynamic aperture:

```text
theta_accept = candidate survival aperture
theta_width  = hydration/splat width aperture
```

Input to theta modulation:

```text
sigma log-norm growth
NARS confidence mass
contradiction mass
thinking style
perturbation class
AriGraph support
Chronos/anomaly hints
source entropy
```

Law:

```text
Sigma bound = guardrail.
Theta = steering/aperture.
```

---

## Hydration/collapse path

Hydration consumes hot objects only:

```text
AwarenessPlane16K
GrammarMarkovLens64
ReasoningWitness64[]
sigma_idx
theta_accept_q8
theta_width_q8
replay_ref
```

It emits:

```text
HydrationCandidate
  factor_a
  factor_b
  factor_c or missing-factor request
  projection kind
  energy
  support mass
  contradiction mass
  replay ref
```

Then:

```text
HydrationCandidate
  -> NARS/AriGraph validation
  -> revised ReasoningWitness64
  -> next ThoughtCycleSoA row
  -> optional ontology candidate
```

The outer ontology receives collapsed objects, never raw hot-layer planes as facts.

---

## Why this is a CAM codebook schema

It has the required properties:

1. Fixed address space

```text
4096x4096 factor-pair grid
```

2. Deterministic projection grammar

```text
SPO/ABC 2^3 masks
```

3. Compact activation state

```text
AwarenessPlane16K = 2 KiB per cycle
```

4. Sidecar transition witness

```text
ReasoningWitness64 = NARS/style/causal/replay angle
```

5. Replayable superposition

```text
multiple witnesses and projections can activate the same plane deterministically
```

6. Controlled collapse

```text
sigma + theta + NARS/AriGraph gates decide candidates
```

7. Codebook-level losslessness

```text
within the fixed 4096 COCA codebook and deterministic projection/witness contract,
the thought-state projection can be replayed without losing encoded factor,
projection, witness, and cycle-lineage information
```

This is why the architecture should be called a deterministic superposition resonance projection rather than a dense graph matrix.

---

## Suggested wording for architecture reviews

Use this:

```text
The 4096x4096 substrate is a pairwise factor-interaction CAM address space calibrated on a 4096-entry COCA meaning/codebook basis derived from frequent English speaker vocabulary. SPO is represented as a generic ABC triadic lens with 2^3 projection masks, allowing any two factors to query or support the third. Each cycle materializes only a 16,384-bit CAM/COCA awareness plane, not a dense matrix. ReasoningWitness64 carries the quantized NARS truth, relation family, projection kind, temporal bucket, thinking style, perturbation class, source lane, and generation for each transition. The resulting plane is a deterministic superposition of replayable witness-weighted projections. Given fixed codebook versions, projection masks, witness layout, seeds, and kernels, this gives a codebook-level losslessness guarantee for replaying the encoded thought state. Sigma controls uncertainty geometry, theta controls dynamic aperture, and NARS/AriGraph validation controls collapse into ontology candidates.
```

Avoid this:

```text
We store all meanings in a 16k x 16k matrix.
```

Better:

```text
We store a compact 16kbit cycle plane that indexes an implicit 4096x4096 triadic completion field over a calibrated 4096-entry COCA codebook.
```

---

## Integration with `thought-cycle-soa-awareness-integration-v1.md`

This document explains why the carrier schema is coherent.

The implementation plan lives in:

```text
.claude/plans/thought-cycle-soa-awareness-integration-v1.md
```

The relationship is:

```text
this document:
  why 4096x4096 SPO 2^3 + ReasoningWitness64 -> 16,384 is a CAM codebook schema

integration plan:
  how to wire the types, SoA, replay, sigma/theta, hydration, and ontology collapse
```
