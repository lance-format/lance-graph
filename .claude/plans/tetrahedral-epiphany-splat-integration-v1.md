# Tetrahedral Epiphany Splat Integration Plan v1

Status: integration plan / architecture proposal. No runtime implementation is claimed here.

## One-sentence thesis

Use the SPOW tetrahedral grid as a constraint board: two known COCA factors and a projection hole ask for the missing factor or witness; paired 64-axis cause/effect trajectories and paired 256-axis attention masks steer Gaussian/Fisher-Z CAM splats across the board; Perturbationslernen learns which thinking effect fills holes, lowers entropy, and creates the felt signature of an epiphany.

In shorthand:

```text
S/P/O/W tetrahedron
  + GestaltCause64
  + ThinkingEffect64
  + AttentionIn256
  + AttentionOut256
  + COCA4096 meaning
  -> spatial splat over holes
  -> witness quorum
  -> insight threshold
  -> replay-certified candidate
```

---

## Core model

The 4096x4096 field is not primarily a distance ledger. It is a deterministic question surface:

```text
[A{4096}, B{4096}]::[projection_mask]() -> C{4096}
```

After adding witness/worldline as the fourth vertex, the local reasoning primitive becomes:

```text
S / P / O / W
```

where:

```text
S = subject / source / factor A
P = predicate / process / factor B
O = object / outcome / factor C
W = witness / why / worldline / weight
```

The SPO face expresses the semantic proposition. The W vertex explains why the move is allowed.

---

## Paired axes

Every layer has two semantic control dimensions:

```text
2 x 64:
  GestaltCause64    = causal/semantic shape of the current hole-state
  ThinkingEffect64  = cognitive move selected to transform that state

2 x 256:
  AttentionIn256    = perception/search aperture
  AttentionOut256   = action/collapse/fanout aperture

4096:
  COCA meaning basis
```

The transition is:

```text
GestaltCause64 + AttentionIn256 + meaning4096
  -> ThinkingEffect64 + AttentionOut256
```

This is the small control grammar for Perturbationslernen.

---

## Tetrahedral holes

The grid can ask for semantic slots or the witness slot:

```text
SP_  asks O
S_O  asks P
_PO  asks S
SPO  asks W
SPW  asks O
SOW  asks P
POW  asks S
```

This expands classic SPO completion into metacognitive completion:

```text
not only: what object completes this relation?
but also: what witness/angle makes this local world consistent?
```

---

## Criminal-case / crossword analogy

A reasoning state is like a case board or spatial crossword.

Holes include:

```text
who / subject
what / predicate
outcome / object
why / witness
motive
means
opportunity
timeline
source reliability
contradiction
```

A candidate answer is useful when it does not merely fill one blank, but makes neighboring blanks easier.

The system should learn to challenge the angle, fan out hypotheses, and rotate thinking styles until:

```text
motive, means, opportunity, timeline, witnesses, and contradictions
become mutually constraining
```

That is the epiphany moment.

---

## Spatial Gaussian / Fisher-Z splat role

Spatial splat search is not nearest-neighbor decoration. It is hypothesis consequence search.

A candidate fill is splatted across neighboring SPOW holes:

```text
candidate answer
  + GestaltCause64
  + ThinkingEffect64
  + AttentionIn256 / AttentionOut256
  + ReasoningWitness64 quorum
  + sigma/theta
  -> pressure over neighboring holes
```

A good splat:

```text
closes the current hole
raises support pressure
localizes contradiction
lowers entropy
creates cleaner downstream holes
survives replay / AriGraph / NARS validation
```

A bad splat:

```text
fills one hole but increases global disorder
spreads contradiction
requires too-wide attention
destroys replay coherence
```

---

## Epiphany as reinforcement signal

An epiphany is a rewarded transition where paired cause/effect trajectories align and attention compresses.

Operational signature:

```text
many possible cause-effect pairings
  -> one high-resonance transition
  -> hole fills
  -> entropy drops
  -> contradiction localizes
  -> next holes become structured
  -> replay coherence rises
```

Reward components:

```text
+ hole closure
+ support gain
+ contradiction isolation
+ attention compression
+ trajectory convergence
+ downstream question clarity
+ replay coherence
- entropy increase
- compute cost
- false promotion penalty
```

The system should optimize for insight transitions, not raw confidence alone.

---

## Core DTOs / contracts

### TetrahedralHole

```rust
pub struct TetrahedralHole {
    pub subject: Option<u16>,      // S / factor A
    pub predicate: Option<u16>,    // P / factor B
    pub object: Option<u16>,       // O / factor C
    pub witness: Option<ReasoningWitness64>, // W

    pub missing_slot: TetraSlot,

    pub gestalt_cause: u8,         // 0..63
    pub attention_in: u8,          // 0..255

    pub witness_offset: u32,
    pub witness_len: u16,
    pub replay_ref: u64,
}

pub enum TetraSlot {
    Subject,
    Predicate,
    Object,
    Witness,
}
```

### TetrahedralMove

```rust
pub struct TetrahedralMove {
    pub thinking_effect: u8,       // 0..63
    pub attention_out: u8,         // 0..255
    pub candidate: TetraCandidate,
    pub epiphany_reward_q8: u8,
    pub replay_ref: u64,
}
```

### TetraCandidate

```rust
pub enum TetraCandidate {
    Subject(u16),
    Predicate(u16),
    Object(u16),
    Witness(ReasoningWitness64),
}
```

### EpiphanySignal

```rust
pub struct EpiphanySignal {
    pub hole_closure_q8: u8,
    pub support_gain_q8: u8,
    pub contradiction_isolation_q8: u8,
    pub attention_compression_q8: u8,
    pub trajectory_convergence_q8: u8,
    pub downstream_question_clarity_q8: u8,
    pub replay_coherence_q8: u8,
    pub entropy_reduction_q8: u8,
    pub compute_cost_q8: u8,
    pub reward_q8: u8,
}
```

### GestaltEffectTransition

```rust
pub struct GestaltEffectTransition {
    pub thought_id: u64,
    pub cycle_id: u64,

    pub gestalt_cause: u8,
    pub thinking_effect: u8,
    pub attention_in: u8,
    pub attention_out: u8,

    pub meaning_anchor_a: Option<u16>,
    pub meaning_anchor_b: Option<u16>,
    pub meaning_answer: Option<u16>,

    pub projection_mask: u8,
    pub witness_offset: u32,
    pub witness_len: u16,

    pub epiphany: EpiphanySignal,
    pub replay_ref: u64,
}
```

---

## Integration with ThoughtCycleSoA

Add or connect lanes:

```text
thought_id
cycle_id
subject / factor_a
predicate / factor_b
object / factor_c
witness_offset
witness_len
ReasoningWitness64[]
gestalt_cause64
thinking_effect64
attention_in256
attention_out256
missing_slot / projection_mask
sigma
theta
replay_ref
epiphany_reward
```

The inner ontology writes these lanes. The promotion membrane reads them. Lance/AriGraph replay certifies them. The outer ontology only receives promoted facts/candidates.

---

## L1 supercycle execution

The expensive 4096x4096/CAM exposure should be amortized by asking all active holes in one supercycle.

```text
1. L1 collects all active SPOW holes.
2. Resonance predicts GestaltCause64 for each hole.
3. Perturbationslernen predicts ThinkingEffect64 + AttentionOut256.
4. Group holes by trajectory/effect/attention/projection.
5. Reorder CAM register / palette view once per group.
6. Run spatial Gaussian/Fisher-Z splat over answer space.
7. Score hole closure and neighboring constraint impact.
8. Append witnesses and update SoA lanes.
9. Only insight-threshold crossings reach the promotion membrane.
```

The rule:

```text
Do not pay 4096x4096 per question.
Pay once per attention supercycle.
```

---

## Perturbationslernen target

The learner optimizes:

```text
P(
  ThinkingEffect64,
  AttentionOut256
  |
  GestaltCause64,
  AttentionIn256,
  meaning4096 region,
  hole pattern,
  witness quorum,
  entropy state
)
```

The learning target is not truth itself. It is the next perturbation that best lowers entropy and structures the case board.

```text
learn where to push
learn how hard to push
learn which angle to challenge
learn which splat radius to use
learn when to replay instead of fan out
```

Truth remains with NARS/AriGraph/replay/promotion membrane.

---

## Splat certificate

Any candidate produced by spatial splatting should carry a certificate:

```rust
pub struct TetraSplatCertificate {
    pub thought_id: u64,
    pub cycle_id: u64,
    pub candidate_id: u64,

    pub gestalt_cause: u8,
    pub thinking_effect: u8,
    pub attention_in: u8,
    pub attention_out: u8,

    pub current_hole: TetraSlot,
    pub filled_slot: TetraSlot,

    pub support_pressure_q8: u8,
    pub contradiction_pressure_q8: u8,
    pub entropy_after_q8: u8,
    pub downstream_clarity_q8: u8,
    pub replay_coherence_q8: u8,

    pub exact_replay_required: bool,
    pub replay_ref: u64,
}
```

Promotion law:

```text
support high + contradiction low + entropy reduced + replay coherent:
  candidate may proceed to NARS/AriGraph validation

support high + contradiction high:
  exact replay required

forecast/counterfactual source:
  scenario or prefetch only, no fact promotion

introspection effect:
  policy update only, no ontology fact
```

---

## PR sequence

### PR 1: Contracts

Add:

```text
TetraSlot
TetrahedralHole
TetraCandidate
TetrahedralMove
EpiphanySignal
GestaltEffectTransition
TetraSplatCertificate
```

### PR 2: SoA lanes

Attach tetrahedral state, cause/effect64, attention in/out256, and epiphany reward to ThoughtCycleSoA.

### PR 3: Hole board

Implement L1 `TetraHoleBoard` for active holes and status tracking.

### PR 4: Style/effect prediction

Implement first deterministic heuristic predictor:

```text
missing object + high support -> deduction
missing cause + observed outcome -> abduction
contradiction in full triad -> revision/counterfactual
single factor only -> association/fanout
```

Later replace or augment with learned Perturbationslernen policy.

### PR 5: Splat search

Implement Gaussian/Fisher-Z CAM splat from candidate to neighboring holes, producing `TetraSplatCertificate`.

### PR 6: Epiphany reward

Compute `EpiphanySignal` from hole closure, entropy delta, support/contradiction, downstream clarity, and replay coherence.

### PR 7: Supercycle batching

Batch holes by GestaltCause64 / ThinkingEffect64 / attention palette / projection mask to amortize CAM exposure.

### PR 8: Replay and validation

Route only insight-threshold candidates to NARS/AriGraph and outer ontology promotion.

### PR 9: Learning table

Add a compact policy table:

```text
GestaltCause64 x meaning_region x AttentionIn256 -> ThinkingEffect64 / AttentionOut256
```

Record reward statistics from `EpiphanySignal`.

---

## Suggested architecture wording

Use this:

```text
The system treats reasoning as a tetrahedral constraint board over S/P/O/W. S/P/O are semantic factors; W is the witness/worldline. Each active hole is classified by GestaltCause64 and an input attention bucket. Resonance-based thinking predicts a ThinkingEffect64 and output attention bucket, then spatial Gaussian/Fisher-Z CAM splatting tests the consequence of candidate fills across neighboring holes. Perturbationslernen rewards moves that close holes, reduce entropy, isolate contradiction, and create cleaner downstream questions. The promotion membrane remains the only path from candidate pressure to outer ontology facts.
```

Avoid this:

```text
The system twists knobs until it feels true.
```

Better:

```text
The system searches effect64/attention256 perturbations and rewards only replayable constraint improvements that survive NARS/AriGraph validation.
```

---

## Final law

```text
The inner field may be playful and speculative.
The outer ontology must be boring and replay-certified.

Splatting proposes.
Perturbationslernen learns the angle.
NARS/AriGraph validate.
The membrane commits.
```
