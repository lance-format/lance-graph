# Markov triplet query quorum model

Status: contextual architecture note. This document reframes the 4096x4096 field as a deterministic query surface rather than a semantic distance table.

## Thesis

The 4096x4096 field has one primary job:

```text
Given two deterministic COCA positions A and B, and a 2^3 ABC/SPO projection mask, ask for the corresponding third factor C.
```

So the hot syntax is closer to:

```text
[A{4096}, B{4096}]::[projection_mask]() -> C candidates
```

The system does not need to store a dense semantic-distance ledger for all pairs. It needs the two factor IDs, the projection mask, a witness quorum, and replay lineage.

## CausalEdge64 as compressed projection story

`ReasoningWitness64` / `CausalEdge64` already carries the compact story of a projection event:

```text
NARS frequency/confidence
support or contradiction polarity
relation family
projection kind
temporal bucket
thinking style
perturbation class
source lane
generation
```

That means it can say which projection was used, which reasoning mode produced it, and what truth texture the transition carried.

## Thinking atoms

Use a small finite set of thinking atoms as provenance for the transition. Start with 16 and leave room for 64.

Examples:

```text
deduction
extrapolation
counterfactual
synthesis
inference
association
abduction
fan-out
intuition
introspection
analogy
revision
retrieval
causal tracing
contradiction seeking
scenario planning
```

A compact mask can preserve the full quorum when more than one atom contributed:

```rust
#[repr(transparent)]
pub struct ThinkingAtomMask64(pub u64);
```

The dominant style may live inside `ReasoningWitness64`; the full contributing set can live in `ThinkingAtomMask64`.

## Projection masks as stated/open slots

For an ABC triad:

```text
ABC   = all factors stated
AB_   = C is queried
A_C   = B is queried
_BC   = A is queried
AOnly = B and C queried
BOnly = A and C queried
COnly = A and B queried
```

A thinking atom transforms one stated/open pattern into another. For example:

```text
deduction:      AB_ -> ABC candidate
abduction:      _BC -> ABC candidate with weak explanatory support
association:    AOnly -> AB_ or A_C candidate family
synthesis:      several partial states -> ABC candidate
counterfactual: ABC -> scenario state under altered premise
introspection:  any state -> policy/style update, not fact promotion
```

## Markov triplet state

The Markov chain can evolve over triplet states rather than only tokens or vectors.

```rust
pub struct MarkovTripletState {
    pub factor_a: Option<u16>,
    pub factor_b: Option<u16>,
    pub factor_c: Option<u16>,
    pub projection: TriadicProjection,
    pub thinking_atoms: ThinkingAtomMask64,
    pub witness_offset: u32,
    pub witness_len: u16,
    pub replay_ref: u64,
}
```

The transition certificate records how the stated/open pattern moved:

```rust
pub struct TripletQuorumCertificate {
    pub thought_id: u64,
    pub cycle_id: u64,
    pub input_projection: TriadicProjection,
    pub output_projection: TriadicProjection,
    pub thinking_atoms: ThinkingAtomMask64,
    pub support_quorum_q8: u8,
    pub contradiction_quorum_q8: u8,
    pub uncertainty_quorum_q8: u8,
    pub witness_offset: u32,
    pub witness_len: u16,
    pub entropy_budget_remaining_q8: u8,
    pub replay_ref: u64,
}
```

## Relation to semantic CAM distance

Semantic CAM distance ranks candidate answers and historical compatibility. The triplet quorum records why the question was asked and how the projection state moved.

```text
semantic CAM distance = candidate compatibility metric
CausalEdge64 quorum   = transition truth/provenance story
Markov triplet        = inherited projection-state machine
```

## Implementation sequence

1. Add `ThinkingAtom16` and `ThinkingAtomMask64`.
2. Add `MarkovTripletState`.
3. Add simple `TripletTransform` trait.
4. Add `TripletQuorumCertificate`.
5. Attach triplet states and certificates to `ThoughtCycleSoA` by `thought_id` and `cycle_id`.

## Review wording

Use this:

```text
4096x4096 is a deterministic query surface: two known COCA factors plus a 2^3 projection mask ask the missing third factor. CausalEdge64 compresses the projection story as NARS truth, projection kind, temporal/source lane, and thinking-style angle. A ThinkingAtomMask64 preserves which reasoning atoms participated. The Markov chain can then evolve over triplet projection states, where each transition records which factors were stated, which were queried, and which witness quorum supported the move.
```
