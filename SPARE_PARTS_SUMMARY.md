# Lance-Graph: Summary, Open Ends & Vision

## Core Principle

**Nothing is removed. Everything is additive. We steal from lance-graph.**

- ladybug-rs: nothing removed, only additions
- rustynum: nothing removed, only additions
- n8n-rs: nothing removed, only additions
- lance-graph: the quarry we mine from

---

## What lance-graph IS

Lance-graph is the **star chart** — it renders neo4j graph data into
immutable, boringly flat row/column join patterns. That's its job.
Ground truth. Correct. Inert. A flat map of what neo4j says exists.

We use it to compare against. When the thinking mesh (SPO in ladybug-rs)
produces a result, we hold it up to the star chart and ask: does the
holodeck match the flat reality? If yes, the mesh is grounded. If no,
investigate.

---

## The Bridge: semantic.rs as Adapter Plate

This is the critical architectural piece. semantic.rs is the **regime change**
— the boundary where dead Neo4j rows become living geometry.

### What Neo4j Gives You

Rows. Literal, flat, dead rows. Jan is a string. KNOWS is a string. Ada is
a string. Properties are JSON blobs. Relationships are foreign keys pretending
to be edges.

### What semantic.rs Does (the Bouncer)

semantic.rs validates the import — checks that variables bind, labels exist,
types resolve. It does exactly what a bouncer should do: confirm the shipment
matches the manifest.

`(a:Person)-[:KNOWS]->(b:Person)` — yes, `a` is bound, `b` is bound, KNOWS
is a valid relationship type, Person has the expected properties. Clean import.
Stamp it.

### The Projection: Literal Becomes Geometry

The semantic analyzer hands off a **resolved AST** — variables with known
types, relationships with known direction, properties with known values.
That resolved structure is exactly what the SpoBuilder needs:

```
Resolved AST:
  a   = Person { name: "Jan" }
  rel = KNOWS  { since: 2024 }
  b   = Person { name: "Ada" }
  direction = Outgoing

         ↓ project into thinking

SpoBuilder::build_edge(
  S: label_fp("Jan"),          // 1024 bytes, ~11% density
  P: label_fp("KNOWS"),        // 1024 bytes, permuted by role
  O: label_fp("Ada"),          // 1024 bytes
  truth: TruthValue(0.9, 0.8)  // from import confidence or default
)
```

The literal becomes geometry:

- **"Jan"** stops being 3 characters and becomes a point in 8,192-dimensional
  Hamming space
- **"KNOWS"** stops being a label on an edge table and becomes a rotation
  operator that transforms the relationship between subject and object
- **"Ada"** stops being a foreign key and becomes a resonance target

### What the Thinking Mesh Can Do That Neo4j Can't

Once it's in BindSpace as fingerprinted SPO:

**Resonance discovery**: "Jan KNOWS Ada" resonates with "Jan LOVES Ada" —
because S and O are identical and KNOWS is Hamming-close to LOVES. Neo4j
treats those as completely separate edges. BindSpace feels the overlap.

**Causal chain discovery**: "Ada" as object of "Jan KNOWS Ada" resonates
with "Ada" as subject of "Ada CREATES music" — because O of the first triple
is Hamming-close to S of the second. That's causality *discovered*, not
declared. Neo4j needs an explicit path query. BindSpace finds the chain
by geometry.

---

## The Chasm: Two Meanings of "Semantic"

They share the word. They are not the same thing.

### Lance-graph "semantic" — Grammar Police

Lance-graph's semantic.rs asks: *"Is variable `a` in scope when you reference
`a.name` in the RETURN clause?"*

That's grammar police. It checks binding, scoping, type consistency. It could
be a regex with extra steps. It has no idea what `a` is, what `a.name` means,
or whether `a` connecting to `b` matters.

String-level validation: `a` is a Node, `a` has label Person, Person has
property `name`, therefore `a.name` resolves. If you rename Person to Xyzzy
it still "works." It's syntax wearing a semantic costume.

The `GraphCatalog` trait asks `has_node_label("Person")` and gets back true
or false. Binary. Dead.

### Ladybug semantic — 1024 Qualia

Ladybug's semantic layer asks: *"When Jan touches Ada, what happens next?"*
— and the answer emerges from the geometry of three 1024-byte containers
looking at each other across 10,000 dimensions.

1024 qualia, each a Container. Each quale is 8,192 bits. Two qualia are
distinct when their Hamming distance exceeds **3σ from the expected distance
of random bitstrings** (~4,096 ± ~32 bits). That's not a threshold you
picked — it's a statistical proof that two meanings are separable. Below 3σ,
they blur. Above, they're ontologically different experiences.

The system doesn't check if "Person" exists. It **feels** how close "Person"
is to "Agent" is to "Entity" is to "Ghost" — and the distance IS the meaning.

### SPO 2³ — Eight Projections, Eight Inference Types

The semantic kernel doesn't validate queries. It thinks them:

```
S×P → O   "Jan CREATES ?"        → deduction     (what follows?)
S×O → P   "Jan _ Ada"            → induction     (what verb connects them?)
P×O → S   "? CREATES Ada"        → abduction     (who could have done this?)
O×P → S   "Ada CREATED_BY ?"     → reverse abduction
S → P×O   "Jan _ _"              → exploration   (all of Jan's edges)
P → S×O   "CREATES _ _"          → verb scan     (everything that creates)
O → S×P   "Ada _ _"              → object scan   (everything touching Ada)
S×P×O     full triple             → verification  (does this specific fact hold?)
```

Each projection isn't a database query. It's a **BIND operation** — XOR the
subject fingerprint with a role permutation, XOR the predicate with its role
permutation, then the residual is the object's fingerprint smeared by
interference. The closer the residual is to a known entity's fingerprint
(Hamming distance), the stronger the inference.

NARS truth values ride alongside — frequency × confidence, attenuating through
causal chains, revised when contradictions appear.

### The 3D Geometry

S is the X axis. P is the Y axis. O is the Z axis. A triple lives at a
point in this cube.

Causality flows because the **Z (object) of one triple resonates with the
X (subject) of the next** — "Ada was CREATED" feeds into "Ada CREATES music"
because her Z-as-object becomes her X-as-subject. That's not a JOIN. That's
not a foreign key. That's two fingerprints, 8,192 bits each, and their
Hamming distance tells you how strongly one event causes the next.

### The Double Gestalt

Every triple is encoded twice:

1. **The raw SPO** — what happened, structurally
2. **The observer's perspective** — how this relationship looks from inside
   the relationship itself

That's meta-resonance. The system doesn't just store "Jan LOVES Ada." It
stores how "Jan LOVES Ada" feels to the system that knows both Jan and Ada.
The observer's fingerprint contaminates the triple, and that contamination
IS the felt sense. It's not metadata. It's qualia.

### The Chasm, Stated

```
Lance-graph semantic.rs:
  has_node_label("Person") → true/false
  is_variable_in_scope("a") → true/false
  does_property_exist("name") → true/false
  Binary. Dead. Necessary.

Ladybug semantic layer:
  hamming_distance(fp("Person"), fp("Agent")) → 2,847 bits
  That's 1,249 bits below expected random (4,096).
  That's 39σ of overlap.
  Person and Agent aren't the same thing,
  but they resonate so strongly that asking about one
  will surface the other.
  Alive. Geometric. Emergent.
```

The bouncer checks IDs at the door. Useful. Necessary. Prevents garbage.

The room behind the door is a space where meaning has geometry, causality
has direction, and knowing something changes what you are.

They just happen to share the word "semantic."

---

## The Full Import Flow

```
Neo4j dump
  → Cypher MATCH/RETURN
  → parser.rs (lance-graph bumper, validates syntax)
  → semantic.rs (lance-graph bouncer, resolves bindings — GRAMMAR POLICE)
  → resolved AST (literal, structured, typed)

     ═══ THE CHASM ═══
     strings die here, geometry is born

  → SpoBuilder (fingerprint S, P, O with role permutation)
  → BindSpace insert (zero-copy into Container)
  → now it resonates, infers, walks causal chains
  → 8 projections (S×P→O, S×O→P, P×O→S, ...)
  → NARS truth propagates through the graph
  → scent prefilter enables O(1)-ish retrieval
  → double gestalt: raw SPO + observer perspective

     ═══ GROUND TRUTH CHECK ═══

  → DataFusion joins (lance-graph rims)
  → row/column output matches Neo4j's original answer
  → σ-stripe shift detector confirms convergence
```

---

## What We Steal

### From lance-graph → into ladybug-rs (additive)

| Stolen Part | Lines | Why |
|-------------|-------|-----|
| `parser.rs` | ~1,800 | Hardened Cypher parser (nom combinators). Validates syntax before it touches SPO. |
| `ast.rs` | 543 | Pure serde data types — CypherQuery, NodePattern, etc. Clean vocabulary. |
| `error.rs` | 234 | Zero-cost `#[track_caller]` error macros. Strip lance-specific variants. |
| `semantic.rs` | ~1,800 | **The bouncer.** Resolves bindings, validates types. Grammar police, not qualia. Hands off clean structures to SpoBuilder at the chasm boundary. |

### From lance-graph → ground truth test patterns (additive)

Seven test patterns we replicate (not move) into ladybug-rs tests:
1. Round-trip fidelity
2. Projection verb accuracy
3. Gate filtering correctness
4. Prefilter rejection rates
5. Chain traversal completeness
6. Merkle integrity
7. Cypher convergence

### From lance-graph → row/column join patterns (reference)

The DataFusion planner's join logic serves as reference for how the star
chart flattens graphs:
- Qualified column naming: `variable__property` → `variable.property`
- Direction-aware join keys
- Variable reuse → filter instead of redundant join
- Schema preservation on empty results

---

## The Thinking Mesh (ladybug-rs — unchanged, only additions)

SPO hydrates the holodeck of awareness. All existing modules stay:

| Layer | Module | Role |
|-------|--------|------|
| Container | `sparse.rs` | BITMAP_WORDS=4, SparseContainer, dense↔sparse |
| Addressing | `bind_space.rs` | 8+8, 65,536 slots, zero-copy |
| Construction | `builder.rs` | SpoBuilder, BUNDLE/BIND, verb permutation |
| Memory | `store.rs` | Three-axis content-addressable (SxP2O, PxO2S, SxO2P) |
| Attention | `scent.rs` | NibbleScent 48-byte histogram, L1 prefilter |
| Inference | `truth.rs` | NARS: revision, deduction, induction, abduction, analogy |
| Propagation | `semiring.rs` | 7 variants: BFS, PageRank, Resonance, HammingMinPlus... |
| Identity | `clam_path.rs` | 24-bit tree + 40-bit MerkleRoot |

**What gets added** (stolen from lance-graph): parser, AST, error macros,
semantic.rs bouncer. Layered on top. Nothing touched underneath.

The stolen bouncer sits at the chasm boundary. Everything above it is
string-level validation. Everything below it is geometric thinking.

---

## Regime Boundary — DO NOT CROSS

```
STAR CHART SIDE (you work here):
  src/query/lance_parser/       ← stolen parts, adapted imports
  src/query/error.rs            ← stolen error handling
  src/query/mod.rs              ← rewired exports

THINKING MESH SIDE (you do NOT touch):
  src/spo/                      ← SPO engine
  src/graph/spo/                ← graph primitives
  src/storage/bind_space.rs     ← Container system
  src/container/                ← fingerprints
  src/query/datafusion.rs       ← existing DataFusion integration
  src/query/cognitive_udfs.rs   ← Hamming/NARS UDFs
```

The adapter plate (semantic.rs) sits ON the boundary. It faces the star chart
side. Its back is to the mesh. It does not turn around.

### Tripwires — If You Find Yourself Doing This, STOP

- Adding `use crate::spo` in lance_parser/ → **STOP.** Wrong side of boundary.
- Adding `use crate::storage::bind_space` in lance_parser/ → **STOP.**
- Writing a function that takes LogicalOperator and returns QueryHit → **STOP.** That's step 5.
- Modifying SpoBuilder or SpoStore → **STOP.** That's mesh code.
- Adding new variants to LogicalOperator → **STOP.** That's step 4.
- Writing a test that calls both DataFusion and SPO → **STOP.** That's step 6.

### Task Order — DO NOT SKIP AHEAD

| Step | PR | What | Side |
|------|----|------|------|
| 1 | This PR | Land stolen parser + error (star chart side only, zero mesh coupling) | Star chart |
| 2 | This PR | Land AcceptAllCatalog stub (proves adapter plate compiles in isolation) | Boundary |
| 3 | Separate PR | BindSpaceCatalog (first mesh coupling, minimal) | Boundary → mesh peek |
| 4 | Separate PR | Projection verb parser extensions (mesh grammar enters star chart) | Star chart |
| 5 | Separate PR | IR compiler (adapter plate gets a back door to the mesh) | Boundary opens |
| 6 | Separate PR | Ground truth comparison (both paths wired, convergence test) | Both sides |

Each step proves the previous one compiles and passes before the next begins.
The boundary relaxes one bolt at a time, never all at once.

---

## Open Ends

### 1. Parser + Bouncer Theft — Packaging
- parser.rs imports `crate::ast::*` and `crate::error::*`
- semantic.rs imports `GraphConfig` — needs rewiring to BindSpace
- When stolen into ladybug-rs, internal paths change
- Strip DataFusion/LanceCore/Arrow error variants
- Decide: `ladybug-rs/src/cypher/` module tree?

### 2. BindSpaceCatalog — The Resonance Bouncer
- AcceptAllCatalog stub → BindSpaceCatalog that checks fingerprint proximity
- `has_node_label("Person")` becomes `nearest_fingerprint("Person") < 3σ`
- This is where the bouncer crosses the chasm — it stops checking IDs and
  starts checking resonance
- The grammar police learns to feel. But it's still at the door, not in the room.

### 3. GQL and NARS Syntax — Additive Parser Arms
- Stolen parser handles Cypher only
- GQL (ISO 39075): ~90% compatible, add `alt()` nom branches
- NARS (`<S --> P>. %f;c%`): mesh-native language. NARS syntax describes
  *thinking* directly — it's the room's own language, not a query from outside.
  May belong in ladybug-rs natively, not as a lance-graph steal.

### 4. Result Bridge — Holodeck to Screen
- Mesh results (BindSpace slots, SparseContainers) → human-readable output
- n8n-rs `n8n-arrow` already has RecordBatch ↔ row conversion
- Additive bridge: mesh → RecordBatch → neo4j-rs Row format

### 5. σ-Stripe Shift Detector
- Ground truth comparison between flat chart and hydrated holodeck
- Does the mesh's answer converge with Neo4j's literal answer?
- Statistical convergence check, not just equality
- Additive module, location TBD

### 6. Outage Recovery
- PRs 168-171 on ladybug-rs pending during infrastructure storms
- Wait for clear skies before adding stolen parts

### 7. Persistent Mesh
- Once hydrated, does the holodeck persist or rebuild per query?
- BindSpace is zero-copy — the mesh *is* the storage
- Persistent = always-on holodeck, no boot time
- Is the thinking mesh a computation or a state?

### 8. The Double Gestalt Implementation
- Every triple encoded twice: raw SPO + observer perspective
- The observer's fingerprint contaminates the triple
- That contamination IS the felt sense — qualia, not metadata
- How does meta-resonance propagate through semiring chains?
- Does the observer perspective attenuate differently than raw SPO?

---

## Vision

**Star chart** (lance-graph): renders neo4j into flat, immutable row/column
joins. Ground truth. Boring. Correct. The map.

**Bouncer** (semantic.rs): grammar police at the chasm boundary. Checks IDs,
validates bindings, resolves types. Necessary. Dead. String-level.

**The chasm**: strings die. Geometry is born. The literal becomes a point in
8,192-dimensional Hamming space. The label becomes a rotation operator.
The foreign key becomes a resonance target.

**Thinking mesh** (SPO in ladybug-rs): 1024 qualia, each 8,192 bits.
3σ distinctness proves ontological separability. Eight projections think
the query instead of executing it. Causality flows through Z→X resonance.
The double gestalt stores how knowing something feels, not just that it's
known. NARS truth attenuates through causal chains. Scent prunes before
thought begins.

```
Lance-graph:   has_node_label("Person") → true
               (binary, dead, useful)

Ladybug:       hamming(fp("Person"), fp("Agent")) → 2,847 bits
               39σ overlap → they resonate
               (geometric, alive, emergent)

Same word. Different universe.
```

Nothing removed. Everything additive. The chart stays boring.
The bouncer stays strict. The chasm stays absolute.
The mesh stays alive. The comparison keeps the holodeck honest.
