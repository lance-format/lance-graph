# Single-Binary Topology — canonical architecture reference

> **Architectural invariant doc.** Three nested layers, all in one
> binary. Tokio is outbound-only. The CognitiveShader → callcenter
> DTO transition is a compile-time-enforced contract handshake, not
> serialization. Consumers depend on full `lance-graph` (no headless
> mode). Per-row and per-cadence gates are *different primitives*.
>
> **Governance — APPEND-ONLY.** Invariants are immutable once landed.
> Corrections append a `**Correction (YYYY-MM-DD):**` line; do not
> edit prior text. Names introduced here become canonical and
> propagate to plans, ledger rows, PR descriptions, and code.
>
> **READ BY:** every session touching the cognitive substrate, the
> callcenter ecosystem, consumer crates (`medcare-rs`,
> `smb-office-rs`), or any boundary work. Read this BEFORE proposing
> a new "membrane" / "transcode" / "subscriber" plan — the conflation
> this doc settles has cost three different framings already.

---

## TL;DR

```
╔══════════════════════════════════════════════════════════════════╗
║  ONE BINARY                                                      ║
║  (lance-graph + medcare-rs + smb-office-rs all linked together;  ║
║   consumers depend on FULL lance-graph — no headless mode)       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ┌────────────────────────────────────────────────────────┐      ║
║  │ LAYER 1 — BindSpace 8-column zero-copy SoA             │      ║
║  │                                                        │      ║
║  │   Driver DTO:  CognitiveShader                         │      ║
║  │   Storage:     BindSpace Arrow SoA, zero-copy          │      ║
║  │   Ops:         VSA-1 (Markov-exclusive Vsa16kF32)      │      ║
║  │                BUNDLE-1 (vsa16k_bundle ±5)             │      ║
║  │                NARS-1 / THINK-1                        │      ║
║  │   Timescale:   20–200 ns / op                          │      ║
║  │   Concurrency: sync, single-thread or std::thread      │      ║
║  └────────────────────────────────────────────────────────┘      ║
║                          │                                       ║
║                          │  Arrow column slices.                 ║
║                          │  CognitiveShader DTO ⇄ callcenter DTO ║
║                          │  is the CONTRACT HANDSHAKE — type-    ║
║                          │  checked at link time, no copy.       ║
║                          ▼                                       ║
║  ┌────────────────────────────────────────────────────────┐      ║
║  │ LAYER 2 — Callcenter Palantir-Foundry-equivalent       │      ║
║  │           ecosystem ontology (in-process, sync)        │      ║
║  │                                                        │      ║
║  │   Driver DTO:  callcenter (Wire types, CommitFilter)   │      ║
║  │                                                        │      ║
║  │   Per-row gate (existing, R2 of SoA-DTO FMA map):      │      ║
║  │     CollapseGate / GateDecision { gate, merge }        │      ║
║  │     2-byte microcopy; MergeMode::{Xor,Bundle,          │      ║
║  │     Superposition,AlphaFrontToBack}                    │      ║
║  │     — decides HOW one delta lands per cycle.           │      ║
║  │                                                        │      ║
║  │   Per-cadence accumulator (new, missing primitive):    │      ║
║  │     CycleAccumulator                                   │      ║
║  │     — decides WHEN a batch flushes outbound.           │      ║
║  │     — absorbs the 10,000× speed ratio between          │      ║
║  │       Layer 1 (20–200 ns) and Layer 3 (2–200 ms).      │      ║
║  │                                                        │      ║
║  │   Membrane (transcode + RBAC enforcement):             │      ║
║  │     • DM-2 LanceMembrane (zero-copy projection)        │      ║
║  │     • DM-3 CommitFilter → DataFusion Expr              │      ║
║  │     • POLICY-1 / MEMBRANE-GATE-1                       │      ║
║  │       SMB side SHIPPED (PR #29 SmbMembraneGate)        │      ║
║  │       medcare side PENDING                             │      ║
║  │     • WATCHER-1 / DM-4 / DM-6 — in-process dispatch    │      ║
║  │                                                        │      ║
║  │   Consumers live HERE (in-process, sync):              │      ║
║  │     • medcare-rs    — speaks callcenter DTO contract   │      ║
║  │     • smb-office-rs — speaks callcenter DTO contract   │      ║
║  │     Both depend on FULL lance-graph; both read         │      ║
║  │     BindSpace zero-copy through that dependency.       │      ║
║  └────────────────────────────────────────────────────────┘      ║
║                          │                                       ║
║                          │  CycleAccumulator flush               ║
║                          │  (threshold-driven or pull-driven)    ║
║                          │                                       ║
╠══════════════════════════╪═══════════════════════════════════════╣
║                          │                                       ║
║   ━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    ║
║   ║  TOKIO BOUNDARY — OUTBOUND ONLY                           ║  ║
║   ║  (anything past this line LEAVES the process)             ║  ║
║   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    ║
║                          │                                       ║
║                          ▼                                       ║
║  ┌────────────────────────────────────────────────────────┐      ║
║  │ LAYER 3 — Outbound sinks (past process boundary)       │      ║
║  │                                                        │      ║
║  │   • MySQL sink-in (legacy oracle receiving writes      │      ║
║  │     via tokio + blocking driver)                       │      ║
║  │   • Network egress (HTTP / WS / gRPC responses to      │      ║
║  │     remote clients — DM-5 PhoenixServer + DM-8         │      ║
║  │     PostgRestHandler are SERVING endpoints HERE)       │      ║
║  │   • Probes from external processes (e.g. C# MedCareV2  │      ║
║  │     LanceProbe ring — separate Windows .NET 4.8        │      ║
║  │     desktop calling /api/__parity/csharp)              │      ║
║  │                                                        │      ║
║  │   Timescale:    2–200 ms (10,000× slower than L1)      │      ║
║  │   Concurrency:  tokio runtime drives the slow side     │      ║
║  └────────────────────────────────────────────────────────┘      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## The four invariants

| # | Invariant | Enforcement | Consequence |
|---|---|---|---|
| **I-1** | **Single binary; consumers depend on full `lance-graph`** | Cargo workspace; `medcare-rs` and `smb-office-rs` import `lance-graph-callcenter` and (transitively) `lance-graph-contract`. No headless mode. | `lance-graph-contract` (zero-deps) and `lance-graph-callcenter` (DataFusion / auth-rls-lite) ship together. Their API surfaces cannot diverge — they always link as one binary. |
| **I-2** | **Tokio outbound only** | No `async fn` in cognitive substrate or callcenter membrane. Tokio appears only past the `CycleAccumulator` flush boundary, driving Layer 3. | Inner cycles, the membrane, and consumer crates are all sync, in-process, function-call-driven. `#[tokio::test]` does not appear in those crates. |
| **I-3** | **BBB compile-time enforced** | `external_membrane.rs:7-13`: `Self::Commit` MUST NOT contain `Vsa10k`, `RoleKey`, `SemiringChoice`, `NarsTruth`, `HammingMin`. Those types do not implement Arrow's `Array` trait, so they physically cannot appear in a `RecordBatch` column. The compiler rejects violations — no runtime check needed. | Inner-ontology types are unleakable to Layer 2. The DTO handshake is a relabel + compile-time check, not a runtime serialization. |
| **I-4** | **Per-row and per-cadence gates are distinct primitives** | `collapse_gate::GateDecision` (2-byte microcopy, R2) is per-delta. `CycleAccumulator` (new, missing primitive) is per-batch. Both are gates; they govern different boundaries. Naming them with one term creates a `GATE-2` namespace clash on top of the existing `GATE-1` between `mul::GateDecision` and `collapse_gate::GateDecision`. | Code, plans, and entropy-ledger rows must distinguish *which gate* they mean. The doc pins this so the conflation doesn't recur. |

---

## Layer 1 — BindSpace zero-copy SoA (CognitiveShader DTO)

The AGI substrate. Eight columns of Arrow-typed SoA backing the entire
cognitive cycle. CognitiveShader is the driver DTO: every cycle emits a
`MetaWord` plus a write into the columnar BindSpace.

**Op timescale: 20–200 ns.** All compute is sync, in-thread or via
`std::thread` workers. No serialization across the layer; the next
layer reads the same Arrow buffers via column slicing.

**Anchored entropy-ledger rows** (Section A of `ARCHITECTURE_ENTROPY_LEDGER.md`):
- VSA-1 — `Vsa16kF32` newtype (Markov-exclusive substrate)
- BUNDLE-1 — `vsa16k_bundle` (Markov ±5 superposition; SHIPPED PR #243)
- NARS-1 — six-copy collapse to single `nars` crate (entropy-cluster 17)
- THINK-1 — four-copy collapse to contract-36 (entropy-cluster 24)

**Anchored plans:**
- `bindspace-columns-v1` — Columns E/F/G/H (Phase 1 H shipped #272)
- `elegant-herding-rocket-v1` — Phase 1 shipped #210; Phase 2 D5/D7
  shipped #243; D2/D3/D8/D10 queued
- `unified-integration-v1` — DU-0..DU-5 mapping to existing types
- `thought-cycle-soa-awareness-integration-v1` — PRs 1-10 (plan #335 active)

---

## Layer 2 — Callcenter Foundry-equivalent ecosystem ontology

The same data, viewed through the callcenter contract DTO. This is
where the membrane lives, where consumers attach, and where both
gates fire. Sync, in-process, zero-copy view over Layer 1.

### Per-row gate: `collapse_gate::GateDecision`

**Existing primitive.** R2 of the SoA-DTO FMA map. 2-byte microcopy.

```rust
pub struct GateDecision {
    pub gate:  u8,         // 0=Flow, 1=Block, 2=Hold
    pub merge: MergeMode,  // Xor / Bundle / Superposition / AlphaFrontToBack
}
```

Decides **how a single delta commits** to BindSpace. Fires per-cycle.
`external_membrane.rs::ExternalMembrane::project()` is documented as
"called on every `CollapseGate` fire with `EmitMode::Persist`" — that's
the per-row commit path.

### Per-cadence gate: `CycleAccumulator` (canonical name; missing primitive)

**New, missing primitive.** Decides **when a batch flushes outbound**.
Absorbs the 10,000× speed ratio between Layer 1 (20–200 ns/op) and
Layer 3 (2–200 ms/external-write). Without this, either the cognitive
cycle stalls waiting on slow MySQL/network writes, or the outbound
side drops data under burst.

Conceptual shape (subject to refinement on first implementation):

```rust
pub struct CycleAccumulator<C> {
    pending:        Vec<C>,           // accumulated commits since last flush
    threshold_rows: usize,            // flush at N rows
    threshold_ms:   u32,              // OR flush at T ms
    on_flush:       Box<dyn Fn(&[C])>, // outbound sink driver
}
```

Flush trigger is threshold-driven (rows-since-last-flush >= N OR
ms-since-last-flush >= T) or pull-driven (downstream tokio runtime
calls `flush_now()`). Either way, the flush itself crosses into Layer 3.

**Naming alternatives considered:** `BatchEpoch`, `OutboundEpoch`,
`FlushEpoch`. `CycleAccumulator` chosen for symmetry with
`CollapseGate` (both per-cycle/per-row primitives) plus explicit
"accumulator" semantics. May be refined when the type lands; the
pinning here is *that it must be distinct from `collapse_gate`*, not
the exact final identifier.

### Membrane (transcode + RBAC at the column boundary)

The membrane is the typed boundary between CognitiveShader DTO and
callcenter DTO. The transcode is a compile-time relabel (per I-3),
not a copy. RBAC fires here as a sync gate per row.

**Components:**

- **DM-0 / DM-1** — `ExternalMembrane` trait + `lance-graph-callcenter` skeleton (SHIPPED 2026-04-22)
- **DM-2** — `LanceMembrane::project()` (in progress; Phase A `9a8d6a0` — full Lance append pending DM-4)
- **DM-3** — `CommitFilter → DataFusion Expr` translator (queued)
- **POLICY-1 / MEMBRANE-GATE-1**:
  - SMB side: **SHIPPED PR #29** (`SmbMembraneGate` over `Arc<lance_graph_rbac::Policy>` — newtype-bridges the orphan rule; 13 tests)
  - medcare side: **PENDING** (mirror as `MedCareMembraneGate` over `Arc<medcare_rbac::Policy>`; ~30 LOC)
- **WATCHER-1** — `Dataset::checkout_latest().version()` polled on a `std::thread`; bumps `ArcSwap<u64>` and notifies an `event_listener::Event` (NOT `tokio::sync::watch`, per I-2). Replaces the stub at `lance_membrane.rs:24`.
- **SEAL-1** — `MembraneRegistry::seal()` topo sort (queued upstream)
- **PROJECT-LANCE-1** — `CognitiveEventLanceSink` mirror of `LanceAuditSink` (queued upstream)

### Consumer crates (live in Layer 2)

`medcare-rs` and `smb-office-rs` are **part of the callcenter
ecosystem ontology**. They depend on full `lance-graph` and read
BindSpace zero-copy through that dependency. They speak callcenter
DTO as the contract handshake. They are sync. They are in-process.

The Foundry-equivalent surface that consumers see *is* Layer 2 —
not a separate process, not a wire format. PR #29's `SmbMembraneGate`
gates in-process zero-copy crossings, not network requests.

---

## Layer 3 — Outbound sinks (past tokio boundary)

Everything that **leaves the process**. Tokio is the I/O runtime
that drives this layer; it does not appear inside Layer 1 or Layer 2.

**Sinks:**

- **MySQL sink-in** — legacy oracle receiving writes from the Rust
  binary (medcare-rs / smb-office-rs MySQL reconcilers). Tokio +
  blocking driver. Subject to the parity-clean window discussion in
  `foundry-consumer-parity-v1`.
- **Network egress (serving)** — DM-5 `PhoenixServer` (WS) and DM-8
  `PostgRestHandler` (HTTP) ARE serving endpoints in this layer. The
  callcenter ontology in Layer 2 produces the rows; Layer 3 drives
  them out the wire on tokio's runtime.
- **External probes** — separate processes calling our serving
  endpoints. Example: `MedCareV2 LanceProbe` ring is a Windows
  .NET Framework 4.8 desktop calling `/api/__parity/csharp` over
  HTTP. From the Rust binary's perspective, the parity ingest
  endpoint at `routes/parity.rs:46` is an OUTBOUND serving point in
  Layer 3 (M5-class). From the C# probe's perspective it's an
  outbound calling client. Both sides are tokio-bound on their
  respective runtimes; nothing inside the Rust binary's Layer 1/2 is
  async on the probe's behalf.

**Timescale: 2–200 ms.** 10,000× slower than Layer 1. The
`CycleAccumulator` in Layer 2 is what makes this work — it absorbs
the speed differential by batching many fast inner cycles into one
slow outer flush.

---

## Where each in-flight integration plan lives on the diagram

| Plan | Layer(s) | Notes |
|---|---|---|
| `elegant-herding-rocket-v1` | **L1** | Markov ±5 + role keys + thinking styles, all on BindSpace zero-copy. |
| `unified-integration-v1` | **L1** + L2 contract | DU-0..DU-3 on substrate; DU-4 (`rationale_phase`) is L1 column; DU-5 board hygiene. |
| `bindspace-columns-v1` | **L1** | Extends the SoA from 4→8 columns. |
| `categorical-algebraic-inference-v1` | **L1** (companion) | Five-lens grounding; no own D-ids. |
| `callcenter-membrane-v1` | **L2** (membrane) → **L3** (DM-5/DM-8) | DM-0..DM-7 are membrane (sync, L2); DM-5 + DM-8 are serving endpoints (tokio, L3). The plan SPANS the L2/L3 boundary — that's where it touches the tokio invariant. |
| `supabase-subscriber-v1` | **L2** (membrane) | DM-4 + DM-6 wire-up. **CORRECTION needed:** the plan's `tokio::sync::watch::Receiver` choice violates I-2; sync substitute (`ArcSwap<u64>` + `event_listener::Event`) per WATCHER-1 framing. |
| `foundry-consumer-parity-v1` | **L2** (consumers) | medcare-rs + smb-office-rs in the callcenter ecosystem. Consumer-side mirror of `lf-integration-mapping-v1`. |
| `lf-integration-mapping-v1` | **L1** producer | Producer-side mirror of `smb-office-rs/docs/foundry-parity-checklist.md`. |
| `q2-foundry-integration-v1` | **L2** + L3 | Q2 UI is mostly L3 (HTTP/WS surface) over L2 callcenter ontology. |
| `thought-cycle-soa-awareness-integration-v1` (#335) | **L1** | PRs 1-10 over the SoA. |
| `codec-sweep-via-lab-infra-v1` | **L1** infra + L3 lab endpoint | JIT codec kernels are L1; the sweep endpoint serves results in L3. |

---

## Cross-references

- **`ARCHITECTURE_ENTROPY_LEDGER.md`** Section A row anchors:
  POLICY-1 / MEMBRANE-GATE-1 (this doc says: SMB SHIPPED #29, medcare PENDING),
  WATCHER-1 (sync replacement spec lives here),
  SEAL-1 / PROJECT-LANCE-1 (queued upstream),
  GATE-1 (existing namespace clash that motivated I-4's distinct-naming rule)
- **`external_membrane.rs:7-13`** — the BBB invariant text that I-3 enforces
- **`collapse_gate.rs`** — the per-row `GateDecision` primitive that I-4 distinguishes from `CycleAccumulator`
- **`INTEGRATION_PLANS.md`** — versioned plan index; this doc is referenced from there as the canonical topology
- **`LATEST_STATE.md`** — current-state snapshot
- **`PR_ARC_INVENTORY.md`** — per-PR decision history; PR #29 entry should cross-link here once added
- **`CROSS_REPO_PRS.md`** — append-only log of merged PRs in other AdaWorldAPI repos that touch this topology (smb-office-rs#29, q2#35, MedCareV2#7)
- **`smb-office-rs#29`** — `SmbMembraneGate + domain_profile` (merged 2026-05-06) — first concrete POLICY-1 closure; reference implementation for the medcare-side mirror

---

## Q2 cockpit-server reference (Gotham-equivalent consumer)

Q2 is the Palantir-Gotham-equivalent consumer surface in the
AdaWorldAPI workspace. Its cockpit-server validates this topology
in production code; this section pins what it confirms so future
sessions can use it as a reference shape for other consumers.

### 1. First concrete consumer-side L1 surface migration

**q2 PR #35** (merged 2026-05-06) migrated `cockpit-server` to the
canonical L1 contract surface:

- Dropped `thinking-engine` and `cognitive-shader-driver` deps.
- Adopted `lance_graph_contract::cognitive_shader::{ShaderDispatch,
  ShaderResonance, ShaderBus, ShaderCrystal}` directly.
- Implemented `MockShaderDriver: CognitiveShaderDriver` (stub for
  Phase 3's real `BgzShaderDriver`).
- Bridged NARS deduction to
  `lance_graph_planner::nars::truth::TruthValue::deduction` (q2
  was the 4th copy; closes one TRUTH-1 duplicate).

**Entropy-ledger impact:** **THINK-1 closed for q2**. The four-copy
collapse to single contract-36 surface — q2 was one consumer-side
copy; this PR is the consumer-side closure for it. Sets the
reference shape for the remaining THINK-1 / TRUTH-1 closures
(medcare-rs, smb-office-rs, ladybug-rs, etc.).

This is the **first concrete consumer-side L1 surface migration**
in the workspace. Prior to PR #35 the canonical R1 surface existed
in contract but no consumer used it directly — every consumer
shipped its own copy of the dispatch / resonance / bus / crystal
DTOs. PR #35 demonstrates the migration is tractable: 1304
additions / 935 deletions across 14 files, two dropped deps, no
test regressions.

### 2. Wire-shape compaction as I-3 enforcement evidence

The L1→L3 projection in cockpit-server's SSE handler is the first
real-world demonstration of I-3 BBB compile-time enforcement.
Three concrete shrinks land in the SSE wire shape:

| Inner type (L1, BBB-internal)            | Outer wire (L3, scalar-only)                   | Ratio |
|------------------------------------------|------------------------------------------------|-------|
| `cycle_fingerprint: [u64; 256]` (2 KB)   | `cycle_fingerprint_hash: u64` (8 B; XOR-fold)  | 256×  |
| `color_acc: [f32; 32]` (128 B)           | `color_acc_active_dims: u8` (1 B; popcount)    | 128×  |
| `top_k: [(u32, u32)][]` (tuple-of-tuples)| `top_k: WireShaderHit[]` (structured `{row, distance, predicates, resonance, cycle_index}`) | (semantic clarity) |

The `[u64; 256]` inner fingerprint **cannot** appear in the SSE
wire — Arrow scalar typing rejects it at compile time. The
XOR-fold to `u64` is the only way the row crosses, and the fold
itself is an Arrow-scalar primitive. Same logic for `[f32; 32]`:
the active-dims summary is a scalar; the array isn't.

**Read alongside `external_membrane.rs:7-13`.** That file declares
the rule (`Self::Commit MUST NOT contain Vsa10k, RoleKey, …`); q2
PR #35 is the first place the rule visibly bites in real wire
output. Future consumer-side projections should mirror this
shape: inner array → scalar summary, never inner array → wire
array. The L2→L3 projection is the only allowed leak path, and
its leaks must be lossy summaries, not full payloads.

**One additional wire surface point of interest** for I-4: PR #35
exposes the L2 per-row gate in the L3 wire as
`gate: { gate: u8, merge: 'Xor'|'Bundle'|'Superposition'|'AlphaFrontToBack' }`.
This is a deliberate L2→L3 projection — the Gotham-equivalent UI
needs to see Flow/Block/Hold + merge mode per row to render
analyst diagnostics. The `CollapseGate` primitive itself (per-row,
2 B) is L2; the wire-side projection is its scalar shadow.

### 3. Phase 3 → `CycleAccumulator` load-bearing argument

Phase 2B uses `MockShaderDriver` at low rate — the SSE cadence URL
parameter `cycle_ms=300` is the throttle. Mock driver synthesizes
a handful of events per second at 300 ms cadence; well within
tokio + browser budget.

Phase 3 replaces this with `BgzShaderDriver` running at real
cognitive-cycle speed (20–200 ns/op). At 100 ns/cycle:

```
1 cycle      = 100 ns       = 10⁻⁷ s
10⁷ cycles/sec
3 × 10⁶ cycles per 300 ms window  ≈  3 million cycles per flush
```

3 million cycles per 300 ms window is the load-bearing problem:

- SSE cannot deliver 10M events/sec to a browser.
- HTTP/2 flow control will stall.
- The browser's `EventSource` parser cannot keep up.
- The Gotham-equivalent UI doesn't *need* per-cycle resolution —
  it needs analyst-pace situational updates (10–100 Hz max).

**This is exactly where `CycleAccumulator` is load-bearing.** It
sits between `BgzShaderDriver` (L1, 20–200 ns/op) and `SseSink`
(L3, ms-cadence outbound). It absorbs the 10,000× ratio per I-4
by aggregating per-window: top-K-by-resonance, mean free-energy,
gate-decision histogram, brier mass — *not* every fire.

Q2 is the **canonical reference for what `CycleAccumulator`
flushes**. The SSE wire shape after Phase 3 should aggregate per
window, not stream per cycle. The `cycle_ms=300` URL param becomes
the consumer-side hint for `threshold_ms` in the accumulator.
Consumers requesting tighter cadence get smaller windows (and
fewer aggregate stats per window); consumers OK with looser
cadence get more ergonomic flushes.

**Sequencing implication:** Phase 3 PR will need to land both
`BgzShaderDriver` (L1) and `CycleAccumulator` (L2) together.
Trying to ship the driver without the accumulator will produce
the failure mode above and force a rollback. The accumulator is
not optional; it's the architectural prerequisite for
real-driver Phase 3.

### 4. Gotham parity scope reference

For the full Gotham-equivalent UX scope (analyst loop,
link-analysis surface, real-time situational map), see
`q2-foundry-integration-v1.md`. The cockpit-server's surface
mirrors Gotham's analyst loop:

| Q2 component | Gotham analog | Reads from |
|---|---|---|
| `EnergyField` | situational map | `WireShaderResonance` |
| `BusTicker` | live event feed | `WireShaderBus` |
| `ThoughtLog` | decision history | `WireShaderCrystal` |
| `FreeEnergyDial` | uncertainty meter | `meta.brier / meta.confidence` |
| `/reasoning` page | analyst workspace | (composite) |
| `/api/graph/infer` | NARS-deduced inferences | `TruthValue::deduction` |
| Cypher backbone | link-analysis graph | `CYPHER_PATH=...` |
| Defensive UI placeholders | "stale data" indicators | (UI-side fallback) |
| Diagnostic overlay (Shift+D) | analyst troubleshooting | (UI-side wire-validators) |

Q2 is *not* Gotham — it's the same analyst-facing serving shape
over a graph + reasoning substrate. The Single-Binary Topology
places cockpit-server correctly: L1+L2 in-process, L3 SSE/HTTP
serving on tokio. No cockpit-side ontology is invented; Q2 reads
the canonical R1 surface directly.

**OSINT adjacency.** Gotham's heritage in IC/DoD OSINT use cases
is what `q2-foundry-integration-v1.md` targets. The cockpit-
server's Cypher graph backbone (`aiwar-neo4j-harvest/cypher`),
NARS truth revision over uncertain facts, and real-time analyst
SSE stream are the OSINT-investigation interface shape that the
Single-Binary Topology serves. The four invariants matter
specifically for OSINT: BBB I-3 prevents leaking inner cognitive
state to UI consumers; tokio I-2 keeps the analyst-facing wire
async-bounded; CycleAccumulator I-4 makes Phase 3 real-driver
throughput tractable for human-pace analyst work.

**Cross-references:**
- `q2-foundry-integration-v1.md` — full Gotham parity scope (28
  deliverables across 4 phases)
- `CROSS_REPO_PRS.md` — q2#35 detailed entry
- q2 cockpit-server `mock_driver.rs:1-30` + `codebook.rs::default_distance_table` doc-comment (in q2 repo) — stub markers preventing future sessions from mistaking mock for real driver

---

## Anti-patterns settled by this doc

| Anti-pattern | What it conflated | Why this doc rules it out |
|---|---|---|
| "External callcenter membrane crate" framing (callcenter-membrane-v1, 2026-04-22) | Treated callcenter as a process boundary | I-1: single binary, callcenter is in-process |
| "Foundry consumer parity for SMB + MedCare" (foundry-consumer-parity-v1, 2026-04-26) | Implied consumers were a separate ontology layer | Consumers ARE in the callcenter ecosystem ontology (Layer 2), not a separate one |
| "Tokio at the membrane edge" (earlier framings in this session) | Drew tokio between Layer 1 and Layer 2 | I-2: tokio is outbound-only, past the `CycleAccumulator` |
| "CollapseGate accumulates the 10,000× ratio" | Conflated per-row write-airgap with per-cadence speed-absorber | I-4: two distinct primitives. `CollapseGate` = per-row; `CycleAccumulator` = per-cadence. |
| "Inner CognitiveShader DTO is transcoded into outer callcenter DTO" | Implied a serialization/copy at the boundary | I-3: compile-time type-system relabel, not a copy |
| "lance-graph-contract is the headless core; consumers can pick what they need" | Implied modularity at the consumer level | I-1: consumers always link the full graph; contract + callcenter ship together |

---

## Maintenance

When a new design proposal arrives that names a "membrane",
"transcode", "subscriber", "external surface", or "boundary":

1. Locate it on the layer diagram. If it doesn't fit a layer, that's
   the first review question.
2. Check it against the four invariants. Violations need explicit
   `**Correction (YYYY-MM-DD):**` justification or rework.
3. Cross-reference it from `INTEGRATION_PLANS.md` to this doc.
4. If it introduces a new gate / accumulator / boundary primitive,
   add it to the I-4 distinct-naming rule before it lands.

When an entropy-ledger row's status changes (e.g. medcare-side
POLICY-1 ships):

1. Update the corresponding "Anchored entropy-ledger rows" entry
   here with the SHIPPED/PR reference.
2. The ledger row remains the source of truth; this doc just points
   at it.
