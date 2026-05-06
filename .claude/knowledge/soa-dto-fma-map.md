# SoA → DTO FMA Map

> **READ BY:** any agent touching cognitive cycle dispatch, BindSpace
> column extension, contract DTO additions, OrchestrationBridge
> routing, ExternalMembrane wiring, callcenter projection, or
> codec-research lab surface.
>
> **Status:** FINDING — built 2026-05-05 from primary source reads
> (`bindspace.rs`, `cognitive_shader.rs`, `orchestration.rs`,
> `collapse_gate.rs`, `external_membrane.rs`, contract `lib.rs`,
> `lance_membrane.rs`). No agent-synthesised claims. Citations
> are file:line.
>
> **Why "FMA":** the architecture's load-bearing operation is a
> fused multiply-add on `Vsa16kF32` — `bind(role, content) +
> bundle(prior)` in one dispatch, no intermediate rounding.
> Each REGION below names its FMA: what it multiplies, what it
> adds, what flows out.

## The 8 regions, one line each

| # | Region | Crate / File | What it owns |
|---|---|---|---|
| **R0** | **BindSpace SoA** | `cognitive-shader-driver::bindspace` | The 8-column read-only address space — the substrate every cycle reads. |
| **R1** | **Per-cycle DTO family** (Φ Ψ B Γ) | `lance-graph-contract::cognitive_shader` | `ShaderDispatch` → `ShaderResonance` → `ShaderBus` → `ShaderCrystal`. The hot path. |
| **R2** | **Write airgap** | `lance-graph-contract::collapse_gate` | `GateDecision` + `MergeMode`. The 2-byte microcopy that crosses the read-only boundary. |
| **R3** | **Cross-domain orchestration** | `lance-graph-contract::orchestration` | `UnifiedStep` + `OrchestrationBridge` + `StepDomain` (7 variants). |
| **R4** | **External boundary (BBB)** | `lance-graph-contract::external_membrane` + `lance-graph-callcenter::lance_membrane` | The compile-time BBB. `ExternalMembrane`, `CommitFilter`, `MembraneGate`, `CognitiveEventRow`. |
| **R5** | **Carrier algebra** | `lance-graph-contract::crystal::fingerprint` + `ndarray::hpc::vsa` + (excluded) `holograph::bitpack` | `Vsa16kF32` ℝ multiply/add (canonical) + `Binary16K` GF(2) XOR (Hamming compare) + a third stack in excluded `holograph`. |
| **R6** | **Domain catalogues** | `contract::grammar/role_keys`, `contract::thinking`, `contract::persona`, `contract::ontology`, `contract::cam` | Identity-fingerprint registries — names → `Vsa16kF32` slices. |
| **R7** | **Storage tissue (cold)** | `lance-graph::graph::versioned`, `lance-graph-callcenter::audit`, `lance-graph-callcenter::transcode::parallelbetrieb` | Lance `Dataset` writers: `versioned.rs` (6 datasets), `LanceAuditSink`, `DriftEvent`. |
| **R8** | **Lab surface** | `cognitive-shader-driver::wire/serve/grpc/codec_research` | Lab-only Wire DTOs + REST/gRPC entry — the codec-cert + thinking-harvest port. |

## Region detail

### R0 — BindSpace SoA (the substrate)

`crates/cognitive-shader-driver/src/bindspace.rs:167-178` — `pub struct BindSpace`. **8 column families**, one row per cognitive atom:

| Column | Type | Width / row | Meaning |
|---|---|---|---|
| `fingerprints.content` | `Box<[u64]>` | 256 × u64 = 2 KB | content / topic identity (Binary16K) |
| `fingerprints.cycle` | `Box<[f32]>` | 16 384 × f32 = 64 KB | the unit of thought, **Vsa16kF32 carrier** |
| `fingerprints.topic` | `Box<[u64]>` | 256 × u64 = 2 KB | topic plane (Binary16K) |
| `fingerprints.angle` | `Box<[u64]>` | 256 × u64 = 2 KB | angle plane (Binary16K) |
| `fingerprints.sigma` | `Box<[u8]>` | 1 B | Σ-codebook index (PR #289, R²=0.9949 at k=256) |
| `edges` (`EdgeColumn`) | `Box<[u64]>` | 8 B | one `CausalEdge64` per row |
| `qualia` (`QualiaColumn`) | `Box<[f32]>` | 18 × f32 = 72 B | dims 0..16 EXPERIENCED CMYK + dim 17 OBSERVED classification distance |
| `meta` (`MetaColumn`) | `Box<[u32]>` | 4 B | packed `MetaWord`: thinking(6) + awareness(4) + nars_f(8) + nars_c(8) + free_e(6) |
| `temporal` | `Box<[u64]>` | 8 B | per-row timestamp |
| `expert` | `Box<[u16]>` | 2 B | expert id (a2a blackboard authorship) |
| `entity_type` | `Box<[u16]>` | 2 B | **Column H — Foundry Object Type** (PR #272) |

Per-row footprint: **71 777 B** (`bindspace.rs:323` test). Builder: `BindSpaceBuilder` with `push` + `push_typed(entity_type)`. Pre-filter: `BindSpace::meta_prefilter(window, &MetaFilter)` — one u32 per row, returns `Vec<u32>` of accepted row indices.

**FMA in R0:** none — R0 is pure substrate. It is *read* multiplicatively (window × filter) by R1, *written* additively (delta `+` MergeMode) by R2.

### R1 — Per-cycle DTO family (Φ Ψ B Γ)

`crates/lance-graph-contract/src/cognitive_shader.rs`. Four DTOs, one per stage:

| Stage | DTO | Contents | Bytes |
|---|---|---|---|
| **Φ Dispatch** | `ShaderDispatch` (line 178-209) | `meta_prefilter: MetaFilter`, `rows: ColumnWindow`, `layer_mask: u8`, `radius: u16`, `style: StyleSelector`, `rung: RungLevel`, `max_cycles: u16`, `entropy_floor: f32`, `emit: EmitMode`, `merge_override: Option<MergeMode>`, `alpha_saturation_override: Option<f32>` | `Copy`, ~32 B |
| **Ψ Resonance** | `ShaderResonance` (line 279-300) | `top_k: [ShaderHit; 8]`, `hit_count`, `cycles_used`, `entropy`, `std_dev`, `style_ord` | `Copy`, 16 B/hit × 8 + 16 |
| **B Bus** | `ShaderBus` (line 349-359) | `cycle_fingerprint: [u64; 256]`, `emitted_edges: [u64; 8]`, `emitted_edge_count: u8`, `gate: GateDecision`, `resonance: ShaderResonance` | ~2 KB |
| **Γ Crystal** | `ShaderCrystal` (line 379-389) | `bus`, `persisted_row: Option<u32>`, `meta: MetaSummary`, `alpha_composite: Option<AlphaComposite>` | ~2 KB + 132 B (Pillar-7) |

Driver trait: `CognitiveShaderDriver` (line 425) — `dispatch(&ShaderDispatch) -> ShaderCrystal` + `dispatch_with_sink<S: ShaderSink>`. Sinks: `ShaderSink::{on_resonance, on_bus, on_crystal}`. `NullSink` is the default.

`AlphaComposite` (line 319, **PR #324 / B5**): `color_acc: [f32; 32]`, `alpha_acc`, `hits_consumed`, `saturated`. Populated only when `merge_override == Some(MergeMode::AlphaFrontToBack)`.

**FMA in R1:**
```
ShaderResonance.top_k = bgz17_distance(BindSpace.fingerprints[rows], query_fp)         ← MULTIPLY
                       ⊕ MetaFilter mask                                                ← ADD (mask)
                       ⊕ NARS truth on edges                                            ← ADD (weight)
ShaderBus.cycle_fingerprint = vsa16k_bundle( unbind(role, hit.content_fp) for hit in top_k ) ← FMA per hit
ShaderCrystal.alpha_composite = Σ_i color_i · α_i · Π_{j<i}(1-α_j)                     ← Pillar-7 FMA, Kerbl 2023
```

The dispatch IS the FMA: bind multiplicands (role × content) and bundle addends (prior cycle, global context) in one driver call.

### R2 — Write airgap

`crates/lance-graph-contract/src/collapse_gate.rs:18-100`.

`MergeMode` (4 variants, `repr(u8)`):
- `Xor = 0` — single-writer, self-inverse, reversible.
- `Bundle = 1` — multi-writer majority vote.
- `Superposition = 2` — preserve all deltas.
- `AlphaFrontToBack = 3` — Pillar-7 Kerbl 2023 EWA splatting (PR #324). `ALPHA_SATURATION_THRESHOLD = 0.99`.

`GateDecision { gate: u8, merge: MergeMode }` — 2 bytes. Constants: `FLOW_XOR`, `FLOW_BUNDLE`, `FLOW_SUPER`, `BLOCK`, `HOLD`. Methods: `is_flow / is_block / is_hold`.

**FMA in R2:** `column[row] := merge(column[row], delta, mode)` — one fused commit per row. The cycle never holds `&mut` into the SoA; the gate decides, the merge applies.

### R3 — Cross-domain orchestration

`crates/lance-graph-contract/src/orchestration.rs`.

**`StepDomain` — 7 variants** (line 37-52): `Crew`, `Ladybug`, `N8n`, `LanceGraph`, `Ndarray`, **`Smb`**, **`Medcare`** (last two added post-PR #278/E5).

`StepDomain::profile() -> DomainProfile` (line 89-119) — STATIC defaults per domain. Medcare requires HIPAA-grade fail-closed (`requires_fail_closed=true`, `audit_retention_days=2190`, `auto_action_confidence=0.92`, `escalation=Human`); SMB is permissive (`90`, `0.75`, `Llm`, `false`); generic infra domains get `30 / 0.70 / Llm / false`.

`Escalation::{Llm, Human, Reject}`. `VerbTaxonomyId::{Generic, Smb, Medcare}` — picks the per-domain 144-cell verb table.

`UnifiedStep { step_id: String, step_type: String, status, thinking, reasoning, confidence, depends_on: Vec<StepId> }` (line 332-346). `id() -> StepId` is `fnv1a_str(step_id)` (line 356) — fixes the `id: 0` DAG-collapse landmine.

`OrchestrationBridge` trait (line 380): `route(&mut UnifiedStep)`, `resolve_thinking(ThinkingStyle, InferenceType) -> ThinkingContext`, `domain_available(StepDomain) -> bool`.

`step_trajectory_hash(domain, step, &[u64; 256])` — feature-gated `trajectory-audit` stub (line 194-201, PR #278 outlook E4); IMPLEMENTATION DEFERRED.

**FMA in R3:** `route(step) = bridge[step.domain].dispatch(thinking × step.step_type) + step.depends_on resolution`. The FMA is the bridge picking the right domain handler and threading the DAG predecessor.

### R4 — External boundary (BBB)

`crates/lance-graph-contract/src/external_membrane.rs` + `crates/lance-graph-callcenter/src/lance_membrane.rs`.

**`ExternalRole` — 8 variants** (line 40-51): inbound (User, Consumer, N8n, OpenClaw, CrewaiUser, CrewaiAgent) + outbound (Rag, Agent).

**`ExternalEventKind`** (line 60-64): `Seed` (start cycle), `Context` (XOR-bind into ±5 trajectory), `Commit` (outbound).

**`CommitFilter`** (line 75-84): scalar-only — `actor_id, max_free_energy, style_ordinal, is_commit`. **No VSA, no semiring** — compile-time BBB invariant (line 7-13: `Self::Commit MUST NOT contain Vsa10k, RoleKey, SemiringChoice, NarsTruth, HammingMin`).

**`MembraneGate`** trait (line 131-139): `should_emit(external_role, faculty_role, expert_id, gate_commit) -> bool`. `AllowAllGate` is the default. **No `impl MembraneGate for rbac::Policy` exists today** — the natural RBAC bridge is missing.

**`ExternalMembrane`** trait (line 167-194): assoc types `Commit`, `Intent`, `Subscription`. Methods: `project(&ShaderBus, MetaWord) -> Self::Commit`, `ingest(Self::Intent) -> UnifiedStep`, `subscribe(CommitFilter) -> Self::Subscription`.

`LanceMembrane` impl in `lance_membrane.rs:115-272`. `MembraneRegistry::with_rls()` + `with_audit()` + `seal()`. `seal()` returns `Ok(())` empty (line 207-213, TODO comment "real topo sort via depends_on() once N>2 plugins"). Phase-D subscription wired to `tokio::sync::watch` only (line 24, "no Lance MVCC binding"). Feature-gate coherence checks at line 45-59 (`compile_error!` if `membrane-plugins-rls` without `auth-*`).

**FMA in R4:** `project(bus, meta) = scalar_strip(bus, meta)` — strip VSA/semiring, emit Arrow scalars. `ingest(intent) = role_bind(payload, role) + UnifiedStep_translate(intent)`. The BBB is enforced at *compile time* by the Arrow type system (line 8-13), not at runtime.

### R5 — Carrier algebra (the literal FMA)

Three parallel stacks, **distinct algebras**:

1. **`lance-graph-contract::crystal::fingerprint`** — `Vsa16kF32 = Box<[f32; 16_384]>`. ℝ multiply/add (the canonical cycle carrier).
   - `vsa16k_zero`, `vsa16k_bind` (multiply), `vsa16k_bundle` (add), `vsa16k_cosine`, `binary16k_to_vsa16k_bipolar`, `vsa16k_to_binary16k_threshold`.
   - Sandwich layout (older `Vsa10kF32` form): lead `[0..3437)` / cells `[3437..6562)` / quorum `[6562..6567)` / sentinel `6567` / tail `[6568..10_000)`.
   - **No `vsa16k_permute` exists.** ρ^d Markov braiding cannot run on the f32 carrier today.
   - Free functions, not methods — Click P-1 litmus violation (CLAUDE.md "method = accept").

2. **`ndarray::hpc::vsa`** — `VsaVector` Binary16K `[u64; 256]`. **GF(2) XOR** algebra: `vsa_bind/unbind` (XOR), `vsa_bundle` (majority), `vsa_permute` (rotate), `vsa_similarity`, `vsa_hamming`, `vsa_clean`, `vsa_sequence`.
   - Same width as `Vsa16kF32` (16 384 bits), **incompatible algebra** (GF(2) vs ℝ).

3. **`holograph::bitpack::BitpackedVector`** + **`holograph::representation::*`** — XOR on bitpacked binary, third copy in **workspace-EXCLUDED** crate. No `lance-graph-contract` dep. `holograph::sentence_crystal::SemanticCrystal` collides with `contract::crystal::sentence::SentenceCrystal`.

**FMA in R5:**
```
Vsa16kF32:  out[i] = role[i] * content[i] + prior[i]                    ← FMA (literally)
Binary16K:  out[i] = role[i] ^ content[i] ^ prior[i]                     ← XOR-FMA (different algebra)
```

The user's "FMA but for our architecture" lives **literally** here on `Vsa16kF32`. Every other region's FMA is metaphorical; R5 is the real one.

### R6 — Domain catalogues (identity → slice)

Per `I-VSA-IDENTITIES`: identity fingerprints point to content. Three ID-only catalogues today:

| Catalogue | File | What it registers |
|---|---|---|
| Grammar role keys | `contract/grammar/role_keys.rs` | 37 LazyLock `RoleKey` instances (Subject, Predicate, Object, TEKAMOLO slots, Finnish cases, tenses, NARS inferences). Slices `[start:end)` over Binary16K, **disjoint**. `bind/unbind/recovery_margin` REMOVED in cleanup commit `cd5c049`. |
| Thinking styles | `contract/thinking.rs` | 36 variants × 6 clusters, `tau()` addresses (0x20/0x40/0x60/0x80/0xA0/0xC0). |
| Personas / archetypes | `contract/persona.rs` | (placeholder for PersonaHub 56-bit compression DU-0). |
| Ontology / Object Type | `contract/ontology.rs` (646 LOC) | `EntityTypeId = u16`, `Locale`, `Label`, `Ontology`, `entity_type_id(&Ontology, &str) -> EntityTypeId` resolver. **The Foundry Object Type ↔ Column H bridge.** |
| CAM-PQ subspaces | `contract/cam.rs` | `CamByte::{Heel=0, Branch=1, TwigA=2, TwigB=3, Leaf=4, Gamma=5}`. **HIP missing** vs I10 doctrine (HEEL/HIP/BRANCH/TWIG/LEAF). |

**FMA in R6:** none — R6 is **register**, not FMA. Per Test 0 (register laziness), use the catalogue *before* reaching for VSA.

### R7 — Storage tissue (cold)

| Writer | File | What it persists |
|---|---|---|
| `VersionedGraph` | `lance-graph/src/graph/versioned.rs:40,205-528` | **6 separate `*.lance` Datasets** under one base path: `nodes`, `edges`, `fingerprints`, `scopes`, `neighborhoods`, `cognitive_nodes`. `WriteMode::Append` / `Create`. |
| `LanceAuditSink` | `lance-graph-callcenter/src/audit.rs:157-389` | One `audit.lance` dataset. RLS audit only. |
| `DriftEvent` writer | `lance-graph-callcenter/src/transcode/parallelbetrieb.rs` | MySQL ↔ SPO drift events (PR #309), no chrono dep. |

**Not yet wired:** `CognitiveEventRow` from `external_intent.rs` is fanned out via `tokio::sync::watch` only (`lance_membrane.rs::project`) — **no Lance writer for projected cognitive events**. The schema-on-paper at `external_intent.rs:88-104` is unbacked.

**`lancedb` SDK (`=0.27.2`) is a phantom dep.** Zero `lancedb::` source hits; feature `lancedb-sdk` has no body. `holograph` has feature `lancedb = ["dep:lance"]` — conflates the file format with the SDK in a single feature name.

**FMA in R7:** `Dataset::write(record_batch, Append)` — fused commit + version bump on Lance MVCC. **`LanceVersionWatcher` does not yet bind to `Dataset::version()`** (`lance_membrane.rs:24`, "Phase D not done").

### R8 — Lab surface (codec-cert + thinking-harvest)

`crates/cognitive-shader-driver/src/{wire, serve, grpc, codec_research, codec_bridge, codec_kernel_cache, decode_kernel, rotation_kernel, token_agreement, auto_detect}.rs`.

Wire DTOs (`wire.rs`, behind `--features serve`):
- `WireCodecParams` + `WireLaneWidth` + `WireDistance` + `WireRotation` + `WireResidualSpec` — serde mirrors of `contract::cam::*`. `TryFrom<WireCodecParams> for CodecParams` runs precision-ladder validation BEFORE JIT compile.
- `WireTensorView` + `AlignedBytes` (64-byte-aligned, decoded once at REST ingress per Rule F).
- `WireCalibrateRequest` / `WireCalibrateResponse` (D0.1, PR #227).
- `WireTokenAgreement` + `WireTokenAgreementResult` (D0.2 in flight).
- `WireSweep` + `WireSweepGrid` + `WireSweepResult` (D0.3 in flight).

Endpoints (`serve.rs`): `POST /v1/shader/{dispatch, calibrate, probe, tensors, plan, token-agreement, sweep}`.

Bridges into the canonical surface:
- `codec_bridge::CodecResearchBridge: OrchestrationBridge` — `StepDomain::Ndarray` consumer, lab-only.
- `cypher_bridge.rs` — **regex stub** for `StepDomain::LanceGraph`. Confidence=0.5. Lossy `to_uppercase().starts_with(...)`.
- `planner_bridge.rs` — `WirePlan` shortcut → `PlannerAwareness` (already impls `OrchestrationBridge` directly; this is a lab adapter).

**FMA in R8:** `WireDispatch + JIT_kernel + Planner = real-token measurement, no relink`. The lab surface IS the codec-cert + thinking-harvest port (per `lab-vs-canonical-surface.md` I11). It is **not** the canonical consumer surface.

## Cross-region matrix — who feeds whom

| Producer (region.thing) | Consumer (region.thing) | Edge type | Notes |
|---|---|---|---|
| R8 `WireDispatch` ingress | R3 `UnifiedStep` (via codec_bridge) | translate | Wire DTO decoded once at REST edge per Rule F |
| R3 `UnifiedStep::route` | R0 `BindSpace` reads | call | bridge picks domain, dispatches |
| R0 `meta_prefilter(window, filter)` | R1 `ShaderDispatch.rows` | filter | one u32 / row, returns `Vec<u32>` |
| R1 `ShaderDispatch` | R5 `vsa16k_bind/bundle` | algebra | the literal FMA |
| R5 `Vsa16kF32` cycle output | R1 `ShaderBus.cycle_fingerprint` | threshold | `vsa16k_to_binary16k_threshold` |
| R1 `ShaderCrystal::on_crystal` | R2 `GateDecision` | decide | sink callback |
| R2 `MergeMode + delta` | R0 `BindSpace` columns | write | only via airgap |
| R1 `ShaderBus + MetaWord` | R4 `ExternalMembrane::project` | project | strips VSA/semiring per BBB |
| R4 `project()` | R7 `LanceMembrane::version_watcher.bump` | notify | `tokio::watch` only — no Lance MVCC bind today |
| R4 `MembraneGate::should_emit` | R4 fan-out | gate | RBAC bridge MISSING |
| R3 `OrchestrationBridge::resolve_thinking` | R6 `contract::thinking::ThinkingStyle` | lookup | catalogue |
| R6 grammar role keys | R5 `vsa16k_bind` | bind | role × content |
| R6 `ontology::EntityTypeId` | R0 `BindSpace.entity_type[row]` | write | Column H, Foundry Object Type |
| R7 `Dataset::write` | (cold archive) | persist | append-only |
| R7 `DriftEvent` | (MedCare oracle parity) | reconcile | F1 shipped per `medcare-foundry-vision.md` |

## Open seams — where the FMA is NOT yet fused

These are the points where multiplicand × multiplicand + addend should chain but doesn't, sourced from primary reads:

1. **R5 carrier-method drift.** `Vsa16kF32` algebra is 8 free functions, not methods. Click P-1 litmus violated. **Add `impl Vsa16kF32 { fn bind, bundle, cosine, permute }`** + the missing `vsa16k_permute` (Markov ρ^d unimplementable today).
2. **R1 ↔ R2 airgap soft-enforced.** `BindSpace` fields are `pub`, not `pub(crate)`. `engine_bridge::persist_cycle` takes `&mut BindSpace` directly. The doc-comment claims "mutations go through CollapseGate" — code does not enforce it.
3. **R4 ↔ R6 MembraneGate bridge missing.** `MembraneGate` trait has only `AllowAllGate`. `lance-graph-rbac::Policy` exists. No `impl MembraneGate for Arc<rbac::Policy>` anywhere.
4. **R7 cold-path stub.** `LanceMembrane::project` emits `CognitiveEventRow` to `tokio::watch`, not Lance. No `cognitive_events.lance` writer mirroring `LanceAuditSink`.
5. **R7 MVCC unbound.** `LanceVersionWatcher` wraps `tokio::sync::watch::channel`, never reads `Dataset::version()`. Phase-D TODO at `lance_membrane.rs:24`.
6. **R8 ↔ R3 cypher_bridge stub.** `cypher_bridge.rs` is regex; the real `lance-graph::parser::parse_cypher_query` (1932 LOC nom) exists in lance-graph and is never invoked from the bridge.
7. **R6 catalogue drift.** `CamByte` has 6 variants `{Heel, Branch, TwigA, TwigB, Leaf, Gamma}`; I10 doctrinal sequence is HEEL/HIP/BRANCH/TWIG/LEAF — **HIP missing**.
8. **R5 `lancedb` phantom.** `=0.27.2` pinned, zero source hits. Either delete or wire.
9. **R3 `step_trajectory_hash` stub.** Feature-gated `unimplemented!()` (orchestration.rs:194-201). PR #278 E4 cross-PR bridge unfinished.
10. **R5 `holograph` island.** Workspace-EXCLUDED, zero `lance_graph_contract` dep, parallel `query/parser.rs` (663 LOC), parallel `SentenceCrystal`, third VSA stack on bitpacked binary.

## What this map is NOT

- **Not a refactor plan.** The `R-list` from the prior review pass was the prompt-triggered reaction the user rejected. This map is the structural understanding the entropy ledger will work from.
- **Not a TODO.** Each "open seam" is a fact about today's code, not a directive.
- **Not synthesised from agent outputs.** Every claim above cites a file:line from primary source. The `LATEST_STATE.md` inventory currently misses 10 modules; this doc is the corrected baseline for the SoA-DTO surface specifically.

## Cross-references

- `.claude/knowledge/lab-vs-canonical-surface.md` — I1-I11 invariants. R0-R4 mapping.
- `.claude/knowledge/cam-pq-unified-pipeline.md` — R6 cam_pq, R7 cold path.
- `.claude/knowledge/cognitive-shader-architecture.md` — R0-R2 deep detail.
- `.claude/knowledge/vsa-switchboard-architecture.md` — R5 three-layer doctrine.
- `CLAUDE.md` — The Click P-1 (Vsa16kF32 FMA), three iron rules.
- `.claude/foundry-roadmap.md` — R3 (StepDomain::Smb/Medcare), R4 (RBAC + RLS).
- `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md` — companion ledger built from this map.
