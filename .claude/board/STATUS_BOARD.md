# Status Board — Cross-Deliverable View

> Deliverable-level status across all active integration plans.
> **Status** and **PR / Evidence** columns are the only mutable
> fields — title, plan-version, and scope are immutable.
>
> For plan-level status see `INTEGRATION_PLANS.md`.
> For per-PR decision history see `PR_ARC_INVENTORY.md`.
> For current contract inventory see `LATEST_STATE.md`.

---

## Status Legend

| Status | Meaning |
|---|---|
| **Shipped** | Merged to main. PR column cites the merge commit. |
| **In PR** | PR open, under review. Not yet merged. |
| **In progress** | Active branch, code in flight, not yet PR. |
| **Queued** | Next up; spec is clear; work not started. |
| **Backlog** | Future; still in scope but not yet queued for a phase. |
| **Deferred** | Explicitly parked. Rationale recorded. Will be revisited. |
| **Abandoned** | Removed from scope. Rationale recorded. Will not be revisited. |

Rules:
- New rows APPEND (at the bottom of the relevant section).
- Status field is the ONLY field that gets edited in place.
- When a deliverable ships, record the PR number — never delete the
  row.
- When a deliverable is superseded by a different design, keep the
  row with Status = Abandoned and cite the replacement.

---

## codec-sweep-via-lab-infra-v1 — JIT-first codec sweep

Active integration plan. 7 Phase 0 deliverables (D0.1–D0.7) + Phases
1–5 queued. One upfront Wire-surface rebuild; every candidate
afterwards is a JIT kernel, not a rebuild. Plan path:
`.claude/plans/codec-sweep-via-lab-infra-v1.md`.

### CI Gate — JC Substrate Proof

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| CI-JC | `.github/workflows/jc-proof.yml` — runs prove_it on every PR touching `crates/jc/` or `cam.rs` | **In PR** | 5-min timeout, exits 0 = substrate sound |

### Phase 0 — API hardening (partial in PR #225; remainder queued)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D0.1 | Extend `WireCalibrate` + `WireTensorView` (64-byte-aligned decode, object-oriented methods) | **Shipped** | #227 — 55/55 tests passing |
| D0.2 | `WireTokenAgreement` endpoint stub — I11 cert gate (Phase 0 surface, Phase 2 harness) | **In PR** | branch — `WireTokenAgreement` + `WireTokenAgreementResult` + `WireBaseline` DTOs + 3 round-trip tests. Stub handler returns `stub:true` / `backend:"stub"` until D2.1–D2.3 wire real decode-and-compare. |
| D0.3 | `WireSweep` streaming endpoint + Lance append stub | **In PR** | branch — `WireSweepGrid` + `cardinality()` + `enumerate()` → `Vec<WireCodecParams>` + `WireMeasure` enum + `WireSweepRequest` / `WireSweepResult` / `WireSweepResponse` DTOs + 5 tests. Streaming handler + Lance writer defer to Phase 3 D3.1. |
| D0.4 | Surface freeze (commit + rebuild) | **Ready** | D0.1–D0.7 all Shipped / In PR; freeze fires on merge of this PR. |
| D0.5 | `auto_detect.rs` — `ModelFingerprint` from `config.json` | **In PR** | branch — `auto_detect::{detect, ModelFingerprint, DetectError}` + HF config.json parser + per-architecture lane/distance heuristics (llama/qwen3/bert/modernbert/xlm-roberta/generic) + 8 tests. CODING_PRACTICES gap 1 remediated. |
| D0.6 | `CodecParamsBuilder` fluent API | **Shipped** | #225 — `contract::cam` +290 LOC of codec-params types, 14 tests (CODING_PRACTICES gap 3) |
| D0.7 | Precision-ladder validation (OPQ↔BF16x32, Hadamard pow2, overfit guard) | **Shipped** | #225 — `CodecParamsError` at `.build()` BEFORE JIT compile |

### Phase 1 — JIT codec kernels

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D1.1 | `CodecKernelCache` — structural cache layer (generic over handle) | **In PR** | branch — `CodecKernelCache<H>` + `StubKernel` + `get_or_compile` / `try_get_or_compile` with RwLock concurrent-safe double-check + compile/hit/ratio counters + 9 tests. Scaffold ships NOW; D1.1b Cranelift IR emission follows. |
| D1.1b | Adapter: `CodecKernelEngine` wrapping `ndarray::hpc::jitson_cranelift::JitEngine` with two-phase BUILD/RUN lifecycle (Arc-freeze). CodecParams → CodecScanParams adapter + codec-specific IR emission in jitson_cranelift/scan_jit analog | **Queued** | target ~250 LOC; `JitEngine` already ships (`/home/user/ndarray/src/hpc/jitson_cranelift/engine.rs`); the work is the CodecParams adapter + codec-specific JITSON template |
| D1.2 | Rotation primitives: Identity / Hadamard / OPQ as `RotationKernel` impls | **In PR** | branch — `RotationKernel` trait (Send+Sync+Debug, object-safe) + `IdentityRotation` (no-op) + `HadamardRotation` (real Sylvester butterfly, O(N log N) in-place, norm²-scaling verified) + `OpqRotationStub` (matrix-blob-id placeholder for D1.1b) + `build(&Rotation, dim)` factory + `RotationError` typed errors + 15 tests. Hadamard stays at Tier-3 F32x16 (add/sub, not matmul → no AMX benefit per Rule C). |
| D1.3 | Residual PQ via decode-kernel composition | **In PR** | branch — `DecodeKernel` trait (Send+Sync+Debug, object-safe, encode/decode/signature/bytes_per_row/dim/backend) + `StubDecodeKernel` (byte-exact round-trip for testing) + `ResidualComposer` (base + residual with subtract/add; nests recursively for depth >1) + `DecodeError` typed errors + 9 tests. Scope clarified: hydration/calibration path, NOT cascade inference (cascade uses `p64_bridge::CognitiveShader` per `cognitive-shader-architecture.md` line 582). |

### Phase 2 — Token-agreement harness (I11 cert gate) — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D2.1 | Token-agreement harness scaffold (reference model stub + top-k comparator + stub result) | **In PR** | branch — `ReferenceModel::{load, stub}` + `TokenAgreementError` + `TopKAgreement::{compare, top1_rate, top5_rate, meets_cert_gate, aggregate}` + `TokenAgreementHarness::{measure_stub, measure_full}` + 13 tests. Real safetensors load + decode loop defer to D2.2. |
| D2.2 | Decode-and-compare loop (top-k, per-layer MSE) | **Queued** | target ~220 LOC |
| D2.3 | Handler wiring for `/v1/shader/token-agreement` | **In PR** | branch — `token_agreement_handler` routes `WireTokenAgreement` → TryFrom(CodecParams) at ingress (precision-ladder + overfit guard fire here) → `ReferenceModel::load` or stub fallback on nonexistent paths → `TokenAgreementHarness::measure_stub()` → `WireTokenAgreementResult { stub:true }`. Route added: `POST /v1/shader/token-agreement`. Phase 0 Wire + Phase 2 harness now round-trip end-to-end. |

### Phase 3 — Sweep driver + Lance logger — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D3.1 | Server-side sweep handler + Lance fragment append | **In PR** | branch — `sweep_handler` batch mode: enumerates `WireSweepGrid::enumerate()`, validates each via TryFrom(CodecParams) at ingress, returns `WireSweepResponse { results: [WireSweepResult { kernel_hash, stub:true }], cardinality, elapsed_ms }`. SSE streaming + real calibrate/token-agreement per point deferred to D3.1b. Route: `POST /v1/shader/sweep`. |
| D3.2 | Client-side driver + config files | **In PR** | branch — 3 starter YAML configs (`configs/codec/{00_pr220_baseline, 10_wider_codebook, 12_hadamard_pre_rotation}.yaml`), `scripts/codec_sweep.sh` curl wrapper, `configs/codec/README.md`, YAML-shape spec-drift guard test. 118/118 tests pass. |

### Phase 4 — Frontier analysis — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D4.1 | DataFusion SQL over `sweep_results` Lance | **Queued** | target ~80 LOC |
| D4.2 | Pareto frontier notebook | **Queued** | target ~120 LOC |

### Phase 5 — Graduation — Fires per-candidate

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D5  | Graduation to canonical `OrchestrationBridge` (per winner) | **Queued** | target ~120 LOC per graduation; gate: ICC ≥ 0.99 held-out + token-agreement top1 ≥ 0.99 |

---

## elegant-herding-rocket-v1 — Phase-structured

Active integration plan, 12 deliverables D0 + D2–D11 (D1 dropped
early — CausalityFlow extension deferred). Plan path:
`.claude/plans/elegant-herding-rocket-v1.md`.

### Phase 1 — Shipped (PR #210, merged 2026-04-19)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D0  | grammar-landscape.md + linguistic-epiphanies + fractal-codec knowledge docs | **Shipped** | #210 — 3 docs, 1151 LOC |
| D4  | ContextChain reasoning ops (coherence / replay / disambiguate / WeightingKernel) | **Shipped** | #210 — 396 LOC, 8 tests |
| D6  | Role-key catalogue with contiguous `[start:stop]` slice addressing | **Shipped** | #210 — 404 LOC, 7 tests |

### Phase 2 — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D2  | DeepNSM emits `FailureTicket` on low coverage (wiring step 4) | **Queued** | — |
| D3  | Grammar Triangle wired into DeepNSM via `triangle_bridge.rs` | **Queued** | — |
| D5  | Markov ±5 bundler + Trajectory + content_fp (wiring steps 1-3) | **Shipped** | PR #243 — `content_fp.rs` (98 LOC, 5 tests), `markov_bundle.rs` (250 LOC, 8 tests), `trajectory.rs` (298 LOC, 4 tests). 63 deepnsm tests pass. |
| D7  | Thinking styles + free-energy + RoleKey-as-operator | **Shipped** | PR #243 — `thinking_styles.rs` (490 LOC, 12 tests), `free_energy.rs` (347 LOC, 7 tests), `role_keys.rs` bind/unbind/recovery_margin (295 LOC added, 14 tests). 175 contract tests pass. |

### Phase 3 — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D8  | Story-context bridge: AriGraph commit + global_context + contradiction (wiring steps 5-6) | **Queued** | — |
| D10 | Forward-validation harness (Animal Farm: chapter-10 > chapter-1 accuracy = AGI test) | **Queued** | — |

### Phase 4 — Backlog

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D9  | ONNX story-arc export + ArcPressure / ArcDerivative awareness hook | **Backlog** | — |
| D11 | Bundle-perturb emergence interface (transformer-free generative stack) | **Backlog** | — |

### Dropped / Deferred from the plan itself

| D-id | Title | Status | Notes |
|---|---|---|---|
| D1  | CausalityFlow 3→9 slot extension (modal/local/instrument + beneficiary/goal/source) | **Deferred** | User decision; follow-up PR after Phase 2 |

---

## Infrastructure / governance (not in elegant-herding-rocket)

Workspace-level bootstrap work. Tracked here rather than PR_ARC
because it's process, not architecture.

| Item | Status | PR / Evidence |
|---|---|---|
| CLAUDE.md §Session Start — three mandatory reads | **Shipped** | #211 |
| CLAUDE.md §A2A Orchestration — two layers (runtime + session) | **Shipped** | #211 |
| CLAUDE.md §Model Policy — grindwork vs accumulation + never Haiku | **Shipped** | #211 |
| CLAUDE.md §GitHub Access Policy — zipball-for-reads | **Shipped** | #211 |
| `.claude/BOOT.md` session entry + prior-art links | **Shipped** | #211 |
| `.claude/agents/BOOT.md` orchestration spec (renamed from README) | **Shipped** | #211 |
| `.claude/agents/README.md` function inventory | **Shipped** | #211 |
| `.claude/board/LATEST_STATE.md` current-state snapshot | **Shipped** | #211 |
| `.claude/board/PR_ARC_INVENTORY.md` append-only decision arc | **Shipped** | #211 |
| `.claude/board/INTEGRATION_PLANS.md` versioned plan index | **Shipped** | #211 |
| `.claude/board/STATUS_BOARD.md` this file | **Shipped** | #211 |
| `.claude/settings.json` team-shared governance (ask/deny + hooks) | **Shipped** | #211 |
| `.claude/hooks/session-start.sh` + `post-compact.sh` | **Shipped** | #211 |
| `.claude/skills/cca2a/` pattern-explanation skill | **Shipped** | #211 |
| `.claude/plans/elegant-herding-rocket-v1.md` plan in workspace | **Shipped** | #211 |

## Infrastructure — queued

| Item | Status | Notes |
|---|---|---|
| `.claude/rules/` with `paths:` frontmatter | **Backlog** | Audit rec 2; replace / complement `READ BY:` headers with path-scoped loading |
| Skill `context: fork` + `agent:` field | **Backlog** | Audit rec 4; read-only isolation for search-only skill variants |
| Auto memory (`~/.claude/projects/<proj>/memory/`) | **Backlog** | Audit rec; unstructured addition to curated LATEST_STATE |

---

## Cross-cutting research threads (orthogonal to grammar work)

Separate research thread — not entangled with grammar/crystal/A2A.
Tracked here so it doesn't get lost.

| Item | Status | Notes |
|---|---|---|
| Named-Entity pre-pass (NER) — biggest OSINT blocker | **Deferred** | Dedicated PR after Phase 2 |
| FP_WORDS = 160 migration (currently 157) | **Deferred** | Needs coordinated ndarray change |
| Crystal4K 41:1 persistence compression | **Deferred** | ladybug-rs owns it; would port later |
| 200–500 YAML TEKAMOLO templates per language | **Deferred** | Training pipeline; future |
| Cross-linguistic active parsers (EN+FI+RU+TR) | **Deferred** | Role keys exist; parsers later |
| Fractal-descriptor leaf codec (MFDFA on Hadamard) | **Research** | `.claude/knowledge/fractal-codec-argmax-regime.md`. 30-min probe first. |
| UK Biobank cardiac MRI benchmark | **Research** | Downstream of fractal-codec probe |
| Chess vertical (ruci + lichess-bot integration) | **Deferred** | Capstone Tier 0, parallel stream |
| Wikidata ingest (1.2 B triples → 14.4 GB) | **Deferred** | `.claude/knowledge/wikidata-spo-nars-at-scale.md` |
| OSINT pipeline (spider + reader-lm + DeepNSM) | **Deferred** | `.claude/knowledge/osint-pipeline-openclaw.md` |
| Python/TypeScript grammar-stack convergence | **Deferred** | `.claude/knowledge/grammar-landscape.md` §7 |

---

## Prior-art audit (61 + 41 = 102 existing docs)

Before this session, the workspace accumulated 61 `.claude/*.md`
top-level docs + 41 `.claude/prompts/*.md` files across prior
sessions. They are indexed in `.claude/BOOT.md §Existing content`
and `CLAUDE.md §Prior art`, but their individual **status** (still
active / superseded / archival) has not been audited.

Status rows per bucket, not per file (102 rows would drown the
board — use filesystem + INTEGRATION_PLANS + PR_ARC for per-file
history):

| Bucket | Count | Status | Notes |
|---|---|---|---|
| `.claude/*.md` top-level calibration reports / handovers / audits / snapshots | 61 | **Indexed** | Pointed at from BOOT.md + CLAUDE.md. Per-file active/superseded status: **Backlog** (needs one-pass audit). |
| `.claude/prompts/*.md` scoped session / probe / handover prompts | 41 | **Indexed** | Pointed at from BOOT.md via `SCOPED_PROMPTS.md` index. Per-file status: **Backlog**. |
| `.claude/knowledge/*.md` structured knowledge | 12 | **Active** | Current; each has `READ BY:` header; used by Knowledge Activation triggers. |
| `.claude/agents/*.md` specialist + meta-agent cards | 24 | **Active** | Current; used by spawning + Knowledge Activation. |
| `.claude/hooks/*.sh` | 2 | **Active** | Wired via settings.json. |
| `.claude/skills/cca2a/*.md` | 3 | **Active** | Current. |
| `.claude/plans/*.md` integration plans | 1 (v1) | **Active** | Elegant herding rocket v1, Phase 1 shipped. |

**Backlog item — prior-art audit.** One-pass sweep across the
61+41 files. Per file: label as active / superseded / archival
with a one-line note. Deliverable = an `ARCHIVE_INDEX.md` that
splits the 102 into current vs historical, plus rename/move of
superseded files into an `archive/` subdirectory. Estimate ~200
LOC of meta work, ~2 hours of reading. **Not urgent**; useful
before the next major planning session.

---

## ADR 0001 — Archetype transcode + Lance/DataFusion stack + Persona 16^32

Three-decision architectural lock, accepted 2026-04-24. First ADR in the
workspace. Path: `.claude/adr/0001-archetype-transcode-stack.md`.

| Decision | Status | Mutability |
|---|---|---|
| **D1 — Archetype is TRANSCODED, not bridged** | **Accepted** | Immutable (unlocking requires new ADR) |
| **D2 — Stack lock** (Lance + DataFusion + Supabase-shape scheduler + Arrow temporal; Polars rejected; Ballista deferred to 1s-P99) | **Accepted** | Ballista threshold mutable; rest immutable |
| **D3 — Persona 16^32 is THE identity space** (56-bit PersonaSignature; atom vector BBB-banned) | **Accepted** | Immutable; shared-DTO unification OPEN for future ADRs |

**Follow-up items tracked** (per ADR implications):

| Item | Priority | Location |
|---|---|---|
| DU-2 clarification (rename "bridge" → "transcode") | P2 | `unified-integration-v1.md` DU-2 |
| First `lance-graph-archetype` skeleton crate | P1 (when deliverable lands) | — |
| Grok gRPC A2A expert adapter | P2 | `TECH_DEBT.md` 2026-04-24 |
| Enrichment-shape follow-up ADR | P2 | `TECH_DEBT.md` 2026-04-24 |
| Ballista threshold tuning (post-benchmark amend) | P3 | `TECH_DEBT.md` 2026-04-24 |

Merged via PR #249 (2026-04-24).

---

## callcenter-membrane-v1 — Supabase-shape over Lance + DataFusion

External callcenter membrane crate. BBB enforced by Arrow type system at
compile time. Plan: `.claude/plans/callcenter-membrane-v1.md`. **Validated
by ADR 0001 Decision 2** (DM-4 `LanceVersionWatcher` + DM-6 `DrainTask`
pattern IS the Supabase-shape transcode approach).

### DM-0 / DM-1 — Shipped in this session

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| DM-0 | `ExternalMembrane` trait + `CommitFilter` in `lance-graph-contract/src/external_membrane.rs` | **Shipped** | session 2026-04-22 — `pub mod external_membrane` added to contract lib.rs |
| DM-1 | `lance-graph-callcenter` crate skeleton: `Cargo.toml` (feature gates) + `src/lib.rs` (stub + UNKNOWN markers) | **Shipped** | session 2026-04-22 — added to workspace members |

### DM-2 through DM-9 — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| DM-2 | `LanceMembrane: ExternalMembrane` impl with `project()` + compile-time BBB leak test | **In progress** | Phase A shipped `9a8d6a0` — `LanceMembrane` struct + `project()` + `ingest()` + `subscribe()` stub. Phase B: full Lance append + version counter pending DM-4. |
| DM-3 | `CommitFilter` → DataFusion `Expr` translator (`[query]` feature) | **Queued** | — |
| DM-4 | `LanceVersionWatcher` — tails Lance version counter, emits Phoenix `postgres_changes` (`[realtime]`) | **In PR** | branch `claude/supabase-subscriber-wire-up` — DM-4a/b/c: `version_watcher.rs` (117 LOC, 4 tests), `lib.rs` `pub mod version_watcher`, `LanceMembrane::watcher` field + `project()` calls `bump()`, `subscribe()` returns `watch::Receiver<CognitiveEventRow>`. |
| DM-5 | `PhoenixServer` — minimal WS server, Phoenix channel subset (`[realtime]`) | **Queued** | Resolve UNKNOWN-2 (which consumers need Phoenix wire?) first |
| DM-6 | `DrainTask` — `steering_intent` Lance read → `UnifiedStep` → `OrchestrationBridge::route()` | **In PR** | branch `claude/supabase-subscriber-wire-up` — DM-6a/b scaffold: `drain.rs` (89 LOC, 2 tests), `lib.rs` `pub mod drain`, `Poll::Pending` until follow-up PR wires real drain loop. |
| DM-7 | `JwtMiddleware` + `ActorContext` → `LogicalPlan` RLS rewriter (`[auth]`) | **Queued** | Resolve UNKNOWN-3 (pgwire?) + UNKNOWN-4 (actor_id type) first |
| DM-8 | `PostgRestHandler` — query-string → DataFusion SQL → Lance scan → Arrow response (`[serve]`) | **Queued** | Confirm PostgREST compat needed (§ 8 stop point 4) before building |
| DM-9 | End-to-end test: shader fires → `LanceMembrane::project()` → Lance append → Phoenix subscriber receives event | **Queued** | Depends on DM-2 through DM-6 |

---

## grammar-foundry-followup-v1 — Wire stubs to existing tissue

Plan: `.claude/plans/grammar-foundry-followup-v1.md`. Session 2026-04-29.
Six explicit stubs in PRs #275-#283 + 1 keystone (LF-12 Pipeline DAG). 13 PRs total in 3 waves.

### Wave 1 — no deps (parallel)

| D-id | Title | Status | Notes |
|---|---|---|---|
| PR-S1 | LF-12 Pipeline DAG: `UnifiedStep.depends_on` + topological executor | **Queued** | Keystone. Unblocks F4, G2, G6 |
| PR-F1 | PolicyRewriter UDF wrap: `RedactionMode` executors (closes `policy.rs:122`) | **Queued** | Unblocks F2, F5 |
| PR-F3 | Audit log Lance-backed writer (closes `lib.rs:100`) | **Queued** | |
| PR-F6 | `dn_path.rs` real scent via CAM-PQ (closes `dn_path.rs:53`) | **Queued** | Risk: bgz-tensor dep |
| PR-G1 | Triangle bridge real Causality footprint (closes `triangle_bridge.rs:90,221`) | **Queued** | |
| PR-G3 | ContextChain real `Binary16K` fingerprint (closes `context_chain.rs:345`) | **Queued** | |
| PR-G4 | verb_table seed 10/12 families (closes empty `default_table()` rows) | **Queued** | |
| PR-G5 | AriGraph episodic unbundle/rebundle (per `integration-plan-grammar-crystal-arigraph.md`) | **Queued** | |

### Wave 2 — depends on Wave 1

| D-id | Title | Status | Notes |
|---|---|---|---|
| PR-F2 | RowEncryption + DifferentialPrivacy executors (closes `policy.rs:147,181`) | **Queued** | After F1; needs key-mgmt ADR |
| PR-F4 | PostgREST → DataFusion dispatch (closes `EchoHandler` stub) | **Queued** | After S1 |
| PR-F5 | `audit_from_plan()` helper (closes `orchestration.rs:202` `unimplemented!`) | **Queued** | After F1 |
| PR-G2 | Disambiguator wiring at parser boundary + FailureTicket emission | **Queued** | After S1 |

### Wave 3 — depends on Waves 1+2

| D-id | Title | Status | Notes |
|---|---|---|---|
| PR-G6 | Animal Farm harness real run (D10 from PR #243) | **Queued** | After G1+G2+G3; text licensing needed |

---

## unified-integration-v1 — PersonaHub × ONNX × Archetype × MM-CoT × RoleDB

Plan: `.claude/plans/unified-integration-v1.md`. Session 2026-04-23.

| D-id | Title | Status | Notes |
|---|---|---|---|
| DU-0 | PersonaHub 56-bit compression: `(atom_bitset: u32, palette_weight: u8, template_id: u16)` offline extraction from 370M HF parquet rows | **Queued** | Runs offline; no code deps. Output: `personas.bin` + `sigs_dedup.bin` + `templates/*.yaml` |
| DU-1 | ONNX persona classifier @ L4/L5 — 288-class `(ExternalRole × ThinkingStyle)` product prediction; `style_oracle: Option<&OnnxPersonaClassifier>` in Think struct | **Queued** | Needs ~10K labeled cycles from Lance internal_cold (DM-2 must ship first); replaces Chronos proposal |
| DU-2 | Archetype ECS bridge crate `lance-graph-archetype-bridge` — `ArchetypeWorld → Blackboard`, `ArchetypeTick → UnifiedStep`, `project() → DataFrame component` adapters | **Queued** | Needs DM-2 (ExternalMembrane impl) before adapter can be built |
| DU-3 | RoleDB DataFusion VSA UDFs: `unbind`, `bundle`, `hamming_dist`, `braid_at`, `top_k` — registers in DataFusion session | **Queued** | Fingerprint column type decision needed first (FixedSizeBinary vs FixedSizeList); see open question in plan § 5 |
| DU-4 | MM-CoT stage split: add `rationale_phase: bool` to `CognitiveEventRow`; surface `FacultyDescriptor.is_asymmetric()` in projected RecordBatch | **Shipped** (Phase A: 2026-04-23 `a05979e`; Phase B: 2026-04-24) | Phase A: field exists. Phase B: `set_faculty_context()` on `LanceMembrane` wires `rationale_phase` from `AtomicBool`; orchestration layer calls it with `FacultyDescriptor::is_asymmetric()` + stage. Column is live, not ghost. |
| DU-5 | Board hygiene: DU-0 through DU-4 registered; INTEGRATION_PLANS.md + LATEST_STATE.md updated | **Shipped** (2026-04-23, commit `a05979e`) | Plan corrections + precision-tier §18 + father-grandfather concept committed in follow-up. |

## splat-osint-ingestion-v1 — Splat contract + EWA OSINT bridge

Active plan, 7 deliverables (D-SPLAT-1..7) staged across 6 PRs of the
`gaussian-splat-cam-plane-workaround.md` doc-sequence. PR 1+2 in flight
on branch `claude/splat-osint-ingestion`.
Plan path: `.claude/plans/2026-05-06-splat-osint-ingestion-v1.md`.

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-SPLAT-1 | `crates/lance-graph-contract/src/splat.rs` — `SplatChannel`, `CamPlaneSplat`, `SplatPlaneSet`, `AwarenessPlane16K`, `CamSplatCertificate`, `SplatDecision`, `TriadicProjection`, `ReasoningWitness64` + 10 unit tests | **In PR** | branch `claude/splat-osint-ingestion` |
| D-SPLAT-2 | `crates/jc/examples/osint_edge_traversal.rs` — EWA-Sandwich Σ-push-forward demo for OSINT 5-hop chain, side-by-side vs naive convolution | **In PR** | branch `claude/splat-osint-ingestion` |
| D-SPLAT-3 | `witness_to_splat()` deterministic conversion (PR 2 of doc-sequence) | **Queued** | — |
| D-SPLAT-4 | Splat deposition into BindSpace columns via `MergeMode::AlphaFrontToBack` lanes (PR 3 of doc-sequence) | **Queued** | — |
| D-SPLAT-5 | `PlanarSplatBundle4096` with local/short/medium/long bands (PR 4 of doc-sequence) | **Queued** | — |
| D-SPLAT-6 | Semantic-CAM-distance integration — survivor tile selection vs splatted pressure planes (PR 5 of doc-sequence) | **Queued** | — |
| D-SPLAT-7 | Replay fallback — exact 4096-cycle ThoughtCycleSoA replay slice when certificate insufficient (PR 6 of doc-sequence) | **Queued** | — |

Cross-ref: SPLAT-1 row in `ARCHITECTURE_ENTROPY_LEDGER.md` (Aspirational → Wired stage 1, entropy 4 → 2).

---


## Update protocol

When a deliverable ships:
1. Edit this file's Status column in place for the row → **Shipped**.
2. Fill in PR / Evidence column with the merge commit or PR #.
3. Append a new section to `PR_ARC_INVENTORY.md` (Added / Locked /
   Deferred / Docs / Confidence).
4. Update `LATEST_STATE.md` (Recently Shipped PRs + Current Inventory
   if types change).

When a deliverable moves phase (e.g. Queued → In progress → In PR):
1. Edit Status column in place. Don't reorder rows.
2. If the move reflects scope correction, also update
   `INTEGRATION_PLANS.md` Status line for the parent plan.

When a new deliverable is added to a plan:
1. Append a new row at the bottom of the plan's section.
2. D-id is sequential in the plan (D12, D13, etc.).
3. Original scope becomes immutable once committed.

When a deliverable is abandoned:
1. Edit Status → **Abandoned**. Don't remove the row.
2. Cite the replacement in Notes.
