# Architecture Entropy Ledger

> **APPEND-ONLY** — same governance as `PR_ARC_INVENTORY.md` /
> `EPIPHANIES.md` / `TECH_DEBT.md` / `ISSUES.md`. New rows append
> below. The `Entropy` and `Plan-Status` columns are the only
> mutable per-row fields; structural claims (Region / Component /
> File / DupCount / Seam) are immutable history.
>
> **Companion to** `.claude/knowledge/soa-dto-fma-map.md`. The
> map describes the architecture. This ledger scores each
> component on **integration state**, **loose-end count**, and
> **duplicate potential** — the three dimensions the user named:
> "what SoA has which integration state and how many lose ends and
> unconnected vs to be refactored, dead ends, vs integration plan
> exists, active or abandoned or stalled".
>
> **Why this exists:** today the workspace has DTO spaghetti —
> 6-copy NARS, 4-copy ThinkingStyle, 3-copy VSA stacks, 2-copy
> SentenceCrystal, 2-copy GateDecision (different shapes). The
> map names them; this ledger scores them so the next session can
> sort by entropy and pick the highest-leverage fix without
> re-discovery.

---

## Scoring rubrics

**Integration state:**
- `Wired` — implementation present, consumers reading/writing it, tests cover it.
- `Stub` — types exist, body is `unimplemented!()`/regex/empty `Ok(())`/single mock.
- `Aspirational` — referenced in docs/plans but **zero source hits in `crates/`**.
- `Dead` — code exists but no consumer / workspace-excluded / superseded.

**Duplicate potential:**
- `None` — single canonical, no parallel.
- `Low` — one shared def + adapters.
- `Med` — 2 copies, no formal bridge.
- `High` — 3+ copies OR same-name-different-shape collision.
- `Spaghetti` — 4+ copies with subtly different semantics.

**Entropy (1-5):**
1. Clean: canonical, wired, doc + tests + plan agree.
2. Mostly clean: small drift (e.g. doc lag), no behavioural risk.
3. Partial: working but loose end (one missing wiring, one stale doc).
4. High: 2-3 unconnected duplicates OR major seam broken.
5. Spaghetti: 4+ duplicates / cross-crate name collision / dead pointer in plan.

**Plan-status:**
- `Shipped` — plan deliverable shipped (cite PR#).
- `Active` — plan v1 active per `INTEGRATION_PLANS.md`.
- `Stalled` — plan exists but D-id has been "In PR" without a merge for >14 days.
- `Abandoned` — plan superseded or explicitly retired.
- `Missing` — no plan in `INTEGRATION_PLANS.md` or `STATUS_BOARD.md`.

---

## 2026-05-05 — initial snapshot

### Section A — Per-SoA / per-DTO inventory

Sorted by entropy DESC. Citations are file:line where the entry can be verified.

| ID | Region | Component | State | DupCount | DupPotential | LooseEnds | Plan | PlanStatus | Entropy |
|---|---|---|---|---|---|---|---|---|---|
| **NARS-1** | R6 | NARS inference enum | Wired (×6) | 6 | **Spaghetti** | `contract::nars::InferenceType`(5), `contract::grammar::inference::NarsInference`(7), `planner::nars::inference::NarsInference`(5), `planner::thinking::nars_dispatch::NarsInferenceType`(5), `causal-edge::edge::InferenceType`, `learning::feedback::NarsInferenceType`(5). 9 revision sites. 3 `TruthValue` copies. | none | Missing | **5** |
| **THINK-1** | R6 | ThinkingStyle enum | Wired (×4) + const(1) + bandit(1) | 6 | **Spaghetti** | `contract::thinking::ThinkingStyle`(36), `planner::thinking::style::ThinkingStyle`(12), `thinking-engine::cognitive_stack::ThinkingStyle`(12), `thinking-engine::superposition::ThinkingStyle`(5), `cognitive-shader-driver::engine_bridge::UNIFIED_STYLES`(12-const), `learning::cognitive_styles::StyleSelector`(RL bandit). `ord_to_thinking_style` (driver.rs:677) hand-picks contract-36 reps from 12-ord. No `From` adapters. | THINKING_RECONCILIATION.md (workspace-root, not plan) | Stalled | **5** |
| **VSA-1** | R5 | VSA carrier algebra | Wired (×3, distinct algebras) | 3 | **High** | `contract::crystal::fingerprint::Vsa16kF32` (ℝ multiply/add) + `ndarray::hpc::vsa::VsaVector` (Binary16K GF(2) XOR) + `holograph::bitpack::BitpackedVector` (XOR, workspace-EXCLUDED). No `vsa16k_permute` on f32 carrier — Markov ρ^d unimplementable. 8 free functions, no methods (Click P-1 violation). | vsa-switchboard-architecture.md (knowledge, not plan) | Missing | **5** |
| **DEEPNSM-NSM-1** | R5/R6 | nsm/ vs deepnsm | Wired (×2 parallel) | 2 | **High** | `lance-graph/src/nsm/{encoder,parser,similarity,tokenizer,nsm_word}.rs` (≈2,405 LOC) parallel to `crates/deepnsm/`. CLAUDE.md Phase 3 task "Consolidate nsm/ module" never ran. | none | Missing | **5** |
| **GATE-1** | R2/R6 | `GateDecision` namespace clash | Wired (×2 different shapes) | 2 | **High** | `contract::collapse_gate::GateDecision` is a struct `{gate: u8, merge: MergeMode}`. `contract::mul::GateDecision` is an enum (Flow/Hold/Block-style). Same name, different types, in the same crate. Quietly tolerated. | none | Missing | **4** |
| **SPO-1** | R7/R6 | Two SPO stores | Wired (×2 distinct) | 2 | Med | `lance-graph::graph::spo::*` (fingerprint-keyed, HammingMin truth-semiring) + `lance-graph::graph::arigraph::triplet_graph` (string-keyed `HashMap<String, Vec<usize>>`, 1,072 LOC). Share only `TruthValue`. **No bridge fn** between them — `to_fingerprints()` is a derive, not a writer. | none | Missing | **4** |
| **POLICY-1** | R4/R6 | Two `policy.rs` modules | Wired (×2 unconnected) | 2 | Med | `lance-graph-rbac::policy::Policy` (role-based, `AccessDecision`, `smb_policy()`) + `lance-graph-callcenter::policy::PolicyRewriter` trait + `ColumnMaskRewriter`. Neither calls into the other; **no `impl MembraneGate for rbac::Policy`**. | foundry-roadmap.md PR-1 (LF-3/DM-7 shipped #278); RBAC↔Membrane bridge has no plan entry | Stalled | **4** |
| **TRUTH-1** | R6 | `TruthValue` algebra | Wired (×3) | 3 | High | `contract::crystal::TruthValue` + `lance-graph-planner::nars::truth::TruthValue` + `causal-edge::tables::PackedTruth`. `revise()` fn at planner/nars/truth.rs:57; `revision()` at lance-graph/spo/truth.rs:62 — **subtly different math** (planner uses `c/(1-c)`, spo uses `c/(1-c+EPS)`). | none | Missing | **4** |
| **CRYSTAL-1** | R5/R6 | `SentenceCrystal` collision | Wired (×2 unaware) | 2 | High | `contract::crystal::sentence::SentenceCrystal` + `holograph::crystal_dejavu::SentenceCrystal` (line 248) + `holograph::sentence_crystal::SemanticCrystal`. `holograph` excluded from workspace; collision is silent. | none | Abandoned (holograph excluded) | **4** |
| **NARS-TRUTH-1** | R6 | `NarsTruth` struct | Wired (×2) | 2 | Med | `contract::exploration::NarsTruth:86` + `holograph::width_16k::schema::NarsTruth:97`. | none | Missing | **3** |
| **PARSER-1** | R3/R8 | Cypher parser | Wired (×1 real) + Stub (×3) + parallel (×1 excluded) | 5 | Spaghetti | `lance-graph::parser::parse_cypher_query` (1,932 LOC nom — REAL); `planner::strategy::cypher_parse.rs` (72 LOC substring); `cognitive-shader-driver::cypher_bridge.rs` (regex stub, confidence=0.5); `lance-graph-cognitive::cypher_bridge.rs` (5th copy); `holograph::query::parser.rs` (663 LOC parallel AST, excluded). **35 `format!("{:?}", logical_plan)` Debug-string introspection sites workspace-wide.** | codec-sweep / elegant-herding-rocket — no parser-unification plan | Missing | **5** |
| **HEEL-1** | R6 | HEEL/HIP/BRANCH/TWIG/LEAF | Wired (×3 different orderings) | 3 | High | `contract::cam::CamByte::{Heel, Branch, TwigA, TwigB, Leaf, Gamma}` — **HIP missing**. `lance-graph::graph::neighborhood::SearchCascade::{heel,hip,twig,leaf_rerank}` — BRANCH missing. `bgz17/router.rs` + `bgz-tensor/hhtl_d.rs` — third invention on bf16 basins. I10 doctrinal sequence violated by all three. | none | Missing | **4** |
| **CAM-DIST-1** | R6/R8 | `cam_distance` UDF wiring | Wired (opt-in only) | 1 | Low | UDF registered at `cam_pq/udf.rs:241`, `:257`, `:326`. Called from `query.rs:470` only when `with_cam_codebook(...)` is opted into. `datafusion_planner/mod.rs` does NOT register; default Cypher path can't reference `cam_distance`. | cam-pq-production-wiring-v1 (DRAFT) | Stalled | **3** |
| **DNTREE-1** | R6 | `DnTree` struct | Wired (in excluded crate) | 1 | None | `holograph/src/dntree.rs:36` (TreeAddr, DnNode, DnEdge, CogVerb). `holograph` excluded → no `lance-graph` consumer. `QueryStrategy::DnTreeFull` (`contract::nars.rs:36`) is a token with no traversal. | none | Abandoned (excluded) | **5** |
| **CONTRACT-INV-1** | R6 | `LATEST_STATE.md` Contract Inventory | Stale | n/a | None | 38 contract modules on disk; LATEST_STATE inventory mentions ~28. **10 modules unlisted:** `auth`, `distance`, `external_membrane`, `faculty`, `hash`, `ontology`, `persona`, `scenario`, `sigma_propagation`, `sla`. `lib.rs:26-32` docstring still says "7 modules". | none | n/a | **3** |
| **PLAN-INDEX-1** | (board) | INTEGRATION_PLANS.md | Stale | n/a | None | 18 plan files on disk; INTEGRATION_PLANS.md indexes 11. **7 plans unindexed:** archetype-scaffold, burn-ndarray-parity-sprint, foundry-roadmap-unified-smb-medcare, oxigraph-arigraph-cognitive-shader-soa-merge, sql-spo-ontology-bridge, tetrahedral-epiphany-splat-integration, thought-cycle-soa-awareness-integration (last shipped via PR #335). | n/a | n/a | **3** |
| **AGENT-LOG-1** | (board) | `.claude/board/AGENT_LOG.md` | Aspirational | 0 | None | Referenced by CLAUDE.md §Mandatory-Board-Hygiene-Rule + §Layer-2 A2A. **File does not exist** at `.claude/board/AGENT_LOG.md`. | none | n/a | **3** |
| **PR-ARC-1** | (board) | `PR_ARC_INVENTORY.md` | Stale | n/a | None | Last PR archived: #243 (2026-04-21). Main HEAD post-#335 (2026-04-30). ~50 PRs unarchived. CLAUDE.md §Mandatory-Board-Hygiene-Rule violated by every merge since #243. | n/a | n/a | **3** |
| **LATEST-RECENT-1** | (board) | `LATEST_STATE.md` Recently Shipped | Stale | n/a | None | Same staleness as PR-ARC-1. | n/a | n/a | **3** |
| **STATUS-CODEC-1** | (board) | STATUS_BOARD codec-sweep | Frozen | n/a | None | D0.2/D0.3/D0.5/D1.1/D1.2/D1.3/D2.1/D2.3/D3.1/D3.2 marked "**In PR**" since 2026-04-21 with **no PR number cited and no merge in `git log`**. Stalled or branch abandoned. | codec-sweep-via-lab-infra-v1 | Stalled | **4** |
| **MUL-ASSESS-1** | R6 | `MulAssessment` | Wired (×4) | 4 | Spaghetti | `contract::mul::MulAssessment:50` + `planner::mul::mod.rs:82` + `arigraph::orchestrator:172` + `cache::triple_model:57`. | none | Missing | **5** |
| **TRUST-1** | R6 | `TrustTexture` | Wired (×3 incompatible) | 3 | High | contract: 4 variants (Calibrated/Over/Under/Uncertain); planner: 5 (Crystalline/Solid/Fuzzy/Murky/Dissonant); arigraph orchestrator: 3 (Crystalline/Fibrous/Fuzzy). | none | Missing | **4** |
| **FLOW-1** | R6 | `FlowState` | Wired (×3) | 3 | High | contract:4, planner:4 (Apathy not Transition), lance-graph:4 with Apathy. | none | Missing | **4** |
| **COMPASS-1** | R6 | `CompassDecision` | Wired (×3 incompatible) | 3 | High | contract:`{StaySurface,GoMeta}` (2); planner:`{ExecuteWithLearning,Exploratory,SurfaceToMeta}` (3); lance-graph:`{None,ForceExplore,ForceSandbox}` (3). | none | Missing | **4** |
| **FREEWILL-1** | R6 | `free_will_modifier` width | Wired (mixed widths) | n/a | Low | `f64` in contract/planner; `f32` in arigraph orchestrator and contract/sensorium.rs and contract/plan.rs. | none | Missing | **2** |
| **MEMBRANE-GATE-1** | R4 | `MembraneGate` impls | Wired (1 default) | 1 | None | Only `AllowAllGate`. **No `impl MembraneGate for Arc<rbac::Policy>`** — natural RBAC bridge missing. | foundry-roadmap.md (RBAC bridge has no D-id) | Missing | **3** |
| **SEAL-1** | R4 | `MembraneRegistry::seal()` | Stub | n/a | None | `lance_membrane.rs:207-213` returns `Ok(())` empty with TODO "real topo sort via depends_on() once N>2 plugins". `Plugin::seal` trait exists, never called. | foundry-roadmap PR-3 | Stalled | **3** |
| **WATCHER-1** | R7 | `LanceVersionWatcher` | Stub (mis-named) | 1 | None | Wraps `tokio::sync::watch::channel`, **does not call `Dataset::version()`**. Phase D TODO at `lance_membrane.rs:24`. | callcenter-membrane-v1 DM-4 | Stalled | **4** |
| **PROJECT-LANCE-1** | R7 | `CognitiveEventRow` Lance writer | Aspirational | 0 | None | `external_intent.rs:88-104` schema-on-paper. `LanceMembrane::project` emits to `tokio::watch` only; **no Lance dataset writer**. Audit log writer exists for `AuditEntry`, not for `CognitiveEventRow`. | callcenter-membrane-v1 DM-9 | Stalled | **4** |
| **LANCEDB-PHANTOM-1** | R7 | `lancedb` SDK dep | Aspirational | 0 | None | `lance-graph/Cargo.toml:39+67` declares `lancedb = "=0.27.2"` behind feature `lancedb-sdk`. **Zero `lancedb::` source hits workspace-wide.** Phantom feature gate. | none | n/a (delete or wire) | **3** |
| **LANCEDB-NAME-1** | R5 | Naming leak | Wired (cosmetic drift) | n/a | None | `holograph/Cargo.toml:19` declares feature `lancedb = ["dep:lance"]` — conflates LanceDB-SDK with Lance-format in feature name. `contract::crystal::fingerprint:9-10,31,67,70` doc-comments call `Vsa16kF32` "lancedb-native VSA". `learning::cam_ops:1814,1840` defines `register_lancedb_ops()` for **Lance** ops. | none | Missing | **2** |
| **PERMUTE-1** | R5 | `vsa16k_permute` | Aspirational | 0 | None | Markov ρ^d braiding requires it; missing. `markov_bundle.rs:135` TODO "per-sentence pre-bundle vsa_permute is a follow-up. Until then, no permutation = aligned bundle." | elegant-herding-rocket-v1 D5 (markov bundler) shipped #243; permute deferred | Stalled | **4** |
| **CONTENT-FP-1** | R5/R6 | `content_fp.rs` (deepnsm) | Aspirational | 0 | None | Brief claimed shipped in PR #243; **does not exist** at `crates/deepnsm/src/content_fp.rs`. Closest: `fingerprint16k.rs::from_centroid_semantic` (which violates I-VSA-IDENTITIES by superposing CAM-PQ-style centroid-derived bits via XOR). | elegant-herding-rocket-v1 D5 region | Stalled | **4** |
| **ROLEKEY-OPS-1** | R6 | `RoleKey::bind/unbind/recovery_margin` | Removed | 0 | None | Deleted in cleanup commit `cd5c049` (`role_keys.rs:57-60,81-90,117-124` are explicit tombstones). **Board still advertises them** (`LATEST_STATE.md:32`, `STATUS_BOARD.md:120`); `TECH_DEBT.md:1057` flags the divergence. | none | n/a (board lag) | **4** |
| **TRAJ-HASH-1** | R3 | `step_trajectory_hash` | Stub | 0 | None | `orchestration.rs:194-201` is feature-gated `unimplemented!()`. PR #278 outlook E4 cross-PR bridge to PR #279's grammar substrate. | foundry-roadmap.md (cross-PR) | Stalled | **3** |
| **ENT-TYPE-1** | R6/R0 | `EntityTypeId` ↔ `BindSpace.entity_type[row]` | Wired | 1 | None | `ontology.rs:81` defines `EntityTypeId = u16` with explicit "Foundry Object Type / Column H" doc. `bindspace.rs:177` exposes `entity_type: Box<[u16]>`. `BindSpaceBuilder::push_typed` writes it. **The bridge IS shipped.** | bindspace-columns-v1 D-H1..4 | Active (PR #272 shipped Phase 1) | **2** |
| **ALPHA-7-1** | R1/R2 | Pillar-7 α-front-to-back | Wired (no JC prover) | 1 | None | `cognitive_shader.rs:319-340` (`AlphaComposite`, `ALPHA_COMPOSITE_DIMS=32`); `collapse_gate.rs:30-53` (`MergeMode::AlphaFrontToBack=3`, `ALPHA_SATURATION_THRESHOLD=0.99`); kernel at `driver.rs:493-529`. No `crates/jc/` prover for Pillar-7 (other pillars have empirical-concentration probes; this one only has unit tests). | none | Missing | **3** |
| **CARTAN-1**, **PRECOND-1** | (jc) | Pillars 2 + 4 | Stub (silent green) | n/a | None | `jc/cartan.rs:15-22` and `jc/precond.rs:12-19` return `PillarResult::deferred(...)` which counts as `pass: true` (`lib.rs:53`). `cargo run -p jc` reports "9/9 PASS" with two pillars stubbed. | (formal-scaffold idea-journal) | Stalled | **3** |
| **SPLAT-1** | R0/R1 | Gaussian-splat → CAM-plane splat | Aspirational | 0 | None | `CamPlaneSplat`, `SplatPlaneSet`, `SplatChannel`, `TetraSplatCertificate` — **zero hits in `crates/`**. Live only in `.claude/knowledge/gaussian-splat-cam-plane-workaround.md` and `.claude/plans/tetrahedral-epiphany-splat-integration-v1.md`. | tetrahedral-epiphany-splat-integration-v1 (NOT in INTEGRATION_PLANS.md) | Active (per disk) but unindexed | **4** |
| **SCENARIO-PROMOTE-1** | R1/R7 | Scenario-vs-evidence promotion gate | Aspirational | 0 | None | Per `holographic-temporal-access-projection.md`. `ShaderHit::source_kind: TemporalCandidateKind` does not exist. `contract::scenario` is a Lance-versioning facade for `ScenarioBranch`/`ScenarioDiff`, not a promotion gate. | none | Missing | **3** |
| **ADJ-THINK-1** | R3/R6 | Thinking-as-`AdjacencyStore` (I5) | Aspirational | 0 | None | I5 doctrine: 36 thinking styles are nodes in a CSR/CSC `AdjacencyStore` at τ-prefix `0x0D`. `tau()` addresses are computed (`thinking.rs`); **nothing writes those rows.** `AdjacencyStore<ThinkingStyle>` does not exist. | none | Missing | **4** |
| **DEBUG-STRINGIFY-1** | R3/R7 | `format!("{:?}", logical_plan)` | Workaround | 35 | High | 35 sites read DataFusion `LogicalPlan` Debug output as a stable surface. `lance_native_planner.rs:76-79` feeds the result back into `Planner::plan(query_hint)` which runs `to_uppercase().contains("MATCH")` against pretty-printed Rust struct syntax. | none | Missing | **5** |

### Section B — Spaghetti cluster (verified groups)

When fixing one row, fix the cluster — these duplicates are mutually entangled.

| Cluster | Members | Total entropy | Suggested order |
|---|---|---|---|
| **NARS** | NARS-1, TRUTH-1, NARS-TRUTH-1, MUL-ASSESS-1 | 17 | Collapse `NarsInference` first (canonical: `contract::grammar::inference::NarsInference` 7-variant). Then `TruthValue`, then `NarsTruth`. |
| **Thinking** | THINK-1, COMPASS-1, TRUST-1, FLOW-1, MUL-ASSESS-1, ADJ-THINK-1 | 24 | Adopt `contract::thinking::ThinkingStyle` (36) as canonical. Replace planner-12 + engine-12 + engine-5 + bandit. Drop `UNIFIED_STYLES[12]` const for `unified_params(style: ThinkingStyle)`. |
| **VSA carrier** | VSA-1, PERMUTE-1, CONTENT-FP-1, ROLEKEY-OPS-1, CRYSTAL-1 | 23 | Add `vsa16k_permute` first (Markov ρ^d unblocks D5). Then methods on `Vsa16kF32`. Then re-introduce role-key ops on f32 carrier (cleanly, not the reverted `[u64;157]`). |
| **Parser** | PARSER-1, DEBUG-STRINGIFY-1 | 10 | Wire `cypher_parse::CypherParse::plan` to call `lance-graph::parser::parse_cypher_query` (real nom). Eradicate the 35 Debug-string sites once a typed visitor exists. |
| **Foundry seal** | POLICY-1, MEMBRANE-GATE-1, SEAL-1, WATCHER-1, PROJECT-LANCE-1 | 18 | RBAC↔MembraneGate impl. Then real `seal()` topo sort. Then `LanceVersionWatcher` MVCC bind. Then `CognitiveEventLanceSink`. |
| **HEEL ladder** | HEEL-1, CAM-DIST-1, DNTREE-1 | 12 | Reconcile `CamByte` with I10 (rename or amend doc). Decide if `holograph::DnTree` graduates into contract, or if `QueryStrategy::DnTreeFull` retires. |
| **Board hygiene** | CONTRACT-INV-1, PLAN-INDEX-1, AGENT-LOG-1, PR-ARC-1, LATEST-RECENT-1, STATUS-CODEC-1 | 19 | Triple-ledger retrofit (in flight as background-agent task). Then create `AGENT_LOG.md`. |

### Section C — Open seams crossing region boundaries

These are NOT the cluster duplicates — they're **single-instance** unfused seams where multiplicand × multiplicand + addend SHOULD chain but doesn't:

| Seam | From → To | Single point of fix |
|---|---|---|
| R0 `BindSpace` write airgap soft-enforced | R0 → R2 | Privatise fields, add `apply(GateDecision, MergeMode, RowDelta)` write API |
| R3 `step_trajectory_hash` | R3 → R7 | Land PR #279 outlook E4 implementation |
| R4 RBAC ↔ MembraneGate | R4 → R6 | `impl MembraneGate for Arc<rbac::Policy>` |
| R7 LanceMembrane projection cold-path | R4 → R7 | `CognitiveEventLanceSink` mirror of `LanceAuditSink` |
| R7 `LanceVersionWatcher` MVCC | R7 → R4 | Bind `Dataset::checkout_latest().version()` |
| R8 cypher_bridge regex | R8 → R3 | Wire to `lance-graph::parser::parse_cypher_query` |
| Pillar-7 prover | (jc) | Build `crates/jc/src/alpha_front_to_back.rs` KS-concentration probe |
| Pillars 2/4 silent-green | (jc) | Fix `PillarResult::deferred` → `pass: false` or split `Deferred` state |

### Section D — Plan-status cross-reference

| Plan file | INTEGRATION_PLANS index? | STATUS_BOARD coverage? | Live deliverables? |
|---|---|---|---|
| `elegant-herding-rocket-v1` | yes | yes | D5/D7 SHIPPED #243; D2/D3/D8/D9/D10/D11 queued |
| `codec-sweep-via-lab-infra-v1` | yes | yes | D0.6/D0.7 SHIPPED #225; D0.1 SHIPPED #227; 10 D-ids stalled "In PR" since 2026-04-21 with no merge |
| `categorical-algebraic-inference-v1` | yes | partial | Companion to elegant-herding-rocket; #243 |
| `callcenter-membrane-v1` | yes | yes | DM-0/DM-1 SHIPPED 2026-04-22; DM-2 in progress; DM-4/DM-6 In PR; DM-7 stalled |
| `unified-integration-v1` | yes | yes | DU-4/DU-5 SHIPPED; DU-0..3 queued |
| `q2-foundry-integration-v1` | yes | NO | Proposed; no D-id rows |
| `lf-integration-mapping-v1` | yes | NO | Active; no STATUS_BOARD section |
| `bindspace-columns-v1` | yes | NO | Active; PR #272 shipped Phase 1 (Column H) — no STATUS_BOARD rows |
| `foundry-consumer-parity-v1` | yes | NO | Active; no STATUS_BOARD rows |
| `cam-pq-production-wiring-v1` | yes (DRAFT) | NO | DRAFT |
| `supabase-subscriber-v1` | yes | partial | DM-4/DM-6 In PR per STATUS_BOARD |
| `archetype-scaffold-v1` | **NO** | NO | unindexed plan on disk |
| `burn-ndarray-parity-sprint-v1` | **NO** | NO | unindexed |
| `foundry-roadmap-unified-smb-medcare-v1` | **NO** | NO | unindexed (separate from `.claude/foundry-roadmap.md` which IS the active roadmap) |
| `oxigraph-arigraph-cognitive-shader-soa-merge-v1` | **NO** | NO | unindexed |
| `sql-spo-ontology-bridge-v1` | **NO** | NO | unindexed |
| `tetrahedral-epiphany-splat-integration-v1` | **NO** | NO | unindexed (Pillar-7 splat work) |
| `thought-cycle-soa-awareness-integration-v1` | **NO** | NO | unindexed (PR #335 shipped) |

### Section E — Aggregate entropy

- **Total rows:** 41 inventory entries.
- **Spaghetti (entropy 5):** 7 (NARS-1, THINK-1, VSA-1, DEEPNSM-NSM-1, PARSER-1, DNTREE-1, MUL-ASSESS-1, DEBUG-STRINGIFY-1).
- **High (entropy 4):** 13.
- **Partial (entropy 3):** 11.
- **Mostly clean (entropy 2):** 2.
- **Clean (entropy 1):** 0.

**Highest-leverage cluster (most entropy / fewest dependencies):** Board hygiene (19 cluster-entropy, no source change required). Already in flight as the triple-ledger retrofit background agent.

**Highest-architectural-leverage cluster:** Thinking (24) — collapsing the 4-copy ThinkingStyle drift unlocks I5 (thinking-as-AdjacencyStore), which removes the `ord_to_thinking_style` hand-pick widening, which simplifies `MUL-ASSESS-1` and `COMPASS-1`.

**Highest-cognitive-leverage cluster:** VSA carrier (23) — adding `vsa16k_permute` unblocks Markov ρ^d braiding (D5 deferred work), which is the literal FMA the architecture is built around.

---

## Update protocol

When a row's state changes (e.g. NARS-1 collapses from spaghetti to single canonical):

1. **Append a new dated entry** below this snapshot. Reference the row ID.
2. **Do NOT edit this 2026-05-05 snapshot table.** Append-only — old rows are history.
3. New row: `## YYYY-MM-DD — <ID> resolution / state change` with old-state, new-state, evidence (PR#, file:line).

When a new SoA / DTO / bridge enters the architecture:

1. Append a new dated entry with `## YYYY-MM-DD — <ID> introduction`.
2. Score Region / DupCount / DupPotential / LooseEnds / Plan / PlanStatus / Entropy.
3. If part of an existing cluster, cite the cluster ID.

Cross-references:
- Companion map: `.claude/knowledge/soa-dto-fma-map.md`.
- Mandatory invariants: `.claude/knowledge/lab-vs-canonical-surface.md` (I1-I11).
- Iron rules: `CLAUDE.md` § Substrate-level iron rules (I-SUBSTRATE-MARKOV / I-NOISE-FLOOR-JIRAK / I-VSA-IDENTITIES).

---

## Section F — Maturity Ladder + Smart/Dumb + Deficit→Genius

> **Why this section is APPENDED, not part of Section A:** Section A
> was first written without the maturity-ladder / Smart-Dumb / Deficit
> dimensions. Per APPEND-ONLY governance, the prior table stays
> immutable; this section adds the missing axes against the same row
> IDs. Read alongside Section A.

### F.1 — Maturity ladder

The user-named stages an SoA / DTO walks through:

| Stage | Name | Means |
|---|---|---|
| **1** | **SoA Draft** | Type defined; ≤ 1 consumer; few/no tests; doc-only or inert. |
| **2** | **Wired** | Real consumer + tests; lives in the canonical hot path or is correctly isolated to the lab surface. |
| **3** | **Deeply Integrated** | Multiple consumers across crate boundaries; board-referenced; load-bearing. |
| **4** | **Only Source of Truth** | Stage 3 + zero parallel duplicates + locked invariant + Click-P-1 carrier-method genius. |

### F.2 — Smart vs Dumb (Click P-1 lens)

- **Smart** = methods on the carrier with the state to reason
  (`MetaWord::thinking()`, `MetaFilter::accepts(MetaWord)`,
  `BindSpace::meta_prefilter`, `ShaderHit::confidence_to_alpha`,
  `UnifiedStep::id()`).
- **Dumb** = free functions on the carrier's state, or fields-only
  with the verb living elsewhere (`vsa16k_bind(a, b)`,
  `vsa16k_bundle(items)`, `ShaderBus { ...fields }`).
- **Trait** = behaviour-shaped contract — neither smart nor dumb.

Per CLAUDE.md Click P-1 litmus: "free function = reject, method = accept".

### F.3 — Per-row maturity + smart/dumb + deficit

Same row IDs as Section A. Maturity is the current stage, not the
target. Deficit names what would elevate this row to Stage 4 + Smart;
**descriptive, not prescriptive** — naming the gap, not proposing a
refactor.

| ID | Maturity (1-4) | Smart/Dumb | Deficit→Genius (descriptive) |
|---|---|---|---|
| **NARS-1** | Stage 2 (Wired ×6, no canonical) | Trait pattern absent | Six-copy collapse to `contract::grammar::inference::NarsInference` (7-variant) + `core()` 7→5 adapter at boundary; deletes 5 enums + 1 NarsTruth duplicate |
| **THINK-1** | Stage 2 (Wired ×4, drift documented in THINKING_RECONCILIATION.md but never resolved) | Trait `ThinkingStyleProvider::select_from_assessment` has zero implementors | Adopt contract-36; replace planner-12 + engine-12 + engine-5 with `pub use`; replace `UNIFIED_STYLES[12]` with `unified_params(style: ThinkingStyle) -> &UnifiedStyle` |
| **VSA-1** | Stage 3 stalled (canonical exists; algebra is free functions) | **Dumb** (8/8 free fns: `vsa16k_bind/bundle/cosine/...`) | Newtype `Vsa16kF32` with `impl { fn bind, bundle, cosine, permute }`; the highest-leverage Click-P-1 deficit |
| **DEEPNSM-NSM-1** | Stage 1 + Stage 3 parallel | Dumb (both sides) | Delete `crates/lance-graph/src/nsm/` (2,405 LOC superseded); replace with thin re-export shim |
| **GATE-1** | Stage 3 (clash unresolved) | Both Smart on their own — but same name, different shapes | Rename `mul::GateDecision` → `mul::CompassGate` (frees the `GateDecision` name for collapse_gate) |
| **SPO-1** | Stage 3 (×2 distinct purposes, **not duplicates by design**) | `triplet_graph` Smart (string-keyed methods); `spo::store` Smart (fingerprint-keyed methods) | Add `arigraph::SpoBridge::promote_to_spo(&TripletGraph, gate, &mut SpoStore)` — promotes warm string-keyed entries into cold fingerprint-keyed store |
| **POLICY-1** | Stage 2 (both wired, no bridge) | Both Smart | `impl MembraneGate for Arc<rbac::Policy>` — natural RBAC↔BBB bridge |
| **TRUTH-1** | Stage 3 (planner) + Stage 3 (spo) + Stage 2 (causal-edge) | All Smart with subtly-different `revise()` math | Promote `TruthValueOps` algebra into `contract::nars`; both consumers re-export type aliases |
| **CRYSTAL-1** | Stage 3 (contract) + Stage 1 (holograph excluded) | Both Smart, name collision | Rename `holograph::sentence_crystal::SemanticCrystal` OR rewrite holograph to consume contract |
| **NARS-TRUTH-1** | Stage 2 (×2) | Both Dumb (struct only) | Single `contract::nars::NarsTruth`; holograph copy retires |
| **PARSER-1** | Stage 3 (lance-graph::parser real) + Stage 1 (planner stubs) + Stage 1 (cypher_bridge) + excluded (holograph parser) + 5th (lance-graph-cognitive copy) | Stage 3 parser is Smart (`CypherQuery` AST methods); stubs are Dumb (string-affinity) | Wire `cypher_parse::CypherParse::plan` to `parse_cypher_query`; promote `cypher_bridge.rs` from regex to thin wrapper |
| **HEEL-1** | Stage 2 (×3 different orderings) | Dumb enums | Reconcile `CamByte` with I10 (HIP missing) — rename or amend doc; pick one |
| **CAM-DIST-1** | Stage 2 (registered, opt-in only) | Smart (UDF function) | Register at `DataFusionPlanner::new` so default Cypher path resolves `cam_distance` |
| **DNTREE-1** | Stage 0 (in workspace-EXCLUDED crate) | Smart in holograph (TreeAddr/CogVerb), dead pointer in lance-graph | Either graduate `holograph::DnTree` into contract or formally retire `QueryStrategy::DnTreeFull` |
| **CONTRACT-INV-1** | n/a (board) | n/a | APPEND missing 10 modules to `LATEST_STATE.md` Current Inventory |
| **PLAN-INDEX-1** | n/a (board) | n/a | PREPEND 7 unindexed plans to `INTEGRATION_PLANS.md` |
| **AGENT-LOG-1** | Stage 0 (file does not exist) | n/a | Create `.claude/board/AGENT_LOG.md` (referenced by CLAUDE.md) |
| **PR-ARC-1** | n/a (board) | n/a | Backfill PRs #244–#335 (in flight by background retrofit agent) |
| **LATEST-RECENT-1** | n/a (board) | n/a | Same as PR-ARC-1 |
| **STATUS-CODEC-1** | n/a (board) | n/a | Audit "In PR" rows: flip to Shipped (with PR#) or Queued (if branch abandoned) |
| **MUL-ASSESS-1** | Stage 2 (×4) | Smart on contract side (`compute()`); Dumb on planner side (free fn `assess`) | Promote `MulAssessment::compute` carrier-method to canonical; delete free-fn copies |
| **TRUST-1** | Stage 2 (×3 incompatible variant sets) | All Dumb (enums only) | Pick contract canonical (4 variants); other two retire |
| **FLOW-1** | Stage 2 (×3) | Dumb enums | Same as TRUST-1 |
| **COMPASS-1** | Stage 2 (×3 incompatible decisions) | Dumb enums | Pick one; the navigation decision-space must be single-source |
| **FREEWILL-1** | Stage 2 (mixed widths) | n/a (scalar) | Pick f32 OR f64 globally; document why |
| **MEMBRANE-GATE-1** | Stage 1 (only `AllowAllGate`) | Trait | `impl MembraneGate for Arc<rbac::Policy>` — same as POLICY-1's seam, viewed from the gate side |
| **SEAL-1** | Stage 1 (empty `Ok(())`) | Trait pattern named, zero impls | Walk `Plugin::depends_on()`, run each `Plugin::seal()`, return first error |
| **WATCHER-1** | Stage 1 (mis-named) | Dumb wrapper around `tokio::watch` | Bind to `Dataset::checkout_latest().version()`; rename to honour what it does |
| **PROJECT-LANCE-1** | Stage 0 (no Lance writer for cognitive events) | n/a | `CognitiveEventLanceSink` mirroring `LanceAuditSink` |
| **LANCEDB-PHANTOM-1** | Stage 0 (zero source hits) | n/a | Delete dep + feature OR add `src/lancedb_sdk.rs` so the gate has a body |
| **LANCEDB-NAME-1** | n/a (cosmetic) | n/a | Rename holograph feature `lancedb` → `lance-format`; scrub "lancedb-native" doc-leak in contract |
| **PERMUTE-1** | Stage 0 (does not exist) | n/a | Implement `vsa16k_permute(v: &Vsa16kF32, rho: usize) -> Vsa16kF32` — Markov ρ^d carrier |
| **CONTENT-FP-1** | Stage 0 (does not exist) | n/a | Land `content_fp.rs` for real on the f32 carrier; eliminate the `fingerprint16k.rs::from_centroid_semantic` I-VSA-IDENTITIES violation |
| **ROLEKEY-OPS-1** | n/a (removed in cleanup commit cd5c049) | n/a | Either re-land `RoleKey::bind/unbind/recovery_margin` cleanly on f32 carrier, or scrub board references |
| **TRAJ-HASH-1** | Stage 1 (`unimplemented!()`) | n/a (stub fn) | Wire trajectory ↔ audit-log per PR #279 outlook E4 |
| **ENT-TYPE-1** | **Stage 3** (Foundry Object Type ↔ Column H bridge **already shipped** PR #272) | Smart (resolver fn) | Lift `EntityTypeId` to a newtype consumed directly by `BindSpace.entity_type[row]` instead of raw `u16` — minor genius polish |
| **ALPHA-7-1** | Stage 2 (wired, no JC prover) | DTO is Dumb; kernel in driver.rs is the carrier | Build `crates/jc/src/alpha_front_to_back.rs` KS-concentration probe |
| **CARTAN-1, PRECOND-1** | Stage 1 (deferred = silent green) | n/a (stub fns) | Either set `pass: false` or split a third state `Deferred` so `cargo run -p jc` reports honestly |
| **SPLAT-1** | Stage 0 (zero source hits in `crates/`) | n/a | Implement per `gaussian-splat-cam-plane-workaround.md`: `SplatChannel::{Support, Contradiction, Forecast, Counterfactual, Style, Source}` as `MergeMode::AlphaFrontToBack` lanes |
| **SCENARIO-PROMOTE-1** | Stage 0 (no `source_kind` field on `ShaderHit`) | n/a | Add `ShaderHit::source_kind: TemporalCandidateKind` per `holographic-temporal-access-projection.md` |
| **ADJ-THINK-1** | Stage 0 (`tau()` computed, never written) | n/a | `contract::thinking_adjacency::ThinkingAdjacency` — CSR-backed, NARS-weighted edges, τ-prefix `0x0D` slots; materialises I5 |
| **DEBUG-STRINGIFY-1** | n/a (workaround pattern) | Anti-pattern (Debug-as-API) | Inventory the 35 sites; replace `format!("{:?}", df_plan)` with typed `LogicalPlan` visitor; file workspace-wide TECH_DEBT entry |

### F.4 — Aggregate (maturity + Smart/Dumb)

| Bucket | Count | Examples |
|---|---|---|
| **Stage 4 (Only Source of Truth)** | 4 | `MetaWord`, `MetaFilter`, `ColumnWindow`, `UnifiedStep` (FNV-1a id-from-string fix is genius-level) |
| **Stage 3 stalled** (Deeply Integrated but blocked from Stage 4) | 8 | `BindSpace`, `Vsa16kF32`, `ThinkingStyle (contract)`, `triplet_graph`, parser (lance-graph), `OrchestrationBridge` (2-impl conflict), `LanceMembrane`, `CamByte` (HIP missing) |
| **Stage 2 (Wired, no canonical)** | 18 | All the duplicate clusters where multiple copies are wired but no consolidation |
| **Stage 1 (Draft / Stub)** | 7 | `cypher_bridge`, `MembraneGate` impls, `MembraneRegistry::seal`, `LanceVersionWatcher`, `step_trajectory_hash`, Pillars 2/4, `temporal/expert/cycle` columns of BindSpace as raw boxes |
| **Stage 0 (Aspirational / Missing / Dead)** | 9 | `vsa16k_permute`, `content_fp.rs`, `lancedb` SDK, `CognitiveEventRow` Lance writer, `SplatChannel`, `ShaderHit::source_kind`, `ThinkingAdjacency`, `holograph::DnTree` (excluded), `RoleKey::bind/unbind` (removed) |
| **Smart (Click P-1 satisfied)** | 12 | `MetaWord`, `MetaFilter`, `BindSpace`, `ShaderHit`, `UnifiedStep`, `StepDomain`, `CommitFilter`, `LanceMembrane`, `MembraneRegistry`, `LanceAuditSink`, `WireCalibrate`, `MulAssessment::compute` |
| **Dumb (free functions / fields-only on a carrier)** | 18 | **`Vsa16kF32`** (worst — algebra is 8 free fns), `ShaderDispatch`, `ShaderBus`, `ShaderCrystal`, `temporal` / `expert` / `cycle` columns, `MergeMode`, `RungLevel`, `EmitMode`, `CognitiveEventRow` |
| **Trait (behaviour contract)** | 7 | `ShaderSink`, `CognitiveShaderDriver`, `OrchestrationBridge`, `BridgeSlot`, `MembraneGate`, `ExternalMembrane`, `Plugin` |

### F.5 — The single biggest deficit-vs-genius gap

`Vsa16kF32` (R5) is the architecture's load-bearing carrier — the
literal FMA `bind(role, content) + bundle(prior)` lives on it — and
its algebra is currently **8 free functions on a type alias**, not
methods on a newtype carrier. Click P-1 explicitly says "free
function = reject, method = accept". This is the canonical
deficit-vs-genius gap: **Stage 3 stalled at Dumb when it should be
Stage 4 + Smart**. The missing `vsa16k_permute` (PERMUTE-1) is a
direct downstream consequence — the operation simply doesn't exist
on the carrier, so Markov ρ^d braiding (the diagram in CLAUDE.md
"The Click") is unimplementable today.

This row alone (combined with PERMUTE-1, CONTENT-FP-1,
ROLEKEY-OPS-1, CRYSTAL-1) carries the highest cognitive leverage
of any single fix in the ledger.

