# Grammar + Foundry Follow-up — v1

> **Status:** Active (2026-04-29)
> **Author:** main thread (Opus 4.7), session 2026-04-29
> **Scope:** Wire the stubs and scaffolds shipped in PRs #275-#283 to existing tissue. 13 PRs across two parallel tracks (6 Foundry + 6 Grammar) sharing one keystone (LF-12 Pipeline DAG). All deliverables target `main` directly; no stacking PRs (avoids the merge-order orphaning that bit #281/#283 → #284/#285).
> **Cross-refs:**
> - `lf-integration-mapping-v1.md` — LF-12 keystone rationale
> - `foundry-roadmap.md` — original PR-1..PR-5 sequence (PR-1/PR-2 shipped as #278/#280; this plan ships PR-3..PR-5 as PR-F1..F4)
> - `integration-plan-grammar-crystal-arigraph.md` — original AriGraph follow-up (now shipped as PR-G5)
> - `grammar-landscape.md` — case inventories that PR-G4 consumes

## Epiphany

The shipped Foundry+Grammar surface is 70% scaffold, 30% executor. Six explicit `stub`/`skeleton`/`placeholder`/`unimplemented!` markers in the merged code (verified by grep) name what remains. The next pass is **wiring**, not new scaffolding.

## Inventory: shipped scaffold vs. remaining stub

### Foundry — `crates/lance-graph-callcenter`

| Module | Wired | Stub |
|---|---|---|
| `rls.rs` (920 LOC) | RLS `OptimizerRule` predicates, sealed registry, 23 tests | actor_id column wiring (`filter_expr.rs:53`) |
| `audit.rs` (310 LOC) | `AuditSink` trait, `InMemoryAuditSink`, FNV hash | **Lance-backed writer** (`lib.rs:100`) |
| `postgrest.rs` (954 LOC) | URL parse, filter ops (eq/in/ilike/like/is), `EchoHandler` | **Real DataFusion dispatch** |
| `lance_membrane.rs` | `with_registry()`, Plugin handshake, atomic ops | — |
| `policy.rs` (309 LOC) | `PolicyRewriter` trait, `ColumnMaskRewriter` skeleton | **UDF column wrap** (`policy.rs:122`); **`RowEncryptionPolicy`** (`policy.rs:147`); **`DifferentialPrivacyPolicy`** (`policy.rs:181`) |
| `dn_path.rs` | DN path parser | **Real scent** (`scent_stub` XOR-fold at `dn_path.rs:53`) |
| `orchestration.rs` (contract) | `StepDomain`, `DomainProfile`, `Escalation`, `VerbTaxonomyId` | **`audit_from_plan()`** (`orchestration.rs:202` `unimplemented!`) |

### Grammar — `crates/lance-graph-contract/src/grammar/` + `crates/deepnsm/`

| Module | Wired | Stub |
|---|---|---|
| `role_keys.rs` (313 LOC) | 16384-dim slice catalogue, `LazyLock` arrays | — |
| `context_chain.rs` (+355) | `coherence_at`, `replay_with_alternative`, `WeightingKernel` | **Real `Binary16K` chosen-fingerprint** (`context_chain.rs:345`) |
| `thinking_styles.rs` (+725) | `GrammarStyleConfig`, NARS revision, 12 YAML configs | **Persistence E6** (`thinking_styles.rs:1209`) |
| `verb_table.rs` (120 LOC) | 144-cell `VerbFamily × Tense` schema | **10/12 family rows** (only Becomes + Causes seeded) |
| `disambiguator.rs` (137 LOC) | `Disambiguatable` trait, free function, 4 tests | **Caller wiring** (no parser callsite) |
| `markov_bundle.rs` | Ring buffer, role-indexed bundling | — |
| `trajectory.rs` | `Trajectory`, `role_bundle`, `role_candidates` | — |
| `ticket_emit.rs` (181 LOC) | `FailureTicket` decomposition | **Caller wiring** (parser doesn't emit) |
| `triangle_bridge.rs` (138 LOC) | NSM × Causality × Qualia merge | **Real Causality footprint** (`triangle_bridge.rs:90,221`) |
| `quantum_mode.rs` | `PhaseTag(u128)`, `HolographicMode` | **Dispatch** (no consumer reads `HolographicMode`) |
| `tests/animal_farm_harness.rs` | `EpiphanyPrediction`, `evaluate()`, 4 tests | **Real run on Animal Farm text** (D10 from PR #243) |

## Deliverables

### Shared keystone

**PR-S1 — LF-12 Pipeline DAG**
- Add `UnifiedStep.depends_on: Vec<StepId>` to `contract::orchestration`
- Topological executor in `lance-graph-planner::pipeline` (~300 LOC)
- Unblocks: PR-F4, PR-G2, PR-G6
- Effort: L. Per `lf-integration-mapping-v1.md` §"Open questions": schema field add + executor as one PR.

### Foundry track (6 PRs)

**PR-F1 — PolicyRewriter UDF wrap**
- File: `policy.rs:120-124` skeleton → real
- Wire `Expr::Column(c) → mask_udf(c)` for each column in `ColumnMaskRewriter::columns`
- Wire `RedactionMode::{Null, Constant, Hash, Truncate}` to four UDF impls
- Tests: 5 tests per `RedactionMode`
- Effort: S–M (~250 LOC)

**PR-F2 — RowEncryption + DifferentialPrivacy executors**
- Files: `policy.rs:147` and `policy.rs:181` stubs → executors
- Encryption: AES-GCM (or chacha20)
- DP: Laplace noise on aggregate columns marked `Marking::Pii`
- Depends on PR-F1 (UDF dispatch shape)
- Effort: M (~200 LOC + dep)
- Open ADR: encryption key management (KMS? in-process? user-supplied?)

**PR-F3 — Audit log Lance-backed writer**
- File: `audit.rs` — new `LanceAuditSink` (~200 LOC)
- One row per `AuditEntry` to a Lance dataset
- Reuse `LanceMembrane::with_registry()` for wiring
- Tests: scan-back-after-flush, 1k roundtrip
- Effort: M
- Cross-ref: `foundry-roadmap.md` PR-2 ("append-only Lance table")

**PR-F4 — PostgREST → DataFusion dispatch**
- File: `postgrest.rs` `EchoHandler` → real DataFusion (~400 LOC)
- Parse → `LogicalPlan` → `MembraneRegistry` (RLS rewrite) → `RecordBatch` → JSON
- Tests: 10–15 PostgREST shape tests with real Lance scans
- Depends on PR-S1
- Effort: L (split if needed: dispatch-without-RLS, then RLS-aware)

**PR-F5 — `audit_from_plan()` helper**
- File: `orchestration.rs:202` `unimplemented!` → helper
- Capture rewritten DataFusion `LogicalPlan` into `AuditEntry.rewritten_plan: Option<LogicalPlan>` after RLS
- Touches: `orchestration.rs`, `audit.rs` (new field)
- Depends on PR-F1 (datafusion-plan feature non-stub)
- Effort: S (~80 LOC)

**PR-F6 — `dn_path.rs` real scent**
- File: `dn_path.rs:53` `scent_stub` → real CAM-PQ lookup
- Use existing `bgz-tensor::CamCodecContract`
- Effort: S
- Risk: pulls bgz-tensor into callcenter dep tree

### Grammar track (6 PRs)

**PR-G1 — Triangle bridge real Causality footprint**
- Files: `triangle_bridge.rs:90,221` neutral 0.5 placeholder → real Pearl 2³ projection
- Each parsed sentence → `CausalEdge64` from SPO triple → triangle's causality channel
- Tests: two NSM-bound sentences, same Subject, different Pearl masks → different triangle outputs
- Effort: M (~100 LOC)

**PR-G2 — Disambiguator wiring at parser boundary**
- File: `crates/deepnsm/src/parser.rs` — call `disambiguate_general` when `coverage < LOCAL_COVERAGE_THRESHOLD`
- Emit `FailureTicket` (already in `ticket_emit.rs`) when can't resolve
- Tests: Wechsel "auf den Tisch / auf dem Tisch" → disambiguator picks Akk vs Dat
- Depends on PR-S1
- Effort: M (~150 LOC)

**PR-G3 — ContextChain real Binary16K fingerprint**
- File: `context_chain.rs:345` placeholder zero-fingerprint → real `MarkovBundler::role_bundle()` output
- Effort: S (~80 LOC)

**PR-G4 — verb_table seed beyond Becomes/Causes**
- File: `verb_table.rs` `default_table()` — populate 10 remaining `VerbFamily` rows × 12 `Tense` cells
- 120 cells of `SlotPrior` data from `grammar-landscape.md` §3
- Tests: 12 sentences (one per VerbFamily) → priors land non-zero
- Effort: M (~300 LOC mostly content)
- Risk: linguistic judgment — flag for `family-codec-smith` review

**PR-G5 — AriGraph episodic unbundle/rebundle**
- File: `crates/lance-graph/src/graph/arigraph/episodic.rs`
- Add `unbundle_hardened()`, `unbundle_targeted()`, `rebundle_cold()` per `integration-plan-grammar-crystal-arigraph.md`
- Wire to `UNBUNDLE_HARDNESS_THRESHOLD`
- Effort: M (~200 LOC)

**PR-G6 — Animal Farm harness real run**
- File: `crates/deepnsm/tests/animal_farm_harness.rs` — feed actual Animal Farm chapters
- Verify `chapter_10_acc > chapter_1_acc` (the AGI thesis from PR #243)
- Depends on PR-G1 + PR-G2 + PR-G3 (otherwise harness measures placeholder behavior)
- Effort: L (text licensing + ground-truth annotation are real work)

## Sequencing

```
Wave 1 (parallel, no deps):     S1, F1, F3, F6, G1, G3, G4, G5
Wave 2 (depends on Wave 1):     F2 (after F1), F4 (after S1), F5 (after F1), G2 (after S1)
Wave 3 (depends on Waves 1+2):  G6 (after G1, G2, G3)
```

## Out of scope

- LF-13/14/15+ (cron, row-level lineage, additional connectors) — separate connector-tier plan after LF-12 lands
- LF-50/52 (ModelRegistry + LlmProvider) — Stage 5 of `lf-integration-mapping-v1.md`
- Cross-linguistic bundling (Finnish/Russian/Turkish) — needs parallel corpora; separate session
- LF-71 redesign / scenario-world — already in PR via `claude/scenario-world-facade`
- Quantum mode dispatch wiring — defer until one consumer requests `SinglePhase` vs `PerRole`

## Open decisions

1. **PR-F2 encryption key management.** KMS? in-process? user-supplied? Block until ADR.
2. **PR-G6 Animal Farm text licensing.** Public domain in some jurisdictions. Gutenberg has it. Repo policy on bundling text data?
3. **PR-F6 bgz-tensor → callcenter dep.** bgz-tensor is currently `exclude`d from workspace. Pull in or keep `dn_path::scent` best-effort?
4. **PR-G4 ownership.** 120 cells × content quality dependence on linguistic judgment. Single-author or split by language family?

## Update protocol

This plan stays Active until all 13 PRs ship. Each PR-X gets a row in `STATUS_BOARD.md` under "grammar-foundry-followup-v1" section. PR merges flip Status: Queued → In progress → In PR → Shipped. Open decisions resolve via ADR in `.claude/decisions/` before their dependent PR opens.

End of v1.
