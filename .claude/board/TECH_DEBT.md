# Technical Debt Log — Open + Paid (double-entry, append-only)

> **Append-only ledger** for knowingly-deferred work: TODOs, shortcuts,
> workarounds, unsafe assumptions, missing probes, hardcoded
> thresholds, stubs, and anything else we shipped with intentional
> debt. Debt moves Open → Paid by status-flip; rows are NEVER
> deleted.
>
> **Purpose:** separate from `ISSUES.md` (bugs) and `IDEAS.md`
> (speculation). Tech debt is **code that works but we know
> something better is owed**. This file is where we admit it.

---

## Double-entry discipline

Same pattern as `ISSUES.md`:

1. **Open Debt** — known shortcut or deferral, captured at shipping
   time.
2. **Paid Debt** — when the shortcut is replaced with the proper
   implementation, append here with PR anchor + Status flip on the
   original Open entry to `Paid YYYY-MM-DD`.

The Open entry stays in its section forever (chronology). The Paid
section accumulates retirements for audit.

---

## Governance

- **Append-only.** Never delete a row.
- **Mutable fields:** `**Status:**` and `**Payoff:**` lines only.
- **`permissions.ask` on Edit** (same rule as other bookkeeping
  files).

## Cross-references

- `ISSUES.md` — an issue may become tech debt when knowingly
  deferred rather than fixed.
- `IDEAS.md` — a rejected idea that was "the better way" often
  leaves behind a debt entry documenting the compromise shipped
  instead.
- `PR_ARC_INVENTORY.md` — which PR introduced the debt + which PR
  paid it.
- `STATUS_BOARD.md` — debt items that block deliverables are
  cross-referenced from the D-id row.
- `EPIPHANIES.md` — an epiphany often retroactively turns something
  into tech debt (the old approach is now known to be suboptimal).

---

## Kanban Format (priority + scope on every entry)

Every debt item carries:
- **Priority** — `P0` must-pay-before-next-phase / `P1` pay-soon /
  `P2` eventual / `P3` keep-tracked-but-low.
- **Scope** — which agent / deliverable / domain owes it:
  `@<agent-name>`, `D<N>`, `domain:<tag>`.

Ticket tag: `[P2 @truth-architect D10 domain:grammar]`. Same
filter discipline — agents pull their own debt by `@`-mention.

## Open Debt

(Seeded with known deferrals from recent PRs. New items PREPEND
with today's date.)

## 2026-04-30 — TD-BGZ-TESTS-1: 5 pre-existing bgz-tensor test failures shipped with PR #308

**Status:** Open
**Priority:** P2
**Scope:** @family-codec-smith @palette-engineer crate:bgz-tensor
**Introduced by:** PR #308 (bgz-tensor pulled into workspace; failures pre-date workspace inclusion)
**Payoff estimate:** ~150-300 LOC across 4 modules; investigation-heavy

### What

PR #308's body claimed "194/200 tests pass, 6 pre-existing failures in
experimental codec paths" without listing which tests. Verified after
the merge — actually **5 failures** today (one was fixed by the master
merge during the PR cycle). The five are deterministic and reproducible
on `main` `540408e`:

| Test (file:line) | Panic |
|---|---|
| `gamma_calibration::tests::calibration_profile_size` (`src/gamma_calibration.rs:435`) | `assertion left == right failed: left: 48, right: 40` — calibration profile size drifted |
| `hhtl_d::tests::hhtl_d_entry_roundtrip` (`src/hhtl_d.rs:564`) | `(decoded.residual_f32() - 1.0).abs() < 1e-3` — residual roundtrip exceeds 1e-3 tolerance |
| `matryoshka::tests::encode_decode_roundtrip_nonzero` (`src/matryoshka.rs:469`) | panic at `band_max` selection in roundtrip path |
| `matryoshka::tests::roundtrip_quality_reasonable` (`src/matryoshka.rs:469`) | same panic site as above |
| `hhtl_cache::tests::test_hhtl_cache_256_size` (`src/hhtl_cache.rs:635`) | `assertion left == right failed: expected 206342 bytes, got 206358` (16-byte drift) |

### Why open

These are pre-existing failures in experimental codec paths
(gamma calibration, HHTL-D residual, Matryoshka encoding,
HHTL cache layout). They were never exercised by CI before
PR #308 because bgz-tensor was workspace-excluded. The
substrate (workspace inclusion + ndarray exports + compat shim
removal) is the actual deliverable of #308; fixing these tests
is its own scope.

### Cross-ref

- Reproduce: `cargo test -p bgz-tensor --lib --no-fail-fast`
- Test logs captured (locally): `/tmp/bgz-tensor-tests.log`, `/tmp/bgz-tests-detail.log`
- Per-failure panic detail: see table above
- PR #308 (merged commit `540408e`)
- Earlier session footprint: agent originally reported 6 failures
  (added `adaptive_codec::tests::adaptive_encode_decode`); that one
  was fixed by master merge `f429dec`

### Payoff path

1. **Quick win**: `hhtl_cache::test_hhtl_cache_256_size` — 16-byte
   size drift, likely a recent struct alignment change. Read
   `hhtl_cache.rs:635` + `git log -p src/hhtl_cache.rs`. ~1 hour.
2. **Medium**: `gamma_calibration::calibration_profile_size` —
   profile size 48 vs expected 40. Either expectation needs update
   or profile structure regressed. ~2 hours.
3. **Hard**: `hhtl_d::hhtl_d_entry_roundtrip` and the two
   `matryoshka::*roundtrip*` failures — encoding-quality
   regressions. Need codec-smith specialist.

Defer until calibration probe queue catches up
(`bf16-hhtl-terrain.md`).

```
## YYYY-MM-DD — <short title>
**Status:** Open
**Priority:** P0 | P1 | P2 | P3
**Scope:** @<agent> D<N> domain:<tag>
**Introduced by:** PR #NNN
**Payoff estimate:** <rough LOC / time>

<one paragraph: what shortcut was taken, why, what the proper fix
looks like, any blocking dependencies>

Cross-ref: <file:line / deliverable D-id / epiphany entry>
```

## 2026-04-24 — CognitiveEventRow ghost columns (`rationale_phase`, `dialect`) need live wiring (MM-CoT Phase B + polyglot Phase B)

**Status:** Open
**Priority:** P1
**Scope:** @host-glove-designer @bus-compiler DU-4 domain:mm-cot domain:callcenter
**Introduced by:** plan `unified-integration-v1.md` DU-4 (Phase A shipped in commit `a05979e`, 2026-04-23)
**Payoff estimate:** Phase B rationale_phase ~60 LOC on `lance_membrane.rs` (add `current_rationale_phase: AtomicBool` + setter + derive call in project); polyglot dialect ~80 LOC front-end parser hook. Total ≤ 1 day.

Per the user framing 2026-04-24 — *"make sure the SoA is not a Coast/Cost of a ghost"* — a column on `CognitiveEventRow` that is always set to the stub default (Phase A) is a ghost column: it exists in the schema but carries no cognitive signal. The SoA classification work (ADR-0002 I1 Codec Regime Split) classifies every column as Index / Argmax / Skip, but a column can be correctly classified as Index yet still be a ghost if nothing fills it.

Ghost columns shipped today:

| Column | Schema status | Current value | Phase B source |
|---|---|---|---|
| `rationale_phase` | bool, Phase A stub | always `false` | `FacultyDescriptor::is_asymmetric()` + explicit Stage-1/Stage-2 tracking (MM-CoT DU-4) |
| `dialect` | u8, Phase A stub | always `0` | polyglot front-end parser (SPARQL / SQL / Cypher / GQL / NARS) |

Proper fix for `rationale_phase`:
1. Add `current_rationale_phase: AtomicBool` to `LanceMembrane` state alongside the existing `current_role` / `current_faculty` / `current_expert` pattern.
2. Expose `set_rationale_phase(bool)` on the membrane.
3. Dispatcher sets it to `true` when entering Stage-1 rationale emission of an asymmetric faculty; `false` otherwise.
4. `project()` reads the atomic into the row.

Acceptance: every column on `CognitiveEventRow` must be either (a) live (changes across cycles based on actual state) or (b) explicitly marked with a phase label in the schema doc-comment. No silent Phase-A stubs.

Cross-ref: `crates/lance-graph-callcenter/src/external_intent.rs:131` (schema); `crates/lance-graph-callcenter/src/lance_membrane.rs:130` (current stub); `crates/lance-graph-contract/src/faculty.rs:98` (`is_asymmetric`); EPIPHANIES 2026-04-24 "I1 Codec Regime Split" + caveat note; STATUS_BOARD.md DU-4 row (Phase A shipped, Phase B pending).

## 2026-04-24 — jc Pillar 5b: direct Pearl 2³ mask-accuracy measurement (three-plane vs CAM-PQ-bundled)

**Status:** Open
**Priority:** P1
**Scope:** @savant-research @family-codec-smith domain:jirak domain:codec
**Introduced by:** this session's Pearl 2³ + CAM-PQ analysis
**Payoff estimate:** ~80 LOC addition to `crates/jc/src/jirak.rs` + test; ≤ 1 day

Today's jc pillar 5 measures *sup-error inflation* under weak dependence (dep 0.013287 vs IID 0.011671 at d=16384, N=5000) — a proxy for the CAM-PQ-contamination penalty. What it does NOT yet measure is the **direct Pearl 2³ mask-classification error**: given ground-truth Pearl masks (e.g. 110 = S✓, P✓, O✗), how often does three-independent-popcount + truth-table disagree with CAM-PQ-unbind + distance? Adding a `pub fn prove_pearl_mask()` arm to `jirak.rs` turns the 14 % sup-error finding into a direct "X % mask-misclassification rate" number that ADR-0002 Spine-Freeze can cite as the quantitative gate for the I1 Codec Regime Split.

Proper fix: extend `jirak.rs` with three disjoint-seed planes (S/P/O) + a bundled CAM-PQ-shaped code over the same content; run N ground-truth mask evaluations; report three-plane accuracy vs CAM-PQ accuracy. Keep the "10-minute proof" runtime promise. Blocks the ADR-0002 citation chain (see 2026-04-24 EPIPHANIES entry "I1 Codec Regime Split").

Cross-ref: `crates/jc/src/jirak.rs:124` (current `prove` function); `crates/lance-graph-contract/src/cam.rs` `CodecRoute::{CamPq, Passthrough}`; EPIPHANIES 2026-04-24 "I1 Codec Regime Split"; CLAUDE.md I-NOISE-FLOOR-JIRAK.

## 2026-04-24 — AriGraph episodic fingerprint as CAM-PQ first-tier cascade filter

**Status:** Open
**Priority:** P3
**Scope:** @family-codec-smith @truth-architect domain:arigraph domain:codec
**Introduced by:** this session's CAM-PQ-vs-AriGraph analysis
**Payoff estimate:** ~60 LOC in `episodic.rs` + CAM-PQ codec integration from `cam.rs` contract + test

`arigraph::episodic::Episode.fingerprint: Fingerprint = [u64; 256]` (2 KB per episode) is an argmax-regime structure per the I1 Codec Regime Split — retrieval is Hamming similarity, not exact identity lookup. It is a legitimate CAM-PQ-compression target (6 B per episode = 340× smaller), usable as a first-tier cascade filter: CAM-PQ ADC narrows N → k ≈ 64 candidates, then exact Hamming on the surviving [u64; 256] fingerprints. Triplets stay string-keyed (index regime, unchanged); only the similarity-retrieval index gets compressed.

Not urgent — current `retrieve_similar(fp, k)` is already O(n) Hamming and not a bottleneck at demo scale. Becomes relevant when episodic capacity grows past ~1M episodes (cascade saves memory + time). Until then, flagged for the future cascade-optimization pass. Must NOT touch triplet strings or archetype `ExpertId` — those are index regime.

Cross-ref: `crates/lance-graph/src/graph/arigraph/episodic.rs:104` (retrieve_similar); `crates/lance-graph-contract/src/cam.rs` `CodecRoute::CamPq` + `CAM_SIZE = 6`; EPIPHANIES 2026-04-24 "I1 Codec Regime Split" argmax-regime row for episodic.

## 2026-04-24 — Frankenstein blast radius on branch `claude/read-claude-md-jh51O` (Vsa10k / L3 / 157 confusion)

**Status:** Open
**Priority:** P0
**Scope:** @integration-lead @truth-architect domain:vsa domain:plans
**Introduced by:** this branch, commits `468357d`, `7a60c42`, `585f8b0`, `489911b` (plus inherited from earlier `dfcf246` / PR #210 / PR #242-#243)
**Payoff estimate:** plan-doc surgery (1 day) + contract rename already scoped in IDEAS.md (see 2026-04-19 entries)

Session on `claude/read-claude-md-jh51O` ran out of context before cleanup. Next session needs these breadcrumbs.

### Root errors (three clustered, one session)

1. **"L3" misread.** User said "L3" = CPU cache hardware budget (~16 MB). Session interpreted as cognitive-stack Layer 3 and built a fabricated precision-tier architecture on top.
2. **`Vsa10k BF16` fabrication.** No such type exists. The real types are `CrystalFingerprint::Vsa10kI8` (10 KB) and `Vsa10kF32` (40 KB legacy) — variants of `CrystalFingerprint`, not precision tiers of `Vsa10k`.
3. **`Vsa10k = [u64; 157]` inherited confusion.** 157-word bit-packed labelled "10K VSA" is the pre-existing error (source: `dfcf246`). Bit-packed never uses 10,000 — uses 16,384 (powers of 2). Only the high-dim numeric 40 KB / 80 KB forms legitimately carry 10K-D framing, and only for grammar ±5 wire bundling. Workspace already filed the rename in `IDEAS.md` 2026-04-19 entries (157 → 256 for binary; Vsa10k* → Vsa16k* for VSA floats); this session ignored the filed correction.

### Blast radius — plans written this session (needs surgery)

| File | Sections | Verdict |
|---|---|---|
| `.claude/plans/callcenter-membrane-v1.md` | §§ 14-17 (commit `468357d`) | Mixed. §14 cold-storage + §15 VSA-UDF-dispatch concept good; concrete sizes/type names poisoned. §16 persona-as-function (32 atoms × 16 weights, PersonaSignature 56-bit) good. §17 four-way multiply + ONNX-replaces-Chronos good; tensor-shape annotations poisoned. |
| `.claude/plans/callcenter-membrane-v1.md` | § 18 (commit `7a60c42`, lines 1139-1224) | **Delete whole section.** Three-tier precision table, father-grandfather compression, CK-safety proof — all built on fabricated `Vsa10k BF16` and L3-as-cognitive-layer. |
| `.claude/plans/callcenter-membrane-v1.md` | line 822 (edited `489911b`) | Already partially fixed. Still references "Fingerprint<256>" as if distinct from Vsa10k — verify correctness after rename. |
| `.claude/plans/unified-integration-v1.md` | DU-3 precision-note (lines 196-201, commit `7a60c42`) | **Delete paragraph.** References §18 that must go. DU-3 body (5 UDFs signature) kept but delegates must use canonical `RoleKey::bind/unbind` from PR #243. |
| `.claude/plans/unified-integration-v1.md` | DU-1 ONNX classifier (`468357d`) | Good core idea. `recent_fingerprints: Tensor[N, 16384]` tensor-shape comment at line 86 needs review against the 16K binary vs 16K-D float substrate split (see IDEAS.md 2026-04-19 CORRECTION). |
| `.claude/plans/unified-integration-v1.md` | DU-2 Archetype ECS bridge | Good. Mapping table (Entity=PersonaCard / World=Blackboard / Tick=CollapseGate fire) is sound. |
| `.claude/plans/unified-integration-v1.md` | DU-4 `rationale_phase: bool` (commit `a05979e`) | Shipped correctly to `CognitiveEventRow`. No change needed. |

### Blast radius — source (this session did NOT modify these; flagging for rename sweep)

- `crates/lance-graph-contract/src/grammar/role_keys.rs` — `VSA_WORDS = 157`, `Vsa10k = [u64; 157]`. Originates `dfcf246` (pre-PR #210). Already in rename sweep per IDEAS.md `CORRECTION-OF 2026-04-19 FP_WORDS`.
- `crates/deepnsm/src/{content_fp,markov_bundle,trajectory,context,encoder}.rs` — all consume `Vsa10k` / `VSA_WORDS = 157`. Added in PR #243 (`c6e69c4`). Follows whichever direction the rename goes.
- `CLAUDE.md` P-1 section — references `[u64; 157]` and `trajectory: Vsa10k`. Prose; updates after source rename.

### Blast radius — vsa_udfs.rs (commit `585f8b0`)

`crates/lance-graph-callcenter/src/vsa_udfs.rs` (628 lines) has three wrong operations — flagged separately for the canonical-delegation pass:

- `unbind_op`: set-bit-counting on role-indexed slice → returns fraction. WRONG — should delegate to `RoleKey::unbind` + `recovery_margin` (shipped in `79ac8f6` / PR #242).
- `bundle_op`: implements XOR, named "vsa_bundle". WRONG NAME — XOR is `MergeMode::Xor` (single-writer delta); violates **I-SUBSTRATE-MARKOV** iron rule if used on state-transition paths. Rename to `vsa_xor` (honest) or implement true CK-safe bundle.
- `braid_at_op`: cyclic word rotation. WRONG — should use `vsa_permute` (PR #209 reference braid).

### Good ideas salvageable (architectural, not in poisoned sections)

Verbatim user framings from this session, recorded so they don't get lost:

1. **Two callcenter modes coexist.** (a) DataFusion polyglot query mode — addresses the 4096 / 0xFFF command space (SPARQL / SQL / Cypher / GQL / NARS); Redis-like UDF access. (b) VSA blackboard orchestration — roles bind work orders into accumulator; CollapseGate XOR flushes into Markov trajectory; new empty ledger starts.
2. **VSA lazy-buffer pattern.** Bind events → accumulate in VSA → CollapseGate XOR flush → Markov trajectory → scalar `CognitiveEventRow` projection → external consumer. The accumulator is the lazy buffer; CollapseGate is the bell.
3. **Kitchen analogy.** Work orders bind into VSA (order sheet) → CollapseGate = bell → callcenter/waiter picks up scalar `CognitiveEventRow` → external consumer gets ticket.
4. **BBB iron rule already in contract source** (`external_membrane.rs:10`): `Self::Commit` MUST NOT contain `Vsa10k`, `RoleKey`, `SemiringChoice`, `NarsTruth`. Any plan proposing these cross the gate is invalid by construction.
5. **Grammar ±5 wire format is the ONLY legitimate 10,000-D / 10K-D / 16K-D float carrier.** 40 KB (f32) / 80 KB (u8 5-lane) per the IDEAS.md rename plan. Bit-packed fingerprint substrate is separate: `Container<[u64; 256]>` = 16,384 bits = 2 KB, Hamming-only, NOT VSA.
6. **Chronos → ONNX replacement** (DU-1): full 288-class `(ExternalRole × ThinkingStyle)` product prediction vs Chronos 1D scalar. Grounding sound; tensor-shape details need re-verification post-rename.
7. **Archetype ECS bridge** (DU-2): `VangelisTech/archetype` sits ABOVE callcenter; Entity = PersonaCard, World = Blackboard, Tick = CollapseGate fire. Thin adapter crate `lance-graph-archetype-bridge`. Mapping table intact.

### Prior art this session ignored

- `.claude/board/IDEAS.md` 2026-04-19 entries: "FP_WORDS = 256 (supersede the 160 plan)" + "CORRECTION-OF 2026-04-19 FP_WORDS = 256" + "REFINEMENT-OF 2026-04-19 CORRECTION-OF FP_WORDS scope". Explicit governance ban: *"There shall be zero occurrences of '10,000-D binary VSA' / '10,000-bit VSA' in any workspace file."* This session wrote multiple such occurrences.
- `.claude/board/TECH_DEBT.md` 2026-04-19 `FP_WORDS = 157 (not 160)` entry (exists above this row once this appends).
- `.claude/board/TECH_DEBT.md` 2026-04-19 `VSA substrate renaming: Vsa10k* → Vsa16k* + float framing` entry.

### Archetype / Chronos — breadcrumbs the plan docs do NOT yet carry

Flagging here what's in this session's conversation context but not in `unified-integration-v1.md` or `callcenter-membrane-v1.md`. Budget honesty: recording only what I can attribute to this session's conversation — not reconstructing.

**Archetype — name collision is undocumented.** "Archetype" in DU-2 refers exclusively to the external `VangelisTech/archetype` Rust ECS crate (simulation layer above callcenter). But the user explicitly noted mid-session: *"What is internal is the person archetypes"* — pointing at `crates/thinking-engine/src/persona.rs` (internal VSA-bound cognitive archetypes). These are two distinct objects sharing the word "archetype":

| Sense | Lives in | Role | Covered in plans? |
|---|---|---|---|
| External ECS Archetype | `VangelisTech/archetype` crate | Simulation tick layer ABOVE callcenter | Yes — DU-2 |
| Internal persona-archetype | `thinking-engine/src/persona.rs` | VSA-bound cognitive identity INSIDE substrate | **No** — only contract-side `PersonaCard` (metadata routing) is named, not the internal archetype |

Next session: add an explicit disambiguation paragraph to DU-2 and to `callcenter-membrane-v1.md` § 16 ("Persona as function") noting that `PersonaCard` (contract, metadata) and `thinking-engine::persona::*` (internal, VSA-bound archetype) are different objects; the callcenter sees the former, never the latter. The "32 atoms × 16 weights" formulation in § 16 sits at the metadata/compression layer — whether it corresponds to or is distinct from the internal archetype is an **open question** (I don't have confident attribution for this from this session's conversation).

**Chronos — what's captured vs what's missing.** Replacement rationale ("1D scalar → 288-class `(ExternalRole × ThinkingStyle)` product"; "ONNX infra already justified by Jina v5 ONNX on disk"; "task = classification, not time-series forecasting") is captured in § 17 of callcenter-membrane-v1.md and DU-1 of unified-integration-v1.md. I do NOT hold confident additional Chronos-specific content from this session's brainstorming that could be added without fabrication. Flagging so the next session knows: if the $200 session's memory of the brainstorm contains richer Chronos material, that's new information to add, not something this session held and failed to record.

**Archetype × Chronos interplay — NOT captured anywhere.** The user grouped "archetype and chronos" together as joint brainstorming content. The plans treat them as independent deliverables (DU-1 Chronos-replacement, DU-2 Archetype-bridge). Whether they compose (e.g. Archetype tick driving ONNX-classifier-as-style-oracle at each tick, feeding Blackboard rounds) is **not documented**. Candidate composition: `ArchetypeTick → UnifiedStep → CollapseGate fire → ONNX classifier predicts next (role, style) → next tick's PersonaCard` — but this is my conjecture, not session-attributable, so flagging as an **open design question** for the next session rather than writing it into a plan.

### ONNX > Chronos — what's in plans vs what's missing

§ 17 of `callcenter-membrane-v1.md` (commit `468357d`) holds a 6-criterion "Why ONNX over Chronos" table: **Output** (1D scalar vs 288 logits) · **Task type** (forecasting vs classification) · **Training** (pre-trained vs Lance E-DEPLOY-1 corpus) · **Precision** (style axis vs role × thinking product) · **Infra** (separate model vs `ort` crate justified by Jina v5 ONNX on disk) · **Fit** (partial vs full product).

**Gap:** the table enumerates where Chronos LOSES but not where Chronos WOULD LEGITIMATELY WIN. User flagged "Chronos only useful for XYZ" as a piece of brainstorm content not captured. I do NOT hold session-attributable XYZ content. Reasoning from first principles (NOT session-attributable, flag if reused): Chronos is appropriate when the task is genuinely temporal forecasting — predicting next F-value N cycles ahead, predicting style-drift onset, predicting gate-commit-rate over a rolling window, auxiliary time-series heads on the Lance internal_cold timeline. These would compose with (not replace) the ONNX classifier's instantaneous prediction. The next session should either fill the XYZ from their own brainstorm recall, or explicitly reject Chronos across all use cases.

### Archetype / persona / thinking-style modeling — epiphany candidates not yet in EPIPHANIES.md

§ 16 and § 17 hold modeling insights that sit in plan docs but have NOT been promoted to dated entries in `.claude/board/EPIPHANIES.md`:

- **Four-way multiply = architecture** (§ 17): `(persona 288 × style 36 × stage 2 × learned-dynamics)` ≈ 20 736 × oracle configurations; F-descent IS the automatic architecture search over this space; misaligned configurations are dropped by the CollapseGate predicate. Epiphany framing: *free-energy minimisation over the four-way product = neural-architecture search without an outer optimiser*.
- **Persona as function** (§ 16): 32 cognitive atoms × 16 weightings = 16^32 addressable persona space, compressed to 56-bit PersonaSignature. YAML runbooks are macro scaffolding for the context loop, not persona identity. Epiphany framing: *persona identity is a coordinate in atom-space, not a YAML artefact; the YAML is a program on the context loop*.
- **MM-CoT stage split = faculty asymmetry** (§ 17 row): `rationale_phase: bool` maps to `FacultyDescriptor::is_asymmetric()` (inbound_style ≠ outbound_style). Epiphany framing: *MM-CoT rationale-vs-answer split is NOT a new axis — it reuses the existing faculty asymmetry from the contract*.

**Already in EPIPHANIES.md:** `E-DEPLOY-1` (commit `5dbdf25`) captures the Supabase-shape thinking-extension 9-dim joint epiphany. The three candidates above are NOT yet prepended there. Board-hygiene rule from CLAUDE.md requires EPIPHANIES entries in the same commit as the plan additions — this session violated that rule for § 16 / § 17.

Next session action: prepend three dated EPIPHANIES.md entries using the framings above, cross-referenced to § 16 / § 17 / commit `468357d`. Do NOT write additional modeling content I don't hold; if the next session has richer brainstorm recall, let them author from that rather than inheriting my reconstruction.

### Correction plan for next session (P0 order)

1. **Delete** `callcenter-membrane-v1.md` § 18 (lines 1139-1224).
2. **Delete** `unified-integration-v1.md` DU-3 precision-note (lines 196-201).
3. **Append** EPIPHANIES.md entry with the three-cluster root-error summary (L3 / Vsa10k-BF16 fabrication / 157 inheritance), referencing this TECH_DEBT row.
4. **Fix** `vsa_udfs.rs` three operations to delegate to canonical `RoleKey::bind/unbind`, `vsa_permute`, and an honest XOR name.
5. **Update** `LATEST_STATE.md` § Current Contract Inventory to include `FacultyRole` + `FacultyDescriptor` (added this session in `2a4a245`, never board-logged — same-commit hygiene rule violation).
6. **Proceed with** the IDEAS.md Vsa10k* → Vsa16k* rename sweep OR scope-limit it per the REFINEMENT entry (grammar / quantum / ladybug allow-list) — that is its own PR, not a cleanup prerequisite.

Cross-ref: EPIPHANIES.md entries needed; IDEAS.md 2026-04-19 rename-sweep entries; PR #242 (`defe928`) The Click categorical-algebraic click; PR #243 (`c6e69c4`) D5 Trajectory; this branch commits `468357d` / `7a60c42` / `585f8b0` / `489911b` / `a05979e` / `2a4a245`.

---

### Seeded from PRs #204–#211

## 2026-04-21 — Vsa10k = [u64; 157] is a fifth representation format (Frankenstein effect)
**Status:** Open
**Priority:** P0
**Scope:** @truth-architect @container-architect D5 D7 domain:vsa domain:grammar
**Introduced by:** PR #242 / #243 (session 2026-04-21)
**Payoff estimate:** ~200 LOC refactor across role_keys.rs + content_fp.rs + markov_bundle.rs + trajectory.rs

`Vsa10k = [u64; 157]` introduced in `role_keys.rs` is a standalone
type alias that doesn't interoperate with the four existing vector
representations (`Binary16K [u64; 256]`, `Vsa10kI8 [i8; 10_000]`,
`Vsa10kF32 [f32; 10_000]`, DeepNSM `VsaVec [u64; 8]`).

**Blast radius (5 disconnection points):**
1. ContextChain operates on Binary16K (256 words). Trajectory uses
   Vsa10k (157 words). Step 8 KL-feedback blocked by type mismatch.
2. EpisodicMemory stores CrystalFingerprint. No Vsa10k conversion.
3. crystal::fingerprint has f32 VSA algebra. role_keys has bitpacked
   VSA algebra. Two parallel surfaces, incompatible types.
4. 157 words not SIMD-aligned (debt entry says 160).
5. Prior debt says rename to 16K + re-scale slices to [0..16384).
   This session went opposite (kept 10K).

**Resolution recommendation:** Option (A) — adopt Binary16K (256
words) as THE bitpacked carrier. Re-scale role-key slices to
[0..16384) proportionally. Eliminates the type mismatch, aligns
with existing debt entry, ContextChain/EpisodicMemory interop free.

Cross-ref: debt "FP_WORDS = 157 (not 160)", debt "VSA substrate
renaming: Vsa10k* → Vsa16k*", `role_keys.rs:41`.

---

## 2026-04-19 — Contract `ContextChain::coherence_at` returns 0 for non-Binary16K variants
**Status:** Open
**Priority:** P2
**Scope:** @resonance-cartographer @container-architect D4 domain:grammar
**Introduced by:** PR #210
**Payoff estimate:** ~80 LOC + tests

D4 shipped with Hamming-based coherence on the `Binary16K` variant
only; other `CrystalFingerprint` variants (`Structured5x5`,
`Vsa10kI8`, `Vsa10kF32`) return 0 as a zero-dep fallback. Cosine
coherence on the f32 variants would unlock richer disambiguation
but requires adding a minimal linear-algebra shim without breaking
the zero-dep invariant of the contract.

Cross-ref: `crates/lance-graph-contract/src/grammar/context_chain.rs`.

## 2026-04-19 — CausalityFlow has 3/9 TEKAMOLO slots; modal/local/instrument + beneficiary/goal/source deferred
**Status:** Open
**Priority:** P1
**Scope:** @integration-lead @truth-architect D1 domain:grammar
**Introduced by:** PR #208 + #210 (deliberate deferral)
**Payoff estimate:** 6 new `Option<String>` fields in
`lance-graph-cognitive/src/grammar/causality.rs` + tests

Full thematic-role inventory needs 6 more slots on `CausalityFlow`.
Deferred per user decision; D2 ticket emission and D3 triangle
bridge map only the 3 existing slots for now. Phase 2 work is
consistent with 3/9; Phase 3 may benefit from the extension.

Cross-ref: `grammar-landscape.md` §3, `STATUS_BOARD.md` D1 row.

## 2026-04-19 — Named-Entity pre-pass (NER) is the biggest OSINT blocker; stubbed out
**Status:** Open
**Introduced by:** architectural choice (all PRs)
**Payoff estimate:** dedicated PR, ~800 LOC new crate / subsystem

COCA 4096 has zero coverage of proper nouns (Altman, Anthropic,
Riyadh). Every unknown entity falls through to hash-bucket
collisions in the SPO graph. Grammar work proceeds without NER;
OSINT pipeline is blocked on this.

Cross-ref: `grammar-tiered-routing.md` §C8,
`STATUS_BOARD.md` Research threads section.

## 2026-04-19 — FP_WORDS = 157 (not 160); SIMD remainder loops remain
**Status:** Open
**Introduced by:** architectural choice (ndarray::hpc::vsa)
**Payoff estimate:** coordinated ndarray + lance-graph change,
~30 LOC in ndarray + 0 in lance-graph if field naming is stable

160 u64 = 10,240 bits (SIMD-clean for AVX-512 / AVX2 / NEON), zero
remainder loop in every SIMD pass. Current 157 u64 has 5-word
scalar tail. Performance delta is measurable but not critical yet.

Cross-ref: `cross-repo-harvest-2026-04-19.md` H6.

## 2026-04-19 — Abduction-threshold for unbundle-to-graph is hand-picked (0.88)
**Status:** Open
**Introduced by:** #208 (inherits PR design)
**Payoff estimate:** empirical calibration run on real corpus +
~30 LOC threshold parameterization

NARS Abduction confidence threshold for promoting facts into the
triplet graph is hand-picked at 0.88. Miscalibration on a specific
corpus (e.g. Animal Farm) would compound errors. Calibration is
pending D10 validation harness.

Cross-ref: `integration-plan-grammar-crystal-arigraph.md` E8,
`STATUS_BOARD.md` D10 row.

---

## Paid Debt

## 2026-04-25 — TD-INT-3/10/14 paid: MUL gate veto, NarsTables lookup, convergence highway (from 2026-04-24)
**Status:** Paid 2026-04-25
**Payoff:** Commit `0f9dcbb` on `claude/teleport-session-setup-wMZfb`

The three P1 wiring gaps that bring the second metacognitive layer online — meta-uncertainty veto, precomputed NARS truth lookup, and the cold→hot knowledge highway — are now wired.

- **TD-INT-3 (MUL gate veto):** `MulAssessment::compute(&SituationInput)` is a carrier method on the contract type (per "object speaks for itself" doctrine). In `driver.rs`, the gate decision builds a SituationInput from current dispatch state (felt_competence ← top_resonance, demonstrated ← `1 - F.total`, skill ← `awareness.recent_success.frequency`, challenge ← std_dev, environment_stability ← `1 - std_dev`), computes MulAssessment, then vetoes homeostatic Flow → Hold whenever MUL flags Mount-Stupid or Overconfident-trust-texture. The system can no longer commit confidently while metacognitively flagging the gap.
- **TD-INT-10 (NarsTables in cascade):** `causal_edge::tables::NarsTables` is a zero-dep crate `cognitive-shader-driver` already depends on, so no circular dep. ShaderDriver gains `nars_tables: Option<Arc<NarsTables>>` + a `with_nars_tables(Arc)` builder. Per cascade hit, when tables are attached, the system revises `(edge.frequency, edge.confidence)` against `(resonance, half_confidence)` via `tables.revise(...)`. Result currently observed only — tuning into the resonance formula is deferred. Call site established; the wiring debt is paid.
- **TD-INT-14 (convergence highway):** ShaderDriver.planes moved into `RwLock<Box<[[u64; 64]; 8]>>` so newly-committed AriGraph SPO knowledge can swap into the live cascade without restart. New `update_planes(&self, [[u64; 64]; 8])` takes the write lock and replaces in place. `dispatch()` reads under the read lock and snapshots so concurrent writes can't tear the topology mid-cycle. Planner-side `run_convergence(triplets, apply: impl FnOnce([[u64; 64]; 8]))` packages the conversion + closure handoff so `cognitive-shader-driver` doesn't need to depend on `lance-graph-planner` (would be circular). Call site: `run_convergence(&triplets, |p| driver.update_planes(p))`.

The cognitive loop now has every metacognitive layer wired: F drives the gate (TD-INT-1), NARS revises every cycle (TD-INT-2), MUL vetoes overconfidence (TD-INT-3), Markov braiding preserves order (TD-INT-4), NarsTables truth-revises per hit (TD-INT-10), and AriGraph commits flow into the cascade via convergence (TD-INT-14). Six P0/P1 dormant intelligence features paid in two days.

Cross-ref: TD-INT-3 / TD-INT-10 / TD-INT-14 original entries in the 2026-04-24 systemic-wiring-gaps log; commit 0f9dcbb.

## 2026-04-25 — TD-INT-1/2/4 paid: cognitive loop closes structurally every dispatch (from 2026-04-24)
**Status:** Paid 2026-04-25
**Payoff:** Commit `474d3eb` (TD-INT-1) + `b7787cf` (TD-INT-2 + TD-INT-4) on `claude/teleport-session-setup-wMZfb`

The three P0 wiring gaps (FreeEnergy compose, NARS revision per cycle, Markov trajectory braiding) are now wired into `cognitive-shader-driver/src/driver.rs`. Every dispatch cycle now executes: encode → Markov braid (positional XOR) → FreeEnergy::compose → Resolution gate → NARS revise → next cycle's F landscape changes accordingly.

- **TD-INT-1 (FreeEnergy gate):** Replaced `collapse_gate(std_dev)` heuristic with principled `FreeEnergy::compose(top_resonance, std_dev)`. Homeostatic F → Flow with `MergeMode::Bundle` (Markov-respecting per I-SUBSTRATE-MARKOV); catastrophic F → Block; epiphany (top-2 within EPIPHANY_MARGIN) → Hold; mid-band → Hold. `MetaSummary.meta_confidence = 1 - F.total` (principled) and `should_admit_ignorance = F.is_catastrophic()` replace the `1 - std_dev` and `confidence < 0.2` surrogates.
- **TD-INT-2 (NARS revision):** Added `awareness: RwLock<Vec<GrammarStyleAwareness>>` to ShaderDriver (12 entries indexed by shader ord). At end of `run()`, `free_energy_to_outcome(F, is_epiphany)` produces a ParseOutcome (LocalSuccess / LocalSuccessConfirmedByLLM / EscalatedButLLMAgreed / LocalFailureLLMSucceeded), which is then folded into `awareness[style_ord]` via `style_aw.revise(ParamKey::NarsPrimary(inference), outcome)`. Hot path stays zero-allocation; lock is brief (write only at end of cycle).
- **TD-INT-4 (Markov braiding, binary-space first step):** Replaced unordered XOR fold of content rows with positional XOR fold — each row's fingerprint is rotated by `cycle_index % WORDS_PER_FP` before XOR. Two cycles with identical hits in different order now produce different `cycle_fp`. This is the binary-space analogue of `vsa_permute + vsa_bundle`. **Deferred:** full f32 VSA bundle requires a Vsa16kF32 trajectory carrier alongside Binary16K — separate tracked debt.

What this means in the larger frame: the system no longer just describes cognition through types; it performs cognition every cycle. The `Think` struct from CLAUDE.md §The Click is now operationally instantiated by `ShaderDriver` — the awareness field is mutated, the F landscape changes, the next dispatch differs from the last. Concrete-operational → formal-operational, in Piaget's terms.

Cross-ref: original entries TD-INT-1 / TD-INT-2 / TD-INT-4 in the 2026-04-24 systemic-wiring-gaps log; CLAUDE.md §The Click; I-SUBSTRATE-MARKOV (Bundle merge mode); commits 474d3eb + b7787cf.

---

(No further debt paid at initial commit. When an Open entry is retired,
APPEND here with same title + PR anchor.)

```
## YYYY-MM-DD — <same title as Open entry> (from YYYY-MM-DD)
**Status:** Paid YYYY-MM-DD
**Payoff:** PR #NNN (commit SHA) — <one-line description>

<verbatim original Open paragraph>

Cross-ref: <same + PR link>
```

---

## How to use this file

**When shipping with a known shortcut** — prepend to **Open Debt**
with `**Status:** Open` + `**Introduced by:** PR #NNN` +
`**Payoff estimate:**`. One paragraph describing what's owed.

**When paying debt** — append to **Paid Debt** with the same title
+ date anchor + `**Status:** Paid YYYY-MM-DD` + `**Payoff:** PR
#NNN`. Flip the Open entry's Status to `Paid YYYY-MM-DD`.

**When debt becomes irrelevant** (e.g. the feature it blocked got
abandoned) — flip Open Status to `Moot YYYY-MM-DD`. Keep the row.

Nothing is lost. Every shortcut has a trail from introduction to
payoff (or abandonment).

## 2026-04-19 — VSA substrate renaming: Vsa10k* → Vsa16k* + float framing
**Status:** Open
**Priority:** P1
**Scope:** @container-architect @truth-architect D6 D0 domain:vsa domain:codec
**Introduced by:** architectural choice (all PRs referring to "10K VSA")
**Payoff estimate:** 1 contract-crate PR + 1 audit-sweep PR across
~28 `.claude/*.md` files (21 lance-graph + 7 ndarray).

The VSA substrate is misnamed and mis-scaled throughout the workspace:

1. **Naming:** `Vsa10kF32`, `Vsa10kI8` in
   `lance-graph-contract::crystal::CrystalFingerprint` should be
   `Vsa16kF32`, `Vsa16kI8`.
2. **Scale:** 10,000-D is a legacy narrower width; move to 16,384-D
   (64 KB lossless f32, 32 KB BF16, 80 KB u8-5-lane, 160 KB BF16-5-lane).
3. **Role-key slices:** `grammar::role_keys` addresses `[0..10000)`;
   re-scale to `[0..16384)` proportionally, keeping slices disjoint.
4. **Framing:** eliminate every occurrence of "10,000 binary VSA" — it
   collapses the 2 KB Hamming fingerprint (Container) with the
   ≥64 KB float VSA substrate. They are different objects.

Cross-ref: `.claude/board/IDEAS.md` CORRECTION-OF entry (2026-04-19).
Audit list of 28 files affected: `grep -rn "10000\|10,000\|Vsa10k\|10 000-D" .claude/`.

## 2026-04-19 — Ladybug 10000-D VSA import caused 700-1100 MB memory blowup
**Status:** Open
**Priority:** P1
**Scope:** @container-architect @integration-lead domain:vsa domain:memory
**Introduced by:** PRs #200-#203 (ladybug-rs / bighorn imports —
cognitive crate + CognitiveShader + BindSpace + adaptive codecs)
**Payoff estimate:** LanceDB zero-copy mmap migration + sparse/lazy
materialization audit + VSA scale decision (10k → 16k adds ~60% per
row: 40 KB → 64 KB at f32, worse memory not better).

Observed (per user, 2026-04-19): importing ladybug-rs at 10,000-D VSA
pushed runtime memory to **700-1,100 MB**. Arithmetic: at 40 KB/row
(Vsa10kF32), this is ~17-27 K live fingerprints in RAM
simultaneously. Pathologies:

1. **Migrating to 16,384-D makes it worse** — 64 KB/row × same
   population = 1.1-1.8 GB. 5-lane encoding (80 KB u8, 160 KB BF16) is
   worse still. Wider substrate without a storage-layer fix inflates
   the problem.

2. **Storage-layer fix is mandatory** — LanceDB `FixedSizeList<Float32,
   N>` natively supports mmap zero-copy. Hot rows stay OS-cached; cold
   rows pay page-fault but not RAM. Target runtime memory: ≤ 100 MB
   working set regardless of row count.

3. **Population audit** — before the 16k rename, audit which consumers
   keep VSA fingerprints live in RAM vs fetch-on-demand. Working set
   should be bounded by cache policy, not row count. Candidates for
   fetch-on-demand: Markov ±5 trajectory (rarely revisited), AriGraph
   episodic (LRU-evictable), VSA table snapshots.

4. **Sparse representation gap** — most VSA role-bundles have a few
   active slices at a time. A sparse encoding (indices-of-nonzero or
   Structured5x5 middle cells only) could drop per-row to a few KB for
   the common case. Only the "full field" queries need the dense
   f32 representation.

**Gate before 16k rename PR:** measure peak RAM on a representative
workload (Animal Farm D10 corpus when it lands) with the current 10k
substrate. If the fix is "mmap only hot rows", the substrate switch
is safe. If the fix requires sparse representation, the rename needs
to redesign the storage contract.

**Cross-ref:** IDEAS CORRECTION-OF + REFINEMENT-OF entries
(2026-04-19). PRs #200-#203 introduced the memory footprint.

## 2026-04-19 — CORRECTION-OF 2026-04-19 Ladybug 700-1100 MB blowup — it's a 10k × 10k GLITCH MATRIX, not HDC population
**Status:** Open
**Priority:** P0
**Scope:** @container-architect @integration-lead domain:memory domain:cleanup
**Introduced by:** PRs #200-#203 (ladybug-rs / bighorn imports — carried
over code that allocates a dense 10,000 × 10,000 structure)
**Payoff estimate:** identify + delete the single glitch allocation.
No migration, no redesign, no substrate change.

The prior entry framed the 700-1,100 MB blowup as a per-row HDC
population cost (17-27 K live fingerprints at 40 KB/row). **This was
wrong.** Per user (2026-04-19):

**There is no 10,000 × 10,000 matrix we actually want.** The memory
blowup came from a dense 10k × 10k structure imported as a glitch
from outdated ladybug-rs / bighorn code. Arithmetic:

- 10,000 × 10,000 × f32 = **400 MB** (single allocation)
- Plus cognitive-stack state + other working memory → 700-1,100 MB.

**Revised fix:**

1. **Identify the glitch** — grep ladybug-imported modules (cognitive
   crate, CognitiveShader, BindSpace, CollapseGate, adaptive codecs)
   for any dense `[[T; 10000]; 10000]`, `Vec<Vec<T>>` of that shape,
   `FixedSizeList<T, 100000000>`, or 400 MB-scale buffer. High-probability
   candidates: a token-token distance matrix, co-occurrence matrix, dense
   attention matrix, or K=10000 CLAM centroid distance table that was
   imported intact without trimming to the workspace's actual scale.
2. **Delete it.** It has no consumer in lance-graph proper (grammar /
   crystal / quantum use 1-D 10k-width HDC vectors, never square
   matrices).
3. **Verify** with peak-RAM measurement on a minimal workload.

**Invalidates:**

- Prior "16k rename makes memory worse" analysis — the per-row HDC
  math was sound but irrelevant to this specific blowup.
- Mmap zero-copy requirement for HDC population — still good hygiene
  but not the fix for the 700-1,100 MB observation.
- Sparse encoding as a Structured5x5-alternative — still architecturally
  useful but unrelated to the glitch.

The 16k rename + f32 → BF16 migration proceed independently of this
debt item. This one is a one-shot deletion.

## 2026-04-19 — SUPERSEDES all "stoneage import" / ladybug-refactor entries — retire ladybug-rs entirely
**Status:** CLOSED (via architectural decision)
**Scope:** @integration-lead @workspace-primer domain:architecture
**Decision:** per user (2026-04-19), ladybug-rs is retired. Migration
target is **ada-rs + lance-graph exclusively**. Ladybug-rs becomes
read-only historical reference; no maintenance, no refactor, no
integration obligation. Harvest-only.

**Consequences (ledger cleanup):**

- "Ladybug 700-1100 MB memory blowup" (glitch matrix): the matrix
  lives in ladybug-import code that is itself scheduled for removal.
  Deletion still P0 **only if** it ends up compiled into the current
  ada-rs / lance-graph binaries; otherwise it goes away with the
  import archival. Downgrade to P2, gate on "does cargo tree show the
  ladybug dep?"
- "Vsa10k → Vsa16k rename sweep" (2026-04-19): scope tightens to
  **ada-rs + lance-graph only**. Ladybug's internal types don't get
  renamed — they get left alone in the archive.
- "Ladybug import refactor resistance" table: obsolete. Don't refactor
  ladybug code; if a pattern is useful (PhaseTag, BindSpace,
  CognitiveShader, adaptive codec), reimplement cleanly in the target
  repo against canonical `Fingerprint<256>`.
- `CLAUDE.md` architecture diagram: "ladybug-rs = The Brain" line is
  stale; ada-rs + lance-graph now carry both the Brain and the Spine.
  Update in a follow-up PR.

**Repos in the canonical stack (post-retirement):**

```
ndarray            = The Foundation  (SIMD, GEMM, HPC, Fingerprint<256>, CAM-PQ)
lance-graph        = The Spine       (query + codec + semantics + contracts)
ada-rs             = The Brain       (BindSpace, SPO server, 4096 surface, cognitive shader)
crewai-rust        = The Agents      (agent orchestration, thinking styles)
n8n-rs             = The Orchestrator (workflow DAG, step routing)
```

(Was 5 repos; still 5, with ada-rs taking ladybug-rs's Brain slot.)

**Cross-ref:** prior entries "Ladybug 700-1100 MB memory blowup",
"Vsa10k* → Vsa16k* rename sweep", "CORRECTION-OF ... 10k × 10k GLITCH
MATRIX" — all carry this decision as their closing context.

## 2026-04-21 — D5 deepnsm files built on wrong VSA substrate (Frankenstein revert needed)
**Status:** Open
**Priority:** P0
**Scope:** @truth-architect @container-architect D5 domain:vsa domain:grammar
**Introduced by:** PR #243 (this session)
**Payoff estimate:** ~200 LOC rewrite — revert four files + reimplement on Vsa16kF32

Three deepnsm files and one contract addition built on `Vsa10k = [u64; 157]`
bitpacked binary + XOR algebra. Correct substrate is `Vsa16kF32 = Box<[f32;
16_384]>` (pending Vsa10k→Vsa16k coordinated rescale) with element-wise
multiply/add (`vsa_bind`/`vsa_bundle`/`vsa_cosine` that already exist in
`crystal/fingerprint.rs`).

Files to rewrite:

1. `crates/deepnsm/src/content_fp.rs` — produce `Box<[f32; 16_384]>`
   bipolar fingerprints (sign bit from SplitMix64), not `[u64; 157]`.
2. `crates/deepnsm/src/markov_bundle.rs` — use `vsa_bundle` (add) not
   `vsa_xor`. Braiding via `vsa_permute` on f32 indices, not bit rotation.
3. `crates/deepnsm/src/trajectory.rs` — `free_energy` likelihood term uses
   `vsa_cosine(unbind, expected)` per role, not Hamming recovery margin.
4. `crates/lance-graph-contract/src/grammar/role_keys.rs` — DELETE
   `RoleKey::bind/unbind/recovery_margin` (algebra belongs on carrier).
   DELETE `vsa_xor`, `vsa_similarity`. DELETE `Vsa10k` type alias.
   Keep role key static definitions but retype as `Box<[f32; 16_384]>`
   bipolar values (sign bit from FNV-seeded SplitMix64, ±1 in slice,
   zero elsewhere).

Dependency: blocks on the Vsa10k→Vsa16k coordinated rescale PR (ndarray
+ contract). Until that lands, interim fix is to use existing `Vsa10kF32`
(10,000 dims) without renaming.

Cross-ref: `.claude/knowledge/vsa-switchboard-architecture.md` (full
three-layer architecture), `EPIPHANIES.md` 2026-04-21 CORRECTION-OF entry,
audit agent report 2026-04-21.

## 2026-04-21 — Vsa10k→Vsa16k coordinated rescale (ndarray + lance-graph-contract)
**Status:** Open
**Priority:** P1
**Scope:** @container-architect @truth-architect D6 D0 domain:vsa domain:codec cross-repo:ndarray
**Introduced by:** architectural choice (originally flagged 2026-04-19 P1 debt, now coordinated two-repo scope)
**Payoff estimate:** 2 coordinated PRs (ndarray + contract) — ~40 LOC ndarray,
~200 LOC contract (rename + rescale + sandwich constants + passthrough funcs)

Rescale the float VSA carrier from 10,000 dims to 16,384 dims. This:
- Makes Binary16K ↔ Vsa16kF32 passthrough 1:1 (currently 16K bits vs 10K dims)
- Gives +60% capacity for orthogonal role superposition (Johnson-Lindenstrauss)
- Allows power-of-2 role slice boundaries (SUBJECT[0..4096), etc.)
- Aligns float VSA dim with Binary16K bit width

NOT a SIMD alignment issue for floats — both 10,000 and 16,384 f32 are
AVX-512-clean. The 157→160 u64 alignment issue was a BINARY-format-only
concern.

ndarray side: `VSA_DIMS = 16_384`, `VSA_WORDS = 256`, rescale VsaVector,
vsa_permute, vsa_bundle, VsaAccumulator.

contract side: rename `Vsa10kF32/I8` → `Vsa16kF32/I8`, rescale Box array
sizes, update sandwich constants (SANDWICH_TAIL_END = 16_384), update
passthrough funcs (to_vsa10k_f32 → to_vsa16k_f32, etc.), update role-key
slice boundaries proportionally.

Cross-ref: `cross-repo-harvest-2026-04-19.md` H6, prior tech debt entry
from 2026-04-19.

## 2026-04-21 — Jirak-derived thresholds probe (replace hand-tuned σ constants)
**Status:** Open
**Priority:** P1
**Scope:** @truth-architect @probe-runner D7 D10 domain:vsa domain:stats
**Introduced by:** CLAUDE.md `I-NOISE-FLOOR-JIRAK` iron rule
**Payoff estimate:** ~100 LOC in contract crate (jirak_threshold function)
+ one-shot calibration probe on Animal Farm to measure ρ

Hand-picked constants:
- `UNBUNDLE_HARDNESS_THRESHOLD = 0.8` (PR #208)
- `UNBUNDLE_ABDUCTION_THRESHOLD = 0.88` (this session, D7)
- `HOMEOSTASIS_FLOOR = 0.2`, `EPIPHANY_MARGIN = 0.05`, `FAILURE_CEILING = 0.8`
  (this session, free_energy.rs)

All hand-tuned. Should become functions of measured bit-level weak
dependence ρ per Jirak 2016 (arxiv 1606.01617).

Probe:
1. Measure ρ on actual Binary16K fingerprint distribution (autocorrelation
   of bit i with bit j across a representative corpus).
2. Implement `jirak_threshold(n: usize, rho: f32, confidence: f32) -> f32`.
3. Replace hand-picked constants with calls to this function.
4. First-run calibration: Animal Farm corpus for the grammar/coref pipeline.

Ship this BEFORE claiming the stack is "grounded." Currently it's
"tuned," which is defensible but must be documented honestly.

Cross-ref: CLAUDE.md `I-NOISE-FLOOR-JIRAK`, `.claude/jc/examples/prove_it.rs`
(3/5 pillars already run in 5s).

## 2026-04-21 — ONNX 16kbit story-arc learning (D9 deferred)
**Status:** Open
**Priority:** P2
**Scope:** @truth-architect @onnx-bridge D9 domain:arc domain:learning
**Introduced by:** `categorical-algebraic-inference-v1.md` D9 architectural
placeholder
**Payoff estimate:** new crate or bridge module (~300 LOC) + ONNX model
training pipeline (out of scope for lance-graph; upstream in thinking-engine)

The 3x16kbit Plane accumulator (ndarray `plane.rs`, shipped in
CROSS_REPO_AUDIT_2026_04_01.md) is the write-path for AriGraph edge
learning. It is NOT a VSA carrier — it's a saturating i8 accumulator.

D9 ONNX arc export should consume Vsa16kF32 trajectory fingerprints
(identity layer) and predict state transitions. The model is trained
on (state, arc_pressure, arc_derivative) tuples emitted per cycle.

Deferred until:
1. D5 rewrite (correct VSA substrate)
2. D8 AriGraph commit bridge
3. D10 Animal Farm benchmark running

Then ONNX training has a corpus to consume.

Cross-ref: `categorical-algebraic-inference-v1.md` §4 Next deferred,
ndarray `plane.rs`.

## 2026-04-21 — Callcenter / persona / archetype catalogues (speculative intent preservation)
**Status:** Open
**Priority:** P3 (tracked but low)
**Scope:** @host-glove-designer @truth-architect domain:persona domain:callcenter
**Introduced by:** architectural discussion 2026-04-21
**Payoff estimate:** deferred — not yet greenlit as deliverables

The three-layer VSA architecture explicitly supports callcenter intent
classification, persona-agent routing, and archetype-based character
inference. None of these are shipped features; they are SPECULATIVE
applications of the switchboard pattern.

Intent preservation (for future planning sessions):
1. Callcenter agents as persona-registry consumers (each agent role =
   one persona identity fingerprint).
2. Intent classification via VSA resonance (caller turn → role bundle
   → cosine-rank against intent codebook → dispatch).
3. Supabase as content layer (NOT VSA layer) — stores persona YAML,
   intent definitions, call transcripts, cumulative awareness. Supabase
   never stores VSA bundles (those are ephemeral compute state).
4. "BBB" references in prior conversations = speculative benchmarks
   (Better Business Bureau complaint handling), not shipped features.
5. Archetype catalogue unification with existing palette archetypes
   (256 per plane in bgz17), VoiceArchetype (16 channels in ndarray::
   hpc::audio), Glyph5B (5-byte archetype addressing).

Do NOT ship without explicit green-light. The architecture supports it;
the current scope doesn't include it. Preserve the framing for when
these become tracked deliverables.

Cross-ref: `.claude/knowledge/vsa-switchboard-architecture.md` § The
Archetype ↔ AriGraph ↔ Persona ↔ ThinkingStyle Unification.


## 2026-04-21 — L3 naming collision: CPU cache L3 vs cognitive-shader Layer 3
**Status:** Open
**Priority:** P2
**Scope:** @truth-architect @integration-lead domain:docs domain:cognitive-shader
**Introduced by:** architectural naming (pre-existing, flagged 2026-04-21)
**Payoff estimate:** ~40 LOC docs cleanup + grep pass over `.claude/*.md`

The 7-layer cognitive stack uses L0..L6 labels (L0 ndarray SIMD, L1
BindSpace, L2 CognitiveShader, L3 CollapseGate, L4 Planner, L5 GPU
meta, L6 LanceDB). This NAMING COLLIDES with CPU cache levels (L1/L2/
L3 caches).

The confusion point: when memory analysis says "Vsa16kF32 (64 KB)
fits in L3 cache" — does L3 mean cache level 3 (typical 8-32 MB, plenty
of room) or cognitive-shader Layer 3 (CollapseGate, a logical dispatch
layer unrelated to physical memory)?

The poisoning risk: documentation or session handovers that casually
reference "L3" may be read by a future session as either (a) CPU
cache level or (b) cognitive-shader layer, and the wrong reading can
lead to wrong design decisions (e.g., sizing the trajectory bundle to
"fit in L3" when "L3" was actually referring to the collapse gate).

**Fix:** Disambiguate in all future writing:
- CPU cache → "CPU-L3" or "L3-cache"
- Cognitive stack layer → "cog-L3" or "Layer 3 (CollapseGate)"
- Grep pass over existing `.claude/*.md` for "L3" with implicit
  context; annotate each occurrence.

Cross-ref: `CLAUDE.md § What This Is` (7-layer stack diagram),
this session's memory analysis discussion of Vsa16kF32 cache fit.


## 2026-04-21 — Lazy-VSA principle (reclassification of queued VSA work)

**Status:** Open (architectural guidance, applies to all VSA items below)
**Priority:** P1 (guidance)
**Scope:** @integration-lead all domain:vsa entries
**Introduced by:** user direction 2026-04-21 post-cleanup

**Principle:** VSA substrate work is PULLED IN by downstream deliverables,
not PUSHED OUT speculatively. Specifically:

1. **Vsa10k→Vsa16k coordinated rescale** — POSTPONED. Deserves its own
   dedicated test session. Cross-repo (ndarray + contract), non-trivial
   calibration impact, merits focused planning. Do NOT bundle with D5
   rewrite or any other in-flight work.

2. **D5 rewrite on Vsa10kF32 (current 10K)** — pull in ONLY IF a
   downstream deliverable (steps 4–8 wiring, D8 AriGraph bridge, D10
   Animal Farm benchmark) concretely needs the VSA trajectory and the
   Vsa16k rescale hasn't landed yet. Until that trigger fires,
   `Vsa10kF32` stays unused by the grammar/persona/callcenter
   consumers. The cleanup has reverted to pre-D5 state — that's the
   terminal state for this session.

3. **Jirak-derived thresholds probe** — pull in ONLY IF a downstream
   deliverable uses the thresholds (currently only `UNBUNDLE_HARDNESS`
   and `ABDUCTION_THRESHOLD` in shipped code; both hand-tuned are
   defensible). First consumer to need calibrated thresholds triggers
   the probe.

   **HOWEVER:** Jirak's ACTIVE role right now (NOT deferred) is the
   scientific framework for **CAM-PQ vs Vsa10k format decisions**.
   `FormatBestPractices.md § 1` quantifies the weak-dependence ρ that
   makes CAM-PQ bitpacked codes POOR candidates for VSA bundling
   (ρ ≈ 0.3–0.5 after centroid quantization; effective capacity drops
   proportionally) vs near-IID role-key-generated bits (ρ ≈ 0.01)
   where `Vsa10kF32` bundling retains full capacity. This is Jirak
   as **DECISION FRAMEWORK**, not Jirak as CALIBRATION PROBE. The
   decision framework is already shipped in the cleanup docs; the
   probe is what measures specific ρ per corpus to replace the
   hand-tuned thresholds when a downstream consumer demands it.

**Why lazy:** every VSA substrate touch has calibration ripple effects.
Doing them in the order "substrate first, then consumers" means calibrating
substrate changes without a real consumer to test against — which is how
the D5 Frankenstein happened. Inverting to "consumer first, pull substrate
when needed" grounds every substrate change in a concrete benchmark.

**Action implication:** next session doesn't start with "rewrite D5 on
Vsa10kF32." Next session starts with "what's the FIRST downstream
deliverable that forces a VSA substrate decision, and what does that
decision need?" If the answer is D10 Animal Farm, then D5+D8+D10 plan
together. If the answer is persona bank for callcenter, D5+persona
catalogue. Etc.

Cross-ref: `FormatBestPractices.md` § 0 (the 5-question checklist
before picking a format), CHANGELOG.md 2026-04-21 entries.


## 2026-04-24 — Ballista trigger threshold tuning (ADR 0001 mutable)

**Status:** Open
**Priority:** P3 (tracked, tuned post-benchmark)
**Scope:** @truth-architect @host-glove-designer ADR-0001 domain:stack
**Introduced by:** ADR 0001 Decision 2 (the only mutable lock)
**Payoff estimate:** 1 benchmark run + 1 ADR-amend commit

ADR 0001 locks Ballista activation to: "single-node query P99 latency
on Animal Farm OR callcenter hot path exceeds 1 second after reasonable
DataFusion optimization." The "1 second" threshold is a placeholder —
empirically tune after first benchmark runs. If 500ms is the right
number, amend ADR 0001's mutable section. If 2s is the right number,
same process.

Threshold amendment does NOT require a new ADR — it is the one field
Decision 2 explicitly leaves mutable.

Cross-ref: `.claude/adr/0001-archetype-transcode-stack.md` § Ballista
activation trigger.

## 2026-04-24 — Context enrichment API for external BBB consumers (ADR 0001 open question)

**Status:** Open
**Priority:** P2
**Scope:** @host-glove-designer @truth-architect domain:bbb domain:external
**Introduced by:** ADR 0001 Decision 3 addendum (OPEN question, not locked)
**Payoff estimate:** Future ADR + ~80 LOC types + BBB deny-list review

External consumers (dashboards, LLM routers, simulation monitors) may
need more than `CognitiveEventRow` scalar projection, but less than
full internal state (which is BBB-banned per `I-VSA-IDENTITIES` +
`external_membrane.rs`).

Candidate enrichment shapes (NOT decided):

- `EnrichedCognitiveRow` — `CognitiveEventRow` + Staunen magnitude +
  arc pressure + recent commit digest as `Fingerprint<256>`
- `TrajectorySummary` — hashed identities (no unbindable VSA) + scalar
  coherence metrics
- `BlackboardRoundDigest` — round ID + expert count + aggregate
  confidence, no per-expert detail

Governance: ANY enrichment must pass the BBB type-system gate in
`external_membrane.rs`. Additions to the `permit` list are ADR-worthy
changes, not ad-hoc edits. The BBB is the most load-bearing invariant
in the workspace.

Next step: identify first concrete external consumer, scope enrichment
shape for that consumer's needs, author ADR that extends `permit`
list and justifies each field.

Cross-ref: `.claude/adr/0001-archetype-transcode-stack.md` § Context
enrichment, `external_membrane.rs:10`, `CognitiveEventRow`.

## 2026-04-24 — Grok gRPC as first external-LLM A2A expert (observation, not locked)

**Status:** Open
**Priority:** P2
**Scope:** @host-glove-designer @adk-coordinator domain:a2a domain:external
**Introduced by:** ADR 0001 Decision 2 addendum (observation)
**Payoff estimate:** ~100 LOC Grok client + Blackboard entry mapping

xAI Grok exposes a gRPC API. The lab `grpc` feature gate on
`cognitive-shader-driver` already serves the transport layer Ballista
needs AND the transport layer Grok speaks. Grok-as-expert could be
deployed pre-Ballista:

```
Grok gRPC response → lab grpc handler → Blackboard expert entry
    (expert_id = "grok", capability = ..., result = ..., confidence, ...)
```

BBB rule holds: Grok receives scalar `CognitiveEventRow` projections
on outbound; the `BlackboardEntry` it fills is internal-only, never
exposed as external content. Grok becomes an EXPERT in the A2A sense,
not a FRONTEND consumer.

Not a commitment — recording that the wire is already shaped for it.
Deliverable trigger: when multi-LLM A2A becomes a tracked requirement.

Cross-ref: `.claude/adr/0001-archetype-transcode-stack.md` § Grok
gRPC addendum, `crates/cognitive-shader-driver/src/grpc.rs`,
`a2a_blackboard::BlackboardEntry`.


## 2026-04-24 — Context-syntax contract for cross-language queries (spine gap)
**Status:** Open
**Priority:** P1
**Scope:** @bus-compiler @integration-lead domain:planner domain:external-surface
**Introduced by:** architectural audit 2026-04-24 (two-SoA framing)
**Payoff estimate:** ~40 LOC contract type + documentation sweep across 16 planner strategies

Cypher / SQL / Gremlin / SPARQL / Redis-DN all parse into DataFusion LogicalPlan, but there is no first-class contract type declaring the SHARED COLUMN SURFACE that external query languages must reference through. Currently the marriage is implicit across the 16 strategies in `lance-graph-planner`.

Proposal: add a `SharedExternalSchema` type to `lance-graph-contract` that enumerates projected column names available to all external query languages, with enforcement at `PlannerContract` planning step. Without it, cross-language queries work by coincidence and each parser can drift into its own naming.

Blocks: parallel transcodes (callcenter + archetype) that open external query surfaces. Both would need to know the shared column names.

Cross-ref: `lance-graph-planner::strategy::*`; `lance-graph-contract/src/plan.rs`; epiphany 2026-04-24 "Context-syntax marriage."

## 2026-04-24 — External boundary as staging + projection columns (formalize into the global SoA)
**Status:** Open
**Priority:** P1
**Scope:** @host-glove-designer @truth-architect domain:bbb domain:soa
**Introduced by:** two-SoA architectural audit 2026-04-24
**Payoff estimate:** ~100 LOC contract types + BindSpace column additions + migration of existing ExternalMembrane impls

Today `ExternalMembrane` is a trait with method-based `ingest()` + `project()`. Architectural direction from 2026-04-24 audit: both crossings become explicit BindSpace columns (`StagingColumn`, `ProjectedRow`) so the data path is columnar and subject to the dual-ledger write discipline (CollapseGate).

BBB stays enforced by type system (staging column accepts only `ExternalEvent` shapes with no VSA / RoleKey / NarsTruth fields; projection exposes only scalar `CognitiveEventRow`). The BOUNDARY becomes visible in the SoA schema.

Cross-ref: `lance-graph-contract/src/external_membrane.rs`; I1 invariant; epiphany 2026-04-24 "External boundary formalized INTO the global SoA."

## 2026-04-24 — Grammar Markov ±5 / ±500 kernel as first-class BindSpace column layout
**Status:** Open
**Priority:** P1
**Scope:** @bus-compiler @ripple-architect domain:markov domain:soa
**Introduced by:** two-SoA audit + pyramid architecture session 2026-04-24
**Payoff estimate:** ~60 LOC column schema + documentation

Grammar Markov kernel (±5 local, ±500 paragraph, ±500000+ document-level per `elegant-herding-rocket-v1.md`) needs a documented first-class column layout in BindSpace. Candidates:
- Ring buffer column: `MarkovRingColumn<Binary16K, RING_SIZE>`
- Paired preceding/following: `fingerprints.preceding[]` + `fingerprints.focal[]` + `fingerprints.following[]`
- Positional offset per row: `fingerprints[row].markov_offset: i16`

Choose ONE, document byte-level, make it the canonical surface for all Markov-window operations. Without this, each shader worker (deepnsm grammar, coref resolution, NARS dispatch) reconstructs the window ad-hoc.

Cross-ref: `elegant-herding-rocket-v1.md` D5 markov_bundle (deferred as LAZY); `cognitive-shader-architecture.md` BindSpace columns; epiphany 2026-04-24 "TWO SoAs."

## 2026-04-24 — dn_redis unification via DataFusion streaming
**Status:** Open
**Priority:** P2
**Scope:** @host-glove-designer domain:external-surface domain:datafusion
**Introduced by:** two-SoA audit 2026-04-24
**Payoff estimate:** ~120 LOC adapter + Redis cache layer + deprecation of flat-KV surface

Current `crates/lance-graph-cognitive/src/container_bs/dn_redis.rs` uses flat `ada:dn:{hex}` Redis keys with subtree-scan operations. Per the two-SoA framing (external query SoA on DataFusion), this should be recast as DataFusion-served queries over Lance with Redis as an optional write-through cache.

The hierarchical DN path from `callcenter-membrane-v1.md` §595 (`/tree/ns/heel/h/hip/x/branch/b/twig/t/leaf/l`) is the natural DataFusion query shape: each path segment is a predicate on a Lance column. heel/hip/branch/twig/leaf are existing cascade-tree levels in `crates/lance-graph/src/graph/blasgraph/heel_hip_twig_leaf.rs`.

Deprecate the flat-key protocol over one migration cycle; retain Redis caching as acceleration layer on top of DataFusion queries.

Cross-ref: `container_bs/dn_redis.rs`; `callcenter-membrane-v1.md` §§595–803; `heel_hip_twig_leaf.rs`; epiphany 2026-04-24 "dn_redis is external."

## 2026-04-24 — Systemic wiring gaps: 14 dormant intelligence features

> **Frame:** Each item is an object-thinks-for-itself method that EXISTS
> but is not CALLED from the dispatch flow. Fix = add call site, not
> add type. All INTERNAL (hot path, inside BBB) unless marked BOUNDARY.
> No reductions proposed.

### TD-INT-1: FreeEnergy::compose() not called from dispatch
**What:** `FreeEnergy::compose(likelihood, kl)` in contract::grammar::free_energy.
**Where:** driver.rs after step [5], before CollapseGate. Replace `confidence < 0.2` heuristic with principled F.
**How:** `FreeEnergy::compose(top_k[0].resonance, awareness_kl)` then `Resolution::from_free_energy(F)`.
**Frame:** Internal | Functional (method on FreeEnergy carrier) | **P0**

### TD-INT-2: NARS revision not called per cycle
**What:** `awareness.revise_truth(key, outcome)` + `divergence_from(prior)` in grammar::thinking_styles.
**Where:** End of driver.rs::run(), after Resolution determined. Updates epistemic state, phi-1 ceiling.
**How:** `awareness.revise(style_key, resolution_outcome)`. Requires `&mut ParamTruths` on dispatch context.
**Frame:** Internal | Functional | **P0**

### TD-INT-3: MulAssessment not computed at dispatch time
**What:** `MulAssessment::compute(SituationInput)` in planner::mul -- DK position, trust texture, compass, homeostasis.
**Where:** Should compose with collapse_gate() in driver.rs. Currently two independent heuristics.
**How:** Build SituationInput from resonance + awareness. MUL can veto Flow to Hold if DK = unskilled-overconfident.
**Frame:** Internal | Functional | **P1** (metacognition)

### TD-INT-4: Trajectory braiding not in dispatch (Markov plus-minus-5)
**What:** trajectory.rs + markov_bundle.rs (PR #243) -- vsa_permute + vsa_bundle.
**Where:** driver.rs step [4] does XOR fold for cycle_fp. Should be VSA bundle with positional braiding.
**How:** Replace XOR fold: `vsa_permute(content_fp, position)` then `vsa_bundle(trajectory, permuted)`.
**Frame:** Internal | SoA storage + Functional algebra | **P0** (I-SUBSTRATE-MARKOV depends on this)

### TD-INT-5: RoleKey bind/unbind not used in content cascade
**What:** RoleKey::bind/unbind/recovery_margin in grammar::role_keys.
**Where:** Content Hamming cascade (PR #259) compares raw content via popcount(XOR).
**How:** Unbind by SUBJECT role key, compare subject-plane only via vsa_cosine instead of Hamming.
**Frame:** Internal | Functional | **P1** (upgrades bag-of-bits to role-indexed semantic similarity)

### TD-INT-6: ContextChain disambiguation not connected to route handler
**What:** ContextChain::disambiguate(WeightingKernel) in grammar/.
**Where:** CypherBridge (PR #258) is regex stub. When real parser returns N parse candidates, ContextChain picks best.
**How:** Build ContextChain from recent dispatch context. disambiguate(kernel) selects winner.
**Frame:** Internal | Functional | **P2** (activates when real Cypher parser is wired)

### TD-INT-7: Pearl 2-cubed causal mask not queried
**What:** CausalEdge64 packs Pearl 2-cubed (3 bits = 8 causal types) into every edge. Packed in dispatch step [6].
**Where:** No query path reads the mask. No "show me only direct causes" filter.
**How:** Add causal_type predicate to graph queries. Cypher WHERE should filter on mask bits.
**Frame:** Internal | SoA storage + Functional query | **P1**

### TD-INT-8: Schema validation not called on SPO commit
**What:** Schema::validate(&present) returns missing Required predicates. codec_route_for() per predicate.
**Where:** SPO commit path (Resolution::Commit to AriGraph). No validation runs today.
**How:** Before commit: schema.validate(present). If missing_required non-empty, emit FailureTicket instead of Commit.
**Frame:** Internal | Functional | **P1** (ontology exists but does not constrain)

### TD-INT-9: RBAC Policy not enforced at membrane projection
**What:** Policy::evaluate(role, entity, operation) returns Allow/Deny/Escalate.
**Where:** LanceMembrane::project() emits without checking RBAC. Any subscriber sees everything.
**How:** Before project() emits: policy.evaluate(actor_role, entity_type, Read{depth}). Skip on Deny.
**Frame:** BOUNDARY (membrane) | Functional | **P1**

### TD-INT-10: NarsTables (4096-head) not accessible from shader driver
**What:** nars_engine::NarsTables in planner::cache -- Pearl 2-cubed + 4096-head DK + Plasticity + Truth.
**Where:** ShaderDriver has no reference to NarsTables. Hot path does not use NARS lookup.
**How:** Pass &NarsTables to ShaderDriver. After cascade, look up NARS truth per hit SPO triple.
**Frame:** Internal | SoA (precomputed table) | **P1** (the 4096 surface the contract references)

### TD-INT-11: neural-debug runtime registry not populated
**What:** NeuronState enum + FunctionMeta + registry. WireHealth.neural_debug = None.
**Where:** health_handler hardcodes None. Runtime registry exists but is not fed by dispatch.
**How:** During run(), record row states (Alive/Static/NaN). Populate registry. health_handler reads it.
**Frame:** Internal | Functional | **P2** (diagnostic, not cognitive)

### TD-INT-12: DrainTask does not drain (Poll::Pending scaffold)
**What:** DrainTask in callcenter::drain returns Poll::Pending forever (PR #255).
**Where:** Should poll Lance for steering_intent rows then OrchestrationBridge::route().
**How:** Implement Future::poll() to scan, build UnifiedStep, route, mark drained.
**Frame:** BOUNDARY (outside-to-inside pump) | Functional | **P2**

### TD-INT-13: CommitFilter not applied server-side on project()
**What:** CommitFilter scalar predicates. Applied subscriber-side only today.
**Where:** LanceMembrane::project() emits all events unconditionally.
**How:** Apply filter inside project() before watcher.bump(row). Server-side predicate pushdown.
**Frame:** BOUNDARY | Functional | **P2**

### TD-INT-14: Convergence highway (AriGraph to p64 to CognitiveShader) not invoked
**What:** convergence.rs in planner::cache -- AriGraph triplets to p64 Palette to shader planes.
**Where:** No runtime invocation. Conversion functions exist but are not called.
**How:** On AriGraph commit, call convergence to update shader [[u64;64];8] planes. Newly committed knowledge reaches palette cascade distance table.
**Frame:** Internal | SoA planes + Functional conversion | **P1** (without this, palette cascade uses static demo planes forever)

### Summary by priority

| Priority | Items | What they activate |
|---|---|---|
| **P0** | TD-INT-1, 2, 4 | Active inference gate, NARS revision, Markov trajectory -- the cognitive loop |
| **P1** | TD-INT-3, 5, 7, 8, 9, 10, 14 | Metacognition, role-indexed similarity, causal queries, schema validation, RBAC enforcement, NARS lookup, convergence highway |
| **P2** | TD-INT-6, 11, 12, 13 | Disambiguation, neural-debug overlay, drain pump, server-side filter |

### Summary by frame

| Frame | Items |
|---|---|
| Internal hot path | TD-INT-1, 2, 3, 4, 5, 6, 7, 10, 14 |
| Boundary (membrane) | TD-INT-8, 9, 12, 13 |
| Diagnostic | TD-INT-11 |

All 14 items are additive (add call site). Zero items require type creation or code deletion.

## 2026-04-26 — TD-DIST-1: Distance trait missing from contract (type-intrinsic dispatch)

**Status:** Open
**Severity:** Medium (no runtime cost today — hard-coded dispatch works — but blocks
generic SoA distance sweeps)

The contract has `CodecRoute` (Passthrough | CamPq) naming the regime and
`DistanceTableProvider` for ADC, but no unified `Distance` trait that each
carrier type implements. Today each call site hard-codes which distance
function to use (`hamming_distance_raw` for Binary16K, `adc_distance` for
CamPq, `cosine_f64_simd` for Vsa16kF32). This works but prevents writing
generic distance sweeps over mixed SoA columns.

**Fix:** Add `pub trait Distance` to `contract::cam` (or a new `contract::distance`
module). Implement for `[u64; 256]`, `CamPqCode`, `PaletteEdge`, `Vsa16kF32`.
Include `similarity_z()` for FisherZ-transformed cosine averaging.
See EPIPHANIES.md 2026-04-26 distance-dispatch entry for full design.

**Blocked by:** nothing — pure additive.
**Unblocks:** generic SoA distance accumulation, multi-column weighted distance,
render-frame similarity for force-directed layout (CAM-PQ pruning + HHTL cascade).

## 2026-04-26 — TD-DIST-2: vector_ops.rs still has scalar dot/norm/cosine (4 loops)

**Status:** Open
**Severity:** High (hot path in DataFusion UDF — L2/cosine queries)

`vector_ops.rs` lines 140, 160, 179, 189 have 4 independent scalar
`.iter().map().sum()` loops for dot product, norm², cosine similarity.
Should swap for `ndarray::hpc::heel_f64x8::{dot_f64_simd, cosine_f64_simd}`.
Estimated 8-12× speedup (chunked F64x8 FMA vs scalar).

## 2026-04-26 — TD-DIST-3: bgz17 Palette::nearest() uses brute-force 256×17 L1

**Status:** Open
**Severity:** Medium (build-time hot path for palette construction)

`bgz17/palette.rs` lines 56-65 iterate all 256 centroids per query.
Should use precomputed distance table from `ndarray::hpc::palette_distance`.
Estimated 100× speedup for encoding (O(1) table lookup vs O(256) L1 per query).

## 2026-04-26 — Paid Debt: TD-DIST-1/2/3 all shipped in commit 8603148

- **TD-DIST-1** (Distance trait): `contract::distance` module with `Distance` trait,
  `fisher_z_inverse`, `mean_similarity_fisher`. Impls for `[u64; 256]`, `[u8; 6]`, `[u8; 3]`.
  11 tests. Status: **PAID**.
- **TD-DIST-2** (vector_ops scalar→SIMD): `cosine_distance`, `cosine_similarity`,
  `dot_product_distance`, `dot_product_similarity` all now delegate to
  `ndarray::hpc::heel_f64x8::cosine_f32_to_f64_simd` / `dot_f64_simd`. Status: **PAID**.
- **TD-DIST-3** (Palette distance table): `Palette::build_distance_table()` →
  `PaletteDistanceTable` with O(1) `distance(a, b)` and `edge_distance(a, b)`.
  128 KB table, L2-resident. Status: **PAID**.

## 2026-04-26 — TD-PALETTE-SENTINEL: 257th sentinel slot in palette distance/compose tables

**Status:** Open (low priority — historical aspirational design, no current need)

The 2026-04-20 resolution-hierarchy epiphany described the bgz17 HIP layer
as `256×257` (256 archetypes + 1 sentinel). Implementation shipped `k×k`
without the sentinel. See EPIPHANIES.md 2026-04-26 CORRECTION for full
context.

**Why deferred:**
- Adding a 257th index requires widening palette indices from `u8` to `u16`
- `PaletteEdge` wire format doubles from 3 bytes to 6 bytes per edge
- `MAX_PALETTE_SIZE = 256` is a deliberate u8-ceiling design choice
- The three sentinel roles (unknown/null/identity) are already covered by
  existing mechanisms: `Palette::nearest()` clamps unknowns, `identity()`
  returns the closest-to-zero archetype.

**Revisit when:** a real "absent edge" code path materializes (e.g., a
sparse mxm that needs to distinguish "no relation" from "relation = 0
distance"), or when the palette grows beyond 256 entries (which would
also force u16 indices).

## 2026-04-26 — TD-AWARENESS-INLINE-1: awareness should be BF16-mantissa-inline, not driver-global

**Status:** Open (P-0 architectural, scope: substrate-wide)

Per EPIPHANIES.md 2026-04-26 "awareness should be BF16-mantissa-inline":
the current `ShaderDriver.awareness: RwLock<Vec<GrammarStyleAwareness>>`
is driver-global and separate from the stream. This wastes the CPU's
20-200 ns random-access advantage and recreates the parser/processor
split that AGI is supposed to dissolve.

**The correct shape:** every stream operation returns `(value, awareness)`,
where awareness (7-8 bits, BF16-mantissa-equivalent) is derived inline
from operation properties (bit-purity, distribution shape, residual norm,
match strength). Awareness composes through the cascade the same way
values compose.

**Wedge for the smallest viable adoption:**
1. Extend `contract::distance::Distance` with
   `distance_with_awareness(&self, other) -> (u32, u8)`. 8 bits per
   measurement; 11% overhead vs raw distance.
2. Add `Aware` trait and `Annotated<T>` to contract.
3. Implement awareness derivation for the four primary operations:
   `vsa_bind`, `vsa_bundle`, `hamming`, `cosine`.
4. Update `ShaderDriver::dispatch` to compose inline awareness over
   the cascade. The driver-global `GrammarStyleAwareness` becomes a
   bootstrap seed, not the per-cycle source of truth.

**Size budget:** 11-12% overhead on stream payloads (vs 43.75% for
BF16 mantissa as a fraction of value), because the value plane is
much wider here than in floating-point.

**Why deferred:** scope is substrate-wide. Touches the contract
Distance trait (just shipped TD-DIST-1), every SIMD operation in
ndarray::hpc, the shader driver's cascade, and the BindSpace SoA.
Should be designed as one coherent commit, not piecemeal.

**Revisit when:** the next architectural sweep covers the awareness
dimension. Until then, awareness stays driver-global. The epiphany
documents the correct direction so future work doesn't re-derive it.

---

## 2026-05-05 — Tech-debt items extracted from PRs #244–#335

> Items below are ONLY those the PR author EXPLICITLY named as debt, deferred work, known limitation, TODO, stub, or "not yet wired". No inference. Each item cites the PR where the author flagged it.

---

### TD-F10-ACTOR-ID — actor_id semantic fix deferred (PR #274)

**Status:** Open
**Priority:** P2
**Scope:** crate:lance-graph-callcenter domain:callcenter
**Introduced by:** PR #274 (F-10 finding explicitly deferred)
**Author's words:** "F-10 · HIGH · actor_id = expert as u64 (DEFERRED). Semantic fix requires adding `actor_id: ActorId` to `ExternalIntent` and plumbing through `ingest()`. Deferred to avoid schema change in this PR — should be its own commit with downstream coordination."

---

### TD-ARROW-58 — Arrow 58 blocked until lance 5+ (PR #273, #275)

**Status:** Open
**Priority:** P2
**Scope:** crate:lance-graph domain:deps
**Introduced by:** PR #273 (documented as "Cannot bump"), confirmed in #275
**Author's words:** "Cannot bump. Both lance 4.0.1 and deltalake 0.31 pin `arrow = "^57"`. Arrow 58 needs lance 5+ (not released). Documented in TD-LANCE-UPGRADE."

---

### TD-ENTITY-TYPE-ID-CACHE — entity_type_id() O(N) linear scan (PR #272)

**Status:** Open
**Priority:** P3
**Scope:** crate:lance-graph-contract domain:ontology
**Introduced by:** PR #272
**Author's words:** "`entity_type_id()` does a linear scan of `Ontology.schemas` — O(N) per lookup. Fine for N < 100 schemas, but if someone has 1000+ entity types this becomes a problem. Should be a `HashMap<&str, EntityTypeId>` cache on `Ontology`. Not worth optimizing now (N is ~10 for SMB), but flagged."

---

### TD-COLUMN-H-DISPATCH — Column H entity_type not written at dispatch time (PR #272)

**Status:** Open
**Priority:** P2
**Scope:** crate:cognitive-shader-driver domain:bindspace
**Introduced by:** PR #272
**Author's words:** "The `dispatch()` step that writes entity_type into the SoA (D-H3 in the plan) is NOT wired yet — this PR adds the FIELD but not the dispatch-time write. That's Phase 2 territory because it requires knowing which `OntologySpec` the current triplet matches, which is the novel-pattern-detection logic in D-E3."

---

### TD-CLIPPY-SHADER-DRIVER — no clippy gate on cognitive-shader-driver (PR #272)

**Status:** Open
**Priority:** P3
**Scope:** crate:cognitive-shader-driver domain:ci
**Introduced by:** PR #272
**Author's words:** "No clippy gate on the shader-driver crate (only contract is gated). The shader-driver has pre-existing clippy debt from the parallel agents."

---

### TD-BINDSPACE-COLUMNS-EFG — Column E/F/G phases not implemented (PR #271)

**Status:** Open
**Priority:** P2
**Scope:** crate:cognitive-shader-driver domain:bindspace
**Introduced by:** PR #271 (plan phase 2/3/4 explicitly deferred)
**Author's words:** "Phase 3 (Column F) needs a proof-of-concept measuring whether inline awareness actually improves meta_confidence vs the current `1 - F.total` before committing to the full 9-deliverable plan. Phase 4 (Column G) is blocked [on LF-50/52] and speculative."

---

### TD-GRAMMAR-ROTATE-RIGHT-FUTURE — post-bundle permute not yet implemented (PR #282)

**Status:** Open
**Priority:** P3
**Scope:** crate:deepnsm domain:grammar
**Introduced by:** PR #282
**Author's words:** "`rotate_right` removed. Documented for future per-sentence pre-bundle permute." (rotation removed as it corrupted role-slice alignment; the correct per-sentence-pre-bundle permute is a future item)

---

### TD-POSTGREST-EDGE-CASES — PostgREST filter parsing edge cases not tested (PR #278)

**Status:** Open
**Priority:** P2
**Scope:** crate:lance-graph-callcenter domain:postgrest
**Introduced by:** PR #278
**Author's words:** "[ ] PostgREST filter parsing edge cases (nested paths, unicode table names)"

---

### TD-RLS-MULTI-TABLE-JOIN — RLS predicate injection on multi-table JOINs not manually reviewed (PR #278)

**Status:** Open
**Priority:** P2
**Scope:** crate:lance-graph-callcenter domain:rls
**Introduced by:** PR #278
**Author's words:** "[ ] Manual review of RLS predicate injection on multi-table JOINs"

---

### TD-GRAMMAR-UNICODE-RESTORE — ASCII fallback in grammar-landscape.md needs unicode restore (PR #279)

**Status:** Open
**Priority:** P3
**Scope:** crate:deepnsm doc:grammar-landscape.md
**Introduced by:** PR #279
**Author's words:** "[ ] ASCII→unicode restore on grammar-landscape.md (Finnish ä/ö, Cyrillic, Japanese particles)"

---

### TD-GRAMMAR-MEXICANHAT-VERIFY — WeightingKernel::MexicanHat zero-crossing not verified (PR #279)

**Status:** Open
**Priority:** P2
**Scope:** crate:deepnsm domain:grammar
**Introduced by:** PR #279
**Author's words:** "[ ] Verify WeightingKernel::MexicanHat zero-crossing matches Ricker wavelet"

---

### TD-SIGMA-CODEBOOK-PRODUCTION — Σ-codebook viability probe used synthetic not production data (PR #288)

**Status:** Open
**Priority:** P2
**Scope:** crate:jc domain:sigma-codebook
**Introduced by:** PR #288
**Author's words:** "Synthesised distribution is plausible, not from Production measured — with real stream run again to confirm. R² = 0.9949 is knapp über Threshold; für >5-Hop-Multi-Hop-Queries kann der kumulierte Fehler relevant werden."

---

### TD-TRANSCODE-PARALLELBETRIEB — parallelbetrieb is explicitly a transitional bandaid (PR #309)

**Status:** Open
**Priority:** P2
**Scope:** crate:lance-graph-callcenter domain:transcode
**Introduced by:** PR #309
**Author's words:** "The bandaid framing is for the parallel-evaluation overhead, not for the witness itself. Even at F5 the reconciler stays — MySQL is permanent — but its mode shifts from 'consensus required for any commit' to 'background witness that emits drift events when something diverges'."

---

### TD-TRANSCODE-PHASE4-NARS-SINK — Phase 4 NARS cold sink not covered (PR #310)

**Status:** Open
**Priority:** P2
**Scope:** crate:lance-graph-callcenter domain:transcode
**Introduced by:** PR #310
**Author's words:** "Phase 4 (NARS cold sink) — not covered. Continues to be a future PR aligned with `.claude/plans/sql-spo-ontology-bridge-v1.md`."

---

### TD-TRANSCODE-PHASE5-BINDSPACE-TO-DTO — BindSpace → outer-DTO reverse path not wired (PR #310)

**Status:** Open
**Priority:** P2
**Scope:** crate:lance-graph-callcenter domain:transcode
**Introduced by:** PR #310
**Author's words:** "Phase 5 (BindSpace → outer-DTO direction) — `zerocopy.rs` still only goes outer → Arrow. The reverse path (BindSpace columns → external SoA batch) needs the producer-side accessor in `cognitive-shader-driver` first."

---

### TD-TRANSCODE-PHASE2B-SPOSTORE — Phase-2-B SpoStore scan not implemented (PR #312)

**Status:** Open
**Priority:** P2
**Scope:** crate:lance-graph-callcenter domain:transcode
**Introduced by:** PR #312
**Author's words:** "Replace `MemTable::scan` delegate with a custom `ExecutionPlan` walking `SpoStore`'s `query_forward` / `query_reverse` / `query_relation` according to the `SpoLookup` shape. Flip recognised filters from `Inexact` → `Exact` once that scan is trusted to strict-enforce."

---

### TD-TRANSCODE-ROUND3-TYPED-DEFERRED — several SemanticType→Arrow conversions deferred (PR #316)

**Status:** Open
**Priority:** P3
**Scope:** crate:lance-graph-callcenter domain:transcode
**Introduced by:** PR #316
**Author's words:** "`Date(Month)` / `Date(Year)` precisions — today only `YYYY-MM-DD` parses; round-4 plumbs the precision into the parser. `Geo` / `File` / `Image` typed reconstruction — round-4 candidates. Async resolver — round-5. `FixedSizeListF32` / `FixedSizeBinary` single-bytes resolver — round-5 wide-payload resolver."

---

### TD-SIGMA-B3-CODEBOOK-BOOT — Σ codebook static + boot-load not yet wired (PR #323)

**Status:** Open
**Priority:** P2
**Scope:** crate:cognitive-shader-driver domain:sigma
**Introduced by:** PR #323
**Author's words:** "The Σ codebook itself is not loaded here — that is a B3 concern. The codebook static + boot-load-from-disk lives in `lance-graph-contract::sigma_propagation`. This PR only allocates the per-row index column."

---

### TD-SIGMA-B4-DISPATCH — Σ-propagate in shader-driver dispatch stage not wired (PR #323, #322)

**Status:** Open
**Priority:** P2
**Scope:** crate:cognitive-shader-driver domain:sigma
**Introduced by:** PR #322 (explicit B4 follow-up item)
**Author's words:** "B4 shader-driver Σ-propagate (later PR): in `ShaderDriver::dispatch()` between [5] edge emission and [6] FreeEnergy gate, propagate `sigma_path = ewa_sandwich(...)` along the resonance chain. Reject cycles whose `log_norm_growth` exceeds `pillar_5plus_bound`."

---

### TD-FNV-COPIES-THINKING-HOLOGRAPH — 2 FNV-1a copies remain in thinking-engine + holograph (PR #307)

**Status:** Open
**Priority:** P3
**Scope:** crate:thinking-engine crate:holograph domain:dedup
**Introduced by:** PR #307
**Author's words:** "2 copies remain in `thinking-engine` and `holograph` (don't depend on contract) — annotated."

---

### TD-CONTRACT-TEST-COVERAGE-CI — lance-graph-contract tests not in CI gate until PR #328 (PR #326)

**Status:** Paid 2026-05-01 (PR #328)
**Priority:** P2
**Scope:** crate:lance-graph-contract domain:ci
**Introduced by:** PR #322 (latent); surfaced in PR #326
**Author's words (PR #326):** "`crates/lance-graph-contract` does have a gating clippy job, but the workspace test job runs `cargo test` against `crates/lance-graph` only — `lance-graph-contract`'s tests don't fail any CI gate today."
**Payoff:** PR #328 added `cargo test --manifest-path crates/lance-graph-contract/Cargo.toml --lib` to CI.

---

### TD-BGZ-TENSOR-5-FAILURES-330 — 5 bgz-tensor platform-sensitive size-assertion test failures (PR #330)

**Status:** Open
**Priority:** P2
**Scope:** crate:bgz-tensor domain:ci
**Introduced by:** PR #308 (workspace inclusion); re-confirmed in PR #330
**Author's words (PR #330):** "bgz-tensor pre-existing failures: 5 platform-sensitive size-assertion tests that are not in the CI gate."

---

### TD-FMT-STANDALONE-CRATES-4400 — ~4400 rustfmt drift hunks in excluded/standalone crates (PR #329)

**Status:** Open
**Priority:** P3
**Scope:** domain:rustfmt crate:thinking-engine crate:holograph crate:cognitive-shader-driver crate:bgz17 crate:deepnsm crate:jc (and others)
**Introduced by:** PR #329 (audit surfaced)
**Author's words:** "Most 'drift' in the standalones is intentional author style (single-line `if`s, visually-aligned struct literals, two-space-comment alignment). No CI gate exists to lock the canonical style. Two viable follow-up paths: Path A (per-crate rustfmt.toml overrides) or Path B (mass-rewrite + CI gate for every crate)."

