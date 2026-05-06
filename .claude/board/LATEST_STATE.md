# LATEST_STATE — What Just Shipped (read this FIRST)

> **Auto-injected at session start via SessionStart hook.**
> Updated after every merged PR.
> **Last updated:** 2026-04-21 post PR #243 (D5+D7 + categorical-algebraic inference architecture).
>
> Purpose: prevent new sessions from hallucinating structure that
> already exists or proposing features already shipped. Read this
> BEFORE proposing any grammar/crystal/contract changes.

---

## Recently Shipped PRs (reverse chronological)

| PR | Merged | Title | What it added |
|---|---|---|---|
| **#243** | *(open)* | D5+D7 categorical-algebraic inference | `thinking_styles.rs` (490 LOC, 12 tests), `free_energy.rs` (347 LOC, 7 tests), `role_keys.rs` bind/unbind/recovery (295 LOC, 14 tests), `content_fp.rs` (98 LOC, 5 tests), `markov_bundle.rs` (250 LOC, 8 tests), `trajectory.rs` (298 LOC, 4 tests). Plans: `categorical-algebraic-inference-v1.md` (496 lines). Knowledge: `paper-landscape-grammar-parsing.md`, `session-2026-04-21-categorical-click.md`. CLAUDE.md § The Click (P-1). 12 epiphanies. |
| **#225** | *(open)* | Codec-sweep plan + D0.6/D0.7 CodecParams | 9-commit plan (`codec-sweep-via-lab-infra-v1.md`, Rules A-F, 9 starter YAMLs, CODING_PRACTICES audit) + `lance-graph-contract::cam` CodecParams/Builder/precision-ladder validation (14 tests). 147/147 contract suite |
| **#224** | 2026-04-20 | lab = API+Planner+JIT, thinking harvest, I11 measurability | `lab-vs-canonical-surface.md` extended: three-part lab stack (API + Planner + JIT), thinking-harvest subsection (REST/Cypher → `{rows, thinking_trace}` = the AGI magic bullet), I11 invariant (every layer L0→L4 emits harvest-ready trace; no black-box short-circuits) |
| **#223** | 2026-04-20 | LAB-ONLY firewall + AGI-as-SoA + I1-I10 | `lab-vs-canonical-surface.md` initial doc: canonical consumer = `UnifiedStep`/`OrchestrationBridge`, Wire DTOs are lab quarantine. AGI = (topic, angle, thinking, planner) = struct-of-arrays consuming cognitive-shader-driver. 10 cross-cutting invariants I1-I10 (BindSpace read-only, canonical `simd::*` import, temporal budgets, temperature hierarchy, thinking IS AdjacencyStore, weights are seeds, per-cycle cascade, 4096 surface, three DTO families, HEEL/HIP/BRANCH/TWIG/LEAF) |
| **#210** | 2026-04-19 | Phase 1 grammar + knowledge docs | ContextChain reasoning ops, role_keys slice catalogue, 3 knowledge docs (grammar-landscape, linguistic-epiphanies E13-E27, fractal-codec) |
| **#209** | 2026-04-19 | sandwich layout + bipolar cells | Crystal fingerprint sandwich, VSA_permute reference, lossless bundling corrections |
| **#208** | 2026-04-19 | grammar + crystal + AriGraph unbundle | Contract grammar/ + crystal/ modules, AriGraph episodic unbundle hooks with SIMD dispatch |
| **#206** | 2026-04-18 | state classification pillars | qualia.rs (17D), proprioception.rs (7 anchors), world_map.rs, sigma_rosetta 64 glyphs + 144 verbs, Pumpkin NPC example |
| **#205** | 2026-04-18 | engine bridge + CMYK/RGB qualia | engine_bridge.rs, 12-style unified mapping, 17D vs 18D qualia distinction |
| **#204** | 2026-04-18 | cognitive-shader-driver | New crate, ShaderDispatch/Resonance/Bus/Crystal DTOs, BindSpace struct-of-arrays, full ladybug-rs import |

## Current Contract Inventory (lance-graph-contract)

Types that EXIST — do NOT re-propose them:

**`grammar/`**: `FailureTicket`, `PartialParse`, `CausalAmbiguity`, `TekamoloSlots`, `TekamoloSlot`, `WechselAmbiguity`, `WechselRole`, `FinnishCase`, `finnish_case_for_suffix`, `NarsInference`, `inference_to_style_cluster`, `ContextChain` (with coherence_at / total_coherence / replay_with_alternative / disambiguate / DisambiguationResult / WeightingKernel), `RoleKey` + 47 `LazyLock<RoleKey>` instances + `Tense` enum + `finnish_case_key / tense_key / nars_inference_key` lookups, **`RoleKey::bind/unbind/recovery_margin`** (slice-masked XOR), **`Vsa10k`** + `VSA_ZERO` + `vsa_xor` + `vsa_similarity`, **`GrammarStyleConfig`** + **`GrammarStyleAwareness`** + `revise_truth` + `ParseOutcome` + `divergence_from`, **`FreeEnergy`** + **`Hypothesis`** + **`Resolution`** (Commit / Epiphany / FailureTicket) + `from_ranked` + thresholds.

**`crystal/`**: `Crystal` trait, `CrystalKind`, `TruthValue`, `UNBUNDLE_HARDNESS_THRESHOLD = 0.8`, `CrystalFingerprint` (Binary16K / Structured5x5 / Vsa10kI8 / Vsa10kF32 / **Vsa16kF32**), `Structured5x5`, `Quorum5D`, `SentenceCrystal`, `ContextCrystal`, `DocumentCrystal`, `CycleCrystal`, `SessionCrystal`, sandwich layout constants, **`vsa16k_zero` / `binary16k_to_vsa16k_bipolar` / `vsa16k_to_binary16k_threshold` / `vsa16k_bind` / `vsa16k_bundle` / `vsa16k_cosine`** (Click switchboard carrier + algebra, 64 KB, inside-BBB only).

**`cognitive_shader`**: `ShaderDispatch`, `ShaderResonance`, `ShaderBus`, `ShaderCrystal`, `MetaWord` (u32 packed), `MetaFilter`, `ColumnWindow`, `StyleSelector`, `RungLevel`, `EmitMode`, `ShaderSink` trait, `CognitiveShaderDriver` trait.

**`cognitive-shader-driver` BindSpace substrate (2026-04-24)**: `FingerprintColumns.cycle` is now `Box<[f32]>` (Vsa16kF32 carrier, 16_384 f32 per row = 64 KB) — migrated from `Box<[u64]>` (Binary16K). New constant `FLOATS_PER_VSA = 16_384`. New methods: `set_cycle(&[f32])`, `set_cycle_from_bits(&[u64; 256])` (adapter with `binary16k_to_vsa16k_bipolar` projection), `cycle_row() -> &[f32]`. `write_cycle_fingerprint()` API unchanged (takes `&[u64; 256]`), converts internally. `byte_footprint()` for 1 row = 71_774 bytes. Other three planes (content/topic/angle) remain `Box<[u64]>`.

## cognitive-shader-driver Wire Surface (lab-only, post D0.1)

Types live in `crates/cognitive-shader-driver/src/wire.rs` behind `--features serve`:

- **`WireCodecParams`** + `WireLaneWidth {F32x16, U8x64, F64x8, BF16x32}` + `WireDistance {AdcU8, AdcI8}` + `WireRotation {Identity, Hadamard{dim}, Opq{matrix_blob_id, dim}}` + `WireResidualSpec {depth, centroids}` — serde mirrors of the `contract::cam::*` types from PR #225. `TryFrom<WireCodecParams> for CodecParams` runs the precision-ladder validation (OPQ↔BF16x32, overfit guard, pow2 Hadamard) at ingress BEFORE any JIT compile.
- **`WireTensorView {shape, lane_width, bytes_base64}`** + methods `row(&AlignedBytes, usize)` / `subspace(&AlignedBytes, row, k, sub_bytes)` / `row_count()` / `col_count()` / `row_bytes()` / `element_bytes()` / `decode() -> AlignedBytes`. Per Rule E (Wire surface IS the SIMD surface, object-oriented) + Rule A (stdlib `slice::array_windows::<N>` + `ndarray::simd::*` loaders).
- **`AlignedBytes`** — heap-allocated, 64-byte-aligned owned buffer produced once by `WireTensorView::decode` per Rule F (decode at REST ingress, never inside). Safe Send/Sync; `Drop` dealloc with matching layout.
- **`WireCalibrateRequest`** extended with optional `params: Option<WireCodecParams>` + `tensor_view: Option<WireTensorView>` (new path) alongside legacy fields (`num_subspaces` / `num_centroids` / `kmeans_iterations` / `max_rows`) for back-compat.
- **`WireCalibrateResponse`** extended with `kernel_hash: u64` (= `CodecParams::kernel_signature()` of the executed kernel) + `compile_time_us: u64` + `backend: String` ("amx" | "vnni" | "avx512" | "avx2" | "legacy"; **never "scalar"** — iron rule).
- **`WireTensorViewError {Base64, SizeMismatch, ZeroShape}`** — typed decode errors.

**`proprioception`**: 7 `StateAnchor` (Intake/Focused/Rest/Flow/Observer/Balanced/Baseline), 11-D `ProprioceptionAxes`, `StateClassifier` trait, `DefaultClassifier`, `hydrate()` softmax-weighted blend.

**`qualia`**: 17-D `QualiaVector`, `qualia_to_state` projection (17→11).

**`world_map`**: `WorldMapDto`, `WorldMapRenderer` trait, `DefaultRenderer`.

**`world_model`**: `WorldModelDto` with `qualia`, `axes`, `proprioception`, `cycle_fingerprint`, `timestamp`, `cycle_index`, `is_self_recognised()` / `is_liminal()`.

**`container`**: `Container = [u64; 256]` (16Kbit = 2KB), `CogRecord`.

**`property`** (new, SMB domain): `PropertyKind` (Required / Optional / Free), `PropertySpec` (predicate + kind + `CodecRoute` + NARS floor), `PropertySchema` (`&'static`-based, const schemas), `Schema` + `SchemaBuilder` (runtime builder: `.required()` / `.optional()` / `.searchable()` / `.free()` / `.validate()`), `CUSTOMER_SCHEMA`, `INVOICE_SCHEMA`. Maps bardioc Required/Optional/Free to I1 Codec Regime Split (ADR-0002).

**`repository`** (new, SMB domain): `EntityStore` + `EntityWriter` + `Batch` + `EntityKey` — Arrow-agnostic row store contract.

**`mail`** (new, SMB domain): `MailParser` + `ThreadLinker` + `ParseHints` + `AttachmentRef` + `PartRef`.

**`ocr`** (new, SMB domain): `OcrProvider` + `PageImage` + `OcrOpts` + `Bbox` + `BlockKind` + `LayoutBlock`.

**`tax`** (new, SMB domain): `TaxEngine` + `TaxPeriod` + `PeriodKind` + `Jurisdiction` + `PostingBatchRef` + `RuleBundle`.

**`reasoning`** (new, SMB domain): `Reasoner` + `ReasoningKind` + `ReasoningContext` + `EvidenceRef` + `Budget`.

**`cam`** (extended by PR #225): `CodecRoute` + `route_tensor` (existing), `CamByte`, `CamStrategy`, `DistanceTableProvider` trait, `CamCodecContract` trait, `IvfContract` trait, plus codec-sweep parameter shape — `LaneWidth` (F32x16 / U8x64 / F64x8 / BF16x32), `Distance` (AdcU8 / AdcI8), `Rotation` (Identity / Hadamard{dim} / Opq{matrix_blob_id, dim}), `ResidualSpec {depth, centroids}`, `CodecParams {subspaces, centroids, residual, lane_width, pre_rotation, distance, calibration_rows, measurement_rows, seed}` with `kernel_signature() -> u64` + `is_matmul_heavy() -> bool`, `CodecParamsBuilder` fluent API, `CodecParamsError {ZeroDimension, OpqRequiresBf16, HadamardDimNotPow2, CalibrationEqualsMeasurement}` — **precision-ladder validation fires at `.build()` BEFORE any JIT compile**.

**`graph_render`** (new, q2 cockpit): `RenderNode`, `RenderEdge`, `InferredConnection`, `Contradiction`, `GraphSnapshot`, `GraphHealth`, `CypherResult`, `CypherValue`, `CypherError`, `EpisodicTrace`, `ShaderEvent`, traits `GraphSnapshotProvider`, `GraphInferenceProvider`, `CypherExecutor`, `EpisodicTraceProvider`, `ShaderEventStream`. Visual render surface for Neo4j/Palantir Gotham cockpit — q2 consumes, lance-graph arigraph produces.

**`a2a_blackboard`**, **`collapse_gate`**, **`exploration`**, **`literal_graph`**, **`orchestration_mode`**, **`jit`**, **`nars`**, **`plan`**, **`orchestration`**, **`thinking`** (36 styles, 6 clusters), **`mul`**, **`sensorium`**, **`high_heel`**.

## Current AriGraph Inventory (lance-graph/src/graph/arigraph/)

4696 LOC shipped, 7 modules:
- `episodic.rs` (210 LOC + unbundle hooks from #208) — `Episode`, `EpisodicMemory`, `unbundle_hardened`, `unbundle_targeted`, `rebundle_cold`, `UnbundleReport`, `RebundleReport`, `UNBUNDLE_HARDNESS_THRESHOLD = 0.8`
- `triplet_graph.rs` (1064 LOC) — SPO graph, NARS truth, BFS, spatial paths
- `retrieval.rs` (447 LOC) — fingerprint retrieval policies
- `sensorium.rs` (539 LOC) — observation → triplets
- `orchestrator.rs` (1562 LOC) — AriGraph coordinator
- `xai_client.rs` (521 LOC) — xAI enrichment
- `language.rs` (339 LOC) — LM bridge

## Workspace Conventions (locked in CLAUDE.md)

1. **Model policy:** main thread Opus + deep thinking; subagent grindwork → Sonnet; accumulation → Opus; NEVER Haiku.
2. **GitHub reads:** zipball to `/tmp/sources/` + local grep for 3+ reads per repo. MCP only for writes (PR, comments) and single-path reads.
3. **Contract zero-dep invariant:** `lance-graph-contract` has no external crate deps. Do not add any.
4. **Read before Write:** always Read a file before overwriting. Write-over-self without Read is the documented failure mode.
5. **No JSON serialization in types.** Serde stays out of types (debug-only). Wire formats are explicit.
6. **Pumpkin framing** for externally-visible examples (clinical / game-AI disguise for the AGI primitives).

## Active Branches (local at /home/user/lance-graph)

- `claude/teleport-session-setup-wMZfb` — shipped PRs #223 (LAB-ONLY + AGI-as-SoA + I1-I10), #224 (three-part stack + thinking harvest + I11), #225 (codec-sweep plan + D0.6 CodecParamsBuilder + D0.7 precision-ladder validation). Next on this branch: board hygiene + CLAUDE.md tightening; then D0.1 / D0.2 / D0.3 / D0.5 Wire-surface code.
- `claude/deepnsm-grammar-phase1` — Phase 1 PR #210, merged into main.
- `main` — up-to-date post #225.

## Active Integration Plans

- **`elegant-herding-rocket-v1`** — grammar / NARS / crystal / AriGraph (Phase 1 shipped in #210; Phase 2 queued).
- **`codec-sweep-via-lab-infra-v1`** (NEW 2026-04-20) — JIT-first codec sweep through lab endpoint; 1 upfront rebuild, unlimited candidates afterwards. D0.6 + D0.7 shipped in #225.

## Immediate Next Work

**`codec-sweep-via-lab-infra-v1` Phase 0 remainder (next up):**

- **D0.1** Extend `WireCalibrate` + `WireTensorView` (object-oriented, 64-byte-aligned decode) (~180 LOC).
- **D0.2** `WireTokenAgreement` endpoint stub — the I11 cert gate (~160 LOC).
- **D0.3** `WireSweep` streaming endpoint + Lance append (~200 LOC).
- **D0.5** `auto_detect.rs` reading `config.json` for `ModelFingerprint` (~140 LOC).
- Four test gates: `kernel_contract_test`, `amx_dispatch_test` (x86_64), `wire_object_surface_test`, `no_internal_serialisation_test`.

Total Phase 0 remainder: ~680 LOC, one upfront rebuild, surface freezes after.

**`elegant-herding-rocket-v1` Phase 2 (still queued):**

Per `.claude/plans/elegant-herding-rocket-v1.md`:

- **D2** DeepNSM emits `FailureTicket` on low coverage (~150 LOC).
- **D3** Grammar Triangle wired into DeepNSM via `triangle_bridge.rs` (~220 LOC).
- **D5** Markov ±5 bundler with role-indexed VSA (~300 LOC).
- **D7** NARS-tested grammar thinking styles as meta-inference policies (~260 LOC).

Total ~930 LOC, one PR when it ships.

## Deferred (do NOT propose these — they're explicitly parked)

- CausalityFlow TEKAMOLO extension (modal/local/instrument + beneficiary/goal/source, 9 total) — struct change deferred until after Phase 2.
- D8 story-context bridge, D9 ONNX arc export, D10 Animal Farm validation, D11 bundle-perturb emergence — Phase 3/4.
- Named Entity pre-pass (NER) — biggest OSINT blocker, separate PR.
- FP_WORDS = 160 migration (currently 157) — coordinated ndarray change.
- Crystal4K 41:1 persistence compression.
- 200-500 YAML TEKAMOLO templates per language — future training pipeline.
- Python/TypeScript grammar-stack convergence.

## If You're Tempted to Propose Something

Check this file first. Then check the KNOWLEDGE_INDEX.md for which
docs cover your domain. Then load only those docs. If you're still
uncertain whether something exists, grep the actual source before
proposing a new type.

The fastest way to waste 30 turns is to re-invent what's already in
the contract. This file exists to prevent that.

---

## 2026-05-05 — Recently Shipped backfill (PRs #244–#335)

> The "Recently Shipped PRs" table above stops at #243 (last refreshed 2026-04-21). Roughly 50 PRs have merged since. This section retrofits them.

| PR | Merged | Title | What it added (one-line) |
|---|---|---|---|
| **#335** | 2026-05-05 | Claude/thought cycle soa integration plan | Two new knowledge docs: gaussian-splat-cam-plane-workaround + entropy-budget-codebook-superposition |
| **#330** | 2026-05-01 | docs: add Cursor Cloud specific instructions to AGENTS.md | AGENTS.md section: ndarray path, CI commands, fmt-drift inventory, bgz-tensor known failures |
| **#329** | 2026-05-01 | style: apply rustfmt to contract lib.rs + python bindings | Tier-A rustfmt drift in contract lib.rs + python bindings (no semantic change) |
| **#328** | 2026-05-01 | ci(test): add lance-graph-contract unit tests to test gate | `cargo test -p lance-graph-contract --lib` added to CI rust-test.yml |
| **#327** | 2026-05-01 | style(shader-driver): drop double-space alignment in bindspace.rs | Two-line rustfmt drift fix in bindspace.rs introduced by #323 |
| **#326** | 2026-05-01 | fix(sigma-propagation): correct log_norm_growth_negative test seed | Fix broken test from #322: seed at 4·I not I so attenuation reduces log-norm |
| **#325** | 2026-04-30 | chore(toolchain): bump pin 1.94.0 → 1.94.1 | rust-toolchain.toml bumped to 1.94.1 to match sibling repos |
| **#324** | 2026-04-30 | feat(shader-driver): Pillar-7 α-front-to-back-merge sink mode (B5) | AlphaFrontToBack MergeMode + EWA Kerbl-2023 compositing in stage [7] |
| **#323** | 2026-04-30 | feat(cognitive-shader-driver): add Σ-codebook-index column to FingerprintColumns (B2) | FingerprintColumns.sigma u8 column (+1 byte/row, 0.02% overhead) |
| **#322** | 2026-04-30 | feat(contract): promote EWA-Sandwich Σ-propagation kernel to contract (B1) | sigma_propagation.rs: Spd2, ewa_sandwich, log_norm_growth, pillar_5plus_bound |
| **#321** | 2026-04-30 | fix: 10 pre-existing test failures (cosine_distance, arigraph, parse_triplets) | Fixed cosine inversion, Stagnant ordering, quality_window clear, SPO arg order |
| **#320** | 2026-04-30 | ci: declare rustfmt + clippy as pinned-toolchain components | rust-toolchain.toml gets components=[rustfmt,clippy]; fixes CI fmt failure |
| **#319** | 2026-04-30 | fix(transcode): per-month day-validity in parse_iso_date_to_days | Gregorian per-month + leap-year gate before civil_to_days |
| **#316** | 2026-04-30 | feat(transcode): round-3 typed-value resolver for triples_to_batch | triples_to_batch_with_resolver: Currency→f32, Date→Date32, Id→u64 |
| **#315** | 2026-04-30 | ci: revert ndarray-branch pin — PR #115 landed on master | Remove temp ndarray branch pin from rust-test.yml + style.yml |
| **#314** | 2026-04-30 | docs(vision): clear post-F1 staleness items in medcare-foundry-vision.md | §1–§4 DRAFT/forward-tense/PR-N placeholders replaced with real anchors |
| **#313** | 2026-04-30 | feat(transcode): Phase-2-B triples_to_batch (ExpandedTriple → RecordBatch) | ExpandedTriple stream → N-row RecordBatch, lenient-Utf8, 19 tests |
| **#312** | 2026-04-30 | feat(transcode): Phase-2-A pushdown classification (Inexact for recognised filters) | OntologyTableProvider classifies entity_type/predicate/nars filters as Inexact |
| **#311** | 2026-04-30 | docs(vision): mark F1 shipped, restate next deliverable as F2 | medcare-foundry-vision.md §7: F1 parity shipped; F2 RBAC is next posture |
| **#310** | 2026-04-30 | feat(transcode): r2 fixes — typed Arrow + codec_route + partial writes + CachedOntology | Currency/Date/Id→typed Arrow; CachedOntology; validate_route; from_columns_partial |
| **#309** | 2026-04-30 | feat(callcenter::transcode): outer ↔ inner ontology mapper + parallelbetrieb | transcode submodule: zerocopy, cam_pq_decode, spo_filter, ontology_table, parallelbetrieb |
| **#308** | 2026-04-30 | feat: bilingual ontology DTO surface + bgz-tensor workspace inclusion | OntologyDto locale projection; smb_ontology + medcare_ontology; bgz-tensor in workspace |
| **#307** | 2026-04-30 | refactor: dedup FNV-1a — one canonical hash::fnv1a in contract | contract::hash::fnv1a const fn; 8 call sites unified |
| **#306** | 2026-04-30 | feat(G4): verb_table tense modulation (Quirk CGEL grounded) | 12 VerbFamily priors + tense_modifier → 144 unique cell values |
| **#305** | 2026-04-30 | feat(G3): DisambiguateOpts builder + deepnsm caller wiring real fingerprint | DisambiguateOpts builder; sign_binarize_to_binary16k; disambiguator_glue.rs |
| **#304** | 2026-04-30 | feat(G1): Pearl 2³ causality footprint with PAD-model qualia mapping | compute_pearl_mask() 3-bit SPO→CausalMask; PAD qualia footprint replaces 0.5 |
| **#303** | 2026-04-30 | feat(F6): FNV-1a scent with scent_u64 accessor + birthday collision tests | scent() FNV-1a fold-to-u8; scent_u64() full 64-bit digest; 10 tests |
| **#302** | 2026-04-30 | feat(F3): LanceAuditSink with temporal timestamps + full schema round-trip | LanceAuditSink → Lance dataset append; temporal timestamp; O(1) scan_back |
| **#301** | 2026-04-30 | feat(F1): ColumnMaskRewriter full-tree expression walk + Hash UDF hard-fail | Full-tree OptimizerRule covering Filter/Aggregate/Join; NotYetWiredHashUdf |
| **#300** | 2026-04-30 | feat(LF-12): Pipeline DAG with StepId derivation + OrchestrationBridge adapter | PipelineDag Kahn's algorithm; FNV-1a StepId; execute_via_bridge; cycle detection |
| **#299** | 2026-04-29 | revert #294/#295/#296 + clean on top | Reverts #294–#296 confabulation; corrects probe routing (M1/P2-P4 → shader-lab) |
| **#296** | 2026-04-29 | ideas: COCA-Bundle vs Jina-CLAM bucket comparison (**REVERTED by #299**) | IDEAS.md Open entry for COCA/Jina probe (premise flawed; reverted) |
| **#295** | 2026-04-29 | docs: probe-queue data-available followup (**REVERTED by #299**) | bf16-hhtl-terrain.md data-available update (inherited bad routing; reverted) |
| **#294** | 2026-04-29 | docs(probe-queue): honest "needs production data" assessment (**REVERTED by #299**) | bf16-hhtl-terrain.md probe routing table (wrong routing; reverted) |
| **#293** | 2026-04-29 | jc: drain Probe P1 (γ-phase-offset ranking discrimination) → PASS | probe_p1_gamma_phase.rs; P1 PASS: min Spearman ρ=-0.963 (Dupain-Sós) |
| **#292** | 2026-04-29 | docs(board): posthoc-correct PRs #290 #291 via canonical board mechanism | CONJECTURE banners; 5 Open IDEAS.md entries; 2 EPIPHANIES.md entries |
| **#291** | 2026-04-29 | docs: idea journal — proposed application pillars 7/8/9 captured | IDEA_JOURNAL_2026_04_29_FUTURE_PILLARS.md with Pillars 7/8/9 + PASS criteria |
| **#290** | 2026-04-29 | docs: idea journal — streaming-hydration + fractal-codec captured | IDEA_JOURNAL_2026_04_29_STREAMING_HYDRATION.md separating two ideas |
| **#289** | 2026-04-29 | jc: Pillar 6 — EWA-Sandwich Σ-push-forward | ewa_sandwich.rs; Pillar 6: 10000/10000 PSD-preserving hops; KS bound tightness 1.467× |
| **#288** | 2026-04-29 | jc: Σ-Codebook Viability Probe — rules out CausalEdge64 8→16B expansion | sigma_codebook_probe.rs; R²=0.9949 at k=256; CausalEdge64 stays 8 bytes |
| **#287** | 2026-04-29 | jc: Pillar 5++ — Düker-Zoubouloglou Hilbert-space CLT | dueker_zoubouloglou.rs; Pillar 5++: bundle-of-N in ℝ^16384 → Gaussian limit in ℓ² |
| **#286** | 2026-04-29 | jc: Pillar 5+ — Köstenberger-Stark concentration on Hadamard 2×2 SPD | koestenberger.rs; Pillar 5+: tightness 0.969× on SPD manifold |
| **#285** | 2026-04-29 | Re-land #283 unlocks (Quantum, Disambiguator, verb_table, animal-farm) | Quantum mode, Disambiguator trait, verb_table, animal-farm harness; PhaseTag overflow fix |
| **#284** | 2026-04-29 | Re-land #281 unlocks (PolicyRewriter, DomainProfile) | PolicyRewriter trait, ColumnMaskRewriter, DomainProfile HIPAA thresholds |
| **#282** | 2026-04-29 | fix: Grammar/Markov hardening — slice unification, kernel wiring | CRITICAL slice fix; rotate_right removed; coherence kernel wired; 363 tests |
| **#280** | 2026-04-29 | fix: Foundry hardening — sealed RLS, VecDeque audit, URL decode, Plugin handshake | Sealed RLS default; O(1) audit ring; FNV-1a; URL decode; Plugin handshake; 58 tests |
| **#279** | 2026-04-29 | feat: DeepNSM grammar parser — Markov ±5 bundler, role keys, thinking styles | D0/D2/D3/D4/D5/D6/D7: MarkovBundler, RoleKeySlice, GrammarStyleConfig, 12 YAML configs |
| **#278** | 2026-04-29 | feat: Foundry parity — RLS rewriter, audit log, PostgREST, with_registry | LF-3/DM-7 RLS; LF-90 audit; DM-8 PostgREST stub; LanceMembrane::with_registry; 35 tests |
| **#277** | 2026-04-28 | plan: unified Foundry roadmap for SMB + MedCare (corrects #276 framing) | foundry-roadmap-unified-v1.md; correct scale decisions per FormatBestPractices.md |
| **#276** | 2026-04-28 | plan: Foundry Consumer Parity — shared ontology + UNKNOWN resolutions | foundry-consumer-parity-v1.md; 5 callcenter UNKNOWNs resolved; DM-8 unblocked |
| **#275** | 2026-04-28 | feat: add lancedb 0.27.2 + pin lance =4.0.0 | lancedb=0.27.2 optional dep; lance exact-pinned =4.0.0 for compat |
| **#274** | 2026-04-27 | fix: F-01 identity-tear race + F-08 bounds check + F-09 poison recovery | Single ActorState RwLock; poison recovery; push bounds check |
| **#273** | 2026-04-27 | feat: bump lance 2→4 + datafusion 51→52 + deltalake 0.30→0.31 | Version bumps + API break fixes (invalid_input, DeltaTableProvider migration) |
| **#272** | 2026-04-27 | feat: Column H — EntityTypeId on BindSpace (Phase 1 of 4) | EntityTypeId u16 on BindSpace; push_typed(); 1-based index; 4 tests |
| **#271** | 2026-04-27 | plan: BindSpace Columns E/F/G/H — 4→8 SoA integration plan | bindspace-columns-v1.md; 24 deliverables; 7 SOUND / 7 CAUTION / 0 WRONG |
| **#270** | 2026-04-26 | ci: remove typos spell-check job (too many false positives) | Removed crate-ci/typos from style.yml; cargo fmt --check remains |
| **#269** | 2026-04-26 | feat: Distance trait + SIMD Hamming/cosine wiring + PaletteDistanceTable + Dockerfile docs | Distance trait; SIMD Hamming/cosine wiring; PaletteDistanceTable 128KB; Dockerfile.md |

