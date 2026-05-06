# Erratum to `jc-pillars-runtime-wiring-v1.md`

> **Filed:** 2026-05-05 (after reading `crates/lance-graph-planner/src/cache/nars_engine.rs`).
> **Scope:** correct layer-attribution for each JC pillar's wiring target. **No phase deletions; no new phases.** Several phases apply to a *different* layer than the v1 draft implied.

## Why this erratum

The v1 draft conflated two distinct layers when describing where each JC pillar wires in:

- **L1 (atomic NARS)** — `crates/lance-graph-planner/src/cache/nars_engine.rs` and `crates/causal-edge` (`SpoHead` + `CausalEdge64`)
- **L2 (preview cache)** — `crates/lance-graph-planner/src/cache/convergence.rs` (the `[[u64;64];8]` bitmap)
- **L3 (SoA hot path)** — `crates/cognitive-shader-driver/src/{wire,bindspace}.rs`
- **L4 (thinking styles + MUL)** — `[f32; 8]` weight vectors over Pearl 2³ + meta-uncertainty layer (DK + trust)

After reading `nars_engine.rs` directly: **L1 is already in the Pillar-5b Index regime.** `SpoDistances` is structurally three separate planes (`s_table`, `p_table`, `o_table`), each 256×256×u16. `causal_distance(mask)` only sums the planes the Pearl mask activates. Pearl 2³ is fully explicit (`MASK_NONE`/`S`/`P`/`O`/`SP`/`SO`/`PO`/`SPO`, mapped to Pearl's three rungs of causation). Wang truth-value algebra is real (`f = fa·fb, c = fa·fb·ca·cb` for Deduction; `w/(w+1)` form for Induction/Abduction). NarsTables hot-path revision is O(1) no-float through a 128 KB L1-cache-resident lookup.

The "bundled regime" the v1 draft proposed to lift is **not** at L1. It's at L2 — `convergence.rs` runs `triplets_to_palette_layers` which OR-bits a single u64 cell per `(s_idx % 64, o_idx % 64, classify_relation(p))` triple. That's the bundled-with-collision step. L2 is a *preview cache* downstream of L1, not the reasoning layer.

## Corrected layer-mapping table

| Pillar / phase | v1 said wires at | Actual layer | Erratum |
|---|---|---|---|
| **Pillar 1** (substrate, d≥10K bundle associativity) | (substrate generally) | embedding/fingerprint substrate, separate from L1's symbolic 8-byte SpoHead | unchanged scope; just clarify it does **not** apply to SpoHead's q8 truth-value layer |
| **Pillar 5b** (three-plane Index vs bundled) | "lift the bitmap to three planes" | **L2 + L3 only.** L1 is already three-plane. | **Phase P2 (`IndexShader` as parallel path) wires at `cognitive-shader-driver/src/bindspace.rs`, NOT at `nars_engine.rs`. nars_engine needs no change for Pillar 5b.** |
| **Pillar 6** (EWA-sandwich Σ-propagation) | "per-edge Σ via sandwich" | **L3 only**, additive | **Phase P3 attaches Σ as a side-channel on `CausalEdge64` propagated through L3's edge-traversal hot path. It does NOT replace L1's Wang composition (`forward_edge` via compose tables); it adds anisotropic uncertainty alongside.** |
| **Pillar 3** (φ-Weyl 144-verb collocation) | "COCA-4096 predicate vocabulary replaces 8-class regex" | **L2 only** | **Phase P4 replaces `convergence.rs::classify_relation` with COCA-4096 predicate lookup. L1's `Inference` enum stays at 7 variants. Compounds/products/variables come via L4 thinking-styles, not L1 atom expansion.** |
| **Pillar 5** (Jirak Berry-Esseen weak-dep noise) | runtime sup-error budget | L3 (instrumented around the SoA hot path); L1's existing skepticism heuristic stays | unchanged scope |
| **Pillars 2 + 4** (Cartan-Kuranishi + γ+φ preconditioner) | deferred per JC `lib.rs` | deferred | unchanged |

## Per-phase scope corrections

**P0 — JC pillars as `cargo test`:** unchanged. Cross-layer CI gate.

**P1 — Promote EWA-sandwich + Spd2 to public JC API:** unchanged. Layer-agnostic refactor.

**P2 — `IndexShader` parallel path:** **Scope narrows to L3 (`cognitive-shader-driver/src/bindspace.rs`).** L1 (`nars_engine.rs`) needs **no IndexShader** because `SpoDistances` is already three-plane. The acceptance criterion "Index shader's mask classification accuracy ≥ Bundled shader's" applies to the L2/L3 boundary — comparing the L2 bitmap-fed bundled path against an L3 three-plane SoA path that consults L1 directly. The synthetic Pillar 5b inputs still apply; the real-data corpus is whatever flows through the cognitive-shader-driver, not synthetic SpoHead constructions.

**P3 — Per-edge Σ propagation (read-only first):** **Scope narrows to L3.** Σ attaches to `CausalEdge64` instances **as carried through the L3 hot path** — the SoA traversal layer. L1 itself doesn't need to know about Σ; `forward_edge`'s scalar compose continues to operate as today. The Σ side-channel runs alongside via a parallel EWA-sandwich step at the L3 traversal boundary. Acceptance criteria unchanged (PSD rate ≥ 0.999, CV ≤ KS bound × 1.75).

**P4 — COCA-4096 predicate vocabulary:** **Scope narrows to L2.** Replaces `convergence.rs::classify_relation` (the 8-class regex). L1's `Inference` enum (Deduction/Induction/Abduction/Revision/Analogy/Resemblance/Synthesis) stays at 7 — those are NAL inference *rule names*, not the predicate *vocabulary*. **Compounds (NAL-3 `&`, `\|`, `-`), products (NAL-4 `(a*b) --> r`), and variables (NAL-6 `$1`) move to L4 thinking-styles + MUL — they do NOT extend `SpoHead`.** This is consistent with the user's directive that compound/application reasoning lives in thinking-styles and MUL (DK + trust), not in atom expansion.

**P5 — Σ-aware promotion gate:** **Scope clarified to L3 + L4.** The Σ-spread threshold check happens at L3 (where Σ is propagated); the *thresholds themselves* come from L4 thinking-styles (`commit_threshold`, `novelty_budget` per the canonical 36 styles in `agi_lego_party_canonical.yaml`). MUL DK + trust modulate the threshold per claim — high DK = require tighter Σ; low trust source = require tighter Σ. L1's NARS-confidence path stays unchanged.

**P6 — Eight materialised Pearl 2³ query paths:** **Scope clarified.** L1 (`nars_engine.rs`) **already has all 8** via `SpoDistances::all_projections` returning `[u32; 8]`. The "materialisation" P6 talks about applies at L3 — eight SoA query-dispatch paths, each calling into L1's existing 8-mask projection and composing per-mask response with the L4 thinking-style weight vector. Per-mask synthetic accuracy criterion unchanged.

## What didn't change

- The seven-pillar measurement framework
- The non-destructive principle (parallel paths, feature-flagged)
- Pillars 2 and 4 deferred status
- The acceptance criteria (just attached to the right layer)
- Sister-plan coordination with `lance-graph-rdf-fma-snomed-v1.md` at the SemanticSpoRow shape decision
- The promotion gate's structural separation (NARS-conf + Σ-spread + Pillar-6 path-bounded — just relocated to L3+L4)

## Mental model update worth carrying forward

| Concern | Wrong layer to fix it | Right layer to fix it |
|---|---|---|
| "Three-plane vs bundled Pearl 2³" | L1 (already correct) | L2 (the bitmap) and L3 (SoA) |
| "Multi-hop noise growth" | L1 (Wang already attenuates scalarly) | L3 (Σ side-channel via EWA-sandwich) |
| "Predicate vocabulary too narrow" | L1 (`Inference` enum is rule-names, not vocab) | L2 (replace regex with COCA-4096) |
| "Compounds, products, variables" | L1 atom expansion (NO) | L4 thinking-styles + MUL DK/trust |
| "DK detection on over-compounded confidence" | L1 (atomic, doesn't know about composition) | L4 MUL |

This keeps L1 atomic and 8-byte. All compositional, contextual, and meta-uncertainty reasoning lives in L3 (Σ + ontology context) and L4 (thinking-styles + MUL). The atom doesn't grow; the layers above carry the structure.

## Pointer back to v1

This erratum supersedes layer-attribution claims in v1's Phase descriptions. The v1 acceptance criteria, deferral policy, dependencies, and risk register all stand. Read this erratum *with* v1, not in place of it.

Filed alongside: `.claude/plans/jc-pillars-runtime-wiring-v1.md`.
