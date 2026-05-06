# JC pillars → runtime wiring (cognitive-shader-driver)

> **Status:** plan, not implementation. v1.
> **Owner crate:** `crates/cognitive-shader-driver/` (consumer); `crates/jc/` (provider).
> **Sister plans:** `lance-graph-rdf-fma-snomed-v1.md` (ontology-context column).
> **Constraint:** non-destructive — every phase adds a parallel path; old paths stay green until pillar invariants pass on real data.

## Why

`crates/jc/` is a five-pillar mathematical proof-in-code that the cognitive-shader substrate **could** be a clean Pearl 2³ Index regime with EWA-sandwich Σ-propagation. The current runtime in `cognitive-shader-driver/src/{wire,bindspace}.rs` is closer to the **bundled regime** that JC Pillar 5b explicitly measures as accuracy-lossy:

- Bitmap `[[u64; 64]; 8]` after `mod 64` collisions
- 8-class predicate regex (`classify_relation`) collapsing P-vocabulary
- No per-edge Σ attached; multi-hop = arithmetic noise growth
- Pearl 2³ flagged via `pearl: u8` but only enforced for SP_ and SPO masks (geometry doesn't exist for the other six)

The JC pillars give us the *math*. Runtime doesn't yet *use* them. Wiring closes the gap and converts each "is this real?" question from "trust the doc-string" to "JC's pillar passed in CI on real data".

## What this plan covers vs disowns

**Owns:**
- Promotion of `crates/jc/src/ewa_sandwich.rs::sandwich` and `Spd2` from proof-internal to public JC API
- Parallel `IndexShader` runtime path alongside existing `BundledShader`, gated by feature flag
- Per-edge `Σ ∈ SPD(d)` attached to claims and propagated via `sandwich(J, Σ)`
- Σ-aware promotion gate (claim promotes only if NARS-conf AND Σ-spread AND path-bounded all pass)
- Pearl 2³ mask geometry — eight materialized projections instead of two implicit (SP_, SPO) ones
- COCA-4096 predicate vocabulary substituting the 8-class regex (with regex fallback for cold-start)
- JC pillars promoted from `cargo run --example prove_it` to `cargo test` — every PR runs the proof-corpus

**Disowns (explicit out-of-scope):**
- Replacing NARS Wang truth-table revision (already real math; keep)
- Replacing `SpoHead.{freq, conf, pearl_mask}` (already real; keep)
- Training COCA-4096 codebooks (data/learning concern; this plan defines the substitution path, not the learning loop)
- Pillar 2 (Cartan-Kuranishi) and Pillar 4 (γ+φ preconditioner) — deferred per JC's `lib.rs`; we proceed without their guarantees
- Replacing the bitmap entirely in P2/P3 — both regimes coexist behind feature flag until Pillar 5b's "≥ bundled" criterion measured on real data
- `lance-graph-rdf` ontology-context wiring (separate plan; sister concern, will join at the SemanticSpoRow layer)

## Phases

Each phase is non-destructive: old paths stay green; new paths add JC-pillar-grounded primitives. Phase passes by **measuring** the corresponding JC pillar property holds in the runtime path on real data, not just in the proof example.

### P0 — JC pillars run as `cargo test`, not just `cargo run --example`

| What | Promote `crates/jc/examples/prove_it.rs` from manual run to test-suite gate. Each pillar's `prove()` returns `PillarResult { pass: bool }` already; wrap in `#[test]` so CI fails on regression. |
|---|---|
| **Why** | JC pillars are currently proof-corpus runnable on demand. Without CI gating, a refactor that breaks substrate associativity or Pearl 2³ accuracy goes unnoticed. |
| **Cost** | Hour-scale. Wiring only — no new math. |
| **Acceptance** | `cargo test -p jc --release` runs all five executable pillars (1, 3, 5, 5b, 6 — pillars 2 and 4 stay deferred). PR CI invokes it. |
| **Risk** | Pillar runtimes — substrate.rs is 10k trials × d=10000 ~ a few seconds; pearl.rs is 4000 trials × d=16384 ~ 10s; ewa_sandwich.rs is 1000 paths × 10 hops ~ 1s. Total ≤ 30s, acceptable for CI. If borderline, gate behind `--release` only. |
| **Reverts cleanly** | Yes — drop the `#[test]` wrappers. |

### P1 — Promote EWA-sandwich and SPD math from proof-internal to public JC API

| What | Move `sandwich()`, `Spd2` (with `eig`, `pow`, `sqrt`, `log_spd`, `frobenius_sq`, `det`, `is_spd`), `sample_step_sigma`, `propagate_path` from `crates/jc/src/ewa_sandwich.rs` (currently private to the pillar) into a public module: `pub mod ewa { pub use spd::*; pub use sandwich::*; }`. The pillar's `prove()` continues to use them; downstream crates (cognitive-shader-driver, lance-graph-rdf) can now depend on JC for the math without copying. |
|---|---|
| **Why** | Mathematical primitives belong in one place. Without this, P3 forces the cognitive shader to either copy the SPD math (drift risk) or re-invent it (correctness risk). |
| **Cost** | Day-scale. Pure refactor — no math change. Add `Spd2: Copy + Clone + Debug`, `Send + Sync` impls explicit. |
| **Acceptance** | `jc::ewa::sandwich(&m, &n)` is callable from `cognitive-shader-driver` tests. `cargo test -p jc` continues to pass (Pillar 6 still works). One downstream consumer (test crate) demonstrates the call. No public API regression. |
| **Risk** | Low. Refactor of existing tested code. |
| **Reverts cleanly** | Yes — git revert. |

### P2 — `IndexShader`: parallel three-plane Pearl 2³ runtime path

| What | Add a new shader implementation in `cognitive-shader-driver/src/index_shader.rs` (gated `#[cfg(feature = "index-shader")]`) alongside the existing bundled bitmap. The Index shader stores three role-keyed planes per Pearl mask — one for S, P, O — at d=16384 binary fingerprints per claim. Recovery via XOR ⊕ R_k Hamming distance, exactly the Pillar 5b Method A geometry. Both shaders run; results compared on every claim insertion and query. |
|---|---|
| **Why** | Pillar 5b proves three-plane Index ≥ bundled at Pearl 2³ mask accuracy. Wiring it in **parallel** lets us measure that gap on real data, not synthetic. If the gap appears on production traffic, we have evidence to retire bundled. If not, we know the synthetic vs real distribution drift and adapt. |
| **Cost** | Week-scale. ~500 LOC for the shader; ~200 LOC for the parallel-runner; tests. |
| **Acceptance** | (a) Both shaders return the same Pearl mask classification on synthetic Pillar 5b inputs (sanity). (b) On real claim traffic, Index shader's mask classification accuracy ≥ bundled shader's accuracy across a 10K-claim corpus. (c) Index shader memory: ≤ 4 KB per claim (3 planes × 16384 bits / 8 = 6 KB; one plane per Pearl-mask-active-role; can be sparse). (d) Latency: Index shader within 2× of bundled shader's lookup time on the hot path. |
| **Risk** | (i) Real-data distribution may not exhibit the Pearl 5b gap if existing claims are mostly SPO (single mask) — partial mitigation: track the distribution of Pearl masks observed and report. (ii) Memory cost scales with claim count — at 1M claims × 6 KB = 6 GB; LRU on cold contexts. (iii) Coordinator code that compares results per claim adds overhead — gate it behind a sampling rate. |
| **Reverts cleanly** | Yes — disable the feature flag. Bundled shader stays the default. |

### P3 — Per-edge Σ propagation via EWA-sandwich (read-only first)

| What | Each edge in the cognitive graph carries a `Σ ∈ SPD(d_eff)` covariance. On multi-hop traversal, `Σ_n+1 = sandwich(J_e, Σ_n)` per hop, where `J_e = sqrt(Σ_e)`. Initially **read-only**: Σ is computed and stored, but no promotion / decision uses it yet. Phase visibility is the goal — we want to measure Pillar 6 properties (PSD preservation rate, CV concentration) on real graph traversals. |
|---|---|
| **Why** | The bitmap implicitly assumes additive (convolution) error growth: every hop adds noise. Pillar 6 proves sandwich gives **bounded** Σ when M_k contractive — geometric error control instead of arithmetic. Multi-hop graph queries at depth >5 lose meaning under convolution; with sandwich, they stay well-conditioned. We need the runtime measurement to confirm contractive regime on real edges. |
| **Cost** | Week-scale. Add `Σ: Spd2` (or 4×4 SPD for higher-d if needed) to the edge type; sandwich-propagate during traversal; export per-traversal stats (PSD rate, CV). |
| **Acceptance** | (a) On 1000 real multi-hop traversals × length 10, PSD preservation rate ≥ 0.999 (Pillar 6 criterion). (b) CV ≤ `√(2/n) · √(1+2σ²n) · 1.75` (Pillar 6 slack). (c) No regression in existing query latency or correctness — Σ is read-only, doesn't affect routing. (d) Report includes Σ-spread distribution per Pearl mask. |
| **Risk** | (i) Edge confidence may not produce contractive Σ in real data — Σ_e with eigenvalues > 1 → divergent paths. Mitigation: cap eigenvalues at 1.0 with logging; investigate divergent edges as data quality issue. (ii) SPD math at d=16384 is expensive — start with 2×2 SPD on a low-dim projection, scale up only if needed. |
| **Reverts cleanly** | Yes — Σ is a side-channel; disable, don't propagate. |

### P4 — COCA-4096 predicate vocabulary substituting the 8-class regex

| What | Replace `convergence::classify_relation` (8-class regex) with a COCA-4096 codebook lookup: each predicate string maps to one of 4096 codebook prototypes via Jina/embedanything embedding nearest-neighbor. Regex stays as **fallback** when codebook is empty (cold-start) or when embedding service is offline. |
|---|---|
| **Why** | The 8-class regex is the most catastrophically lossy step in the current shader: it throws away ~99% of predicate vocabulary distinguishing power. Pillar 3 (φ-Weyl 144-verb collocation) suggests 144 is the *minimum* Weyl-optimal coverage; 8 is far below. COCA-4096 gives us the full bilinear S × O × P field at the resolution Pillar 5b runs at. |
| **Cost** | Two-week-scale. Codebook bootstrapping (Jina embed all observed predicates → cluster → 4096 centroids) is a separate pipeline. Lookup integration is a day. The training data and embedding service dependency are the cost. |
| **Acceptance** | (a) On the same 10K-claim corpus from P2, COCA-4096 lookup mask classification ≥ regex mask classification — measured by Pearl 2³ accuracy gap. (b) Cold-start fallback to regex works (zero codebook → regex path). (c) Codebook is hot-reloadable without shader restart. (d) Predicate distribution: at least 80% of observed predicates land in non-noise codebook entries (well-clustered). |
| **Risk** | (i) Bootstrapping requires representative predicate corpus — if observed predicates are skewed, codebook is skewed. Mitigation: bootstrap from public ontology predicate vocabularies (FMA `regional_part_of`, RadLex `RID:has*`, SNOMED relationship types) for outer-ontology coverage; let local predicates fill 40% of the codebook. (ii) Embedding service latency on hot path — pre-compute, batch, or fall back to regex. (iii) Codebook drift over time — version it; retain old versions for replay. |
| **Reverts cleanly** | Yes — `--features regex-predicate` falls back. |

### P5 — Σ-aware promotion gate

| What | Wire Σ from P3 into the promotion membrane. A claim promotes only when **all three** hold: (a) NARS `conf ≥ threshold` (existing); (b) `Σ-spread ≤ threshold` (new — claim has low uncertainty); (c) Pillar 6 path-bounded check passes for the multi-hop derivation chain (new — the path that reached this claim stayed well-conditioned). Bitmap presence-OR demoted to coarse first-stage filter; real promotion is Σ-driven. |
|---|---|
| **Why** | Currently "promotion" passes if NARS conf is high. But high NARS conf with high Σ-spread = the system is confident in a mean that has wide uncertainty → false confidence. Σ-aware promotion catches this. |
| **Cost** | Week-scale. Promotion path already exists; add the two new checks. |
| **Acceptance** | (a) Calibration test: on a labeled set of "should-promote" vs "should-not-promote" claims, Σ-aware gate's precision ≥ NARS-only gate's precision. (b) No false-positive regressions: claims that the NARS-only gate promoted correctly still promote. (c) Σ-spread threshold is configurable per ontology context (FMA-anatomy may tolerate looser; SNOMED-clinical tighter). |
| **Risk** | (i) Σ-spread threshold tuning is per-deployment — start conservative (high threshold = permissive), tighten with measurement. (ii) Path-bounded check requires the entire derivation chain — adds query latency for promotion calls; mitigate with cached Σ-projections. |
| **Reverts cleanly** | Yes — feature flag falls back to NARS-only promotion. |

### P6 — Eight materialised Pearl 2³ query paths

| What | Each Pearl 2³ mask `m ∈ {∅, S, P, O, SP, SO, PO, SPO}` has its own query dispatcher in `index_shader::query::dispatch(mask, query) → result`. Queries route to the appropriate marginalized geometry (closed-form Σ marginalization per Pillar 6 + role-plane subset selection per Pillar 5b). The "boring 8× shader runs" are explicit and SIMD-broadcastable. |
|---|---|
| **Why** | Pearl 2³ enforcement is currently partial: only SP_ → ?O and SPO → ?validation have geometries. The other six masks fall back to the same SPO grid with implicit role assumption. Materialising all eight removes the Pearl-mask information loss the user identified. |
| **Cost** | Week-scale. Eight dispatch handlers, each ~100 LOC; shared SIMD kernel; tests per mask. |
| **Acceptance** | (a) Each Pearl mask passes a synthetic Pillar-5b-style accuracy test: given (active-role bindings, query) → recover correct Pearl mask classification with accuracy ≥ 0.95. (b) Hot path latency for SP_ (most common) ≤ existing bundled bitmap latency. (c) `___` (prior, no roles active) returns the unconditional prior over claims — well-defined fallback. (d) Marginalization correctness: `query(SPO).reduce(P) == query(S_O)` (within tolerance). |
| **Risk** | (i) Memory cost — eight materialized geometries per ontology context. Lazy materialization, populate on first query per (context, mask) pair. (ii) Cross-mask consistency — drift between SP_ output and SPO-marginalized-over-P output is a data-quality issue; test catches it. |
| **Reverts cleanly** | Yes — feature flag, dispatcher falls back to existing bundled path. |

## Architecture (where things sit after all phases)

```
                          ┌──────────────────────────┐
                          │  crates/jc (proof crate) │
                          │   • Pillar 1: substrate  │
                          │   • Pillar 5b: Pearl 2³  │
                          │   • Pillar 6: EWA-sandwich│
                          │   • Pillar 5: Jirak floor │
                          │   • pub mod ewa { … }    │ ← P1 promotes API
                          └────────────┬──────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────┐
                  │  crates/cognitive-shader-driver     │
                  │                                     │
                  │  Existing path (default):           │
                  │   • BundledShader (bitmap, regex)   │
                  │                                     │
                  │  New paths (feature-flagged):       │
                  │   • IndexShader (3-plane × 8 mask)  │ ← P2, P6
                  │   • Σ-edge propagation (sandwich)   │ ← P3
                  │   • COCA-4096 predicate lookup      │ ← P4
                  │   • Σ-aware promotion gate          │ ← P5
                  │                                     │
                  │  Shared (always-on):                │
                  │   • SpoHead {freq, conf, pearl}     │
                  │   • NARS Wang revision tables       │
                  └────────────┬────────────────────────┘
                               │
                               ▼
                  ┌─────────────────────────────────────┐
                  │  Promotion membrane                 │
                  │   • NARS-conf threshold (existing)  │
                  │   • + Σ-spread threshold (P5)       │
                  │   • + Pillar 6 path-bounded (P5)    │
                  └─────────────────────────────────────┘
```

## Acceptance criteria summary (per pillar, runtime-grounded)

| Pillar | Property certified by JC proof | Runtime acceptance after wiring |
|---|---|---|
| 1 (substrate) | Bundle associativity at d=10000, cosine ≥ 0.99 | At runtime d=16384, sample 1000 bundle compositions; cosine left-vs-right ≥ 0.99. CI gate (P0). |
| 5 (Jirak) | Berry-Esseen sup-error inflation bounded under weak dep at d=16384 | Runtime measures sup-error on real claim traffic; ≤ Jirak bound × 1.5 slack. (Plumb in P3.) |
| 5b (Pearl 2³) | Three-plane mask accuracy ≥ bundled at d=16384 | Index shader's mask accuracy ≥ Bundled shader's on real corpus (P2). All 8 masks materialised (P6). |
| 6 (EWA-sandwich) | PSD preservation rate ≥ 0.999, CV ≤ KS bound × 1.75 | Real multi-hop traversals × 1000: PSD rate ≥ 0.999, CV bound holds (P3). |
| 3 (φ-Weyl) | 144-verb collocation coverage Weyl-optimal | COCA-4096 at minimum 144 well-clustered predicate centroids; gap from regex measured (P4). |
| 2 (Cartan-Kuranishi) | role_keys ≡ Cartan characters | **Deferred** — not blocking; revisit when JC pillar reactivates. |
| 4 (γ+φ preconditioner) | Prolongation step reduction | **Deferred** — not blocking; revisit when JC pillar reactivates. |

## Out of scope (explicit)

- **Replacing NARS Wang truth tables.** The (f, c) revision arithmetic is real math; promotion + revision math stays.
- **Replacing `SpoHead`.** It's the real promotion vehicle; we extend with Σ side-channel, don't replace.
- **Training COCA-4096 codebooks.** This plan defines the substitution interface; the embedding pipeline + bootstrap data is a separate concern.
- **Pillars 2 and 4 (deferred in JC).** We proceed without their guarantees; phases that would benefit (P3 contraction guarantee on edges; promotion convergence rate) operate with conservative thresholds + measurement instead of proof.
- **`lance-graph-rdf` integration.** Sister plan covers ontology-context column. Joins this plan at the SemanticSpoRow layer when both land — not now.
- **The full reasoning lattice (64×64 cause-effect, 256×256 palette-attention, 4096³ SPO bilinear).** This plan wires the *substrate* pillars (1, 5, 5b, 6) into runtime. The lattice (Pillar 3 + 4 + future tensors) is a separate-plan concern.

## Dependencies + risks

**Hard prerequisites:**
- P1 (EWA API promotion) blocks P3, P5
- P2 (IndexShader) blocks P6
- P4 (COCA-4096) is independent; can run in parallel with P2/P3/P5

**Soft prerequisites:**
- P0 (CI gate) is recommended before any of P1–P6 lands; without it, regressions to JC pillars go undetected during integration
- Pillar 1's d=10000 floor is sufficient for the planned d=16384 runtime; weaker d (4096) would force re-running substrate.rs at lower d and confirming the floor

**Risk register:**
- **Real-data Pearl-mask distribution skew (P2 risk i)**: most production claims are SPO single-mask, so the Pillar 5b gap may not show on real data. Mitigation: track Pearl-mask distribution, expose it on the cockpit, drive feature flag flip-in with a per-mask sample size policy (don't claim "Index wins" without N ≥ 1000 per mask).
- **Σ-spread eigenvalue overflow (P3 risk i)**: real edges with high uncertainty can have Σ eigenvalues > 1, breaking Pillar 6's contractive assumption. Mitigation: cap eigenvalues; treat overflow as data-quality alert, not silent failure.
- **COCA bootstrap distribution skew (P4 risk i)**: local predicate corpus skews codebook. Mitigation: seed from public ontology predicate vocabularies (FMA, RadLex, SNOMED).
- **Σ-spread threshold tuning (P5 risk i)**: per-deployment. Conservative defaults; tighten with measurement.
- **Memory cost at scale (P2 + P6 risks)**: Index shader × 8 masks × 1M claims × few KB = multi-GB. Lazy materialization + LRU eviction.
- **Pillars 2 and 4 deferred**: P3 path-bounded check operates on Pillar 6 alone (without Pillar 4 prolongation guarantee). Conservative thresholds compensate. Revisit when pillars activate.

## Open questions

1. **Per-claim fingerprint width**: current `Base17` (272 bits) vs Pillar 5b/6's d=16384 (16K bits). Promoting per-claim fingerprint to d=16384 is 60× memory growth — at 1M claims, 2 GB → 16 GB. Tractable but worth confirming the dimensionality story before P2.
2. **Per-edge Σ rank**: 2×2 SPD (Pillar 6's proof rank) vs higher rank (per-role × per-Pearl-mask anisotropy). Start at 2×2, lift only if measurement demands.
3. **COCA-4096 vs full Jina embedding (D ≈ 1024)**: codebook quantization vs continuous embedding. Codebook is cache-friendly and matches the existing palette pattern; continuous loses the discrete-projection that Pearl 5b's geometry assumes. **Recommend codebook quantization at 4096.** Worth confirming Jina embedding distance distribution clusters cleanly into 4096 modes (a separate study).
4. **Pillar 5 (Jirak) plumbing**: currently the noise floor is a per-pillar measurement. To wire it into the runtime, sup-error has to be measurable per query — needs instrumentation.
5. **Branch / merge story with `lance-graph-rdf`**: when both plans land, SemanticSpoRow needs both `ontology_context_id` (from rdf plan) and Σ-edge-propagation (from this plan). The merge point is `cognitive-shader-driver/src/wire.rs::SemanticSpoRow` definition. Coordination is straightforward but needs a single-row-shape decision before either lands in code.
6. **Pillars 2 and 4 reactivation**: JC's `lib.rs` calls them "stubs pending coupled-revival-track activation" — when do they reactivate, and what runtime checks become available once they do?

## Reference

- Brainstorm session 2026-05-04: math vs wishful analysis (in conversation)
- `crates/jc/src/substrate.rs` — Pillar 1
- `crates/jc/src/jirak.rs` — Pillar 5
- `crates/jc/src/pearl.rs` — Pillar 5b (the key proof for this plan)
- `crates/jc/src/ewa_sandwich.rs` — Pillar 6 (the EWA-sandwich math to promote in P1)
- `crates/cognitive-shader-driver/src/{wire,bindspace}.rs` — runtime integration points
- `crates/lance-graph-planner/src/cache/convergence.rs` — current bitmap path; `classify_relation` (the regex P4 replaces)
- Sister plan: `lance-graph-rdf-fma-snomed-v1.md` (ontology-context column; joins at SemanticSpoRow)
- Architectural Commitment #4 (MedCare-rs CLAUDE.md): thinking lives only in lance-graph; this plan does not introduce reasoning, only enforces Pearl 2³ geometry that JC pillars certify.

---

**Phasing summary one-liner:** P0 turns JC into a CI gate; P1 promotes EWA-sandwich math from proof to lib; P2 adds Index shader as a parallel path; P3 attaches Σ to edges and propagates; P4 swaps the 8-class regex for COCA-4096; P5 makes promotion Σ-aware; P6 materialises all eight Pearl 2³ query paths. Each phase is non-destructive and has a JC pillar as its measurement gate.
