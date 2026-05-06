# Splat-OSINT Ingestion v1 — SPLAT-1 stage 0->1 + EWA OSINT bridge

**Status:** Active — PR 1+2 of 6 in flight on `claude/splat-osint-ingestion`.

**Author + date:** Claude (for Jan), 2026-05-06.

**Branch:** `claude/splat-osint-ingestion`.

---

## Scope

SPLAT-1 ledger row (Section A of `ARCHITECTURE_ENTROPY_LEDGER.md`) was
**Aspirational, entropy 4** — `CamPlaneSplat` / `SplatPlaneSet` /
`SplatChannel` / `TetraSplatCertificate` lived only in
`.claude/knowledge/gaussian-splat-cam-plane-workaround.md` and
`.claude/plans/tetrahedral-epiphany-splat-integration-v1.md`, with
**zero hits in `crates/`**.

This plan ships **PR 1 + PR 2** of the 6-PR sequence laid out in
`gaussian-splat-cam-plane-workaround.md § Implementation sequence`,
moving SPLAT-1 from **Aspirational** to **Wired (x1, stage 1)**:

- **PR 1 — splat contract types.** Materialise `SplatChannel`,
  `CamPlaneSplat`, `SplatPlaneSet`, `AwarenessPlane16K`,
  `CamSplatCertificate`, `SplatDecision`, `TriadicProjection`,
  `ReasoningWitness64` in `lance-graph-contract::splat` (zero deps,
  no serde, no float on hot path).
- **PR 2 — EWA-sandwich Sigma-push-forward OSINT bridge.** A working
  `crates/jc/examples/osint_edge_traversal.rs` demonstrating the
  EWA-Sandwich kernel (PR #289 / `crates/jc/src/ewa_sandwich.rs`)
  performing a 5-hop OSINT chain traversal as a substitute for
  Cypher edge-hydration. Side-by-side vs naive convolution to
  certify the Sigma-push-forward viability for OSINT.

PR 1 and PR 2 ship together because the EWA example consumes the
new contract types — splitting them defeats the demo.

---

## Why

**Originating question (q2 PR #35 review):**
*"Can the Gaussian splat substitute neo4j edge hydration for new OSINT?"*

**Answer, in two layers:**

1. **Option 1 — shippable today (no contract change required).**
   The existing Cypher parser (`lance-graph::parser::parse_cypher_query`,
   1,932 LOC nom) plus the EWA-Sandwich Sigma-push-forward kernel
   (PR #289, `crates/jc/src/ewa_sandwich.rs`) already deliver
   neo4j-edge-hydration substitution: each hop SPD covariance
   pushes forward through the kernel, preserving PSD across 10 000
   /10 000 hops with a Koestenberger-Stark tightness bound of
   1.467x (PR #286, PR #287). For OSINT 5-hop chains this is well
   inside the certified regime.
2. **Option 2 — canonical path (this plan).**
   `gaussian-splat-cam-plane-workaround.md` is the architecturally
   correct surface: `ReasoningWitness64 + sigma + theta +
   projection lens -> semantic CAM plane deposition`, with
   per-channel hot pressure (support / contradiction / forecast /
   counterfactual / style / source) and exact 4096-cycle replay as
   cold authority. PR 1 + PR 2 of this plan ship the contract types
   plus the OSINT EWA demo; PRs 3-6 (queued) close the loop into
   BindSpace columns and the certificate-driven decision pipeline.

The `tetrahedral-epiphany-splat-integration-v1.md` plan covers the
SPOW tetrahedron / 4096x4096 question-surface side of this work; it
remains active and complementary. This plan is the **runtime-contract
slice** of the same architecture.

---

## Deliverables (D-ids)

| D-id | Title | Status | Owner |
|---|---|---|---|
| **D-SPLAT-1** | `crates/lance-graph-contract/src/splat.rs` — `SplatChannel` (6 variants), `CamPlaneSplat` (q8 amplitude / width / theta_accept + 16-byte witness identity + 8-byte `replay_ref`), `SplatPlaneSet` (6 channel planes), `AwarenessPlane16K` (256 x u64 = 2 KB pressure tile), `CamSplatCertificate` (q8 pressure measurements + replay decision), `SplatDecision` (Proceed / RequireExactReplay / PrefetchOnly / ScenarioOnly / Drop), `TriadicProjection`, `ReasoningWitness64`. 10 unit tests minimum. | **In PR** | contract-splat agent |
| **D-SPLAT-2** | `crates/jc/examples/osint_edge_traversal.rs` — EWA-Sandwich Sigma-push-forward demo for an OSINT 5-hop chain. Side-by-side vs naive convolution (Koestenberger-Stark tightness comparison). | **In PR** | ewa-osint-example agent |
| **D-SPLAT-3** | `witness_to_splat()` deterministic conversion (PR 2 of doc-sequence). Maps `(factor_a, factor_b, projection, ReasoningWitness64, sigma_idx, ThetaDecision, replay_ref)` -> `CamPlaneSplat` deterministically under fixed codebook + sigma codebook + theta policy + witness layout + seeds. | **Queued** | future PR |
| **D-SPLAT-4** | Splat deposition into BindSpace columns via `MergeMode::AlphaFrontToBack` lanes (PR 3 of doc-sequence). Q8 / bit-tile accumulation per Pillar-7 alpha-front-to-back-merge sink mode (B5 from PR #324). | **Queued** | future PR |
| **D-SPLAT-5** | `PlanarSplatBundle4096` with local (8-16 cycles) / short (64) / medium (512) / long (4096) bands (PR 4 of doc-sequence). | **Queued** | future PR |
| **D-SPLAT-6** | Semantic-CAM-distance integration (PR 5 of doc-sequence). Survivor tile selection compares against splatted pressure planes, not raw Hamming. | **Queued** | future PR |
| **D-SPLAT-7** | Replay fallback (PR 6 of doc-sequence). When `CamSplatCertificate` is insufficient (e.g. high support **and** high contradiction), load exact 4096-cycle ThoughtCycleSoA replay slice. | **Queued** | future PR |

---

## Test plan

```bash
# D-SPLAT-1 — splat contract types
cargo test -p lance-graph-contract --lib splat

# D-SPLAT-2 — OSINT edge traversal example
cargo run --example osint_edge_traversal --release \
    --manifest-path crates/jc/Cargo.toml
```

Acceptance gates:

- D-SPLAT-1: 10/10 unit tests pass; zero serde imports; zero float
  fields on hot path; `lance-graph-contract` total suite still green
  (175/175 -> 185/185 expected).
- D-SPLAT-2: example runs in `--release`; prints per-hop PSD-preserved
  Sigma trace + Koestenberger-Stark tightness bound vs naive-convolution
  baseline; all 5 hops PSD-preserved.

---

## Architectural compliance

- **I-VSA-IDENTITIES (iron rule).** `CamPlaneSplat` carries a
  16-byte witness identity (`ReasoningWitness64`) + 8-byte
  `replay_ref` — splats POINT TO content, never bundle content
  bits. The 6 channel planes are addressable by content identity,
  not by anonymous superposition.
- **Zero-dep contract preserved.** `lance-graph-contract` keeps its
  zero external-crate-dep invariant. `splat.rs` compiles with no
  serde, no nom, no datafusion.
- **No serde on types.** Wire formats are explicit. (CLAUDE.md
  Workspace Conventions rule 5.)
- **No floats on hot path.** All splat amplitudes / widths /
  apertures are q8 (`u8`). Float accumulation, if it ever appears,
  is confined to calibration paths.
- **Click P-1 method discipline.** Each splat type carries methods
  on its own state — `CamPlaneSplat::pressure_q8()`,
  `SplatPlaneSet::deposit(&CamPlaneSplat)`,
  `CamSplatCertificate::decide() -> SplatDecision`. No free
  functions on the carrier state. (CLAUDE.md The Click litmus.)
- **Pillar-6 math grounding.** EWA-Sandwich Sigma-push-forward is
  certified PSD-preserving across 10 000 hops (PR #289), with
  tightness 1.467x (PR #286 Koestenberger-Stark, PR #287
  Dueker-Zoubouloglou). PR 2 inherits this proof.
- **Pillar-7 deposition mode.** Splat deposition (queued, D-SPLAT-4)
  uses `MergeMode::AlphaFrontToBack` (PR #324, B5). The contract
  type ships in PR 1; the kernel wiring lands in PR 3 of the
  doc-sequence.

---

## Reference

- **Source spec:** `.claude/knowledge/gaussian-splat-cam-plane-workaround.md`
  (already-existing knowledge doc; do NOT modify in this plan).
- **Companion plan:** `.claude/plans/tetrahedral-epiphany-splat-integration-v1.md`
  (SPOW tetrahedron / 4096x4096 question surface — orthogonal axis
  of the same architecture).
- **EWA prover:** `crates/jc/src/ewa_sandwich.rs` (PR #289).
- **PSD-tightness bounds:** PR #286 (Koestenberger-Stark on Hadamard
  2x2 SPD) and PR #287 (Dueker-Zoubouloglou Hilbert-space CLT).
- **Pillar-7 sink mode:** PR #324 (`MergeMode::AlphaFrontToBack`,
  EWA Kerbl-2023 compositing in stage [7]).
- **Sigma-codebook column:** PR #323 (`FingerprintColumns.sigma` u8
  column, +1 byte/row).
- **Sigma-propagation contract:** PR #322 (`sigma_propagation.rs`
  promoted to contract — `Spd2`, `ewa_sandwich`,
  `log_norm_growth`, `pillar_5plus_bound`).
- **Ledger row resolved:** SPLAT-1 in
  `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md` Section A
  (entropy 4 -> 2, Aspirational -> Wired stage 1).

---

## Out of scope (this plan)

- Changes to `gaussian-splat-cam-plane-workaround.md` itself (the
  source-of-truth knowledge doc stays immutable here).
- BindSpace column wiring (deferred to D-SPLAT-4 / PR 3 of
  doc-sequence).
- 4096-band PlanarSplatBundle bands (deferred to D-SPLAT-5).
- Semantic-CAM-distance integration (deferred to D-SPLAT-6).
- Replay fallback (deferred to D-SPLAT-7).
- SPOW tetrahedron question-surface work (covered by
  `tetrahedral-epiphany-splat-integration-v1.md`).
