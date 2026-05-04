# lance-graph-rdf — FMA / SNOMED CT / RadLex import + named-graph context

> **Status:** plan, not implementation. v1.
> **Scope:** carve-out from `oxigraph-arigraph-cognitive-shader-soa-merge-v1` (parent brief).
> **Owner crate:** `crates/lance-graph-rdf/` (new).
> **Border position:** between upstream oxigraph public crates and `cognitive-shader-driver` SoA.

## Why

The cognitive shader's hot path needs an `ontology_context_id` lane so the same SPO row can live in multiple ontology contexts without semantic mud:

- `graph:fma` — anatomy (validated outer ontology)
- `graph:radlex` — radiology imaging labels (validated outer)
- `graph:snomed` — clinical findings (validated outer, license-gated)
- `graph:medcare-local` — local clinical claims
- `graph:ai-candidates` — speculative AI-generated
- `graph:doctor-corrections` — clinician-supervised
- `graph:histology` / `graph:final-befund` — outcome evidence
- `graph:billing` — financial, must not merge with clinical

Without this column, FMA tendon facts and billing rows end up in the same lane. With it, `(s,p,o)` is shape-stable across lanes; pressure patterns are computable.

`lance-graph-rdf` is the **smallest possible border crate** that:
1. Ingests RDF dumps (Turtle, N-Quads, RDF/XML, OWL)
2. Mints stable `OntologyContextId: u32` per RDF named graph
3. Mints stable `TermId: u64` per IRI within a context
4. Emits canonical `SemanticQuad` rows for downstream

It is deliberately **not** the cognitive shader, **not** the SoA storage, **not** the NARS reasoner, **not** the clinical DTO layer.

## What lance-graph-rdf owns vs disowns

**Owns:**
- Public crate import surface (oxrdf / oxttl / oxrdfxml / oxrdfio from crates.io)
- `OntologyContextId` (u32) and `NamedGraphRegistry` (IRI ↔ ContextId)
- `TermId` (u64) and `TermInterner` (per-context IRI ↔ TermId)
- `SemanticQuad` row type and `From<oxrdf::Quad>` compilation
- `OntologyManifest` (version, source URL, SHA, license SPDX, last-imported)
- Three importers: `fma`, `radlex`, `snomed`
- License gate for SNOMED (refuse load without affiliate attestation)
- Manifest-based re-load (do not double-register on restart)

**Disowns (explicit out-of-scope):**
- `SemanticSpoRow` / `SemanticSpoSoA` (those land in `cognitive-shader-driver/src/wire.rs` and `bindspace.rs` — separate F2 issue)
- AriGraph episode/source/witness/truth/replay lanes (added downstream of this crate)
- Clinical DTOs (`FrameDto`, `RoiEvidenceDto`, `FindingCandidateDto`) — those live in `medcare-bridge`
- SPARQL evaluation (we use DataFusion, not `spareval`)
- Inference / transitive closure (`regional_part_of` traversal etc.) — that's the cognitive shader's job
- Mapping table generation (UMLS, RadLex↔FMA, SNOMED↔RadLex crosswalks) — consumed as input from upstream tools, not generated here
- Ontology version diff / reconciliation (FMA 5.0 vs 5.1) — separate item

## Crate shape

### `Cargo.toml`

```toml
[package]
name = "lance-graph-rdf"
version = "0.1.0"
edition = "2024"

[dependencies]
# oxigraph public lib crates — ALL rocksdb-free
oxrdf         = "0.3"   # NamedNode, BlankNode, Literal, Triple, Quad, Dataset
oxsdatatypes  = "0.2"   # XSD primitive types
oxttl         = "0.2"   # Turtle / N-Triples / N-Quads / TriG
oxrdfio       = "0.2"   # unified parser façade (auto-dispatch by format)
oxrdfxml      = "0.2"   # RDF/XML for FMA OWL exports

# Workspace
lance-graph-contract = { workspace = true }  # canonical SemanticQuad re-export

# Standard
serde         = { version = "1", features = ["derive"] }
sha2          = "0.10"
thiserror     = "1"
tracing       = "0.1"

# DELIBERATELY ABSENT:
#   oxigraph (main crate) — depends on rocksdb
#   oxrocksdb-sys         — rocksdb FFI
#   spareval              — replaced by DataFusion in lance-graph
```

### Module tree

```
crates/lance-graph-rdf/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── context.rs       # OntologyContextId, NamedGraphRegistry
│   ├── term.rs          # TermId, TermInterner (per-context)
│   ├── quad.rs          # SemanticQuad row + From<oxrdf::Quad>
│   ├── manifest.rs      # OntologyManifest + sidecar persistence
│   ├── importers/
│   │   ├── mod.rs
│   │   ├── generic.rs   # Turtle / N-Quads loader (used by RadLex, SNOMED)
│   │   ├── fma.rs       # OWL/RDF-XML loader (FMA-specific)
│   │   ├── radlex.rs    # Turtle loader + FMA cross-ref resolution
│   │   └── snomed.rs    # N-Quads loader with affiliate-license gate
│   └── error.rs
└── tests/
    ├── fixtures/
    │   ├── fma-mini.rdf       # 50 anatomy nodes (synthetic, real FMA shape)
    │   ├── radlex-mini.ttl    # 50 imaging labels
    │   └── snomed-mini.nq     # 50 clinical concepts (synthetic, NOT real RF2)
    ├── roundtrip.rs           # SemanticQuad → Quad → SemanticQuad identity
    ├── deterministic.rs       # SHA-256 of materialised dataset stable across re-runs
    └── snomed_license_gate.rs # Refuses load without affiliate attestation
```

### Core types

```rust
/// Stable identifier for a named graph / ontology context.
/// Allocated at first registration, persisted in NamedGraphRegistry.
/// Per-deployment, not per-process — same IRI gets same ContextId across restarts.
#[derive(Copy, Clone, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct OntologyContextId(pub u32);

/// IRI ↔ OntologyContextId registry.
/// Persists to a Lance dataset sidecar; reloads on startup.
pub struct NamedGraphRegistry { /* ... */ }

/// Stable identifier for a term IRI.
/// Per-context — TermId(0) means different things in FMA vs SNOMED.
/// (See open question 2 on per-context vs global namespace.)
#[derive(Copy, Clone, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TermId(pub u64);

/// Per-context TermId interner.
pub struct TermInterner { /* ... */ }

/// The pre-shader ontology row.
/// Compiled from oxrdf::Quad; emitted to downstream cognitive-shader-driver.
/// No episode / witness / truth / replay lanes here — those land downstream.
#[repr(C)]
pub struct SemanticQuad {
    pub subject_id: TermId,                  // u64
    pub predicate_id: TermId,                // u64
    pub object_id: TermId,                   // u64 (or literal-ref via separate table)
    pub ontology_context_id: OntologyContextId,  // u32
}

/// Manifest entry per imported ontology.
/// Captures provenance + license + version for clinical audit.
pub struct OntologyManifest {
    pub context_iri: oxrdf::NamedNode,
    pub context_id: OntologyContextId,
    pub version: String,                     // "FMA 5.0.0", "RadLex 4.1", etc.
    pub source_url: String,
    pub source_sha256: [u8; 32],
    pub license_spdx: String,                // "CC-BY-3.0", "RadLex-Free", "SNOMED-Affiliate"
    pub commercial_use: bool,
    pub last_imported: chrono::DateTime<chrono::Utc>,
}
```

### Importer behaviour (shared shape)

```
1. Open source file (RDF/XML, Turtle, N-Quads) via oxrdfio
2. Stream parse Quads (do not buffer — ontologies can be 10M+ triples)
3. For each Quad { s, p, o, g }:
     a. Register g in NamedGraphRegistry → OntologyContextId
     b. Intern s/p/o IRIs in per-context TermInterner → TermIds
     c. Emit SemanticQuad
4. Materialise SemanticQuad rows to Lance dataset (columnar, append-only)
5. Write OntologyManifest sidecar
6. Re-runs: read existing manifest, skip already-imported (SHA match)
```

## The three importers (concrete)

### FMA — Foundational Model of Anatomy

| Field | Value |
|---|---|
| Source | BioPortal FMA OWL download (RDF/XML) |
| Format | RDF/XML (use `oxrdfxml`) |
| License | CC-BY-3.0 (open, commercial OK) |
| Size | ~80,000 anatomy classes, ~1.5M triples |
| Named graph | `<http://purl.org/sig/ont/fma>` |
| Canonical ContextId | `OntologyContextId(1)` (allocated at first registration) |
| Key relations | `regional_part_of`, `constitutional_part_of`, `member_of` |

### RadLex

| Field | Value |
|---|---|
| Source | RSNA RadLex download (Turtle) |
| Format | Turtle (use `oxttl`) |
| License | RadLex-Free (RSNA terms — commercial + non-commercial OK) |
| Size | ~75,000 imaging concepts, ~600K triples |
| Named graph | `<http://radlex.org/>` |
| Canonical ContextId | `OntologyContextId(2)` |
| Cross-refs | `RID:hasFmaId` style — resolves to FMA via TermInterner cross-context lookup |

### SNOMED CT

| Field | Value |
|---|---|
| Source | NHS / IHTSDO RF2 release converted to N-Quads upstream |
| Format | N-Quads (use `oxttl`) |
| License | SNOMED-Affiliate (per-country, attestation required) |
| Size | ~350,000 concepts core; ~2M+ triples |
| Named graph | `<http://snomed.info/sct>` |
| Canonical ContextId | `OntologyContextId(3)` |
| **License gate** | Importer **MUST** check `manifest.license_spdx == "SNOMED-Affiliate"` AND `manifest.commercial_use` matches deployment's affiliate status. Otherwise: refuse load with explicit `Error::SnomedLicenseNotAttested`. Tested as both a happy-path and refusal-path acceptance criterion. |

### Cross-reference context

A separate context for IRI-to-IRI mapping triples (`owl:sameAs`, RadLex↔FMA crosswalk, SNOMED↔RadLex):

| Field | Value |
|---|---|
| Named graph | `<http://medcare.local/mappings>` |
| Canonical ContextId | `OntologyContextId(99)` |
| Load order | **After** all participating ontologies, so cross-referenced TermIds resolve. |
| Source | UMLS-style mapping tables (consumed as input, generation out-of-scope). |

## Where lance-graph-rdf sits

```
        oxrdf, oxttl, oxrdfio, oxrdfxml, oxsdatatypes  (upstream public crates)
                                  │
                                  ▼
                        lance-graph-rdf  ← THIS CRATE
                          • NamedGraphRegistry
                          • TermInterner per context
                          • OntologyManifest
                          • SemanticQuad emission
                                  │
                                  ▼
                       lance-graph-contract
                          • re-exports SemanticQuad as canonical
                          • SoA columnar shape declared here
                                  │
                                  ▼
                     cognitive-shader-driver
                          • SemanticSpoRow / SemanticSpoSoA
                          • adds episode / source / witness / truth / replay
                                  │
                                  ▼
                     AriGraph + NARS (existing in lance-graph)
                          • episodic semantic memory + reasoning
```

## Acceptance criteria

- [ ] `crates/lance-graph-rdf` compiles standalone; `cargo tree` shows zero `oxigraph` (main), zero `oxrocksdb-sys`, zero `spareval` deps.
- [ ] FMA mini-fixture (50 anatomy nodes) imports; output SHA-256 stable across re-runs given same input.
- [ ] RadLex mini-fixture imports through same `importers::generic` Turtle path as FMA's OWL/XML path.
- [ ] SNOMED happy-path: with valid affiliate attestation in manifest, mini-fixture imports.
- [ ] SNOMED refusal-path: without `SNOMED-Affiliate` attestation, importer returns `Error::SnomedLicenseNotAttested`; nothing materialises.
- [ ] OntologyManifest sidecar written after import; re-running with same input is a no-op (SHA check).
- [ ] Cross-reference context (`<http://medcare.local/mappings>`) loaded **after** FMA + RadLex resolves all TermIds; cross-context lookup query (e.g. "all SNOMED concepts mapped to FMA `Tendon of supraspinatus`") returns expected set.
- [ ] Round-trip: `SemanticQuad → oxrdf::Quad → SemanticQuad` is identity (TermId stability, no IRI loss).
- [ ] Performance smoke: FMA-mini (50 nodes) imports in <100 ms; full FMA (~1.5M triples) imports in <30 s on stock dev box.
- [ ] No `unsafe` in this crate.

## Dependencies + phases

**Upstream public crates** (crates.io, no fork needed for v1):
`oxrdf` 0.3, `oxsdatatypes` 0.2, `oxttl` 0.2, `oxrdfio` 0.2, `oxrdfxml` 0.2

**Workspace internal:** `lance-graph-contract` (re-exports `SemanticQuad`)

**Soft downstream:** F2 issue (cognitive-shader-driver `SemanticSpoSoA` wiring) — consumes lance-graph-rdf output but is not blocked by this plan.

**Independent of:** A1 (lance-graph#331 TripletGraph promotion). lance-graph-rdf operates one layer **above** AriGraph triplets — it imports RDF named graphs, then downstream consumers compile those into AriGraph or shader rows.

### Phasing

| Phase | Scope | Effort |
|---|---|---|
| **P0 — scaffold** | New crate skeleton, deps wired, empty modules, smoke test (5-quad N-Triples → SemanticQuad). | Day 1 |
| **P1 — FMA importer** | `oxrdfxml`-based loader, TermInterner, manifest emission. Acceptance: full FMA import < 30s, deterministic SHA. | Week 1 |
| **P2 — RadLex importer** | `oxttl`-based loader through `importers::generic`. Cross-reference resolution to FMA. | Week 1–2 |
| **P3 — SNOMED + license gate** | `oxttl`-based loader for N-Quads form. License attestation gate. Both happy + refusal acceptance tests. | Week 2 |
| **P4 — Mapping context loader** | Generic loader for cross-reference context. Order discipline (dependent ontologies first). End-to-end query: SNOMED↔FMA cross-walk works. | Week 2–3 |

## Open questions

1. **Persistence backend** — Lance dataset (consistent with rest of stack) vs Arrow IPC files vs Parquet. Lance is the obvious answer; worth confirming the schema fits cleanly (mostly fixed-width u32/u64 + one bytes column for IRI strings on demand).
2. **TermId scope** — per-context (TermId(0..N) within each OntologyContextId) vs global (single u64 namespace across all contexts). Per-context is simpler and avoids ID exhaustion; global makes cross-context joins cheaper. **Recommend per-context**, with optional `(context_id, term_id) → global_id` index built lazily.
3. **IRI string storage** — keep IRI strings forever (so we can re-emit RDF for export) vs hash-only (compact, one-way). **Recommend keep** — clinical records need traceable provenance, IRI re-emission is a real export need.
4. **Versioning** — when FMA bumps to a new release, do we mint a new `OntologyContextId` (e.g. `OntologyContextId(11)` for FMA 5.1) or update in place? **Recommend new ContextId per major version**, with a manifest pointer chain. Old ContextId stays readable forever — clinical records reference the version they were minted against.
5. **AdaWorldAPI/oxigraph fork** — does the org fork carry patches that diverge from upstream `oxrdf`/`oxttl`/etc? If yes, swap `Cargo.toml` deps from crates.io to fork path. If no patches, drop the fork from the dependency graph entirely. **Decision needed before P0.**

## Reference

- Parent brief (this is a carve-out): `oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` (to be filed alongside)
- Clinical DTO architecture: `AnatomyTargetDto` carries `fma_id: Option<u32>`, `snomed_id: Option<u64>`, `radlex_id: Option<u32>` — these resolve through the `NamedGraphRegistry` + `TermInterner` provided here
- Architectural Commitment #1 (MedCare-rs CLAUDE.md): MySQL is permanent oracle. lance-graph-rdf does not write to MySQL; ontology contexts are read-side only.
- Architectural Commitment #4 (MedCare-rs CLAUDE.md): thinking lives only in lance-graph. lance-graph-rdf does no inference; importer only.
- External standards anchored: FMA (BioPortal), RadLex (RSNA), SNOMED CT (IHTSDO/NHS RF2). FHIR `ImagingStudy`/`Observation`/`DiagnosticReport` are downstream consumers, not in this crate.
