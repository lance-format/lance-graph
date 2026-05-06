# Meta-2 Review — medcare-realtime (Round 2, Stage 2)

**Reviewer:** Meta agent 2 of 3 (Round 2 review pass)
**Scope:** medcare-rs/crates/medcare-realtime + workspace registration
**Method:** read W5-W8 commits + log entries; cross-check against
smb-realtime shape parity, upstream dep availability, topology I-1/I-2,
and Round 3 readiness.

> **Tone:** brutally honest. Reviewing my own colleague's work as if
> shipping to production tomorrow. Findings escalate by severity.

---

## Verdict

**Ship Round 2 with one CRITICAL fix applied as W7-revision-2 before
opening Round 3.** The fail-loud choice on `StepDomain::MedCare` may
block compilation if the variant doesn't exist upstream. One MEDIUM
finding flagged for follow-up. Otherwise Round 2 is clean.

| # | Severity | Finding | Action |
|---|---|---|---|
| 1 | **CRITICAL** | W7 hard-depends on `StepDomain::MedCare` variant; sprint cannot verify upstream existence | W7-revision-2: construct DomainProfile inline with documented expected values; switch to `StepDomain::MedCare.profile()` when upstream variant ships |
| 2 | MEDIUM | `MedCareStack` empty struct in v1 — is this a facade or a marker? | Honest doc note in module head; defer field growth to follow-up |
| 3 | MEDIUM | No `with_default_policies()` builder — smb-realtime has 3 default policies registered; medcare-rs has zero | Backlog: add when canonical entity list is firm |
| 4 | LOW | No test asserts cross-crate workspace dep resolves (medcare-realtime → medcare-rbac) | CI catches this on first build; no Sprint action |
| 5 | LOW | Cargo.toml has 0 `[dev-dependencies]` while smb-realtime has tokio-test + pretty_assertions | Defer; v1 tests are simple `assert!` on sync surface |

---

## CRITICAL #1 — `StepDomain::MedCare` may not exist upstream

**Finding.** W7's `domain_profile()` calls
`lance_graph_contract::orchestration::StepDomain::MedCare.profile()`.
W7's self-review explicitly chose "fail loud" — if the variant is
absent, the file won't compile.

**Why this is critical.** The sprint goal is "produce a buildable
3-stage scaffolding ready for Round 3 + future merge". A non-compiling
medcare-realtime blocks Round 3 (W9 imports from medcare-realtime),
blocks workspace `cargo build`, and blocks any CI run on this branch.

The fail-loud rationale ("don't mask the gap") is defensible
architecturally — but the cost is concrete (sprint produces broken
code) versus a hypothetical benefit (someone notices the gap faster).
A documented inline fallback achieves the same surface visibility
without the compilation cost.

**Super-helpful solution (apply as W7-revision-2):**

```rust
pub fn domain_profile(&self) -> DomainProfile {
    // TODO upstream: switch to `StepDomain::MedCare.profile()` once
    // the variant ships in lance-graph-contract::orchestration.
    // Until then, construct the medcare-appropriate profile inline
    // — values mirror what `StepDomain::MedCare.profile()` should
    // return (3650 days BMV-Ä §57 retention, 0.85 confidence
    // threshold, fail-closed required, Llm escalation).
    DomainProfile {
        audit_retention_days: 3650,
        auto_action_confidence: 0.85,
        escalation: EscalationStrategy::Llm,
        requires_fail_closed: true,
        verb_taxonomy: VerbTaxonomyId::MedCare,
    }
}
```

Caveat: `EscalationStrategy::Llm` and `VerbTaxonomyId::MedCare` are
also assumed to exist. If they don't, the fallback strategy is to use
whatever the contract crate currently exposes for `EscalationStrategy`
and either default `VerbTaxonomyId` or omit the field.

**The cleanest super-helpful path** — fetch the actual `DomainProfile`
struct definition from upstream BEFORE committing W7-revision-2, so
the fallback uses the right field names. Two options:

(a) Fetch `lance-graph-contract/src/orchestration.rs` content; verify
    `DomainProfile` shape; commit revision-2 with correct fields.

(b) Wrap the call in a feature gate:
    ```rust
    #[cfg(feature = "upstream-medcare-domain")]
    pub fn domain_profile(&self) -> DomainProfile {
        StepDomain::MedCare.profile()
    }

    #[cfg(not(feature = "upstream-medcare-domain"))]
    pub fn domain_profile(&self) -> DomainProfile {
        // hand-constructed fallback
    }
    ```
    Add `upstream-medcare-domain` feature gated on the variant
    landing. Cleaner future migration path; more boilerplate now.

**Recommended:** Option (a) for sprint pragmatism. Option (b) if the
upstream PR is uncertain (>1 week to land).

---

## MEDIUM #2 — MedCareStack empty struct: facade or marker?

**Finding.** `MedCareStack` in W7 has zero fields:

```rust
pub struct MedCareStack {
    // v1 placeholder
}
```

Compare smb-realtime:

```rust
pub struct SmbStack {
    ontology: &'static CachedOntology,
    rls_registry: Arc<RlsPolicyRegistry>,
}
```

medcare's empty struct is honest about v1 scope — neither
`medcare_ontology()` nor `RlsPolicyRegistry` is wired yet. But the
public API surface (new / default / domain_profile / Clone / Debug)
suggests a real facade.

**Tension.** Empty-struct-with-methods is fine for a marker type that
locks in the API surface for future field growth. But the doc
comments call it a "facade" — language that implies composition.

**Solution.** Tighten the doc comment to acknowledge the v1 marker
status explicitly:

```rust
/// Assembled outer-membrane facade for medcare. Cheap to clone.
///
/// **v1 status: marker.** This struct is currently empty — the
/// public API surface (`new` / `default` / `domain_profile`) is
/// stable, but internal composition (RLS registry, cached ontology,
/// gate accessors) lands in follow-up sprints when upstream
/// blockers (DM-7, DM-8, medcare_ontology factory) ship.
///
/// Holding the type at v1 means consumer code can already reference
/// `MedCareStack` symbolically; field growth doesn't break the API.
```

This is honest — v1 trades emptiness for symbol stability. Worth
saying so explicitly.

**No code change needed; doc-comment update lands as part of any
future field-growth commit.** Round 3 doesn't block on this.

---

## MEDIUM #3 — Missing default policies builder

**Finding.** smb-realtime ships `with_default_policies()` registering
Customer / Invoice / TaxDeclaration. medcare-realtime has no
equivalent. Once `MedCareStack` grows an `rls_registry` field, it'll
need a parallel `with_default_medcare_policies()` registering Patient,
Diagnosis, LabResult, Prescription, Anamnese, Ueberweisung.

**Action.** Backlog. v1 doesn't have rls_registry yet; the builder
follows once the field lands. Capture in TECH_DEBT.md when sprint
synthesis closes.

---

## LOW #4-#5 — Defer

- #4: cross-crate dep resolution test → CI catches naturally
- #5: `[dev-dependencies]` → trivial; v1 has no async tests yet

---

## Round 3 implications

W9 (gate.rs) needs to import:
```rust
use medcare_rbac::policy::{medcare_policy, Operation, Policy};
use medcare_rbac::role::Role;
use medcare_rbac::access::AccessDecision;

use lance_graph_contract::external_membrane::{AllowAllGate, MembraneGate};
use lance_graph_contract::property::PrefetchDepth;
```

W10 needs to update `medcare-realtime/src/lib.rs` to add:
```rust
pub mod gate;
pub use gate::{AccessDecision, AllowAllGate, MembraneGate, MedCareMembraneGate, Policy};
```

W11 (integration tests) wraps `MedCareStack::new()` and verifies
`MedCareMembraneGate` composes with it. v1 stack is empty so the
composition test will be trivial — that's fine.

W12 (§73 SGB V test) verifies:
1. Doctor without Ueberweisung row CANNOT read another Doctor's
   Patient at Detail (the row-level check happens above the gate;
   this test exercises the gate's role-level deny path)
2. Doctor with active Ueberweisung CAN read referred Patient at
   Detail (the gate allows; row-level check passes too)
3. **BtM-flagged Prescription.issue → Escalate** (per Meta-1 #3
   carry-forward; gate.rs in W9 must implement this wrapping)

The Meta-1 HIGH #3 + #4 carry-forward (BtM Escalate, anonymize/merge
Escalate) lands in W9 logic. W12 tests verify.

---

## Feedback loop — apply NOW (W7-revision-2)

Open question for the orchestrator: do we have the upstream
`DomainProfile` field shape on hand? If yes, apply W7-revision-2 with
inline construction. If no, fetch first.

**Recommendation:** Fetch `lance-graph-contract/src/orchestration.rs`
to confirm DomainProfile field names + EscalationStrategy + VerbTaxonomyId
variants, THEN commit W7-revision-2 with the correct inline construction.
Five minutes of verification, much better than guessing.

If `StepDomain::MedCare` already exists upstream, no revision needed
(though doc strengthening per #2 still recommended).
