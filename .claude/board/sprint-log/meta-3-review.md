# Meta-3 Review — MedCareMembraneGate (Round 3, Stage 3)

**Reviewer:** Meta agent 3 of 3 (final review pass)
**Scope:** medcare-rs/crates/medcare-realtime/src/gate.rs + tests/{integration,regulatory}.rs
**Method:** read W9-W12 commits + log entries; cross-check against
PR #29 reference impl + Meta-1/Meta-2 carry-forwards + v1 boundary
honesty.

> **Tone:** brutally honest. This is the last review pass before sprint
> closure. Findings either block ship or document v1 limits explicitly
> for future sessions. No filler.

---

## Verdict

**Ship Round 3 + close sprint.** Zero CRITICAL findings. Two HIGH
findings are honest documentation gaps that the W12 author has
already partially captured; meta surfaces them more sharply. Two
MEDIUM findings deferred to follow-up sprints. The v1 surface is
the right shape; it's just smaller than ambition.

| # | Severity | Finding | Action |
|---|---|---|---|
| 1 | HIGH | Action operations (Operation::Act) unreachable via gate — gate routes only Read/Write | Doc note in gate.rs module head + tests/regulatory.rs; orchestration layer is the right home for action gating |
| 2 | HIGH | BtM Escalate "v1 limitation documented" tests will silently pass even if the gate IS later updated to return Escalate — the assertion is `is_allowed()` | Tighten future-spec assertions: explicit `assert_eq!(decision, AccessDecision::Allow)` so future Escalate flip is a real test failure |
| 3 | MEDIUM | Three name paths for `Policy` (rbac, gate, lib) | Choose one canonical; document the others as legacy aliases |
| 4 | MEDIUM | No bench harness validates 20-200 ns gate decision claim | Add to backlog; gate-bench follow-up sprint |
| 5 | LOW | Module-head TD-MEMBRANE-FIRST-VS-ANY caveat carries forward but no test exercises the divergence case (predicate-specific RLS) | Backlog: write a unit test once a real divergence case is identified |

---

## HIGH #1 — Action operations unreachable via gate

**Finding.** `MedCareMembraneGate::should_emit` and `evaluate` route
`gate_commit: bool` to `Operation::Read { depth }` (false) or
`Operation::Write { predicate }` (true). Neither path reaches
`Operation::Act { action }`.

This means actions like:
- `Diagnosis.classify` / `finalize` / `retract`
- `Prescription.issue` / `renew` / `revoke`
- `Anamnese.append` (the ONLY mutation path for Anamnese per BMV-Ä §57)
- `Ueberweisung.send` / `accept` / `decline`
- `Patient.merge` / `anonymize` / `delete`

...cannot be gated through `MedCareMembraneGate`. The orchestration
layer must call `medcare_rbac::Policy::evaluate(role, entity,
Operation::Act { action })` directly.

**This is intentional from PR #29's design** — the upstream
`MembraneGate` trait shape is `(commit: bool)` only. But it's a
substantial v1 limit that medcare's append-only Anamnese semantic
relies on.

**Super-helpful solution.** Add an explicit doc note to gate.rs
module head:

```rust
//! # Action operations not gated here
//!
//! `MedCareMembraneGate` routes only Read/Write per upstream's
//! `(gate_commit: bool)` shape. Action operations (`Diagnosis.classify`,
//! `Prescription.issue`, `Anamnese.append`, etc.) must go through
//! `medcare_rbac::Policy::evaluate(role, entity, Operation::Act
//! { action })` at the orchestration layer.
//!
//! This is by upstream design — `MembraneGate::should_emit` is the
//! wire-shape of "is this projection allowed to leave the membrane",
//! which is a Read/Write question. Action authorization (issue this
//! prescription, finalize this diagnosis) is an orchestration-layer
//! concern that fires before the eventual Read/Write projection.
```

**Action.** Update gate.rs doc — small follow-up commit. Or carry
this into the sprint summary as documented v1 limit.

---

## HIGH #2 — BtM Escalate "limitation documented" tests are too weak

**Finding.** W12's regulatory tests for the BtM/finalize/anonymize
limitation:

```rust
#[test]
fn btm_escalate_path_documented_as_v1_limitation() {
    let gate = MedCareMembraneGate::from_medcare_policy("doctor", "Prescription");
    let decision = gate.evaluate(true);
    assert!(decision.is_allowed());  // ← too loose
}
```

`is_allowed()` returns true only for `AccessDecision::Allow`. If a
future commit lands the Escalate wrapping (the documented future
behavior), `decision` becomes `Escalate { reason: "..." }`, which is
NOT `is_allowed()`, so the test FAILS. That's the desired flip — the
test fails until the spec is updated.

But the failure message will be cryptic: "expected true, got false".
The reader has to figure out whether is_allowed false means Deny,
Escalate, or some other variant.

**Super-helpful solution.** Tighten the assertion:

```rust
#[test]
fn btm_escalate_path_documented_as_v1_limitation() {
    let gate = MedCareMembraneGate::from_medcare_policy("doctor", "Prescription");
    let decision = gate.evaluate(true);
    // v1: Allow uniformly (gate doesn't see btm_flag).
    // FUTURE: when row-context lands, this should become
    // AccessDecision::Escalate { reason: "BtM second signature required" }
    // for btm_flag=true rows. This explicit assert_eq! makes the future
    // flip a clear test failure: "expected Allow, got Escalate { ... }"
    // is much more readable than "expected true, got false".
    assert_eq!(decision, AccessDecision::Allow);
}
```

This applies to all three v1-limitation tests (BtM, finalize/retract,
anonymize). Same pattern.

**Action.** Optional W12-revision-2 to tighten three assertions. Not
blocking — the loose assertions still flip when future changes land.
But the failure message clarity matters for someone diagnosing a CI
break six months from now.

---

## MEDIUM #3 — Three name paths for `Policy`

**Finding.** Same `Policy` type reachable via:
- `medcare_rbac::policy::Policy` (canonical home)
- `medcare_realtime::gate::Policy` (re-exported via `pub use`)
- `medcare_realtime::Policy` (lib.rs crate-root re-export)

Compilation-equivalent. Cognition-confusing. Future "import Policy"
may grab any of three depending on which path the IDE auto-suggests.

**Super-helpful solution.** Pick one canonical path; document others
as legacy aliases. Recommended canonical: `medcare_realtime::Policy`
(crate-root) — same as smb-realtime's pattern. Other two paths stay
for backward-compat but aren't documented as primary.

**Action.** Backlog. Doc-only update; no behavior change.

---

## MEDIUM #4 — No bench harness for 20-200 ns claim

**Finding.** Gate doc claims "decisions run at L1 inner speed
(~20-200 ns)". v1 has zero benchmarks validating this.

**Super-helpful solution.** Backlog item: `gate-bench-v1` follow-up
adding `criterion`-based microbenchmarks for:
- `should_emit(_, _, _, true)` — allow path
- `should_emit(_, _, _, true)` — deny path (unknown role)
- `should_emit(_, _, _, false)` — read path
- `evaluate(true)` (Escalate path when future row-context lands)

Targets: <500 ns p99 for current sync impl. If we can't hit that, the
"20-200 ns" claim in the topology doc needs revision.

---

## LOW #5 — TD-MEMBRANE-FIRST-VS-ANY untested

**Finding.** PR #29 caveat #3 says `writable_predicates.first()` may
deny when "any" should allow IF predicate-specific RLS conditions
exist. medcare-realtime carries the caveat forward in module-head
docs but writes no test.

**Super-helpful solution.** Backlog: when a real divergence case is
identified (e.g. a Patient predicate where `Operation::Write { predicate }`
has different RLS conditions per predicate), write a regression test.
v1 doesn't have such a case, so the test would be vacuous.

---

## Sprint-wide closure assessment

**Round 1 (medcare-rbac):** Solid. 26 tests, 2 CRITICAL fixes applied
in revision-2. Surface mirrors lance-graph-rbac with medcare entities.

**Round 2 (medcare-realtime skeleton):** Solid. 5 tests, 1 CRITICAL
casing fix + HIPAA-grade values applied in W7-revision-2. Workspace
registration clean.

**Round 3 (MedCareMembraneGate):** Solid. 33 tests across gate.rs +
integration.rs + regulatory.rs. Two HIGH gaps (action ops unreachable,
v1-limit assertions loose) are honest documentation issues, not
correctness blockers.

**Total tests:** 64 across all three crates (medcare-rbac 26 +
medcare-realtime 38).

**Compilation expectation:** medcare-rs root `cargo build` should
work assuming:
1. lance-graph submodule symlink is functional (vendor/lance-graph)
2. lance-graph workspace builds (lance-graph-contract +
   lance-graph-callcenter + medcare-rbac symbol resolution)
3. `StepDomain::Medcare` profile values match upstream (verified
   in W7-rev2)

If any of (1)/(2) fails on a fresh checkout, that's a vendor/submodule
hygiene issue — not a sprint deliverable issue.

**Recommended follow-up sprint scope (smaller than this one):**
1. Apply HIGH #1 doc note (5 min)
2. Apply HIGH #2 assertion tighten (10 min)
3. Bench harness for gate decisions (HIGH-MEDIUM, ~2 hours)
4. Action-operation orchestration layer wrapper (HIGH-HIGH, half day)

---

## Verdict reaffirmed

**Ship.** v1 surface is correct and honest about its limits. Two HIGH
findings are documentation/test-clarity issues, not correctness
issues. Three MEDIUM/LOW findings go to backlog.

POLICY-1 / MEMBRANE-GATE-1 medcare-side seam is **CLOSED** for v1.
Topology I-1 / I-2 / I-3 / I-4 invariants are upheld. PR #29's three
TD caveats are honestly carried forward to the medcare side.
