# Agent W7 — medcare-realtime/src/stack.rs

**Round:** 2 (Stage 2 — medcare-realtime skeleton)
**Owner:** crates/medcare-realtime/src/stack.rs (`MedCareStack`)
**Commit:** medcare-rs `ffa6c187`
**Status:** ✅ committed

## Action

`MedCareStack` empty-struct facade with:
- `new()` / `default()` / `Clone` / `Debug` (smoke-test surface)
- `domain_profile()` returning `StepDomain::MedCare.profile()`

5 tests covering smoke + 3 regulatory invariants:
- `medcare_audit_retention_meets_bmv_ae_57` (≥3650 days = 10 years)
- `medcare_requires_fail_closed` (medical safety)
- `medcare_auto_action_confidence_higher_than_smb_default` (>0.75)

## Compilation dependency surfaced

**Requires `StepDomain::MedCare` variant in
`lance-graph-contract::orchestration`.** If absent, this file fails to
compile — exposing a concrete upstream gap. Doc on `domain_profile()`
explicitly says "fall back to hand-constructed DomainProfile would
mask the gap" — fail loud rather than hide.

If Meta-2 surfaces this as a real upstream gap, the fix is a small
lance-graph PR adding the variant. Doesn't block this sprint's other
deliverables (W9-W12 don't depend on `domain_profile()`).

## Self-review

- ✅ Empty-struct v1 keeps public API surface stable as fields grow
- ✅ Regulatory invariants (BMV-Ä §57, fail-closed) pinned in tests
- ⚠️ No RlsPolicyRegistry yet (smb-realtime has it via DM-7 upstream
  feature gate). medcare side waits for DM-7 + medcare-rbac wiring.
- ⚠️ No `with_default_policies()` builder — defer until upstream
  medcare_ontology() ships canonical entity list.
