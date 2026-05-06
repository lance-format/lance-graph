# Agent W11 — medcare-realtime/tests/integration.rs

**Round:** 3 (Stage 3)
**Owner:** crates/medcare-realtime/tests/integration.rs (7 tests)
**Commit:** medcare-rs `cec95f5`
**Status:** ✅ committed

## Action

Integration tests living outside lib.rs (the `tests/` directory in
Cargo treats each .rs file as a separate test crate, exercising only
the public API). 7 tests covering gate × stack composition.

## Coverage

| Test | Verifies |
|---|---|
| stack_and_gate_compose | smoke test for canonical wiring |
| doctor_gate_full_clinical_authority | doctor reads all 6 entities (loop) |
| auditor_gate_read_only_across_entities | auditor split, all 6 entities (loop) |
| unknown_role_uniformly_denied | unknown role denial, all 6 entities (loop) |
| admin_gate_full_authority_including_anamnese_redact | GDPR Art.17 path |
| doctor_anamnese_predicate_write_denied_via_gate | BMV-Ä §57 (loop over 4 predicates) |
| stack_clone_is_cheap | clone preserves domain_profile |

## Self-review

- ✅ Loop-over-entity tests catch regressions on missing permissions
- ✅ All public-API symbols imported via `medcare_realtime::*` (no
  back-channel into internal modules)
- ⚠️ No performance test for the 20-200 ns claim — defer to bench harness
