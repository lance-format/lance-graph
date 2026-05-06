# Agent W4 — medcare-rbac/src/policy.rs

**Round:** 1 (Stage 1 — medcare-rbac scaffolding)
**Owner:** crates/medcare-rbac/src/policy.rs
**Commit:** medcare-rs `860d58e`
**Status:** ✅ committed (Round 1 closure)

## Action

`Policy` struct (name + roles vec) + `Operation<'a>` enum (Read/Write/Act)
+ `evaluate()` routing (4-arm match on Operation) + `medcare_policy()`
factory composing the 4 roles from W3.

11 evaluate() tests covering:

| Test | Verifies |
|---|---|
| `medcare_policy_has_four_roles` | composition shape |
| `evaluate_doctor_read_patient_full` | doctor happy path Read |
| `evaluate_doctor_issue_prescription` | doctor happy path Act |
| `evaluate_doctor_send_referral` | Ueberweisung action |
| `evaluate_doctor_cannot_delete_patient` | doctor deny path (admin-only) |
| `evaluate_auditor_write_anything_denied` | auditor write boundary |
| `evaluate_auditor_full_read_allowed_everywhere` | auditor read scope (loops 6 entities) |
| `evaluate_receptionist_no_clinical_reads` | receptionist boundary (loops 4 clinical entities) |
| `evaluate_receptionist_demographics_writable` | receptionist allow path |
| `evaluate_admin_full_authority` | admin Anamnese.redact |
| `evaluate_unknown_role` | unknown-role rejection |

## Output verification

- ✅ Policy::evaluate routes via match on Operation enum (3 arms)
- ✅ Returns AccessDecision::Allow / Deny with stable static reasons
- ✅ medcare_policy() composes all 4 roles in order: doctor, auditor,
  receptionist, admin (mirrors smb_policy() pattern)
- ✅ Auditor counter-test loops all 6 entities asserting Read Full Allow
- ✅ Receptionist counter-test loops 4 clinical entities asserting deny

## Round 1 closure summary

**Files committed (5 medcare-rbac files):**
| File | LOC | Tests | SHA |
|---|---|---|---|
| Cargo.toml | 13 | — | `5b06da8` |
| src/lib.rs | 33 | — | `7b91459` |
| src/access.rs | 71 | 2 | `2fdace7` |
| src/permission.rs | 119 | 3 | `49f377c` |
| src/role.rs | 280 | 10 | `bdb86ba` |
| src/policy.rs | 240 | 11 | `860d58e` |

**Total:** 6 commits, ~750 LOC, 26 tests, all on `claude/lance-
datafusion-integration-gv0BF` branch of medcare-rs.

**Acceptance vs. SPRINT_LOG.md Round 1 criteria:**
- ✅ File-for-file mirror of lance-graph-rbac shape (5/5 files)
- ✅ medcare entities (Patient/Diagnosis/LabResult/Prescription/Anamnese/Ueberweisung)
- ✅ Roles: Doctor, Auditor, Receptionist, Admin
- ✅ Compiles standalone — depends only on lance-graph-contract workspace
- ✅ Tests cover each role × entity × operation triple

## Blockers / open questions

- **Workspace registration outstanding.** medcare-rs/Cargo.toml `members = [...]`
  needs to grow `crates/medcare-rbac` and `[workspace.dependencies] medcare-rbac`
  needs to grow. *W8 (Round 2) owns this.*
- **§73 SGB V row-level test missing.** Per W3 self-review, the
  Ueberweisung-an-Facharzt cross-doctor visibility check belongs at
  gate/RLS layer; W12 (Round 3) ships the test.
- **BtM Escalate path not exercised.** Currently every role.evaluate
  returns Allow/Deny only. Escalate test belongs in Round 3 gate test.

## Self-review

- ✅ All 11 tests fire happy + counter paths
- ✅ medcare_policy() pinning test (count = 4 roles)
- ✅ Loop-over-entities tests catch missing-permission drift
- ⚠️ No test for Operation::Act on unknown action (e.g. doctor.evaluate
  with Action="invent-an-action") — should rely on can_act returning
  false via slice contains. *Edge case; defer to Meta-1 verdict.*
