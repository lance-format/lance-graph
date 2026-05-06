# Agent W3 — medcare-rbac src/permission.rs + src/role.rs

**Round:** 1 (Stage 1 — medcare-rbac scaffolding)
**Owner:** crates/medcare-rbac/src/permission.rs + src/role.rs
**Commits:** medcare-rs `49f377c` (permission.rs) + `bdb86ba` (role.rs)
**Status:** ✅ committed

## Action

### File 1 — src/permission.rs (3725 bytes)

`PermissionSpec` struct + 3 builders (`read_only` / `full` / `read_at`)
+ 3 predicates (`can_read_at` / `can_write` / `can_act`). Identical
shape to upstream — domain-agnostic at this layer. Tests adapted to
medcare entities (Patient, Prescription, btm_flag) but logic identical.

### File 2 — src/role.rs (12071 bytes — the bulk of Round 1)

Four medcare role factories covering 6-entity catalogue:

| Role | Patient | Diagnosis | LabResult | Prescription | Anamnese | Ueberweisung |
|---|---|---|---|---|---|---|
| `doctor()` | Full (5 demo predicates) | Full + 3 actions | Full + 2 actions | Full + 3 actions | Full + append | Full + 3 actions |
| `auditor()` | Read Full | Read Full | Read Full | Read Full | Read Full | Read Full |
| `receptionist()` | Full (3 demo predicates) | — | — | — | — | Read Detail |
| `admin()` | Full + 3 actions (incl. delete) | Full + 4 actions | Full + 4 actions | Full + 4 actions | Full + redact | Full + 4 actions |

10 unit tests cover each role × entity decision (doctor_can_classify,
auditor_reads_full, receptionist_demographics_only, admin_can_do_everything,
+ counter-tests for boundary cases like doctor_cannot_delete_anything).

## Output verification

- ✅ Role struct: name + permissions vec, builder pattern (matches upstream)
- ✅ can_read / can_write / can_act delegate via permission_for() lookup
- ✅ §73 SGB V row-level note documented in module head
- ✅ All 4 role factories return Role with at least one permission entry

## Blockers / open questions

**Surfaced for Meta-1 review:**

1. **Anamnese append-only modelling.** The Doctor role grants `Full`
   write on Anamnese predicates (complaint, family_history, etc.) and
   action `append`. But Anamnese is logically append-only (BMV-Ä
   retention). Current model relies on the consumer to interpret
   "append" as the only allowed mutation. A future iteration may want
   to model "append-only" as a permission-spec attribute (third axis)
   rather than relying on convention. *Pending Meta-1 verdict.*

2. **BtM (controlled substance) escalation.** Doctor role grants
   action `issue` on Prescription, INCLUDING when btm_flag is true.
   The escalation surface lives at evaluate() level (returning Escalate
   decision), not at role-grant level. But `policy.rs::evaluate` doesn't
   currently return Escalate for BtM — it returns Allow. *Round 3
   gate.rs needs to wrap the BtM-flagged check.*

3. **§73 SGB V cross-doctor visibility.** Doctor role grants Full read
   on Patient, but per §73 SGB V cross-doctor visibility requires an
   active Ueberweisung row. Current model treats this as gate/RLS
   concern (correct architectural call) but the test surface here
   doesn't exercise it. *Round 3 (W12) ships the per-row Ueberweisung
   visibility test.*

## Self-review

- ✅ 10 tests cover happy + counter paths for each role
- ✅ Doctor cannot_delete_anything counter-test pins the admin-only
  delete invariant
- ✅ Doctor cannot_write_lab_test_value test pins the lab-system-as-source
  invariant
- ⚠️ Receptionist has only ~2 permissions; thin coverage. Defensible
  (MFA is a narrow role) but Meta-1 may want broader scheduling
  predicates.
- ⚠️ No test exercises `Escalate` return path — by construction medcare
  roles never escalate at the role layer; escalation is decision-layer.
  *Round 3 gate test (W12) should fire BtM Escalate path.*
