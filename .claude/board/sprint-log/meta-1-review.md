# Meta-1 Review — medcare-rbac (Round 1, Stage 1)

**Reviewer:** Meta agent 1 of 3 (Round 1 review pass)
**Scope:** medcare-rs/crates/medcare-rbac (W1 + W2 + W3 + W4)
**Method:** read W1-W4 commits + log entries; cross-check against
lance-graph-rbac shape parity + §73 SGB V + BMV-Ä regulatory norms +
PR #29 SmbMembraneGate three-caveat surface.

> **Tone:** brutally honest. "Looks fine, ship it" reviews waste the
> author's time AND the consumer's. Every finding here is either a
> real correctness issue, a real regulatory issue, or a deferred
> concern with explicit rationale. No filler.

---

## Verdict

**Ship Round 1 with two CRITICAL fixes applied as W3-revision-2 before
opening Round 2.** Two fixes are correctness/regulatory blockers; the
remainder are useful-but-not-blocking findings recorded for backlog.

| # | Severity | Finding | Action |
|---|---|---|---|
| 1 | **CRITICAL** | Doctor role grants Full WRITE on Anamnese predicates | W3-revision-2: empty writable_predicates on Anamnese, keep only "append" action |
| 2 | **CRITICAL** | Receptionist has no Identity-read on clinical entities → fails safety (allergy lookup) | W3-revision-2: add Identity read on Patient + Diagnosis (for allergy lookup before scheduling) |
| 3 | HIGH | Doctor finalize/retract Diagnosis returns Allow; should escalate per medical liability conventions | Round 3 gate.rs wraps BtM + finalize/retract → Escalate |
| 4 | HIGH | Admin `anonymize` on Patient is GDPR Art.17 + §35 BDSG territory; current Allow is too permissive | Round 3 gate or follow-up: Escalate path for anonymize/merge |
| 5 | MEDIUM | Termin (appointment) entity missing — Receptionist primary workflow not covered | Backlog; add to entity catalogue v2 |
| 6 | MEDIUM | Recall/Erinnerung (Krebsvorsorge) entity missing | Backlog |
| 7 | MEDIUM | ePA (elektronische Patientenakte) entity missing — regulatory MUST for §73 | Backlog; cross-ref ePA spec |
| 8 | LOW | evaluate() emits no audit trail (regulatory compliance: KV billing audit) | Round 3 gate.rs: AuditSink hook |
| 9 | LOW | Versicherungstyp (PKV/GKV) modulation not present | Backlog (most installs are GKV-only) |
| 10 | LOW | "unknown role" reason cannot embed which role was unknown | Backlog; tracing integration |

---

## CRITICAL #1 — Doctor.Anamnese Full write violates BMV-Ä §57

**Finding.** W3 role.rs grants the Doctor role:

```rust
PermissionSpec::full(
    "Anamnese",
    &[
        "complaint",
        "family_history",
        "social_history",
        "medication_history",
    ],
    &["append"],
)
```

This grants Full WRITE on every Anamnese predicate. Anamnese is
append-only by regulation (BMV-Ä §57 retention + ärztliche
Schweigepflicht implementation). Granting a doctor the ability to
overwrite `social_history` lets a malicious or pressured actor revise
patient narratives retroactively — exactly what the append-only
discipline forbids.

The W3 author flagged this themselves in agent-W3.md self-review under
"open questions" #1 ("Anamnese append-only modelling") — the issue is
acknowledged but the fix is deferred to "future iteration". That's
not adequate. **Doctor's Anamnese permission must be expressed as
"actions only, no predicates" in v1**, not a TODO.

**Super-helpful solution (apply as W3-revision-2):**

```rust
PermissionSpec::full(
    "Anamnese",
    &[],          // ← empty: no predicate-level writes
    &["append"],  // ← only the append action
)
```

This requires the consumer to go through the `append` action to add
narrative content. The action call site is where the orchestration
layer can enforce "create new row, never UPDATE old row" semantics.
The role-level API now correctly says "Doctor can append, not edit."

Admin keeps Full write on Anamnese (admin/data-protection-officer can
redact for GDPR compliance — that's the intended escape hatch).

**Why this is critical, not high.** A v1 ship with Doctor.Anamnese
predicate-write enabled bakes a regulatory non-conformity into the
public RBAC API. Future consumers will rely on the surface. Fixing
later requires a breaking change. Fix in v1 = $0; fix in v2 = N
consumers' code.

---

## CRITICAL #2 — Receptionist clinical-blind fails safety

**Finding.** W3 role.rs grants Receptionist exactly two permissions:

```rust
Role::new("receptionist")
    .with_permission(PermissionSpec::full("Patient", &["phone", "email", "address"], &[]))
    .with_permission(PermissionSpec::read_at("Ueberweisung", PrefetchDepth::Detail))
```

This means a receptionist scheduling an appointment cannot:
- See if the patient has a known anaphylaxis allergy (`Patient.allergies`
  is Full-read-only for Doctor/Auditor)
- See if the patient has a chronic condition relevant to scheduling
  (e.g. diabetes patient must come fasting before lab draw — needs
  Diagnosis.icd10_code visibility at Identity)
- See if a recent Lab is pending (needs LabResult.status at Identity)

The receptionist test `receptionist_demographics_only` actively asserts
no clinical reads at Identity depth. **That assertion encodes a safety
hazard** — real-world MFA workflow needs Identity-read on clinical
entities for triage.

**Super-helpful solution (apply as W3-revision-2):**

```rust
pub fn receptionist() -> Role {
    use lance_graph_contract::property::PrefetchDepth;
    // MFA needs Identity-read on clinical entities for safe scheduling:
    // - Patient.allergies (anaphylaxis check)
    // - Diagnosis.icd10_code (triage relevant conditions)
    // - LabResult.status (don't schedule lab draw if one is pending)
    // Full clinical content (Anamnese narrative, LabResult value) stays
    // gated to Doctor/Auditor.
    Role::new("receptionist")
        .with_permission(PermissionSpec::full(
            "Patient",
            &["phone", "email", "address"],
            &[],
        ))
        .with_permission(PermissionSpec::read_at("Patient", PrefetchDepth::Identity))
        .with_permission(PermissionSpec::read_at("Diagnosis", PrefetchDepth::Identity))
        .with_permission(PermissionSpec::read_at("LabResult", PrefetchDepth::Identity))
        .with_permission(PermissionSpec::read_at("Ueberweisung", PrefetchDepth::Detail))
}
```

**Caveat for the implementer.** The above has TWO `Patient` permissions —
one with Full demographics + one with Identity. The current
`Role::permission_for` returns the FIRST matching entity, so order
matters. Either:

- (a) Reorder so Identity comes first (loses demographic writes)
- (b) Merge into a single PermissionSpec with `max_depth: Detail` and
  the 3 demographic writable predicates
- (c) Extend `Role::permission_for` to merge across multiple matches

The cleanest fix is **(b) — merge into a single PermissionSpec**:

```rust
.with_permission(PermissionSpec {
    entity_type: "Patient",
    max_depth: PrefetchDepth::Detail,  // demographics ARE Detail
    writable_predicates: &["phone", "email", "address"],
    allowed_actions: &[],
})
.with_permission(PermissionSpec::read_at("Patient", PrefetchDepth::Identity))  // remove this — covered by Detail
```

Wait — Detail subsumes Identity (per the can_read_at impl `depth <= max_depth`).
So the single-permission form already covers Identity-read on Patient:

```rust
.with_permission(PermissionSpec {
    entity_type: "Patient",
    max_depth: PrefetchDepth::Detail,
    writable_predicates: &["phone", "email", "address"],
    allowed_actions: &[],
})
.with_permission(PermissionSpec::read_at("Diagnosis", PrefetchDepth::Identity))
.with_permission(PermissionSpec::read_at("LabResult", PrefetchDepth::Identity))
.with_permission(PermissionSpec::read_at("Ueberweisung", PrefetchDepth::Detail))
```

Plus update test `receptionist_demographics_only` — it currently
asserts NO clinical reads. After the fix, it should assert "Identity
clinical reads, no Full clinical reads, no clinical writes".

**Why this is critical, not high.** A v1 ship with Receptionist
clinical-blind ships a workflow-breaking RBAC design that real MFAs
will route around (probably by sharing Doctor credentials — exactly
the scenario RBAC exists to prevent). Fixing later means re-auditing
real-world deployments.

---

## HIGH #3 — Diagnosis finalize/retract should be Escalate

**Finding.** W3 grants doctor:

```rust
PermissionSpec::full(
    "Diagnosis",
    &["icd10_code", "severity", "onset_date", "status", "notes"],
    &["classify", "finalize", "retract"],
)
```

`finalize` is the moment of medical-legal commitment. Once a
diagnosis is finalized, it's the basis for billing, pharmacy claims,
and insurance disputes. `retract` is rolling that commitment back —
non-trivial action that affects downstream claims.

Current code returns Allow for both. Clinical governance norm: at
minimum log the action; ideally require attending-physician second
signature or audit-team review.

**Super-helpful solution (defer to Round 3 gate.rs).** The role-level
permission grants the underlying capability; the gate.rs decision
layer wraps `finalize` and `retract` actions with Escalate when
clinical governance config requires it. v1 default could be Allow
(matches current behaviour), but the gate API surface accepts a
governance-config struct that flips Escalate on. Cleanly extends.

**Action.** No Round 1 revision needed. Round 3 gate impl (W9) must
implement the wrapping; W12 test must verify.

---

## HIGH #4 — admin.anonymize is GDPR Art.17 territory

**Finding.** Admin role has action `anonymize` on Patient. Anonymization
is irreversible per GDPR Article 17. Allowing it via simple Allow
means the admin can permanently destroy a patient record's identity
binding — no recovery, no audit, no second-eye check.

The same concern applies to `merge` (which logically deletes the
secondary record) and `delete` everywhere.

**Super-helpful solution (defer to Round 3 gate.rs).** Same pattern as
#3 — the role grants capability, the gate decision layer wraps with
Escalate (require data-protection-officer signoff). Document the
escalation expectation in the role doc.

**Action.** No Round 1 revision needed. Round 3 gate impl must include
this in the wrapping. Add `data_protection_officer` role to v2
backlog.

---

## MEDIUM #5–#7 — Missing entities (Termin, Recall, ePA)

**Defer to v2 entity-catalogue extension.** Each is real but each is
non-blocking for Round 2/3. Capture in TECH_DEBT.md / IDEAS.md so
the gap is visible.

**Quick recommended prioritization:**
- Termin (appointment scheduling) — P-1 (Receptionist's primary tool)
- ePA (elektronische Patientenakte) — P-2 (regulatory mandate)
- Recall/Erinnerung — P-3 (Krebsvorsorge tracking)

---

## MEDIUM #8 — evaluate() audit trail

**Finding.** Every `Policy::evaluate()` call returns AccessDecision but
emits no audit log. KV billing review (auditor role's primary use case)
expects an "every-access-logged" trail.

**Solution.** Round 3 gate.rs `MedCareMembraneGate::evaluate` wraps
`policy.evaluate` and emits an audit entry to `lance-graph-callcenter`'s
AuditSink (when audit-log feature is on). Doesn't block Round 2.

---

## LOW #9-#10 — Defer

- #9 PKV/GKV modulation: real but most deployments are GKV-only
- #10 dynamic reason strings: needs tracing integration; v2 with
  proper observability story

---

## Round 2 implications

W5 (medcare-realtime/Cargo.toml) needs `medcare-rbac = { workspace = true }` dep.

W8 (workspace Cargo.toml update) needs:
- `members = [..., "crates/medcare-rbac", "crates/medcare-realtime"]`
- `[workspace.dependencies] medcare-rbac = { path = "crates/medcare-rbac" }`

## Round 3 implications

- W9 (gate.rs) MUST implement Escalate wrapping for:
  - Prescription.issue when btm_flag=true → Escalate "BtM second signature required"
  - Diagnosis.finalize / retract → Escalate "clinical governance review" (gated by config; default Allow)
  - Patient.anonymize / merge / delete → Escalate "data-protection-officer signoff"
- W12 (§73 SGB V test) MUST cover:
  - Doctor without Ueberweisung row CANNOT read another Doctor's Patient
  - Doctor with active Ueberweisung CAN read referred Patient at Detail
  - BtM-flagged Prescription.issue returns Escalate (not Allow)

---

## Feedback loop — apply NOW (W3-revision-2)

The two CRITICAL findings get a follow-up commit on the same Round-1
medcare-rbac surface. After that lands, Round 1 is closed and Round 2
opens.

Concrete diff for revision-2:

```diff
 pub fn doctor() -> Role {
     Role::new("doctor")
         // ...
         .with_permission(PermissionSpec::full(
             "Anamnese",
-            &[
-                "complaint",
-                "family_history",
-                "social_history",
-                "medication_history",
-            ],
+            &[],  // append-only via action; no predicate writes
             &["append"],
         ))
         // ...
 }

 pub fn receptionist() -> Role {
     use lance_graph_contract::property::PrefetchDepth;
+    // MFA needs Identity-read on clinical entities for safe
+    // scheduling: Patient.allergies, Diagnosis.icd10_code, LabResult.status.
     Role::new("receptionist")
-        .with_permission(PermissionSpec::full(
-            "Patient",
-            &["phone", "email", "address"],
-            &[],
-        ))
-        .with_permission(PermissionSpec::read_at(
-            "Ueberweisung",
-            PrefetchDepth::Detail,
-        ))
+        .with_permission(PermissionSpec {
+            entity_type: "Patient",
+            max_depth: PrefetchDepth::Detail,
+            writable_predicates: &["phone", "email", "address"],
+            allowed_actions: &[],
+        })
+        .with_permission(PermissionSpec::read_at("Diagnosis", PrefetchDepth::Identity))
+        .with_permission(PermissionSpec::read_at("LabResult", PrefetchDepth::Identity))
+        .with_permission(PermissionSpec::read_at("Ueberweisung", PrefetchDepth::Detail))
 }
```

Plus test fixes:
- `receptionist_demographics_only` → split into `receptionist_demographics_writable`
  + `receptionist_clinical_identity_read_only`

Apply as W3-revision-2.
