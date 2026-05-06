# Sprint Log — MedCare Policy Scaffolding (3 stages, 12 + 3 agents)

> **Append-only operational log for the medcare scaffolding sprint.**
> 12 worker agents (4 per stage × 3 stages) + 3 meta agents (1 per
> stage). Each agent appends a structured entry as a separate file
> in `agents/` (akin to `tee -a agent-NN.md`). Meta agents read the
> per-agent files and emit `meta-N-review.md` with brutally honest
> findings + super-helpful solutions, fed back into the next round.
>
> **Why this file exists.** The cca2a pattern says coordination state
> goes in append-only files visible to all agents. In the GitHub-MCP
> environment the equivalent of `tee -a /var/log/sprint.log` is a
> directory of per-agent commits in this folder. `git log` is the
> read interface for any session monitoring the sprint.

---

## Sprint manifest

**Goal:** close `MEDCARE_POLICY_GAP.md` Stages 1+2+3 in one sprint.
Lands `medcare-rbac` crate, `medcare-realtime` skeleton, and the
`MedCareMembraneGate` impl on the medcare-rs side. Mirrors PR #29
(`SmbMembraneGate`) for SMB but adapted to the medcare regulatory
context (§73 SGB V, BMV-Ä retention, Patient/Diagnosis/LabResult/
Prescription/Anamnese/Ueberweisung entity set).

**Branch:** `claude/lance-datafusion-integration-gv0BF` on both
`AdaWorldAPI/medcare-rs` (worker commits) and
`AdaWorldAPI/lance-graph` (sprint-log + meta reviews).

**References used by all agents:**
- `lance-graph/crates/lance-graph-rbac/{Cargo.toml,src/{lib,access,role,policy,permission}.rs}` — Stage 1 mirror target
- `smb-office-rs/crates/smb-realtime/{Cargo.toml,src/{lib,stack,gate}.rs}` — Stage 2+3 mirror target
- `smb-office-rs#29` `SmbMembraneGate` — Stage 3 reference impl
- `lance-graph/.claude/board/MEDCARE_POLICY_GAP.md` — scoping doc
- `lance-graph/.claude/board/SINGLE_BINARY_TOPOLOGY.md` — invariants
  the gate must respect (I-1 single binary, I-2 sync, I-3 BBB,
  I-4 distinct gates)

---

## Round 1 — medcare-rbac crate (Stage 1, ~300 LOC)

| Agent | File | Status |
|---|---|---|
| W1 | `medcare-rs/crates/medcare-rbac/Cargo.toml` | pending |
| W2 | `medcare-rs/crates/medcare-rbac/src/lib.rs` + `access.rs` | pending |
| W3 | `medcare-rs/crates/medcare-rbac/src/permission.rs` + `role.rs` | pending |
| W4 | `medcare-rs/crates/medcare-rbac/src/policy.rs` + `medcare_policy()` | pending |
| M1 | brutally honest review of Round 1 → `meta-1-review.md` | pending |

**Round 1 acceptance criteria:**
- Mirrors lance-graph-rbac shape file-for-file (Cargo.toml, lib.rs,
  access.rs, role.rs, policy.rs, permission.rs)
- Domain: medcare entities (Patient, Diagnosis, LabResult,
  Prescription, Anamnese, Ueberweisung)
- Roles: Doctor, Auditor, Receptionist, Admin (working set; M1 may
  surface §73 SGB V regulatory gaps for follow-up)
- Compiles standalone (no medcare-realtime dep yet)
- Tests cover each role × entity × operation triple

## Round 2 — medcare-realtime skeleton (Stage 2, ~200 LOC)

| Agent | File | Status |
|---|---|---|
| W5 | `medcare-rs/crates/medcare-realtime/Cargo.toml` | pending |
| W6 | `medcare-rs/crates/medcare-realtime/src/lib.rs` (no gate yet) | pending |
| W7 | `medcare-rs/crates/medcare-realtime/src/stack.rs` (`MedCareStack`) | pending |
| W8 | `medcare-rs/Cargo.toml` workspace member update + `medcare_rbac` workspace dep | pending |
| M2 | brutally honest review of Round 2 → `meta-2-review.md` | pending |

**Round 2 acceptance criteria:**
- Mirrors smb-realtime shape (Cargo.toml feature gates, lib.rs
  module map, stack.rs facade)
- `MedCareStack::domain_profile()` returns `StepDomain::MedCare.profile()`
  per upstream R3
- Workspace Cargo.toml lists both `medcare-rbac` and
  `medcare-realtime` as members
- Compiles with stub `MedCareStack` (no gate.rs yet)

## Round 3 — MedCareMembraneGate impl (Stage 3, ~300 LOC)

| Agent | File | Status |
|---|---|---|
| W9 | `medcare-rs/crates/medcare-realtime/src/gate.rs` (newtype + builders + impl) | pending |
| W10 | re-exports in `medcare-realtime/src/lib.rs` (gate types) | pending |
| W11 | integration tests (gate × MedCareStack composition) | pending |
| W12 | §73 SGB V referral-visibility test (Ueberweisung-an-Facharzt) | pending |
| M3 | brutally honest review against PR #29 + 3 TD caveats → `meta-3-review.md` | pending |

**Round 3 acceptance criteria:**
- `impl MembraneGate for MedCareMembraneGate` mirrors PR #29
- Builders: `new`, `from_medcare_policy`, `with_write_predicate`,
  `with_read_depth`, `evaluate`
- 11+ unit tests covering each role × entity × commit/read path
- §73 SGB V regulatory test covers referral-visibility (`Ueberweisung`
  → only the referred-to physician should read after acceptance)
- Three TD caveats from PR #29 captured (faculty-blind,
  escalate-lossy, first-vs-any)

---

## Coordination notes

- **Worker agents commit to `medcare-rs` branch** for code; **append
  to `lance-graph/.claude/board/sprint-log/agents/`** for log entries.
- **Meta agents commit ONLY to `lance-graph`**; their reviews land
  in `meta-N-review.md` files in this directory.
- **No cross-agent file conflicts:** each worker owns 1-2 distinct
  files; same-file collisions (e.g. lib.rs re-export in W6 vs W10)
  are sequenced (W10 happens after W6 in Round 3).
- **Append-only:** worker log entries are immutable. Revisions
  spawn a new log entry (e.g. `agent-W3-revision-2.md`).
- **Meta feedback loop:** each Meta posts blockers; if any are
  critical, a follow-up worker is spawned in the same round before
  closing it.

---

## Status timeline (mutable ONLY in the table above)

This file is the index. Per-agent log entries land in
`agents/agent-WN.md`; meta reviews land in `meta-N-review.md`.
The status column above is the only mutable surface here.
