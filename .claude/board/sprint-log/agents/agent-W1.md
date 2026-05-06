# Agent W1 — medcare-rbac/Cargo.toml

**Round:** 1 (Stage 1 — medcare-rbac scaffolding)
**Owner:** crates/medcare-rbac/Cargo.toml
**Commit:** medcare-rs `5b06da8`
**Status:** ✅ committed

## Action

Wrote `crates/medcare-rbac/Cargo.toml` mirroring lance-graph-rbac/Cargo.toml
shape with medcare-rs workspace inheritance:

```toml
[package]
name = "medcare-rbac"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true
description = "Role-based access control for the medcare cognitive stack — mirror of lance-graph-rbac for medcare entities (Patient/Diagnosis/LabResult/Prescription/Anamnese/Ueberweisung)."

[dependencies]
lance-graph-contract = { workspace = true }
```

## Output verification

- Single workspace dep (lance-graph-contract); zero external deps
- Convention matches medcare-rs (workspace inheritance, not literal values)
- Description references the 6 medcare entities for discoverability

## Blockers / open questions

- **Workspace registration deferred to W8** (Round 2). Until W8 lands the
  `medcare-rs/Cargo.toml` workspace member entry, `cargo build` on the
  medcare-rs root won't pick this crate up. The crate file itself is
  correct; the workspace plumbing is W8's responsibility.

## Self-review

- ✅ File-for-file mirror of lance-graph-rbac
- ✅ workspace = true inheritance pattern
- ✅ Zero external deps beyond contract
- ⚠️ No `[features]` block — should we have one for future regulatory toggles?
  *Defer to Meta-1 review.*
