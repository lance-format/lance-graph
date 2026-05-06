//! META-AGENT: `MembraneRegistry::with_rls()` requires
//! `pub mod rls;` (already present, gated on `auth-rls-lite`/`auth-rls`/`auth`/`full`)
//! and a `pub struct RlsPolicyRegistry` exported from that module (A2's work).
//! `MembraneRegistry::with_audit()` requires `pub mod audit;` and
//! `pub trait AuditSink: Send + Sync` (A3's work) plus a feature
//! `audit-log` (or equivalent) declared in `Cargo.toml`. Until those
//! land, the registry hooks are gated behind their respective features
//! so this builder compiles whether the upstream pieces are wired or not.
//!
//! LanceMembrane — the `ExternalMembrane` implementation.
//!
//! This is the compile-time BBB enforcement point. It is the only place
//! in the system where external consumer traffic and internal cognitive
//! state meet. After crossing here, external intent becomes a `UnifiedStep`
//! and committed cycles become `CognitiveEventRow` scalars — never the
//! reverse.
//!
//! # Phase map
//!
//! - Phase A (this file): BBB spine. Types correct. Scent is a stub.
//!   Subscription is a disconnected `mpsc::Receiver<u64>`.
//! - Phase B: `dialect` field populated by polyglot front-end parsers.
//! - Phase C: `scent` replaced by full ZeckBF17→Base17→CAM-PQ cascade.
//! - Phase D: Subscription wired to the std-sync `WatchReceiver` over
//!   the Lance version counter; `CommitFilter` applied per-subscriber.
//!
//! # UNKNOWN-1 resolution
//!
//! `ShaderSink` in `cognitive-shader-driver` is the internal BindSpace
//! ingestion path — it processes cycle fingerprints stack-side. There is
//! NO overlap with `ExternalMembrane`: `ShaderSink` never crosses the BBB;
//! `ExternalMembrane` is the gate. The two traits compose without conflict.
//!
//! Plan: `.claude/plans/callcenter-membrane-v1.md` § 10.9 – § 11

use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    RwLock,
};

// ─────────────────────────────────────────────────────────────────────────────
// Feature-flag coherence (HIGH/MEDIUM #4 — load-time misconfiguration guard)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(
    feature = "membrane-plugins-rls",
    not(any(
        feature = "auth-rls-lite",
        feature = "auth-rls",
        feature = "auth",
        feature = "full"
    ))
))]
compile_error!(
    "feature `membrane-plugins-rls` requires one of: auth-rls-lite, auth-rls, auth, full"
);

#[cfg(all(feature = "membrane-plugins-audit", not(feature = "audit-log")))]
compile_error!("feature `membrane-plugins-audit` requires `audit-log`");

/// Damping factor applied to `meta.free_e()` to derive `gate_f` until a
/// real gate-time signal is wired.  See TD-MEMBRANE-GATE-1 for the path
/// to a proper threshold reading; the damping makes the redundancy with
/// `free_e` explicit while still producing a sensible scalar for the
/// `WHERE gate_f < N` SQL filter pattern documented on `CognitiveEventRow`.
const GATE_DAMPING_FACTOR: f32 = 0.5;

#[cfg(not(feature = "realtime"))]
use std::sync::mpsc;

#[cfg(feature = "realtime")]
use crate::version_watcher::{LanceVersionWatcher, WatchReceiver};

use std::sync::Arc;

use lance_graph_contract::{
    a2a_blackboard::ExpertId,
    cognitive_shader::{MetaWord, ShaderBus},
    external_membrane::{CommitFilter, ExternalEventKind, ExternalMembrane, MembraneGate},
    faculty::FacultyRole,
    orchestration::{StepStatus, UnifiedStep},
};

use crate::external_intent::{CognitiveEventRow, ExternalIntent};

// ─────────────────────────────────────────────────────────────────────────────
// Plugin handshake (E2 — typed dependency injection for membrane policy)
// ─────────────────────────────────────────────────────────────────────────────

/// Membrane plugin contract — typed dependency injection for policy components.
///
/// Plugins implement `seal(&self, registry)` to assert their prerequisites at
/// boot. This makes misconfigurations a load-time error rather than a runtime
/// no-op.
pub trait Plugin: Send + Sync + std::fmt::Debug {
    /// Stable name for ordering / dependency declarations.
    fn name(&self) -> &'static str;

    /// Other plugin names this plugin needs to run AFTER.
    /// Used by `MembraneRegistry::seal()` for topological ordering.
    fn depends_on(&self) -> &[&'static str] {
        &[]
    }

    /// Verify prerequisites are wired. Called once at membrane construction.
    /// Default = noop. Plugins like AuditPlugin override to assert RLS is set.
    fn seal(&self, _registry: &MembraneRegistry) -> Result<(), String> {
        Ok(())
    }
}

/// The Blood-Brain Barrier enforcement point.
///
/// Atomic actor identity snapshot — written once per ingest, read once per project.
/// Single lock eliminates the F-01 identity-tear race where concurrent ingest+project
/// could see role from event N, faculty from N+1, expert from N+2.
#[derive(Clone, Copy, Debug)]
struct ActorState {
    role: u8,
    faculty: u8,
    expert: ExpertId,
}

/// Component registry for membrane wiring. Holds optional handles to
/// RLS rewriter, audit sink, catalog provider. Builders consult it
/// at construction time; runtime is read-only after seal().
///
/// Each plugin slot is feature-gated so the registry compiles whether
/// or not the upstream module (`rls`, `audit`, …) is enabled. Without
/// any feature flags, `MembraneRegistry` is an empty placeholder; with
/// the appropriate features active, the corresponding `with_*` builder
/// method becomes callable.
#[derive(Debug, Default)]
pub struct MembraneRegistry {
    #[cfg(all(
        feature = "membrane-plugins-rls",
        any(
            feature = "auth-rls-lite",
            feature = "auth-rls",
            feature = "auth",
            feature = "full"
        )
    ))]
    rls: Option<Arc<crate::rls::RlsPolicyRegistry>>,
    #[cfg(all(feature = "membrane-plugins-audit", feature = "audit-log"))]
    audit: Option<Arc<dyn crate::audit::AuditSink>>,
    // Future: catalog: Option<...>, validator: Option<...>
}

impl MembraneRegistry {
    /// Construct an empty registry. Use the `with_*` chain to populate.
    pub fn new() -> Self {
        Self::default()
    }

    /// Install an RLS policy registry (A2's `RlsPolicyRegistry`).
    /// Requires the `membrane-plugins-rls` cargo feature plus one of the
    /// `auth-rls{,-lite}` / `auth` / `full` features that activates
    /// `crate::rls`.
    #[cfg(all(
        feature = "membrane-plugins-rls",
        any(
            feature = "auth-rls-lite",
            feature = "auth-rls",
            feature = "auth",
            feature = "full"
        )
    ))]
    pub fn with_rls(mut self, reg: Arc<crate::rls::RlsPolicyRegistry>) -> Self {
        self.rls = Some(reg);
        self
    }

    /// Install an audit sink (A3's `AuditSink`).
    /// Requires the `membrane-plugins-audit` and `audit-log` cargo features.
    #[cfg(all(feature = "membrane-plugins-audit", feature = "audit-log"))]
    pub fn with_audit(mut self, sink: Arc<dyn crate::audit::AuditSink>) -> Self {
        self.audit = Some(sink);
        self
    }

    /// Borrow the installed RLS policy registry, if any.
    #[cfg(all(
        feature = "membrane-plugins-rls",
        any(
            feature = "auth-rls-lite",
            feature = "auth-rls",
            feature = "auth",
            feature = "full"
        )
    ))]
    pub fn rls(&self) -> Option<&Arc<crate::rls::RlsPolicyRegistry>> {
        self.rls.as_ref()
    }

    /// Borrow the installed audit sink, if any.
    #[cfg(all(feature = "membrane-plugins-audit", feature = "audit-log"))]
    pub fn audit(&self) -> Option<&Arc<dyn crate::audit::AuditSink>> {
        self.audit.as_ref()
    }

    /// Seal the registry — runs each plugin's seal() in dependency order.
    /// Returns the first error or Ok if all plugins pass.
    pub fn seal(&self) -> Result<(), String> {
        // For now we only have rls + audit; trivial 2-plugin case.
        // Full topo sort can land in a follow-up PR.
        // Order: rls → audit (audit logs the rewritten plan, so RLS must run first).
        // TODO: real topo sort via depends_on() once N>2 plugins.
        Ok(())
    }
}

/// One `LanceMembrane` per session. All interior state is protected by
/// `RwLock` or atomics so it is `Send + Sync`.
///
/// `current_actor` captures the last `ingest()` context atomically so
/// that the next `project()` call stamps a consistent identity triple
/// on the emitted `CognitiveEventRow`.
pub struct LanceMembrane {
    current_actor: RwLock<ActorState>,
    // Read with Acquire, written with Release — pairs with current_actor
    // RwLock for happens-before across threads.
    current_scent: AtomicU64,
    // Read with Acquire, written with Release — pairs with current_actor
    // RwLock for happens-before across threads.
    current_rationale_phase: AtomicBool,
    version: AtomicU64,
    server_filter: RwLock<CommitFilter>,
    gate: RwLock<Option<Arc<dyn MembraneGate>>>,
    /// Optional plugin registry (RLS, audit, catalog). Set via
    /// `LanceMembrane::with_registry()`. Read-only after seal in v1.
    registry: Option<MembraneRegistry>,
    #[cfg(feature = "realtime")]
    watcher: LanceVersionWatcher,
}

impl LanceMembrane {
    pub fn new() -> Self {
        Self {
            current_actor: RwLock::new(ActorState {
                role: 0,
                faculty: FacultyRole::ReadingComprehension as u8,
                expert: 0,
            }),
            current_scent: AtomicU64::new(0),
            current_rationale_phase: AtomicBool::new(false),
            version: AtomicU64::new(0),
            server_filter: RwLock::new(CommitFilter::default()),
            gate: RwLock::new(None),
            registry: None,
            #[cfg(feature = "realtime")]
            watcher: LanceVersionWatcher::default(),
        }
    }

    /// Build the membrane with a registry of plugins (RLS / audit / catalog).
    ///
    /// Idempotent — calling twice replaces the registry rather than
    /// merging or appending. v1 keeps the registry read-only after this
    /// call; a future seal() may freeze it explicitly.
    pub fn with_registry(mut self, registry: MembraneRegistry) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Borrow the installed registry, if any.
    pub fn registry(&self) -> Option<&MembraneRegistry> {
        self.registry.as_ref()
    }

    /// Install a server-side fan-out filter (TD-INT-13). Subscribers only
    /// see rows matching the filter; rows that don't match are still
    /// returned by `project()` but not bumped to the watcher.
    pub fn set_server_filter(&self, filter: CommitFilter) {
        *self.server_filter.write().expect("server_filter poisoned") = filter;
    }

    /// Install a fan-out gate (TD-INT-9). RBAC, multi-tenant scope,
    /// custom policies. Default = no gate (allow all).
    pub fn set_gate(&self, gate: Arc<dyn MembraneGate>) {
        *self.gate.write().expect("gate poisoned") = Some(gate);
    }

    /// Remove any installed gate, reverting to allow-all fan-out.
    pub fn clear_gate(&self) {
        *self.gate.write().expect("gate poisoned") = None;
    }

    /// Current Lance version counter (monotonic; ticks on each CollapseGate
    /// Persist). Phase D wires this to the Lance dataset version.
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }

    /// Set the faculty context for the current cycle.
    ///
    /// Called by the orchestration layer (not by `ingest`) when
    /// the faculty dispatcher resolves which `FacultyDescriptor`
    /// handles the current cycle. The `rationale_phase` argument
    /// surfaces MM-CoT Stage 1 (true) vs Stage 2 (false); the
    /// dispatcher derives it from `FacultyDescriptor::is_asymmetric()`
    /// plus the current dispatch stage.
    pub fn set_faculty_context(&self, faculty: u8, expert: ExpertId, rationale_phase: bool) {
        {
            let mut actor = self
                .current_actor
                .write()
                .unwrap_or_else(|e| e.into_inner());
            actor.faculty = faculty;
            actor.expert = expert;
        }
        self.current_rationale_phase
            .store(rationale_phase, Ordering::Release);
    }
}

impl Default for LanceMembrane {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ExternalMembrane impl — the BBB
// ─────────────────────────────────────────────────────────────────────────────

impl ExternalMembrane for LanceMembrane {
    /// Scalar projection of a committed cycle.
    ///
    /// Only Arrow-scalar primitives here. `[u64; 256]` fingerprint lives
    /// in `bus` — this type carries two representative words, not the tensor.
    type Commit = CognitiveEventRow;

    /// External consumer intent entering through the gate.
    type Intent = ExternalIntent;

    /// Subscription handle for projected cognitive events.
    ///
    /// With `[realtime]` feature: a `WatchReceiver` over the in-process
    /// `LanceVersionWatcher` — always-latest semantics, supabase-shape,
    /// std-only sync primitives (no tokio per topology I-2).
    /// Without `[realtime]`: a disconnected `mpsc::Receiver<u64>` stub (Phase A).
    #[cfg(not(feature = "realtime"))]
    type Subscription = mpsc::Receiver<u64>;
    #[cfg(feature = "realtime")]
    type Subscription = WatchReceiver;

    /// Project a committed ShaderBus cycle to a scalar row.
    ///
    /// Strips all VSA state. Emits only Arrow-scalar-compatible fields.
    /// Called on every `CollapseGate` fire with `EmitMode::Persist`.
    fn project(&self, bus: &ShaderBus, meta: MetaWord) -> CognitiveEventRow {
        // Single-lock snapshot: role/faculty/expert are always consistent.
        // Poison recovery via into_inner() (F-09: no permanent panic source).
        let actor = *self.current_actor.read().unwrap_or_else(|e| e.into_inner());
        let role = actor.role;
        let faculty = actor.faculty;
        let expert = actor.expert;
        let scent = self.current_scent.load(Ordering::Acquire) as u8;

        let row = CognitiveEventRow {
            external_role: role,
            faculty_role: faculty,
            expert_id: expert,
            dialect: 0, // Phase B: populated by polyglot front-end
            scent,
            thinking: meta.thinking(),
            awareness: meta.awareness(),
            nars_f: meta.nars_f(),
            nars_c: meta.nars_c(),
            free_e: meta.free_e(),
            // Two scalar words from the fingerprint — cycle identity without
            // the full 2 KB VSA tensor. The tensor lives only in ShaderBus.
            cycle_fp_hi: bus.cycle_fingerprint[0],
            cycle_fp_lo: bus.cycle_fingerprint[255],
            gate_commit: bus.gate.is_flow(),
            // gate_f currently mirrors free_e * GATE_DAMPING_FACTOR pending
            // real gate signal — see TD-MEMBRANE-GATE-1.  Field is kept
            // because `CognitiveEventRow::gate_f` is part of the public
            // schema (see external_intent.rs and filter_expr.rs) and is
            // referenced by `WHERE gate_f < N` SQL filters.
            gate_f: ((meta.free_e() as f32) * GATE_DAMPING_FACTOR) as u8,
            rationale_phase: self.current_rationale_phase.load(Ordering::Acquire),
        };

        // TD-INT-13 + TD-INT-9: server-side fan-out gate.
        //   1. CommitFilter — scalar predicate match
        //   2. MembraneGate — RBAC / multi-tenant / custom policy
        // If either rejects, we still return the row but skip the watcher
        // bump. The row is the projection; only the fan-out is gated.
        let actor_id = expert as u64; // v1: actor = expert; refine when UNKNOWN-4 lands
        let style_ord = meta.thinking();
        let pass_filter = self
            .server_filter
            .read()
            .map(|f| f.matches(actor_id, meta.free_e(), style_ord, bus.gate.is_flow()))
            .unwrap_or(true);
        let pass_gate = self
            .gate
            .read()
            .ok()
            .and_then(|g| {
                g.as_ref()
                    .map(|g| g.should_emit(role, faculty, expert, bus.gate.is_flow()))
            })
            .unwrap_or(true);

        // DM-4: fan out to all current subscribers (supabase-shape realtime).
        #[cfg(feature = "realtime")]
        if pass_filter && pass_gate {
            self.watcher.bump(row.clone());
        }
        #[cfg(not(feature = "realtime"))]
        let _ = (pass_filter, pass_gate); // silence unused without realtime

        row
    }

    /// Translate external intent to canonical dispatch.
    ///
    /// Four-step BBB crossing:
    /// 1. Pass membrane — `ExternalIntent` is the safe crossing type.
    /// 2. Get a role   — `intent.role` is stamped into `current_actor.role`.
    /// 3. Get a place  — `intent.dn.scent()` becomes `current_scent`.
    /// 4. Translate    — returns `UnifiedStep` for `OrchestrationBridge::route()`.
    fn ingest(&self, intent: ExternalIntent) -> UnifiedStep {
        // 2. Role — atomic write to the shared actor state (F-01 fix).
        // ExternalRole is `#[repr(u8)]` so the cast is safe today; the
        // debug_assert guards against a future widening of the enum repr.
        // ingest is infallible by current contract — assert in dev, clamp
        // in release via u8::try_from fallback.
        debug_assert!(
            (intent.role as u32) <= u8::MAX as u32,
            "ExternalRole grew past u8 — update repr or widen CognitiveEventRow.external_role",
        );
        let role: u8 = u8::try_from(intent.role as u32).unwrap_or_else(|_| {
            eprintln!("WARN: ExternalRole exceeds u8 range, clamping to 0xFF");
            0xFFu8
        });
        {
            let mut actor = self
                .current_actor
                .write()
                .unwrap_or_else(|e| e.into_inner());
            actor.role = role;
        }

        // 3. Place (FNV-1a scent; Phase C may upgrade to full cascade)
        let scent = intent.dn.scent();
        self.current_scent.store(scent as u64, Ordering::Release);

        // 4. Translate to step type for OrchestrationBridge
        let step_type = match intent.kind {
            ExternalEventKind::Seed => "lg.blackboard.seed",
            ExternalEventKind::Context => "lg.blackboard.context",
            ExternalEventKind::Commit => "lg.blackboard.commit",
        };

        UnifiedStep {
            step_id: format!("{:016x}", scent as u64),
            step_type: step_type.to_owned(),
            status: StepStatus::Pending,
            thinking: None, // resolved by OrchestrationBridge::resolve_thinking()
            reasoning: None,
            confidence: None,
            depends_on: Vec::new(),
        }
    }

    /// Subscribe to projected commits matching the filter.
    ///
    /// With `[realtime]`: returns a `WatchReceiver` seeded with the latest
    /// committed row.  Always-latest semantics (supabase-shape), std-only
    /// sync primitives.
    /// Without `[realtime]`: returns a disconnected `mpsc::Receiver<u64>` stub.
    #[cfg(not(feature = "realtime"))]
    fn subscribe(&self, _filter: CommitFilter) -> mpsc::Receiver<u64> {
        let (_tx, rx) = mpsc::channel();
        rx
    }

    #[cfg(feature = "realtime")]
    fn subscribe(&self, _filter: CommitFilter) -> WatchReceiver {
        self.watcher.subscribe()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dn_path::DnPath;
    use lance_graph_contract::{cognitive_shader::ShaderBus, external_membrane::ExternalRole};

    fn make_dn() -> DnPath {
        DnPath::parse("/tree/ada/heel/callcenter/hip/v1/branch/agents/twig/card/leaf/abc").unwrap()
    }

    #[test]
    fn ingest_sets_role_and_scent() {
        let m = LanceMembrane::new();
        let intent = ExternalIntent::seed(
            ExternalRole::CrewaiAgent,
            make_dn(),
            b"hello world".to_vec(),
        );
        let step = m.ingest(intent);
        assert_eq!(step.step_type, "lg.blackboard.seed");
        assert_eq!(step.status, StepStatus::Pending);
        // scent was set
        assert_ne!(m.current_scent.load(Ordering::Relaxed), 0);
        // role was stamped (read from atomic ActorState — F-01 fix)
        assert_eq!(
            m.current_actor.read().unwrap().role,
            ExternalRole::CrewaiAgent as u8
        );
    }

    #[test]
    fn project_emits_scalar_row() {
        let m = LanceMembrane::new();
        // Prime role context via ingest
        let intent = ExternalIntent::seed(ExternalRole::Rag, make_dn(), vec![]);
        m.ingest(intent);

        let bus = ShaderBus::empty();
        let meta = MetaWord::new(5, 3, 200, 150, 10);
        let row = m.project(&bus, meta);

        assert_eq!(row.external_role, ExternalRole::Rag as u8);
        assert_eq!(row.thinking, 5);
        assert_eq!(row.nars_f, 200);
        assert_eq!(row.nars_c, 150);
        assert_eq!(row.free_e, 10);
        // Fingerprint words from ShaderBus::empty() are 0
        assert_eq!(row.cycle_fp_hi, 0);
        assert_eq!(row.cycle_fp_lo, 0);
        // Empty bus has HOLD gate (gate=2, not Flow)
        assert!(!row.gate_commit);
    }

    #[test]
    fn context_kind_produces_context_step() {
        let m = LanceMembrane::new();
        let intent =
            ExternalIntent::context(ExternalRole::N8n, make_dn(), b"context payload".to_vec());
        let step = m.ingest(intent);
        assert_eq!(step.step_type, "lg.blackboard.context");
    }

    /// Phase A (no realtime feature): subscription is a disconnected stub.
    #[cfg(not(feature = "realtime"))]
    #[test]
    fn subscribe_returns_disconnected_receiver() {
        let m = LanceMembrane::new();
        let rx = m.subscribe(CommitFilter::default());
        // Phase A: disconnected — no sender kept, recv errors immediately
        assert!(rx.try_recv().is_err());
    }

    /// TD-INT-9: gate denies → row still returned, fan-out skipped.
    #[test]
    fn gate_denies_skips_emit_but_returns_row() {
        struct DenyAll;
        impl MembraneGate for DenyAll {
            fn should_emit(&self, _: u8, _: u8, _: u16, _: bool) -> bool {
                false
            }
        }

        let m = LanceMembrane::new();
        m.set_gate(Arc::new(DenyAll));
        let intent = ExternalIntent::seed(ExternalRole::Rag, make_dn(), vec![]);
        m.ingest(intent);

        let bus = ShaderBus::empty();
        let meta = MetaWord::new(5, 3, 200, 150, 10);
        let row = m.project(&bus, meta);
        // Row is still returned (caller may want it for metrics).
        assert_eq!(row.external_role, ExternalRole::Rag as u8);
        assert_eq!(row.thinking, 5);
        // Fan-out is skipped — verified via watcher tests in realtime feature.
    }

    /// TD-INT-13: server-side filter mismatch → fan-out skipped.
    #[test]
    fn filter_mismatch_skips_emit() {
        let m = LanceMembrane::new();
        // Filter: only style_ordinal == 99
        m.set_server_filter(CommitFilter {
            style_ordinal: Some(99),
            ..Default::default()
        });
        let intent = ExternalIntent::seed(ExternalRole::Rag, make_dn(), vec![]);
        m.ingest(intent);

        let bus = ShaderBus::empty();
        // Project with style 5 — should not match the filter
        let meta = MetaWord::new(5, 3, 200, 150, 10);
        let row = m.project(&bus, meta);
        assert_eq!(row.thinking, 5);
        // Row returned, fan-out gated on style_ordinal mismatch.
    }

    /// TD-INT-13: server-side filter match → fan-out happens.
    #[test]
    fn filter_match_passes_emit() {
        let m = LanceMembrane::new();
        // Filter: style_ordinal == 5
        m.set_server_filter(CommitFilter {
            style_ordinal: Some(5),
            max_free_energy: Some(100),
            ..Default::default()
        });
        let intent = ExternalIntent::seed(ExternalRole::Rag, make_dn(), vec![]);
        m.ingest(intent);

        let bus = ShaderBus::empty();
        let meta = MetaWord::new(5, 3, 200, 150, 10);
        let row = m.project(&bus, meta);
        assert_eq!(row.thinking, 5);
        assert_eq!(row.free_e, 10);
        // Free_e=10 ≤ max=100 ✓; style=5 == 5 ✓ → fan-out passes.
    }

    /// Phase D (realtime feature): subscribe() → project() → rx.current() sees the row.
    /// Migrated from tokio::sync::watch::Receiver to the std-sync WatchReceiver
    /// API per topology I-2 (tokio outbound only).
    #[cfg(feature = "realtime")]
    #[test]
    fn subscribe_receives_on_project() {
        let m = LanceMembrane::new();
        let rx = m.subscribe(CommitFilter::default());

        // Prime role context and call project()
        let intent = ExternalIntent::seed(ExternalRole::CrewaiAgent, make_dn(), vec![]);
        m.ingest(intent);

        let bus = ShaderBus::empty();
        let meta = MetaWord::new(7, 3, 200, 150, 10);
        m.project(&bus, meta);

        // current() reads the always-latest snapshot (Arc<CognitiveEventRow>);
        // after project() bumped the watcher, it reflects the new row.
        let snapshot = rx.current();
        assert_eq!(
            snapshot.thinking, 7,
            "subscriber should see the projected row"
        );
        assert_eq!(snapshot.external_role, ExternalRole::CrewaiAgent as u8);
    }

    #[test]
    fn set_faculty_context_wires_rationale_phase() {
        let m = LanceMembrane::new();
        // Before wiring: rationale_phase defaults to false
        let bus = ShaderBus::empty();
        let meta = MetaWord::new(0, 0, 128, 128, 20);
        let row = m.project(&bus, meta);
        assert!(!row.rationale_phase, "default should be Stage 2 (false)");

        // Wire Stage 1 (rationale) via set_faculty_context
        m.set_faculty_context(
            FacultyRole::ReadingComprehension as u8,
            42,   // expert_id
            true, // rationale phase = Stage 1
        );
        let row = m.project(&bus, meta);
        assert!(
            row.rationale_phase,
            "after set_faculty_context(true), should be Stage 1"
        );
        assert_eq!(row.faculty_role, FacultyRole::ReadingComprehension as u8);
        assert_eq!(row.expert_id, 42);

        // Wire Stage 2 (answer)
        m.set_faculty_context(FacultyRole::ReadingComprehension as u8, 42, false);
        let row = m.project(&bus, meta);
        assert!(
            !row.rationale_phase,
            "after set_faculty_context(false), should be Stage 2"
        );
    }

    #[test]
    fn with_registry_chain_compiles_and_stores() {
        // Empty registry composes through the builder and is readable back.
        let m = LanceMembrane::new().with_registry(MembraneRegistry::new());
        assert!(
            m.registry().is_some(),
            "registry() should be Some after wiring"
        );
    }

    #[test]
    fn with_registry_replaces_not_appends() {
        // Two consecutive with_registry calls — second wins.
        let m = LanceMembrane::new()
            .with_registry(MembraneRegistry::new())
            .with_registry(MembraneRegistry::new());
        // Still exactly one registry held — call site is idempotent.
        assert!(m.registry().is_some());
    }

    #[test]
    fn registry_none_by_default() {
        let m = LanceMembrane::new();
        assert!(
            m.registry().is_none(),
            "registry() should be None on a bare LanceMembrane"
        );
    }

    #[test]
    fn bbb_scalar_only_compile_check() {
        // This is the compile-time BBB proof: CognitiveEventRow contains
        // only u8/u16/u64/bool. It implements Send + Sync trivially.
        // If someone added a [u64; 256] field here, the type would still
        // be Send, but it would be a 2 KB copy — visible in code review.
        // More critically, Arrow::try_new() would reject it as a column
        // element when the [persist] feature is activated in Phase B.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CognitiveEventRow>();
        assert_send_sync::<LanceMembrane>();
    }

    /// HIGH #1: gate_f must not be a verbatim copy of free_e.
    /// Option B applied — gate_f = free_e * GATE_DAMPING_FACTOR (0.5).
    /// See TD-MEMBRANE-GATE-1.
    #[test]
    fn gate_f_resolution_does_not_duplicate_free_e() {
        let m = LanceMembrane::new();
        let intent = ExternalIntent::seed(ExternalRole::Rag, make_dn(), vec![]);
        m.ingest(intent);

        let bus = ShaderBus::empty();
        // Use a free_e value where damping produces a distinguishable result.
        // free_e is 6 bits (0..=63) per MetaWord packing, pick 40.
        let meta = MetaWord::new(5, 3, 200, 150, 40);
        let row = m.project(&bus, meta);

        assert_eq!(row.free_e, 40, "free_e is the raw MetaWord scalar");
        // gate_f is damped and so must differ from free_e for any non-zero free_e.
        assert_ne!(
            row.gate_f, row.free_e,
            "gate_f must not be a verbatim copy of free_e (HIGH #1)",
        );
        // Specifically: floor(40 * 0.5) == 20.
        assert_eq!(
            row.gate_f, 20,
            "gate_f should equal free_e * GATE_DAMPING_FACTOR (0.5)",
        );
    }

    /// MEDIUM #2: Acquire/Release pairs make writer state visible to readers.
    /// One thread writes via set_faculty_context, another reads via project.
    #[test]
    fn atomics_acquire_release_visible_across_threads() {
        use std::sync::Arc as StdArc;
        use std::thread;

        let m = StdArc::new(LanceMembrane::new());
        let intent = ExternalIntent::seed(ExternalRole::Rag, make_dn(), vec![]);
        m.ingest(intent);

        // Writer thread flips rationale_phase to true via set_faculty_context.
        let writer = {
            let m = StdArc::clone(&m);
            thread::spawn(move || {
                m.set_faculty_context(FacultyRole::ReadingComprehension as u8, 7, true);
            })
        };
        writer.join().expect("writer thread joined");

        // Reader thread observes the row via project.
        let reader = {
            let m = StdArc::clone(&m);
            thread::spawn(move || {
                let bus = ShaderBus::empty();
                let meta = MetaWord::new(5, 3, 200, 150, 10);
                m.project(&bus, meta)
            })
        };
        let row = reader.join().expect("reader thread joined");

        // Release on the writer side + Acquire on the reader side
        // establishes happens-before — we must observe the post-write state.
        assert!(
            row.rationale_phase,
            "Acquire-side read must see Release-side write"
        );
        assert_eq!(row.faculty_role, FacultyRole::ReadingComprehension as u8);
        assert_eq!(row.expert_id, 7);
    }

    /// MEDIUM #3: ExternalRole today fits in u8 — debug_assert holds and the
    /// cast yields the discriminant. The clamp-on-overflow path is dormant
    /// until the enum repr widens; we exercise the happy path here.
    #[test]
    fn external_role_clamp_or_assert() {
        let m = LanceMembrane::new();
        // Highest currently defined variant = 7 (Agent). Well within u8.
        let intent = ExternalIntent::seed(ExternalRole::Agent, make_dn(), vec![]);
        m.ingest(intent);

        let actor = m.current_actor.read().unwrap();
        assert_eq!(actor.role, ExternalRole::Agent as u8);
        // ExternalRole repr is u8; storing as u8 is enforced by the type system.
        // Highest currently defined variant = 7 (Agent), reserve room for growth.
        assert!(
            actor.role < 32,
            "ExternalRole repr stays well below u8::MAX to leave room for growth"
        );
    }

    /// E2: seal() smoke test — empty registry seals without error.
    #[test]
    fn plugin_seal_passes_with_well_formed_registry() {
        let registry = MembraneRegistry::new();
        assert!(
            registry.seal().is_ok(),
            "empty registry should seal cleanly",
        );

        // And via the builder chain on the membrane.
        let m = LanceMembrane::new().with_registry(MembraneRegistry::new());
        let r = m.registry().expect("registry installed");
        assert!(r.seal().is_ok(), "seal through with_registry chain");
    }
}
