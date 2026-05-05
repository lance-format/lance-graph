//! # cognitive-shader-driver — the shader IS the driver
//!
//! End-to-end wiring of the four components that were previously isolated:
//!
//! ```text
//! lance-graph-contract::cognitive_shader  ←  zero-dep DTOs
//!              │
//!              ▼
//! cognitive-shader-driver (THIS CRATE)
//!    ├── BindSpace (struct-of-arrays, genius packed layout)
//!    ├── p64_bridge::CognitiveShader (8 planes × bgz17 O(1) distance)
//!    └── thinking-engine hook (optional, behind `with-engine` feature)
//!                                            │
//!                                            ▼
//!              cycle_fingerprint emitted through ShaderSink
//! ```
//!
//! ## Role Reversal (the genius DTO API)
//!
//! The shader no longer sits below the engine. It holds the BindSpace columns,
//! reads the packed u32 MetaColumn FIRST (cheapest prefilter), then loads only
//! the Fingerprint rows that passed. Each cycle emits exactly one
//! `Fingerprint<256>` (the cycle_fingerprint), which is simultaneously:
//!
//! - a cache key (A2A blackboard lookup),
//! - a retrieval key (BindSpace similarity sweep),
//! - a replay key (deterministic given the same dispatch),
//! - a cursor (next-cycle seed).
//!
//! ## EmbedAnything Patterns
//!
//! - **Builder** (`CognitiveShaderBuilder::new().rows(..).style(..).sink(..).build()`)
//! - **Auto-detect** (`StyleSelector::Auto` routes from qualia column)
//! - **Commit sinks** (`ShaderSink` trait; `NullSink` as zero-cost default)
//! - **Feature gates** (`with-engine` pulls thinking-engine; core stays lean)
//! - **No forward pass at runtime** — bgz17 distance is precomputed
//!
//! ---
//!
//! ## STOP — Canonical vs LAB-ONLY (read before touching this crate)
//!
//! **Do NOT add new REST endpoints, new per-op Wire DTOs, or new
//! HTTP/gRPC handlers without reading
//! `.claude/knowledge/lab-vs-canonical-surface.md` first.**
//!
//! **`cognitive-shader-driver` IS the unified API.** Its public
//! re-exports form the super DTO family that every consumer speaks:
//!
//! - **Per-cycle hot path:** `ShaderDispatch` → `ShaderDriver` →
//!   `ShaderHit` + `MetaWord` via `ShaderSink`. `BindSpace*` columns
//!   are the substrate.
//! - **Cross-domain composition:** `UnifiedStep` routed via
//!   `OrchestrationBridge` by `StepDomain`. Every consumer
//!   (thinking-engine, planner, codec research) plugs in through this
//!   trait.
//!
//! Both layers are zero-dep, always compiled, and consumer-facing.
//! That is the stable downstream-visible API.
//!
//! Everything else in this crate — `wire.rs`, `serve.rs`, `grpc.rs`,
//! `codec_research.rs`, `codec_bridge.rs`, `planner_bridge.rs`, and the
//! per-op `Wire*` DTOs — is **LAB-ONLY** scaffolding. It exists so the
//! Claude Code backend can test the canonical bridge over HTTP/gRPC.
//! It is **not** part of the consumer API. Future sessions must not
//! "extend the REST API" with new per-op endpoints; they must extend the
//! canonical bridge and let the lab transport follow automatically.
//!
//! Feature layout:
//!
//! ```text
//! default       = canonical library surface only (no HTTP, no gRPC, no wire DTOs)
//! serve         = lab REST transport
//! grpc          = lab gRPC transport
//! with-engine   = thinking-engine consumer wired in
//! with-planner  = planner consumer wired in
//! lab           = umbrella: all of the above → the shader-lab binary
//! ```
//!
//! Research (codec_research, codec_bridge) is gated under serve/grpc
//! because it only exists to be tested through the lab transport. It is
//! one consumer of the unified DTO — not the reason the DTO exists.

#![warn(rust_2018_idioms)]

// ──────────────────────────────────────────────────────────────────────
// Canonical surface — always compiled. Consumers depend on exactly this.
// ──────────────────────────────────────────────────────────────────────
//
// The canonical consumer-facing DTO is `UnifiedStep` + `OrchestrationBridge`
// re-exported from `lance-graph-contract` (see the `pub use` block below).
// Everything below dispatches through that trait; research, planner, and
// engine are just consumers that plug in via the bridge.
pub mod bindspace;
pub mod driver;
pub mod auto_style;
pub mod engine_bridge;
pub mod sigma_rosetta;

// ──────────────────────────────────────────────────────────────────────
// LAB-ONLY modules — compiled only into the shader-lab binary. Never
// shipped to downstream consumers; never part of the canonical API.
// ──────────────────────────────────────────────────────────────────────
//
// Umbrella switch: `--features lab` turns on everything here at once.
// Individual features (`serve`, `grpc`, `with-planner`, `with-engine`)
// remain selectable for narrower lab builds.
//
// Architecture rule: these modules expose per-op Wire DTOs and REST/gRPC
// transports for test convenience in the Claude Code backend. They all
// ultimately route through `OrchestrationBridge` + `UnifiedStep` — the
// canonical surface. No business logic lives here that isn't also
// reachable through the canonical bridge.

// Per-op Wire DTOs (REST + protobuf). LAB-ONLY.
// Gated on `any(serve, grpc)` because both transports share the same
// DTOs; gRPC consumers (grpc.rs) and REST consumers (serve.rs) both
// convert to/from the `Wire*` types in wire.rs.
#[cfg(any(feature = "serve", feature = "grpc"))]
pub mod wire;

// D0.5 — model architecture auto-detection from config.json.
// CODING_PRACTICES.md gap 1 remediation. LAB-ONLY.
#[cfg(any(feature = "serve", feature = "grpc"))]
pub mod auto_detect;

// D1.1 — JIT kernel cache keyed by CodecParams::kernel_signature().
// Structural layer; actual Cranelift IR emission defers to D1.1b. LAB-ONLY.
#[cfg(any(feature = "serve", feature = "grpc"))]
pub mod codec_kernel_cache;

// D1.2 — rotation primitives (Identity / Hadamard / OPQ-stub). LAB-ONLY.
// Hadamard is real (in-place butterfly); OPQ is stub pending D1.1b's
// ndarray::hpc::jitson_cranelift::JitEngine adapter + matrix-blob loader.
#[cfg(any(feature = "serve", feature = "grpc"))]
pub mod rotation_kernel;

// D1.3 — decode-kernel trait + residual composition.
// Hydration/calibration path (NOT cascade inference — that uses
// p64_bridge::CognitiveShader per cognitive-shader-architecture.md
// line 582). LAB-ONLY.
#[cfg(any(feature = "serve", feature = "grpc"))]
pub mod decode_kernel;

// D2.1 — token-agreement harness scaffold (I11 cert gate infra).
// Reference model loader stub + top-k comparator + stub result with
// machine-checkable `stub:true` flag. D2.2 adds real safetensors decode.
// LAB-ONLY.
#[cfg(any(feature = "serve", feature = "grpc"))]
pub mod token_agreement;

// Axum REST server. LAB-ONLY.
#[cfg(feature = "serve")]
pub mod serve;

// tonic gRPC server. LAB-ONLY.
#[cfg(feature = "grpc")]
pub mod grpc;

// Codec research consumer — backing logic for the research Wire DTOs.
// LAB-ONLY. Research is one consumer of the unified DTO, not its reason.
#[cfg(any(feature = "serve", feature = "grpc"))]
pub mod codec_research;

// OrchestrationBridge impl for the codec research consumer — owns
// StepDomain::Ndarray. LAB-ONLY (sits next to its backing logic).
#[cfg(any(feature = "serve", feature = "grpc"))]
pub mod codec_bridge;

// OrchestrationBridge impl for Cypher queries (lg.cypher step_type) —
// routes to AriGraph SPO / BindSpace. Phase 1 stub classifier; Phase 2
// will pull the real `lance_graph::parser::parse_cypher_query` once the
// core crate dep is worth its build-time cost. LAB-ONLY.
#[cfg(any(feature = "serve", feature = "grpc"))]
pub mod cypher_bridge;

// Planner bridge — lab test-shortcut for the per-op WirePlan DTOs.
// PlannerAwareness implements OrchestrationBridge directly in the
// planner crate; that's the canonical path. LAB-ONLY.
#[cfg(feature = "with-planner")]
pub mod planner_bridge;

pub use lance_graph_contract::cognitive_shader::{
    CognitiveShaderDriver, ColumnWindow, EmitMode, MetaFilter, MetaSummary, MetaWord,
    NullSink, RungLevel, ShaderBus, ShaderCrystal, ShaderDispatch, ShaderHit,
    ShaderResonance, ShaderSink, StyleSelector,
};

pub use lance_graph_contract::collapse_gate::{GateDecision, MergeMode};

pub use bindspace::{BindSpace, BindSpaceBuilder, EdgeColumn, FingerprintColumns,
                     MetaColumn, QualiaColumn};
pub use driver::{CognitiveShaderBuilder, ShaderDriver};
pub use engine_bridge::{
    EngineBusBridge, UnifiedStyle, UNIFIED_STYLES, unified_style,
    ingest_codebook_indices, dispatch_from_top_k,
    write_qualia_17d, read_qualia_17d, persist_cycle,
};
