//! # lance-graph-contract — The Single Source of Truth
//!
//! Zero-dependency trait crate that defines the contract between:
//! - **lance-graph-planner** (implements these traits)
//! - **ladybug-rs** (calls Planner + CamPq + OrchestrationBridge)
//! - **crewai-rust** (calls ThinkingStyleContract + MulContract)
//! - **n8n-rs** (calls JitContract + OrchestrationBridge)
//!
//! # Why This Exists
//!
//! Before this crate, each consumer duplicated thinking style enums,
//! field modulation structs, and query plan types. Now:
//!
//! ```text
//! crewai-rust ──┐
//!               │
//! n8n-rs ───────┤── depend on ──► lance-graph-contract (traits only)
//!               │
//! ladybug-rs ───┘
//!
//! lance-graph-planner ──► implements lance-graph-contract traits
//! ```
//!
//! # Module Layout
//!
//! - [`thinking`] — 36 thinking styles, 6 clusters, τ addresses, 23D vectors
//! - [`mul`] — MUL assessment (Dunning-Kruger, trust, flow, compass)
//! - [`plan`] — Query planning traits (PlanStrategy, PlanResult)
//! - [`cam`] — CAM-PQ distance contract (6-byte fingerprint ops)
//! - [`jit`] — JIT compilation contract (jitson template → kernel)
//! - [`orchestration`] — Bridge trait for single-binary routing
//! - [`nars`] — NARS inference types shared across all consumers

pub mod a2a_blackboard;
pub mod auth;
pub mod cam;
pub mod cognitive_shader;
pub mod collapse_gate;
pub mod container;
pub mod crystal;
pub mod distance;
pub mod exploration;
pub mod external_membrane;
pub mod faculty;
pub mod grammar;
pub mod graph_render;
pub mod hash;
pub mod high_heel;
pub mod jit;
pub mod literal_graph;
pub mod mail;
pub mod mul;
pub mod nars;
pub mod ocr;
pub mod ontology;
pub mod orchestration;
pub mod orchestration_mode;
pub mod persona;
pub mod plan;
pub mod property;
pub mod proprioception;
pub mod qualia;
pub mod reasoning;
pub mod repository;
pub mod scenario;
pub mod sensorium;
pub mod sigma_propagation;
pub mod sla;
pub mod splat;
pub mod tax;
pub mod thinking;
pub mod world_map;
pub mod world_model;
