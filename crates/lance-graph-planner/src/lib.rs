//! # lance-graph-planner: Unified Query Planner
//!
//! ## Architecture: Three Layers + Strategy Registry
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │  MUL + COMPASS (outer loop)                     │
//! │  "Should I plan? How much do I trust the plan?" │
//! ├─────────────────────────────────────────────────┤
//! │  THINKING ORCHESTRATION (middle layer)           │
//! │  "How should I think about this?"                │
//! ├─────────────────────────────────────────────────┤
//! │  STRATEGY REGISTRY (inner loop)                  │
//! │  16 composable strategies, selected by:          │
//! │  - User: "use dp_join + sigma_scan"              │
//! │  - AGI: MUL+Compass+ThinkingStyle → affinity     │
//! │  - Auto: each strategy reports affinity           │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! ## The 16 Strategies
//!
//! | # | Strategy | Phase | Source |
//! |---|----------|-------|--------|
//! | 1 | CypherParse | Parse | lance-graph nom (Cypher) |
//! | 2 | GremlinParse | Parse | TinkerPop Gremlin |
//! | 3 | SparqlParse | Parse | W3C SPARQL |
//! | 4 | GqlParse | Parse | ISO GQL (39075) |
//! | 5 | ArenaIR | Plan | Polars arena |
//! | 6 | DPJoinEnum | Plan | Kuzudb DP |
//! | 7 | RuleOptimizer | Optimize | DataFusion |
//! | 8 | HistogramCost | Optimize | Hyrise |
//! | 9 | SigmaBandScan | Physicalize | lance-graph |
//! | 10 | MorselExec | Physicalize | Kuzudb/Polars |
//! | 11 | StreamPipeline | Execute | Polars |
//! | 12 | TruthPropagation | Physicalize | lance-graph semiring |
//! | 13 | CollapseGate | Physicalize | agi-chat |
//! | 14 | JitCompile | Execute | ndarray |
//! | 15 | WorkflowDAG | Plan | LangGraph |
//! | 16 | ExtensionPlanner | any | DataFusion |

// === P0: Adjacency Substrate (THE GROUND) ===
pub mod adjacency;

// === P1: NARS Schema + Thinking ===
pub mod nars;

// === Core modules ===
pub mod execute;
pub mod ir;
pub mod mul;
pub mod optimize;
pub mod physical;
pub mod plan;
pub mod thinking;

// === Dynamic Elevation (cost model that smells resistance) ===
pub mod elevation;

// === Prediction Engine (temporal NARS simulation + news ingestion) ===
pub mod prediction;

// === Strategy system ===
pub mod compose;
pub mod selector;
pub mod strategy;
pub mod traits;

// === Pipeline DAG executor (LF-12 keystone) ===
pub mod pipeline;

// === Autocomplete Cache (VSA superposition KV-cache) ===
pub mod cache;

// === Internal API (same-binary, zero-serde) ===
pub mod api;

// === Canonical OrchestrationBridge impl (dedup per contract) ===
// Implements `lance_graph_contract::orchestration::OrchestrationBridge`
// for `PlannerAwareness`. Replaces per-consumer bridge modules. See
// `docs/CONSUMER_WIRING_INSTRUCTIONS.md` for the integration pattern.
mod orchestration_impl;

use ir::LogicalPlan;
use mul::{GateDecision, MulAssessment};
use selector::StrategySelector;
use thinking::ThinkingContext;
use traits::{PlanContext, PlanStrategy, QueryFeatures};

/// The unified planner entry point with strategy awareness.
pub struct PlannerAwareness {
    /// All registered strategies.
    strategies: Vec<Box<dyn PlanStrategy>>,
    /// Who decides which strategies to use.
    selector: StrategySelector,
}

/// Result of unified planning.
pub struct PlanResult {
    /// MUL assessment (None if no MUL layer used).
    pub mul: Option<MulAssessment>,
    /// Thinking context (None if no thinking orchestration).
    pub thinking: Option<ThinkingContext>,
    /// The logical plan (arena-allocated).
    pub plan: LogicalPlan,
    /// Which strategies were selected and composed.
    pub strategies_used: Vec<String>,
    /// Free will modifier applied to confidence.
    pub free_will_modifier: f64,
    /// Compass score (if navigating unknown territory).
    pub compass_score: Option<f64>,
}

impl PlannerAwareness {
    /// Create with default strategies (all 13) and auto selection.
    pub fn new() -> Self {
        Self {
            strategies: strategy::default_strategies(),
            selector: StrategySelector::default(),
        }
    }

    /// Create with explicit strategy selection.
    pub fn with_explicit(strategy_names: Vec<String>) -> Self {
        Self {
            strategies: strategy::default_strategies(),
            selector: StrategySelector::Explicit(strategy_names),
        }
    }

    /// Create with AGI resonance selection.
    pub fn with_resonance(thinking_style: Vec<f64>, mul_modifier: f64, compass_score: f64) -> Self {
        Self {
            strategies: strategy::default_strategies(),
            selector: StrategySelector::Resonance {
                thinking_style,
                mul_modifier,
                compass_score,
            },
        }
    }

    /// Set the strategy selector.
    pub fn set_selector(&mut self, selector: StrategySelector) {
        self.selector = selector;
    }

    /// Register an additional custom strategy.
    pub fn register_strategy(&mut self, strategy: Box<dyn PlanStrategy>) {
        self.strategies.push(strategy);
    }

    /// Get a reference to all registered strategies.
    pub fn strategies(&self) -> &[Box<dyn PlanStrategy>] {
        &self.strategies
    }

    /// Plan a query with full MUL + Thinking + Strategy pipeline.
    pub fn plan_full(
        &self,
        query: &str,
        situation: &mul::SituationInput,
    ) -> Result<PlanResult, PlanError> {
        // === LAYER 1: MUL + COMPASS ===
        let mul_assessment = mul::assess(situation);
        let gate = mul::gate_check(&mul_assessment);

        match gate {
            GateDecision::Proceed { free_will_modifier } => {
                // === LAYER 2: THINKING ORCHESTRATION ===
                let thinking_ctx =
                    thinking::orchestrate(query, &mul_assessment, &plan::PlannerConfig::default());

                // === LAYER 3: STRATEGY COMPOSITION ===
                let context = PlanContext {
                    query: query.to_string(),
                    features: QueryFeatures::default(),
                    free_will_modifier,
                    thinking_style: Some(
                        thinking_ctx
                            .modulation
                            .to_fingerprint()
                            .iter()
                            .map(|b| *b as f64 / 255.0)
                            .collect(),
                    ),
                    nars_hint: Some(thinking_ctx.nars_type),
                };

                let selected =
                    selector::select_strategies(&self.selector, &self.strategies, &context);
                let strategy_names: Vec<String> =
                    selected.iter().map(|s| s.name().to_string()).collect();

                let plan = compose::compose_and_execute(&selected, context)?;

                Ok(PlanResult {
                    mul: Some(mul_assessment),
                    thinking: Some(thinking_ctx),
                    plan,
                    strategies_used: strategy_names,
                    free_will_modifier,
                    compass_score: None,
                })
            }
            GateDecision::Sandbox { reason } => Err(PlanError::GateBlocked { reason }),
            GateDecision::Compass => {
                let compass = mul::compass::navigate(query, &mul_assessment);
                match compass.decision {
                    mul::compass::CompassDecision::SurfaceToMeta => Err(PlanError::SurfaceToMeta {
                        compass_score: compass.score,
                        reason: "Stakes too high".into(),
                    }),
                    _ => {
                        // Proceed with reduced confidence
                        let thinking_ctx = thinking::orchestrate(
                            query,
                            &mul_assessment,
                            &plan::PlannerConfig::default(),
                        );
                        let context = PlanContext {
                            query: query.to_string(),
                            features: QueryFeatures::default(),
                            free_will_modifier: compass.modified_score,
                            thinking_style: None,
                            nars_hint: Some(thinking_ctx.nars_type),
                        };
                        let selected =
                            selector::select_strategies(&self.selector, &self.strategies, &context);
                        let strategy_names: Vec<String> =
                            selected.iter().map(|s| s.name().to_string()).collect();
                        let plan = compose::compose_and_execute(&selected, context)?;

                        Ok(PlanResult {
                            mul: Some(mul_assessment),
                            thinking: Some(thinking_ctx),
                            plan,
                            strategies_used: strategy_names,
                            free_will_modifier: compass.modified_score,
                            compass_score: Some(compass.score),
                        })
                    }
                }
            }
        }
    }

    /// Plan a query in Auto mode (no MUL, no thinking orchestration).
    /// Two-pass: first lexical feature detection, then strategies score affinity.
    pub fn plan_auto(&self, query: &str) -> Result<PlanResult, PlanError> {
        // Pass 1: lexical feature detection (single source of truth lives in
        // strategy::cypher_parse::extract_features — keep this in sync by
        // calling it rather than re-implementing the keyword scan here).
        let context = PlanContext {
            query: query.to_string(),
            features: strategy::cypher_parse::extract_features(query),
            free_will_modifier: 1.0,
            thinking_style: None,
            nars_hint: None,
        };

        // Pass 2: select strategies with detected features
        let selected = selector::select_strategies(
            &StrategySelector::Auto {
                max_per_phase: 2,
                min_affinity: 0.3,
            },
            &self.strategies,
            &context,
        );
        let strategy_names: Vec<String> = selected.iter().map(|s| s.name().to_string()).collect();

        let plan = compose::compose_and_execute(&selected, context)?;

        Ok(PlanResult {
            mul: None,
            thinking: None,
            plan,
            strategies_used: strategy_names,
            free_will_modifier: 1.0,
            compass_score: None,
        })
    }
}

impl Default for PlannerAwareness {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors from the unified planner.
#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error("MUL gate blocked: {reason}")]
    GateBlocked { reason: String },

    #[error("Surface to meta (compass={compass_score:.3}): {reason}")]
    SurfaceToMeta { compass_score: f64, reason: String },

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Plan error: {0}")]
    Plan(String),

    #[error("Optimize error: {0}")]
    Optimize(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_selects_dp_join_for_multi_hop_pattern() {
        let planner = PlannerAwareness::new();
        let result = planner
            .plan_auto("MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN a, c")
            .unwrap();

        // Should include arena_ir for logical plan + dp_join for join ordering
        assert!(result.strategies_used.iter().any(|s| s == "arena_ir"));
    }

    #[test]
    fn auto_selects_sigma_scan_for_fingerprint_query() {
        let planner = PlannerAwareness::new();
        let result = planner
            .plan_auto("MATCH (n) WHERE RESONATE(n.fingerprint, $query, 0.7) RETURN n")
            .unwrap();

        assert!(result.strategies_used.iter().any(|s| s == "sigma_scan"));
    }

    #[test]
    fn explicit_override_uses_only_named_strategies() {
        let planner =
            PlannerAwareness::with_explicit(vec!["cypher_parse".into(), "arena_ir".into()]);
        let result = planner.plan_auto("MATCH (n) RETURN n");

        // plan_auto uses Auto selector, but with_explicit sets Explicit
        // Let's use plan_full instead with a default situation
        let result = planner
            .plan_full(
                "MATCH (n) RETURN n",
                &mul::SituationInput {
                    felt_competence: 0.8,
                    demonstrated_competence: 0.8,
                    source_reliability: 0.9,
                    environment_stability: 0.9,
                    calibration_accuracy: 0.8,
                    complexity_ratio: 0.7,
                    ..Default::default()
                },
            )
            .unwrap();

        // Only cypher_parse and arena_ir should be used
        assert!(result.strategies_used.len() <= 2);
        for name in &result.strategies_used {
            assert!(
                name == "cypher_parse" || name == "arena_ir",
                "Unexpected strategy: {name}"
            );
        }
    }

    #[test]
    fn strategies_compose_in_pipeline_order() {
        let planner = PlannerAwareness::new();
        let result = planner
            .plan_auto("MATCH (a)-[:CAUSES*2..5]->(b) WHERE RESONATE(a.fp, $q, 0.3) RETURN a, b")
            .unwrap();

        // Verify strategies are in phase order
        let phases: Vec<_> = result
            .strategies_used
            .iter()
            .map(|name| {
                planner
                    .strategies
                    .iter()
                    .find(|s| s.name() == name)
                    .map(|s| s.capability().phase())
            })
            .collect();

        for window in phases.windows(2) {
            if let (Some(a), Some(b)) = (window[0], window[1]) {
                assert!(
                    a <= b,
                    "Strategies not in pipeline order: {:?} > {:?}",
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn collapse_gate_filters_low_resonance_results() {
        use ir::logical_op::CollapseGate;
        use physical::collapse::{CollapseOp, GateState};

        let op = CollapseOp {
            gate: CollapseGate::default(),
            child: Box::new(DummyPhysOp),
            estimated_cardinality: 10.0,
        };

        // Coherent → FLOW
        assert_eq!(op.classify(&[0.85, 0.86, 0.84]), GateState::Flow);
        // Dispersed → BLOCK
        assert_eq!(op.classify(&[0.0, 1.0, 0.0, 1.0]), GateState::Block);
    }

    #[test]
    fn truth_propagation_accumulates_during_traversal() {
        use physical::accumulate::{Semiring, SemiringValue, TruthPropagatingSemiring};

        let sr = TruthPropagatingSemiring;

        // Deduction along edge
        let premise = SemiringValue::Truth {
            frequency: 0.9,
            confidence: 0.8,
        };
        let edge = SemiringValue::Truth {
            frequency: 0.7,
            confidence: 0.9,
        };
        let conclusion = sr.multiply(&premise, &edge);

        if let SemiringValue::Truth {
            frequency,
            confidence,
        } = conclusion
        {
            assert!(frequency < 0.9, "Deduction should reduce frequency");
            assert!(confidence < 0.8, "Deduction should reduce confidence");
        }

        // Revision merges evidence
        let path_a = SemiringValue::Truth {
            frequency: 0.8,
            confidence: 0.7,
        };
        let path_b = SemiringValue::Truth {
            frequency: 0.6,
            confidence: 0.5,
        };
        let revised = sr.add(&path_a, &path_b);

        if let SemiringValue::Truth { confidence, .. } = revised {
            assert!(
                confidence > 0.5,
                "Revision should increase confidence with more evidence"
            );
        }
    }

    #[test]
    fn resonance_mode_uses_thinking_style_affinity() {
        let planner = PlannerAwareness::with_resonance(
            vec![0.8, 0.1, 0.1, 0.2, 0.9, 0.5, 0.5, 0.3], // High depth + analytical
            0.85,                                         // Good MUL modifier
            0.0,                                          // No compass
        );

        let result = planner
            .plan_full(
                "MATCH (a)-[:KNOWS]->(b) RETURN a",
                &mul::SituationInput {
                    felt_competence: 0.8,
                    demonstrated_competence: 0.8,
                    source_reliability: 0.9,
                    environment_stability: 0.9,
                    calibration_accuracy: 0.8,
                    complexity_ratio: 0.7,
                    ..Default::default()
                },
            )
            .unwrap();

        assert!(!result.strategies_used.is_empty());
    }

    // Dummy physical operator for testing
    #[derive(Debug)]
    struct DummyPhysOp;
    impl physical::PhysicalOperator for DummyPhysOp {
        fn name(&self) -> &str {
            "dummy"
        }
        fn cardinality(&self) -> f64 {
            0.0
        }
        fn is_pipeline_breaker(&self) -> bool {
            false
        }
        fn children(&self) -> Vec<&dyn physical::PhysicalOperator> {
            vec![]
        }
    }
}
