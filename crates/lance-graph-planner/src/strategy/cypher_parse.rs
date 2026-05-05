//! Strategy #1: CypherParse — Intent parsing via lance-graph's nom parser.

use crate::ir::{Arena, LogicalOp};
use crate::traits::*;
use crate::PlanError;

#[derive(Debug)]
pub struct CypherParse;

impl PlanStrategy for CypherParse {
    fn name(&self) -> &str {
        "cypher_parse"
    }
    fn capability(&self) -> PlanCapability {
        PlanCapability::Parse
    }

    fn affinity(&self, context: &PlanContext) -> f32 {
        // Always high affinity — every query needs parsing
        if context.query.to_uppercase().contains("MATCH")
            || context.query.to_uppercase().contains("CREATE")
            || context.query.to_uppercase().contains("RETURN")
        {
            0.95
        } else {
            0.5
        }
    }

    fn plan(
        &self,
        mut input: PlanInput,
        _arena: &mut Arena<LogicalOp>,
    ) -> Result<PlanInput, PlanError> {
        let q = input.context.query.to_uppercase();

        // Detect query features from syntax
        input.context.features.has_graph_pattern = q.contains("MATCH");
        input.context.features.has_fingerprint_scan =
            q.contains("HAMMING") || q.contains("FINGERPRINT") || q.contains("RESONATE");
        input.context.features.has_variable_length_path =
            q.contains("*..") || q.contains("*1..") || q.contains("*2..");
        input.context.features.has_aggregation =
            q.contains("COUNT") || q.contains("SUM") || q.contains("AVG") || q.contains("COLLECT");
        input.context.features.has_mutation = q.contains("CREATE")
            || q.contains("SET")
            || q.contains("DELETE")
            || q.contains("MERGE");
        input.context.features.has_resonance = q.contains("RESONATE");
        input.context.features.has_truth_values = q.contains("TRUTH") || q.contains("CONFIDENCE");
        input.context.features.has_workflow = q.contains("WORKFLOW") || q.contains("TASK");
        input.context.features.num_match_clauses = q.matches("MATCH").count();

        // Estimate complexity from detected features
        let mut complexity = input.context.features.num_match_clauses as f64 * 0.2;
        if input.context.features.has_variable_length_path {
            complexity += 0.3;
        }
        if input.context.features.has_fingerprint_scan {
            complexity += 0.2;
        }
        if input.context.features.has_aggregation {
            complexity += 0.1;
        }
        input.context.features.estimated_complexity = complexity.min(1.0);

        // Real implementation: call lance-graph's parser::parse_cypher_query()
        // to produce a full AST. For now, feature detection is the output.

        Ok(input)
    }
}
