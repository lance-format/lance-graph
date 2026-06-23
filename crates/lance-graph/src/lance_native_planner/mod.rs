// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Native physical planner.
//!
//! Lowers a supported single-hop `Expand` onto CSR-backed extension nodes
//! (`CsrExpandNode` + `LanceTakeNode`). Any plan it cannot serve natively is
//! delegated wholesale to `DataFusionPlanner`, so `LanceNative` execution is
//! always correct on valid Cypher — it simply uses joins when CSR cannot help.

mod csr_expand;
mod direction;
mod extension_planner;
mod take;

pub use direction::NativeDirection;
pub use extension_planner::CsrQueryPlanner;

use std::sync::Arc;

use arrow_schema::{Field, Schema};
use datafusion::common::DFSchema;
use datafusion::logical_expr::{Extension, LogicalPlan, LogicalPlanBuilder};

use crate::ast::RelationshipDirection;
use crate::case_insensitive::qualify_column;
use crate::config::GraphConfig;
use crate::datafusion_planner::expression::{to_df_boolean_expr, to_df_value_expr};
use crate::datafusion_planner::{analysis, DataFusionPlanner, GraphPhysicalPlanner, PlanningContext};
use crate::error::{GraphError, Result};
use crate::logical_plan::{LogicalOperator, ProjectionItem, SortItem};

use csr_expand::CsrExpandNode;
use take::LanceTakeNode;

/// Lance-native planner: CSR single-hop expand with DataFusion fallback.
pub struct LanceNativePlanner {
    config: GraphConfig,
    df: DataFusionPlanner,
}

impl LanceNativePlanner {
    pub fn new(config: GraphConfig) -> Self {
        Self {
            df: DataFusionPlanner::new(config.clone()),
            config,
        }
    }

    pub fn with_catalog(
        config: GraphConfig,
        catalog: Arc<dyn lance_graph_catalog::GraphSourceCatalog>,
    ) -> Self {
        Self {
            df: DataFusionPlanner::with_catalog(config.clone(), catalog),
            config,
        }
    }
}

impl GraphPhysicalPlanner for LanceNativePlanner {
    fn plan(&self, logical_plan: &LogicalOperator) -> Result<LogicalPlan> {
        if !can_plan_natively(logical_plan) {
            return self.df.plan(logical_plan);
        }
        let analysis = analysis::analyze(logical_plan)?;
        let mut ctx = PlanningContext::new(&analysis);
        self.build_native(&mut ctx, logical_plan)
    }
}

impl LanceNativePlanner {
    fn build_native(
        &self,
        ctx: &mut PlanningContext,
        op: &LogicalOperator,
    ) -> Result<LogicalPlan> {
        match op {
            LogicalOperator::ScanByLabel {
                variable,
                label,
                properties,
            } => self.df.build_scan(ctx, variable, label, properties),

            LogicalOperator::Filter { input, predicate } => {
                let child = self.build_native(ctx, input)?;
                let expr = to_df_boolean_expr(predicate);
                LogicalPlanBuilder::from(child)
                    .filter(expr)
                    .map_err(|e| self.plan_err("filter", e))?
                    .build()
                    .map_err(|e| self.plan_err("filter build", e))
            }

            LogicalOperator::Project { input, projections } => {
                let child = self.build_native(ctx, input)?;
                self.build_project_on(child, projections)
            }

            LogicalOperator::Sort { input, sort_items } => {
                let child = self.build_native(ctx, input)?;
                self.build_sort_on(child, sort_items)
            }

            LogicalOperator::Limit { input, count } => {
                let child = self.build_native(ctx, input)?;
                LogicalPlanBuilder::from(child)
                    .limit(0, Some(*count as usize))
                    .map_err(|e| self.plan_err("limit", e))?
                    .build()
                    .map_err(|e| self.plan_err("limit build", e))
            }

            LogicalOperator::Offset { input, offset } => {
                let child = self.build_native(ctx, input)?;
                LogicalPlanBuilder::from(child)
                    .limit(*offset as usize, None)
                    .map_err(|e| self.plan_err("offset", e))?
                    .build()
                    .map_err(|e| self.plan_err("offset build", e))
            }

            LogicalOperator::Distinct { input } => {
                let child = self.build_native(ctx, input)?;
                LogicalPlanBuilder::from(child)
                    .distinct()
                    .map_err(|e| self.plan_err("distinct", e))?
                    .build()
                    .map_err(|e| self.plan_err("distinct build", e))
            }

            LogicalOperator::Expand {
                input,
                source_variable,
                target_variable,
                target_label,
                relationship_types,
                direction,
                ..
            } => self.build_expand_native(
                ctx,
                input,
                source_variable,
                target_variable,
                target_label,
                relationship_types,
                direction,
            ),

            other => Err(GraphError::PlanError {
                message: format!("native planner reached unsupported operator: {:?}", other),
                location: snafu::Location::new(file!(), line!(), column!()),
            }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_expand_native(
        &self,
        ctx: &mut PlanningContext,
        input: &LogicalOperator,
        source_variable: &str,
        target_variable: &str,
        target_label: &str,
        relationship_types: &[String],
        direction: &RelationshipDirection,
    ) -> Result<LogicalPlan> {
        let source_plan = self.build_native(ctx, input)?;

        let rel_type = &relationship_types[0];
        let rel_map = self
            .config
            .get_relationship_mapping(rel_type)
            .ok_or_else(|| GraphError::ConfigError {
                message: format!("No relationship mapping for '{}'", rel_type),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        let src_label = ctx
            .analysis
            .var_to_label
            .get(source_variable)
            .ok_or_else(|| GraphError::PlanError {
                message: format!("No label for source variable '{}'", source_variable),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        let src_node = self
            .config
            .get_node_mapping(src_label)
            .ok_or_else(|| GraphError::ConfigError {
                message: format!("No node mapping for label '{}'", src_label),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        let tgt_node = self
            .config
            .get_node_mapping(target_label)
            .ok_or_else(|| GraphError::ConfigError {
                message: format!("No node mapping for label '{}'", target_label),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        let catalog = self.df.catalog_ref().ok_or_else(|| GraphError::ConfigError {
            message: "LanceNativePlanner requires a catalog for native expand".to_string(),
            location: snafu::Location::new(file!(), line!(), column!()),
        })?;
        let tgt_source = catalog.node_source(target_label).ok_or_else(|| {
            GraphError::ConfigError {
                message: format!("No table source for target label '{}'", target_label),
                location: snafu::Location::new(file!(), line!(), column!()),
            }
        })?;
        let tgt_arrow = tgt_source.schema();

        let native_direction = match direction {
            RelationshipDirection::Outgoing => NativeDirection::Outgoing,
            RelationshipDirection::Incoming => NativeDirection::Incoming,
            RelationshipDirection::Undirected => {
                return Err(GraphError::PlanError {
                    message: "undirected expand is not natively supported".to_string(),
                    location: snafu::Location::new(file!(), line!(), column!()),
                });
            }
        };

        let source_id_column = qualify_column(source_variable, &src_node.id_field);
        let neighbor_column = qualify_column(target_variable, &tgt_node.id_field);
        let id_lower = tgt_node.id_field.to_lowercase();
        let neighbor_data_type = tgt_arrow
            .field_with_name(&id_lower)
            .map_err(|e| GraphError::ConfigError {
                message: format!(
                    "target id field '{}' not found in '{}': {}",
                    tgt_node.id_field, target_label, e
                ),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?
            .data_type()
            .clone();

        // CsrExpandNode output schema = source schema + neighbor column.
        let src_arrow = source_plan.schema().inner();
        let mut expand_fields: Vec<Field> =
            src_arrow.fields().iter().map(|f| f.as_ref().clone()).collect();
        expand_fields.push(Field::new(&neighbor_column, neighbor_data_type.clone(), true));
        let expand_arrow = Schema::new(expand_fields);
        // Fields are intentionally unqualified: names are already the pre-qualified
        // `{var}__{col}` strings the reused DataFusion builders resolve by name.
        let expand_schema = Arc::new(
            DFSchema::try_from(expand_arrow).map_err(|e| self.plan_err("expand schema", e))?,
        );

        let expand_node = CsrExpandNode {
            input: source_plan,
            rel_type: rel_type.to_lowercase(),
            src_field: rel_map.source_id_field.to_lowercase(),
            dst_field: rel_map.target_id_field.to_lowercase(),
            direction: native_direction,
            source_id_column,
            neighbor_column: neighbor_column.clone(),
            neighbor_data_type,
            schema: expand_schema,
        };
        let expand_plan = LogicalPlan::Extension(Extension {
            node: Arc::new(expand_node),
        });

        // LanceTakeNode: materialize all target columns except the id field.
        let take_cols: Vec<String> = tgt_arrow
            .fields()
            .iter()
            .map(|f| f.name().to_lowercase())
            .filter(|n| n != &id_lower)
            .collect();

        let mut take_fields: Vec<Field> = expand_plan
            .schema()
            .inner()
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        for raw in &take_cols {
            let f = tgt_arrow
                .field_with_name(raw)
                .map_err(|e| self.plan_err("target field", e))?;
            take_fields.push(Field::new(
                qualify_column(target_variable, raw),
                f.data_type().clone(),
                true,
            ));
        }
        let take_arrow = Schema::new(take_fields);
        let take_schema =
            Arc::new(DFSchema::try_from(take_arrow).map_err(|e| self.plan_err("take schema", e))?);

        let take_node = LanceTakeNode {
            input: expand_plan,
            target_table: target_label.to_lowercase(),
            row_id_column: neighbor_column,
            take_cols,
            schema: take_schema,
        };
        Ok(LogicalPlan::Extension(Extension {
            node: Arc::new(take_node),
        }))
    }

    fn build_project_on(
        &self,
        input: LogicalPlan,
        projections: &[ProjectionItem],
    ) -> Result<LogicalPlan> {
        let has_agg = projections
            .iter()
            .any(|p| crate::datafusion_planner::expression::contains_aggregate(&p.expression));
        if has_agg {
            self.df.build_project_with_aggregates(input, projections)
        } else {
            self.df.build_simple_project(input, projections)
        }
    }

    fn build_sort_on(&self, input: LogicalPlan, sort_items: &[SortItem]) -> Result<LogicalPlan> {
        use datafusion::logical_expr::SortExpr;
        let sort_exprs: Vec<SortExpr> = sort_items
            .iter()
            .map(|item| {
                let expr = to_df_value_expr(&item.expression);
                let asc = matches!(item.direction, crate::ast::SortDirection::Ascending);
                SortExpr {
                    expr,
                    asc,
                    nulls_first: true,
                }
            })
            .collect();
        LogicalPlanBuilder::from(input)
            .sort(sort_exprs)
            .map_err(|e| self.plan_err("sort", e))?
            .build()
            .map_err(|e| self.plan_err("sort build", e))
    }

    fn plan_err<E: std::fmt::Display>(&self, what: &str, e: E) -> GraphError {
        GraphError::PlanError {
            message: format!("native {}: {}", what, e),
            location: snafu::Location::new(file!(), line!(), column!()),
        }
    }
}

/// True iff the plan is a single-hop expand the native planner can serve.
fn can_plan_natively(op: &LogicalOperator) -> bool {
    let mut expands = 0usize;
    if !walk_supported(op, &mut expands) {
        return false;
    }
    expands == 1
}

/// Recursively check every operator is natively supportable, accumulating the
/// total number of `Expand` nodes into `expands`. Returns false on any
/// unsupported operator. The caller enforces exactly one expand (`expands == 1`);
/// this function does not early-exit on the second expand.
fn walk_supported(op: &LogicalOperator, expands: &mut usize) -> bool {
    match op {
        LogicalOperator::ScanByLabel { .. } => true,
        LogicalOperator::Filter { input, .. }
        | LogicalOperator::Project { input, .. }
        | LogicalOperator::Sort { input, .. }
        | LogicalOperator::Limit { input, .. }
        | LogicalOperator::Offset { input, .. }
        | LogicalOperator::Distinct { input } => walk_supported(input, expands),
        LogicalOperator::Expand {
            input,
            relationship_types,
            direction,
            properties,
            target_properties,
            ..
        } => {
            *expands += 1;
            if relationship_types.len() != 1 {
                return false;
            }
            if matches!(direction, RelationshipDirection::Undirected) {
                return false;
            }
            // Inline relationship/target-node property filters are not handled by
            // the native path yet; fall back to the DataFusion join planner.
            if !properties.is_empty() || !target_properties.is_empty() {
                return false;
            }
            walk_supported(input, expands)
        }
        LogicalOperator::VariableLengthExpand { .. }
        | LogicalOperator::Join { .. }
        | LogicalOperator::Unwind { .. } => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{PropertyRef, ValueExpression};
    use crate::datafusion_planner::test_fixtures::{make_catalog, person_knows_config, person_scan};
    use crate::logical_plan::{LogicalOperator, ProjectionItem};

    fn knows_expand(direction: RelationshipDirection) -> LogicalOperator {
        LogicalOperator::Expand {
            input: Box::new(person_scan("a")),
            source_variable: "a".to_string(),
            target_variable: "b".to_string(),
            target_label: "Person".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction,
            relationship_variable: None,
            properties: Default::default(),
            target_properties: Default::default(),
        }
    }

    #[test]
    fn test_can_plan_natively_single_hop() {
        let plan = LogicalOperator::Project {
            input: Box::new(knows_expand(RelationshipDirection::Outgoing)),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef::new("b", "name")),
                alias: None,
            }],
        };
        assert!(can_plan_natively(&plan));
    }

    #[test]
    fn test_cannot_plan_undirected_or_multitype() {
        assert!(!can_plan_natively(&knows_expand(
            RelationshipDirection::Undirected
        )));
        let mut multi = knows_expand(RelationshipDirection::Outgoing);
        if let LogicalOperator::Expand {
            relationship_types, ..
        } = &mut multi
        {
            relationship_types.push("LIKES".to_string());
        }
        assert!(!can_plan_natively(&multi));
    }

    #[test]
    fn test_cannot_plan_zero_expands() {
        assert!(!can_plan_natively(&person_scan("a")));
    }

    #[test]
    fn test_native_plan_contains_extension_nodes() {
        let plan = LogicalOperator::Project {
            input: Box::new(knows_expand(RelationshipDirection::Outgoing)),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef::new("b", "name")),
                alias: None,
            }],
        };
        let planner = LanceNativePlanner::with_catalog(person_knows_config(), make_catalog());
        let df_plan = planner.plan(&plan).unwrap();
        let s = format!("{:?}", df_plan);
        assert!(s.contains("CsrExpand"), "missing CsrExpand: {}", s);
        assert!(s.contains("LanceTake"), "missing LanceTake: {}", s);
    }

    #[test]
    fn test_unsupported_falls_back_to_join() {
        let vlexpand = LogicalOperator::VariableLengthExpand {
            input: Box::new(person_scan("a")),
            source_variable: "a".into(),
            target_variable: "b".into(),
            relationship_types: vec!["KNOWS".into()],
            direction: RelationshipDirection::Outgoing,
            relationship_variable: None,
            min_length: Some(1),
            max_length: Some(2),
            target_properties: Default::default(),
        };
        let planner = LanceNativePlanner::with_catalog(person_knows_config(), make_catalog());
        let df_plan = planner.plan(&vlexpand).unwrap();
        let s = format!("{:?}", df_plan);
        assert!(!s.contains("CsrExpand"), "should not be native: {}", s);
    }

    #[test]
    fn test_cannot_plan_expand_with_inline_property_filters() {
        use crate::ast::PropertyValue;
        // Target-node inline filter -> must fall back.
        let mut op = knows_expand(RelationshipDirection::Outgoing);
        if let LogicalOperator::Expand { target_properties, .. } = &mut op {
            target_properties.insert("active".into(), PropertyValue::Boolean(true));
        }
        assert!(!can_plan_natively(&op));

        // Relationship inline filter -> must fall back.
        let mut op2 = knows_expand(RelationshipDirection::Outgoing);
        if let LogicalOperator::Expand { properties, .. } = &mut op2 {
            properties.insert("since".into(), PropertyValue::Integer(2020));
        }
        assert!(!can_plan_natively(&op2));
    }
}
