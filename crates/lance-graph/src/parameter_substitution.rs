use crate::ast::*;
use crate::error::{GraphError, Result};
use std::collections::HashMap;

/// Substitute parameters with literal values in the AST
pub fn substitute_parameters(
    query: &mut CypherQuery,
    parameters: &HashMap<String, serde_json::Value>,
) -> Result<()> {
    // Substitute in READING clauses
    for clause in &mut query.reading_clauses {
        substitute_in_reading_clause(clause, parameters)?;
    }

    // Substitute in WHERE clause
    if let Some(where_clause) = &mut query.where_clause {
        substitute_in_where_clause(where_clause, parameters)?;
    }

    // Substitute in WITH clause
    if let Some(with_clause) = &mut query.with_clause {
        substitute_in_with_clause(with_clause, parameters)?;
    }

    // Substitute in post-WITH READING clauses
    for clause in &mut query.post_with_reading_clauses {
        substitute_in_reading_clause(clause, parameters)?;
    }

    // Substitute in post-WITH WHERE clause
    if let Some(post_where) = &mut query.post_with_where_clause {
        substitute_in_where_clause(post_where, parameters)?;
    }

    // Substitute in RETURN clause
    substitute_in_return_clause(&mut query.return_clause, parameters)?;

    // Substitute in ORDER BY clause
    if let Some(order_by) = &mut query.order_by {
        substitute_in_order_by_clause(order_by, parameters)?;
    }

    Ok(())
}

fn substitute_in_reading_clause(
    clause: &mut ReadingClause,
    parameters: &HashMap<String, serde_json::Value>,
) -> Result<()> {
    match clause {
        ReadingClause::Match(match_clause) => {
            for pattern in &mut match_clause.patterns {
                substitute_in_graph_pattern(pattern, parameters)?;
            }
        }
        ReadingClause::Unwind(unwind_clause) => {
            substitute_in_value_expression(&mut unwind_clause.expression, parameters)?;
        }
    }
    Ok(())
}

fn substitute_in_graph_pattern(
    pattern: &mut GraphPattern,
    parameters: &HashMap<String, serde_json::Value>,
) -> Result<()> {
    match pattern {
        GraphPattern::Node(node) => {
            for value in node.properties.values_mut() {
                substitute_in_property_value(value, parameters)?;
            }
        }
        GraphPattern::Path(path) => {
            substitute_in_node_pattern(&mut path.start_node, parameters)?;
            for segment in &mut path.segments {
                substitute_in_relationship_pattern(&mut segment.relationship, parameters)?;
                substitute_in_node_pattern(&mut segment.end_node, parameters)?;
            }
        }
    }
    Ok(())
}

fn substitute_in_node_pattern(
    node: &mut NodePattern,
    parameters: &HashMap<String, serde_json::Value>,
) -> Result<()> {
    for value in node.properties.values_mut() {
        substitute_in_property_value(value, parameters)?;
    }
    Ok(())
}

fn substitute_in_relationship_pattern(
    rel: &mut RelationshipPattern,
    parameters: &HashMap<String, serde_json::Value>,
) -> Result<()> {
    for value in rel.properties.values_mut() {
        substitute_in_property_value(value, parameters)?;
    }
    Ok(())
}

fn substitute_in_property_value(
    value: &mut PropertyValue,
    parameters: &HashMap<String, serde_json::Value>,
) -> Result<()> {
    if let PropertyValue::Parameter(name) = value {
        let param_value = parameters.get(name).ok_or_else(|| GraphError::PlanError {
            message: format!("Missing parameter: ${}", name),
            location: snafu::Location::new(file!(), line!(), column!()),
        })?;

        *value = json_to_property_value(param_value)?;
    }
    Ok(())
}

fn substitute_in_where_clause(
    where_clause: &mut WhereClause,
    parameters: &HashMap<String, serde_json::Value>,
) -> Result<()> {
    substitute_in_boolean_expression(&mut where_clause.expression, parameters)
}

fn substitute_in_with_clause(
    with_clause: &mut WithClause,
    parameters: &HashMap<String, serde_json::Value>,
) -> Result<()> {
    for item in &mut with_clause.items {
        substitute_in_value_expression(&mut item.expression, parameters)?;
    }
    if let Some(order_by) = &mut with_clause.order_by {
        substitute_in_order_by_clause(order_by, parameters)?;
    }
    Ok(())
}

fn substitute_in_return_clause(
    return_clause: &mut ReturnClause,
    parameters: &HashMap<String, serde_json::Value>,
) -> Result<()> {
    for item in &mut return_clause.items {
        substitute_in_value_expression(&mut item.expression, parameters)?;
    }
    Ok(())
}

fn substitute_in_order_by_clause(
    order_by: &mut OrderByClause,
    parameters: &HashMap<String, serde_json::Value>,
) -> Result<()> {
    for item in &mut order_by.items {
        substitute_in_value_expression(&mut item.expression, parameters)?;
    }
    Ok(())
}

fn substitute_in_boolean_expression(
    expr: &mut BooleanExpression,
    parameters: &HashMap<String, serde_json::Value>,
) -> Result<()> {
    match expr {
        BooleanExpression::Comparison { left, right, .. } => {
            substitute_in_value_expression(left, parameters)?;
            substitute_in_value_expression(right, parameters)?;
        }
        BooleanExpression::And(left, right) | BooleanExpression::Or(left, right) => {
            substitute_in_boolean_expression(left, parameters)?;
            substitute_in_boolean_expression(right, parameters)?;
        }
        BooleanExpression::Not(inner) => {
            substitute_in_boolean_expression(inner, parameters)?;
        }
        BooleanExpression::Exists(_) => {}
        BooleanExpression::In { expression, list } => {
            substitute_in_value_expression(expression, parameters)?;
            for item in list {
                substitute_in_value_expression(item, parameters)?;
            }
        }
        BooleanExpression::Like { expression, .. }
        | BooleanExpression::ILike { expression, .. }
        | BooleanExpression::Contains { expression, .. }
        | BooleanExpression::StartsWith { expression, .. }
        | BooleanExpression::EndsWith { expression, .. }
        | BooleanExpression::IsNull(expression)
        | BooleanExpression::IsNotNull(expression) => {
            substitute_in_value_expression(expression, parameters)?;
        }
    }
    Ok(())
}

fn substitute_in_value_expression(
    expr: &mut ValueExpression,
    parameters: &HashMap<String, serde_json::Value>,
) -> Result<()> {
    match expr {
        ValueExpression::Parameter(name) => {
            let param_value = parameters.get(name).ok_or_else(|| GraphError::PlanError {
                message: format!("Missing parameter: ${}", name),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

            // Check for array to VectorLiteral conversion
            if let serde_json::Value::Array(arr) = param_value {
                let mut floats = Vec::new();
                for v in arr {
                    if let Some(f) = v.as_f64() {
                        floats.push(f as f32);
                    } else {
                        return Err(GraphError::PlanError {
                            message: format!(
                                "Parameter ${} is a list but contains non-numeric values. Only float vectors are supported as list parameters currently.",
                                name
                            ),
                            location: snafu::Location::new(file!(), line!(), column!()),
                        });
                    }
                }
                *expr = ValueExpression::VectorLiteral(floats);
                return Ok(());
            }

            // Scalar conversion
            let prop_val = json_to_property_value(param_value)?;
            *expr = ValueExpression::Literal(prop_val);
        }
        ValueExpression::ScalarFunction { args, .. }
        | ValueExpression::AggregateFunction { args, .. } => {
            for arg in args {
                substitute_in_value_expression(arg, parameters)?;
            }
        }
        ValueExpression::Arithmetic { left, right, .. } => {
            substitute_in_value_expression(left, parameters)?;
            substitute_in_value_expression(right, parameters)?;
        }
        ValueExpression::VectorDistance { left, right, .. }
        | ValueExpression::VectorSimilarity { left, right, .. } => {
            substitute_in_value_expression(left, parameters)?;
            substitute_in_value_expression(right, parameters)?;
        }
        _ => {}
    }
    Ok(())
}

fn json_to_property_value(value: &serde_json::Value) -> Result<PropertyValue> {
    match value {
        serde_json::Value::Null => Ok(PropertyValue::Null),
        serde_json::Value::Bool(b) => Ok(PropertyValue::Boolean(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(PropertyValue::Integer(i))
            } else if let Some(f) = n.as_f64() {
                Ok(PropertyValue::Float(f))
            } else {
                Ok(PropertyValue::Null)
            }
        }
        serde_json::Value::String(s) => Ok(PropertyValue::String(s.clone())),
        serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
            Err(GraphError::PlanError {
                message: "Complex types (List, Map) are not fully supported as parameters yet (except float vectors).".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            })
        }
    }
}
