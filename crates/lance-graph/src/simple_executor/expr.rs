// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Expression translation helpers for the simple executor

pub(super) fn to_df_boolean_expr_with_vars<F>(
    expr: &crate::ast::BooleanExpression,
    qualify: &F,
) -> Option<datafusion::logical_expr::Expr>
where
    F: Fn(&str, &str) -> String,
{
    use crate::ast::{BooleanExpression as BE, ComparisonOperator as CO, ValueExpression as VE};
    use datafusion::logical_expr::{col, Expr, Operator};
    match expr {
        BE::Comparison {
            left,
            operator,
            right,
        } => {
            let (var, prop, lit_expr) = match (left, right) {
                (VE::Property(p), VE::Literal(val)) => {
                    (p.variable.as_str(), p.property.as_str(), to_df_literal(val))
                }
                (VE::Literal(val), VE::Property(p)) => {
                    (p.variable.as_str(), p.property.as_str(), to_df_literal(val))
                }
                _ => return None,
            };
            let qualified = qualify(var, prop);
            let op = match operator {
                CO::Equal => Operator::Eq,
                CO::NotEqual => Operator::NotEq,
                CO::LessThan => Operator::Lt,
                CO::LessThanOrEqual => Operator::LtEq,
                CO::GreaterThan => Operator::Gt,
                CO::GreaterThanOrEqual => Operator::GtEq,
            };
            Some(Expr::BinaryExpr(datafusion::logical_expr::BinaryExpr {
                left: Box::new(col(&qualified)),
                op,
                right: Box::new(lit_expr),
            }))
        }
        BE::And(l, r) => Some(datafusion::logical_expr::Expr::BinaryExpr(
            datafusion::logical_expr::BinaryExpr {
                left: Box::new(to_df_boolean_expr_with_vars(l, qualify)?),
                op: Operator::And,
                right: Box::new(to_df_boolean_expr_with_vars(r, qualify)?),
            },
        )),
        BE::Or(l, r) => Some(datafusion::logical_expr::Expr::BinaryExpr(
            datafusion::logical_expr::BinaryExpr {
                left: Box::new(to_df_boolean_expr_with_vars(l, qualify)?),
                op: Operator::Or,
                right: Box::new(to_df_boolean_expr_with_vars(r, qualify)?),
            },
        )),
        BE::Not(inner) => Some(datafusion::logical_expr::Expr::Not(Box::new(
            to_df_boolean_expr_with_vars(inner, qualify)?,
        ))),
        _ => None,
    }
}

pub(super) fn to_df_literal(val: &crate::ast::PropertyValue) -> datafusion::logical_expr::Expr {
    use datafusion::logical_expr::lit;
    match val {
        crate::ast::PropertyValue::String(s) => lit(s.clone()),
        crate::ast::PropertyValue::Integer(i) => lit(*i),
        crate::ast::PropertyValue::Float(f) => lit(*f),
        crate::ast::PropertyValue::Boolean(b) => lit(*b),
        crate::ast::PropertyValue::Null => {
            datafusion::logical_expr::Expr::Literal(datafusion::scalar::ScalarValue::Null, None)
        }
        crate::ast::PropertyValue::Parameter(_) => lit(0),
        crate::ast::PropertyValue::Property(prop) => datafusion::logical_expr::col(&prop.property),
    }
}

/// Minimal translator for simple boolean expressions into DataFusion Expr
pub(crate) fn to_df_boolean_expr_simple(
    expr: &crate::ast::BooleanExpression,
) -> Option<datafusion::logical_expr::Expr> {
    use crate::ast::{BooleanExpression as BE, ComparisonOperator as CO, ValueExpression as VE};
    use datafusion::logical_expr::{col, Expr, Operator};
    match expr {
        BE::Comparison {
            left,
            operator,
            right,
        } => {
            let (col_name, lit_expr) = match (left, right) {
                (VE::Property(prop), VE::Literal(val)) => {
                    (prop.property.clone(), to_df_literal(val))
                }
                (VE::Literal(val), VE::Property(prop)) => {
                    (prop.property.clone(), to_df_literal(val))
                }
                _ => return None,
            };
            let op = match operator {
                CO::Equal => Operator::Eq,
                CO::NotEqual => Operator::NotEq,
                CO::LessThan => Operator::Lt,
                CO::LessThanOrEqual => Operator::LtEq,
                CO::GreaterThan => Operator::Gt,
                CO::GreaterThanOrEqual => Operator::GtEq,
            };
            Some(Expr::BinaryExpr(datafusion::logical_expr::BinaryExpr {
                left: Box::new(col(col_name)),
                op,
                right: Box::new(lit_expr),
            }))
        }
        BE::And(l, r) => Some(datafusion::logical_expr::Expr::BinaryExpr(
            datafusion::logical_expr::BinaryExpr {
                left: Box::new(to_df_boolean_expr_simple(l)?),
                op: Operator::And,
                right: Box::new(to_df_boolean_expr_simple(r)?),
            },
        )),
        BE::Or(l, r) => Some(datafusion::logical_expr::Expr::BinaryExpr(
            datafusion::logical_expr::BinaryExpr {
                left: Box::new(to_df_boolean_expr_simple(l)?),
                op: Operator::Or,
                right: Box::new(to_df_boolean_expr_simple(r)?),
            },
        )),
        BE::Not(inner) => Some(datafusion::logical_expr::Expr::Not(Box::new(
            to_df_boolean_expr_simple(inner)?,
        ))),
        BE::Exists(prop) => Some(datafusion::logical_expr::Expr::IsNotNull(Box::new(
            datafusion::logical_expr::Expr::Column(datafusion::common::Column::from_name(
                prop.property.clone(),
            )),
        ))),
        _ => None,
    }
}

/// Build ORDER BY expressions for simple queries
pub(crate) fn to_df_order_by_expr_simple(
    items: &[crate::ast::OrderByItem],
) -> Vec<datafusion::logical_expr::SortExpr> {
    use datafusion::logical_expr::SortExpr;
    items
        .iter()
        .map(|item| {
            let expr = to_df_value_expr_simple(&item.expression);
            let asc = matches!(item.direction, crate::ast::SortDirection::Ascending);
            SortExpr {
                expr,
                asc,
                nulls_first: false,
            }
        })
        .collect()
}

/// Build value expressions for simple queries
pub(crate) fn to_df_value_expr_simple(
    expr: &crate::ast::ValueExpression,
) -> datafusion::logical_expr::Expr {
    use crate::ast::ValueExpression as VE;
    use datafusion::functions::string::{lower, upper};
    use datafusion::logical_expr::{col, lit, BinaryExpr, Expr, Operator};
    match expr {
        VE::Property(prop) => col(&prop.property),
        VE::Variable(v) => col(v),
        VE::Literal(v) => to_df_literal(v),
        VE::Function { name, args } => match name.to_lowercase().as_str() {
            "tolower" | "lower" => {
                if args.len() == 1 {
                    lower().call(vec![to_df_value_expr_simple(&args[0])])
                } else {
                    Expr::Literal(datafusion::scalar::ScalarValue::Null, None)
                }
            }
            "toupper" | "upper" => {
                if args.len() == 1 {
                    upper().call(vec![to_df_value_expr_simple(&args[0])])
                } else {
                    Expr::Literal(datafusion::scalar::ScalarValue::Null, None)
                }
            }
            _ => Expr::Literal(datafusion::scalar::ScalarValue::Null, None),
        },
        VE::Arithmetic {
            left,
            operator,
            right,
        } => {
            use crate::ast::ArithmeticOperator as AO;
            let l = to_df_value_expr_simple(left);
            let r = to_df_value_expr_simple(right);
            let op = match operator {
                AO::Add => Operator::Plus,
                AO::Subtract => Operator::Minus,
                AO::Multiply => Operator::Multiply,
                AO::Divide => Operator::Divide,
                AO::Modulo => Operator::Modulo,
            };
            Expr::BinaryExpr(BinaryExpr {
                left: Box::new(l),
                op,
                right: Box::new(r),
            })
        }
        VE::VectorDistance { .. } => lit(0.0f32),
        VE::VectorSimilarity { .. } => lit(1.0f32),
        VE::Parameter(_) => lit(0),
        VE::VectorLiteral(_) => lit(0.0f32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{ArithmeticOperator, PropertyRef, ValueExpression};
    use datafusion::logical_expr::Expr;
    use datafusion::scalar::ScalarValue;

    #[test]
    fn test_simple_expr_unknown_function_returns_null() {
        let expr = ValueExpression::Function {
            name: "replace".to_string(),
            args: vec![ValueExpression::Property(PropertyRef::new("p", "name"))],
        };
        let df_expr = to_df_value_expr_simple(&expr);
        assert!(matches!(df_expr, Expr::Literal(ScalarValue::Null, _)));
    }

    #[test]
    fn test_simple_expr_lower_wrong_arity_returns_null() {
        let expr = ValueExpression::Function {
            name: "lower".to_string(),
            args: vec![
                ValueExpression::Property(PropertyRef::new("p", "name")),
                ValueExpression::Property(PropertyRef::new("p", "name")),
            ],
        };
        let df_expr = to_df_value_expr_simple(&expr);
        assert!(matches!(df_expr, Expr::Literal(ScalarValue::Null, _)));
    }

    #[test]
    fn test_simple_expr_arithmetic_builds_binary_expr() {
        let expr = ValueExpression::Arithmetic {
            left: Box::new(ValueExpression::Variable("x".to_string())),
            operator: ArithmeticOperator::Add,
            right: Box::new(ValueExpression::Literal(
                crate::ast::PropertyValue::Integer(1),
            )),
        };
        let df_expr = to_df_value_expr_simple(&expr);
        assert!(matches!(df_expr, Expr::BinaryExpr(_)));
    }
}
