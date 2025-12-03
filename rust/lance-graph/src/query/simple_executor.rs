// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Helper functions for simple single-table query execution

/// Minimal translator for simple boolean expressions into DataFusion Expr
pub(super) fn to_df_boolean_expr_simple(
    expr: &crate::ast::BooleanExpression,
) -> Option<datafusion::logical_expr::Expr> {
    use crate::ast::{BooleanExpression as BE, ComparisonOperator as CO, ValueExpression as VE};
    use crate::query::expr::to_df_literal;
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
pub(super) fn to_df_order_by_expr_simple(
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
pub(super) fn to_df_value_expr_simple(
    expr: &crate::ast::ValueExpression,
) -> datafusion::logical_expr::Expr {
    use crate::ast::ValueExpression as VE;
    use crate::query::expr::to_df_literal;
    use datafusion::logical_expr::{col, lit};
    match expr {
        VE::Property(prop) => col(&prop.property),
        VE::Variable(v) => col(v),
        VE::Literal(v) => to_df_literal(v),
        VE::Function { .. } | VE::Arithmetic { .. } => lit(0),
    }
}
