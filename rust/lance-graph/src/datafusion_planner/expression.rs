// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Expression Translation
//!
//! Converts AST expressions to DataFusion expressions

use crate::ast::{BooleanExpression, PropertyValue, ValueExpression};
use crate::datafusion_planner::udf;
use datafusion::functions::string::lower;
use datafusion::functions::string::upper;
use datafusion::logical_expr::{col, lit, BinaryExpr, Expr, Operator};
use datafusion_functions_aggregate::average::avg;
use datafusion_functions_aggregate::count::count;
use datafusion_functions_aggregate::min_max::max;
use datafusion_functions_aggregate::min_max::min;
use datafusion_functions_aggregate::sum::sum;

/// Helper function to create LIKE expressions with consistent settings
fn create_like_expr(expression: &ValueExpression, pattern: &str, case_insensitive: bool) -> Expr {
    Expr::Like(datafusion::logical_expr::Like {
        negated: false,
        expr: Box::new(to_df_value_expr(expression)),
        pattern: Box::new(lit(pattern.to_string())),
        escape_char: None,
        case_insensitive,
    })
}

/// Convert BooleanExpression to DataFusion Expr
pub(crate) fn to_df_boolean_expr(expr: &BooleanExpression) -> Expr {
    use crate::ast::{BooleanExpression as BE, ComparisonOperator as CO};
    match expr {
        BE::Comparison {
            left,
            operator,
            right,
        } => {
            let l = to_df_value_expr(left);
            let r = to_df_value_expr(right);
            let op = match operator {
                CO::Equal => Operator::Eq,
                CO::NotEqual => Operator::NotEq,
                CO::LessThan => Operator::Lt,
                CO::LessThanOrEqual => Operator::LtEq,
                CO::GreaterThan => Operator::Gt,
                CO::GreaterThanOrEqual => Operator::GtEq,
            };
            Expr::BinaryExpr(BinaryExpr {
                left: Box::new(l),
                op,
                right: Box::new(r),
            })
        }
        BE::In { expression, list } => {
            use datafusion::logical_expr::expr::InList as DFInList;
            let expr = to_df_value_expr(expression);
            let list_exprs = list.iter().map(to_df_value_expr).collect::<Vec<_>>();
            Expr::InList(DFInList::new(Box::new(expr), list_exprs, false))
        }
        BE::And(l, r) => Expr::BinaryExpr(BinaryExpr {
            left: Box::new(to_df_boolean_expr(l)),
            op: Operator::And,
            right: Box::new(to_df_boolean_expr(r)),
        }),
        BE::Or(l, r) => Expr::BinaryExpr(BinaryExpr {
            left: Box::new(to_df_boolean_expr(l)),
            op: Operator::Or,
            right: Box::new(to_df_boolean_expr(r)),
        }),
        BE::Not(inner) => Expr::Not(Box::new(to_df_boolean_expr(inner))),
        BE::Exists(prop) => Expr::IsNotNull(Box::new(to_df_value_expr(
            &ValueExpression::Property(prop.clone()),
        ))),
        BE::IsNull(expression) => Expr::IsNull(Box::new(to_df_value_expr(expression))),
        BE::IsNotNull(expression) => Expr::IsNotNull(Box::new(to_df_value_expr(expression))),
        BE::Like {
            expression,
            pattern,
        } => create_like_expr(expression, pattern, false),
        BE::ILike {
            expression,
            pattern,
        } => create_like_expr(expression, pattern, true),
        BE::Contains {
            expression,
            substring,
        } => {
            // CONTAINS is equivalent to LIKE '%substring%'
            let pattern = format!("%{}%", substring);
            create_like_expr(expression, &pattern, false)
        }
        BE::StartsWith { expression, prefix } => {
            // STARTS WITH is equivalent to LIKE 'prefix%'
            let pattern = format!("{}%", prefix);
            create_like_expr(expression, &pattern, false)
        }
        BE::EndsWith { expression, suffix } => {
            // ENDS WITH is equivalent to LIKE '%suffix'
            let pattern = format!("%{}", suffix);
            create_like_expr(expression, &pattern, false)
        }
    }
}

/// Convert ValueExpression to DataFusion Expr
pub(crate) fn to_df_value_expr(expr: &ValueExpression) -> Expr {
    use crate::ast::{PropertyValue as PV, ValueExpression as VE};
    match expr {
        VE::Property(prop) => {
            // Create qualified column name: variable__property
            let qualified_name = format!("{}__{}", prop.variable, prop.property);
            col(&qualified_name)
        }
        VE::Variable(v) => col(v),
        VE::Literal(PV::String(s)) => lit(s.clone()),
        VE::Literal(PV::Integer(i)) => lit(*i),
        VE::Literal(PV::Float(f)) => lit(*f),
        VE::Literal(PV::Boolean(b)) => lit(*b),
        VE::Literal(PV::Null) => {
            datafusion::logical_expr::Expr::Literal(datafusion::scalar::ScalarValue::Null, None)
        }
        VE::Literal(PV::Parameter(_)) => lit(0),
        VE::Literal(PV::Property(prop)) => {
            // Create qualified column name: variable__property
            let qualified_name = format!("{}__{}", prop.variable, prop.property);
            col(&qualified_name)
        }
        VE::Function { name, args } => {
            // Handle aggregation functions
            match name.to_lowercase().as_str() {
                "count" => {
                    if args.len() == 1 {
                        let arg_expr = if let VE::Variable(v) = &args[0] {
                            if v == "*" {
                                // COUNT(*) - count all rows including NULLs
                                lit(1)
                            } else {
                                // COUNT(p) - count non-NULL rows by using a representative column
                                // Use <variable>__id as a null-sensitive column
                                // This ensures optional matches with NULL variables are not counted
                                col(format!("{}__id", v))
                            }
                        } else {
                            // COUNT(p.property) - count non-null values of that property
                            to_df_value_expr(&args[0])
                        };

                        // Use DataFusion's count helper function
                        count(arg_expr)
                    } else {
                        // Invalid argument count - return placeholder
                        lit(0)
                    }
                }
                "sum" => {
                    if args.len() == 1 {
                        // Note: SUM(variable) is rejected by semantic validation
                        // So we only handle valid cases here
                        let arg_expr = to_df_value_expr(&args[0]);
                        sum(arg_expr)
                    } else {
                        // Invalid argument count - return placeholder
                        lit(0)
                    }
                }
                "avg" => {
                    if args.len() == 1 {
                        let arg_expr = to_df_value_expr(&args[0]);
                        avg(arg_expr)
                    } else {
                        lit(0)
                    }
                }
                "min" => {
                    if args.len() == 1 {
                        let arg_expr = to_df_value_expr(&args[0]);
                        min(arg_expr)
                    } else {
                        lit(0)
                    }
                }
                "max" => {
                    if args.len() == 1 {
                        let arg_expr = to_df_value_expr(&args[0]);
                        max(arg_expr)
                    } else {
                        lit(0)
                    }
                }
                // String functions
                "tolower" | "lower" => {
                    if args.len() == 1 {
                        let arg_expr = to_df_value_expr(&args[0]);
                        lower().call(vec![arg_expr])
                    } else {
                        // Invalid argument count - return NULL
                        Expr::Literal(datafusion::scalar::ScalarValue::Null, None)
                    }
                }
                "toupper" | "upper" => {
                    if args.len() == 1 {
                        let arg_expr = to_df_value_expr(&args[0]);
                        upper().call(vec![arg_expr])
                    } else {
                        // Invalid argument count - return NULL
                        Expr::Literal(datafusion::scalar::ScalarValue::Null, None)
                    }
                }
                _ => {
                    // Unsupported function - return NULL which coerces to any type
                    // This prevents type coercion errors in both string and numeric contexts
                    Expr::Literal(datafusion::scalar::ScalarValue::Null, None)
                }
            }
        }
        VE::Arithmetic {
            left,
            operator,
            right,
        } => {
            use crate::ast::ArithmeticOperator as AO;
            let l = to_df_value_expr(left);
            let r = to_df_value_expr(right);
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
        VE::VectorDistance {
            left,
            right,
            metric,
        } => {
            // Create UDF for vector distance computation
            let udf = udf::create_vector_distance_udf(metric);
            let left_expr = to_df_value_expr(left);
            let right_expr = to_df_value_expr(right);
            Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
                udf,
                vec![left_expr, right_expr],
            ))
        }
        VE::VectorSimilarity {
            left,
            right,
            metric,
        } => {
            // Create UDF for vector similarity computation
            let udf = udf::create_vector_similarity_udf(metric);
            let left_expr = to_df_value_expr(left);
            let right_expr = to_df_value_expr(right);
            Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
                udf,
                vec![left_expr, right_expr],
            ))
        }
        VE::VectorLiteral(values) => {
            // Convert Vec<f32> to DataFusion scalar FixedSizeList
            use arrow::array::{FixedSizeListArray, Float32Array};
            use arrow::datatypes::{DataType, Field};
            use datafusion::scalar::ScalarValue;
            use std::sync::Arc;

            let dim = values.len() as i32;
            let field = Arc::new(Field::new("item", DataType::Float32, true));
            let float_array = Arc::new(Float32Array::from(values.clone()));

            let list_array = FixedSizeListArray::try_new(field.clone(), dim, float_array, None)
                .expect("Failed to create FixedSizeListArray for vector literal");

            let scalar = ScalarValue::try_from_array(&list_array, 0)
                .expect("Failed to create scalar from array");

            lit(scalar)
        }
        VE::Parameter(name) => {
            // TODO: Implement proper parameter resolution
            // Parameters ($param) should be resolved to literal values from the query's
            // parameter map (CypherQuery::parameters()) before or during planning.
            //
            // Current limitation: This creates a column reference as a placeholder,
            // which will fail at execution if the column doesn't exist.
            //
            // Proper fix requires one of:
            // 1. Resolve parameters during semantic analysis (substitute before planning)
            // 2. Pass parameter map to to_df_value_expr and resolve here
            // 3. Use DataFusion's parameter binding mechanism
            col(format!("${}", name))
        }
    }
}

/// Check if a ValueExpression contains an aggregate function
pub(crate) fn contains_aggregate(expr: &ValueExpression) -> bool {
    use crate::ast::ValueExpression as VE;
    match expr {
        VE::Function { name, args } => {
            // Check if this is an aggregate function
            let is_aggregate = matches!(
                name.to_lowercase().as_str(),
                "count" | "sum" | "avg" | "min" | "max"
            );
            // Also check arguments recursively
            is_aggregate || args.iter().any(contains_aggregate)
        }
        VE::Arithmetic { left, right, .. } => contains_aggregate(left) || contains_aggregate(right),
        VE::VectorDistance { left, right, .. } => {
            contains_aggregate(left) || contains_aggregate(right)
        }
        VE::VectorSimilarity { left, right, .. } => {
            contains_aggregate(left) || contains_aggregate(right)
        }
        _ => false,
    }
}

/// Convert a ValueExpression to Cypher dot notation for column naming
///
/// This generates user-friendly column names following Cypher conventions:
/// - Property references: `p.name` (variable.property)
/// - Functions: `function_name(arg)` with simplified argument representation
/// - Other expressions: Use the expression as-is
///
/// This is used when no explicit alias is provided in RETURN clauses.
pub(crate) fn to_cypher_column_name(expr: &ValueExpression) -> String {
    use crate::ast::ValueExpression as VE;
    match expr {
        VE::Property(prop) => {
            // Convert to Cypher dot notation: variable.property
            format!("{}.{}", prop.variable, prop.property)
        }
        VE::Variable(v) => v.clone(),
        VE::Literal(PropertyValue::Property(prop)) => {
            // Handle nested property references
            format!("{}.{}", prop.variable, prop.property)
        }
        VE::Function { name, args } => {
            // Generate descriptive function name: count(*), count(p.name), etc.
            if args.len() == 1 {
                let arg_repr = match &args[0] {
                    VE::Variable(v) => v.clone(),
                    VE::Property(prop) => format!("{}.{}", prop.variable, prop.property),
                    _ => "expr".to_string(),
                };
                format!("{}({})", name.to_lowercase(), arg_repr)
            } else if args.is_empty() {
                format!("{}()", name.to_lowercase())
            } else {
                // Multiple args - just use function name
                name.to_lowercase()
            }
        }
        _ => {
            // For other expressions (literals, arithmetic), use a generic name
            // In practice, these should always have explicit aliases
            "expr".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BooleanExpression, PropertyRef, PropertyValue, ValueExpression};
    use datafusion::logical_expr::Expr;

    // ========================================================================
    // Unit tests for to_df_boolean_expr()
    // ========================================================================

    #[test]
    fn test_boolean_expr_comparison_equal() {
        use crate::ast::ComparisonOperator;
        let expr = BooleanExpression::Comparison {
            left: ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "age".into(),
            }),
            operator: ComparisonOperator::Equal,
            right: ValueExpression::Literal(PropertyValue::Integer(30)),
        };

        let df_expr = to_df_boolean_expr(&expr);
        let s = format!("{:?}", df_expr);
        assert!(s.contains("p__age"), "Should contain qualified column");
        assert!(
            s.contains("30") || s.contains("Int64(30)"),
            "Should contain value"
        );
    }

    #[test]
    fn test_boolean_expr_comparison_operators() {
        use crate::ast::ComparisonOperator;

        let operators = vec![
            (ComparisonOperator::NotEqual, "!="),
            (ComparisonOperator::LessThan, "<"),
            (ComparisonOperator::LessThanOrEqual, "<="),
            (ComparisonOperator::GreaterThan, ">"),
            (ComparisonOperator::GreaterThanOrEqual, ">="),
        ];

        for (op, _op_str) in operators {
            let expr = BooleanExpression::Comparison {
                left: ValueExpression::Property(PropertyRef {
                    variable: "p".into(),
                    property: "age".into(),
                }),
                operator: op,
                right: ValueExpression::Literal(PropertyValue::Integer(30)),
            };

            let df_expr = to_df_boolean_expr(&expr);
            // Should successfully translate without panicking
            assert!(format!("{:?}", df_expr).contains("p__age"));
        }
    }

    #[test]
    fn test_boolean_expr_and() {
        use crate::ast::ComparisonOperator;
        let expr = BooleanExpression::And(
            Box::new(BooleanExpression::Comparison {
                left: ValueExpression::Property(PropertyRef {
                    variable: "p".into(),
                    property: "age".into(),
                }),
                operator: ComparisonOperator::GreaterThan,
                right: ValueExpression::Literal(PropertyValue::Integer(20)),
            }),
            Box::new(BooleanExpression::Comparison {
                left: ValueExpression::Property(PropertyRef {
                    variable: "p".into(),
                    property: "age".into(),
                }),
                operator: ComparisonOperator::LessThan,
                right: ValueExpression::Literal(PropertyValue::Integer(50)),
            }),
        );

        let df_expr = to_df_boolean_expr(&expr);
        let s = format!("{:?}", df_expr);
        assert!(s.contains("And"), "Should contain AND operator");
        assert!(s.contains("p__age"), "Should contain column reference");
    }

    #[test]
    fn test_boolean_expr_or() {
        use crate::ast::ComparisonOperator;
        let expr = BooleanExpression::Or(
            Box::new(BooleanExpression::Comparison {
                left: ValueExpression::Property(PropertyRef {
                    variable: "p".into(),
                    property: "name".into(),
                }),
                operator: ComparisonOperator::Equal,
                right: ValueExpression::Literal(PropertyValue::String("Alice".into())),
            }),
            Box::new(BooleanExpression::Comparison {
                left: ValueExpression::Property(PropertyRef {
                    variable: "p".into(),
                    property: "name".into(),
                }),
                operator: ComparisonOperator::Equal,
                right: ValueExpression::Literal(PropertyValue::String("Bob".into())),
            }),
        );

        let df_expr = to_df_boolean_expr(&expr);
        let s = format!("{:?}", df_expr);
        assert!(s.contains("Or"), "Should contain OR operator");
    }

    #[test]
    fn test_boolean_expr_not() {
        use crate::ast::ComparisonOperator;
        let expr = BooleanExpression::Not(Box::new(BooleanExpression::Comparison {
            left: ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "active".into(),
            }),
            operator: ComparisonOperator::Equal,
            right: ValueExpression::Literal(PropertyValue::Boolean(true)),
        }));

        let df_expr = to_df_boolean_expr(&expr);
        let s = format!("{:?}", df_expr);
        assert!(s.contains("Not"), "Should contain NOT operator");
    }

    #[test]
    fn test_boolean_expr_exists() {
        let expr = BooleanExpression::Exists(PropertyRef {
            variable: "p".into(),
            property: "email".into(),
        });

        let df_expr = to_df_boolean_expr(&expr);
        let s = format!("{:?}", df_expr);
        assert!(
            s.contains("IsNotNull") || s.contains("p__email"),
            "Should translate EXISTS to IsNotNull"
        );
    }

    #[test]
    fn test_boolean_expr_in_list() {
        let expr = BooleanExpression::In {
            expression: ValueExpression::Property(PropertyRef {
                variable: "rel".into(),
                property: "relationship_type".into(),
            }),
            list: vec![
                ValueExpression::Literal(PropertyValue::String("WORKS_FOR".into())),
                ValueExpression::Literal(PropertyValue::String("PART_OF".into())),
            ],
        };

        if let Expr::InList(in_list) = to_df_boolean_expr(&expr) {
            assert!(!in_list.negated);
            assert_eq!(in_list.list.len(), 2);
            match *in_list.expr {
                Expr::Column(ref col_expr) => {
                    assert_eq!(col_expr.name(), "rel__relationship_type");
                }
                other => panic!("Expected column expression, got {:?}", other),
            }
        } else {
            panic!("Expected InList expression");
        }
    }

    #[test]
    fn test_boolean_expr_like() {
        let expr = BooleanExpression::Like {
            expression: ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "name".into(),
            }),
            pattern: "A%".into(),
        };

        if let Expr::Like(like_expr) = to_df_boolean_expr(&expr) {
            assert!(!like_expr.negated, "Should not be negated");
            assert!(!like_expr.case_insensitive, "Should be case sensitive");
            assert_eq!(like_expr.escape_char, None, "Should have no escape char");
            match *like_expr.expr {
                Expr::Column(ref col_expr) => {
                    assert_eq!(col_expr.name(), "p__name");
                }
                other => panic!("Expected column expression, got {:?}", other),
            }
            // Check pattern is a literal
            match *like_expr.pattern {
                Expr::Literal(..) => {} // Success
                other => panic!("Expected literal pattern, got {:?}", other),
            }
        } else {
            panic!("Expected Like expression");
        }
    }

    #[test]
    fn test_boolean_expr_ilike() {
        let expr = BooleanExpression::ILike {
            expression: ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "name".into(),
            }),
            pattern: "alice%".into(),
        };

        if let Expr::Like(like_expr) = to_df_boolean_expr(&expr) {
            assert!(!like_expr.negated, "Should not be negated");
            assert!(
                like_expr.case_insensitive,
                "ILIKE should be case insensitive"
            );
            assert_eq!(like_expr.escape_char, None, "Should have no escape char");
            match *like_expr.expr {
                Expr::Column(ref col_expr) => {
                    assert_eq!(col_expr.name(), "p__name");
                }
                other => panic!("Expected column expression, got {:?}", other),
            }
            // Check pattern is a literal
            match *like_expr.pattern {
                Expr::Literal(ref scalar, _) => {
                    let s = format!("{:?}", scalar);
                    assert!(
                        s.contains("alice%"),
                        "Pattern should be 'alice%', got: {}",
                        s
                    );
                }
                other => panic!("Expected literal pattern, got {:?}", other),
            }
        } else {
            panic!("Expected Like expression");
        }
    }

    #[test]
    fn test_boolean_expr_like_vs_ilike_case_sensitivity() {
        // Test LIKE (case-sensitive)
        let like_expr = BooleanExpression::Like {
            expression: ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "name".into(),
            }),
            pattern: "Test%".into(),
        };

        if let Expr::Like(like) = to_df_boolean_expr(&like_expr) {
            assert!(
                !like.case_insensitive,
                "LIKE should be case-sensitive (case_insensitive = false)"
            );
        } else {
            panic!("Expected Like expression");
        }

        // Test ILIKE (case-insensitive)
        let ilike_expr = BooleanExpression::ILike {
            expression: ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "name".into(),
            }),
            pattern: "Test%".into(),
        };

        if let Expr::Like(ilike) = to_df_boolean_expr(&ilike_expr) {
            assert!(
                ilike.case_insensitive,
                "ILIKE should be case-insensitive (case_insensitive = true)"
            );
        } else {
            panic!("Expected Like expression");
        }
    }

    #[test]
    fn test_boolean_expr_like_with_wildcard() {
        let expr = BooleanExpression::Like {
            expression: ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "email".into(),
            }),
            pattern: "%@example.com".into(),
        };

        let df_expr = to_df_boolean_expr(&expr);
        let s = format!("{:?}", df_expr);
        assert!(
            s.contains("Like") || s.contains("like"),
            "Should be a LIKE expression"
        );
        assert!(s.contains("p__email"), "Should contain column reference");
    }

    #[test]
    fn test_boolean_expr_contains() {
        let expr = BooleanExpression::Contains {
            expression: ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "name".into(),
            }),
            substring: "ali".into(),
        };

        if let Expr::Like(like_expr) = to_df_boolean_expr(&expr) {
            assert!(!like_expr.negated, "Should not be negated");
            assert!(!like_expr.case_insensitive, "Should be case sensitive");
            assert_eq!(like_expr.escape_char, None, "Should have no escape char");

            // Check the expression is the column
            match *like_expr.expr {
                Expr::Column(ref col_expr) => {
                    assert_eq!(col_expr.name(), "p__name");
                }
                other => panic!("Expected column expression, got {:?}", other),
            }

            // Check pattern is '%ali%'
            match *like_expr.pattern {
                Expr::Literal(ref scalar, _) => {
                    let s = format!("{:?}", scalar);
                    assert!(s.contains("%ali%"), "Pattern should be '%ali%', got: {}", s);
                }
                other => panic!("Expected literal pattern, got {:?}", other),
            }
        } else {
            panic!("Expected Like expression");
        }
    }

    #[test]
    fn test_boolean_expr_starts_with() {
        let expr = BooleanExpression::StartsWith {
            expression: ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "email".into(),
            }),
            prefix: "admin".into(),
        };

        if let Expr::Like(like_expr) = to_df_boolean_expr(&expr) {
            assert!(!like_expr.negated, "Should not be negated");
            assert!(!like_expr.case_insensitive, "Should be case sensitive");

            // Check the expression is the column
            match *like_expr.expr {
                Expr::Column(ref col_expr) => {
                    assert_eq!(col_expr.name(), "p__email");
                }
                other => panic!("Expected column expression, got {:?}", other),
            }

            // Check pattern is 'admin%'
            match *like_expr.pattern {
                Expr::Literal(ref scalar, _) => {
                    let s = format!("{:?}", scalar);
                    assert!(
                        s.contains("admin%"),
                        "Pattern should be 'admin%', got: {}",
                        s
                    );
                }
                other => panic!("Expected literal pattern, got {:?}", other),
            }
        } else {
            panic!("Expected Like expression");
        }
    }

    #[test]
    fn test_boolean_expr_ends_with() {
        let expr = BooleanExpression::EndsWith {
            expression: ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "email".into(),
            }),
            suffix: "@example.com".into(),
        };

        if let Expr::Like(like_expr) = to_df_boolean_expr(&expr) {
            assert!(!like_expr.negated, "Should not be negated");
            assert!(!like_expr.case_insensitive, "Should be case sensitive");

            // Check the expression is the column
            match *like_expr.expr {
                Expr::Column(ref col_expr) => {
                    assert_eq!(col_expr.name(), "p__email");
                }
                other => panic!("Expected column expression, got {:?}", other),
            }

            // Check pattern is '%@example.com'
            match *like_expr.pattern {
                Expr::Literal(ref scalar, _) => {
                    let s = format!("{:?}", scalar);
                    assert!(
                        s.contains("%@example.com"),
                        "Pattern should be '%@example.com', got: {}",
                        s
                    );
                }
                other => panic!("Expected literal pattern, got {:?}", other),
            }
        } else {
            panic!("Expected Like expression");
        }
    }

    #[test]
    fn test_boolean_expr_contains_case_sensitivity() {
        // Test that CONTAINS is case-sensitive (case_insensitive = false)
        let expr = BooleanExpression::Contains {
            expression: ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "name".into(),
            }),
            substring: "Test".into(),
        };

        if let Expr::Like(like_expr) = to_df_boolean_expr(&expr) {
            assert!(
                !like_expr.case_insensitive,
                "CONTAINS should be case-sensitive by default"
            );
        } else {
            panic!("Expected Like expression");
        }
    }

    #[test]
    fn test_boolean_expr_string_operators_with_variable() {
        // Test that string operators work with variable references, not just properties
        let expr = BooleanExpression::Contains {
            expression: ValueExpression::Variable("name".into()),
            substring: "test".into(),
        };

        if let Expr::Like(like_expr) = to_df_boolean_expr(&expr) {
            match *like_expr.expr {
                Expr::Column(ref col_expr) => {
                    assert_eq!(
                        col_expr.name(),
                        "name",
                        "Should reference variable directly"
                    );
                }
                other => panic!("Expected column expression, got {:?}", other),
            }
        } else {
            panic!("Expected Like expression");
        }
    }

    // ========================================================================
    // Unit tests for to_df_value_expr()
    // ========================================================================

    #[test]
    fn test_value_expr_property() {
        let expr = ValueExpression::Property(PropertyRef {
            variable: "person".into(),
            property: "name".into(),
        });

        let df_expr = to_df_value_expr(&expr);
        let s = format!("{:?}", df_expr);
        assert_eq!(
            s,
            "Column(Column { relation: None, name: \"person__name\" })"
        );
    }

    #[test]
    fn test_value_expr_literal_integer() {
        let expr = ValueExpression::Literal(PropertyValue::Integer(42));
        let df_expr = to_df_value_expr(&expr);
        let s = format!("{:?}", df_expr);
        assert!(s.contains("42") || s.contains("Int64(42)"));
    }

    #[test]
    fn test_value_expr_literal_float() {
        let expr = ValueExpression::Literal(PropertyValue::Float(std::f64::consts::PI));
        let df_expr = to_df_value_expr(&expr);
        let s = format!("{:?}", df_expr);
        assert!(s.contains("3.14") || s.contains("Float64"));
    }

    #[test]
    fn test_value_expr_literal_string() {
        let expr = ValueExpression::Literal(PropertyValue::String("hello".into()));
        let df_expr = to_df_value_expr(&expr);
        let s = format!("{:?}", df_expr);
        assert!(s.contains("hello") || s.contains("Utf8"));
    }

    #[test]
    fn test_value_expr_literal_boolean() {
        let expr = ValueExpression::Literal(PropertyValue::Boolean(true));
        let df_expr = to_df_value_expr(&expr);
        let s = format!("{:?}", df_expr);
        assert!(s.contains("true") || s.contains("Boolean"));
    }

    #[test]
    fn test_value_expr_literal_null() {
        let expr = ValueExpression::Literal(PropertyValue::Null);
        let df_expr = to_df_value_expr(&expr);
        let s = format!("{:?}", df_expr);
        // Null literals are translated to Literal with Null value
        assert!(s.contains("Literal"), "Should be a Literal expression");
    }

    #[test]
    fn test_value_expr_arithmetic_add() {
        use crate::ast::ArithmeticOperator;
        let expr = ValueExpression::Arithmetic {
            left: Box::new(ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "age".into(),
            })),
            operator: ArithmeticOperator::Add,
            right: Box::new(ValueExpression::Literal(PropertyValue::Integer(5))),
        };

        let df_expr = to_df_value_expr(&expr);
        let s = format!("{:?}", df_expr);
        // Arithmetic expressions should now return a BinaryExpr with Plus operator
        assert!(s.contains("BinaryExpr"), "Should be a BinaryExpr");
        assert!(s.contains("Plus"), "Should contain Plus operator");
        assert!(
            s.contains("p__age"),
            "Should contain the left operand column"
        );
    }

    #[test]
    fn test_value_expr_arithmetic_operators() {
        use crate::ast::ArithmeticOperator;

        let operators = vec![
            (ArithmeticOperator::Add, "Plus"),
            (ArithmeticOperator::Subtract, "Minus"),
            (ArithmeticOperator::Multiply, "Multiply"),
            (ArithmeticOperator::Divide, "Divide"),
        ];

        for (op, expected_op_str) in operators {
            let expr = ValueExpression::Arithmetic {
                left: Box::new(ValueExpression::Literal(PropertyValue::Integer(10))),
                operator: op,
                right: Box::new(ValueExpression::Literal(PropertyValue::Integer(2))),
            };

            let df_expr = to_df_value_expr(&expr);
            let s = format!("{:?}", df_expr);
            // Should translate to BinaryExpr with the correct operator
            assert!(
                s.contains("BinaryExpr"),
                "Arithmetic should produce BinaryExpr"
            );
            assert!(
                s.contains(expected_op_str),
                "Expected operator {} in expression: {}",
                expected_op_str,
                s
            );
        }
    }

    #[test]
    fn test_value_expr_function_count_star() {
        let expr = ValueExpression::Function {
            name: "COUNT".into(),
            args: vec![ValueExpression::Literal(PropertyValue::String("*".into()))],
        };

        let df_expr = to_df_value_expr(&expr);
        let s = format!("{:?}", df_expr);
        assert!(
            s.contains("count") || s.contains("Count"),
            "Should be COUNT function"
        );
    }

    #[test]
    fn test_value_expr_function_count_property() {
        let expr = ValueExpression::Function {
            name: "COUNT".into(),
            args: vec![ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "id".into(),
            })],
        };

        let df_expr = to_df_value_expr(&expr);
        let s = format!("{:?}", df_expr);
        assert!(
            s.contains("count") || s.contains("Count"),
            "Should be COUNT function"
        );
        assert!(s.contains("p__id"), "Should contain column reference");
    }

    #[test]
    fn test_value_expr_function_sum() {
        let expr = ValueExpression::Function {
            name: "SUM".into(),
            args: vec![ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "amount".into(),
            })],
        };

        let df_expr = to_df_value_expr(&expr);
        let s = format!("{:?}", df_expr);
        assert!(
            s.contains("sum") || s.contains("Sum"),
            "Should be SUM function"
        );
        assert!(s.contains("p__amount"), "Should contain column reference");
    }

    #[test]
    fn test_value_expr_function_avg() {
        let expr = ValueExpression::Function {
            name: "AVG".into(),
            args: vec![ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "amount".into(),
            })],
        };

        let df_expr = to_df_value_expr(&expr);
        let s = format!("{:?}", df_expr);
        assert!(
            s.contains("avg") || s.contains("Avg"),
            "Should be AVG function"
        );
        assert!(s.contains("p__amount"), "Should contain column reference");
    }

    #[test]
    fn test_value_expr_function_min() {
        let expr = ValueExpression::Function {
            name: "MIN".into(),
            args: vec![ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "amount".into(),
            })],
        };

        let df_expr = to_df_value_expr(&expr);
        let s = format!("{:?}", df_expr);
        assert!(
            s.contains("min") || s.contains("Min"),
            "Should be MIN function"
        );
        assert!(s.contains("p__amount"), "Should contain column reference");
    }

    #[test]
    fn test_value_expr_function_max() {
        let expr = ValueExpression::Function {
            name: "MAX".into(),
            args: vec![ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "amount".into(),
            })],
        };

        let df_expr = to_df_value_expr(&expr);
        let s = format!("{:?}", df_expr);
        assert!(
            s.contains("max") || s.contains("Max"),
            "Should be MAX function"
        );
        assert!(s.contains("p__amount"), "Should contain column reference");
    }

    #[test]
    fn test_value_expr_function_tolower() {
        let expr = ValueExpression::Function {
            name: "toLower".into(),
            args: vec![ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "name".into(),
            })],
        };

        let df_expr = to_df_value_expr(&expr);
        let s = format!("{:?}", df_expr);
        // Should be a ScalarFunction with lower
        assert!(
            s.contains("lower") || s.contains("Lower"),
            "Should use lower function, got: {}",
            s
        );
        assert!(s.contains("p__name"), "Should contain column reference");
    }

    #[test]
    fn test_value_expr_function_toupper() {
        let expr = ValueExpression::Function {
            name: "toUpper".into(),
            args: vec![ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "name".into(),
            })],
        };

        let df_expr = to_df_value_expr(&expr);
        let s = format!("{:?}", df_expr);
        // Should be a ScalarFunction with upper
        assert!(
            s.contains("upper") || s.contains("Upper"),
            "Should use upper function, got: {}",
            s
        );
        assert!(s.contains("p__name"), "Should contain column reference");
    }

    #[test]
    fn test_value_expr_function_lower_alias() {
        // Test that 'lower' also works (SQL-style alias)
        let expr = ValueExpression::Function {
            name: "lower".into(),
            args: vec![ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "name".into(),
            })],
        };

        let df_expr = to_df_value_expr(&expr);
        let s = format!("{:?}", df_expr);
        assert!(
            s.contains("lower") || s.contains("Lower"),
            "Should use lower function, got: {}",
            s
        );
    }

    #[test]
    fn test_value_expr_function_upper_alias() {
        // Test that 'upper' also works (SQL-style alias)
        let expr = ValueExpression::Function {
            name: "upper".into(),
            args: vec![ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "name".into(),
            })],
        };

        let df_expr = to_df_value_expr(&expr);
        let s = format!("{:?}", df_expr);
        assert!(
            s.contains("upper") || s.contains("Upper"),
            "Should use upper function, got: {}",
            s
        );
    }

    #[test]
    fn test_tolower_with_contains_produces_valid_like() {
        // This is the bug scenario: toLower(s.name) CONTAINS 'offer'
        // Previously returned lit(0) which caused type coercion error
        let tolower_expr = ValueExpression::Function {
            name: "toLower".into(),
            args: vec![ValueExpression::Property(PropertyRef {
                variable: "s".into(),
                property: "name".into(),
            })],
        };

        let contains_expr = BooleanExpression::Contains {
            expression: tolower_expr,
            substring: "offer".into(),
        };

        let df_expr = to_df_boolean_expr(&contains_expr);
        let s = format!("{:?}", df_expr);

        // Should be a Like expression with lower() on the column, not lit(0)
        assert!(
            s.contains("Like"),
            "Should be a LIKE expression, got: {}",
            s
        );
        assert!(
            s.contains("lower") || s.contains("Lower"),
            "Should contain lower function, got: {}",
            s
        );
        assert!(
            !s.contains("Int32(0)") && !s.contains("Int64(0)") && !s.contains("Utf8(\"\")"),
            "Should NOT contain literal 0 or empty string (placeholder bugs), got: {}",
            s
        );
    }

    // ========================================================================
    // Unit tests for contains_aggregate()
    // ========================================================================

    #[test]
    fn test_contains_aggregate_count() {
        let expr = ValueExpression::Function {
            name: "COUNT".into(),
            args: vec![ValueExpression::Literal(PropertyValue::String("*".into()))],
        };

        assert!(
            contains_aggregate(&expr),
            "COUNT should be detected as aggregate"
        );
    }

    #[test]
    fn test_contains_aggregate_sum() {
        let expr = ValueExpression::Function {
            name: "SUM".into(),
            args: vec![ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "value".into(),
            })],
        };

        assert!(
            contains_aggregate(&expr),
            "SUM should be detected as aggregate"
        );
    }

    #[test]
    fn test_contains_aggregate_min() {
        let expr = ValueExpression::Function {
            name: "MIN".into(),
            args: vec![ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "value".into(),
            })],
        };

        assert!(
            contains_aggregate(&expr),
            "MIN should be detected as aggregate"
        );
    }

    #[test]
    fn test_contains_aggregate_max() {
        let expr = ValueExpression::Function {
            name: "MAX".into(),
            args: vec![ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "value".into(),
            })],
        };

        assert!(
            contains_aggregate(&expr),
            "MAX should be detected as aggregate"
        );
    }

    #[test]
    fn test_contains_aggregate_property() {
        let expr = ValueExpression::Property(PropertyRef {
            variable: "p".into(),
            property: "name".into(),
        });

        assert!(
            !contains_aggregate(&expr),
            "Property should not be aggregate"
        );
    }

    #[test]
    fn test_contains_aggregate_literal() {
        let expr = ValueExpression::Literal(PropertyValue::Integer(42));
        assert!(
            !contains_aggregate(&expr),
            "Literal should not be aggregate"
        );
    }

    #[test]
    fn test_contains_aggregate_arithmetic_with_aggregate() {
        use crate::ast::ArithmeticOperator;
        let expr = ValueExpression::Arithmetic {
            left: Box::new(ValueExpression::Function {
                name: "COUNT".into(),
                args: vec![ValueExpression::Literal(PropertyValue::String("*".into()))],
            }),
            operator: ArithmeticOperator::Multiply,
            right: Box::new(ValueExpression::Literal(PropertyValue::Integer(2))),
        };

        assert!(
            contains_aggregate(&expr),
            "Arithmetic with COUNT should be detected as aggregate"
        );
    }

    #[test]
    fn test_contains_aggregate_arithmetic_without_aggregate() {
        use crate::ast::ArithmeticOperator;
        let expr = ValueExpression::Arithmetic {
            left: Box::new(ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "age".into(),
            })),
            operator: ArithmeticOperator::Add,
            right: Box::new(ValueExpression::Literal(PropertyValue::Integer(5))),
        };

        assert!(
            !contains_aggregate(&expr),
            "Arithmetic without aggregates should not be detected as aggregate"
        );
    }

    #[test]
    fn test_contains_aggregate_nested_function() {
        let expr = ValueExpression::Function {
            name: "UPPER".into(),
            args: vec![ValueExpression::Function {
                name: "COUNT".into(),
                args: vec![ValueExpression::Literal(PropertyValue::String("*".into()))],
            }],
        };

        assert!(
            contains_aggregate(&expr),
            "Nested function with COUNT should be detected as aggregate"
        );
    }

    // ========================================================================
    // Unit tests for to_cypher_column_name()
    // ========================================================================

    #[test]
    fn test_cypher_column_name_property() {
        let expr = ValueExpression::Property(PropertyRef {
            variable: "person".into(),
            property: "name".into(),
        });

        let name = to_cypher_column_name(&expr);
        assert_eq!(name, "person.name", "Should convert to Cypher dot notation");
    }

    #[test]
    fn test_cypher_column_name_function_count_star() {
        let expr = ValueExpression::Function {
            name: "COUNT".into(),
            args: vec![ValueExpression::Literal(PropertyValue::String("*".into()))],
        };

        let name = to_cypher_column_name(&expr);
        // Literal arguments (including "*") are converted to "expr" by the implementation
        // This is expected behavior since we can't distinguish "*" from other literals
        assert_eq!(
            name, "count(expr)",
            "COUNT with literal arg should be 'count(expr)'"
        );
    }

    #[test]
    fn test_cypher_column_name_function_count_property() {
        let expr = ValueExpression::Function {
            name: "COUNT".into(),
            args: vec![ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "id".into(),
            })],
        };

        let name = to_cypher_column_name(&expr);
        assert_eq!(name, "count(p.id)", "Should format as count(p.id)");
    }

    #[test]
    fn test_cypher_column_name_function_sum() {
        let expr = ValueExpression::Function {
            name: "SUM".into(),
            args: vec![ValueExpression::Property(PropertyRef {
                variable: "order".into(),
                property: "amount".into(),
            })],
        };

        let name = to_cypher_column_name(&expr);
        assert_eq!(
            name, "sum(order.amount)",
            "Should format as sum(order.amount)"
        );
    }

    #[test]
    fn test_cypher_column_name_literal() {
        let expr = ValueExpression::Literal(PropertyValue::Integer(42));
        let name = to_cypher_column_name(&expr);
        assert_eq!(name, "expr", "Literals should use generic name");
    }

    #[test]
    fn test_cypher_column_name_arithmetic() {
        use crate::ast::ArithmeticOperator;
        let expr = ValueExpression::Arithmetic {
            left: Box::new(ValueExpression::Property(PropertyRef {
                variable: "p".into(),
                property: "age".into(),
            })),
            operator: ArithmeticOperator::Add,
            right: Box::new(ValueExpression::Literal(PropertyValue::Integer(5))),
        };

        let name = to_cypher_column_name(&expr);
        assert_eq!(name, "expr", "Arithmetic should use generic name");
    }
}
