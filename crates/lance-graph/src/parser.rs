// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cypher query parser
//!
//! This module provides parsing functionality for Cypher queries using nom parser combinators.
//! It supports a subset of Cypher syntax focused on graph pattern matching and property access.

use crate::ast::*;
use crate::error::{GraphError, Result};
use nom::{
    branch::alt,
    bytes::complete::{tag, tag_no_case, take_while1},
    character::complete::{char, digit0, digit1, multispace0, multispace1, one_of},
    combinator::{map, map_res, opt, peek, recognize},
    multi::{many0, separated_list0, separated_list1},
    sequence::{delimited, pair, preceded, tuple},
    IResult,
};
use std::collections::HashMap;

/// Parse a complete Cypher query
pub fn parse_cypher_query(input: &str) -> Result<CypherQuery> {
    let (remaining, query) = cypher_query(input).map_err(|e| GraphError::ParseError {
        message: format!("Failed to parse Cypher query: {}", e),
        position: 0,
        location: snafu::Location::new(file!(), line!(), column!()),
    })?;

    if !remaining.trim().is_empty() {
        return Err(GraphError::ParseError {
            message: format!("Unexpected input after query: {}", remaining),
            position: input.len() - remaining.len(),
            location: snafu::Location::new(file!(), line!(), column!()),
        });
    }

    Ok(query)
}

// Top-level parser for a complete Cypher query
fn cypher_query(input: &str) -> IResult<&str, CypherQuery> {
    let (input, _) = multispace0(input)?;
    let (input, match_clauses) = many0(match_clause)(input)?;
    let (input, pre_with_where) = opt(where_clause)(input)?;

    // Optional WITH clause with optional post-WITH MATCH and WHERE
    let (input, with_result) = opt(with_clause)(input)?;
    // Only try to parse post-WITH clauses if we have a WITH clause
    let (input, post_with_matches, post_with_where) = match with_result {
        Some(_) => {
            let (input, matches) = many0(match_clause)(input)?;
            let (input, where_cl) = opt(where_clause)(input)?;
            (input, matches, where_cl)
        }
        None => (input, vec![], None),
    };

    let (input, return_clause) = return_clause(input)?;
    let (input, order_by) = opt(order_by_clause)(input)?;
    let (input, (skip, limit)) = pagination_clauses(input)?;
    let (input, _) = multispace0(input)?;

    Ok((
        input,
        CypherQuery {
            match_clauses,
            where_clause: pre_with_where,
            with_clause: with_result,
            post_with_match_clauses: post_with_matches,
            post_with_where_clause: post_with_where,
            return_clause,
            limit,
            order_by,
            skip,
        },
    ))
}

// Parse a MATCH clause
fn match_clause(input: &str) -> IResult<&str, MatchClause> {
    let (input, _) = multispace0(input)?;
    let (input, _) = tag_no_case("MATCH")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, patterns) = separated_list0(comma_ws, graph_pattern)(input)?;

    Ok((input, MatchClause { patterns }))
}

// Parse a graph pattern (node or path)
fn graph_pattern(input: &str) -> IResult<&str, GraphPattern> {
    alt((
        map(path_pattern, GraphPattern::Path),
        map(node_pattern, GraphPattern::Node),
    ))(input)
}

// Parse a path pattern (only if there are segments)
fn path_pattern(input: &str) -> IResult<&str, PathPattern> {
    let (input, start_node) = node_pattern(input)?;
    let (input, segments) = many0(path_segment)(input)?;

    // Only succeed if we actually have path segments
    if segments.is_empty() {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Tag,
        )));
    }

    Ok((
        input,
        PathPattern {
            start_node,
            segments,
        },
    ))
}

// Parse a path segment (relationship + node)
fn path_segment(input: &str) -> IResult<&str, PathSegment> {
    let (input, relationship) = relationship_pattern(input)?;
    let (input, end_node) = node_pattern(input)?;

    Ok((
        input,
        PathSegment {
            relationship,
            end_node,
        },
    ))
}

// Parse a node pattern: (variable:Label {prop: value})
fn node_pattern(input: &str) -> IResult<&str, NodePattern> {
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(')(input)?;
    let (input, _) = multispace0(input)?;
    let (input, variable) = opt(identifier)(input)?;
    let (input, labels) = many0(preceded(char(':'), identifier))(input)?;
    let (input, _) = multispace0(input)?;
    let (input, properties) = opt(property_map)(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')')(input)?;

    Ok((
        input,
        NodePattern {
            variable: variable.map(|s| s.to_string()),
            labels: labels.into_iter().map(|s| s.to_string()).collect(),
            properties: properties.unwrap_or_default(),
        },
    ))
}

// Parse a relationship pattern: -[variable:TYPE {prop: value}]->
fn relationship_pattern(input: &str) -> IResult<&str, RelationshipPattern> {
    let (input, _) = multispace0(input)?;

    // Parse direction and bracket content
    let (input, (direction, content)) = alt((
        // Outgoing: -[...]->
        map(
            tuple((
                char('-'),
                delimited(char('['), relationship_content, char(']')),
                tag("->"),
            )),
            |(_, content, _)| (RelationshipDirection::Outgoing, content),
        ),
        // Incoming: <-[...]-
        map(
            tuple((
                tag("<-"),
                delimited(char('['), relationship_content, char(']')),
                char('-'),
            )),
            |(_, content, _)| (RelationshipDirection::Incoming, content),
        ),
        // Undirected: -[...]-
        map(
            tuple((
                char('-'),
                delimited(char('['), relationship_content, char(']')),
                char('-'),
            )),
            |(_, content, _)| (RelationshipDirection::Undirected, content),
        ),
    ))(input)?;

    let (variable, types, properties, length) = content;

    Ok((
        input,
        RelationshipPattern {
            variable: variable.map(|s| s.to_string()),
            types: types.into_iter().map(|s| s.to_string()).collect(),
            direction,
            properties: properties.unwrap_or_default(),
            length,
        },
    ))
}

// Type alias for complex relationship content return type
type RelationshipContentResult<'a> = (
    Option<&'a str>,
    Vec<&'a str>,
    Option<HashMap<String, PropertyValue>>,
    Option<LengthRange>,
);

// Parse relationship content inside brackets
fn relationship_content(input: &str) -> IResult<&str, RelationshipContentResult<'_>> {
    let (input, _) = multispace0(input)?;
    let (input, variable) = opt(identifier)(input)?;
    let (input, types) = many0(preceded(char(':'), identifier))(input)?;
    let (input, _) = multispace0(input)?;
    let (input, length) = opt(length_range)(input)?;
    let (input, _) = multispace0(input)?;
    let (input, properties) = opt(property_map)(input)?;
    let (input, _) = multispace0(input)?;

    Ok((input, (variable, types, properties, length)))
}

// Parse a property map: {key: value, key2: value2}
fn property_map(input: &str) -> IResult<&str, HashMap<String, PropertyValue>> {
    let (input, _) = multispace0(input)?;
    let (input, _) = char('{')(input)?;
    let (input, _) = multispace0(input)?;
    let (input, pairs) = separated_list0(comma_ws, property_pair)(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('}')(input)?;

    Ok((input, pairs.into_iter().collect()))
}

// Parse a property key-value pair
fn property_pair(input: &str) -> IResult<&str, (String, PropertyValue)> {
    let (input, _) = multispace0(input)?;
    let (input, key) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(':')(input)?;
    let (input, _) = multispace0(input)?;
    let (input, value) = property_value(input)?;

    Ok((input, (key.to_string(), value)))
}

// Parse a property value
fn property_value(input: &str) -> IResult<&str, PropertyValue> {
    alt((
        map(string_literal, PropertyValue::String),
        map(float_literal, PropertyValue::Float), // Try float BEFORE integer (more specific)
        map(integer_literal, PropertyValue::Integer),
        map(boolean_literal, PropertyValue::Boolean),
        map(tag("null"), |_| PropertyValue::Null),
        map(parameter, PropertyValue::Parameter),
    ))(input)
}

// Parse a WHERE clause
fn where_clause(input: &str) -> IResult<&str, WhereClause> {
    let (input, _) = multispace0(input)?;
    let (input, _) = tag_no_case("WHERE")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, expression) = boolean_expression(input)?;

    Ok((input, WhereClause { expression }))
}

// Parse a boolean expression with OR precedence
fn boolean_expression(input: &str) -> IResult<&str, BooleanExpression> {
    boolean_or_expression(input)
}

fn boolean_or_expression(input: &str) -> IResult<&str, BooleanExpression> {
    let (input, first) = boolean_and_expression(input)?;
    let (input, rest) = many0(preceded(
        tuple((multispace0, tag_no_case("OR"), multispace1)),
        boolean_and_expression,
    ))(input)?;
    let expr = rest.into_iter().fold(first, |acc, item| {
        BooleanExpression::Or(Box::new(acc), Box::new(item))
    });
    Ok((input, expr))
}

fn boolean_and_expression(input: &str) -> IResult<&str, BooleanExpression> {
    let (input, first) = boolean_not_expression(input)?;
    let (input, rest) = many0(preceded(
        tuple((multispace0, tag_no_case("AND"), multispace1)),
        boolean_not_expression,
    ))(input)?;
    let expr = rest.into_iter().fold(first, |acc, item| {
        BooleanExpression::And(Box::new(acc), Box::new(item))
    });
    Ok((input, expr))
}

fn boolean_not_expression(input: &str) -> IResult<&str, BooleanExpression> {
    let (input, _) = multispace0(input)?;
    alt((
        map(
            preceded(
                tuple((tag_no_case("NOT"), multispace1)),
                boolean_not_expression,
            ),
            |expr| BooleanExpression::Not(Box::new(expr)),
        ),
        boolean_primary_expression,
    ))(input)
}

fn boolean_primary_expression(input: &str) -> IResult<&str, BooleanExpression> {
    let (input, _) = multispace0(input)?;
    alt((
        map(
            delimited(
                tuple((char('('), multispace0)),
                boolean_expression,
                tuple((multispace0, char(')'))),
            ),
            |expr| expr,
        ),
        comparison_expression,
    ))(input)
}

fn comparison_expression(input: &str) -> IResult<&str, BooleanExpression> {
    let (input, _) = multispace0(input)?;
    let (input, left) = value_expression(input)?;
    let (input, _) = multispace0(input)?;
    let left_clone = left.clone();

    if let Ok((input_after_in, (_, _, list))) =
        tuple((tag_no_case("IN"), multispace0, value_expression_list))(input)
    {
        return Ok((
            input_after_in,
            BooleanExpression::In {
                expression: left,
                list,
            },
        ));
    }
    // Match LIKE pattern
    if let Ok((input_after_like, (_, _, pattern))) =
        tuple((tag_no_case("LIKE"), multispace0, string_literal))(input)
    {
        return Ok((
            input_after_like,
            BooleanExpression::Like {
                expression: left,
                pattern,
            },
        ));
    }
    // Match ILIKE pattern (case-insensitive LIKE)
    if let Ok((input_after_ilike, (_, _, pattern))) =
        tuple((tag_no_case("ILIKE"), multispace0, string_literal))(input)
    {
        return Ok((
            input_after_ilike,
            BooleanExpression::ILike {
                expression: left,
                pattern,
            },
        ));
    }
    // Match CONTAINS substring
    if let Ok((input_after_contains, (_, _, substring))) =
        tuple((tag_no_case("CONTAINS"), multispace0, string_literal))(input)
    {
        return Ok((
            input_after_contains,
            BooleanExpression::Contains {
                expression: left,
                substring,
            },
        ));
    }
    // Match STARTS WITH prefix (note: multi-word operator)
    if let Ok((input_after_starts, (_, _, _, _, prefix))) = tuple((
        tag_no_case("STARTS"),
        multispace1,
        tag_no_case("WITH"),
        multispace0,
        string_literal,
    ))(input)
    {
        return Ok((
            input_after_starts,
            BooleanExpression::StartsWith {
                expression: left,
                prefix,
            },
        ));
    }
    // Match ENDS WITH suffix (note: multi-word operator)
    if let Ok((input_after_ends, (_, _, _, _, suffix))) = tuple((
        tag_no_case("ENDS"),
        multispace1,
        tag_no_case("WITH"),
        multispace0,
        string_literal,
    ))(input)
    {
        return Ok((
            input_after_ends,
            BooleanExpression::EndsWith {
                expression: left,
                suffix,
            },
        ));
    }
    // Match is null
    if let Ok((rest, ())) = is_null_comparison(input) {
        return Ok((rest, BooleanExpression::IsNull(left_clone)));
    }
    // Match is not null
    if let Ok((rest, ())) = is_not_null_comparison(input) {
        return Ok((rest, BooleanExpression::IsNotNull(left_clone)));
    }

    let (input, operator) = comparison_operator(input)?;
    let (input, _) = multispace0(input)?;
    let (input, right) = value_expression(input)?;

    Ok((
        input,
        BooleanExpression::Comparison {
            left: left_clone,
            operator,
            right,
        },
    ))
}

// Parse a comparison operator
fn comparison_operator(input: &str) -> IResult<&str, ComparisonOperator> {
    alt((
        map(tag("="), |_| ComparisonOperator::Equal),
        map(tag("<>"), |_| ComparisonOperator::NotEqual),
        map(tag("!="), |_| ComparisonOperator::NotEqual),
        map(tag("<="), |_| ComparisonOperator::LessThanOrEqual),
        map(tag(">="), |_| ComparisonOperator::GreaterThanOrEqual),
        map(tag("<"), |_| ComparisonOperator::LessThan),
        map(tag(">"), |_| ComparisonOperator::GreaterThan),
    ))(input)
}

// Parse a basic value expression (without vector functions to avoid circular dependency)
fn basic_value_expression(input: &str) -> IResult<&str, ValueExpression> {
    alt((
        parse_vector_literal, // Try vector literal first [0.1, 0.2]
        parse_parameter,      // Try $parameter
        function_call,        // Regular function calls
        map(property_value, ValueExpression::Literal), // Try literals BEFORE property references
        map(property_reference, ValueExpression::Property),
        map(identifier, |id| ValueExpression::Variable(id.to_string())),
    ))(input)
}

// Parse a value expression
// Optimization: Use peek to avoid expensive backtracking for non-vector queries
fn value_expression(input: &str) -> IResult<&str, ValueExpression> {
    // Peek at first identifier to dispatch to correct parser
    // This eliminates failed parser attempts for every non-vector expression
    if let Ok((_, first_ident)) = peek(identifier)(input) {
        let ident_lower = first_ident.to_lowercase();

        match ident_lower.as_str() {
            "vector_distance" => return parse_vector_distance(input),
            "vector_similarity" => return parse_vector_similarity(input),
            _ => {} // Not a vector function, continue to basic expressions
        }
    }

    // Fast path for common expressions
    basic_value_expression(input)
}

// Parse distance metric: cosine, l2, dot
fn parse_distance_metric(input: &str) -> IResult<&str, DistanceMetric> {
    alt((
        map(tag_no_case("cosine"), |_| DistanceMetric::Cosine),
        map(tag_no_case("l2"), |_| DistanceMetric::L2),
        map(tag_no_case("dot"), |_| DistanceMetric::Dot),
    ))(input)
}

// Parse vector_distance(expr, expr, metric)
fn parse_vector_distance(input: &str) -> IResult<&str, ValueExpression> {
    let (input, _) = tag_no_case("vector_distance")(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(')(input)?;
    let (input, _) = multispace0(input)?;

    // Parse left expression - use basic_value_expression to avoid circular dependency
    let (input, left) = basic_value_expression(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(',')(input)?;
    let (input, _) = multispace0(input)?;

    // Parse right expression - use basic_value_expression to avoid circular dependency
    let (input, right) = basic_value_expression(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(',')(input)?;
    let (input, _) = multispace0(input)?;

    // Parse metric
    let (input, metric) = parse_distance_metric(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')')(input)?;

    Ok((
        input,
        ValueExpression::VectorDistance {
            left: Box::new(left),
            right: Box::new(right),
            metric,
        },
    ))
}

// Parse vector_similarity(expr, expr, metric)
fn parse_vector_similarity(input: &str) -> IResult<&str, ValueExpression> {
    let (input, _) = tag_no_case("vector_similarity")(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(')(input)?;
    let (input, _) = multispace0(input)?;

    // Parse left expression - use basic_value_expression to avoid circular dependency
    let (input, left) = basic_value_expression(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(',')(input)?;
    let (input, _) = multispace0(input)?;

    // Parse right expression - use basic_value_expression to avoid circular dependency
    let (input, right) = basic_value_expression(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(',')(input)?;
    let (input, _) = multispace0(input)?;

    // Parse metric
    let (input, metric) = parse_distance_metric(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')')(input)?;

    Ok((
        input,
        ValueExpression::VectorSimilarity {
            left: Box::new(left),
            right: Box::new(right),
            metric,
        },
    ))
}

// Parse parameter reference: $name
fn parse_parameter(input: &str) -> IResult<&str, ValueExpression> {
    let (input, _) = char('$')(input)?;
    let (input, name) = identifier(input)?;
    Ok((input, ValueExpression::Parameter(name.to_string())))
}

// Parse a function call: function_name(args)
fn function_call(input: &str) -> IResult<&str, ValueExpression> {
    let (input, name) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(')(input)?;
    let (input, _) = multispace0(input)?;

    // Handle COUNT(*) special case - only allow * for COUNT function
    if let Ok((input_after_star, _)) = char::<_, nom::error::Error<&str>>('*')(input) {
        // Validate that this is COUNT function
        if name.to_lowercase() == "count" {
            let (input, _) = multispace0(input_after_star)?;
            let (input, _) = char(')')(input)?;
            return Ok((
                input,
                ValueExpression::Function {
                    name: name.to_string(),
                    args: vec![ValueExpression::Variable("*".to_string())],
                },
            ));
        } else {
            // Not COUNT - fail parsing to try regular argument parsing
            // This will naturally fail since * is not a valid value_expression
        }
    }

    // Parse regular function arguments
    let (input, args) = separated_list0(
        tuple((multispace0, char(','), multispace0)),
        value_expression,
    )(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')')(input)?;

    Ok((
        input,
        ValueExpression::Function {
            name: name.to_string(),
            args,
        },
    ))
}

fn value_expression_list(input: &str) -> IResult<&str, Vec<ValueExpression>> {
    delimited(
        tuple((char('['), multispace0)),
        separated_list1(
            tuple((multispace0, char(','), multispace0)),
            value_expression,
        ),
        tuple((multispace0, char(']'))),
    )(input)
}

// Parse a float32 literal for vectors
fn float32_literal(input: &str) -> IResult<&str, f32> {
    map_res(
        recognize(tuple((
            opt(char('-')),
            alt((
                // Scientific notation: 1e-3, 2.5e2
                recognize(tuple((
                    digit1,
                    opt(tuple((char('.'), digit0))),
                    one_of("eE"),
                    opt(one_of("+-")),
                    digit1,
                ))),
                // Regular float: 1.23 or integer: 123
                recognize(tuple((digit1, opt(tuple((char('.'), digit0)))))),
            )),
        ))),
        |s: &str| s.parse::<f32>(),
    )(input)
}

// Parse vector literal: [0.1, 0.2, 0.3]
fn parse_vector_literal(input: &str) -> IResult<&str, ValueExpression> {
    let (input, _) = char('[')(input)?;
    let (input, _) = multispace0(input)?;

    let (input, values) = separated_list1(
        tuple((multispace0, char(','), multispace0)),
        float32_literal,
    )(input)?;

    let (input, _) = multispace0(input)?;
    let (input, _) = char(']')(input)?;

    Ok((input, ValueExpression::VectorLiteral(values)))
}

// Parse a property reference: variable.property
fn property_reference(input: &str) -> IResult<&str, PropertyRef> {
    let (input, variable) = identifier(input)?;
    let (input, _) = char('.')(input)?;
    let (input, property) = identifier(input)?;

    Ok((
        input,
        PropertyRef {
            variable: variable.to_string(),
            property: property.to_string(),
        },
    ))
}

// Parse a WITH clause (intermediate projection/aggregation)
fn with_clause(input: &str) -> IResult<&str, WithClause> {
    let (input, _) = multispace0(input)?;
    let (input, _) = tag_no_case("WITH")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, items) = separated_list0(comma_ws, return_item)(input)?;
    let (input, order_by) = opt(order_by_clause)(input)?;
    let (input, limit) = opt(limit_clause)(input)?;

    Ok((
        input,
        WithClause {
            items,
            order_by,
            limit,
        },
    ))
}

// Parse a RETURN clause
fn return_clause(input: &str) -> IResult<&str, ReturnClause> {
    let (input, _) = multispace0(input)?;
    let (input, _) = tag_no_case("RETURN")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, distinct) = opt(tag_no_case("DISTINCT"))(input)?;
    let (input, _) = if distinct.is_some() {
        multispace1(input)?
    } else {
        (input, "")
    };
    let (input, items) = separated_list0(comma_ws, return_item)(input)?;

    Ok((
        input,
        ReturnClause {
            distinct: distinct.is_some(),
            items,
        },
    ))
}

// Parse a return item
fn return_item(input: &str) -> IResult<&str, ReturnItem> {
    let (input, expression) = value_expression(input)?;
    let (input, _) = multispace0(input)?;
    let (input, alias) = opt(preceded(
        tuple((tag_no_case("AS"), multispace1)),
        identifier,
    ))(input)?;

    Ok((
        input,
        ReturnItem {
            expression,
            alias: alias.map(|s| s.to_string()),
        },
    ))
}

// Match IS NULL in WHERE clause
fn is_null_comparison(input: &str) -> IResult<&str, ()> {
    let (input, _) = multispace0(input)?;
    let (input, _) = tag_no_case("IS")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, _) = tag_no_case("NULL")(input)?;
    let (input, _) = multispace0(input)?;

    Ok((input, ()))
}

// Match IS NOT NULL in WHERE clause
fn is_not_null_comparison(input: &str) -> IResult<&str, ()> {
    let (input, _) = multispace0(input)?;
    let (input, _) = tag_no_case("IS")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, _) = tag_no_case("NOT")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, _) = tag_no_case("NULL")(input)?;
    let (input, _) = multispace0(input)?;

    Ok((input, ()))
}

// Parse an ORDER BY clause
fn order_by_clause(input: &str) -> IResult<&str, OrderByClause> {
    let (input, _) = multispace0(input)?;
    let (input, _) = tag_no_case("ORDER")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, _) = tag_no_case("BY")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, items) = separated_list0(comma_ws, order_by_item)(input)?;

    Ok((input, OrderByClause { items }))
}

// Parse an order by item
fn order_by_item(input: &str) -> IResult<&str, OrderByItem> {
    let (input, expression) = value_expression(input)?;
    let (input, _) = multispace0(input)?;
    let (input, direction) = opt(alt((
        map(tag_no_case("ASC"), |_| SortDirection::Ascending),
        map(tag_no_case("DESC"), |_| SortDirection::Descending),
    )))(input)?;

    Ok((
        input,
        OrderByItem {
            expression,
            direction: direction.unwrap_or(SortDirection::Ascending),
        },
    ))
}

// Parse a LIMIT clause
fn limit_clause(input: &str) -> IResult<&str, u64> {
    let (input, _) = multispace0(input)?;
    let (input, _) = tag_no_case("LIMIT")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, limit) = integer_literal(input)?;

    Ok((input, limit as u64))
}

// Parse a SKIP clause
fn skip_clause(input: &str) -> IResult<&str, u64> {
    let (input, _) = multispace0(input)?;
    let (input, _) = tag_no_case("SKIP")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, skip) = integer_literal(input)?;

    Ok((input, skip as u64))
}

// Parse pagination clauses (SKIP and LIMIT)
fn pagination_clauses(input: &str) -> IResult<&str, (Option<u64>, Option<u64>)> {
    let (mut remaining, _) = multispace0(input)?;
    let mut skip: Option<u64> = None;
    let mut limit: Option<u64> = None;

    loop {
        let before = remaining;

        if skip.is_none() {
            if let Ok((i, s)) = skip_clause(remaining) {
                skip = Some(s);
                remaining = i;
                continue;
            }
        }

        if limit.is_none() {
            if let Ok((i, l)) = limit_clause(remaining) {
                limit = Some(l);
                remaining = i;
                continue;
            }
        }

        if before == remaining {
            break;
        }
    }

    Ok((remaining, (skip, limit)))
}

// Helper parsers

// Parse an identifier
fn identifier(input: &str) -> IResult<&str, &str> {
    take_while1(|c: char| c.is_alphanumeric() || c == '_')(input)
}

// Parse a string literal
fn string_literal(input: &str) -> IResult<&str, String> {
    alt((double_quoted_string, single_quoted_string))(input)
}

fn double_quoted_string(input: &str) -> IResult<&str, String> {
    let (input, _) = char('"')(input)?;
    let (input, content) = take_while1(|c| c != '"')(input)?;
    let (input, _) = char('"')(input)?;
    Ok((input, content.to_string()))
}

fn single_quoted_string(input: &str) -> IResult<&str, String> {
    let (input, _) = char('\'')(input)?;
    let (input, content) = take_while1(|c| c != '\'')(input)?;
    let (input, _) = char('\'')(input)?;
    Ok((input, content.to_string()))
}

// Parse an integer literal
fn integer_literal(input: &str) -> IResult<&str, i64> {
    let (input, digits) = recognize(pair(
        opt(char('-')),
        take_while1(|c: char| c.is_ascii_digit()),
    ))(input)?;

    Ok((input, digits.parse().unwrap()))
}

// Parse a float literal
fn float_literal(input: &str) -> IResult<&str, f64> {
    let (input, number) = recognize(tuple((
        opt(char('-')),
        take_while1(|c: char| c.is_ascii_digit()),
        char('.'),
        take_while1(|c: char| c.is_ascii_digit()),
    )))(input)?;

    Ok((input, number.parse().unwrap()))
}

// Parse a boolean literal
fn boolean_literal(input: &str) -> IResult<&str, bool> {
    alt((
        map(tag_no_case("true"), |_| true),
        map(tag_no_case("false"), |_| false),
    ))(input)
}

// Parse a parameter reference
fn parameter(input: &str) -> IResult<&str, String> {
    let (input, _) = char('$')(input)?;
    let (input, name) = identifier(input)?;
    Ok((input, name.to_string()))
}

// Parse comma with optional whitespace
fn comma_ws(input: &str) -> IResult<&str, ()> {
    let (input, _) = multispace0(input)?;
    let (input, _) = char(',')(input)?;
    let (input, _) = multispace0(input)?;
    Ok((input, ()))
}

// Parse variable-length path syntax: *1..2, *..3, *2.., *
fn length_range(input: &str) -> IResult<&str, LengthRange> {
    let (input, _) = char('*')(input)?;
    let (input, _) = multispace0(input)?;

    // Parse different length patterns
    alt((
        // *min..max (e.g., *1..3)
        map(
            tuple((
                nom::character::complete::u32,
                tag(".."),
                nom::character::complete::u32,
            )),
            |(min, _, max)| LengthRange {
                min: Some(min),
                max: Some(max),
            },
        ),
        // *..max (e.g., *..3)
        map(preceded(tag(".."), nom::character::complete::u32), |max| {
            LengthRange {
                min: None,
                max: Some(max),
            }
        }),
        // *min.. (e.g., *2..)
        map(
            tuple((nom::character::complete::u32, tag(".."))),
            |(min, _)| LengthRange {
                min: Some(min),
                max: None,
            },
        ),
        // *min (e.g., *2)
        map(nom::character::complete::u32, |min| LengthRange {
            min: Some(min),
            max: Some(min),
        }),
        // * (unlimited)
        map(multispace0, |_| LengthRange {
            min: None,
            max: None,
        }),
    ))(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BooleanExpression, ComparisonOperator, PropertyValue, ValueExpression};

    #[test]
    fn test_parse_simple_node_query() {
        let query = "MATCH (n:Person) RETURN n.name";
        let result = parse_cypher_query(query).unwrap();

        assert_eq!(result.match_clauses.len(), 1);
        assert_eq!(result.return_clause.items.len(), 1);
    }

    #[test]
    fn test_parse_node_with_properties() {
        let query = r#"MATCH (n:Person {name: "John", age: 30}) RETURN n"#;
        let result = parse_cypher_query(query).unwrap();

        if let GraphPattern::Node(node) = &result.match_clauses[0].patterns[0] {
            assert_eq!(node.labels, vec!["Person"]);
            assert_eq!(node.properties.len(), 2);
        } else {
            panic!("Expected node pattern");
        }
    }

    #[test]
    fn test_parse_simple_relationship_query() {
        let query = "MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a.name, b.name";
        let result = parse_cypher_query(query).unwrap();

        assert_eq!(result.match_clauses.len(), 1);
        assert_eq!(result.return_clause.items.len(), 2);

        if let GraphPattern::Path(path) = &result.match_clauses[0].patterns[0] {
            assert_eq!(path.segments.len(), 1);
            assert_eq!(path.segments[0].relationship.types, vec!["KNOWS"]);
        } else {
            panic!("Expected path pattern");
        }
    }

    #[test]
    fn test_parse_variable_length_path() {
        let query = "MATCH (a:Person)-[:FRIEND_OF*1..2]-(b:Person) RETURN a.name, b.name";
        let result = parse_cypher_query(query).unwrap();

        assert_eq!(result.match_clauses.len(), 1);

        if let GraphPattern::Path(path) = &result.match_clauses[0].patterns[0] {
            assert_eq!(path.segments.len(), 1);
            assert_eq!(path.segments[0].relationship.types, vec!["FRIEND_OF"]);

            let length = path.segments[0].relationship.length.as_ref().unwrap();
            assert_eq!(length.min, Some(1));
            assert_eq!(length.max, Some(2));
        } else {
            panic!("Expected path pattern");
        }
    }

    #[test]
    fn test_parse_query_with_where_clause() {
        let query = "MATCH (n:Person) WHERE n.age > 30 RETURN n.name";
        let result = parse_cypher_query(query).unwrap();

        assert!(result.where_clause.is_some());
    }

    #[test]
    fn test_parse_query_with_single_quoted_literal() {
        let query = "MATCH (n:Person) WHERE n.name = 'Alice' RETURN n.name";
        let result = parse_cypher_query(query).unwrap();

        assert!(result.where_clause.is_some());
    }

    #[test]
    fn test_parse_query_with_and_conditions() {
        let query = "MATCH (src:Entity)-[rel:RELATIONSHIP]->(dst:Entity) WHERE rel.relationship_type = 'WORKS_ON' AND dst.name_lower = 'presto' RETURN src.name, src.entity_id";
        let result = parse_cypher_query(query).unwrap();

        let where_clause = result.where_clause.expect("Expected WHERE clause");
        match where_clause.expression {
            BooleanExpression::And(left, right) => {
                match *left {
                    BooleanExpression::Comparison {
                        left: ValueExpression::Property(ref prop),
                        operator,
                        right: ValueExpression::Literal(PropertyValue::String(ref value)),
                    } => {
                        assert_eq!(prop.variable, "rel");
                        assert_eq!(prop.property, "relationship_type");
                        assert_eq!(operator, ComparisonOperator::Equal);
                        assert_eq!(value, "WORKS_ON");
                    }
                    _ => panic!("Expected comparison for relationship_type filter"),
                }

                match *right {
                    BooleanExpression::Comparison {
                        left: ValueExpression::Property(ref prop),
                        operator,
                        right: ValueExpression::Literal(PropertyValue::String(ref value)),
                    } => {
                        assert_eq!(prop.variable, "dst");
                        assert_eq!(prop.property, "name_lower");
                        assert_eq!(operator, ComparisonOperator::Equal);
                        assert_eq!(value, "presto");
                    }
                    _ => panic!("Expected comparison for destination name filter"),
                }
            }
            other => panic!("Expected AND expression, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_query_with_in_clause() {
        let query = "MATCH (src:Entity)-[rel:RELATIONSHIP]->(dst:Entity) WHERE rel.relationship_type IN ['WORKS_FOR', 'PART_OF'] RETURN src.name";
        let result = parse_cypher_query(query).unwrap();

        let where_clause = result.where_clause.expect("Expected WHERE clause");
        match where_clause.expression {
            BooleanExpression::In { expression, list } => {
                match expression {
                    ValueExpression::Property(prop_ref) => {
                        assert_eq!(prop_ref.variable, "rel");
                        assert_eq!(prop_ref.property, "relationship_type");
                    }
                    _ => panic!("Expected property reference in IN expression"),
                }
                assert_eq!(list.len(), 2);
                match &list[0] {
                    ValueExpression::Literal(PropertyValue::String(val)) => {
                        assert_eq!(val, "WORKS_FOR");
                    }
                    _ => panic!("Expected first list item to be a string literal"),
                }
                match &list[1] {
                    ValueExpression::Literal(PropertyValue::String(val)) => {
                        assert_eq!(val, "PART_OF");
                    }
                    _ => panic!("Expected second list item to be a string literal"),
                }
            }
            other => panic!("Expected IN expression, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_query_with_is_null() {
        let query = "MATCH (n:Person) WHERE n.age IS NULL RETURN n.name";
        let result = parse_cypher_query(query).unwrap();

        let where_clause = result.where_clause.expect("Expected WHERE clause");

        match where_clause.expression {
            BooleanExpression::IsNull(expr) => match expr {
                ValueExpression::Property(prop_ref) => {
                    assert_eq!(prop_ref.variable, "n");
                    assert_eq!(prop_ref.property, "age");
                }
                _ => panic!("Expected property reference in IS NULL expression"),
            },
            other => panic!("Expected IS NULL expression, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_query_with_is_not_null() {
        let query = "MATCH (n:Person) WHERE n.age IS NOT NULL RETURN n.name";
        let result = parse_cypher_query(query).unwrap();

        let where_clause = result.where_clause.expect("Expected WHERE clause");

        match where_clause.expression {
            BooleanExpression::IsNotNull(expr) => match expr {
                ValueExpression::Property(prop_ref) => {
                    assert_eq!(prop_ref.variable, "n");
                    assert_eq!(prop_ref.property, "age");
                }
                _ => panic!("Expected property reference in IS NOT NULL expression"),
            },
            other => panic!("Expected IS NOT NULL expression, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_query_with_limit() {
        let query = "MATCH (n:Person) RETURN n.name LIMIT 10";
        let result = parse_cypher_query(query).unwrap();

        assert_eq!(result.limit, Some(10));
    }

    #[test]
    fn test_parse_query_with_skip() {
        let query = "MATCH (n:Person) RETURN n.name SKIP 5";
        let result = parse_cypher_query(query).unwrap();

        assert_eq!(result.skip, Some(5));
        assert_eq!(result.limit, None);
    }

    #[test]
    fn test_parse_query_with_skip_and_limit() {
        let query = "MATCH (n:Person) RETURN n.name SKIP 5 LIMIT 10";
        let result = parse_cypher_query(query).unwrap();

        assert_eq!(result.skip, Some(5));
        assert_eq!(result.limit, Some(10));
    }

    #[test]
    fn test_parse_query_with_skip_and_order_by() {
        let query = "MATCH (n:Person) RETURN n.name ORDER BY n.age SKIP 5";
        let result = parse_cypher_query(query).unwrap();

        assert_eq!(result.skip, Some(5));
        assert!(result.order_by.is_some());
    }

    #[test]
    fn test_parse_query_with_skip_order_by_and_limit() {
        let query = "MATCH (n:Person) RETURN n.name ORDER BY n.age SKIP 5 LIMIT 10";
        let result = parse_cypher_query(query).unwrap();

        assert_eq!(result.skip, Some(5));
        assert_eq!(result.limit, Some(10));
        assert!(result.order_by.is_some());
    }

    #[test]
    fn test_parse_count_star() {
        let query = "MATCH (n:Person) RETURN count(*) AS total";
        let result = parse_cypher_query(query).unwrap();

        assert_eq!(result.return_clause.items.len(), 1);
        let item = &result.return_clause.items[0];
        assert_eq!(item.alias, Some("total".to_string()));

        match &item.expression {
            ValueExpression::Function { name, args } => {
                assert_eq!(name, "count");
                assert_eq!(args.len(), 1);
                match &args[0] {
                    ValueExpression::Variable(v) => assert_eq!(v, "*"),
                    _ => panic!("Expected Variable(*) in count(*)"),
                }
            }
            _ => panic!("Expected Function expression"),
        }
    }

    #[test]
    fn test_parse_count_property() {
        let query = "MATCH (n:Person) RETURN count(n.age)";
        let result = parse_cypher_query(query).unwrap();

        assert_eq!(result.return_clause.items.len(), 1);
        let item = &result.return_clause.items[0];

        match &item.expression {
            ValueExpression::Function { name, args } => {
                assert_eq!(name, "count");
                assert_eq!(args.len(), 1);
                match &args[0] {
                    ValueExpression::Property(prop) => {
                        assert_eq!(prop.variable, "n");
                        assert_eq!(prop.property, "age");
                    }
                    _ => panic!("Expected Property in count(n.age)"),
                }
            }
            _ => panic!("Expected Function expression"),
        }
    }

    #[test]
    fn test_parse_non_count_function_rejects_star() {
        // FOO(*) should fail to parse since * is only allowed for COUNT
        let query = "MATCH (n:Person) RETURN foo(*)";
        let result = parse_cypher_query(query);
        assert!(result.is_err(), "foo(*) should not parse successfully");
    }

    #[test]
    fn test_parse_count_with_multiple_args() {
        // COUNT with multiple arguments parses successfully
        // but will be rejected during semantic validation
        let query = "MATCH (n:Person) RETURN count(n.age, n.name)";
        let result = parse_cypher_query(query);
        assert!(
            result.is_ok(),
            "Parser should accept multiple args (validation happens in semantic phase)"
        );

        // Verify the AST structure
        let ast = result.unwrap();
        match &ast.return_clause.items[0].expression {
            ValueExpression::Function { name, args } => {
                assert_eq!(name, "count");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected Function expression"),
        }
    }

    #[test]
    fn test_parse_like_pattern() {
        let query = "MATCH (n:Person) WHERE n.name LIKE 'A%' RETURN n.name";
        let result = parse_cypher_query(query);
        assert!(result.is_ok(), "LIKE pattern should parse successfully");

        let ast = result.unwrap();
        let where_clause = ast.where_clause.expect("Expected WHERE clause");

        match where_clause.expression {
            BooleanExpression::Like {
                expression,
                pattern,
            } => {
                match expression {
                    ValueExpression::Property(prop) => {
                        assert_eq!(prop.variable, "n");
                        assert_eq!(prop.property, "name");
                    }
                    _ => panic!("Expected property expression"),
                }
                assert_eq!(pattern, "A%");
            }
            _ => panic!("Expected LIKE expression"),
        }
    }

    #[test]
    fn test_parse_like_with_double_quotes() {
        let query = r#"MATCH (n:Person) WHERE n.email LIKE "%@example.com" RETURN n.email"#;
        let result = parse_cypher_query(query);
        assert!(result.is_ok(), "LIKE with double quotes should parse");

        let ast = result.unwrap();
        let where_clause = ast.where_clause.expect("Expected WHERE clause");

        match where_clause.expression {
            BooleanExpression::Like { pattern, .. } => {
                assert_eq!(pattern, "%@example.com");
            }
            _ => panic!("Expected LIKE expression"),
        }
    }

    #[test]
    fn test_parse_like_in_complex_where() {
        let query = "MATCH (n:Person) WHERE n.age > 20 AND n.name LIKE 'J%' RETURN n.name";
        let result = parse_cypher_query(query);
        assert!(result.is_ok(), "LIKE in complex WHERE should parse");

        let ast = result.unwrap();
        let where_clause = ast.where_clause.expect("Expected WHERE clause");

        match where_clause.expression {
            BooleanExpression::And(left, right) => {
                // Left should be age > 20
                match *left {
                    BooleanExpression::Comparison { .. } => {}
                    _ => panic!("Expected comparison on left"),
                }
                // Right should be LIKE
                match *right {
                    BooleanExpression::Like { pattern, .. } => {
                        assert_eq!(pattern, "J%");
                    }
                    _ => panic!("Expected LIKE expression on right"),
                }
            }
            _ => panic!("Expected AND expression"),
        }
    }

    #[test]
    fn test_parse_contains() {
        let query = "MATCH (n:Person) WHERE n.name CONTAINS 'Jo' RETURN n.name";
        let result = parse_cypher_query(query);
        assert!(result.is_ok());

        let query = result.unwrap();
        assert!(query.where_clause.is_some());

        match &query.where_clause.unwrap().expression {
            BooleanExpression::Contains {
                expression,
                substring,
            } => {
                assert_eq!(substring, "Jo");
                match expression {
                    ValueExpression::Property(prop) => {
                        assert_eq!(prop.variable, "n");
                        assert_eq!(prop.property, "name");
                    }
                    _ => panic!("Expected property reference"),
                }
            }
            _ => panic!("Expected CONTAINS expression"),
        }
    }

    #[test]
    fn test_parse_starts_with() {
        let query = "MATCH (n:Person) WHERE n.name STARTS WITH 'Alice' RETURN n.name";
        let result = parse_cypher_query(query);
        assert!(result.is_ok());

        let query = result.unwrap();
        assert!(query.where_clause.is_some());

        match &query.where_clause.unwrap().expression {
            BooleanExpression::StartsWith { expression, prefix } => {
                assert_eq!(prefix, "Alice");
                match expression {
                    ValueExpression::Property(prop) => {
                        assert_eq!(prop.variable, "n");
                        assert_eq!(prop.property, "name");
                    }
                    _ => panic!("Expected property reference"),
                }
            }
            _ => panic!("Expected STARTS WITH expression"),
        }
    }

    #[test]
    fn test_parse_ends_with() {
        let query = "MATCH (n:Person) WHERE n.email ENDS WITH '@example.com' RETURN n.email";
        let result = parse_cypher_query(query);
        assert!(result.is_ok());

        let query = result.unwrap();
        assert!(query.where_clause.is_some());

        match &query.where_clause.unwrap().expression {
            BooleanExpression::EndsWith { expression, suffix } => {
                assert_eq!(suffix, "@example.com");
                match expression {
                    ValueExpression::Property(prop) => {
                        assert_eq!(prop.variable, "n");
                        assert_eq!(prop.property, "email");
                    }
                    _ => panic!("Expected property reference"),
                }
            }
            _ => panic!("Expected ENDS WITH expression"),
        }
    }

    #[test]
    fn test_parse_contains_case_insensitive_keyword() {
        let query = "MATCH (n:Person) WHERE n.name contains 'test' RETURN n.name";
        let result = parse_cypher_query(query);
        assert!(result.is_ok());

        match &result.unwrap().where_clause.unwrap().expression {
            BooleanExpression::Contains { substring, .. } => {
                assert_eq!(substring, "test");
            }
            _ => panic!("Expected CONTAINS expression"),
        }
    }

    #[test]
    fn test_parse_string_operators_in_complex_where() {
        let query =
            "MATCH (n:Person) WHERE n.name CONTAINS 'Jo' AND n.email ENDS WITH '.com' RETURN n";
        let result = parse_cypher_query(query);
        assert!(result.is_ok());

        match &result.unwrap().where_clause.unwrap().expression {
            BooleanExpression::And(left, right) => {
                // Left should be CONTAINS
                match **left {
                    BooleanExpression::Contains { ref substring, .. } => {
                        assert_eq!(substring, "Jo");
                    }
                    _ => panic!("Expected CONTAINS expression on left"),
                }
                // Right should be ENDS WITH
                match **right {
                    BooleanExpression::EndsWith { ref suffix, .. } => {
                        assert_eq!(suffix, ".com");
                    }
                    _ => panic!("Expected ENDS WITH expression on right"),
                }
            }
            _ => panic!("Expected AND expression"),
        }
    }

    #[test]
    fn test_parse_ilike_pattern() {
        let query = "MATCH (n:Person) WHERE n.name ILIKE 'alice%' RETURN n.name";
        let result = parse_cypher_query(query);
        assert!(result.is_ok(), "ILIKE pattern should parse successfully");

        let ast = result.unwrap();
        let where_clause = ast.where_clause.expect("Expected WHERE clause");

        match where_clause.expression {
            BooleanExpression::ILike {
                expression,
                pattern,
            } => {
                match expression {
                    ValueExpression::Property(prop) => {
                        assert_eq!(prop.variable, "n");
                        assert_eq!(prop.property, "name");
                    }
                    _ => panic!("Expected property expression"),
                }
                assert_eq!(pattern, "alice%");
            }
            _ => panic!("Expected ILIKE expression"),
        }
    }

    #[test]
    fn test_parse_like_and_ilike_together() {
        let query =
            "MATCH (n:Person) WHERE n.name LIKE 'Alice%' OR n.name ILIKE 'bob%' RETURN n.name";
        let result = parse_cypher_query(query);
        assert!(result.is_ok(), "LIKE and ILIKE together should parse");

        let ast = result.unwrap();
        let where_clause = ast.where_clause.expect("Expected WHERE clause");

        match where_clause.expression {
            BooleanExpression::Or(left, right) => {
                // Left should be LIKE (case-sensitive)
                match *left {
                    BooleanExpression::Like { pattern, .. } => {
                        assert_eq!(pattern, "Alice%");
                    }
                    _ => panic!("Expected LIKE expression on left"),
                }
                // Right should be ILIKE (case-insensitive)
                match *right {
                    BooleanExpression::ILike { pattern, .. } => {
                        assert_eq!(pattern, "bob%");
                    }
                    _ => panic!("Expected ILIKE expression on right"),
                }
            }
            _ => panic!("Expected OR expression"),
        }
    }

    #[test]
    fn test_parse_vector_distance() {
        let query = "MATCH (p:Person) WHERE vector_distance(p.embedding, $query_vec, cosine) < 0.5 RETURN p.name";
        let result = parse_cypher_query(query);
        assert!(result.is_ok(), "vector_distance should parse successfully");

        let ast = result.unwrap();
        let where_clause = ast.where_clause.expect("Expected WHERE clause");

        // Verify it's a comparison with vector_distance
        match where_clause.expression {
            BooleanExpression::Comparison { left, operator, .. } => {
                match left {
                    ValueExpression::VectorDistance {
                        left,
                        right,
                        metric,
                    } => {
                        assert_eq!(metric, DistanceMetric::Cosine);
                        // Verify left is property reference
                        assert!(matches!(*left, ValueExpression::Property(_)));
                        // Verify right is parameter
                        assert!(matches!(*right, ValueExpression::Parameter(_)));
                    }
                    _ => panic!("Expected VectorDistance"),
                }
                assert_eq!(operator, ComparisonOperator::LessThan);
            }
            _ => panic!("Expected comparison"),
        }
    }

    #[test]
    fn test_parse_vector_similarity() {
        let query =
            "MATCH (p:Person) WHERE vector_similarity(p.embedding, $vec, l2) > 0.8 RETURN p";
        let result = parse_cypher_query(query);
        assert!(
            result.is_ok(),
            "vector_similarity should parse successfully"
        );

        let ast = result.unwrap();
        let where_clause = ast.where_clause.expect("Expected WHERE clause");

        match where_clause.expression {
            BooleanExpression::Comparison { left, operator, .. } => {
                match left {
                    ValueExpression::VectorSimilarity { metric, .. } => {
                        assert_eq!(metric, DistanceMetric::L2);
                    }
                    _ => panic!("Expected VectorSimilarity"),
                }
                assert_eq!(operator, ComparisonOperator::GreaterThan);
            }
            _ => panic!("Expected comparison"),
        }
    }

    #[test]
    fn test_parse_parameter() {
        let query = "MATCH (p:Person) WHERE p.age = $min_age RETURN p";
        let result = parse_cypher_query(query);
        assert!(result.is_ok(), "Parameter should parse successfully");

        let ast = result.unwrap();
        let where_clause = ast.where_clause.expect("Expected WHERE clause");

        match where_clause.expression {
            BooleanExpression::Comparison { right, .. } => match right {
                ValueExpression::Parameter(name) => {
                    assert_eq!(name, "min_age");
                }
                _ => panic!("Expected Parameter"),
            },
            _ => panic!("Expected comparison"),
        }
    }

    #[test]
    fn test_vector_distance_metrics() {
        for metric in &["cosine", "l2", "dot"] {
            let query = format!(
                "MATCH (p:Person) RETURN vector_distance(p.emb, $v, {}) AS dist",
                metric
            );
            let result = parse_cypher_query(&query);
            assert!(result.is_ok(), "Failed to parse metric: {}", metric);

            let ast = result.unwrap();
            let return_item = &ast.return_clause.items[0];

            match &return_item.expression {
                ValueExpression::VectorDistance {
                    metric: parsed_metric,
                    ..
                } => {
                    let expected = match *metric {
                        "cosine" => DistanceMetric::Cosine,
                        "l2" => DistanceMetric::L2,
                        "dot" => DistanceMetric::Dot,
                        _ => panic!("Unexpected metric"),
                    };
                    assert_eq!(*parsed_metric, expected);
                }
                _ => panic!("Expected VectorDistance"),
            }
        }
    }

    #[test]
    fn test_vector_search_in_order_by() {
        let query = "MATCH (p:Person) RETURN p.name ORDER BY vector_distance(p.embedding, $query_vec, cosine) ASC LIMIT 10";
        let result = parse_cypher_query(query);
        assert!(result.is_ok(), "vector_distance in ORDER BY should parse");

        let ast = result.unwrap();
        let order_by = ast.order_by.expect("Expected ORDER BY clause");

        assert_eq!(order_by.items.len(), 1);
        match &order_by.items[0].expression {
            ValueExpression::VectorDistance { .. } => {
                // Success
            }
            _ => panic!("Expected VectorDistance in ORDER BY"),
        }
    }

    #[test]
    fn test_hybrid_query_with_vector_and_property_filters() {
        let query = "MATCH (p:Person) WHERE p.age > 25 AND vector_similarity(p.embedding, $query_vec, cosine) > 0.7 RETURN p.name";
        let result = parse_cypher_query(query);
        assert!(result.is_ok(), "Hybrid query should parse");

        let ast = result.unwrap();
        let where_clause = ast.where_clause.expect("Expected WHERE clause");

        // Should be an AND expression
        match where_clause.expression {
            BooleanExpression::And(left, right) => {
                // Left should be age > 25
                match *left {
                    BooleanExpression::Comparison { .. } => {}
                    _ => panic!("Expected comparison on left"),
                }
                // Right should be vector_similarity > 0.7
                match *right {
                    BooleanExpression::Comparison { left, .. } => match left {
                        ValueExpression::VectorSimilarity { .. } => {}
                        _ => panic!("Expected VectorSimilarity"),
                    },
                    _ => panic!("Expected comparison on right"),
                }
            }
            _ => panic!("Expected AND expression"),
        }
    }

    #[test]
    fn test_parse_vector_literal() {
        let result = parse_vector_literal("[0.1, 0.2, 0.3]");
        assert!(result.is_ok());
        let (_, expr) = result.unwrap();
        match expr {
            ValueExpression::VectorLiteral(vec) => {
                assert_eq!(vec.len(), 3);
                assert_eq!(vec[0], 0.1);
                assert_eq!(vec[1], 0.2);
                assert_eq!(vec[2], 0.3);
            }
            _ => panic!("Expected VectorLiteral"),
        }
    }

    #[test]
    fn test_parse_vector_literal_with_negative_values() {
        let result = parse_vector_literal("[-0.1, 0.2, -0.3]");
        assert!(result.is_ok());
        let (_, expr) = result.unwrap();
        match expr {
            ValueExpression::VectorLiteral(vec) => {
                assert_eq!(vec.len(), 3);
                assert_eq!(vec[0], -0.1);
                assert_eq!(vec[2], -0.3);
            }
            _ => panic!("Expected VectorLiteral"),
        }
    }

    #[test]
    fn test_parse_vector_literal_scientific_notation() {
        let result = parse_vector_literal("[1e-3, 2.5e2, -3e-1]");
        assert!(result.is_ok());
        let (_, expr) = result.unwrap();
        match expr {
            ValueExpression::VectorLiteral(vec) => {
                assert_eq!(vec.len(), 3);
                assert!((vec[0] - 0.001).abs() < 1e-6);
                assert!((vec[1] - 250.0).abs() < 1e-6);
                assert!((vec[2] - (-0.3)).abs() < 1e-6);
            }
            _ => panic!("Expected VectorLiteral"),
        }
    }

    #[test]
    fn test_vector_distance_with_literal() {
        let query =
            "MATCH (p:Person) WHERE vector_distance(p.embedding, [0.1, 0.2], l2) < 0.5 RETURN p";
        let result = parse_cypher_query(query);
        assert!(result.is_ok());

        let ast = result.unwrap();
        let where_clause = ast.where_clause.expect("Expected WHERE clause");

        match where_clause.expression {
            BooleanExpression::Comparison { left, operator, .. } => {
                match left {
                    ValueExpression::VectorDistance {
                        left,
                        right,
                        metric,
                    } => {
                        // Left should be property reference
                        assert!(matches!(*left, ValueExpression::Property(_)));
                        // Right should be vector literal
                        match *right {
                            ValueExpression::VectorLiteral(vec) => {
                                assert_eq!(vec.len(), 2);
                                assert_eq!(vec[0], 0.1);
                                assert_eq!(vec[1], 0.2);
                            }
                            _ => panic!("Expected VectorLiteral"),
                        }
                        assert_eq!(metric, DistanceMetric::L2);
                    }
                    _ => panic!("Expected VectorDistance"),
                }
                assert_eq!(operator, ComparisonOperator::LessThan);
            }
            _ => panic!("Expected comparison"),
        }
    }
}
