// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Spark SQL dialect for the DataFusion unparser.
//!
//! This module provides a Spark SQL dialect built using DataFusion's
//! [`CustomDialectBuilder`].
//!
//! Key Spark SQL differences from standard SQL:
//! - Backtick (`` ` ``) identifier quoting
//! - `EXTRACT(field FROM expr)` for date field extraction
//! - `STRING` type for casting (not `VARCHAR`)
//! - `BIGINT`/`INT` for integer types
//! - `TIMESTAMP` for all timestamp types (no timezone info in cast)
//! - `LENGTH()` instead of `CHARACTER_LENGTH()`
//! - Subqueries in FROM require aliases

use datafusion_sql::sqlparser::ast::{self, Ident, ObjectName, TimezoneInfo};
use datafusion_sql::unparser::dialect::{
    CharacterLengthStyle, CustomDialect, CustomDialectBuilder, DateFieldExtractStyle,
};

/// Build a Spark SQL dialect using DataFusion's `CustomDialectBuilder`.
pub fn build_spark_dialect() -> CustomDialect {
    CustomDialectBuilder::new()
        .with_identifier_quote_style('`')
        .with_supports_nulls_first_in_sort(true)
        .with_use_timestamp_for_date64(true)
        .with_utf8_cast_dtype(ast::DataType::Custom(
            ObjectName::from(vec![Ident::new("STRING")]),
            vec![],
        ))
        .with_large_utf8_cast_dtype(ast::DataType::Custom(
            ObjectName::from(vec![Ident::new("STRING")]),
            vec![],
        ))
        .with_date_field_extract_style(DateFieldExtractStyle::Extract)
        .with_character_length_style(CharacterLengthStyle::Length)
        .with_int64_cast_dtype(ast::DataType::BigInt(None))
        .with_int32_cast_dtype(ast::DataType::Int(None))
        .with_timestamp_cast_dtype(
            ast::DataType::Timestamp(None, TimezoneInfo::None),
            ast::DataType::Timestamp(None, TimezoneInfo::None),
        )
        .with_date32_cast_dtype(ast::DataType::Date)
        .with_supports_column_alias_in_table_alias(true)
        .with_requires_derived_table_alias(true)
        .with_full_qualified_col(false)
        .with_unnest_as_table_factor(false)
        .with_float64_ast_dtype(ast::DataType::Double(ast::ExactNumberInfo::None))
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_sql::unparser::dialect::Dialect;

    #[test]
    fn test_spark_dialect_identifier_quoting() {
        let dialect = build_spark_dialect();
        assert_eq!(dialect.identifier_quote_style("table_name"), Some('`'));
        assert_eq!(dialect.identifier_quote_style("column"), Some('`'));
    }

    #[test]
    fn test_spark_dialect_type_mappings() {
        let dialect = build_spark_dialect();
        assert!(matches!(
            dialect.utf8_cast_dtype(),
            ast::DataType::Custom(..)
        ));
        assert!(matches!(
            dialect.int64_cast_dtype(),
            ast::DataType::BigInt(None)
        ));
        assert!(matches!(
            dialect.int32_cast_dtype(),
            ast::DataType::Int(None)
        ));
        assert!(matches!(dialect.date32_cast_dtype(), ast::DataType::Date));
    }

    #[test]
    fn test_spark_dialect_requires_derived_table_alias() {
        let dialect = build_spark_dialect();
        assert!(dialect.requires_derived_table_alias());
    }

    #[test]
    fn test_spark_dialect_extract_style() {
        let dialect = build_spark_dialect();
        assert!(matches!(
            dialect.date_field_extract_style(),
            DateFieldExtractStyle::Extract
        ));
    }

    #[test]
    fn test_spark_dialect_character_length_style() {
        let dialect = build_spark_dialect();
        assert!(matches!(
            dialect.character_length_style(),
            CharacterLengthStyle::Length
        ));
    }
}
