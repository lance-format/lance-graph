// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! SQL dialect support for the DataFusion unparser.
//!
//! This module provides a [`SqlDialect`] enum for selecting which SQL dialect
//! to use when unparsing DataFusion logical plans to SQL strings, and includes
//! a [`SparkDialect`] implementation for Spark SQL.
//!
//! Key Spark SQL differences from standard SQL:
//! - Backtick (`` ` ``) identifier quoting
//! - `EXTRACT(field FROM expr)` for date field extraction
//! - `STRING` type for casting (not `VARCHAR`)
//! - `BIGINT`/`INT` for integer types
//! - `TIMESTAMP` for all timestamp types (no timezone info in cast)
//! - `LENGTH()` instead of `CHARACTER_LENGTH()`
//! - Subqueries in FROM require aliases

use std::sync::Arc;

use arrow::datatypes::TimeUnit;
use datafusion_common::Result;
use datafusion_expr::Expr;
use datafusion_sql::unparser::dialect::{
    CharacterLengthStyle, DateFieldExtractStyle, DefaultDialect, Dialect, IntervalStyle,
    MySqlDialect, PostgreSqlDialect, SqliteDialect,
};
use datafusion_sql::unparser::Unparser;
use datafusion_sql::sqlparser::ast::{self, Ident, ObjectName, TimezoneInfo};

/// SQL dialect to use when generating SQL from Cypher queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SqlDialect {
    /// Generic SQL (DataFusion default dialect)
    #[default]
    Default,
    /// Spark SQL dialect (backtick quoting, STRING type, EXTRACT, etc.)
    Spark,
    /// PostgreSQL dialect
    PostgreSql,
    /// MySQL dialect
    MySql,
    /// SQLite dialect
    Sqlite,
}

impl SqlDialect {
    /// Create a DataFusion `Unparser` configured for this dialect.
    pub fn unparser(&self) -> DialectUnparser {
        match self {
            SqlDialect::Default => DialectUnparser::Default(DefaultDialect {}),
            SqlDialect::Spark => DialectUnparser::Spark(SparkDialect),
            SqlDialect::PostgreSql => DialectUnparser::PostgreSql(PostgreSqlDialect {}),
            SqlDialect::MySql => DialectUnparser::MySql(MySqlDialect {}),
            SqlDialect::Sqlite => DialectUnparser::Sqlite(SqliteDialect {}),
        }
    }
}

/// Wrapper to hold the concrete dialect type and provide an `Unparser` reference.
pub enum DialectUnparser {
    Default(DefaultDialect),
    Spark(SparkDialect),
    PostgreSql(PostgreSqlDialect),
    MySql(MySqlDialect),
    Sqlite(SqliteDialect),
}

impl DialectUnparser {
    pub fn as_unparser(&self) -> Unparser<'_> {
        match self {
            DialectUnparser::Default(d) => Unparser::new(d),
            DialectUnparser::Spark(d) => Unparser::new(d),
            DialectUnparser::PostgreSql(d) => Unparser::new(d),
            DialectUnparser::MySql(d) => Unparser::new(d),
            DialectUnparser::Sqlite(d) => Unparser::new(d),
        }
    }
}

/// A Spark SQL dialect for unparsing DataFusion logical plans to Spark-compatible SQL.
pub struct SparkDialect;

impl Dialect for SparkDialect {
    fn identifier_quote_style(&self, _identifier: &str) -> Option<char> {
        Some('`')
    }

    fn supports_nulls_first_in_sort(&self) -> bool {
        true
    }

    fn use_timestamp_for_date64(&self) -> bool {
        true
    }

    fn interval_style(&self) -> IntervalStyle {
        IntervalStyle::SQLStandard
    }

    fn float64_ast_dtype(&self) -> ast::DataType {
        ast::DataType::Double(ast::ExactNumberInfo::None)
    }

    fn utf8_cast_dtype(&self) -> ast::DataType {
        ast::DataType::Custom(
            ObjectName::from(vec![Ident::new("STRING")]),
            vec![],
        )
    }

    fn large_utf8_cast_dtype(&self) -> ast::DataType {
        ast::DataType::Custom(
            ObjectName::from(vec![Ident::new("STRING")]),
            vec![],
        )
    }

    fn date_field_extract_style(&self) -> DateFieldExtractStyle {
        DateFieldExtractStyle::Extract
    }

    fn character_length_style(&self) -> CharacterLengthStyle {
        CharacterLengthStyle::Length
    }

    fn int64_cast_dtype(&self) -> ast::DataType {
        ast::DataType::BigInt(None)
    }

    fn int32_cast_dtype(&self) -> ast::DataType {
        ast::DataType::Int(None)
    }

    fn timestamp_cast_dtype(
        &self,
        _time_unit: &TimeUnit,
        _tz: &Option<Arc<str>>,
    ) -> ast::DataType {
        ast::DataType::Timestamp(None, TimezoneInfo::None)
    }

    fn date32_cast_dtype(&self) -> ast::DataType {
        ast::DataType::Date
    }

    fn supports_column_alias_in_table_alias(&self) -> bool {
        true
    }

    fn requires_derived_table_alias(&self) -> bool {
        true
    }

    fn full_qualified_col(&self) -> bool {
        false
    }

    fn unnest_as_table_factor(&self) -> bool {
        false
    }

    fn scalar_function_to_sql_overrides(
        &self,
        _unparser: &Unparser,
        _func_name: &str,
        _args: &[Expr],
    ) -> Result<Option<ast::Expr>> {
        // character_length -> length is handled by CharacterLengthStyle::Length
        // Additional Spark-specific function mappings can be added here as needed
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spark_dialect_identifier_quoting() {
        let dialect = SparkDialect;
        assert_eq!(dialect.identifier_quote_style("table_name"), Some('`'));
        assert_eq!(dialect.identifier_quote_style("column"), Some('`'));
    }

    #[test]
    fn test_spark_dialect_type_mappings() {
        let dialect = SparkDialect;
        assert!(matches!(dialect.utf8_cast_dtype(), ast::DataType::Custom(..)));
        assert!(matches!(dialect.int64_cast_dtype(), ast::DataType::BigInt(None)));
        assert!(matches!(dialect.int32_cast_dtype(), ast::DataType::Int(None)));
        assert!(matches!(dialect.date32_cast_dtype(), ast::DataType::Date));
    }

    #[test]
    fn test_spark_dialect_requires_derived_table_alias() {
        let dialect = SparkDialect;
        assert!(dialect.requires_derived_table_alias());
    }

    #[test]
    fn test_spark_dialect_extract_style() {
        let dialect = SparkDialect;
        assert!(matches!(
            dialect.date_field_extract_style(),
            DateFieldExtractStyle::Extract
        ));
    }

    #[test]
    fn test_spark_dialect_character_length_style() {
        let dialect = SparkDialect;
        assert!(matches!(
            dialect.character_length_style(),
            CharacterLengthStyle::Length
        ));
    }
}
