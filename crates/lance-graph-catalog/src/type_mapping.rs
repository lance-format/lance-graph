// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Type mapping from Unity Catalog types to Arrow data types.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};

use crate::catalog_provider::{CatalogError, CatalogResult, ColumnInfo};

/// Map a Unity Catalog `type_name` string to an Arrow `DataType`.
///
/// Handles the standard UC type names. Case-insensitive.
pub fn uc_type_to_arrow(type_name: &str) -> CatalogResult<DataType> {
    match type_name.to_uppercase().as_str() {
        // Boolean
        "BOOLEAN" => Ok(DataType::Boolean),

        // Integer types
        "BYTE" | "TINYINT" => Ok(DataType::Int8),
        "SHORT" | "SMALLINT" => Ok(DataType::Int16),
        "INT" | "INTEGER" => Ok(DataType::Int32),
        "LONG" | "BIGINT" => Ok(DataType::Int64),

        // Floating-point types
        "FLOAT" | "REAL" => Ok(DataType::Float32),
        "DOUBLE" => Ok(DataType::Float64),

        // Decimal (default precision/scale; future: parse from type_text)
        "DECIMAL" | "NUMERIC" | "DEC" => Ok(DataType::Decimal128(38, 10)),

        // String types
        "STRING" | "VARCHAR" | "CHAR" | "TEXT" => Ok(DataType::Utf8),

        // Binary
        "BINARY" => Ok(DataType::Binary),

        // Date and time
        "DATE" => Ok(DataType::Date32),
        "TIMESTAMP" | "TIMESTAMP_NTZ" => Ok(DataType::Timestamp(
            arrow_schema::TimeUnit::Microsecond,
            None,
        )),

        // Null
        "NULL" | "VOID" => Ok(DataType::Null),

        // Complex types — represented as Utf8 (JSON string) for now.
        // A future iteration could parse type_text for ARRAY<T>, MAP<K,V>, STRUCT<...>.
        "ARRAY" => Ok(DataType::Utf8),
        "MAP" => Ok(DataType::Utf8),
        "STRUCT" => Ok(DataType::Utf8),

        other => Err(CatalogError::TypeMappingError(format!(
            "Unsupported Unity Catalog type: '{}'",
            other
        ))),
    }
}

/// Convert a slice of [`ColumnInfo`] to an Arrow [`Schema`].
///
/// Columns are sorted by `position` to ensure correct field order.
pub fn columns_to_arrow_schema(columns: &[ColumnInfo]) -> CatalogResult<SchemaRef> {
    let mut sorted: Vec<&ColumnInfo> = columns.iter().collect();
    sorted.sort_by_key(|c| c.position);

    let fields: Vec<Field> = sorted
        .iter()
        .map(|col| {
            let data_type = uc_type_to_arrow(&col.type_name)?;
            Ok(Field::new(&col.name, data_type, col.nullable))
        })
        .collect::<CatalogResult<Vec<_>>>()?;

    Ok(Arc::new(Schema::new(fields)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_type_mappings() {
        assert_eq!(uc_type_to_arrow("BOOLEAN").unwrap(), DataType::Boolean);
        assert_eq!(uc_type_to_arrow("INT").unwrap(), DataType::Int32);
        assert_eq!(uc_type_to_arrow("INTEGER").unwrap(), DataType::Int32);
        assert_eq!(uc_type_to_arrow("LONG").unwrap(), DataType::Int64);
        assert_eq!(uc_type_to_arrow("BIGINT").unwrap(), DataType::Int64);
        assert_eq!(uc_type_to_arrow("FLOAT").unwrap(), DataType::Float32);
        assert_eq!(uc_type_to_arrow("DOUBLE").unwrap(), DataType::Float64);
        assert_eq!(uc_type_to_arrow("STRING").unwrap(), DataType::Utf8);
        assert_eq!(uc_type_to_arrow("VARCHAR").unwrap(), DataType::Utf8);
        assert_eq!(uc_type_to_arrow("BINARY").unwrap(), DataType::Binary);
        assert_eq!(uc_type_to_arrow("DATE").unwrap(), DataType::Date32);
        assert_eq!(uc_type_to_arrow("BYTE").unwrap(), DataType::Int8);
        assert_eq!(uc_type_to_arrow("SHORT").unwrap(), DataType::Int16);
    }

    #[test]
    fn test_timestamp_types() {
        assert_eq!(
            uc_type_to_arrow("TIMESTAMP").unwrap(),
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None)
        );
        assert_eq!(
            uc_type_to_arrow("TIMESTAMP_NTZ").unwrap(),
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None)
        );
    }

    #[test]
    fn test_decimal() {
        assert_eq!(
            uc_type_to_arrow("DECIMAL").unwrap(),
            DataType::Decimal128(38, 10)
        );
        assert_eq!(
            uc_type_to_arrow("NUMERIC").unwrap(),
            DataType::Decimal128(38, 10)
        );
    }

    #[test]
    fn test_case_insensitive() {
        assert_eq!(uc_type_to_arrow("int").unwrap(), DataType::Int32);
        assert_eq!(uc_type_to_arrow("String").unwrap(), DataType::Utf8);
        assert_eq!(uc_type_to_arrow("boolean").unwrap(), DataType::Boolean);
    }

    #[test]
    fn test_complex_types_fallback_to_utf8() {
        assert_eq!(uc_type_to_arrow("ARRAY").unwrap(), DataType::Utf8);
        assert_eq!(uc_type_to_arrow("MAP").unwrap(), DataType::Utf8);
        assert_eq!(uc_type_to_arrow("STRUCT").unwrap(), DataType::Utf8);
    }

    #[test]
    fn test_unsupported_type() {
        let err = uc_type_to_arrow("UNKNOWN_TYPE").unwrap_err();
        assert!(err.to_string().contains("UNKNOWN_TYPE"));
    }

    #[test]
    fn test_columns_to_schema() {
        let columns = vec![
            ColumnInfo {
                name: "id".into(),
                type_text: "INT".into(),
                type_name: "INT".into(),
                position: 0,
                nullable: false,
                comment: None,
            },
            ColumnInfo {
                name: "name".into(),
                type_text: "STRING".into(),
                type_name: "STRING".into(),
                position: 1,
                nullable: true,
                comment: None,
            },
            ColumnInfo {
                name: "age".into(),
                type_text: "INT".into(),
                type_name: "INT".into(),
                position: 2,
                nullable: true,
                comment: Some("User age".into()),
            },
        ];
        let schema = columns_to_arrow_schema(&columns).unwrap();
        assert_eq!(schema.fields().len(), 3);
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(*schema.field(0).data_type(), DataType::Int32);
        assert!(!schema.field(0).is_nullable());
        assert_eq!(schema.field(1).name(), "name");
        assert_eq!(*schema.field(1).data_type(), DataType::Utf8);
        assert!(schema.field(1).is_nullable());
    }

    #[test]
    fn test_columns_sorted_by_position() {
        let columns = vec![
            ColumnInfo {
                name: "b".into(),
                type_text: "STRING".into(),
                type_name: "STRING".into(),
                position: 2,
                nullable: true,
                comment: None,
            },
            ColumnInfo {
                name: "a".into(),
                type_text: "INT".into(),
                type_name: "INT".into(),
                position: 0,
                nullable: false,
                comment: None,
            },
            ColumnInfo {
                name: "c".into(),
                type_text: "DOUBLE".into(),
                type_name: "DOUBLE".into(),
                position: 1,
                nullable: true,
                comment: None,
            },
        ];
        let schema = columns_to_arrow_schema(&columns).unwrap();
        assert_eq!(schema.field(0).name(), "a");
        assert_eq!(schema.field(1).name(), "c");
        assert_eq!(schema.field(2).name(), "b");
    }

    #[test]
    fn test_empty_columns() {
        let schema = columns_to_arrow_schema(&[]).unwrap();
        assert_eq!(schema.fields().len(), 0);
    }
}
