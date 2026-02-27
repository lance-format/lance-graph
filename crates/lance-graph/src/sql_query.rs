// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Direct SQL query interface for Lance datasets
//!
//! This module provides a way to execute standard SQL queries directly against
//! in-memory datasets (as RecordBatches) or a pre-configured DataFusion SessionContext,
//! without requiring a GraphConfig or Cypher parsing.

use crate::error::{GraphError, Result};
use crate::query::{normalize_record_batch, normalize_schema};
use arrow_array::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;
use std::collections::HashMap;
use std::sync::Arc;

/// A SQL query that can be executed against in-memory datasets or a DataFusion SessionContext.
///
/// Unlike `CypherQuery`, this does not require a `GraphConfig` — users write standard SQL
/// with explicit JOINs against their node/relationship tables.
///
/// # Example
///
/// ```no_run
/// use lance_graph::SqlQuery;
/// use arrow_array::RecordBatch;
/// use std::collections::HashMap;
///
/// # async fn example() -> lance_graph::Result<()> {
/// let mut datasets: HashMap<String, RecordBatch> = HashMap::new();
/// // datasets.insert("person".to_string(), person_batch);
///
/// let query = SqlQuery::new("SELECT name, age FROM person WHERE age > 30");
/// // let result = query.execute(datasets).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct SqlQuery {
    sql: String,
}

impl SqlQuery {
    /// Create a new SQL query from a SQL string.
    ///
    /// No parsing is done at construction time — the SQL is validated when executed.
    pub fn new(sql: &str) -> Self {
        Self {
            sql: sql.to_string(),
        }
    }

    /// Get the SQL query text.
    pub fn sql(&self) -> &str {
        &self.sql
    }

    /// Execute the SQL query against in-memory datasets.
    ///
    /// Each entry in `datasets` is registered as a table in a fresh DataFusion
    /// SessionContext. Table names are lowercased for consistency.
    ///
    /// # Arguments
    /// * `datasets` - HashMap of table name to RecordBatch
    ///
    /// # Returns
    /// A single `RecordBatch` containing all result rows.
    pub async fn execute(&self, datasets: HashMap<String, RecordBatch>) -> Result<RecordBatch> {
        let ctx = self.build_context(datasets)?;
        self.execute_with_context(ctx).await
    }

    /// Execute the SQL query against a pre-configured DataFusion SessionContext.
    ///
    /// Use this when tables are already registered (e.g., CSV/Parquet files,
    /// external data sources, or a context shared across queries).
    pub async fn execute_with_context(&self, ctx: SessionContext) -> Result<RecordBatch> {
        let df = ctx
            .sql(&self.sql)
            .await
            .map_err(|e| GraphError::PlanError {
                message: format!("SQL execution error: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        let batches = df.collect().await.map_err(|e| GraphError::PlanError {
            message: format!("Failed to collect SQL results: {}", e),
            location: snafu::Location::new(file!(), line!(), column!()),
        })?;

        if batches.is_empty() {
            // Return an empty batch with the schema from the logical plan
            let schema = df_schema_from_ctx(&ctx, &self.sql).await?;
            return Ok(RecordBatch::new_empty(schema));
        }

        let schema = batches[0].schema();
        arrow::compute::concat_batches(&schema, &batches).map_err(|e| GraphError::PlanError {
            message: format!("Failed to concatenate result batches: {}", e),
            location: snafu::Location::new(file!(), line!(), column!()),
        })
    }

    /// Return the DataFusion execution plan as a formatted string.
    ///
    /// Useful for debugging and understanding how the query will be executed.
    pub async fn explain(&self, datasets: HashMap<String, RecordBatch>) -> Result<String> {
        let ctx = self.build_context(datasets)?;

        let df = ctx
            .sql(&self.sql)
            .await
            .map_err(|e| GraphError::PlanError {
                message: format!("SQL explain error: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        let logical_plan = df.logical_plan();

        let physical_plan = ctx
            .state()
            .create_physical_plan(logical_plan)
            .await
            .map_err(|e| GraphError::PlanError {
                message: format!("Failed to create physical plan: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        let physical_plan_str = datafusion::physical_plan::displayable(physical_plan.as_ref())
            .indent(true)
            .to_string();

        Ok(format!(
            "== Logical Plan ==\n{}\n\n== Physical Plan ==\n{}",
            logical_plan.display_indent(),
            physical_plan_str,
        ))
    }

    /// Build a DataFusion SessionContext from in-memory datasets.
    fn build_context(&self, datasets: HashMap<String, RecordBatch>) -> Result<SessionContext> {
        let ctx = SessionContext::new();

        for (name, batch) in datasets {
            let normalized_batch = normalize_record_batch(&batch)?;
            let schema = normalized_batch.schema();
            let mem_table = Arc::new(
                MemTable::try_new(schema, vec![vec![normalized_batch]]).map_err(|e| {
                    GraphError::PlanError {
                        message: format!("Failed to create MemTable for {}: {}", name, e),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    }
                })?,
            );

            let normalized_name = name.to_lowercase();
            ctx.register_table(&normalized_name, mem_table)
                .map_err(|e| GraphError::PlanError {
                    message: format!("Failed to register table {}: {}", name, e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
        }

        Ok(ctx)
    }
}

/// Helper to get the output schema from a SQL query without executing it.
async fn df_schema_from_ctx(ctx: &SessionContext, sql: &str) -> Result<Arc<arrow_schema::Schema>> {
    let df = ctx.sql(sql).await.map_err(|e| GraphError::PlanError {
        message: format!("Failed to plan SQL for schema: {}", e),
        location: snafu::Location::new(file!(), line!(), column!()),
    })?;
    let arrow_schema = Arc::new(arrow_schema::Schema::from(df.schema()));
    normalize_schema(arrow_schema)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float64Array, Int64Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    fn person_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::Int64, false),
            Field::new("city", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
                Arc::new(StringArray::from(vec!["Alice", "Bob", "Carol", "David"])),
                Arc::new(Int64Array::from(vec![28, 34, 29, 42])),
                Arc::new(StringArray::from(vec![
                    "New York",
                    "San Francisco",
                    "New York",
                    "Chicago",
                ])),
            ],
        )
        .unwrap()
    }

    fn datasets_with(name: &str, batch: RecordBatch) -> HashMap<String, RecordBatch> {
        let mut datasets = HashMap::new();
        datasets.insert(name.to_string(), batch);
        datasets
    }

    #[tokio::test]
    async fn test_basic_select() {
        let query = SqlQuery::new("SELECT name, age FROM person WHERE age > 30 ORDER BY age");
        let result = query
            .execute(datasets_with("person", person_batch()))
            .await
            .unwrap();

        let names: Vec<&str> = result
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|v| v.unwrap())
            .collect();
        assert_eq!(names, vec!["Bob", "David"]);
    }

    #[tokio::test]
    async fn test_select_star() {
        let query = SqlQuery::new("SELECT * FROM person");
        let result = query
            .execute(datasets_with("person", person_batch()))
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 4);
        assert_eq!(result.num_columns(), 4);
    }

    #[tokio::test]
    async fn test_limit() {
        let query = SqlQuery::new("SELECT name FROM person ORDER BY name LIMIT 2");
        let result = query
            .execute(datasets_with("person", person_batch()))
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 2);
    }

    #[tokio::test]
    async fn test_aggregation() {
        let query = SqlQuery::new(
            "SELECT COUNT(*) as cnt, AVG(age) as avg_age, SUM(age) as total_age FROM person",
        );
        let result = query
            .execute(datasets_with("person", person_batch()))
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 1);

        let cnt = result
            .column_by_name("cnt")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);
        assert_eq!(cnt, 4);

        let avg_age = result
            .column_by_name("avg_age")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0);
        assert!((avg_age - 33.25).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_group_by() {
        let query = SqlQuery::new(
            "SELECT city, COUNT(*) as cnt FROM person GROUP BY city ORDER BY cnt DESC",
        );
        let result = query
            .execute(datasets_with("person", person_batch()))
            .await
            .unwrap();

        let cities: Vec<&str> = result
            .column_by_name("city")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|v| v.unwrap())
            .collect();
        // New York has 2, others have 1
        assert_eq!(cities[0], "New York");
    }

    #[tokio::test]
    async fn test_invalid_sql() {
        let query = SqlQuery::new("INVALID SQL STATEMENT");
        let result = query.execute(datasets_with("person", person_batch())).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_explain() {
        let query = SqlQuery::new("SELECT name FROM person WHERE age > 30");
        let plan = query
            .explain(datasets_with("person", person_batch()))
            .await
            .unwrap();
        assert!(plan.contains("Logical Plan"));
        assert!(plan.contains("Physical Plan"));
    }

    #[tokio::test]
    async fn test_execute_with_context() {
        // Build context manually and execute against it
        let ctx = SessionContext::new();
        let batch = person_batch();
        let schema = batch.schema();
        let mem_table = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
        ctx.register_table("people", mem_table).unwrap();

        let query = SqlQuery::new("SELECT name FROM people ORDER BY name LIMIT 1");
        let result = query.execute_with_context(ctx).await.unwrap();

        let names: Vec<&str> = result
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|v| v.unwrap())
            .collect();
        assert_eq!(names, vec!["Alice"]);
    }

    #[tokio::test]
    async fn test_sql_text_accessor() {
        let query = SqlQuery::new("SELECT 1");
        assert_eq!(query.sql(), "SELECT 1");
    }
}
