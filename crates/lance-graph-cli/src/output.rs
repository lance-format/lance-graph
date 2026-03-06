// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Output formatting for query results.

use anyhow::Result;
use arrow::util::pretty::pretty_format_batches;
use arrow_array::RecordBatch;
use arrow_cast::display::{ArrayFormatter, FormatOptions};
use std::io::{self, IsTerminal, Write};

/// Output format for query results.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum OutputFormat {
    /// Full JSON object with metadata envelope.
    Json,
    /// One JSON object per row (streaming-friendly, best for agents).
    Jsonl,
    /// Human-readable ASCII table.
    Table,
    /// Comma-separated values with header.
    Csv,
}

impl OutputFormat {
    /// Pick a sensible default: table for TTY, jsonl when piped.
    pub fn default_for_stdout() -> Self {
        if io::stdout().is_terminal() {
            OutputFormat::Table
        } else {
            OutputFormat::Jsonl
        }
    }
}

/// Format and print a `RecordBatch` to stdout.
pub fn print_batch(batch: &RecordBatch, format: OutputFormat) -> Result<()> {
    let stdout = io::stdout();
    let mut out = stdout.lock();

    match format {
        OutputFormat::Json => print_json(&mut out, batch)?,
        OutputFormat::Jsonl => print_jsonl(&mut out, batch)?,
        OutputFormat::Table => print_table(&mut out, batch)?,
        OutputFormat::Csv => print_csv(&mut out, batch)?,
    }

    Ok(())
}

fn print_json(out: &mut impl Write, batch: &RecordBatch) -> Result<()> {
    let rows = batch_to_json_rows(batch)?;
    let envelope = serde_json::json!({
        "columns": batch.schema().fields().iter().map(|f| f.name().as_str()).collect::<Vec<_>>(),
        "row_count": batch.num_rows(),
        "rows": rows,
    });
    serde_json::to_writer_pretty(&mut *out, &envelope)?;
    writeln!(out)?;
    Ok(())
}

fn print_jsonl(out: &mut impl Write, batch: &RecordBatch) -> Result<()> {
    let rows = batch_to_json_rows(batch)?;
    for row in rows {
        serde_json::to_writer(&mut *out, &row)?;
        writeln!(out)?;
    }
    Ok(())
}

fn print_table(out: &mut impl Write, batch: &RecordBatch) -> Result<()> {
    let formatted = pretty_format_batches(std::slice::from_ref(batch))?;
    write!(out, "{formatted}")?;
    writeln!(out)?;
    Ok(())
}

fn print_csv(out: &mut impl Write, batch: &RecordBatch) -> Result<()> {
    let mut writer = arrow_csv::WriterBuilder::new().with_header(true).build(out);
    writer.write(batch)?;
    Ok(())
}

/// Convert a RecordBatch to a Vec of JSON objects.
fn batch_to_json_rows(batch: &RecordBatch) -> Result<Vec<serde_json::Value>> {
    let schema = batch.schema();
    let opts = FormatOptions::default();
    let formatters: Vec<ArrayFormatter> = batch
        .columns()
        .iter()
        .map(|col| ArrayFormatter::try_new(col.as_ref(), &opts))
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let mut rows = Vec::with_capacity(batch.num_rows());
    for row_idx in 0..batch.num_rows() {
        let mut map = serde_json::Map::new();
        for (col_idx, field) in schema.fields().iter().enumerate() {
            let value = if batch.column(col_idx).is_null(row_idx) {
                serde_json::Value::Null
            } else {
                let display = formatters[col_idx].value(row_idx);
                let s = display.to_string();
                // Try to parse as number, otherwise use string.
                if let Ok(n) = s.parse::<i64>() {
                    serde_json::Value::Number(n.into())
                } else if let Ok(n) = s.parse::<f64>() {
                    serde_json::Number::from_f64(n)
                        .map(serde_json::Value::Number)
                        .unwrap_or(serde_json::Value::String(s))
                } else {
                    serde_json::Value::String(s)
                }
            };
            map.insert(field.name().clone(), value);
        }
        rows.push(serde_json::Value::Object(map));
    }
    Ok(rows)
}
