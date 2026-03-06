// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! lgraph — CLI for lance-graph query engine.

mod catalog_helpers;
mod commands;
mod config;
mod namespace_helpers;
mod output;

use clap::{Parser, Subcommand};
use output::OutputFormat;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "lgraph", about = "Lance Graph query engine CLI")]
struct Cli {
    /// Path to config file.
    #[arg(short, long, default_value = "lgraph.toml")]
    config: PathBuf,

    /// Output format (auto-detected if not specified).
    #[arg(short, long)]
    format: Option<OutputFormat>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a Cypher query.
    Cypher {
        /// The Cypher query string.
        query: String,
    },
    /// Run a SQL query.
    Sql {
        /// The SQL query string.
        query: String,
    },
    /// List available tables.
    Tables,
    /// Show schema for a table.
    Schema {
        /// Table name.
        table: String,
    },
    /// Create a template lgraph.toml config file.
    Init,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let format = cli.format.unwrap_or_else(OutputFormat::default_for_stdout);

    let result = match &cli.command {
        Command::Init => commands::init::run(&cli.config),
        cmd => {
            let cfg = match config::LGraphConfig::load(&cli.config) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Error: {e:#}");
                    eprintln!("Run 'lgraph init' to create a config file.");
                    std::process::exit(1);
                }
            };
            match cmd {
                Command::Cypher { query } => commands::cypher::run(&cfg, query, format).await,
                Command::Sql { query } => commands::sql::run(&cfg, query, format).await,
                Command::Tables => commands::tables::run(&cfg, format).await,
                Command::Schema { table } => commands::schema::run(&cfg, table, format).await,
                Command::Init => unreachable!(),
            }
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }
}
