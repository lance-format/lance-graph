// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use anyhow::Result;
use std::path::Path;

use crate::config::template_config;

pub fn run(path: &Path) -> Result<()> {
    if path.exists() {
        eprintln!("Config file already exists: {}", path.display());
        std::process::exit(1);
    }
    std::fs::write(path, template_config())?;
    println!("Created {}", path.display());
    Ok(())
}
