// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Catalog and namespace utilities for Lance Graph.

pub mod namespace;
pub mod source_catalog;

pub use namespace::DirNamespace;
pub use source_catalog::{GraphSourceCatalog, InMemoryCatalog, SimpleTableSource};
