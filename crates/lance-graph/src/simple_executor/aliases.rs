// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::case_insensitive::qualify_column;

/// Qualify a column name for internal DataFusion operations.
/// Returns format: `alias__property` (e.g., "p__name").
/// Both alias and property are normalized to lowercase for case-insensitive behavior.
/// Note: This is for internal use only. Final output uses Cypher dot notation.
#[inline]
pub(super) fn qualify_alias_property(alias: &str, property: &str) -> String {
    qualify_column(alias, property)
}

/// Convert to Cypher-style column name for query results.
/// Returns format: `alias.property` (e.g., "p.name").
/// This matches the output format used by the DataFusion executor.
pub(super) fn to_cypher_column_name(alias: &str, property: &str) -> String {
    format!("{}.{}", alias, property)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qualify_alias_property() {
        // Should normalize to lowercase
        assert_eq!(qualify_alias_property("p", "name"), "p__name");
        assert_eq!(qualify_alias_property("person", "age"), "person__age");
        assert_eq!(qualify_alias_property("P", "Name"), "p__name");
        assert_eq!(qualify_alias_property("PERSON", "AGE"), "person__age");
        assert_eq!(
            qualify_alias_property("Person", "fullName"),
            "person__fullname"
        );
    }

    #[test]
    fn test_to_cypher_column_name() {
        assert_eq!(to_cypher_column_name("p", "name"), "p.name");
        assert_eq!(to_cypher_column_name("c", "company_name"), "c.company_name");
    }
}
