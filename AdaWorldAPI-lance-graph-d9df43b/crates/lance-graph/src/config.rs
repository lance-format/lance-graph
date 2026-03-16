// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Graph configuration for mapping Lance datasets to property graphs

use crate::error::{GraphError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for mapping Lance datasets to property graphs
///
/// # Important: Case-Insensitive Keys
///
/// **WARNING**: `node_mappings` and `relationship_mappings` store keys as **lowercase** for
/// case-insensitive behavior. If you directly insert into these maps, you **must** normalize
/// keys to lowercase to maintain invariants.
///
/// **Recommended**: Use `GraphConfigBuilder` instead of direct field access. The builder
/// automatically normalizes keys.
///
/// # Future API Changes
///
/// These fields may become private in a future major version to enforce invariants.
/// Code should migrate to using accessor methods (`get_node_mapping()`, `get_relationship_mapping()`)
/// rather than direct field access.
///
/// # TODO: API Safety
///
/// TODO: Make `node_mappings` and `relationship_mappings` private to prevent external code
/// from bypassing key normalization. This would require:
/// 1. Adding iterator methods for external access (e.g., `iter_node_mappings()`)
/// 2. Making this a breaking API change in next major version
/// 3. Ensuring all internal code uses accessor methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    /// Mapping of node labels to their field configurations
    ///
    /// **Keys are stored as lowercase** for case-insensitive lookups.
    /// Use `get_node_mapping()` for lookups instead of direct access.
    ///
    /// TODO: Make this private to enforce key normalization invariants
    pub node_mappings: HashMap<String, NodeMapping>,

    /// Mapping of relationship types to their field configurations
    ///
    /// **Keys are stored as lowercase** for case-insensitive lookups.
    /// Use `get_relationship_mapping()` for lookups instead of direct access.
    ///
    /// TODO: Make this private to enforce key normalization invariants
    pub relationship_mappings: HashMap<String, RelationshipMapping>,

    /// Default node ID field if not specified in mappings
    pub default_node_id_field: String,

    /// Default relationship type field if not specified in mappings
    pub default_relationship_type_field: String,
}

/// Configuration for mapping node labels to dataset fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMapping {
    /// The node label (e.g., "Person", "Product")
    pub label: String,
    /// Field name that serves as the node identifier
    pub id_field: String,
    /// Optional fields that define node properties
    pub property_fields: Vec<String>,
    /// Optional filter conditions for this node type
    pub filter_conditions: Option<String>,
}

/// Configuration for mapping relationship types to dataset fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipMapping {
    /// The relationship type (e.g., "KNOWS", "PURCHASED")
    pub relationship_type: String,
    /// Field containing the source node ID
    pub source_id_field: String,
    /// Field containing the target node ID
    pub target_id_field: String,
    /// Optional field containing the relationship type
    pub type_field: Option<String>,
    /// Optional fields that define relationship properties
    pub property_fields: Vec<String>,
    /// Optional filter conditions for this relationship type
    pub filter_conditions: Option<String>,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            node_mappings: HashMap::new(),
            relationship_mappings: HashMap::new(),
            default_node_id_field: "id".to_string(),
            default_relationship_type_field: "type".to_string(),
        }
    }
}

impl GraphConfig {
    /// Create a new builder for GraphConfig
    pub fn builder() -> GraphConfigBuilder {
        GraphConfigBuilder::new()
    }

    /// Get node mapping for a given label (case-insensitive)
    ///
    /// Looks up the node mapping using case-insensitive comparison.
    /// For example, "Person", "PERSON", and "person" all refer to the same label.
    pub fn get_node_mapping(&self, label: &str) -> Option<&NodeMapping> {
        self.node_mappings.get(&label.to_lowercase())
    }

    /// Get relationship mapping for a given type (case-insensitive)
    ///
    /// Looks up the relationship mapping using case-insensitive comparison.
    /// For example, "FOLLOWS", "follows", and "Follows" all refer to the same type.
    pub fn get_relationship_mapping(&self, rel_type: &str) -> Option<&RelationshipMapping> {
        self.relationship_mappings.get(&rel_type.to_lowercase())
    }

    /// Validate the configuration
    ///
    /// Checks for:
    /// - Empty ID fields
    /// - Non-normalized keys (must be lowercase)
    /// - Case-insensitive duplicates
    pub fn validate(&self) -> Result<()> {
        // Validate node mappings
        for (label, mapping) in &self.node_mappings {
            // Check that keys are normalized (lowercase)
            if label != &label.to_lowercase() {
                return Err(GraphError::ConfigError {
                    message: format!(
                        "Node mapping key '{}' is not normalized. \
                         Keys must be lowercase. Use GraphConfigBuilder to ensure proper normalization.",
                        label
                    ),
                    location: snafu::Location::new(file!(), line!(), column!()),
                });
            }

            if mapping.id_field.is_empty() {
                return Err(GraphError::ConfigError {
                    message: format!("Node mapping for '{}' has empty id_field", label),
                    location: snafu::Location::new(file!(), line!(), column!()),
                });
            }
        }

        // Validate relationship mappings
        for (rel_type, mapping) in &self.relationship_mappings {
            // Check that keys are normalized (lowercase)
            if rel_type != &rel_type.to_lowercase() {
                return Err(GraphError::ConfigError {
                    message: format!(
                        "Relationship mapping key '{}' is not normalized. \
                         Keys must be lowercase. Use GraphConfigBuilder to ensure proper normalization.",
                        rel_type
                    ),
                    location: snafu::Location::new(file!(), line!(), column!()),
                });
            }

            if mapping.source_id_field.is_empty() || mapping.target_id_field.is_empty() {
                return Err(GraphError::ConfigError {
                    message: format!(
                        "Relationship mapping for '{}' has empty source or target id field",
                        rel_type
                    ),
                    location: snafu::Location::new(file!(), line!(), column!()),
                });
            }
        }

        Ok(())
    }
}

/// Builder for GraphConfig
#[derive(Debug, Default, Clone)]
pub struct GraphConfigBuilder {
    node_mappings: HashMap<String, NodeMapping>,
    relationship_mappings: HashMap<String, RelationshipMapping>,
    default_node_id_field: Option<String>,
    default_relationship_type_field: Option<String>,
}

impl GraphConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node label mapping
    ///
    /// Note: Labels are case-insensitive. If you add "Person" and "person", the second
    /// will overwrite the first. Keys are stored as lowercase to prevent duplicates.
    pub fn with_node_label<S: Into<String>>(mut self, label: S, id_field: S) -> Self {
        let label_str = label.into();
        let normalized_key = label_str.to_lowercase();
        self.node_mappings.insert(
            normalized_key,
            NodeMapping {
                label: label_str, // Keep original case for display
                id_field: id_field.into(),
                property_fields: Vec::new(),
                filter_conditions: None,
            },
        );
        self
    }

    /// Add a node mapping with additional configuration
    pub fn with_node_mapping(mut self, mapping: NodeMapping) -> Self {
        let normalized_key = mapping.label.to_lowercase();
        self.node_mappings.insert(normalized_key, mapping);
        self
    }

    /// Add a relationship type mapping
    pub fn with_relationship<S: Into<String>>(
        mut self,
        rel_type: S,
        source_field: S,
        target_field: S,
    ) -> Self {
        let type_str = rel_type.into();
        let normalized_key = type_str.to_lowercase();
        self.relationship_mappings.insert(
            normalized_key,
            RelationshipMapping {
                relationship_type: type_str, // Keep original case for display
                source_id_field: source_field.into(),
                target_id_field: target_field.into(),
                type_field: None,
                property_fields: Vec::new(),
                filter_conditions: None,
            },
        );
        self
    }

    /// Add a relationship mapping with additional configuration
    pub fn with_relationship_mapping(mut self, mapping: RelationshipMapping) -> Self {
        let normalized_key = mapping.relationship_type.to_lowercase();
        self.relationship_mappings.insert(normalized_key, mapping);
        self
    }

    /// Set the default node ID field
    pub fn with_default_node_id_field<S: Into<String>>(mut self, field: S) -> Self {
        self.default_node_id_field = Some(field.into());
        self
    }

    /// Set the default relationship type field
    pub fn with_default_relationship_type_field<S: Into<String>>(mut self, field: S) -> Self {
        self.default_relationship_type_field = Some(field.into());
        self
    }

    /// Build the GraphConfig
    pub fn build(self) -> Result<GraphConfig> {
        let config = GraphConfig {
            node_mappings: self.node_mappings,
            relationship_mappings: self.relationship_mappings,
            default_node_id_field: self
                .default_node_id_field
                .unwrap_or_else(|| "id".to_string()),
            default_relationship_type_field: self
                .default_relationship_type_field
                .unwrap_or_else(|| "type".to_string()),
        };

        config.validate()?;
        Ok(config)
    }
}

impl NodeMapping {
    /// Create a new node mapping
    pub fn new<S: Into<String>>(label: S, id_field: S) -> Self {
        Self {
            label: label.into(),
            id_field: id_field.into(),
            property_fields: Vec::new(),
            filter_conditions: None,
        }
    }

    /// Add property fields to the mapping
    pub fn with_properties(mut self, fields: Vec<String>) -> Self {
        self.property_fields = fields;
        self
    }

    /// Add filter conditions for this node type
    pub fn with_filter<S: Into<String>>(mut self, filter: S) -> Self {
        self.filter_conditions = Some(filter.into());
        self
    }
}

impl RelationshipMapping {
    /// Create a new relationship mapping
    pub fn new<S: Into<String>>(rel_type: S, source_field: S, target_field: S) -> Self {
        Self {
            relationship_type: rel_type.into(),
            source_id_field: source_field.into(),
            target_id_field: target_field.into(),
            type_field: None,
            property_fields: Vec::new(),
            filter_conditions: None,
        }
    }

    /// Set the type field for this relationship
    pub fn with_type_field<S: Into<String>>(mut self, type_field: S) -> Self {
        self.type_field = Some(type_field.into());
        self
    }

    /// Add property fields to the mapping
    pub fn with_properties(mut self, fields: Vec<String>) -> Self {
        self.property_fields = fields;
        self
    }

    /// Add filter conditions for this relationship type
    pub fn with_filter<S: Into<String>>(mut self, filter: S) -> Self {
        self.filter_conditions = Some(filter.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_config_builder() {
        let config = GraphConfig::builder()
            .with_node_label("Person", "person_id")
            .with_node_label("Company", "company_id")
            .with_relationship("WORKS_FOR", "person_id", "company_id")
            .build()
            .unwrap();

        assert_eq!(config.node_mappings.len(), 2);
        assert_eq!(config.relationship_mappings.len(), 1);

        let person_mapping = config.get_node_mapping("Person").unwrap();
        assert_eq!(person_mapping.id_field, "person_id");

        let works_for_mapping = config.get_relationship_mapping("WORKS_FOR").unwrap();
        assert_eq!(works_for_mapping.source_id_field, "person_id");
        assert_eq!(works_for_mapping.target_id_field, "company_id");
    }

    #[test]
    fn test_validation_empty_id_field() {
        let mut config = GraphConfig::default();
        config.node_mappings.insert(
            "Person".to_string(),
            NodeMapping {
                label: "Person".to_string(),
                id_field: "".to_string(),
                property_fields: Vec::new(),
                filter_conditions: None,
            },
        );

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_node_mapping_with_properties() {
        let mapping = NodeMapping::new("Person", "id")
            .with_properties(vec!["name".to_string(), "age".to_string()])
            .with_filter("age > 18".to_string());

        assert_eq!(mapping.property_fields.len(), 2);
        assert!(mapping.filter_conditions.is_some());
    }

    #[test]
    fn test_case_insensitive_node_label_lookup() {
        // Test that node label lookups are case-insensitive
        let config = GraphConfig::builder()
            .with_node_label("Person", "person_id")
            .with_node_label("Company", "company_id")
            .build()
            .unwrap();

        // All case variations should work
        assert!(config.get_node_mapping("Person").is_some());
        assert!(config.get_node_mapping("person").is_some());
        assert!(config.get_node_mapping("PERSON").is_some());
        assert!(config.get_node_mapping("PeRsOn").is_some());

        assert!(config.get_node_mapping("Company").is_some());
        assert!(config.get_node_mapping("company").is_some());
        assert!(config.get_node_mapping("COMPANY").is_some());

        // Non-existent labels should return None
        assert!(config.get_node_mapping("Unknown").is_none());
        assert!(config.get_node_mapping("unknown").is_none());

        // Verify we get the same mapping regardless of case
        let mapping1 = config.get_node_mapping("Person").unwrap();
        let mapping2 = config.get_node_mapping("person").unwrap();
        let mapping3 = config.get_node_mapping("PERSON").unwrap();

        assert_eq!(mapping1.id_field, mapping2.id_field);
        assert_eq!(mapping2.id_field, mapping3.id_field);
        assert_eq!(mapping1.id_field, "person_id");
    }

    #[test]
    fn test_case_insensitive_relationship_type_lookup() {
        // Test that relationship type lookups are case-insensitive
        let config = GraphConfig::builder()
            .with_relationship("FOLLOWS", "src_id", "dst_id")
            .with_relationship("WORKS_FOR", "person_id", "company_id")
            .build()
            .unwrap();

        // All case variations should work
        assert!(config.get_relationship_mapping("FOLLOWS").is_some());
        assert!(config.get_relationship_mapping("follows").is_some());
        assert!(config.get_relationship_mapping("Follows").is_some());

        assert!(config.get_relationship_mapping("WORKS_FOR").is_some());
        assert!(config.get_relationship_mapping("works_for").is_some());
        assert!(config.get_relationship_mapping("Works_For").is_some());

        // Non-existent types should return None
        assert!(config.get_relationship_mapping("UNKNOWN").is_none());
        assert!(config.get_relationship_mapping("unknown").is_none());

        // Verify we get the same mapping regardless of case
        let mapping1 = config.get_relationship_mapping("FOLLOWS").unwrap();
        let mapping2 = config.get_relationship_mapping("follows").unwrap();
        let mapping3 = config.get_relationship_mapping("Follows").unwrap();

        assert_eq!(mapping1.source_id_field, mapping2.source_id_field);
        assert_eq!(mapping2.source_id_field, mapping3.source_id_field);
        assert_eq!(mapping1.source_id_field, "src_id");
    }

    #[test]
    fn test_duplicate_label_different_case_should_overwrite() {
        let builder = GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_node_label("person", "id2"); // Should overwrite first entry

        // Should have only 1 entry (second overwrites first due to lowercase key normalization)
        assert_eq!(builder.node_mappings.len(), 1);

        // The second one should have won (id2)
        let mapping = builder.node_mappings.get("person").unwrap();
        assert_eq!(mapping.id_field, "id2");
    }
}
