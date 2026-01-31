//! Case-insensitive string utilities
//!
//! We use case-insensitive identifiers throughout to provide a SQL-like,
//! forgiving user experience.
//!
//! All identifiers (labels, properties, relationships) are case-insensitive,
//! meaning `Person`, `PERSON`, and `person` all refer to the same entity.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// A string wrapper that performs case-insensitive comparisons and hashing.
///
/// Internally stores the lowercase version for efficient comparison.
/// This is the core building block for case-insensitive identifier handling.
///
/// # Examples
///
/// ```
/// use lance_graph::case_insensitive::CaseInsensitiveStr;
///
/// let a = CaseInsensitiveStr::new("Person");
/// let b = CaseInsensitiveStr::new("person");
/// let c = CaseInsensitiveStr::new("PERSON");
///
/// assert_eq!(a, b);
/// assert_eq!(b, c);
/// ```
#[derive(Debug, Clone)]
pub struct CaseInsensitiveStr {
    normalized: String, // Always lowercase
}

impl CaseInsensitiveStr {
    /// Create a new case-insensitive string from any string-like type.
    ///
    /// The input is immediately converted to lowercase for storage.
    pub fn new(s: impl Into<String>) -> Self {
        Self {
            normalized: s.into().to_lowercase(),
        }
    }

    /// Get the normalized (lowercase) string representation.
    pub fn as_str(&self) -> &str {
        &self.normalized
    }
}

impl PartialEq for CaseInsensitiveStr {
    fn eq(&self, other: &Self) -> bool {
        self.normalized == other.normalized
    }
}

impl Eq for CaseInsensitiveStr {}

impl Hash for CaseInsensitiveStr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.normalized.hash(state);
    }
}

impl From<&str> for CaseInsensitiveStr {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for CaseInsensitiveStr {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl AsRef<str> for CaseInsensitiveStr {
    fn as_ref(&self) -> &str {
        &self.normalized
    }
}

/// Case-insensitive HashMap type alias for convenience
///
/// Use this type when you need a HashMap that performs case-insensitive
/// key lookups throughout the system.
pub type CaseInsensitiveMap<V> = HashMap<CaseInsensitiveStr, V>;

/// Create a qualified column name for internal DataFusion operations.
///
/// Returns format: `alias__property` (e.g., "p__name").
/// Both alias and property are normalized to lowercase for case-insensitive behavior.
///
/// This is the central utility for creating qualified column names throughout
/// the codebase. All join keys, scan projections, and expression translations
/// should use this function to ensure consistent case-insensitive behavior.
///
/// # Examples
///
/// ```
/// use lance_graph::case_insensitive::qualify_column;
///
/// assert_eq!(qualify_column("Person", "Name"), "person__name");
/// assert_eq!(qualify_column("p", "fullName"), "p__fullname");
/// ```
#[inline]
pub fn qualify_column(alias: &str, property: &str) -> String {
    format!("{}__{}", alias.to_lowercase(), property.to_lowercase())
}

/// Helper trait for case-insensitive lookups on standard HashMap<String, V>
///
/// This trait provides extension methods for performing case-insensitive
/// lookups on existing String-keyed HashMaps without requiring migration
/// to CaseInsensitiveMap.
///
/// # Performance
///
/// - **Best case**: O(1) when exact case matches (uses HashMap's fast path)
/// - **Worst case**: O(n) when case differs (iterates all keys to find match)
///
/// **For hot paths** (executed per-row or frequently), store normalized (lowercase)
/// keys in the HashMap to guarantee O(1) lookups. This trait is most appropriate for:
/// - Small HashMaps (< 100 entries)
/// - Cold paths (planning phase, executed once per query)
/// - Cases where preserving original case in keys is important
///
/// See `SemanticAnalyzer::variables` for an example of the optimized pattern where
/// keys are normalized to lowercase on insertion, ensuring all lookups hit the O(1) fast path.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// use lance_graph::case_insensitive::CaseInsensitiveLookup;
///
/// let mut map = HashMap::new();
/// map.insert("Person".to_string(), 1);
///
/// assert_eq!(map.get_ci("person"), Some(&1));
/// assert_eq!(map.get_ci("PERSON"), Some(&1));
/// assert_eq!(map.get_ci("Person"), Some(&1));
/// ```
pub trait CaseInsensitiveLookup<V> {
    /// Get a value with case-insensitive key lookup.
    ///
    /// Returns `Some(&V)` if a key matches (case-insensitively), `None` otherwise.
    fn get_ci(&self, key: &str) -> Option<&V>;

    /// Check if a key exists with case-insensitive lookup.
    fn contains_key_ci(&self, key: &str) -> bool;

    /// Get a mutable reference with case-insensitive key lookup.
    fn get_mut_ci(&mut self, key: &str) -> Option<&mut V>;
}

impl<V> CaseInsensitiveLookup<V> for HashMap<String, V> {
    fn get_ci(&self, key: &str) -> Option<&V> {
        // Try exact match first (fast path for common case)
        if let Some(v) = self.get(key) {
            return Some(v);
        }
        // Fall back to case-insensitive search
        let key_lower = key.to_lowercase();
        self.iter()
            .find(|(k, _)| k.to_lowercase() == key_lower)
            .map(|(_, v)| v)
    }

    fn contains_key_ci(&self, key: &str) -> bool {
        self.get_ci(key).is_some()
    }

    fn get_mut_ci(&mut self, key: &str) -> Option<&mut V> {
        // Find the actual key first
        let key_lower = key.to_lowercase();
        let actual_key = self.keys().find(|k| k.to_lowercase() == key_lower).cloned();

        // Then get mutable reference using the actual key
        actual_key.and_then(|k| self.get_mut(&k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_case_insensitive_str_equality() {
        let a = CaseInsensitiveStr::new("Person");
        let b = CaseInsensitiveStr::new("person");
        let c = CaseInsensitiveStr::new("PERSON");
        let d = CaseInsensitiveStr::new("PeRsOn");

        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
        assert_eq!(c, d);
    }

    #[test]
    fn test_case_insensitive_str_inequality() {
        let a = CaseInsensitiveStr::new("Person");
        let b = CaseInsensitiveStr::new("Company");

        assert_ne!(a, b);
    }

    #[test]
    fn test_case_insensitive_str_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(CaseInsensitiveStr::new("Person"));

        // All variations should be found
        assert!(set.contains(&CaseInsensitiveStr::new("person")));
        assert!(set.contains(&CaseInsensitiveStr::new("PERSON")));
        assert!(set.contains(&CaseInsensitiveStr::new("Person")));

        // Different value should not be found
        assert!(!set.contains(&CaseInsensitiveStr::new("Company")));
    }

    #[test]
    fn test_case_insensitive_map() {
        let mut map: CaseInsensitiveMap<i32> = HashMap::new();
        map.insert(CaseInsensitiveStr::new("Person"), 1);
        map.insert(CaseInsensitiveStr::new("Company"), 2);

        // Test various cases
        assert_eq!(map.get(&CaseInsensitiveStr::new("person")), Some(&1));
        assert_eq!(map.get(&CaseInsensitiveStr::new("PERSON")), Some(&1));
        assert_eq!(map.get(&CaseInsensitiveStr::new("Person")), Some(&1));
        assert_eq!(map.get(&CaseInsensitiveStr::new("PeRsOn")), Some(&1));

        assert_eq!(map.get(&CaseInsensitiveStr::new("company")), Some(&2));
        assert_eq!(map.get(&CaseInsensitiveStr::new("COMPANY")), Some(&2));

        assert_eq!(map.get(&CaseInsensitiveStr::new("Unknown")), None);
    }

    #[test]
    fn test_case_insensitive_lookup_trait() {
        let mut map = HashMap::new();
        map.insert("Person".to_string(), 1);
        map.insert("Company".to_string(), 2);
        map.insert("fullName".to_string(), 3);

        // Test get_ci
        assert_eq!(map.get_ci("person"), Some(&1));
        assert_eq!(map.get_ci("PERSON"), Some(&1));
        assert_eq!(map.get_ci("Person"), Some(&1));
        assert_eq!(map.get_ci("PeRsOn"), Some(&1));

        assert_eq!(map.get_ci("company"), Some(&2));
        assert_eq!(map.get_ci("COMPANY"), Some(&2));

        assert_eq!(map.get_ci("fullname"), Some(&3));
        assert_eq!(map.get_ci("FULLNAME"), Some(&3));
        assert_eq!(map.get_ci("FullName"), Some(&3));

        assert_eq!(map.get_ci("Unknown"), None);

        // Test contains_key_ci
        assert!(map.contains_key_ci("person"));
        assert!(map.contains_key_ci("COMPANY"));
        assert!(map.contains_key_ci("FullName"));
        assert!(!map.contains_key_ci("Unknown"));
    }

    #[test]
    fn test_case_insensitive_lookup_exact_match_fast_path() {
        let mut map = HashMap::new();
        map.insert("Person".to_string(), 1);

        // Exact match should use fast path
        assert_eq!(map.get_ci("Person"), Some(&1));

        // Case variations should still work
        assert_eq!(map.get_ci("person"), Some(&1));
        assert_eq!(map.get_ci("PERSON"), Some(&1));
    }

    #[test]
    fn test_case_insensitive_str_as_str() {
        let s = CaseInsensitiveStr::new("Person");
        assert_eq!(s.as_str(), "person"); // Stored as lowercase
    }

    #[test]
    fn test_case_insensitive_str_from_string() {
        let s = String::from("Person");
        let ci_str: CaseInsensitiveStr = s.into();
        assert_eq!(ci_str.as_str(), "person");
    }

    #[test]
    fn test_case_insensitive_str_from_str() {
        let ci_str: CaseInsensitiveStr = "Person".into();
        assert_eq!(ci_str.as_str(), "person");
    }

    #[test]
    fn test_case_insensitive_map_insertion_deduplication() {
        let mut map: CaseInsensitiveMap<i32> = HashMap::new();

        // Insert with different cases - should overwrite
        map.insert(CaseInsensitiveStr::new("Person"), 1);
        map.insert(CaseInsensitiveStr::new("person"), 2);
        map.insert(CaseInsensitiveStr::new("PERSON"), 3);

        // Should have only one entry with the latest value
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&CaseInsensitiveStr::new("person")), Some(&3));
    }

    #[test]
    fn test_get_mut_ci() {
        let mut map = HashMap::new();
        map.insert("Person".to_string(), 1);
        map.insert("Company".to_string(), 2);

        // Test mutable access with different cases
        if let Some(v) = map.get_mut_ci("person") {
            *v = 10;
        }
        assert_eq!(map.get_ci("Person"), Some(&10));

        if let Some(v) = map.get_mut_ci("COMPANY") {
            *v = 20;
        }
        assert_eq!(map.get_ci("company"), Some(&20));

        // Non-existent key
        assert!(map.get_mut_ci("Unknown").is_none());
    }

    #[test]
    fn test_property_name_normalization() {
        // Test realistic property names from Issue #105
        let mut map = HashMap::new();
        map.insert("fullName".to_string(), 1);
        map.insert("isActive".to_string(), 2);
        map.insert("numFollowers".to_string(), 3);

        // All variations should work
        assert_eq!(map.get_ci("fullname"), Some(&1));
        assert_eq!(map.get_ci("FULLNAME"), Some(&1));
        assert_eq!(map.get_ci("FullName"), Some(&1));

        assert_eq!(map.get_ci("isactive"), Some(&2));
        assert_eq!(map.get_ci("ISACTIVE"), Some(&2));
        assert_eq!(map.get_ci("IsActive"), Some(&2));

        assert_eq!(map.get_ci("numfollowers"), Some(&3));
        assert_eq!(map.get_ci("NUMFOLLOWERS"), Some(&3));
        assert_eq!(map.get_ci("NumFollowers"), Some(&3));
    }

    #[test]
    fn test_qualify_column() {
        use super::qualify_column;

        // Basic usage
        assert_eq!(qualify_column("p", "name"), "p__name");
        assert_eq!(qualify_column("person", "age"), "person__age");

        // Case normalization
        assert_eq!(qualify_column("P", "Name"), "p__name");
        assert_eq!(qualify_column("PERSON", "AGE"), "person__age");
        assert_eq!(qualify_column("Person", "fullName"), "person__fullname");

        // Mixed case
        assert_eq!(qualify_column("MyVar", "IsActive"), "myvar__isactive");
        assert_eq!(qualify_column("a", "NumFollowers"), "a__numfollowers");
    }
}
