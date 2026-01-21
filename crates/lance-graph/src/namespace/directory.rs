// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;
use lance_namespace::models::{DescribeTableRequest, DescribeTableResponse};
use lance_namespace::{Error as NamespaceError, LanceNamespace, Result};
use snafu::location;

/// A namespace that resolves table names relative to a base directory or URI.
#[derive(Debug, Clone)]
pub struct DirNamespace {
    base_uri: String,
}

impl DirNamespace {
    /// Create a new directory-backed namespace rooted at `base_uri`.
    ///
    /// The URI is normalized so that it does not end with a trailing slash.
    pub fn new(base_uri: impl Into<String>) -> Self {
        let uri = base_uri.into();
        let clean_uri = uri.trim_end_matches('/').to_string();
        Self {
            base_uri: clean_uri,
        }
    }

    /// Return the normalized base URI.
    pub fn base_uri(&self) -> &str {
        &self.base_uri
    }
}

#[async_trait]
impl LanceNamespace for DirNamespace {
    fn namespace_id(&self) -> String {
        format!("DirNamespace {{ base_uri: '{}' }}", self.base_uri)
    }

    async fn describe_table(&self, request: DescribeTableRequest) -> Result<DescribeTableResponse> {
        let id = request.id.ok_or_else(|| {
            NamespaceError::invalid_input(
                "DirNamespace requires the table identifier to be provided",
                location!(),
            )
        })?;

        if id.len() != 1 {
            return Err(NamespaceError::invalid_input(
                format!(
                    "DirNamespace expects identifiers with a single component, got {:?}",
                    id
                ),
                location!(),
            ));
        }

        let table_name = &id[0];
        let location = format!("{}/{}.lance", self.base_uri, table_name);

        let mut response = DescribeTableResponse::new();
        response.location = Some(location);
        response.storage_options = None;
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn describe_table_returns_clean_location() {
        let namespace = DirNamespace::new("s3://bucket/path/");
        let mut request = DescribeTableRequest::new();
        request.id = Some(vec!["users".to_string()]);

        let response = namespace.describe_table(request).await.unwrap();
        assert_eq!(
            response.location.as_deref(),
            Some("s3://bucket/path/users.lance")
        );
    }

    #[tokio::test]
    async fn describe_table_rejects_missing_identifier() {
        let namespace = DirNamespace::new("file:///tmp");
        let request = DescribeTableRequest::new();

        let err = namespace.describe_table(request).await.unwrap_err();
        assert!(
            err.to_string()
                .contains("DirNamespace requires the table identifier"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn describe_table_rejects_multi_component_identifier() {
        let namespace = DirNamespace::new("memory://namespace");
        let mut request = DescribeTableRequest::new();
        request.id = Some(vec!["foo".into(), "bar".into()]);

        let err = namespace.describe_table(request).await.unwrap_err();
        assert!(
            err.to_string()
                .contains("expects identifiers with a single component"),
            "unexpected error: {err}"
        );
    }
}
