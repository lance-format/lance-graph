// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

pub(super) fn qualify_alias_property(alias: &str, property: &str) -> String {
    format!("{}__{}", alias, property)
}
