// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # GraphBLAS Descriptors
//!
//! Descriptors control the behaviour of GraphBLAS operations, such as whether
//! to transpose inputs, complement masks, or replace outputs.

use std::fmt;

/// Descriptor field identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DescField {
    /// Controls the output container.
    Output,
    /// Controls the mask.
    Mask,
    /// Controls the first input.
    Input0,
    /// Controls the second input.
    Input1,
}

/// Descriptor value for a given field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DescValue {
    /// Default behaviour (no modification).
    Default,
    /// Transpose the matrix operand.
    Transpose,
    /// Complement the mask (use structural complement).
    Complement,
    /// Replace mode: clear the output before writing results.
    Replace,
    /// Structure-only mask: ignore values, use pattern only.
    Structure,
}

/// A descriptor that controls GraphBLAS operation behaviour.
///
/// Each field can be set independently. Unset fields default to [`DescValue::Default`].
#[derive(Clone, PartialEq, Eq)]
pub struct Descriptor {
    /// Output field setting.
    pub output: DescValue,
    /// Mask field setting.
    pub mask: DescValue,
    /// First input field setting.
    pub inp0: DescValue,
    /// Second input field setting.
    pub inp1: DescValue,
}

impl Default for Descriptor {
    fn default() -> Self {
        Self {
            output: DescValue::Default,
            mask: DescValue::Default,
            inp0: DescValue::Default,
            inp1: DescValue::Default,
        }
    }
}

impl fmt::Debug for Descriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Descriptor")
            .field("output", &self.output)
            .field("mask", &self.mask)
            .field("inp0", &self.inp0)
            .field("inp1", &self.inp1)
            .finish()
    }
}

impl Descriptor {
    /// Create a new default descriptor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a descriptor field.
    pub fn set(&mut self, field: DescField, value: DescValue) -> &mut Self {
        match field {
            DescField::Output => self.output = value,
            DescField::Mask => self.mask = value,
            DescField::Input0 => self.inp0 = value,
            DescField::Input1 => self.inp1 = value,
        }
        self
    }

    /// Get the value of a descriptor field.
    pub fn get(&self, field: DescField) -> DescValue {
        match field {
            DescField::Output => self.output,
            DescField::Mask => self.mask,
            DescField::Input0 => self.inp0,
            DescField::Input1 => self.inp1,
        }
    }

    /// Whether the first input should be transposed.
    pub fn transpose_input0(&self) -> bool {
        self.inp0 == DescValue::Transpose
    }

    /// Whether the second input should be transposed.
    pub fn transpose_input1(&self) -> bool {
        self.inp1 == DescValue::Transpose
    }

    /// Whether the mask should be complemented.
    pub fn complement_mask(&self) -> bool {
        self.mask == DescValue::Complement
    }

    /// Whether the output should be replaced (cleared before writing).
    pub fn replace_output(&self) -> bool {
        self.output == DescValue::Replace
    }

    /// Whether the mask is structure-only.
    pub fn structure_mask(&self) -> bool {
        self.mask == DescValue::Structure
    }
}

/// Pre-built descriptor constants.
pub mod GrBDesc {
    use super::*;

    /// Default descriptor — no modifications.
    pub fn default() -> Descriptor {
        Descriptor::default()
    }

    /// Transpose the first input.
    pub fn t0() -> Descriptor {
        let mut d = Descriptor::default();
        d.set(DescField::Input0, DescValue::Transpose);
        d
    }

    /// Transpose the second input.
    pub fn t1() -> Descriptor {
        let mut d = Descriptor::default();
        d.set(DescField::Input1, DescValue::Transpose);
        d
    }

    /// Transpose both inputs.
    pub fn t0t1() -> Descriptor {
        let mut d = Descriptor::default();
        d.set(DescField::Input0, DescValue::Transpose);
        d.set(DescField::Input1, DescValue::Transpose);
        d
    }

    /// Complement the mask.
    pub fn comp() -> Descriptor {
        let mut d = Descriptor::default();
        d.set(DescField::Mask, DescValue::Complement);
        d
    }

    /// Replace the output.
    pub fn replace() -> Descriptor {
        let mut d = Descriptor::default();
        d.set(DescField::Output, DescValue::Replace);
        d
    }

    /// Replace the output and complement the mask.
    pub fn replace_comp() -> Descriptor {
        let mut d = Descriptor::default();
        d.set(DescField::Output, DescValue::Replace);
        d.set(DescField::Mask, DescValue::Complement);
        d
    }

    /// Structure-only mask.
    pub fn structure() -> Descriptor {
        let mut d = Descriptor::default();
        d.set(DescField::Mask, DescValue::Structure);
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_descriptor() {
        let d = Descriptor::default();
        assert_eq!(d.get(DescField::Output), DescValue::Default);
        assert_eq!(d.get(DescField::Mask), DescValue::Default);
        assert_eq!(d.get(DescField::Input0), DescValue::Default);
        assert_eq!(d.get(DescField::Input1), DescValue::Default);
        assert!(!d.transpose_input0());
        assert!(!d.transpose_input1());
        assert!(!d.complement_mask());
        assert!(!d.replace_output());
    }

    #[test]
    fn test_set_fields() {
        let mut d = Descriptor::new();
        d.set(DescField::Input0, DescValue::Transpose);
        assert!(d.transpose_input0());
        assert!(!d.transpose_input1());

        d.set(DescField::Mask, DescValue::Complement);
        assert!(d.complement_mask());
    }

    #[test]
    fn test_presets() {
        let t0 = GrBDesc::t0();
        assert!(t0.transpose_input0());
        assert!(!t0.transpose_input1());

        let t1 = GrBDesc::t1();
        assert!(!t1.transpose_input0());
        assert!(t1.transpose_input1());

        let t0t1 = GrBDesc::t0t1();
        assert!(t0t1.transpose_input0());
        assert!(t0t1.transpose_input1());

        let comp = GrBDesc::comp();
        assert!(comp.complement_mask());

        let repl = GrBDesc::replace();
        assert!(repl.replace_output());

        let rc = GrBDesc::replace_comp();
        assert!(rc.replace_output());
        assert!(rc.complement_mask());

        let s = GrBDesc::structure();
        assert!(s.structure_mask());
    }

    #[test]
    fn test_debug_format() {
        let d = GrBDesc::t0();
        let s = format!("{:?}", d);
        assert!(s.contains("Transpose"));
    }
}
