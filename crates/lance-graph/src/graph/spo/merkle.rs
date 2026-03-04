// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Merkle root and ClamPath integrity for BindSpace nodes.
//!
//! Each node in the BindSpace has a ClamPath (DN address) and a MerkleRoot
//! stamped at write time. Verification checks whether the fingerprint
//! content still matches the stamped root.
//!
//! **Known gap**: `verify_lineage` currently performs a structural check only —
//! it does not re-hash the fingerprint data to detect bit-flip corruption.
//! This is documented and tested (test 6 expects this gap).

use crate::graph::fingerprint::{Fingerprint, ZERO_FP};
#[cfg(test)]
use crate::graph::fingerprint::FINGERPRINT_WORDS;

/// A Merkle root stamped on a BindSpace node at write time.
///
/// Computed from the fingerprint content via simple XOR-fold hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MerkleRoot(pub u64);

impl MerkleRoot {
    /// Compute merkle root from a fingerprint.
    pub fn from_fingerprint(fp: &Fingerprint) -> Self {
        let mut h: u64 = 0xa5a5a5a5a5a5a5a5;
        for &w in fp.iter() {
            h = h.rotate_left(7) ^ w;
            h = h.wrapping_mul(0x517cc1b727220a95);
        }
        MerkleRoot(h)
    }

    /// Check if this root is the zero/unset value.
    pub fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

/// A ClamPath is a hierarchical address (distinguished name path).
///
/// e.g., "agent:test:node" → depth=3, segments=["agent","test","node"]
#[derive(Debug, Clone)]
pub struct ClamPath {
    /// The full path string.
    pub path: String,
    /// Depth (number of segments).
    pub depth: u32,
}

impl ClamPath {
    /// Parse a colon-separated DN path.
    pub fn parse(path: &str) -> Self {
        let depth = path.split(':').count() as u32;
        Self {
            path: path.to_string(),
            depth,
        }
    }
}

/// A node in the BindSpace, addressed by ClamPath.
#[derive(Debug, Clone)]
pub struct BindNode {
    /// The ClamPath address.
    pub clam_path: ClamPath,
    /// The fingerprint data stored at this node.
    pub fingerprint: Fingerprint,
    /// Merkle root stamped at write time.
    pub merkle_root: MerkleRoot,
    /// Depth hint (from the write call).
    pub depth: u32,
}

/// Verification status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyStatus {
    /// Content matches the stamped merkle root.
    Consistent,
    /// Content has been modified since the root was stamped.
    Corrupted,
    /// Node not found.
    NotFound,
}

/// In-memory BindSpace for ClamPath → BindNode mapping.
///
/// Provides write, read, and merkle verification.
pub struct BindSpace {
    nodes: Vec<BindNode>,
}

impl BindSpace {
    /// Create an empty BindSpace.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Write a node into the BindSpace.
    ///
    /// Returns the address (index) of the new node.
    /// The merkle root is stamped at this point from the fingerprint.
    pub fn write_dn_path(&mut self, path: &str, fp: Fingerprint, depth: u32) -> usize {
        let merkle_root = MerkleRoot::from_fingerprint(&fp);
        let node = BindNode {
            clam_path: ClamPath::parse(path),
            fingerprint: fp,
            merkle_root,
            depth,
        };
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    /// Read a node by address.
    pub fn read(&self, addr: usize) -> Option<&BindNode> {
        self.nodes.get(addr)
    }

    /// Read a mutable node by address.
    pub fn read_mut(&mut self, addr: usize) -> Option<&mut BindNode> {
        self.nodes.get_mut(addr)
    }

    /// Get ClamPath and MerkleRoot for a node.
    pub fn clam_merkle(&self, addr: usize) -> Option<(&ClamPath, &MerkleRoot)> {
        self.nodes
            .get(addr)
            .map(|n| (&n.clam_path, &n.merkle_root))
    }

    /// Verify lineage integrity for a node.
    ///
    /// **Known gap**: This performs structural verification only — it checks
    /// that the merkle root is non-zero and the node exists. It does NOT
    /// re-compute the hash from current fingerprint content to detect
    /// bit-level corruption. This is a documented limitation.
    ///
    /// A full implementation would:
    /// ```ignore
    /// let recomputed = MerkleRoot::from_fingerprint(&node.fingerprint);
    /// if recomputed != node.merkle_root { return VerifyStatus::Corrupted; }
    /// ```
    pub fn verify_lineage(&self, addr: usize) -> VerifyStatus {
        match self.nodes.get(addr) {
            None => VerifyStatus::NotFound,
            Some(node) => {
                if node.merkle_root.is_zero() || node.fingerprint == ZERO_FP {
                    VerifyStatus::Corrupted
                } else {
                    // KNOWN GAP: does not re-hash and compare.
                    // Always returns Consistent if root is non-zero
                    // and fingerprint is non-zero.
                    VerifyStatus::Consistent
                }
            }
        }
    }

    /// Full integrity verification (re-hashes and compares).
    ///
    /// This is the correct implementation that `verify_lineage` should
    /// eventually use. Kept separate to document the gap.
    pub fn verify_integrity(&self, addr: usize) -> VerifyStatus {
        match self.nodes.get(addr) {
            None => VerifyStatus::NotFound,
            Some(node) => {
                let recomputed = MerkleRoot::from_fingerprint(&node.fingerprint);
                if recomputed == node.merkle_root {
                    VerifyStatus::Consistent
                } else {
                    VerifyStatus::Corrupted
                }
            }
        }
    }
}

impl Default for BindSpace {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_root_from_fingerprint() {
        let fp = [0xDEADu64; FINGERPRINT_WORDS];
        let root = MerkleRoot::from_fingerprint(&fp);
        assert!(!root.is_zero());

        // Same input → same root
        let root2 = MerkleRoot::from_fingerprint(&fp);
        assert_eq!(root, root2);
    }

    #[test]
    fn test_merkle_root_different() {
        let fp1 = [0xDEADu64; FINGERPRINT_WORDS];
        let fp2 = [0xBEEFu64; FINGERPRINT_WORDS];
        assert_ne!(
            MerkleRoot::from_fingerprint(&fp1),
            MerkleRoot::from_fingerprint(&fp2)
        );
    }

    #[test]
    fn test_clam_path_parse() {
        let cp = ClamPath::parse("agent:test:node");
        assert_eq!(cp.depth, 3);
        assert_eq!(cp.path, "agent:test:node");
    }

    #[test]
    fn test_bind_space_write_read() {
        let mut space = BindSpace::new();
        let fp = [0xDEADu64; FINGERPRINT_WORDS];
        let addr = space.write_dn_path("agent:test:node", fp, 3);

        let node = space.read(addr).unwrap();
        assert_eq!(node.fingerprint, fp);
        assert_eq!(node.depth, 3);
        assert!(!node.merkle_root.is_zero());
    }

    #[test]
    fn test_verify_lineage_gap() {
        let mut space = BindSpace::new();
        let fp = [0xDEADu64; FINGERPRINT_WORDS];
        let addr = space.write_dn_path("agent:test:node", fp, 3);

        // Before corruption: consistent
        assert_eq!(space.verify_lineage(addr), VerifyStatus::Consistent);

        // Corrupt the fingerprint
        space.read_mut(addr).unwrap().fingerprint[5] ^= 0xFFFF;

        // verify_lineage still says Consistent (KNOWN GAP)
        assert_eq!(space.verify_lineage(addr), VerifyStatus::Consistent);

        // verify_integrity correctly detects corruption
        assert_eq!(space.verify_integrity(addr), VerifyStatus::Corrupted);
    }
}
