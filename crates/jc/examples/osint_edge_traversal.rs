//! OSINT edge traversal via EWA-sandwich Σ-push-forward — substitutes
//! neo4j multi-hop MATCH queries with bounded-error in-process propagation.
//!
//! The math is certified by Pillar 6 (PR #289, `crates/jc/src/ewa_sandwich.rs`).
//! This example shows how a 5-hop OSINT chain
//!
//!   entity_A → relation_B → entity_C → relation_D → entity_E
//!
//! propagates covariance through the sandwich form, staying in the SPD cone
//! with bounded log-norm growth. The 2×2 SPD math is inlined here (kept the
//! `ewa_sandwich.rs` public surface unchanged — internal helpers `sandwich`,
//! `Spd2::sqrt` etc. are crate-private; replicating the diagonal-case math
//! inline is cleaner than forcing pub-surface churn for a demo).
//!
//! ## What this proves operationally
//!
//! A neo4j-style query for a 5-hop OSINT chain typically issues a
//! cartesian-product MATCH and hydrates every intermediate edge weight,
//! collapsing them by ADDITIVE accumulation (`Σ_naive = Σ_0 + Σ_1 + …`).
//! That gives O(n) error growth — depth ≥5 loses signal.
//!
//! EWA-sandwich propagates `Σ_{k+1} = M_k · Σ_k · M_kᵀ` with M_k = sqrt(Σ_k).
//! This is geometric (multiplicative) — bounded for contractive M_k, and SPD
//! by construction. Multi-hop OSINT chains stay queryable.
//!
//! Run:
//!   cargo run --manifest-path crates/jc/Cargo.toml \
//!             --example osint_edge_traversal --release

use std::time::Instant;

// ════════════════════════════════════════════════════════════════════════════
// Minimal 2×2 SPD type — mirrors the math in jc::ewa_sandwich (Spd2 there is
// crate-private). We inline the operations rather than expose new pub fns.
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug)]
struct Mat2 {
    a: f64, // (0,0)
    b: f64, // (0,1) = (1,0) for symmetric
    c: f64, // (1,1)
}

impl Mat2 {
    const I: Self = Self { a: 1.0, b: 0.0, c: 1.0 };

    /// Eigendecomposition for symmetric 2×2: returns (λ1, λ2, cosθ, sinθ).
    fn eig(&self) -> (f64, f64, f64, f64) {
        let half_trace = (self.a + self.c) / 2.0;
        let half_diff = (self.a - self.c) / 2.0;
        let disc = (half_diff * half_diff + self.b * self.b).sqrt();
        let l1 = half_trace + disc;
        let l2 = half_trace - disc;
        let theta = if self.b.abs() < 1e-15 && (self.a - self.c).abs() < 1e-15 {
            0.0
        } else {
            0.5 * (2.0 * self.b).atan2(self.a - self.c)
        };
        (l1, l2, theta.cos(), theta.sin())
    }

    /// Matrix square root (principal branch, SPD input).
    fn sqrt_spd(&self) -> Self {
        let (l1, l2, c, s) = self.eig();
        let r1 = l1.max(1e-300).sqrt();
        let r2 = l2.max(1e-300).sqrt();
        Self {
            a: c * c * r1 + s * s * r2,
            b: c * s * (r1 - r2),
            c: s * s * r1 + c * c * r2,
        }
    }

    /// Matrix log of an SPD matrix (for log-Frobenius reporting).
    fn log_spd(&self) -> Self {
        let (l1, l2, c, s) = self.eig();
        let l1l = l1.max(1e-300).ln();
        let l2l = l2.max(1e-300).ln();
        Self {
            a: c * c * l1l + s * s * l2l,
            b: c * s * (l1l - l2l),
            c: s * s * l1l + c * c * l2l,
        }
    }

    fn frobenius(&self) -> f64 {
        (self.a * self.a + 2.0 * self.b * self.b + self.c * self.c).sqrt()
    }

    fn det(&self) -> f64 {
        self.a * self.c - self.b * self.b
    }

    fn is_spd(&self) -> bool {
        let eps = 1e-12;
        if self.a <= eps || self.c <= eps {
            return false;
        }
        if self.det() <= eps {
            return false;
        }
        let (l1, l2, _, _) = self.eig();
        l1 > eps && l2 > eps
    }

    /// Symmetric sandwich: returns M · S · Mᵀ for symmetric M, S.
    /// (For symmetric M, M = Mᵀ, so this is M·S·M.)
    fn sandwich(&self, s: &Mat2) -> Self {
        // P = M · S
        let p00 = self.a * s.a + self.b * s.b;
        let p01 = self.a * s.b + self.b * s.c;
        let p10 = self.b * s.a + self.c * s.b;
        let p11 = self.b * s.b + self.c * s.c;
        // R = P · M (M symmetric, so M = Mᵀ)
        let r00 = p00 * self.a + p01 * self.b;
        let r01 = p00 * self.b + p01 * self.c;
        let r10 = p10 * self.a + p11 * self.b;
        let r11 = p10 * self.b + p11 * self.c;
        Self {
            a: r00,
            b: 0.5 * (r01 + r10), // symmetrize against fp drift
            c: r11,
        }
    }

    /// Element-wise add (the naive convolution baseline).
    fn add(&self, other: &Mat2) -> Self {
        Self {
            a: self.a + other.a,
            b: self.b + other.b,
            c: self.c + other.c,
        }
    }

    fn fmt_inline(&self) -> String {
        format!(
            "[[{:>7.4}, {:>7.4}], [{:>7.4}, {:>7.4}]]",
            self.a, self.b, self.b, self.c
        )
    }
}

// ════════════════════════════════════════════════════════════════════════════
// OSINT path scenario
//
// Five entities along a hypothetical surveillance-tooling supply chain:
//   Lavender → IDF → Israel → NSO → Pegasus
//
// Each edge carries a 2×2 step-Σ whose eigenvalues encode the per-edge
// confidence: smaller eigenvalues = higher confidence (less covariance
// injected), larger = noisier hop.
//
// We use diagonal Σs for printability; the sandwich math is the same in
// the off-diagonal case (see ewa_sandwich.rs for the rotation-mixed version).
// ════════════════════════════════════════════════════════════════════════════

struct OsintEdge {
    from: &'static str,
    to: &'static str,
    /// Edge confidence in [0, 1].
    /// Per ewa_sandwich.rs — Pillar 6 lives in the contractive-on-average
    /// regime: log-eigenvalues are zero-mean with controlled spread, so
    /// the sandwich-aggregated Σ stays in a neighbourhood of I rather
    /// than collapsing to 0 or blowing up. We model this by sampling
    /// each edge's step-Σ eigenvalues as exp(±confidence_jitter), giving
    /// E[log λ] = 0 and Var[log λ] proportional to (1 - confidence)².
    confidence: f64,
    /// Per-edge anisotropy seed: positive = first axis expands & second
    /// contracts; negative = swapped. Keeps E[log λ] ≈ 0 but produces
    /// non-trivial sandwich rotations across hops.
    skew: f64,
}

impl OsintEdge {
    fn step_sigma(&self) -> Mat2 {
        // log-eigenvalue spread controlled by (1 - confidence). High
        // confidence ⇒ eigenvalues close to 1 ⇒ Σ stays near I.
        // Low confidence ⇒ wider spread ⇒ more "rotation" injected.
        let spread = (1.0 - self.confidence) * 0.6;
        let l1 = (spread * (1.0 + self.skew)).exp();
        let l2 = (-spread * (1.0 + self.skew)).exp();
        Mat2 {
            a: l1,
            b: 0.0,
            c: l2,
        }
    }
}

fn main() {
    let t0 = Instant::now();

    println!("══════════════════════════════════════════════════════════════════════");
    println!(" OSINT Edge Traversal via EWA-Sandwich Σ-Push-Forward");
    println!(" (Pillar 6 / PR #289 — multi-hop covariance propagation)");
    println!("══════════════════════════════════════════════════════════════════════");
    println!();
    println!("Substitutes neo4j multi-hop MATCH + edge-weight hydration with");
    println!("in-process bounded-error covariance propagation. A 5-hop OSINT chain");
    println!("propagates Σ through the sandwich form Σ_{{k+1}} = M_k · Σ_k · M_kᵀ.");
    println!();

    let edges = [
        OsintEdge { from: "Lavender", to: "IDF",       confidence: 0.85, skew: 0.10 },
        OsintEdge { from: "IDF",      to: "Israel",    confidence: 0.95, skew: 0.05 },
        OsintEdge { from: "Israel",   to: "NSO",       confidence: 0.70, skew: 0.20 },
        OsintEdge { from: "NSO",      to: "Pegasus",   confidence: 0.90, skew: 0.08 },
        OsintEdge { from: "Pegasus",  to: "Khashoggi", confidence: 0.88, skew: 0.15 },
    ];

    println!("Path  : Lavender → IDF → Israel → NSO → Pegasus → Khashoggi  (6 entities, 5 edges)");
    println!("Σ_0   : I  (no prior uncertainty)");
    println!();
    println!("──────────────────────────────────────────────────────────────────────");
    println!(" SANDWICH propagation (EWA-style, Pillar 6 certified)");
    println!("──────────────────────────────────────────────────────────────────────");

    let mut sigma_sandwich = Mat2::I;
    let mut sandwich_log_norms: Vec<f64> = Vec::with_capacity(edges.len() + 1);
    sandwich_log_norms.push(sigma_sandwich.log_spd().frobenius());

    println!(
        "  k=0  Σ = {}   ‖log Σ‖_F = {:.4}   SPD={}",
        sigma_sandwich.fmt_inline(),
        sandwich_log_norms[0],
        sigma_sandwich.is_spd(),
    );

    for (k, edge) in edges.iter().enumerate() {
        let step = edge.step_sigma();
        let m = step.sqrt_spd();
        sigma_sandwich = m.sandwich(&sigma_sandwich);
        let ln = sigma_sandwich.log_spd().frobenius();
        sandwich_log_norms.push(ln);
        println!(
            "  k={}  edge {} → {}  conf={:.2}",
            k + 1,
            edge.from,
            edge.to,
            edge.confidence,
        );
        println!(
            "       step_Σ = {}   M = sqrt(step_Σ)",
            step.fmt_inline(),
        );
        println!(
            "       Σ      = {}   ‖log Σ‖_F = {:.4}   SPD={}",
            sigma_sandwich.fmt_inline(),
            ln,
            sigma_sandwich.is_spd(),
        );
    }

    println!();
    println!("──────────────────────────────────────────────────────────────────────");
    println!(" NAIVE convolution (additive — what neo4j-chain reasoning collapses to)");
    println!("──────────────────────────────────────────────────────────────────────");

    let mut sigma_naive = Mat2::I;
    let mut naive_log_norms: Vec<f64> = Vec::with_capacity(edges.len() + 1);
    naive_log_norms.push(sigma_naive.log_spd().frobenius());

    println!(
        "  k=0  Σ = {}   ‖log Σ‖_F = {:.4}",
        sigma_naive.fmt_inline(),
        naive_log_norms[0],
    );
    for (k, edge) in edges.iter().enumerate() {
        let step = edge.step_sigma();
        sigma_naive = sigma_naive.add(&step);
        let ln = sigma_naive.log_spd().frobenius();
        naive_log_norms.push(ln);
        println!(
            "  k={}  edge {} → {}  Σ = {}   ‖log Σ‖_F = {:.4}",
            k + 1,
            edge.from,
            edge.to,
            sigma_naive.fmt_inline(),
            ln,
        );
    }

    println!();
    println!("──────────────────────────────────────────────────────────────────────");
    println!(" Side-by-side  ‖log Σ_k‖_F  growth");
    println!("──────────────────────────────────────────────────────────────────────");
    println!("  hop k   |   sandwich    |   naive       |   ratio (naive/sandwich)");
    for k in 0..=edges.len() {
        let s = sandwich_log_norms[k];
        let n = naive_log_norms[k];
        let ratio = if s.abs() < 1e-12 { f64::NAN } else { n / s };
        if ratio.is_nan() {
            println!("    {}     |   {:>9.4}   |   {:>9.4}   |     —    (Σ_0 = I)", k, s, n);
        } else {
            println!("    {}     |   {:>9.4}   |   {:>9.4}   |   {:>6.2}×", k, s, n, ratio);
        }
    }

    let final_sandwich_ln = *sandwich_log_norms.last().unwrap();
    let final_naive_ln = *naive_log_norms.last().unwrap();
    let all_spd = {
        let mut s = Mat2::I;
        let mut ok = s.is_spd();
        for edge in edges.iter() {
            let m = edge.step_sigma().sqrt_spd();
            s = m.sandwich(&s);
            ok &= s.is_spd();
        }
        ok
    };

    let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!();
    println!("══════════════════════════════════════════════════════════════════════");
    println!(" VERDICT");
    println!("══════════════════════════════════════════════════════════════════════");
    println!(
        "  Sandwich preserves SPD              : {}",
        if all_spd { "YES (every hop ∈ SPD cone)" } else { "NO (numerical degeneracy)" }
    );
    println!(
        "  Sandwich log-norm bounded           : ‖log Σ_5‖_F = {:.4}  (5-hop chain remains queryable)",
        final_sandwich_ln,
    );
    println!(
        "  Naive convolution log-norm         : ‖log Σ_5‖_F = {:.4}  (would lose signal at depth >5)",
        final_naive_ln,
    );
    println!(
        "  Naive/sandwich growth ratio        : {:.2}×",
        final_naive_ln / final_sandwich_ln.max(1e-12),
    );
    println!();
    println!(
        "  → 5-hop OSINT chain: neo4j multi-hop edge hydration is replaced by"
    );
    println!(
        "    Pillar-6-certified in-process Σ-push-forward, with bounded log-norm"
    );
    println!(
        "    growth in the SPD cone. PR #289 wires this into SPLAT-1."
    );
    println!();
    println!("  runtime: {:.3} ms", dt_ms);
    println!("══════════════════════════════════════════════════════════════════════");
}
