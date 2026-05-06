//! `CycleAccumulator` — per-cadence speed-ratio absorber.
//!
//! Sits between Layer 1 (BindSpace, 20–200 ns/op) and Layer 3 (outbound
//! sinks, 2–200 ms/flush) per `.claude/board/SINGLE_BINARY_TOPOLOGY.md`.
//! Absorbs the ~10,000× timescale ratio by batching many fast inner
//! cycles into one slow outer flush.
//!
//! # I-4: distinct from `CollapseGate`
//!
//! `collapse_gate::GateDecision` is the **per-row** write-airgap. It
//! decides HOW a single delta commits to BindSpace (Xor / Bundle /
//! Superposition / AlphaFrontToBack) and fires once per cycle.
//!
//! `CycleAccumulator` is the **per-cadence** flush gate. It decides
//! WHEN a batch of already-committed rows flushes outbound (rows-since-
//! flush ≥ N, OR ms-since-flush ≥ T). Fires once per outbound batch.
//!
//! Both are gates; they govern different boundaries. Conflating them
//! creates a `GATE-2` namespace clash on top of the existing `GATE-1`
//! between `mul::GateDecision` and `collapse_gate::GateDecision`.
//!
//! # Pure data + decision logic, no callbacks
//!
//! The accumulator is dep-free. It does not own a flush callback,
//! does not call `tokio::spawn`, does not do I/O. The caller drives
//! the flush:
//!
//! ```ignore
//! match accumulator.push(commit) {
//!     AccumulatorAction::Hold => {}
//!     AccumulatorAction::Flush => {
//!         let batch = accumulator.drain();
//!         outbound_sink.write_batch(batch);  // L3 work, possibly tokio
//!     }
//! }
//! ```
//!
//! This keeps the contract crate zero-dep and gives the consumer
//! (`lance-graph-callcenter`) full control over the flush mechanism.
//!
//! # Why this matters for q2 Phase 3
//!
//! Phase 2B q2 cockpit-server uses `MockShaderDriver` at low rate; SSE
//! `cycle_ms=300` is tractable without an accumulator. Phase 3 replaces
//! it with `BgzShaderDriver` at real cognitive-cycle speed (~10⁷ cycles
//! /sec). At 300 ms cadence that's ~3M cycles per window — the SSE
//! pipe / browser cannot absorb that. `CycleAccumulator` is the
//! architectural prerequisite for Phase 3 to ship.
//!
//! Plan: `.claude/board/SINGLE_BINARY_TOPOLOGY.md` § Per-cadence gate.

use std::time::{Duration, Instant};

/// Decision returned by `CycleAccumulator::push`. Either keep
/// accumulating or flush now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccumulatorAction {
    /// Below both thresholds — caller continues without flushing.
    Hold,
    /// At or above one of the thresholds — caller should call
    /// [`drain`](CycleAccumulator::drain) and ship the batch.
    Flush,
}

/// Per-cadence accumulator absorbing the L1↔L3 speed ratio.
///
/// Holds pending commits in an in-memory `Vec<C>`. A push triggers a
/// flush decision when EITHER:
///
/// - rows-since-last-flush ≥ `threshold_rows`, OR
/// - ms-since-last-flush ≥ `threshold_ms`.
///
/// `drain()` returns the accumulated batch and resets the timer +
/// rows. The caller is responsible for actually pushing the batch
/// out to a Layer-3 sink (MySQL, Phoenix WS, PostgREST, etc.) —
/// the accumulator never does I/O.
///
/// # Threading
///
/// Not internally synchronised. Wrap in `Mutex` or `RwLock` if shared
/// across threads. Typical pattern: one accumulator per outbound sink,
/// one writer thread that calls `push` on each new commit and `drain`
/// when the action is `Flush`.
#[derive(Debug)]
pub struct CycleAccumulator<C> {
    pending: Vec<C>,
    threshold_rows: usize,
    threshold_ms: u32,
    last_flush: Instant,
}

impl<C> CycleAccumulator<C> {
    /// Build an accumulator with row + ms thresholds.
    ///
    /// `threshold_rows = 0` is treated as "rows threshold disabled"
    /// (only the time threshold fires). `threshold_ms = 0` means
    /// "every push flushes" (degenerate but valid for testing).
    ///
    /// Initial capacity is `max(threshold_rows, 64)` to avoid
    /// reallocation on the steady-state hot path.
    pub fn new(threshold_rows: usize, threshold_ms: u32) -> Self {
        Self {
            pending: Vec::with_capacity(threshold_rows.max(64)),
            threshold_rows,
            threshold_ms,
            last_flush: Instant::now(),
        }
    }

    /// Append a commit. Returns whether the caller should drain now.
    ///
    /// Decision rules:
    /// - `threshold_rows > 0` AND `pending.len() >= threshold_rows` → `Flush`
    /// - elapsed since last flush ≥ `threshold_ms` → `Flush`
    /// - otherwise → `Hold`
    pub fn push(&mut self, commit: C) -> AccumulatorAction {
        self.pending.push(commit);
        if self.threshold_rows > 0 && self.pending.len() >= self.threshold_rows {
            return AccumulatorAction::Flush;
        }
        if self.last_flush.elapsed() >= Duration::from_millis(self.threshold_ms as u64) {
            return AccumulatorAction::Flush;
        }
        AccumulatorAction::Hold
    }

    /// Take ownership of the pending batch and reset the timer.
    ///
    /// Returns the accumulated commits in insertion order. Call after
    /// receiving an `AccumulatorAction::Flush`, or to force-flush a
    /// partial batch (e.g. on shutdown or explicit pull from a tokio-
    /// driven outbound runtime).
    pub fn drain(&mut self) -> Vec<C> {
        self.last_flush = Instant::now();
        std::mem::take(&mut self.pending)
    }

    /// Number of commits currently waiting for the next flush.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Whether the pending batch is empty.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Milliseconds since the last `drain()` (or construction).
    ///
    /// Useful for outbound runtimes that want to pull-flush stale
    /// batches even when neither threshold has fired.
    pub fn pending_age_ms(&self) -> u64 {
        self.last_flush.elapsed().as_millis() as u64
    }

    /// Configured row threshold (immutable after construction).
    pub fn threshold_rows(&self) -> usize {
        self.threshold_rows
    }

    /// Configured time threshold in ms (immutable after construction).
    pub fn threshold_ms(&self) -> u32 {
        self.threshold_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn push_below_row_threshold_holds() {
        let mut acc: CycleAccumulator<u32> = CycleAccumulator::new(10, 60_000);
        for i in 0..9 {
            assert_eq!(acc.push(i), AccumulatorAction::Hold);
        }
        assert_eq!(acc.pending_len(), 9);
    }

    #[test]
    fn push_at_row_threshold_flushes() {
        let mut acc: CycleAccumulator<u32> = CycleAccumulator::new(3, 60_000);
        assert_eq!(acc.push(1), AccumulatorAction::Hold);
        assert_eq!(acc.push(2), AccumulatorAction::Hold);
        assert_eq!(acc.push(3), AccumulatorAction::Flush);
    }

    #[test]
    fn drain_returns_batch_and_resets() {
        let mut acc: CycleAccumulator<u32> = CycleAccumulator::new(3, 60_000);
        acc.push(10);
        acc.push(20);
        acc.push(30);
        let batch = acc.drain();
        assert_eq!(batch, vec![10, 20, 30]);
        assert_eq!(acc.pending_len(), 0);
        assert!(acc.is_empty());
    }

    #[test]
    fn ms_threshold_triggers_flush() {
        // 100 rows OR 5 ms — sleep past 5 ms then push; should flush.
        let mut acc: CycleAccumulator<u32> = CycleAccumulator::new(100, 5);
        assert_eq!(acc.push(1), AccumulatorAction::Hold);
        thread::sleep(Duration::from_millis(10));
        assert_eq!(acc.push(2), AccumulatorAction::Flush);
    }

    #[test]
    fn ms_threshold_resets_on_drain() {
        let mut acc: CycleAccumulator<u32> = CycleAccumulator::new(100, 5);
        thread::sleep(Duration::from_millis(10));
        let _ = acc.drain();
        // drain reset the timer; next push within 1 ms should Hold.
        assert_eq!(acc.push(1), AccumulatorAction::Hold);
    }

    #[test]
    fn rows_threshold_zero_disables_count_path() {
        // threshold_rows = 0 → only ms threshold fires.
        let mut acc: CycleAccumulator<u32> = CycleAccumulator::new(0, 60_000);
        for i in 0..1000 {
            assert_eq!(
                acc.push(i),
                AccumulatorAction::Hold,
                "rows=0 should never flush on count"
            );
        }
        assert_eq!(acc.pending_len(), 1000);
    }

    #[test]
    fn ms_threshold_zero_flushes_immediately() {
        // threshold_ms = 0 → every push flushes (degenerate but valid).
        let mut acc: CycleAccumulator<u32> = CycleAccumulator::new(100, 0);
        // Sleep a tiny bit so elapsed > 0; some platforms have very
        // coarse Instant resolution.
        thread::sleep(Duration::from_millis(1));
        assert_eq!(acc.push(1), AccumulatorAction::Flush);
    }

    #[test]
    fn pending_age_ms_grows_then_resets() {
        let mut acc: CycleAccumulator<u32> = CycleAccumulator::new(100, 60_000);
        acc.push(1);
        thread::sleep(Duration::from_millis(10));
        assert!(
            acc.pending_age_ms() >= 10,
            "age should be ≥ 10 ms after sleep"
        );
        acc.drain();
        assert!(
            acc.pending_age_ms() < 5,
            "age should reset on drain (got {} ms)",
            acc.pending_age_ms()
        );
    }

    #[test]
    fn drain_empty_is_safe() {
        let mut acc: CycleAccumulator<u32> = CycleAccumulator::new(10, 100);
        let batch = acc.drain();
        assert!(batch.is_empty());
        assert!(acc.is_empty());
    }

    #[test]
    fn threshold_accessors_return_construction_values() {
        let acc: CycleAccumulator<u32> = CycleAccumulator::new(42, 137);
        assert_eq!(acc.threshold_rows(), 42);
        assert_eq!(acc.threshold_ms(), 137);
    }

    /// I-4 invariant: the accumulator carries no flush callback —
    /// callers drive the flush. This test pins the structural shape:
    /// `CycleAccumulator<C>` must NOT contain any `Box<dyn Fn>` or
    /// equivalent callback field. If a future refactor adds one, the
    /// contract crate's zero-dep promise risks growing a `Send + Sync`
    /// callable trait surface that doesn't belong here.
    #[test]
    fn pure_data_no_callback_field() {
        // Send + Sync iff C: Send + Sync. No callable trait objects.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CycleAccumulator<u32>>();
        assert_send_sync::<CycleAccumulator<()>>();
    }
}
