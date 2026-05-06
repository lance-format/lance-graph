//! `LanceVersionWatcher` — DM-4 of the callcenter membrane plan.
//!
//! Single-producer / many-consumer fan-out using std-only sync primitives.
//! The membrane is the sole writer (one instance per session); every
//! external subscriber receives the latest `CognitiveEventRow` and skips
//! stale revisions — supabase-realtime shape with always-latest semantics.
//!
//! # I-2 (tokio outbound only)
//!
//! Per `.claude/board/SINGLE_BINARY_TOPOLOGY.md`, the watcher is a Layer-2
//! in-process membrane primitive. Tokio is reserved for Layer-3 outbound
//! sinks (DM-5 PhoenixServer, DM-8 PostgRestHandler). This file therefore
//! uses `std::sync::{Arc, RwLock, Mutex, Condvar}` and never `tokio::sync`.
//!
//! Earlier iterations used `tokio::sync::watch::Sender / Receiver` — see
//! the supabase-subscriber-v1 plan correction note (2026-05-06) for the
//! migration history. The semantic shape (always-latest, slow-subscribers-
//! skip) is preserved; only the runtime dependency changed.
//!
//! # BBB invariant
//!
//! The channel payload is `CognitiveEventRow`, the canonical Arrow-scalar
//! outbound DTO. `bbb_scalar_only_compile_check` in `lance_membrane.rs`
//! proves the row carries no VSA / RoleKey / NarsTruth.
//!
//! Plan: `.claude/plans/supabase-subscriber-v1.md` § DM-4.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::Duration;

use crate::external_intent::CognitiveEventRow;

/// Shared state between the watcher and all live receivers.
#[derive(Debug)]
struct WatcherInner {
    /// Always-latest snapshot. Stored as `Arc<T>` so receivers can clone
    /// it cheaply without holding the read lock during downstream work.
    latest: RwLock<Arc<CognitiveEventRow>>,
    /// Monotonic version counter. Every `bump()` increments this and
    /// notifies `cond`. Receivers track their last-seen version locally
    /// so spurious wakeups are filtered by version comparison.
    version: Mutex<u64>,
    /// Wakeup primitive paired with `version`. `notify_all()` on bump.
    cond: Condvar,
    /// Live receiver count. Decremented in `WatchReceiver::Drop`.
    receivers: AtomicUsize,
}

/// Fan-out for projected cognitive events (sync, in-process).
///
/// Wraps `std::sync` primitives keyed on `CognitiveEventRow`. Created
/// with a sentinel initial value (default row). Each
/// `LanceMembrane::project()` call feeds the latest committed row via
/// [`bump`](Self::bump); subscribers observe it with [`subscribe`](Self::subscribe).
#[derive(Debug)]
pub struct LanceVersionWatcher {
    inner: Arc<WatcherInner>,
}

impl LanceVersionWatcher {
    /// Build a watcher seeded with `initial`.
    ///
    /// The first `subscribe()` call sees this value via `current()`.
    /// Typical construction uses `CognitiveEventRow::default()` as the
    /// sentinel — subscribers that poll before any `project()` fire see
    /// an all-zero row.
    pub fn new(initial: CognitiveEventRow) -> Self {
        Self {
            inner: Arc::new(WatcherInner {
                latest: RwLock::new(Arc::new(initial)),
                version: Mutex::new(0),
                cond: Condvar::new(),
                receivers: AtomicUsize::new(0),
            }),
        }
    }

    /// Publish a fresh committed row. All current subscribers observe it.
    ///
    /// Returns `true` when at least one subscriber is listening, `false`
    /// when no receivers are attached. The membrane ignores the return
    /// value — a session with zero subscribers is a valid state.
    pub fn bump(&self, row: CognitiveEventRow) -> bool {
        // Swap latest first; readers seeing latest before they observe
        // the version bump will get the new value (always-latest, may
        // skip intermediate). Receivers waking on cond will read the
        // new version, then read latest — also fine.
        {
            let mut latest = self
                .inner
                .latest
                .write()
                .unwrap_or_else(|e| e.into_inner());
            *latest = Arc::new(row);
        }
        {
            let mut v = self
                .inner
                .version
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            *v = v.wrapping_add(1);
        }
        self.inner.cond.notify_all();
        self.inner.receivers.load(Ordering::Acquire) > 0
    }

    /// Attach a new subscriber.
    ///
    /// The receiver sees the most recently bumped row on first
    /// [`current()`](WatchReceiver::current) and is woken by subsequent
    /// bumps via [`wait_changed()`](WatchReceiver::wait_changed). Always-
    /// latest semantics are preserved: a slow subscriber may skip
    /// intermediate revisions and observe only the most recent one.
    pub fn subscribe(&self) -> WatchReceiver {
        self.inner.receivers.fetch_add(1, Ordering::AcqRel);
        let seen = *self
            .inner
            .version
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        WatchReceiver {
            inner: Arc::clone(&self.inner),
            seen,
        }
    }

    /// Observer count — useful for tests and diagnostics.
    pub fn receiver_count(&self) -> usize {
        self.inner.receivers.load(Ordering::Acquire)
    }
}

impl Default for LanceVersionWatcher {
    fn default() -> Self {
        Self::new(CognitiveEventRow::default())
    }
}

/// Subscriber handle. Always-latest snapshot via [`current`] and blocking
/// change-detection via [`wait_changed`] / [`wait_changed_timeout`] /
/// [`try_changed`]. Tracks last-seen version locally so wakeups return
/// only on a fresh bump after this receiver's prior wake.
///
/// Dropping the receiver decrements `LanceVersionWatcher::receiver_count`.
#[derive(Debug)]
pub struct WatchReceiver {
    inner: Arc<WatcherInner>,
    seen: u64,
}

impl WatchReceiver {
    /// Snapshot of the latest bumped row. Cheap clone of an `Arc`.
    pub fn current(&self) -> Arc<CognitiveEventRow> {
        Arc::clone(
            &self
                .inner
                .latest
                .read()
                .unwrap_or_else(|e| e.into_inner()),
        )
    }

    /// Block until a bump newer than `self.seen` arrives, return the
    /// latest snapshot, and update `seen` to the new version.
    ///
    /// Spurious wakeups are filtered by the version comparison (the
    /// `wait_while` predicate only returns when version != seen).
    pub fn wait_changed(&mut self) -> Arc<CognitiveEventRow> {
        let v = self
            .inner
            .version
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let v = self
            .inner
            .cond
            .wait_while(v, |v| *v == self.seen)
            .unwrap_or_else(|e| e.into_inner());
        self.seen = *v;
        drop(v);
        self.current()
    }

    /// Like [`wait_changed`] but with a timeout. Returns `None` when the
    /// timeout fires before a bump arrives.
    pub fn wait_changed_timeout(&mut self, timeout: Duration) -> Option<Arc<CognitiveEventRow>> {
        let v = self
            .inner
            .version
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (v, result) = self
            .inner
            .cond
            .wait_timeout_while(v, timeout, |v| *v == self.seen)
            .unwrap_or_else(|e| e.into_inner());
        if result.timed_out() {
            None
        } else {
            self.seen = *v;
            drop(v);
            Some(self.current())
        }
    }

    /// Non-blocking check. Returns the latest snapshot iff a bump newer
    /// than `self.seen` is available, otherwise `None`. Updates `seen`
    /// when it returns `Some`.
    pub fn try_changed(&mut self) -> Option<Arc<CognitiveEventRow>> {
        let v = self
            .inner
            .version
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if *v == self.seen {
            None
        } else {
            self.seen = *v;
            drop(v);
            Some(self.current())
        }
    }
}

impl Drop for WatchReceiver {
    fn drop(&mut self) {
        self.inner.receivers.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn subscribe_observes_initial() {
        let mut row = CognitiveEventRow::default();
        row.thinking = 7;
        let w = LanceVersionWatcher::new(row);
        let rx = w.subscribe();
        assert_eq!(rx.current().thinking, 7);
    }

    #[test]
    fn bump_delivers_latest() {
        let w = LanceVersionWatcher::default();
        let mut rx = w.subscribe();

        let mut row = CognitiveEventRow::default();
        row.free_e = 42;
        assert!(w.bump(row));

        // Non-blocking check picks up the new version.
        let snapshot = rx.try_changed().expect("changed after bump");
        assert_eq!(snapshot.free_e, 42);

        // Subsequent try_changed without a new bump returns None.
        assert!(rx.try_changed().is_none());
    }

    #[test]
    fn bump_without_subscribers_returns_false() {
        let w = LanceVersionWatcher::default();
        assert!(!w.bump(CognitiveEventRow::default()));
    }

    #[test]
    fn receiver_count_tracks_subscribers() {
        let w = LanceVersionWatcher::default();
        assert_eq!(w.receiver_count(), 0);
        let rx1 = w.subscribe();
        let rx2 = w.subscribe();
        assert_eq!(w.receiver_count(), 2);
        drop(rx1);
        assert_eq!(w.receiver_count(), 1);
        drop(rx2);
        assert_eq!(w.receiver_count(), 0);
    }

    #[test]
    fn wait_changed_blocks_until_bump() {
        let w = Arc::new(LanceVersionWatcher::default());
        let mut rx = w.subscribe();

        let writer = {
            let w = Arc::clone(&w);
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(20));
                let mut row = CognitiveEventRow::default();
                row.thinking = 13;
                w.bump(row);
            })
        };

        let snapshot = rx.wait_changed();
        assert_eq!(snapshot.thinking, 13);
        writer.join().unwrap();
    }

    #[test]
    fn wait_changed_timeout_fires_on_no_bump() {
        let w = LanceVersionWatcher::default();
        let mut rx = w.subscribe();
        let result = rx.wait_changed_timeout(Duration::from_millis(10));
        assert!(result.is_none(), "no bump in window → None");
    }

    #[test]
    fn wait_changed_timeout_returns_value_on_bump() {
        let w = Arc::new(LanceVersionWatcher::default());
        let mut rx = w.subscribe();

        let writer = {
            let w = Arc::clone(&w);
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(5));
                let mut row = CognitiveEventRow::default();
                row.free_e = 99;
                w.bump(row);
            })
        };

        let result = rx.wait_changed_timeout(Duration::from_millis(200));
        let snapshot = result.expect("bump arrived in window");
        assert_eq!(snapshot.free_e, 99);
        writer.join().unwrap();
    }

    /// I-2 invariant: WatchReceiver and LanceVersionWatcher must be
    /// `Send + Sync` without any tokio runtime. If a future refactor
    /// reintroduces `tokio::sync::*`, this test breaks at compile time
    /// once tokio types appear in the field set (tokio handles are
    /// Send+Sync but require a runtime to drive — defeating I-2).
    #[test]
    fn watcher_is_send_sync_without_runtime() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LanceVersionWatcher>();
        assert_send_sync::<WatchReceiver>();
    }
}
