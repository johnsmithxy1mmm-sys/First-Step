"""WS-triggered arbitrage re-checks: from "poll every minute" to "on the tick".

Arbitrage windows (neg-risk baskets, chain ladders) live seconds; the REST
polling jobs walk the whole board every 60-90s and pick up leftovers. The
fastlane closes that gap for the structures we ALREADY track: when a WS tick
lands on a token belonging to a prefiltered basket/pair, that one structure is
re-verified against live books immediately.

Threading contract:
  * `flag()` is called from the WS receive thread — it only adds a key to a
    set and sets an event (no locks held long, no I/O), so the tick loop
    never stalls.
  * The worker thread drains flagged keys and runs the actual verification
    (REST book fetches + possible execution) OFF the WS thread.
  * A per-key cooldown stops a tick storm from hammering the books API: one
    re-check per structure per `min_recheck_sec`, the rest coalesce.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Hashable

log = logging.getLogger(__name__)


class ArbFastlane(threading.Thread):
    daemon = True

    def __init__(self, check: Callable[[Hashable], None],
                 min_recheck_sec: float = 5.0):
        super().__init__(name="arb-fastlane")
        self._check = check
        self._min_recheck = min_recheck_sec
        self._dirty: set[Hashable] = set()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._halt = threading.Event()
        self._last_checked: dict[Hashable, float] = {}

    # --- WS-thread side (must never block) ---

    def flag(self, key: Hashable) -> None:
        with self._lock:
            self._dirty.add(key)
        self._wake.set()

    # --- worker side ---

    def stop(self) -> None:
        self._halt.set()
        self._wake.set()

    def drain(self, now: float | None = None) -> int:
        """Process pending keys once; returns how many were actually checked.

        Split out of run() so tests drive it synchronously.
        """
        with self._lock:
            keys, self._dirty = self._dirty, set()
        now = time.time() if now is None else now
        checked = 0
        for key in keys:
            if now - self._last_checked.get(key, 0.0) < self._min_recheck:
                continue          # cooling down; coalesced with a later tick
            self._last_checked[key] = now
            checked += 1
            try:
                self._check(key)
            except Exception:
                log.exception("arb fastlane check %r", key)
        # Prune stale cooldown entries so the map cannot grow unbounded.
        if len(self._last_checked) > 10_000:
            cutoff = now - self._min_recheck
            self._last_checked = {k: t for k, t in self._last_checked.items()
                                  if t >= cutoff}
        return checked

    def run(self) -> None:  # pragma: no cover — thin loop over drain()
        while not self._halt.is_set():
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            if self._halt.is_set():
                return
            self.drain()
