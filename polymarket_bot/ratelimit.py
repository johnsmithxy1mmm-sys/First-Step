"""Token-bucket rate limiter: even with the raised V2 limits the bot must
self-throttle — a Cloudflare queue on overshoot is worse than waiting
locally."""

from __future__ import annotations

import threading
import time


class RateLimited(RuntimeError):
    """Raised when a throttled READ could not get a token.

    Deliberately an exception rather than an empty result: `open_orders` and
    `api_positions` feed the idempotency reconcile and the desync kill-switch,
    and an empty list there reads as "nothing on the exchange" — which would
    silently disable the desync detector and let the bot double-enter. Every
    caller already treats an exception conservatively (assume a position
    exists / skip the cycle), so raising fails safe.
    """


class TokenBucket:
    def __init__(self, rate_per_sec: float, burst: float):
        self._rate = float(rate_per_sec)
        self._capacity = float(burst)
        self._tokens = float(burst)
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self._capacity, self._tokens + (now - self._updated) * self._rate)
        self._updated = now

    def try_acquire(self, tokens: float = 1.0) -> bool:
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def acquire(self, tokens: float = 1.0, timeout: float = 30.0) -> bool:
        """Blocking wait for tokens (with a timeout)."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                deficit = tokens - self._tokens
                wait = deficit / self._rate if self._rate > 0 else timeout
            if time.monotonic() + wait > deadline:
                return False
            time.sleep(min(wait, 0.5))
