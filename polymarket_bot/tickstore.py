"""Continuous market-data recording: the raw material of the learning loop.

Every learned model downstream (category correlations, fill calibration,
adverse-selection priors) needs history; without it "self-calibration" is a
word. This store records two streams into its own sqlite file (kept separate
from the ledger so analytical bulk writes never contend with trade records):

  * ticks       — top-of-book updates for the tokens the WS already watches
                  (quoted markets, open positions, tracked arb structures);
  * cat_index   — one row per category per main cycle: the volume-weighted
                  mean daily price change of that category's markets. A
                  cheap, honest factor proxy — correlations of these series
                  replace the expert matrix once there is enough history.

Threading contract: `record_*` only enqueue (bounded queue, drop-oldest on
overflow) — safe to call from the WS receive thread. A daemon writer thread
owns the sqlite connection (sqlite's one-thread rule), batches inserts, and
prunes rows past the retention window.
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ticks (
    ts REAL NOT NULL,
    token TEXT NOT NULL,
    bid REAL, ask REAL, bid_size REAL, ask_size REAL
);
CREATE INDEX IF NOT EXISTS idx_ticks_token_ts ON ticks(token, ts);
CREATE INDEX IF NOT EXISTS idx_ticks_ts ON ticks(ts);
CREATE TABLE IF NOT EXISTS cat_index (
    ts REAL NOT NULL,
    category TEXT NOT NULL,
    value REAL NOT NULL,
    n INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cat_ts ON cat_index(category, ts);
"""


class TickStore:
    def __init__(self, db_path: str, retention_days: float = 30.0,
                 flush_sec: float = 2.0, max_queue: int = 50_000):
        from pathlib import Path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._retention_sec = retention_days * 86_400.0
        self._flush_sec = flush_sec
        self._queue: queue.Queue[tuple] = queue.Queue(maxsize=max_queue)
        self._halt = threading.Event()
        self._last_prune = 0.0
        self._writer = threading.Thread(target=self._run, name="tick-writer",
                                        daemon=True)
        self.dropped = 0          # ticks lost to backpressure (observability)

    def start(self) -> None:
        self._writer.start()

    # --- producers (any thread; never block) ---

    def record_tick(self, token: str, bid: float, ask: float,
                    bid_size: float, ask_size: float,
                    ts: float | None = None) -> None:
        now = time.time() if ts is None else ts   # a real ts=0.0 is not "missing"
        self._put(("tick", now, token, bid, ask, bid_size, ask_size))

    def record_category_index(self, category: str, value: float, n: int,
                              ts: float | None = None) -> None:
        now = time.time() if ts is None else ts
        self._put(("cat", now, category, value, n))

    def _put(self, row: tuple) -> None:
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            # Never block a hot path for analytics: drop the OLDEST, keep the new.
            self.dropped += 1
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(row)
            except (queue.Empty, queue.Full):
                pass

    # --- reader side (short-lived connections; any thread) ---

    def category_series(self, limit_rows: int = 5000) -> dict[str, list[tuple[float, float]]]:
        """(ts, value) series per category, oldest first — correlation input."""
        conn = sqlite3.connect(self._db_path)
        try:
            rows = conn.execute(
                "SELECT ts, category, value FROM cat_index "
                "ORDER BY ts DESC LIMIT ?", (limit_rows,)).fetchall()
        except sqlite3.OperationalError:
            return {}
        finally:
            conn.close()
        out: dict[str, list[tuple[float, float]]] = {}
        for ts, cat, value in reversed(rows):
            out.setdefault(cat, []).append((ts, value))
        return out

    def tick_count(self) -> int:
        conn = sqlite3.connect(self._db_path)
        try:
            return conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
        except sqlite3.OperationalError:
            return 0
        finally:
            conn.close()

    def token_series(self, tokens: list[str], limit_per_token: int = 2000
                     ) -> dict[str, list[tuple[float, float]]]:
        """(ts, mid) series per token, oldest first — lead-lag analysis input."""
        if not tokens:
            return {}
        conn = sqlite3.connect(self._db_path)
        out: dict[str, list[tuple[float, float]]] = {}
        try:
            for token in tokens:
                rows = conn.execute(
                    "SELECT ts, bid, ask FROM ticks WHERE token=? "
                    "ORDER BY ts DESC LIMIT ?", (token, limit_per_token)).fetchall()
                series = []
                for ts, bid, ask in reversed(rows):
                    mid = ((bid + ask) / 2.0 if bid and ask and bid > 0 and ask > 0
                           else (bid or ask or 0.0))
                    if mid > 0:
                        series.append((ts, mid))
                if series:
                    out[token] = series
        except sqlite3.OperationalError:
            return {}
        finally:
            conn.close()
        return out

    def token_book_series(self, tokens: list[str], limit_per_token: int = 4000
                          ) -> dict[str, list[tuple]]:
        """(ts, bid, ask, bid_size, ask_size) per token, oldest first — the
        counterfactual replay's input (needs the full top of book, not just mid)."""
        if not tokens:
            return {}
        conn = sqlite3.connect(self._db_path)
        out: dict[str, list[tuple]] = {}
        try:
            for token in tokens:
                rows = conn.execute(
                    "SELECT ts, bid, ask, bid_size, ask_size FROM ticks "
                    "WHERE token=? ORDER BY ts DESC LIMIT ?",
                    (token, limit_per_token)).fetchall()
                if rows:
                    out[token] = [tuple(r) for r in reversed(rows)]
        except sqlite3.OperationalError:
            return {}
        finally:
            conn.close()
        return out

    # --- writer thread ---

    def flush_now(self) -> None:
        """Synchronous drain+write on the CALLER's connection — tests/shutdown."""
        conn = sqlite3.connect(self._db_path)
        try:
            conn.executescript(_SCHEMA)
            self._drain_into(conn)
            conn.commit()
        finally:
            conn.close()

    def _drain_into(self, conn: sqlite3.Connection) -> int:
        ticks, cats = [], []
        while True:
            try:
                row = self._queue.get_nowait()
            except queue.Empty:
                break
            if row[0] == "tick":
                ticks.append(row[1:])
            else:
                cats.append(row[1:])
        if ticks:
            conn.executemany(
                "INSERT INTO ticks (ts, token, bid, ask, bid_size, ask_size) "
                "VALUES (?, ?, ?, ?, ?, ?)", ticks)
        if cats:
            conn.executemany(
                "INSERT INTO cat_index (ts, category, value, n) "
                "VALUES (?, ?, ?, ?)", cats)
        return len(ticks) + len(cats)

    def _prune(self, conn: sqlite3.Connection, now: float) -> None:
        if now - self._last_prune < 3600.0:
            return
        self._last_prune = now
        cutoff = now - self._retention_sec
        conn.execute("DELETE FROM ticks WHERE ts < ?", (cutoff,))
        conn.execute("DELETE FROM cat_index WHERE ts < ?", (cutoff,))

    def _run(self) -> None:  # pragma: no cover — thin loop over tested parts
        conn = sqlite3.connect(self._db_path)
        conn.executescript(_SCHEMA)
        while not self._halt.is_set():
            self._halt.wait(self._flush_sec)
            try:
                self._drain_into(conn)
                self._prune(conn, time.time())
                conn.commit()
            except Exception:
                log.exception("tick writer")
        self._drain_into(conn)
        conn.commit()
        conn.close()

    def close(self) -> None:
        self._halt.set()
        if self._writer.is_alive():
            self._writer.join(timeout=self._flush_sec + 2)
