"""Monitoring: rich terminal dashboard + optional Telegram alerts.

alert() must never block a hot path (the WS tick thread, the MM lock): it only
enqueues; a daemon worker thread does the actual HTTP send with its timeout.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time

from datetime import datetime, timezone

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .models import Estimate, Position

log = logging.getLogger(__name__)

_queue: "queue.Queue[str]" = queue.Queue()
_worker_lock = threading.Lock()
_worker_started = False

# Alert dedupe: a repeating OPPORTUNITY (an arb window seen every cycle) should
# not re-notify every time. Keyed by a stable structure id -> (last_sent_ts,
# last_value); a repeat within cooldown is suppressed unless its tracked value
# (e.g. net edge) moved by at least min_change. One-off events (fills, entries,
# risk trips) pass no key and are never suppressed.
_dedupe_lock = threading.Lock()
_dedupe: "dict[str, tuple[float, float | None]]" = {}


def _reset_alert_state() -> None:
    """Test hook: forget dedupe history."""
    with _dedupe_lock:
        _dedupe.clear()


def _should_send(key: str | None, cooldown_sec: float, value: float | None,
                 min_change: float | None, now: float) -> bool:
    if not key or cooldown_sec <= 0:
        return True
    with _dedupe_lock:
        prev = _dedupe.get(key)
        if prev is not None and now - prev[0] < cooldown_sec:
            unchanged = (value is None or min_change is None or prev[1] is None
                         or abs(value - prev[1]) < min_change)
            if unchanged:
                return False        # within cooldown, nothing material changed
        if len(_dedupe) > 5000:     # bound the map: drop entries past a day
            cutoff = now - 86_400
            for k in [k for k, v in _dedupe.items() if v[0] < cutoff]:
                _dedupe.pop(k, None)
        _dedupe[key] = (now, value)
        return True


def _send(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    try:
        resp = httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text[:4000]},
            timeout=10,
        )
        return resp.status_code == 200
    except httpx.HTTPError as exc:
        log.warning("telegram alert failed: %s", exc)
        return False


def _drain() -> None:
    while True:
        text = _queue.get()
        try:
            _send(text)
        except Exception:
            log.exception("telegram worker")
        finally:
            _queue.task_done()


def alert(text: str, *, key: str | None = None, cooldown_sec: float = 0.0,
          value: float | None = None, min_change: float | None = None) -> bool:
    """Queue a Telegram alert (non-blocking); False if TELEGRAM_* vars unset.

    Pass `key` + `cooldown_sec` to dedupe a repeating opportunity: a repeat of
    the same key within the window is dropped unless `value` moved by at least
    `min_change` since the last send. One-off events pass no key.
    """
    if not os.environ.get("TELEGRAM_BOT_TOKEN") or not os.environ.get("TELEGRAM_CHAT_ID"):
        return False
    if not _should_send(key, cooldown_sec, value, min_change, time.time()):
        return False
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            threading.Thread(target=_drain, daemon=True, name="tg-alerts").start()
            _worker_started = True
    _queue.put(text)
    return True


class Dashboard:
    def __init__(self) -> None:
        self._console = Console()

    def render(self, *, mode: str, equity: float, drawdown: float,
               observe_only: bool, positions: list[Position],
               marks: dict[str, float], top_estimates: list[Estimate],
               errors: list[str]) -> None:
        c = self._console
        status = "[red]OBSERVE-ONLY (kill-switch)[/red]" if observe_only else "[green]active[/green]"
        c.print(Panel(
            f"mode=[bold]{mode}[/bold]  equity=[bold]${equity:,.2f}[/bold]  "
            f"drawdown=[bold]{drawdown * 100:.1f}%[/bold]  status={status}  "
            f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            title="Polymarket Longshot Bot",
        ))

        if positions:
            t = Table(title=f"Open positions ({len(positions)})")
            for col in ("Category", "Outcome / Question", "Size", "Entry", "Now", "x"):
                t.add_column(col)
            for p in sorted(positions, key=lambda p: p.cost_usd, reverse=True)[:20]:
                mark = marks.get(p.token_id, 0.0)
                mult = mark / p.avg_price if p.avg_price > 0 and mark > 0 else 0.0
                t.add_row(p.category, f"[{p.outcome}] {p.question[:55]}",
                          f"{p.size:,.0f}", f"{p.avg_price:.4f}",
                          f"{mark:.4f}" if mark else "-",
                          f"{mult:.1f}x" if mult else "-")
            c.print(t)

        if top_estimates:
            t = Table(title="Top candidates by edge")
            for col in ("edge", "p_mkt", "p_est", "Signals", "Question"):
                t.add_column(col)
            for e in top_estimates[:10]:
                signal_names = ",".join(s.name for s in e.signals if s.name != "market")
                t.add_row(f"{e.edge_ratio:.2f}", f"{e.p_mkt:.4f}", f"{e.p_est:.4f}",
                          signal_names or "-", e.candidate.market.question[:55])
            c.print(t)

        if errors:
            c.print(Panel("\n".join(errors[-8:]), title="[red]Cycle errors[/red]"))
