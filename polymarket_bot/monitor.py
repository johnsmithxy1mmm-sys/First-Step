"""Monitoring: rich terminal dashboard + optional Telegram alerts.

alert() must never block a hot path (the WS tick thread, the MM lock): it only
enqueues; a daemon worker thread does the actual HTTP send with its timeout.
"""

from __future__ import annotations

import logging
import os
import queue
import threading

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


def alert(text: str) -> bool:
    """Queue a Telegram alert (non-blocking); False if TELEGRAM_* vars unset."""
    if not os.environ.get("TELEGRAM_BOT_TOKEN") or not os.environ.get("TELEGRAM_CHAT_ID"):
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
