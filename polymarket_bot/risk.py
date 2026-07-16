"""Risk framework: absolute limits and kill-switch (hard requirements).

The kill-switch trips automatically on: a risk-limit breach (daily loss,
drawdown from HWM), a WS disconnect longer than the threshold, a mismatch
between local order state and the exchange (reconcile), signature errors.
Action: cancel all orders, notify, stop the strategies.

Two levels:
  PAUSE — temporary block on new orders (WS outage): auto-cleared once the
          stream recovers;
  HALT  — full trading stop until a manual restart (daily stop, drawdown,
          desync, signature).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable

from .config import BotConfig
from .ledger import Ledger

log = logging.getLogger(__name__)


class KillSwitch:
    def __init__(self, cfg: BotConfig, ledger: Ledger, mode: str,
                 cancel_all: Callable[[], None], alert: Callable[[str], bool]):
        self._cfg = cfg.risk
        self._ledger = ledger
        self._mode = mode
        self._cancel_all = cancel_all
        self._alert = alert
        self.halted = False
        # Pause sources are independent: "ws" (stream outage) and "data" (feed
        # anomaly). Each resumes only from its own source, so a healthy WS
        # cannot clear a data-anomaly pause.
        self._pause_sources: set[str] = set()
        self.reason = ""

    # --- state ---

    @property
    def paused(self) -> bool:
        return bool(self._pause_sources)

    @property
    def trading_allowed(self) -> bool:
        return not self.halted and not self.paused

    def trip_halt(self, reason: str) -> None:
        if self.halted:
            return
        self.halted = True
        self.reason = reason
        log.error("KILL-SWITCH HALT: %s", reason)
        self._safe_cancel()
        self._alert(f"KILL-SWITCH (halt until manual restart): {reason}")

    def trip_pause(self, reason: str, source: str = "ws") -> None:
        if self.halted or source in self._pause_sources:
            return
        first = not self.paused
        self._pause_sources.add(source)
        self.reason = reason
        log.warning("KILL-SWITCH PAUSE [%s]: %s", source, reason)
        if first:
            self._safe_cancel()
            self._alert(f"Trading paused (auto-resume): {reason}")

    def resume_from_pause(self, source: str = "ws") -> None:
        if source in self._pause_sources and not self.halted:
            self._pause_sources.discard(source)
            if not self._pause_sources:
                log.info("kill-switch: pause lifted, trading resumed")

    def _safe_cancel(self) -> None:
        try:
            self._cancel_all()
        except Exception:
            log.exception("kill-switch: bulk-cancel")

    # --- checks (called every cycle) ---

    def check_daily_loss(self, equity_now: float) -> None:
        """Daily stop: drawdown from equity at the start of the UTC day."""
        day_start = self._day_start_equity(equity_now)
        loss = day_start - equity_now
        if loss >= self._cfg.max_daily_loss_usd:
            self.trip_halt(f"daily loss ${loss:,.2f} >= "
                           f"${self._cfg.max_daily_loss_usd:,.2f}")

    def _day_start_equity(self, fallback: float) -> float:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row = self._ledger.first_bank_equity_since(f"{today}T00:00:00")
        return row if row is not None else fallback

    def check_drawdown(self, equity_now: float, hwm: float) -> None:
        if hwm > 0 and (hwm - equity_now) / hwm >= self._cfg.max_drawdown_pct:
            self.trip_halt(f"drawdown {((hwm - equity_now) / hwm) * 100:.1f}% "
                           f">= {self._cfg.max_drawdown_pct * 100:.0f}% from HWM")

    def check_global_exposure(self, exposure_usd: float) -> bool:
        """True = total exposure limit reached (new entries forbidden)."""
        return exposure_usd >= self._cfg.max_global_exposure_usd

    def reconcile(self, local_order_ids: set[str], exchange_order_ids: set[str]) -> None:
        """State desync with the exchange — cannot trade, state is unreliable."""
        ghost = exchange_order_ids - local_order_ids
        if ghost:
            self.trip_halt(f"reconcile: {len(ghost)} unknown orders on the exchange "
                           f"(first: {next(iter(ghost))[:20]})")

    def on_ws_disconnect(self, gap_sec: float) -> None:
        self.trip_pause(f"WS stream dead for {gap_sec:.0f}s")

    def on_ws_recovered(self) -> None:
        self.resume_from_pause()

    def on_signature_error(self, exc: Exception) -> None:
        self.trip_halt(f"order signature error: {exc}")
