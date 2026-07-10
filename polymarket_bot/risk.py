"""Риск-фреймворк: абсолютные лимиты и kill-switch (жёсткие требования).

Kill-switch срабатывает автоматически при: превышении риск-лимита
(дневной убыток, просадка от HWM), WS-disconnect дольше порога,
расхождении локального состояния ордеров с биржевым (reconcile),
ошибках подписи. Действие: отменить все ордера, уведомить, остановить
стратегии.

Два уровня:
  PAUSE — временная блокировка новых ордеров (WS-outage): автоснятие
          после восстановления потока;
  HALT  — полная остановка торговли до ручного рестарта (дневной стоп,
          просадка, рассинхрон, подпись).
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
        self.paused = False
        self.reason = ""

    # --- состояние ---

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
        self._alert(f"KILL-SWITCH (halt до ручного рестарта): {reason}")

    def trip_pause(self, reason: str) -> None:
        if self.paused or self.halted:
            return
        self.paused = True
        self.reason = reason
        log.warning("KILL-SWITCH PAUSE: %s", reason)
        self._safe_cancel()
        self._alert(f"Пауза торговли (авто-возврат): {reason}")

    def resume_from_pause(self) -> None:
        if self.paused and not self.halted:
            self.paused = False
            log.info("kill-switch: пауза снята, торговля возобновлена")

    def _safe_cancel(self) -> None:
        try:
            self._cancel_all()
        except Exception:
            log.exception("kill-switch: bulk-cancel")

    # --- проверки (вызываются каждый цикл) ---

    def check_daily_loss(self, equity_now: float) -> None:
        """Дневной стоп: просадка от equity на начало суток UTC."""
        day_start = self._day_start_equity(equity_now)
        loss = day_start - equity_now
        if loss >= self._cfg.max_daily_loss_usd:
            self.trip_halt(f"дневной убыток ${loss:,.2f} >= "
                           f"${self._cfg.max_daily_loss_usd:,.2f}")

    def _day_start_equity(self, fallback: float) -> float:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row = self._ledger.first_bank_equity_since(f"{today}T00:00:00")
        return row if row is not None else fallback

    def check_drawdown(self, equity_now: float, hwm: float) -> None:
        if hwm > 0 and (hwm - equity_now) / hwm >= self._cfg.max_drawdown_pct:
            self.trip_halt(f"просадка {((hwm - equity_now) / hwm) * 100:.1f}% "
                           f">= {self._cfg.max_drawdown_pct * 100:.0f}% от HWM")

    def check_global_exposure(self, exposure_usd: float) -> bool:
        """True = лимит общей экспозиции исчерпан (новые входы запрещены)."""
        return exposure_usd >= self._cfg.max_global_exposure_usd

    def reconcile(self, local_order_ids: set[str], exchange_order_ids: set[str]) -> None:
        """Рассинхрон стейта с биржей — торговать нельзя, состояние ненадёжно."""
        ghost = exchange_order_ids - local_order_ids
        if ghost:
            self.trip_halt(f"reconcile: на бирже {len(ghost)} неизвестных ордеров "
                           f"(первый: {next(iter(ghost))[:20]})")

    def on_ws_disconnect(self, gap_sec: float) -> None:
        self.trip_pause(f"WS-поток мёртв {gap_sec:.0f}с")

    def on_ws_recovered(self) -> None:
        self.resume_from_pause()

    def on_signature_error(self, exc: Exception) -> None:
        self.trip_halt(f"ошибка подписи ордера: {exc}")
