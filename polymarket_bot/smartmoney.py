"""Трекер «умных денег» (уровень 2, №5): следим за сильными кошельками.

Исторически громкие частные плюсы на Polymarket — люди, знавшие домен глубже
толпы. Их сделки видны публично: Data API отдаёт позиции любого адреса. Модуль
следит за списком таких кошельков и алертит, когда они **впервые** заходят в
рынок — с пометкой, если это хвостовой исход (дёшево, потенциал иксов) или
рынок из вашей ниши.

Только алерты, без автоторговли: копировать вслепую нельзя (разный размер
банка, разное время входа), но знать, куда зашли сильные, — сильный сигнал
для ручного решения и для лонгшот-оценщика.

Где взять адреса: leaderboard на polymarket.com/leaderboard — скопируйте
адреса стабильно прибыльных игроков в smart_money.watch_wallets.
"""

from __future__ import annotations

import logging

import httpx
from pydantic import BaseModel

from .config import BotConfig
from .http_client import get_with_backoff, make_client
from .ledger import Ledger
from .monitor import alert
from .niche import NicheWatcher

log = logging.getLogger(__name__)


class WalletPosition(BaseModel):
    wallet: str
    asset: str                 # token id
    title: str
    outcome: str
    size: float
    avg_price: float
    cur_price: float
    cash_pnl: float

    @property
    def usd(self) -> float:
        return self.size * self.avg_price

    @property
    def key(self) -> str:
        return f"{self.wallet}:{self.asset}"


def parse_positions(wallet: str, raw: list) -> list[WalletPosition]:
    """Дефенсивный разбор ответа Data API /positions (форма может отличаться)."""
    out: list[WalletPosition] = []
    for p in raw or []:
        if not isinstance(p, dict):
            continue
        asset = str(p.get("asset") or p.get("tokenId") or "")
        if not asset:
            continue
        try:
            out.append(WalletPosition(
                wallet=wallet, asset=asset,
                title=str(p.get("title") or p.get("market") or "?"),
                outcome=str(p.get("outcome") or ""),
                size=float(p.get("size") or 0),
                avg_price=float(p.get("avgPrice") or 0),
                cur_price=float(p.get("curPrice") or 0),
                cash_pnl=float(p.get("cashPnl") or 0),
            ))
        except (TypeError, ValueError):
            continue
    return out


class SmartMoneyTracker:
    def __init__(self, cfg: BotConfig, ledger: Ledger, niche: NicheWatcher,
                 client: httpx.Client | None = None):
        self._cfg = cfg.smart_money
        self._data_host = cfg.runtime.data_api_host
        self._max_retries = cfg.runtime.max_retries
        self._ledger = ledger
        self._niche = niche
        self._client = client or make_client(cfg.runtime.request_timeout_sec)

    def fetch_positions(self, wallet: str) -> list[WalletPosition]:
        try:
            resp = get_with_backoff(
                self._client, f"{self._data_host}/positions",
                params={"user": wallet, "limit": 500}, max_retries=2)
        except httpx.HTTPError as exc:
            log.warning("smart-money: %s -> %s", wallet[:10], exc)
            return []
        return parse_positions(wallet, resp.json())

    def _is_tail(self, pos: WalletPosition) -> bool:
        price = pos.avg_price or pos.cur_price
        return 0 < price <= self._cfg.tail_max_price

    def _describe(self, pos: WalletPosition, niche: str | None) -> str:
        tags = []
        if self._is_tail(pos):
            tags.append("ХВОСТ")
        if niche:
            tags.append(f"ниша:{niche}")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        return (f"🐋 УМНЫЕ ДЕНЬГИ{tag_str}\n"
                f"Кошелёк {pos.wallet[:10]}… зашёл: {pos.title[:80]}\n"
                f"Исход [{pos.outcome}] по {pos.avg_price:.3f} | "
                f"${pos.usd:,.0f} | PnL кошелька по позиции ${pos.cash_pnl:,.0f}\n"
                f"https://polymarket.com/profile/{pos.wallet}")

    def cycle(self) -> list[WalletPosition]:
        """Проверяет отслеживаемые кошельки, алертит про новые позиции."""
        if not self._cfg.enabled or not self._cfg.watch_wallets:
            return []
        seen = self._ledger.smart_money_seen_keys()
        alerted: list[WalletPosition] = []
        fresh_keys: list[str] = []

        for wallet in self._cfg.watch_wallets:
            for pos in self.fetch_positions(wallet):
                if pos.key in seen or pos.usd < self._cfg.min_position_usd:
                    continue
                fresh_keys.append(pos.key)   # помечаем виденной независимо от алерта
                niche = self._niche.classify(pos.title)
                if self._cfg.only_niche_or_tail and not (niche or self._is_tail(pos)):
                    continue
                alerted.append(pos)
                text = self._describe(pos, niche)
                log.info(text)
                alert(text)

        if fresh_keys:
            self._ledger.mark_smart_money_seen(fresh_keys)
        return alerted
