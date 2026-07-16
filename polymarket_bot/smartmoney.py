"""Smart-money tracker (level 2, #5): follow strong wallets.

Historically loud private winners on Polymarket are people who knew a domain
deeper than the crowd. Their trades are public: the Data API returns any
address's positions. This module watches a list of such wallets and alerts
when they enter a market for the **first** time — flagging it if it's a tail
outcome (cheap, multi-x potential) or a market in your niche.

Alerts only, no auto-trading: blind copying is unwise (different bankroll,
different entry timing), but knowing where the strong players went is a
powerful input for a manual decision and for the longshot estimator.

Where to get addresses: the leaderboard at polymarket.com/leaderboard — copy
the addresses of consistently profitable players into smart_money.watch_wallets.
"""

from __future__ import annotations

import logging

import httpx
from pydantic import BaseModel

from .config import BotConfig
from .http_client import get_with_backoff, make_client
from .ledger import Ledger
from .models import Candidate, Signal
from .monitor import alert
from .niche import NicheWatcher

log = logging.getLogger(__name__)


class SmartMoneySignal:
    """Ensemble signal: a profitable watched wallet holding a tail nudges p_est up.

    Not blind copying — a bounded contribution to the estimate, weighted by the
    wallet's realized PnL. Hot-token map is refreshed by the smart-money tracker.
    """

    name = "smart_money"

    def __init__(self, max_confidence: float = 0.40):
        self._max = max_confidence
        self._hot: dict[str, float] = {}     # token_id -> confidence

    def set_hot(self, hot: dict[str, float]) -> None:
        self._hot = hot

    def evaluate(self, candidate: Candidate) -> Signal | None:
        conf = self._hot.get(candidate.token_id)
        if not conf:
            return None
        p_est = min(candidate.p_mkt * 1.5, 0.999)   # they see value the crowd doesn't
        return Signal(name=self.name, p_est=p_est,
                      confidence=min(conf, self._max),
                      rationale="a profitable watched wallet holds this token")


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
    """Defensive parse of the Data API /positions response (shape may vary)."""
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
            tags.append("TAIL")
        if niche:
            tags.append(f"niche:{niche}")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        return (f"🐋 SMART MONEY{tag_str}\n"
                f"Wallet {pos.wallet[:10]}… entered: {pos.title[:80]}\n"
                f"Outcome [{pos.outcome}] at {pos.avg_price:.3f} | "
                f"${pos.usd:,.0f} | wallet PnL on this position ${pos.cash_pnl:,.0f}\n"
                f"https://polymarket.com/profile/{pos.wallet}")

    def snapshot(self) -> dict[str, list[WalletPosition]]:
        """One Data API pass over the watched wallets (shared by cycle + signal)."""
        return {w: self.fetch_positions(w) for w in self._cfg.watch_wallets}

    def cycle(self, snapshot: dict[str, list[WalletPosition]] | None = None
              ) -> list[WalletPosition]:
        """Checks the watched wallets, alerts on new positions."""
        if not self._cfg.enabled or not self._cfg.watch_wallets:
            return []
        snapshot = snapshot if snapshot is not None else self.snapshot()
        seen = self._ledger.smart_money_seen_keys()
        alerted: list[WalletPosition] = []
        fresh_keys: list[str] = []

        for wallet in self._cfg.watch_wallets:
            for pos in snapshot.get(wallet, []):
                if pos.key in seen or pos.usd < self._cfg.min_position_usd:
                    continue
                fresh_keys.append(pos.key)   # mark seen regardless of alerting
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

    def build_hot_tokens(self, snapshot: dict[str, list[WalletPosition]] | None = None
                         ) -> dict[str, float]:
        """{token_id: confidence} for tails held by profitable watched wallets."""
        if not self._cfg.as_signal or not self._cfg.watch_wallets:
            return {}
        snapshot = snapshot if snapshot is not None else self.snapshot()
        hot: dict[str, float] = {}
        for wallet in self._cfg.watch_wallets:
            positions = snapshot.get(wallet, [])
            wallet_pnl = sum(p.cash_pnl for p in positions)
            if wallet_pnl < self._cfg.signal_min_pnl_usd:
                continue
            weight = min(self._cfg.signal_max_confidence,
                         0.10 + (wallet_pnl / 50_000.0) * 0.30)
            for p in positions:
                if self._is_tail(p) and p.usd >= self._cfg.min_position_usd:
                    hot[p.asset] = max(hot.get(p.asset, 0.0), weight)
        return hot
