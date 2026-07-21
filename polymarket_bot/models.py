"""Pydantic data models: markets, candidates, signals, estimates, trade plans."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _json_list(value: Any) -> list:
    """Gamma returns outcomes/outcomePrices/clobTokenIds as JSON-encoded strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _num(raw: dict, *keys: str) -> float:
    for key in keys:
        value = raw.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _normalize_spread(value: float) -> float:
    """Gamma returns rewardsMaxSpread in cents (3.5) — normalize to a probability."""
    return value / 100.0 if value > 1.0 else value


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


class Market(BaseModel):
    """A single binary Polymarket market (Yes/No) with event context."""

    model_config = ConfigDict(frozen=False)

    id: str
    question: str
    slug: str = ""
    description: str = ""
    category: str = ""
    outcomes: list[str] = Field(default_factory=list)
    outcome_prices: list[float] = Field(default_factory=list)
    clob_token_ids: list[str] = Field(default_factory=list)
    liquidity_usd: float = 0.0
    volume_usd: float = 0.0
    volume_24h_usd: float = 0.0
    one_day_price_change: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    end_date: datetime | None = None
    neg_risk: bool = False
    enable_order_book: bool = True
    tick_size: float = 0.001
    min_order_size: float = 5.0
    resolution_source: str = ""
    closed: bool = False
    # UMA oracle state as reported by Gamma (e.g. "proposed", "challenged",
    # "resolved", "" when absent/unknown). "proposed"/"resolved" means an
    # oracle answer EXISTS on-chain — a fact, unlike the price/volume
    # imminence heuristic.
    uma_resolution_status: str = ""
    # CTF condition id — required to redeem winnings on-chain.
    condition_id: str = ""
    # Liquidity Rewards Program: params come from Gamma — do not hardcode.
    rewards_min_size: float = 0.0     # minimum quote size for rewards
    rewards_max_spread: float = 0.0   # max deviation from midpoint (as probability)
    # Event context (market groups) — needed for cross-market coherence.
    event_id: str = ""
    event_title: str = ""
    event_neg_risk: bool = False

    @classmethod
    def from_gamma(cls, raw: dict, event: dict | None = None) -> "Market | None":
        outcomes = [str(o) for o in _json_list(raw.get("outcomes"))]
        prices_raw = _json_list(raw.get("outcomePrices"))
        token_ids = [str(t) for t in _json_list(raw.get("clobTokenIds"))]
        try:
            prices = [float(p) for p in prices_raw]
        except (TypeError, ValueError):
            return None
        if not (len(outcomes) == len(prices) == len(token_ids)) or not outcomes:
            return None
        event = event or {}
        return cls(
            id=str(raw.get("id", "")),
            question=raw.get("question") or "",
            slug=raw.get("slug") or "",
            description=raw.get("description") or "",
            category=raw.get("category") or "",
            outcomes=outcomes,
            outcome_prices=prices,
            clob_token_ids=token_ids,
            liquidity_usd=_num(raw, "liquidityNum", "liquidity"),
            volume_usd=_num(raw, "volumeNum", "volume"),
            volume_24h_usd=_num(raw, "volume24hr", "volume24hrClob"),
            one_day_price_change=_num(raw, "oneDayPriceChange"),
            best_bid=_num(raw, "bestBid"),
            best_ask=_num(raw, "bestAsk"),
            end_date=_parse_dt(raw.get("endDate")),
            neg_risk=bool(raw.get("negRisk", False)),
            enable_order_book=bool(raw.get("enableOrderBook", True)),
            tick_size=_num(raw, "orderPriceMinTickSize") or 0.001,
            min_order_size=_num(raw, "orderMinSize") or 5.0,
            resolution_source=raw.get("resolutionSource") or "",
            closed=bool(raw.get("closed", False)),
            uma_resolution_status=str(raw.get("umaResolutionStatus") or "").lower(),
            condition_id=str(raw.get("conditionId") or ""),
            rewards_min_size=_num(raw, "rewardsMinSize"),
            rewards_max_spread=_normalize_spread(_num(raw, "rewardsMaxSpread")),
            event_id=str(event.get("id", "")),
            event_title=event.get("title") or "",
            event_neg_risk=bool(event.get("negRisk", False)),
        )

    @property
    def in_rewards_program(self) -> bool:
        return self.rewards_min_size > 0 and self.rewards_max_spread > 0

    def days_to_resolution(self, now: datetime | None = None) -> float | None:
        if self.end_date is None:
            return None
        now = now or datetime.now(timezone.utc)
        return (self.end_date - now).total_seconds() / 86400.0

    def resolved_winner_index(self) -> int | None:
        """For closed markets: index of the winning outcome by final prices."""
        if not self.closed or not self.outcome_prices:
            return None
        best = max(self.outcome_prices)
        if best < 0.95:  # resolution is ambiguous / market voided
            return None
        return self.outcome_prices.index(best)


class BookLevel(BaseModel):
    price: float
    size: float


class OrderBook(BaseModel):
    bids: list[BookLevel] = Field(default_factory=list)
    asks: list[BookLevel] = Field(default_factory=list)

    @property
    def best_bid(self) -> float:
        return max((l.price for l in self.bids if l.size > 0), default=0.0)

    @property
    def best_ask(self) -> float:
        return min((l.price for l in self.asks if l.size > 0), default=0.0)

    @property
    def mid(self) -> float:
        bid, ask = self.best_bid, self.best_ask
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        return bid or ask

    def bid_depth_usd_within(self, pct_from_mid: float) -> float:
        """Dollar depth of bids within pct of mid — support on our side."""
        mid = self.mid
        if mid <= 0:
            return 0.0
        floor_price = mid * (1.0 - pct_from_mid)
        return sum(l.price * l.size for l in self.bids if l.price >= floor_price)


class Candidate(BaseModel):
    """A cheap outcome that passed the first-level filters."""

    market: Market
    outcome_index: int
    token_id: str
    p_mkt: float                      # market price = implied probability
    book: OrderBook | None = None

    @property
    def outcome(self) -> str:
        return self.market.outcomes[self.outcome_index]

    @property
    def payout_multiple(self) -> float:
        return 1.0 / self.p_mkt if self.p_mkt > 0 else 0.0


class Signal(BaseModel):
    """Result from a single probability-estimation source."""

    name: str
    p_est: float | None = None        # None = the signal abstained
    confidence: float = 0.0           # 0..1, used as the weight in the ensemble
    rationale: str = ""


class Estimate(BaseModel):
    """Final ensemble estimate for a candidate."""

    candidate: Candidate
    p_mkt: float
    p_est: float
    signals: list[Signal] = Field(default_factory=list)

    @property
    def edge_ratio(self) -> float:
        return self.p_est / self.p_mkt if self.p_mkt > 0 else 0.0

    def signals_dump(self) -> str:
        return json.dumps(
            [s.model_dump() for s in self.signals], ensure_ascii=False, default=str
        )


class TradePlan(BaseModel):
    """A trade approved by the portfolio module."""

    estimate: Estimate
    category: str
    size_usd: float
    limit_price_cap: float            # above this price the edge vanishes — do not pay more

    @property
    def token_id(self) -> str:
        return self.estimate.candidate.token_id


class ExecutionResult(BaseModel):
    status: str                       # filled | resting | skipped | canceled | failed
    filled_size: float = 0.0
    avg_price: float = 0.0
    order_ids: list[str] = Field(default_factory=list)
    detail: str = ""


def simple_estimate(market: Market, outcome_index: int, price: float) -> Estimate:
    """Minimal estimate for non-longshot strategy trades (arbitrage, MM)."""
    return Estimate(
        candidate=Candidate(
            market=market,
            outcome_index=outcome_index,
            token_id=market.clob_token_ids[outcome_index],
            p_mkt=price,
        ),
        p_mkt=price,
        p_est=price,
        signals=[],
    )


class Position(BaseModel):
    token_id: str
    market_id: str
    question: str
    outcome: str
    category: str
    size: float
    avg_price: float
    event_id: str = ""
    neg_risk: bool = False       # part of a mutually-exclusive (neg-risk) event

    @property
    def cost_usd(self) -> float:
        return self.size * self.avg_price
