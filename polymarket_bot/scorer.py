"""Market selection for MM: filters + scoring (per the master prompt).

Filters: 24h volume above threshold; resolution far enough out; market in
the rewards program; probability stable (daily midpoint move below
threshold); spread >= 2x tick; resolution rules unambiguous (UMA risk:
subjective criteria are flagged).

Scoring: rewards attractiveness / maker competition (estimated from book
depth around the midpoint). Priority — sports with distant resolution and
politics, NOT short-dated crypto markets.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .config import BotConfig
from .models import Market, OrderBook
from .portfolio import classify_category
from .rewards import reward_share

log = logging.getLogger(__name__)

# Subjective resolution criteria — risk of UMA disputes.
SUBJECTIVE_RESOLUTION_MARKERS = (
    "in the opinion", "significant", "substantial", "credible", "widely reported",
    "generally accepted", "commonly understood", "at the discretion", "majority of",
    "notable", "major news outlets",
)

CATEGORY_BOOST = {
    "sports": 1.5,
    "elections": 1.4,      # politics (midterms)
    "economy": 1.0,
    "geopolitics": 0.9,
    "nature": 0.9,
    "crypto": 0.5,         # short-dated crypto markets — not our MM profile
    "other": 1.0,
}


class MarketScorer:
    def __init__(self, cfg: BotConfig):
        self._cfg = cfg.market_maker

    def uma_risk(self, market: Market) -> bool:
        text = market.description.lower()
        return any(marker in text for marker in SUBJECTIVE_RESOLUTION_MARKERS)

    def reject_reason(self, m: Market, now: datetime | None = None) -> str | None:
        """None = market eligible for MM; else a reason string (for diagnostics)."""
        c = self._cfg
        if m.closed or not m.enable_order_book or len(m.clob_token_ids) < 2:
            return "no book / closed"
        if m.volume_24h_usd < c.min_volume_24h_usd:
            return f"24h volume < ${c.min_volume_24h_usd:,.0f}"
        days = m.days_to_resolution(now or datetime.now(timezone.utc))
        if days is None or days < c.min_days_to_resolution:
            return f"resolution < {c.min_days_to_resolution:.0f}d away"
        if c.require_rewards_program and not m.in_rewards_program:
            return "not in rewards program"
        if abs(m.one_day_price_change) > c.max_daily_midpoint_move:
            return "volatile (midpoint move > threshold)"
        if m.best_bid > 0 and m.best_ask > 0 \
                and (m.best_ask - m.best_bid) < 2 * m.tick_size:
            return "spread < 2 ticks"
        if not m.resolution_source and len(m.description.strip()) < 80:
            return "unclear resolution rules"
        if self.uma_risk(m):
            return "UMA risk (subjective resolution)"
        return None

    def eligible(self, m: Market, now: datetime | None = None) -> bool:
        return self.reject_reason(m, now) is None

    def score(self, m: Market, book: OrderBook | None) -> float:
        """Expected share of this market's reward pool, times a category boost.

        The competition term is no longer notional depth inside a flat band but
        the actual Q_min resting in the book under the published quadratic
        scoring rule (see rewards.py). That distinction decides where a small
        quote is worth placing: notional depth treats a wall sitting near the
        band EDGE as heavy competition, while the real rule scores it at a few
        percent — so a market that looks crowded can in fact be wide open to a
        quote placed near the midpoint.

        Turnover still enters, but only as a tie-break: reward share is the
        thing being maximised, and volume proxies how much spread income and
        fill flow rides along with it.
        """
        category = classify_category(m.question, m.category)
        boost = CATEGORY_BOOST.get(category, 1.0)
        mid = book.mid if book is not None else 0.0
        if mid <= 0 or m.rewards_max_spread <= 0:
            # Not scoreable for rewards (no book or not in the program): fall
            # back to turnover alone so such markets rank below real candidates.
            return boost * m.volume_24h_usd * 1e-9
        size = self._cfg.quote_size_usd / max(mid, 0.01)
        half = self._reward_half_spread(m)
        share = reward_share(size=size, half_spread=half,
                             max_spread=m.rewards_max_spread, mid=mid, book=book)
        # Tie-break on turnover: among equal reward shares prefer the market
        # that also pays spread and fills.
        return boost * share * (1.0 + m.volume_24h_usd * 1e-6)

    def _reward_half_spread(self, m: Market) -> float:
        """Distance from mid we would realistically quote at, for scoring."""
        half = self._cfg.half_spread
        if m.rewards_max_spread > 0:
            half = min(half, m.rewards_max_spread * 0.9)
        return max(half, m.tick_size)

    def top_markets(self, markets: list[Market],
                    books: dict[str, OrderBook | None]) -> list[Market]:
        eligible = [m for m in markets if self.eligible(m)]
        scored = sorted(
            eligible,
            key=lambda m: self.score(m, books.get(m.clob_token_ids[0])),
            reverse=True,
        )
        log.info("scorer: %d markets eligible for MM, taking top %d",
                 len(eligible), self._cfg.max_markets)
        return scored[: self._cfg.max_markets]
