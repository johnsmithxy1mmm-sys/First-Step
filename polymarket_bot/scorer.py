"""Отбор рынков для MM: фильтры + скоринг (по мастер-промпту).

Фильтры: дневной объём выше порога; до резолюции достаточно далеко;
рынок в rewards-программе; вероятность стабильна (дневное движение
midpoint ниже порога); спред >= 2x тика; правила резолюции однозначны
(UMA-риск: субъективные критерии во флаг).

Скоринг: привлекательность rewards / конкуренция мейкеров (оценка по
глубине стакана возле midpoint). Приоритет — спорт с далёкой резолюцией
и политика, НЕ короткие крипторынки.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .config import BotConfig
from .models import Market, OrderBook
from .portfolio import classify_category

log = logging.getLogger(__name__)

# Субъективные критерии резолюции — риск UMA-споров.
SUBJECTIVE_RESOLUTION_MARKERS = (
    "in the opinion", "significant", "substantial", "credible", "widely reported",
    "generally accepted", "commonly understood", "at the discretion", "majority of",
    "notable", "major news outlets",
)

CATEGORY_BOOST = {
    "sports": 1.5,
    "elections": 1.4,      # политика (midterms)
    "economy": 1.0,
    "geopolitics": 0.9,
    "nature": 0.9,
    "crypto": 0.5,         # короткие крипторынки — не наш профиль для MM
    "other": 1.0,
}


class MarketScorer:
    def __init__(self, cfg: BotConfig):
        self._cfg = cfg.market_maker

    def uma_risk(self, market: Market) -> bool:
        text = market.description.lower()
        return any(marker in text for marker in SUBJECTIVE_RESOLUTION_MARKERS)

    def reject_reason(self, m: Market, now: datetime | None = None) -> str | None:
        """None = рынок пригоден для MM; иначе строка-причина отсева (для диагностики)."""
        c = self._cfg
        if m.closed or not m.enable_order_book or len(m.clob_token_ids) < 2:
            return "нет стакана / закрыт"
        if m.volume_24h_usd < c.min_volume_24h_usd:
            return f"объём 24h < ${c.min_volume_24h_usd:,.0f}"
        days = m.days_to_resolution(now or datetime.now(timezone.utc))
        if days is None or days < c.min_days_to_resolution:
            return f"до резолюции < {c.min_days_to_resolution:.0f} дн."
        if c.require_rewards_program and not m.in_rewards_program:
            return "не в rewards-программе"
        if abs(m.one_day_price_change) > c.max_daily_midpoint_move:
            return "волатилен (движение midpoint > порога)"
        if m.best_bid > 0 and m.best_ask > 0 \
                and (m.best_ask - m.best_bid) < 2 * m.tick_size:
            return "спред < 2 тиков"
        if not m.resolution_source and len(m.description.strip()) < 80:
            return "мутные правила резолюции"
        if self.uma_risk(m):
            return "UMA-риск (субъективная резолюция)"
        return None

    def eligible(self, m: Market, now: datetime | None = None) -> bool:
        return self.reject_reason(m, now) is None

    def score(self, m: Market, book: OrderBook | None) -> float:
        """rewards-привлекательность / конкуренция мейкеров возле midpoint."""
        category = classify_category(m.question, m.category)
        boost = CATEGORY_BOOST.get(category, 1.0)
        # Прокси rewards-пула рынка: оборот * ширина reward-диапазона.
        attractiveness = m.volume_24h_usd * max(m.rewards_max_spread, 0.01)
        competition_usd = 1.0
        if book is not None and book.mid > 0:
            band = max(m.rewards_max_spread, 0.02)
            lo, hi = book.mid - band, book.mid + band
            competition_usd += sum(l.price * l.size for l in book.bids if l.price >= lo)
            competition_usd += sum(l.price * l.size for l in book.asks if l.price <= hi)
        return boost * attractiveness / competition_usd

    def top_markets(self, markets: list[Market],
                    books: dict[str, OrderBook | None]) -> list[Market]:
        eligible = [m for m in markets if self.eligible(m)]
        scored = sorted(
            eligible,
            key=lambda m: self.score(m, books.get(m.clob_token_ids[0])),
            reverse=True,
        )
        log.info("scorer: %d рынков пригодно для MM, берём топ-%d",
                 len(eligible), self._cfg.max_markets)
        return scored[: self._cfg.max_markets]
