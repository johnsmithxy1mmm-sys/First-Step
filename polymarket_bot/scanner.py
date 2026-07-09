"""Сканер: фильтры первого уровня поверх всех активных рынков.

Задача сканера — не найти сделку, а быстро отсечь то, что сделкой быть
не может: неликвид, рынки без стакана, мутные правила резолюции, неудобные
горизонты. Оценкой вероятности занимается estimator.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .clob import ClobReader
from .config import BotConfig
from .models import Candidate, Market

log = logging.getLogger(__name__)


class Scanner:
    def __init__(self, cfg: BotConfig, clob: ClobReader | None = None):
        self._cfg = cfg
        self._clob = clob

    def first_level_filter(self, markets: list[Market],
                           now: datetime | None = None) -> list[Candidate]:
        """Фильтры, не требующие похода в стакан."""
        s = self._cfg.scanner
        now = now or datetime.now(timezone.utc)
        out: list[Candidate] = []
        for m in markets:
            if m.closed or not m.enable_order_book:
                continue
            if m.volume_24h_usd < s.min_volume_24h_usd:
                continue

            days = m.days_to_resolution(now)
            if days is None or not s.min_days_to_resolution <= days <= s.max_days_to_resolution:
                continue

            q = m.question.lower()
            if s.include_keywords and not any(k.lower() in q for k in s.include_keywords):
                continue
            if any(k.lower() in q for k in s.exclude_keywords):
                continue

            # Неоднозначные правила резолюции — источник «правильно угадал,
            # но рынок зарезолвили не так». Требуем источник или внятное описание.
            if s.require_resolution_clarity and not m.resolution_source:
                if len(m.description.strip()) < s.min_description_chars:
                    continue

            for idx, price in enumerate(m.outcome_prices):
                if not s.price_min <= price <= s.price_max:
                    continue
                if idx >= len(m.clob_token_ids) or not m.clob_token_ids[idx]:
                    continue
                out.append(Candidate(
                    market=m, outcome_index=idx,
                    token_id=m.clob_token_ids[idx], p_mkt=price,
                ))

        # Самые ликвидные — первыми: на них реалистичнее исполниться.
        out.sort(key=lambda c: c.market.volume_24h_usd, reverse=True)
        return out[: s.max_candidates_per_cycle]

    def verify_depth(self, candidates: list[Candidate]) -> list[Candidate]:
        """Второй уровень: реальная глубина книги на нашей стороне."""
        s = self._cfg.scanner
        if not s.verify_book_depth or self._clob is None:
            return candidates
        verified: list[Candidate] = []
        for c in candidates:
            book = self._clob.order_book(c.token_id)
            if book is None:
                continue
            depth = book.bid_depth_usd_within(s.book_depth_pct_from_mid)
            if depth < s.min_book_depth_usd:
                continue
            c.book = book
            verified.append(c)
        log.info("scanner: %d кандидатов после проверки глубины (%d до)",
                 len(verified), len(candidates))
        return verified

    def scan(self, markets: list[Market]) -> list[Candidate]:
        candidates = self.first_level_filter(markets)
        log.info("scanner: %d кандидатов после фильтров 1-го уровня", len(candidates))
        return self.verify_depth(candidates)
