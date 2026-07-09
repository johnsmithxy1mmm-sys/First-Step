"""Кросс-рыночная когерентность — приоритет №1 среди сигналов.

Логические связки между рынками дают структурный edge, не зависящий от
чьих-либо мнений:

1. Neg-risk корзины. В событии с взаимоисключающими исходами (выборы и т.п.)
   сумма цен Yes всех рынков обязана быть ~1. Если S = Σp ≠ 1, каждая цена
   систематически смещена: честная вероятность кандидата ≈ p_mkt / S.
   При S < 1 корзина в сумме недооценена — чистейший edge.

2. Календарные цепочки. «Событие до 31 марта» логически влечёт «до 30 июня»,
   значит P(до ранней даты) ≤ P(до поздней). Если рынок с поздней датой стоит
   ДЕШЕВЛЕ раннего — нарушение монотонности: поздний недооценён минимум до
   цены раннего.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from ..models import Candidate, Market, Signal

log = logging.getLogger(__name__)

# Порог, ниже которого расхождение суммы корзины считаем шумом спреда.
BASKET_TOLERANCE = 0.02

_DATE_NOISE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b"
    r"|\b\d{1,4}(st|nd|rd|th)?\b|[,.?]",
    re.IGNORECASE,
)


def normalize_question(question: str) -> str:
    """Убирает даты/числа: «X by March 31?» и «X by June 30?» дают один ключ."""
    stripped = _DATE_NOISE.sub(" ", question.lower())
    return " ".join(stripped.split())


class CoherenceSignal:
    name = "coherence"

    def __init__(self, markets: list[Market]):
        # Корзины neg-risk событий: сумма Yes-цен по событию.
        self._basket_sum: dict[str, float] = {}
        self._basket_size: dict[str, int] = {}
        by_event: dict[str, list[Market]] = defaultdict(list)
        for m in markets:
            if m.event_neg_risk and m.event_id and m.outcome_prices:
                by_event[m.event_id].append(m)
        for event_id, group in by_event.items():
            if len(group) >= 2:
                self._basket_sum[event_id] = sum(m.outcome_prices[0] for m in group)
                self._basket_size[event_id] = len(group)

        # Календарные цепочки: рынки с одинаковым нормализованным вопросом.
        self._chains: dict[str, list[Market]] = defaultdict(list)
        for m in markets:
            if m.end_date is not None and m.outcome_prices:
                self._chains[normalize_question(m.question)].append(m)

    def evaluate(self, candidate: Candidate) -> Signal | None:
        return self._basket_check(candidate) or self._calendar_check(candidate)

    def _basket_check(self, candidate: Candidate) -> Signal | None:
        if candidate.outcome_index != 0:
            return None
        s = self._basket_sum.get(candidate.market.event_id)
        if s is None or s <= 0 or abs(s - 1.0) <= BASKET_TOLERANCE:
            return None
        p_fair = min(candidate.p_mkt / s, 0.999)
        # Недооценённая корзина (S < 1) — структурный арбитраж, доверие максимальное.
        confidence = 0.9 if s < 1.0 else 0.7
        return Signal(
            name=self.name,
            p_est=p_fair,
            confidence=confidence,
            rationale=f"neg-risk basket «{candidate.market.event_title[:50]}»: "
                      f"Σp={s:.3f} по {self._basket_size[candidate.market.event_id]} исходам "
                      f"-> fair={p_fair:.4f}",
        )

    def _calendar_check(self, candidate: Candidate) -> Signal | None:
        if candidate.outcome_index != 0:
            return None
        m = candidate.market
        chain = self._chains.get(normalize_question(m.question), [])
        if len(chain) < 2 or m.end_date is None:
            return None
        # Максимальная Yes-цена среди рынков той же цепочки с БОЛЕЕ РАННИМ дедлайном.
        earlier_max = max(
            (o.outcome_prices[0] for o in chain
             if o.id != m.id and o.end_date is not None and o.end_date < m.end_date),
            default=None,
        )
        if earlier_max is None or earlier_max <= candidate.p_mkt + BASKET_TOLERANCE:
            return None
        # Нарушение монотонности: наш (поздний) обязан стоить >= раннего.
        return Signal(
            name=self.name,
            p_est=min(earlier_max, 0.999),
            confidence=0.85,
            rationale=f"calendar chain: более ранний дедлайн торгуется по {earlier_max:.3f}, "
                      f"наш (позже, {m.end_date.date()}) по {candidate.p_mkt:.3f} — "
                      f"нарушение P(early) <= P(late)",
        )
