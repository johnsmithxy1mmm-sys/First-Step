"""Отбор и скоринг кандидатов: дешёвые исходы на живых рынках.

Скоринг (0..1) — честные эвристики без магии, веса ниже:
  momentum  — цена исхода росла за сутки: кто-то накапливает позицию;
  activity  — доля суточного объёма в общем: рынок проснулся;
  horizon   — «сладкое» окно резолюции ~30 дней: событию есть когда случиться,
              но деньги не заморожены на год;
  liquidity — глубина рынка: дешёвую цену реально забрать;
  value     — чем дешевле исход, тем выше потенциал выплаты.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import BotConfig
from .gamma import market_float, parse_json_list

SCORE_WEIGHTS = {
    "momentum": 0.35,
    "activity": 0.20,
    "horizon": 0.15,
    "liquidity": 0.15,
    "value": 0.15,
}


@dataclass
class Candidate:
    market_id: str
    question: str
    slug: str
    outcome: str
    outcome_index: int
    token_id: str
    price: float          # цена из Gamma (может отставать от стакана)
    liquidity_usd: float
    volume_usd: float
    end_date: datetime | None
    neg_risk: bool
    tick_size: float
    min_order_size: float
    score: float = 0.0
    score_parts: dict = field(default_factory=dict)

    @property
    def payout_multiple(self) -> float:
        """Во сколько раз вырастут деньги, если исход победит."""
        return 1.0 / self.price if self.price > 0 else 0.0


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def score_candidate(c: Candidate, market: dict, cfg: BotConfig) -> None:
    """Заполняет c.score и c.score_parts. Все компоненты в [0, 1]."""
    parts: dict[str, float] = {}

    # Моментум: oneDayPriceChange в Gamma — по первому исходу (Yes).
    # Для No знак обратный. Рост дешёвого исхода = сигнал накопления.
    day_change = market_float(market, "oneDayPriceChange")
    if c.outcome_index == 1:
        day_change = -day_change
    if c.outcome_index in (0, 1) and c.price > 0 and day_change > 0:
        parts["momentum"] = _clamp(2.0 * day_change / c.price)
    else:
        parts["momentum"] = 0.0

    # Активность: суточный объём против всего объёма. 5%+ оборота за день = максимум.
    vol24 = market_float(market, "volume24hr", "volume24hrClob")
    parts["activity"] = _clamp(20.0 * vol24 / c.volume_usd) if c.volume_usd > 0 else 0.0

    # Горизонт: пик на ~30 днях до резолюции, линейный спад к краям окна.
    if c.end_date is not None:
        days_left = (c.end_date - datetime.now(timezone.utc)).total_seconds() / 86400
        parts["horizon"] = _clamp(1.0 - abs(days_left - 30.0) / 90.0)
    else:
        parts["horizon"] = 0.3  # дата неизвестна — нейтрально-низко

    # Ликвидность: log-шкала, $1M+ = 1.0, $1k ≈ 0.5.
    if c.liquidity_usd > 1:
        parts["liquidity"] = _clamp(math.log10(c.liquidity_usd) / 6.0)
    else:
        parts["liquidity"] = 0.0

    # Ценность: дешевле = больше иксов при победе.
    span = max(cfg.max_price - cfg.min_price, 1e-9)
    parts["value"] = _clamp((cfg.max_price - c.price) / span)

    c.score_parts = parts
    c.score = sum(SCORE_WEIGHTS[k] * v for k, v in parts.items())


def stake_for(c: Candidate, cfg: BotConfig) -> float:
    """Размер ставки: базовый stake_usd, при stake_scaling — 0.5x..1.5x по скору."""
    if not cfg.stake_scaling:
        return cfg.stake_usd
    return round(cfg.stake_usd * (0.5 + c.score), 2)


def _parse_end_date(raw) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def _keywords_ok(question: str, cfg: BotConfig) -> bool:
    q = question.lower()
    if cfg.include_keywords and not any(k.lower() in q for k in cfg.include_keywords):
        return False
    return not any(k.lower() in q for k in cfg.exclude_keywords)


def find_candidates(markets, cfg: BotConfig, skip_token_ids: set[str] = frozenset()) -> list[Candidate]:
    """Фильтрует рынки, скорит дешёвые исходы и сортирует по убыванию скора."""
    now = datetime.now(timezone.utc)
    out: list[Candidate] = []
    for m in markets:
        if not m.get("enableOrderBook", True):
            continue  # без стакана CLOB ставку не разместить

        question = m.get("question") or ""
        if not _keywords_ok(question, cfg):
            continue

        liquidity = market_float(m, "liquidityNum", "liquidity")
        volume = market_float(m, "volumeNum", "volume")
        if liquidity < cfg.min_liquidity_usd or volume < cfg.min_volume_usd:
            continue

        end_date = _parse_end_date(m.get("endDate"))
        if end_date is not None:
            days_left = (end_date - now).total_seconds() / 86400
            if not cfg.min_days_to_resolution <= days_left <= cfg.max_days_to_resolution:
                continue

        outcomes = parse_json_list(m.get("outcomes"))
        prices = parse_json_list(m.get("outcomePrices"))
        token_ids = parse_json_list(m.get("clobTokenIds"))
        if not (len(outcomes) == len(prices) == len(token_ids)):
            continue

        per_market = 0
        for idx, (outcome, price_raw, token_id) in enumerate(zip(outcomes, prices, token_ids)):
            try:
                price = float(price_raw)
            except (TypeError, ValueError):
                continue
            if not cfg.min_price <= price <= cfg.max_price:
                continue
            if not token_id or token_id in skip_token_ids:
                continue
            if per_market >= cfg.max_bets_per_market:
                break
            c = Candidate(
                market_id=str(m.get("id", "")),
                question=question,
                slug=m.get("slug") or "",
                outcome=str(outcome),
                outcome_index=idx,
                token_id=str(token_id),
                price=price,
                liquidity_usd=liquidity,
                volume_usd=volume,
                end_date=end_date,
                neg_risk=bool(m.get("negRisk", False)),
                tick_size=market_float(m, "orderPriceMinTickSize") or 0.001,
                min_order_size=market_float(m, "orderMinSize") or 5.0,
            )
            score_candidate(c, m, cfg)
            if c.score < cfg.min_score:
                continue
            out.append(c)
            per_market += 1

    out.sort(key=lambda c: c.score, reverse=True)
    return out


def plan_bets(candidates: list[Candidate], cfg: BotConfig) -> list[Candidate]:
    """Обрезает список кандидатов по бюджету и лимиту количества ставок."""
    planned: list[Candidate] = []
    budget = cfg.total_budget_usd
    for c in candidates:
        if len(planned) >= cfg.max_bets:
            break
        stake = stake_for(c, cfg)
        if stake > budget:
            continue
        planned.append(c)
        budget -= stake
    return planned
