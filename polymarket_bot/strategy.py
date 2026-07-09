"""Отбор кандидатов: дешёвые исходы на живых рынках."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .config import BotConfig
from .gamma import market_float, parse_json_list


@dataclass
class Candidate:
    market_id: str
    question: str
    slug: str
    outcome: str
    token_id: str
    price: float          # цена из Gamma (может отставать от стакана)
    liquidity_usd: float
    volume_usd: float
    end_date: datetime | None
    neg_risk: bool
    tick_size: float
    min_order_size: float

    @property
    def payout_multiple(self) -> float:
        """Во сколько раз вырастут деньги, если исход победит."""
        return 1.0 / self.price if self.price > 0 else 0.0


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
    """Фильтрует рынки и возвращает дешёвые исходы, отсортированные по ликвидности."""
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
        for outcome, price_raw, token_id in zip(outcomes, prices, token_ids):
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
            out.append(Candidate(
                market_id=str(m.get("id", "")),
                question=question,
                slug=m.get("slug") or "",
                outcome=str(outcome),
                token_id=str(token_id),
                price=price,
                liquidity_usd=liquidity,
                volume_usd=volume,
                end_date=end_date,
                neg_risk=bool(m.get("negRisk", False)),
                tick_size=market_float(m, "orderPriceMinTickSize") or 0.001,
                min_order_size=market_float(m, "orderMinSize") or 5.0,
            ))
            per_market += 1

    # Сначала самые ликвидные: там дешёвую цену реально забрать из стакана.
    out.sort(key=lambda c: c.liquidity_usd, reverse=True)
    return out


def plan_bets(candidates: list[Candidate], cfg: BotConfig) -> list[Candidate]:
    """Обрезает список кандидатов по бюджету и лимиту количества ставок."""
    max_by_budget = int(cfg.total_budget_usd // cfg.stake_usd)
    return candidates[: min(cfg.max_bets, max_by_budget)]
