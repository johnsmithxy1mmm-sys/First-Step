"""Портфель: fractional Kelly, лимиты концентрации, kill-switch, правила выхода."""

from __future__ import annotations

import logging
import math

from .config import BotConfig
from .ledger import Ledger
from .models import Estimate, Position, TradePlan

log = logging.getLogger(__name__)

CATEGORIES = ("geopolitics", "crypto", "elections", "nature", "sports", "economy", "other")

_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "geopolitics": ("war", "invasion", "invade", "missile", "airstrike", "ceasefire", "nato",
                    "nuclear", "sanction", "treaty", "military"),
    "crypto": ("bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "token", "defi"),
    "elections": ("election", "president", "prime minister", "senate", "parliament",
                  "vote", "poll", "nominee", "primary", "mayor"),
    "nature": ("earthquake", "hurricane", "volcano", "flood", "wildfire", "tsunami",
               "tornado", "temperature", "climate"),
    "sports": ("champion", "world cup", "super bowl", "olympics", "nba", "nfl",
               "premier league", "grand slam", "f1"),
    "economy": ("fed", "interest rate", "inflation", "recession", "gdp", "default",
                "shutdown", "tariff", "stock", "s&p"),
}

# Экспертная матрица корреляций категорий: насколько шоки перетекают между ними.
_CORRELATION: dict[frozenset[str], float] = {
    frozenset({"geopolitics", "economy"}): 0.5,
    frozenset({"geopolitics", "crypto"}): 0.3,
    frozenset({"economy", "crypto"}): 0.5,
    frozenset({"elections", "geopolitics"}): 0.4,
    frozenset({"elections", "economy"}): 0.3,
}


def classify_category(question: str, hint: str = "") -> str:
    text = f"{hint} {question}".lower()
    best, best_hits = "other", 0
    for category, keywords in _CATEGORY_KEYWORDS.items():
        hits = sum(1 for k in keywords if k in text)
        if hits > best_hits:
            best, best_hits = category, hits
    return best


def correlation(cat_a: str, cat_b: str) -> float:
    if cat_a == cat_b:
        return 1.0
    return _CORRELATION.get(frozenset({cat_a, cat_b}), 0.1)


def kelly_fraction(p: float, price: float) -> float:
    """Доля банка по Келли для бинарной ставки: купить по price, выплата $1.

    f* = (p - price) / (1 - price); отрицательный edge -> 0.
    """
    if not 0 < price < 1:
        return 0.0
    return max((p - price) / (1.0 - price), 0.0)


class Portfolio:
    def __init__(self, cfg: BotConfig, ledger: Ledger, mode: str):
        self._cfg = cfg.portfolio
        self._ledger = ledger
        self._mode = mode

    # --- состояние банка ---

    def equity(self, marks: dict[str, float] | None = None) -> float:
        """Банк + реализованный PnL + открытые позиции (по марку или по коcту)."""
        positions = self._ledger.open_positions(self._mode)
        marks = marks or {}
        open_value = sum(
            p.size * marks.get(p.token_id, p.avg_price) for p in positions
        )
        cash = self._cfg.bankroll_usd + self._ledger.realized_pnl(self._mode) \
            - sum(p.cost_usd for p in positions)
        return cash + open_value

    def snapshot(self, marks: dict[str, float] | None = None) -> None:
        exposure = self._ledger.total_exposure(self._mode)
        self._ledger.snapshot_bank(self.equity(marks) - exposure, exposure)

    def drawdown(self, marks: dict[str, float] | None = None) -> float:
        hwm = max(self._ledger.high_water_mark(), self._cfg.bankroll_usd)
        eq = self.equity(marks)
        return max(0.0, (hwm - eq) / hwm) if hwm > 0 else 0.0

    def observe_only(self, marks: dict[str, float] | None = None) -> bool:
        """Kill-switch: просадка ≥ max_drawdown_pct переводит бота в наблюдение."""
        dd = self.drawdown(marks)
        if dd >= self._cfg.max_drawdown_pct:
            log.error("KILL-SWITCH: просадка %.1f%% >= %.0f%% — observe-only",
                      dd * 100, self._cfg.max_drawdown_pct * 100)
            return True
        return False

    # --- сайзинг ---

    def size_trade(self, estimate: Estimate) -> TradePlan | None:
        cfg = self._cfg
        c = estimate.candidate
        bankroll = self._cfg.bankroll_usd
        category = classify_category(c.market.question, c.market.category)

        f_star = kelly_fraction(estimate.p_est, estimate.p_mkt)
        if f_star <= 0:
            return None
        size = cfg.kelly_fraction * f_star * bankroll

        # Жёсткий кэп на рынок.
        size = min(size, cfg.max_market_pct * bankroll)

        # Кэп на категорию с учётом корреляций: эффективная экспозиция категории
        # включает долю коррелированных категорий.
        exposure = self._ledger.exposure_by_category(self._mode)
        effective = exposure.get(category, 0.0) + sum(
            correlation(category, other) * usd
            for other, usd in exposure.items() if other != category
        )
        cat_room = cfg.max_category_pct * bankroll - effective
        if cat_room <= 0:
            log.info("skip %s: категория %s заполнена (eff=%.0f)", c.market.id, category, effective)
            return None
        size = min(size, cat_room)

        # Кэп на суммарную экспозицию.
        total_room = cfg.max_total_exposure_pct * bankroll - self._ledger.total_exposure(self._mode)
        if total_room <= 0:
            return None
        size = min(size, total_room)

        # Минимальный размер ордера биржи.
        min_usd = c.market.min_order_size * estimate.p_mkt
        if size < max(min_usd, 1.0):
            return None

        # Выше этой цены edge падает ниже порога — executor не должен платить больше.
        min_edge = max(estimate.edge_ratio / 2, 1.2)  # запас: половина найденного edge
        price_cap = min(estimate.p_est / min_edge, 0.99)

        return TradePlan(
            estimate=estimate,
            category=category,
            size_usd=round(size, 2),
            limit_price_cap=price_cap,
        )

    # --- выходы ---

    def exit_plan(self, position: Position, current_price: float) -> tuple[float, float] | None:
        """(size_to_sell, min_price) если позиция выросла до take-profit multiple.

        Продаём take_profit_fraction, остаток — бесплатный лотерейный билет.
        """
        cfg = self._cfg
        if position.avg_price <= 0 or current_price <= 0:
            return None
        multiple = current_price / position.avg_price
        if multiple < cfg.take_profit_multiple:
            return None
        sell_size = math.floor(position.size * cfg.take_profit_fraction)
        if sell_size <= 0:
            return None
        # Не сливать сильно ниже триггерного уровня.
        min_price = position.avg_price * cfg.take_profit_multiple * 0.8
        return float(sell_size), min_price
