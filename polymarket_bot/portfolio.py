"""Portfolio: fractional Kelly, concentration limits, kill-switch, exit rules."""

from __future__ import annotations

import logging
import math
from collections import defaultdict

from .config import BotConfig
from .ledger import Ledger
from .models import Estimate, Position, TradePlan

log = logging.getLogger(__name__)


def risk_group_key(p: Position) -> tuple[str, str]:
    """Positions that cannot lose simultaneously share a key (netted worst-case).

    In a neg-risk event exactly one outcome wins, so at most one NO leg there
    can lose. Everything else is its own solo group.
    """
    if p.neg_risk and p.event_id:
        return ("event", p.event_id)
    return ("solo", p.token_id)


def event_netted_exposure(positions: list[Position]) -> dict[str, float]:
    """Worst-case USD exposure per category, netting neg-risk event baskets.

    A neg-risk basket's worst case is its largest single leg (at most one loses),
    not the sum — so self-hedged clusters stop eating the category budget.
    """
    groups: dict[tuple[str, str], list[Position]] = defaultdict(list)
    for p in positions:
        groups[risk_group_key(p)].append(p)
    by_cat: dict[str, float] = defaultdict(float)
    for key, ps in groups.items():
        worst = max(p.cost_usd for p in ps) if (key[0] == "event" and len(ps) > 1) \
            else sum(p.cost_usd for p in ps)
        cat = max(ps, key=lambda p: p.cost_usd).category
        by_cat[cat] += worst
    return dict(by_cat)


def event_exposure_breakdown(positions: list[Position]) -> list[tuple[str, int, float, float]]:
    """Per risk-group rows for reporting: (label, legs, gross_usd, worst_case_usd)."""
    groups: dict[tuple[str, str], list[Position]] = defaultdict(list)
    for p in positions:
        groups[risk_group_key(p)].append(p)
    rows: list[tuple[str, int, float, float]] = []
    for key, ps in groups.items():
        gross = sum(p.cost_usd for p in ps)
        netted = key[0] == "event" and len(ps) > 1
        worst = max(p.cost_usd for p in ps) if netted else gross
        label = (f"{len(ps)}x {ps[0].question[:33]}" if netted
                 else ps[0].question[:38])
        rows.append((label, len(ps), gross, worst))
    rows.sort(key=lambda r: r[3], reverse=True)
    return rows

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

# Expert category-correlation matrix: how much shocks spill between them.
# This is the PRIOR — set_learned_correlation() overlays pairs actually
# learned from recorded category-index history (calibration.CorrelationLearner);
# unlearned pairs keep the prior.
_CORRELATION: dict[frozenset[str], float] = {
    frozenset({"geopolitics", "economy"}): 0.5,
    frozenset({"geopolitics", "crypto"}): 0.3,
    frozenset({"economy", "crypto"}): 0.5,
    frozenset({"elections", "geopolitics"}): 0.4,
    frozenset({"elections", "economy"}): 0.3,
}
_LEARNED_CORRELATION: dict[frozenset, float] = {}


def set_learned_correlation(learned: dict[frozenset, float]) -> None:
    global _LEARNED_CORRELATION
    _LEARNED_CORRELATION = dict(learned)


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
    key = frozenset({cat_a, cat_b})
    if key in _LEARNED_CORRELATION:
        return _LEARNED_CORRELATION[key]
    return _CORRELATION.get(key, 0.1)


def kelly_fraction(p: float, price: float) -> float:
    """Kelly bankroll fraction for a binary bet: buy at price, payout $1.

    f* = (p - price) / (1 - price); negative edge -> 0.
    """
    if not 0 < price < 1:
        return 0.0
    return max((p - price) / (1.0 - price), 0.0)


class Portfolio:
    def __init__(self, cfg: BotConfig, ledger: Ledger, mode: str):
        self._cfg = cfg.portfolio
        self._ledger = ledger
        self._mode = mode

    # --- bankroll state ---

    def equity(self, marks: dict[str, float] | None = None) -> float:
        """Bankroll + realized PnL + open positions (at mark or at cost)."""
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
        """Kill-switch: drawdown >= max_drawdown_pct puts the bot in observe mode."""
        dd = self.drawdown(marks)
        if dd >= self._cfg.max_drawdown_pct:
            log.error("KILL-SWITCH: drawdown %.1f%% >= %.0f%% — observe-only",
                      dd * 100, self._cfg.max_drawdown_pct * 100)
            return True
        return False

    # --- sizing ---

    def size_usd(self, category: str, p_est: float, p_mkt: float,
                 min_order_notional: float = 1.0, scale: float = 1.0) -> float | None:
        """Position size by Kelly with all caps; None = no room.

        Single sizing path for longshots and fade: kelly from edge, cap per
        market, per category (with the correlation matrix) and on total exposure.
        `scale` is the Sharpe allocator's tilt — applied BEFORE the caps, so it
        can shift capital between strategies but never break a hard limit.
        """
        cfg = self._cfg
        bankroll = cfg.bankroll_usd

        f_star = kelly_fraction(p_est, p_mkt)
        if f_star <= 0:
            return None
        size = cfg.kelly_fraction * f_star * bankroll * scale
        size = min(size, cfg.max_market_pct * bankroll)

        # Event-netted category exposure: a self-hedged neg-risk basket counts
        # as its worst-case leg, not the sum, freeing room for such clusters.
        exposure = event_netted_exposure(self._ledger.open_positions(self._mode))
        effective = exposure.get(category, 0.0) + sum(
            correlation(category, other) * usd
            for other, usd in exposure.items() if other != category
        )
        cat_room = cfg.max_category_pct * bankroll - effective
        if cat_room <= 0:
            return None
        size = min(size, cat_room)

        total_room = cfg.max_total_exposure_pct * bankroll - self._ledger.total_exposure(self._mode)
        if total_room <= 0:
            return None
        size = min(size, total_room)

        if size < max(min_order_notional, 1.0):
            return None
        return round(size, 2)

    def size_trade(self, estimate: Estimate, scale: float = 1.0) -> TradePlan | None:
        c = estimate.candidate
        category = classify_category(c.market.question, c.market.category)
        size = self.size_usd(category, estimate.p_est, estimate.p_mkt,
                             c.market.min_order_size * estimate.p_mkt, scale=scale)
        if size is None:
            return None
        # Above this price the edge falls below threshold — the executor must not pay more.
        min_edge = max(estimate.edge_ratio / 2, 1.2)  # margin: half the found edge
        price_cap = min(estimate.p_est / min_edge, 0.99)
        return TradePlan(estimate=estimate, category=category,
                         size_usd=size, limit_price_cap=price_cap)

    # --- exits ---

    def exit_plan(self, position: Position, current_price: float) -> tuple[float, float] | None:
        """(size_to_sell, min_price) if the position grew to the take-profit multiple.

        Sell take_profit_fraction; the remainder is a free lottery ticket.
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
        # Do not dump far below the trigger level.
        min_price = position.avg_price * cfg.take_profit_multiple * 0.8
        return float(sell_size), min_price
