"""Strategy: fading overpriced tails (the profitable side of longshot bias).

Cheap outcomes on prediction markets are systematically OVERpriced (the crowd
loves buying "lottery tickets"). Naively buying such tails is negative-EV — the
opposite side is positive: buy NO when the YES tail costs more than fair.

Mechanics on a binary market [Yes, No], prices [p, 1-p]:
  - the Yes tail is overpriced: p_mkt > fair probability;
  - then No is underpriced: 1-p_mkt < 1 - p_fair;
  - buy No near ~(1-p_mkt); at resolution it pays $1.

Fair Yes probability: min(ensemble estimate, p_mkt*(1-bias_discount)) — the
lower of "what the estimator says" and "the systematic bias correction". Even
with no active signal, bias_discount alone makes the tail fadeable.

Reuses the estimator, portfolio limits (Kelly + caps) and the executor.
Maker-only limit orders (fee ~0). Idempotency is handled by the executor.
Positions are held to resolution (the win is small and frequent).
"""

from __future__ import annotations

import logging
import math
from typing import NamedTuple

from .config import BotConfig, FadeConfig
from .executor import Executor
from .ledger import Ledger
from .models import Candidate, Estimate, Position, TradePlan
from .monitor import alert
from .portfolio import Portfolio, classify_category

log = logging.getLogger(__name__)


def payoff_ratio(entry_price: float) -> float:
    """Max gain / max loss for a leg bought at `entry_price` and paid $1 or $0.

    (1 - entry) / entry. This is the number that says how many wins it takes to
    repay one loss: 0.02 -> 50, 0.0101 (an entry at 0.99) -> 99.
    """
    if not 0.0 < entry_price < 1.0:
        return 0.0
    return (1.0 - entry_price) / entry_price


class FadeExit(NamedTuple):
    size: float
    min_price: float
    reason: str          # "tail-stop" | "edge-captured"


def fade_exit_plan(position: Position, mark: float,
                   cfg: FadeConfig) -> FadeExit | None:
    """The fade's real exit; None = keep holding.

    portfolio.exit_plan is unusable here. It triggers on
    `mark / avg_price >= take_profit_multiple` (7.0 by default), and a fade leg
    bought at 0.976 would have to reach 6.83 — prices stop at 1.0, so the most it
    can ever reach is 1.024 and the plan returns None on every call for the life
    of the position. That machinery was written for longshots (0.03 -> 0.21) and
    the fade inherited it wholesale, which left these legs with exactly one exit:
    resolution, at the full notional.

    Two rules, both in price space where the arithmetic is reachable:

      tail-stop         the implied tail probability has multiplied by
                        `tail_stop_multiple` (2.4% -> 7.2%). Caps the loss at a
                        few cents instead of ~98. This does NOT add edge — it
                        pays the spread and turns some spikes-that-revert into
                        realized losses. It buys survivability, so the strategy
                        lives long enough to be measured.

      edge-captured      `early_take_captured` of THIS trade's maximum gain has
                        been realized. At entry 0.976 the whole prize is 2.4c, so
                        0.75 means out at 0.994 with 1.8c banked and the capital
                        recycled instead of waiting months for the last 0.6c.
                        Keyed to the entry, so it can never fire at a loss.
    """
    entry = position.avg_price
    # Guard the direction of the arithmetic: a fade leg is a NO bought high (the
    # YES tail is <= fade_max_price, so entry > 0.9 by construction). Anything
    # cheap here is not a fade and must not be routed through tail logic.
    if not 0.5 <= entry < 1.0 or not 0.0 < mark < 1.0:
        return None

    reason: str | None = None
    tail_entry = 1.0 - entry
    if cfg.tail_stop_multiple > 0 and tail_entry > 0:
        if (1.0 - mark) >= tail_entry * cfg.tail_stop_multiple:
            reason = "tail-stop"
    if reason is None and cfg.early_take_captured > 0 and tail_entry > 0:
        # Fraction of THIS trade's prize that is already banked. Requires
        # mark > entry, so a take can never realize a loss — the earlier
        # remaining-ratio form could, and did, on a legacy book.
        if (mark - entry) / tail_entry >= cfg.early_take_captured:
            reason = "edge-captured"
    if reason is None:
        return None

    size = math.floor(position.size * max(0.0, min(cfg.exit_fraction, 1.0)))
    if size <= 0:
        return None
    # A stop must actually get out, so it accepts slippage; a take is selling
    # into strength and should not dump.
    min_price = mark * 0.9 if reason == "tail-stop" else mark * 0.99
    return FadeExit(float(size), round(max(min_price, 0.01), 4), reason)


class FadeStrategy:
    def __init__(self, cfg: BotConfig, ledger: Ledger, portfolio: Portfolio,
                 executor: Executor, mode: str, calibrator=None):
        self._cfg = cfg.fade
        self._ledger = ledger
        self._portfolio = portfolio
        self._executor = executor
        self._mode = mode
        self._calibrator = calibrator     # TailBiasCalibrator | None
        # Sharpe allocator tilt (set by the bot); caps still apply after it.
        self.size_scale: float = 1.0
        # Does the estimator change any decision here? p_est only moves the fade
        # when it is BELOW the bias-corrected price, i.e. when the min() picks it
        # — otherwise the whole trade rests on the bias_discount prior and the
        # LLM is being paid for nothing. Counted, not assumed.
        self.est_seen = 0
        self.est_binding = 0

    def _bias(self, category: str, p_mkt_yes: float) -> float:
        """Learned bias_discount for this bucket, or the config prior."""
        if self._calibrator is not None:
            return self._calibrator.bias(category, p_mkt_yes)
        return self._cfg.bias_discount

    def refresh_calibration(self) -> None:
        """Refit the bias calibrator from resolved fade tails (called by a job)."""
        if self._calibrator is not None:
            self._calibrator.fit(self._ledger.resolved_for_calibration(self._mode, "fade"))

    def reject_reason(self, estimate: Estimate) -> str | None:
        """None = the tail is fadeable; else the reject reason (for diagnostics).

        Covers the pre-sizing filters only; the portfolio cap is applied later
        in plan() where the bankroll/exposure state is available.
        """
        cfg = self._cfg
        c = estimate.candidate
        p_mkt_yes = estimate.p_mkt
        if not cfg.min_tail_price <= p_mkt_yes <= cfg.fade_max_price:
            return f"tail price outside [{cfg.min_tail_price}, {cfg.fade_max_price}]"
        if c.outcome_index not in (0, 1) or len(c.market.clob_token_ids) < 2:
            return "not a binary market"
        # Payoff SHAPE, independent of time and of edge: buying NO at (1 - tail)
        # risks that price to earn `tail`. At a 1c tail it takes 99 wins to repay
        # one loss, which is not a book you can run — one adverse resolution wipes
        # out the maximum upside of twenty positions and half as much again.
        entry_no = 1.0 - p_mkt_yes
        if payoff_ratio(entry_no) < cfg.min_payoff_ratio:
            return f"payoff ratio below {cfg.min_payoff_ratio}"
        # Near-term only: the fade edge is fixed per bet, so a distant resolution
        # means a tiny IRR (capital locked for months). Recycle capital fast.
        days = c.market.days_to_resolution()
        if days is None or days > cfg.max_days_to_resolution:
            return "resolution too far out"
        # If the estimator sees a REAL longshot (p_est >= ratio * p_mkt, the same
        # threshold the longshot strategy BUYS Yes on), don't fade our own signal.
        if estimate.p_est >= p_mkt_yes * cfg.longshot_veto_ratio:
            return "estimator sees a real longshot"
        category = classify_category(c.market.question, c.market.category)
        bias_fair = p_mkt_yes * (1.0 - self._bias(category, p_mkt_yes))
        p_fair_yes = min(estimate.p_est, bias_fair)
        self.est_seen += 1
        if estimate.p_est < bias_fair:
            self.est_binding += 1        # the LLM, not the prior, set fair value
        edge = (1.0 - p_fair_yes) - (1.0 - p_mkt_yes)   # = p_mkt_yes - p_fair_yes
        if edge < cfg.min_edge_after_fees:
            return "edge below min after fees"
        # Hard IRR floor. _irr_score already computes this quantity, but it only
        # RANKS by it — and ranking alone changes nothing while the portfolio caps
        # are not binding, which is why a leg earning 0.4c over 90 days was
        # entered alongside one earning 2.2c over 4.
        if cfg.min_edge_per_day > 0 and edge / max(days, 0.5) < cfg.min_edge_per_day:
            return f"edge per day below {cfg.min_edge_per_day}"
        return None

    def plan(self, estimate: Estimate) -> TradePlan | None:
        """Builds a NO-buy plan for an overpriced YES tail (or None)."""
        if self.reject_reason(estimate) is not None:
            return None

        cfg = self._cfg
        c = estimate.candidate
        market = c.market
        p_mkt_yes = estimate.p_mkt

        category = classify_category(market.question, market.category)
        # Fair Yes probability with the (learned or prior) systematic bias.
        p_fair_yes = min(estimate.p_est, p_mkt_yes * (1.0 - self._bias(category, p_mkt_yes)))
        p_fair_no = 1.0 - p_fair_yes
        entry_no = 1.0 - p_mkt_yes                    # market price of No
        no_index = 1 - c.outcome_index
        no_token = market.clob_token_ids[no_index]

        size = self._portfolio.size_usd(category, p_fair_no, entry_no,
                                        market.min_order_size * entry_no,
                                        scale=self.size_scale)
        if size is None:
            return None

        # Synthetic No-side estimate for sizing / ledger.
        no_est = Estimate(
            candidate=Candidate(market=market, outcome_index=no_index,
                                token_id=no_token, p_mkt=entry_no),
            p_mkt=entry_no, p_est=p_fair_no, signals=estimate.signals,
        )
        # Don't pay more for No than leaves the minimum edge.
        price_cap = min(p_fair_no - cfg.min_edge_after_fees, 0.99)
        return TradePlan(estimate=no_est, category=category,
                         size_usd=size, limit_price_cap=price_cap)

    def _irr_score(self, est: Estimate) -> float:
        """Fade IRR proxy: edge per day. Capital goes to the fastest recyclers.

        Uses the same (learned or prior) bias as plan(), so the ranking agrees
        with the sizing.
        """
        m = est.candidate.market
        days = m.days_to_resolution()
        category = classify_category(m.question, m.category)
        edge = est.p_mkt * self._bias(category, est.p_mkt)   # ~fade edge, NO side
        return edge / max(days if days is not None else 999.0, 0.5)

    def cycle(self, estimates: list[Estimate]) -> int:
        """Fades overpriced tails among the scored candidates. -> entries."""
        if not self._cfg.enabled:
            return 0
        entered = 0
        # IRR planner: enter shortest-horizon / highest-edge first, so the best
        # opportunities get capital before the portfolio caps fill.
        for est in sorted(estimates, key=self._irr_score, reverse=True):
            plan = self.plan(est)
            if plan is None:
                continue
            result = self._executor.execute(plan, strategy="fade")
            if result.status == "filled":
                entered += 1
                m = plan.estimate.candidate.market
                side = plan.estimate.candidate.outcome
                log.info("FADE %s %.3f x %.0f = $%.2f (fair Yes %.3f vs market %.3f) [%s]",
                         side, result.avg_price, result.filled_size,
                         result.avg_price * result.filled_size,
                         1 - plan.estimate.p_est, est.p_mkt, m.question[:50])
                alert(f"FADE fill [{self._mode}] {side} {result.avg_price:.3f} "
                      f"x {result.filled_size:,.0f} "
                      f"= ${result.avg_price * result.filled_size:,.2f} "
                      f"(fair Yes {1 - plan.estimate.p_est:.3f} vs market {est.p_mkt:.3f}) "
                      f"— {m.question[:60]}")
        if entered:
            log.info("fades this cycle: %d", entered)
        if self.est_seen:
            log.info("fade: estimator set fair value in %d/%d scored tails "
                     "(%.1f%%) — the rest ran on the bias_discount prior",
                     self.est_binding, self.est_seen,
                     100.0 * self.est_binding / self.est_seen)
        return entered
