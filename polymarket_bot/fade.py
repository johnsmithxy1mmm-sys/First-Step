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

from .config import BotConfig
from .executor import Executor
from .ledger import Ledger
from .models import Candidate, Estimate, TradePlan
from .monitor import alert
from .portfolio import Portfolio, classify_category

log = logging.getLogger(__name__)


class FadeStrategy:
    def __init__(self, cfg: BotConfig, ledger: Ledger, portfolio: Portfolio,
                 executor: Executor, mode: str, calibrator=None):
        self._cfg = cfg.fade
        self._ledger = ledger
        self._portfolio = portfolio
        self._executor = executor
        self._mode = mode
        self._calibrator = calibrator     # TailBiasCalibrator | None

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
        p_fair_yes = min(estimate.p_est, p_mkt_yes * (1.0 - self._bias(category, p_mkt_yes)))
        edge = (1.0 - p_fair_yes) - (1.0 - p_mkt_yes)   # = p_mkt_yes - p_fair_yes
        if edge < cfg.min_edge_after_fees:
            return "edge below min after fees"
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
                                        market.min_order_size * entry_no)
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
        """Fade IRR proxy: edge per day. Capital goes to the fastest recyclers."""
        days = est.candidate.market.days_to_resolution()
        edge = est.p_mkt * self._cfg.bias_discount     # ~fade edge on the NO side
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
        return entered
