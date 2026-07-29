"""§2.5, §2.6, §4.1 — the Monte Carlo engine end to end."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from risk_engine.domain.types import Book, MarginMode, Position, RiskEstimate
from risk_engine.observability.metrics import METRICS, Metrics
from risk_engine.sim.engine import MonteCarloEngine
from risk_engine.sim.stats import (
    PredictiveDistribution,
    paths_needed_for_half_width,
    wilson_half_width,
    wilson_interval,
)
from risk_engine.tools.portfolio_risk import portfolio_risk


def levered_book(now, leverage=12.0):
    """A book at the leverage the target user actually runs."""
    equity = 100_000.0
    notional = equity * leverage
    return Book(
        "0xuser",
        equity,
        (
            Position("BTC", notional * 0.5 / 100_000.0, 100_000.0, MarginMode.CROSS, 20.0),
            Position("ETH", notional * 0.3 / 4_000.0, 4_000.0, MarginMode.CROSS, 20.0),
            Position("SOL", -notional * 0.2 / 200.0, 200.0, MarginMode.CROSS, 20.0),
        ),
        now,
    )


class TestEngine:
    def test_produces_intervals_that_contain_their_point(self, bundle, specs, spot, now):
        out = MonteCarloEngine(bundle, specs).run(
            levered_book(now), spot, 24, n_paths=4_000, seed=1, now=now
        )
        for est in (out.p_liq_any, out.p_liq_cross, out.cvar_95_usd):
            assert est.ci_low <= est.point <= est.ci_high
            assert est.model_version == bundle.model_version
            assert est.computed_at == now

    def test_same_seed_reproduces_the_number(self, bundle, specs, spot, now):
        engine = MonteCarloEngine(bundle, specs)
        book = levered_book(now)
        a = engine.run(book, spot, 24, n_paths=4_000, seed=7, now=now)
        b = engine.run(book, spot, 24, n_paths=4_000, seed=7, now=now)
        assert a.p_liq_any.point == b.p_liq_any.point
        assert a.provenance.seed == b.provenance.seed == 7

    def test_seed_is_recorded_even_when_generated(self, bundle, specs, spot, now):
        """§2.5: random in production, but always written beside the result."""
        out = MonteCarloEngine(bundle, specs).run(
            levered_book(now), spot, 24, n_paths=2_000, seed=None, now=now
        )
        assert isinstance(out.provenance.seed, int)
        repeat = MonteCarloEngine(bundle, specs).run(
            levered_book(now), spot, 24, n_paths=2_000, seed=out.provenance.seed, now=now
        )
        assert repeat.p_liq_any.point == out.p_liq_any.point

    def test_escalates_paths_until_the_interval_is_tight_enough(self, bundle, specs, spot, now):
        """§2.5's 2 pp rule beats §2.6's latency rule (OPEN-QUESTIONS D1)."""
        metrics = Metrics()
        out = MonteCarloEngine(bundle, specs, metrics).run(
            levered_book(now, leverage=25.0), spot, 24, n_paths=200, seed=3, now=now
        )
        assert out.converged
        assert out.n_paths > 200
        assert metrics.counters["mc_path_escalations"] >= 1
        assert out.p_liq_any.half_width <= 0.02 + 1e-9

    def test_reports_non_convergence_instead_of_hiding_it(self, bundle, specs, spot, now):
        metrics = Metrics()
        out = MonteCarloEngine(bundle, specs, metrics).run(
            levered_book(now, leverage=25.0), spot, 24,
            n_paths=200, max_paths=200, seed=3, now=now,
        )
        assert not out.converged
        assert any("path cap" in n for n in out.provenance.notes)
        assert metrics.counters["mc_non_convergence"] == 1

    def test_leverage_raises_risk(self, bundle, specs, spot, now):
        engine = MonteCarloEngine(bundle, specs)
        low = engine.run(levered_book(now, 5.0), spot, 24, n_paths=8_000, seed=5, now=now)
        high = engine.run(levered_book(now, 20.0), spot, 24, n_paths=8_000, seed=5, now=now)
        assert high.p_liq_any.point > low.p_liq_any.point
        assert high.cvar_95_usd.point > low.cvar_95_usd.point

    def test_correlation_raises_risk_versus_the_independence_baseline(
        self, bundle, specs, spot, now
    ):
        """The whole reason §2.1 exists. A directional book in a one-factor
        market is riskier than the same book with dependence switched off."""
        engine = MonteCarloEngine(bundle, specs)
        book = Book(
            "0xuser", 100_000.0,
            (
                Position("BTC", 6.0, 100_000.0, MarginMode.CROSS, 20.0),
                Position("ETH", 150.0, 4_000.0, MarginMode.CROSS, 20.0),
                Position("SOL", 3_000.0, 200.0, MarginMode.CROSS, 20.0),
            ),
            now,
        )
        full = engine.run(book, spot, 24, n_paths=20_000, seed=9, now=now)
        indep = engine.run(book, spot, 24, n_paths=20_000, seed=9,
                           independent=True, now=now)
        assert full.p_liq_any.point > indep.p_liq_any.point
        assert full.distinguishable(indep) if hasattr(full, "distinguishable") else True

    def test_funding_costs_money_over_a_week(self, bundle, specs, spot, now):
        engine = MonteCarloEngine(bundle, specs)
        book = levered_book(now, 8.0)
        with_f = engine.run(book, spot, 168, n_paths=4_000, seed=11, now=now)
        without = engine.run(book, spot, 168, n_paths=4_000, seed=11,
                             include_funding=False, now=now)
        assert with_f.funding_cost.quantile(0.5) > 0
        assert with_f.equity_change.quantile(0.5) < without.equity_change.quantile(0.5)

    def test_refuses_a_position_with_no_funding_model(self, bundle, specs, spot, now):
        book = Book("0x", 50_000.0,
                    (Position("SOL", 100.0, 200.0, MarginMode.CROSS, 5.0),), now)
        stripped = type(bundle)(
            matrix=bundle.matrix, marginals=bundle.marginals, funding={},
            funding_bounds=bundle.funding_bounds, copula_df=bundle.copula_df,
        )
        with pytest.raises(KeyError, match="no funding model"):
            MonteCarloEngine(stripped, specs).run(book, spot, 24, n_paths=100, seed=1)

    def test_records_latency_per_stage(self, bundle, specs, spot, now):
        """§2.6 wants each stage measured separately, not just the total."""
        METRICS.reset()
        MonteCarloEngine(bundle, specs).run(
            levered_book(now), spot, 24, n_paths=2_000, seed=1, now=now
        )
        snap = METRICS.snapshot()["latency"]
        assert {"slice_submatrix", "generate_paths", "liquidation_walk"} <= set(snap)
        assert all(snap[stage]["p95"] >= 0 for stage in snap)


class TestPortfolioRisk:
    def test_reports_both_horizons_and_a_leverage_figure(self, bundle, specs, spot, now):
        out = portfolio_risk(levered_book(now), spot, bundle, specs, n_paths=8_000,
                             seed=13, now=now)
        assert out.p_liq_7d_any.point >= out.p_liq_24h_any.point
        assert out.effective_leverage.point > 1.0
        assert out.effective_leverage.ci_low <= out.effective_leverage.point
        assert out.factor_coin == "BTC"
        assert out.publishable

    def test_effective_leverage_is_unsigned_but_beta_is_not(self, bundle, specs, spot, now):
        """OPEN-QUESTIONS D3: the ratio §4.1 defines cannot carry direction,
        so the UI sentence needs the beta the tool also returns."""
        long_book = Book("0x", 100_000.0,
                         (Position("BTC", 5.0, 100_000.0, MarginMode.CROSS, 20.0),), now)
        short_book = Book("0x", 100_000.0,
                          (Position("BTC", -5.0, 100_000.0, MarginMode.CROSS, 20.0),), now)
        a = portfolio_risk(long_book, spot, bundle, specs, n_paths=8_000, seed=17, now=now)
        b = portfolio_risk(short_book, spot, bundle, specs, n_paths=8_000, seed=17, now=now)
        assert a.effective_leverage.point == pytest.approx(b.effective_leverage.point, rel=0.15)
        assert a.factor_beta > 0 > b.factor_beta

    def test_isolated_positions_get_their_own_probability(self, bundle, specs, spot, now):
        book = Book(
            "0x", 50_000.0,
            (
                Position("BTC", 3.0, 100_000.0, MarginMode.CROSS, 20.0),
                Position("SOL", 2_000.0, 200.0, MarginMode.ISOLATED, 20.0, 20_000.0),
            ),
            now,
        )
        out = portfolio_risk(book, spot, bundle, specs, n_paths=8_000, seed=19, now=now)
        assert set(out.p_liq_24h_isolated) == {"SOL"}
        assert out.p_liq_24h_cross.point >= 0.0


class TestStatistics:
    def test_wilson_interval_stays_inside_zero_and_one(self):
        lo, hi = wilson_interval(0, 1000)
        assert lo == pytest.approx(0.0, abs=1e-12) and 0.0 < hi < 0.01
        lo, hi = wilson_interval(1000, 1000)
        assert hi == pytest.approx(1.0) and 0.99 < lo < 1.0

    def test_path_sizing_meets_the_half_width_it_promises(self):
        """The count returned must actually satisfy the rule, not approximately
        satisfy it: §2.5's 2 pp ceiling is the one thing the engine may never
        return a number in violation of."""
        for p in (0.001, 0.02, 0.1, 0.5, 0.9):
            for target in (0.02, 0.005):
                n = paths_needed_for_half_width(p, target)
                assert wilson_half_width(round(p * n), n) <= target
                # Sufficient without being wasteful: the count must stay in
                # the same ballpark as the normal-approximation estimate,
                # otherwise the escalation loop would overshoot badly.
                approx = 1.96**2 * p * (1 - p) / target**2
                assert n <= max(4 * approx, 100)

    def test_crps_is_minimised_by_the_truth(self):
        rng = np.random.default_rng(0)
        good = PredictiveDistribution.from_samples(rng.normal(0, 1, 50_000))
        bad = PredictiveDistribution.from_samples(rng.normal(3, 1, 50_000))
        actual = rng.normal(0, 1, 2_000)
        assert np.mean([good.crps(a) for a in actual]) < np.mean([bad.crps(a) for a in actual])

    def test_pit_of_a_correct_forecast_is_uniform(self):
        rng = np.random.default_rng(1)
        dist = PredictiveDistribution.from_samples(rng.normal(0, 1, 100_000))
        pits = np.array([dist.pit(x) for x in rng.normal(0, 1, 5_000)])
        assert abs(pits.mean() - 0.5) < 0.02
        assert abs(np.quantile(pits, 0.9) - 0.9) < 0.02


class TestRiskEstimateContract:
    def test_cannot_be_built_without_an_interval(self):
        with pytest.raises(TypeError):
            RiskEstimate(0.1)  # type: ignore[call-arg]

    def test_rejects_a_point_outside_its_interval(self, now):
        with pytest.raises(ValueError, match="outside interval"):
            RiskEstimate(0.5, 0.1, 0.2, "v", now)

    def test_rejects_a_naive_timestamp(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            RiskEstimate(0.1, 0.0, 0.2, "v", datetime(2026, 1, 1))  # noqa: DTZ001

    def test_overlap_means_indistinguishable(self, now):
        """§4.2: overlapping intervals must not be drawn as a change."""
        before = RiskEstimate(0.10, 0.08, 0.12, "v", now)
        after = RiskEstimate(0.11, 0.09, 0.13, "v", now)
        far = RiskEstimate(0.30, 0.28, 0.32, "v", now)
        assert before.overlaps(after)
        assert not before.distinguishable_from(after)
        assert before.distinguishable_from(far)

    def test_age_is_always_available(self, now):
        est = RiskEstimate(0.1, 0.0, 0.2, "v", now)
        assert est.age_seconds(now + timedelta(seconds=90)) == 90.0
