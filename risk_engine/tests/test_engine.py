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
    ks_uniformity,
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
        # Audit A-12: the previous form was `assert ... if hasattr(...) else True`
        # against an attribute RiskResult does not have, i.e. vacuously true.
        assert full.p_liq_any.distinguishable_from(indep.p_liq_any)

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
        pits = np.array([dist.pit(x, rng.random()) for x in rng.normal(0, 1, 5_000)])
        assert abs(pits.mean() - 0.5) < 0.02
        assert abs(np.quantile(pits, 0.9) - 0.9) < 0.02

    def test_randomized_pit_stays_uniform_when_the_forecast_has_an_atom(self):
        """Audit A-02, the killer test.

        The liquidation model writes a wiped account to exactly zero equity
        (§1.6), so the predicted distribution carries a mass point. A plain
        `F(x)` maps every realised liquidation to one identical value and KS
        rejects a by-construction-perfect forecast at p ~ 1e-104. The
        randomized transform must not.
        """
        rng = np.random.default_rng(11)

        def draw(n):  # 20% atom at -1000, otherwise N(0, 100)
            x = rng.normal(0, 100, n)
            x[rng.random(n) < 0.20] = -1000.0
            return x

        dist = PredictiveDistribution.from_samples(draw(200_000))
        actual = draw(3_000)
        pits = np.array([dist.pit(x, rng.random()) for x in actual])

        # The atom's PIT values must be spread, not a single point.
        atom_pits = pits[actual == -1000.0]
        assert atom_pits.std() > 0.01
        assert len(np.unique(atom_pits)) > 100

        _stat, p = ks_uniformity(pits)
        assert p > 0.01, f"KS rejected a perfectly calibrated forecast: p={p:.3g}"

    def test_cdf_interval_exposes_the_atom(self):
        dist = PredictiveDistribution.from_samples(
            np.concatenate([np.full(2_000, -50.0), np.linspace(0.0, 100.0, 8_000)])
        )
        lo, hi = dist.cdf_interval(-50.0)
        assert hi - lo > 0.15  # ~20% of mass sits on the atom
        lo2, hi2 = dist.cdf_interval(50.0)
        assert lo2 == pytest.approx(hi2)  # continuous region: no interval

    def test_pit_rejects_an_out_of_range_uniform(self):
        dist = PredictiveDistribution.from_samples(np.linspace(-1, 1, 1001))
        with pytest.raises(ValueError, match=r"u must be in"):
            dist.pit(0.0, 1.5)

    def test_non_finite_quantiles_are_refused(self):
        """Audit A-07: NaN passes the non-decreasing check, since every
        comparison against NaN is False."""
        with pytest.raises(ValueError, match="finite"):
            PredictiveDistribution.from_samples(np.full(100, np.nan))


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


class TestBaselineBIsActuallyIndependent:
    """Audit A-03. rho=0 under a t-copula is not independence: the shared
    chi-square mixer still drives every asset into its tail together. §3.2's
    baseline B must remove dependence, not merely decorrelate it -- otherwise
    the null hypothesis secretly contains the effect being tested, and §3.2's
    prescribed conclusion ('B wins, so drop the correlation machinery') would
    be drawn from a baseline built on that same machinery."""

    def test_independent_paths_have_no_joint_tail_dependence(self, bundle, specs, now):
        from risk_engine.sim.paths import draw_base_randomness, generate_log_returns

        spec = bundle.path_spec(("BTC", "ETH", "SOL"), independent=True)
        assert spec.copula_df is None, "a t-copula cannot express independence"

        base = draw_base_randomness(200_000, 1, 3, 0, spec.copula_df,
                                    np.random.default_rng(5))
        r = generate_log_returns(spec, base)[:, 0, :]
        for i, j in ((0, 1), (0, 2), (1, 2)):
            qi = np.quantile(r[:, i], 0.05)
            qj = np.quantile(r[:, j], 0.05)
            in_i = r[:, i] <= qi
            joint = float((in_i & (r[:, j] <= qj)).sum() / in_i.sum())
            assert joint == pytest.approx(0.05, abs=0.01), f"pair {(i, j)}: {joint:.4f}"

    def test_the_full_model_still_uses_the_t_copula(self, bundle):
        assert bundle.path_spec(("BTC", "ETH"), independent=False).copula_df == bundle.copula_df

    def test_marginals_are_untouched_by_the_independence_switch(self, bundle):
        dep = bundle.path_spec(("BTC", "ETH", "SOL"), independent=False)
        ind = bundle.path_spec(("BTC", "ETH", "SOL"), independent=True)
        assert dep.marginal_df == ind.marginal_df
        assert np.allclose(dep.step_vol, ind.step_vol)


class TestHorizonMonotonicity:
    """Audit A-10: two `run` calls with the same seed do NOT share paths --
    the (paths, steps, assets) draw shape differs, so the streams diverge
    after the first path and a 7d probability could land below the 24h one."""

    def test_shared_walk_makes_liquidation_flags_nested(self, bundle, specs, spot, now):
        results = MonteCarloEngine(bundle, specs).run_horizons(
            levered_book(now, 18.0), spot, (24, 168), n_paths=8_000, seed=21, now=now
        )
        day, week = results[24], results[168]
        assert week.p_liq_any.point >= day.p_liq_any.point
        assert week.p_liq_cross.point >= day.p_liq_cross.point

    def test_portfolio_risk_never_inverts_the_horizons(self, bundle, specs, spot, now):
        for lev in (4.0, 8.0, 15.0, 22.0):
            out = portfolio_risk(levered_book(now, lev), spot, bundle, specs,
                                 n_paths=4_000, seed=23, now=now)
            assert out.p_liq_7d_any.point >= out.p_liq_24h_any.point, lev

    def test_separate_runs_do_not_share_paths(self):
        """The fact the old comment got wrong, pinned so it cannot creep back."""
        from risk_engine.sim.paths import draw_base_randomness

        a = draw_base_randomness(4, 24, 1, 0, None, np.random.default_rng(7))
        b = draw_base_randomness(4, 168, 1, 0, None, np.random.default_rng(7))
        assert np.allclose(a.z[0, :24, 0], b.z[0, :24, 0])   # path 0 coincides
        assert not np.allclose(a.z[1, :24, 0], b.z[1, :24, 0])  # everything after diverges


class TestMetricsAreBounded:
    """Audit A-11: this is a long-running service; unbounded sample lists are
    a slow leak with no ceiling."""

    def test_the_latency_bound_is_declared_and_sane(self):
        """Checked as a property of the container, not by overflowing it: a
        mutant that sets the cap to 1e9 is still nominally 'bounded', and a
        test that loops to the cap would hang rather than fail."""
        from risk_engine.observability.metrics import MAX_EVENT_SAMPLES, MAX_LATENCY_SAMPLES

        m = Metrics()
        m.observe_latency("stage", 1.0)
        assert m.latencies_ms["stage"].maxlen == MAX_LATENCY_SAMPLES
        assert m.psd_corrections.maxlen == MAX_EVENT_SAMPLES
        assert m.df_clamps.maxlen == MAX_EVENT_SAMPLES
        # A bound large enough to be a leak in its own right is not a bound.
        assert 0 < MAX_LATENCY_SAMPLES <= 100_000
        assert 0 < MAX_EVENT_SAMPLES <= 100_000

    def test_latency_window_discards_the_oldest_samples(self):
        from risk_engine.observability.metrics import MAX_LATENCY_SAMPLES

        # Guard before the loop: without it, a mutant that raises the cap
        # makes this test hang for a billion iterations instead of failing.
        assert MAX_LATENCY_SAMPLES <= 100_000
        m = Metrics()
        for i in range(MAX_LATENCY_SAMPLES + 500):
            m.observe_latency("stage", float(i))
        assert len(m.latencies_ms["stage"]) == MAX_LATENCY_SAMPLES
        # The window keeps the most recent samples, not the oldest.
        assert m.percentiles("stage")["p99"] > MAX_LATENCY_SAMPLES

    def test_event_lists_are_capped_but_counters_are_not(self):
        from risk_engine.model.psd import project_to_correlation
        from risk_engine.observability.metrics import MAX_EVENT_SAMPLES

        m = Metrics()
        bad = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
        for _ in range(MAX_EVENT_SAMPLES + 50):
            project_to_correlation(bad, metrics=m)
        assert len(m.psd_corrections) == MAX_EVENT_SAMPLES
        # The lifetime tally survives even though the samples rolled over.
        assert m.counters["psd_projection_corrections"] == MAX_EVENT_SAMPLES + 50


class TestParallelBlocks:
    """OPEN-QUESTIONS D7. Threads are only admissible here because they do
    not change what the model predicts -- only which sample is drawn from
    it. That distinction is what keeps this a PATCH release and leaves the
    shadow window intact (§3.3)."""

    def test_blocks_are_planned_for_both_memory_and_parallelism(self):
        from risk_engine.sim.engine import plan_chunks

        # One block per worker when memory allows: the previous sizing
        # produced a single block and left every other core idle.
        assert plan_chunks(20_000, 75, workers=4) == [5_000] * 4
        assert plan_chunks(20_000, 75, workers=1) == [20_000]
        # Memory still wins when a block would be too large.
        wide = plan_chunks(200_000, 4_000, workers=4)
        assert max(wide) <= 4_000_000 // 4_000
        assert sum(wide) == 200_000
        # Balanced, so no worker waits on a straggler.
        uneven = plan_chunks(10_001, 75, workers=4)
        assert sum(uneven) == 10_001
        assert max(uneven) - min(uneven) <= 1

    def test_streams_are_independent_and_reproducible(self):
        from risk_engine.sim.engine import spawn_streams

        a = [g.standard_normal(5) for g in spawn_streams(42, 4)]
        b = [g.standard_normal(5) for g in spawn_streams(42, 4)]
        for x, y in zip(a, b, strict=True):
            assert np.array_equal(x, y)
        # Distinct blocks must not replay the same numbers.
        assert not np.array_equal(a[0], a[1])

    def test_the_result_does_not_depend_on_completion_order(self, bundle, specs, spot, now):
        """Every block's randomness is decided before any thread starts, so
        two runs agree exactly however the threads interleave."""
        engine = MonteCarloEngine(bundle, specs, workers=4)
        a = engine.run(levered_book(now, 18.0), spot, 24, n_paths=8_000, seed=5, now=now)
        b = engine.run(levered_book(now, 18.0), spot, 24, n_paths=8_000, seed=5, now=now)
        assert a.p_liq_any.point == b.p_liq_any.point
        assert np.array_equal(a.raw_equity_change, b.raw_equity_change)

    def test_worker_count_changes_the_sample_but_not_the_answer(
        self, bundle, specs, spot, now
    ):
        """Different blocking means different draws from the SAME law. The
        estimates must therefore agree to within Monte Carlo error, not
        bit-for-bit -- and this is exactly why the change is a PATCH."""
        book = levered_book(now, 18.0)
        serial = MonteCarloEngine(bundle, specs, workers=1).run(
            book, spot, 24, n_paths=20_000, seed=5, now=now
        )
        threaded = MonteCarloEngine(bundle, specs, workers=4).run(
            book, spot, 24, n_paths=20_000, seed=5, now=now
        )
        assert serial.p_liq_any.point != threaded.p_liq_any.point
        # Two independent 20k estimates: 3 standard errors is a generous bound.
        se = (serial.p_liq_any.point * (1 - serial.p_liq_any.point) / 20_000) ** 0.5
        assert abs(serial.p_liq_any.point - threaded.p_liq_any.point) < 4 * max(se, 1e-4)

    def test_the_benchmarks_are_untouched_by_blocking(self):
        """The §3.1 gate drives the simulator directly with an explicit
        BaseRandomness, so it never goes through the block planner and its
        numbers are unaffected by any of this."""
        import inspect

        from risk_engine.validation import benchmarks

        source = inspect.getsource(benchmarks)
        assert "run_blocks" not in source
        assert "simulate_paths(" in source
