"""The B1 power analysis, checked against cases with known answers.

A simulation that produces a decision-shaped table is worth exactly as much
as its calibration. These tests pin the three things that would make the
table wrong while still looking plausible: the generator has to produce the
correlation it was asked for, the fast interval has to equal the general one
it stands in for, and the naive criterion has to behave correctly in the one
regime where it *is* correct -- independent observations -- because if it
does not, every clustered number is measuring a broken baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

from risk_engine.sim.stats import clustered_bootstrap_ci
from risk_engine.validation.power import (
    NOMINAL_BREACH_RATE,
    clustered_rate_ci,
    evaluate,
    simulate_days,
)


class TestGenerator:
    def test_the_marginal_rate_is_the_one_requested(self):
        """Varying the clustering must not move the average breach rate --
        otherwise the sweep would be comparing models with different tail
        behaviour, not the same model observed differently."""
        rng = np.random.default_rng(0)
        for icc in (0.0, 0.05, 0.2, 0.5):
            sums, counts = simulate_days(4_000, 200, NOMINAL_BREACH_RATE, icc, rng)
            assert sums.sum() / counts.sum() == pytest.approx(
                NOMINAL_BREACH_RATE, abs=0.004
            ), icc

    def test_the_realised_correlation_matches_the_requested_one(self):
        """The swept parameter has to be the quantity it is labelled as. The
        intra-class correlation of a beta-binomial shows up as variance
        inflation over the binomial: Var = n·p(1-p)·[1 + (n-1)·icc]."""
        rng = np.random.default_rng(1)
        n, days = 200, 20_000
        for icc in (0.05, 0.2, 0.4):
            sums, _ = simulate_days(days, n, NOMINAL_BREACH_RATE, icc, rng)
            binomial_var = n * NOMINAL_BREACH_RATE * (1 - NOMINAL_BREACH_RATE)
            inflation = float(np.var(sums)) / binomial_var
            assert inflation == pytest.approx(1 + (n - 1) * icc, rel=0.12), icc

    def test_zero_correlation_is_exactly_binomial(self):
        rng = np.random.default_rng(2)
        n = 200
        sums, _ = simulate_days(20_000, n, NOMINAL_BREACH_RATE, 0.0, rng)
        expected = n * NOMINAL_BREACH_RATE * (1 - NOMINAL_BREACH_RATE)
        assert float(np.var(sums)) == pytest.approx(expected, rel=0.06)

    def test_total_correlation_makes_the_day_the_only_unit(self):
        """At icc=1 a day is all-or-nothing, so the effective sample size is
        the day count however many addresses are watched."""
        rng = np.random.default_rng(3)
        sums, counts = simulate_days(2_000, 200, NOMINAL_BREACH_RATE, 1.0, rng)
        assert set(np.unique(sums)) <= {0, 200}
        assert sums.sum() / counts.sum() == pytest.approx(NOMINAL_BREACH_RATE, abs=0.01)


class TestIntervalAgreement:
    def test_the_fast_interval_equals_the_general_clustered_one(self):
        """`clustered_rate_ci` is a specialisation for a ratio of sums. If it
        drifts from `clustered_bootstrap_ci` the whole table is measuring
        something other than the interval the gate is actually read from."""
        rng = np.random.default_rng(4)
        sums, counts = simulate_days(30, 200, NOMINAL_BREACH_RATE, 0.15, rng)

        fast = clustered_rate_ci(sums, counts, np.random.default_rng(9), n_boot=4_000)

        # The same data as one observation per address, with its day as the
        # cluster id -- the form the general implementation takes.
        values = np.concatenate([
            np.concatenate([np.ones(s), np.zeros(c - s)])
            for s, c in zip(sums, counts, strict=True)
        ])
        ids = np.concatenate([
            np.full(c, d) for d, c in enumerate(counts)
        ])
        general = clustered_bootstrap_ci(
            values, ids, lambda v: float(v.mean()), np.random.default_rng(9),
            n_boot=4_000,
        )
        assert fast[0] == pytest.approx(general[0], abs=0.002)
        assert fast[1] == pytest.approx(general[1], abs=0.002)

    def test_the_interval_widens_with_clustering(self):
        rng = np.random.default_rng(5)
        widths = []
        for icc in (0.0, 0.1, 0.4):
            sums, counts = simulate_days(21, 200, NOMINAL_BREACH_RATE, icc, rng)
            lo, hi = clustered_rate_ci(sums, counts, rng, n_boot=2_000)
            widths.append(hi - lo)
        assert widths[0] < widths[1] < widths[2]


class TestCalibration:
    def test_the_naive_criterion_is_correct_when_observations_are_independent(self):
        """The baseline the whole argument rests on. §0.3's interval is not
        wrong in general -- it is wrong under clustering. If it misbehaved at
        icc=0 the harness would be measuring its own bug."""
        cell = evaluate(days=21, addresses_per_day=200, icc=0.0,
                        n_trials=400, n_boot=400, seed=11)
        assert cell.naive_false_rejection == pytest.approx(0.05, abs=0.035)

    def test_clustering_breaks_the_naive_criterion(self):
        """B1's claim, measured rather than argued: a correctly calibrated
        model is rejected far more often than the 5% the interval promises."""
        cell = evaluate(days=21, addresses_per_day=200, icc=0.2,
                        n_trials=200, n_boot=400, seed=12)
        assert cell.naive_false_rejection > 0.5

    def test_the_honest_interval_cannot_detect_real_miscalibration_in_21_days(self):
        """The other half of B1, and the reason "use the clustered interval"
        is not on its own a fix. A model whose true breach rate is 10% --
        double what it claims -- is not reliably caught in 21 days."""
        cell = evaluate(days=21, addresses_per_day=200, icc=0.2,
                        n_trials=200, n_boot=400, seed=13)
        assert cell.power_at_10pct < 0.8
        assert cell.clustered_half_width_pp > 2.0

    def test_a_longer_window_recovers_power(self):
        """Which is what makes the window length the decision, rather than
        the criterion being unusable in principle."""
        short = evaluate(days=21, addresses_per_day=200, icc=0.2,
                         n_trials=150, n_boot=400, seed=14)
        long = evaluate(days=180, addresses_per_day=200, icc=0.2,
                        n_trials=150, n_boot=400, seed=14)
        assert long.power_at_10pct > short.power_at_10pct
        assert long.clustered_half_width_pp < short.clustered_half_width_pp

    def test_more_addresses_barely_help_under_clustering(self):
        """The finding that decides how to spend effort: at a realistic
        intra-day correlation the day is the unit, so 500 addresses buy very
        little over 200. Sampling harder is not a substitute for waiting."""
        few = evaluate(days=21, addresses_per_day=200, icc=0.2,
                       n_trials=150, n_boot=400, seed=15)
        many = evaluate(days=21, addresses_per_day=500, icc=0.2,
                        n_trials=150, n_boot=400, seed=15)
        # Under independence 2.5x the sample would cut the half-width by ~37%.
        assert many.clustered_half_width_pp > 0.85 * few.clustered_half_width_pp
