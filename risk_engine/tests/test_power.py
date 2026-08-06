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

from risk_engine.sim.stats import clustered_mean_ci
from risk_engine.validation.power import (
    NOMINAL_BREACH_RATE,
    clustered_rate_ci,
    evaluate,
    power_at,
    power_ci_at,
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
        drifts from `clustered_mean_ci` the whole table is measuring
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
        general = clustered_mean_ci(
            values, ids, np.random.default_rng(9), n_boot=4_000,
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

    def test_power_at_matches_the_tabulated_columns(self):
        """`power_at` exists so window sizing is never quantised to the two
        rates the table shows (audit F-1). It must agree with the table at
        the rates the table does show."""
        cell = evaluate(days=30, addresses_per_day=200, icc=0.10,
                        n_trials=200, n_boot=400, seed=21)
        at_10 = power_at(30, 200, 0.10, 0.10, n_trials=200, n_boot=400, seed=99)
        assert at_10 == pytest.approx(cell.power_at_10pct, abs=0.12)

    def test_power_at_is_monotone_in_the_target(self):
        """The property F-1 violated: a true rate closer to the nominal 5% is
        harder to detect, so power must fall as the target tightens."""
        powers = [
            power_at(30, 200, 0.10, p, n_trials=250, n_boot=400, seed=31)
            for p in (0.06, 0.08, 0.12)
        ]
        assert powers[0] < powers[1] < powers[2]

    def test_power_at_refuses_a_degenerate_target(self):
        with pytest.raises(ValueError, match="false-rejection"):
            power_at(30, 200, 0.10, NOMINAL_BREACH_RATE, 10, 50, 0)
        with pytest.raises(ValueError, match="rate in"):
            power_at(30, 200, 0.10, 1.5, 10, 50, 0)

    def test_power_ci_at_agrees_with_power_at_on_the_point(self):
        """Both walk the identical simulation (`_power_trials`); they must
        not drift into reporting different points for the same inputs."""
        point = power_at(45, 200, 0.15, 0.10, n_trials=150, n_boot=300, seed=7)
        ci_point, lo, hi = power_ci_at(45, 200, 0.15, 0.10, n_trials=150, n_boot=300, seed=7)
        assert ci_point == point
        assert lo <= point <= hi

    def test_power_ci_at_bound_narrows_with_more_trials(self):
        """The whole reason this exists: the interval is what tells a caller
        whether the point estimate is a decision or a coin flip."""
        _, lo_small, hi_small = power_ci_at(45, 200, 0.15, 0.10, n_trials=50,
                                            n_boot=200, seed=3)
        _, lo_big, hi_big = power_ci_at(45, 200, 0.15, 0.10, n_trials=800,
                                        n_boot=200, seed=3)
        assert (hi_big - lo_big) < (hi_small - lo_small)

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


class TestTheClusteredIntervalActuallyCovers:
    """Nobody had measured the one property an interval exists to have.

    The shipped interval was a PERCENTILE bootstrap over days. Resampling
    days is necessary — B1's whole argument — and it is not sufficient: a
    percentile bootstrap undercovers badly when the clusters are few, and 21
    days is few. Measured at 200 addresses/day against a nominal 95%:

    | ICC  | percentile (was) | studentised (now) |
    |------|------------------|-------------------|
    | 0.00 |  94.8%           |  95.2%            |
    | 0.10 |  89.0%           |  94.0%            |
    | 0.20 |  82.8%           |  94.5%            |
    | 0.40 |  79.2%           |  91.2%            |

    Two costs, and the second is worse. A gate criterion read off an interval
    covering 83% rejects a correct model ~17% of the time instead of 5%. And
    `clustered_half_width_pp` — the column B1 sizes the window from — was
    taken from that interval: 3.58pp at ICC 0.20 where the honest figure is
    7.31pp, so a window sized off it looks informative and is not.

    B1 records this exact lesson, learned on the ICC estimator in
    `clustering.py` ("the day-clustered percentile bootstrap covers 43% ...
    Both were discarded"). It never reached this interval.
    """

    TRIALS = 200

    def _coverage(self, icc, days=21, per_day=200, seed=0):
        rng = np.random.default_rng(seed)
        hits = 0
        for t in range(self.TRIALS):
            sums, counts = simulate_days(days, per_day, NOMINAL_BREACH_RATE, icc, rng)
            lo, hi = clustered_rate_ci(sums, counts,
                                       np.random.default_rng(7717 + t), n_boot=200)
            hits += lo <= NOMINAL_BREACH_RATE <= hi
        return hits / self.TRIALS

    @pytest.mark.parametrize("icc", [0.0, 0.10, 0.20])
    def test_it_covers_at_about_its_nominal_level(self, icc):
        """0.90 rather than 0.95 as the bar: 200 trials carry a standard
        error near 1.5pp, so a tighter threshold would be flaky. It is still
        far above the 82.8% the percentile version scored at ICC 0.20, which
        is the regression this guards."""
        assert self._coverage(icc) >= 0.90, icc

    def test_the_naive_interval_does_not_cover_at_all(self):
        """The contrast that justifies the machinery. §0.3's binomial interval
        assumes independence; at ICC 0.20 it contains the true rate about a
        quarter of the time, which is B1's recorded 77.7% false rejection
        seen from the other side."""
        from risk_engine.sim.stats import wilson_interval

        rng = np.random.default_rng(1)
        hits = 0
        for _ in range(self.TRIALS):
            sums, counts = simulate_days(21, 200, NOMINAL_BREACH_RATE, 0.20, rng)
            lo, hi = wilson_interval(int(sums.sum()), int(counts.sum()))
            hits += lo <= NOMINAL_BREACH_RATE <= hi
        assert hits / self.TRIALS < 0.5

    def test_a_degenerate_sample_gets_a_point_not_a_fabricated_width(self):
        """Every day identical means there is no between-day variation to
        studentise against. A width invented from nothing would be worse than
        admitting the interval is a point."""
        sums = np.zeros(21, dtype=np.int64)
        counts = np.full(21, 200, dtype=np.int64)
        lo, hi = clustered_rate_ci(sums, counts, np.random.default_rng(0), n_boot=50)
        assert lo == hi == 0.0

    def test_it_refuses_a_single_day(self):
        """One cluster cannot bound between-cluster variation, and a silent
        answer there would be the undercoverage this class exists to stop,
        taken to its limit."""
        with pytest.raises(ValueError, match="two days"):
            clustered_rate_ci(np.array([10]), np.array([200]),
                              np.random.default_rng(0))
