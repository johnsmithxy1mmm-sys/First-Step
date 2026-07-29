"""The clustering estimators, against data whose correlation is known.

This number decides how long the shadow window has to be, so an estimator
that is quietly biased low would pick a short window and nothing downstream
would notice — a gate sized off a too-low ICC passes while establishing
nothing, which is §10's forbidden direction reached by arithmetic. Every
test here generates data with a correlation it chose and asks whether the
estimator finds it back, and the interval tests measure *coverage* rather
than checking that some interval exists.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from risk_engine.shadow.clustering import (
    NOMINAL_BREACH_RATE,
    breach_icc_confidence_set,
    breach_icc_from_latent,
    estimate_breach_icc,
    estimate_latent_correlation,
    recommend_window,
)
from risk_engine.validation.power import simulate_days

BREACH_Z = stats.norm.ppf(NOMINAL_BREACH_RATE)


def latent_world(n_days: int, n_per_day: int, rho: float, seed: int):
    """A day's common shock plus per-address noise, on the latent scale.

    Returns (pit, breached, days). This is the structure the estimator
    assumes, so recovery tests use it; the assumption-free tests below use
    the independent beta-binomial generator instead.
    """
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal((n_days, 1))
    idio = rng.standard_normal((n_days, n_per_day))
    z = np.sqrt(rho) * factor + np.sqrt(1.0 - rho) * idio
    pit = stats.norm.cdf(z).ravel()
    breached = (z < BREACH_Z).astype(float).ravel()
    days = np.repeat(np.arange(n_days), n_per_day)
    return pit, breached, days


class TestLatentCorrelation:
    @pytest.mark.parametrize("rho", [0.05, 0.15, 0.30, 0.50])
    def test_it_recovers_the_correlation_it_was_given(self, rho):
        """Averaged over datasets, because the claim is about the estimator
        and not about one draw: at 30 days the standard deviation of a single
        estimate at rho=0.3 is around 0.06, so a per-seed tolerance tight
        enough to be meaningful would fail on noise."""
        estimates = [
            estimate_latent_correlation(
                *latent_world(30, 200, rho, seed=int(rho * 1000) + s)[::2],
                np.random.default_rng(s), n_boot=100,
            ).point
            for s in range(8)
        ]
        assert float(np.mean(estimates)) == pytest.approx(rho, rel=0.15)

    def test_independent_days_estimate_near_zero(self):
        pit, _, days = latent_world(30, 200, 1e-9, seed=5)
        est = estimate_latent_correlation(pit, days, np.random.default_rng(2), n_boot=200)
        assert est.point < 0.02

    def test_the_interval_covers_at_pilot_length(self):
        """The property the whole pilot rests on: the number the window is
        sized from must not be confidently wrong at fourteen days."""
        covered = 0
        trials = 40
        for s in range(trials):
            pit, _, days = latent_world(14, 200, 0.30, seed=300 + s)
            est = estimate_latent_correlation(pit, days, np.random.default_rng(s), n_boot=1_000)
            if est.ci_low <= 0.30 <= est.ci_high:
                covered += 1
        assert covered / trials >= 0.85

    def test_the_bootstrap_alternative_undercovers_and_is_not_the_default(self):
        """Measured, and the reason the default is the inverted test. The
        day-clustered bootstrap covers about three quarters of the time at
        fourteen days against a nominal 95%, and it fails *low* — a ceiling
        that is too low shortens the window, which is the §10-forbidden
        direction reached by arithmetic."""
        covered = 0
        trials = 30
        for s in range(trials):
            pit, _, days = latent_world(14, 200, 0.30, seed=600 + s)
            est = estimate_latent_correlation(
                pit, days, np.random.default_rng(s), n_boot=400, method="bootstrap"
            )
            if est.ci_low <= 0.30 <= est.ci_high:
                covered += 1
        assert covered / trials < 0.95

    def test_an_unknown_interval_method_is_refused(self):
        pit, _, days = latent_world(4, 100, 0.2, seed=1)
        with pytest.raises(ValueError, match="unknown method"):
            estimate_latent_correlation(pit, days, method="jackknife")

    def test_one_day_is_refused_rather_than_guessed(self):
        with pytest.raises(ValueError, match="at least two days"):
            estimate_latent_correlation(np.array([0.4, 0.6]), np.array(["d", "d"]))

    def test_mismatched_arrays_are_refused(self):
        with pytest.raises(ValueError, match="same length"):
            estimate_latent_correlation(np.array([0.4, 0.6]), np.array(["a"]))

    def test_degenerate_pit_values_do_not_produce_infinities(self):
        """A PIT of exactly 0 or 1 is an infinity on the latent scale, and an
        infinity there would silently poison every moment."""
        rng = np.random.default_rng(9)
        pit = rng.random(4_000)
        pit[:50] = 0.0
        pit[50:100] = 1.0
        days = np.repeat(np.arange(20), 200)
        est = estimate_latent_correlation(pit, days, np.random.default_rng(9), n_boot=100)
        assert np.isfinite(est.point)
        assert 0.0 <= est.point <= 1.0


class TestLatentToBreachMap:
    @pytest.mark.parametrize("rho", [0.05, 0.15, 0.30, 0.50])
    def test_the_gaussian_map_matches_the_empirical_breach_icc(self, rho):
        """The map is the load-bearing step: it is what lets a precise latent
        estimate stand in for an imprecise breach one. Checked against the
        breach ICC actually realised in latent-world data."""
        from risk_engine.shadow.clustering import _icc_from_counts

        realised = []
        for s in range(8):
            _, breached, days = latent_world(200, 200, rho, seed=int(rho * 777) + s)
            _, index = np.unique(days, return_inverse=True)
            counts = np.bincount(index)
            sums = np.bincount(index, weights=breached)
            value, _ = _icc_from_counts(sums, counts)
            realised.append(value)
        predicted = breach_icc_from_latent(rho, copula="gaussian")
        assert predicted == pytest.approx(float(np.mean(realised)), abs=0.02)

    def test_the_map_compresses(self):
        """Why the required window is shorter than "clustering is obviously
        severe" suggests: strong latent co-movement produces much weaker
        correlation between rare breach events."""
        assert breach_icc_from_latent(0.30, copula="gaussian") < 0.30 / 2
        assert breach_icc_from_latent(0.50, copula="gaussian") < 0.50 / 2

    def test_the_t_copula_maps_higher_than_the_gaussian(self):
        """Tail dependence makes joint breaches likelier at the same
        correlation, so the t map lengthens the window. That is why it is the
        default: between two stated assumptions, §10 permits the one that
        does not shorten validation."""
        for rho in (0.10, 0.30, 0.50):
            assert breach_icc_from_latent(rho, copula="t") > breach_icc_from_latent(
                rho, copula="gaussian"
            )

    def test_it_is_monotone_in_the_latent_correlation(self):
        values = [breach_icc_from_latent(r, copula="gaussian") for r in (0.0, 0.1, 0.3, 0.6, 0.9)]
        assert values == sorted(values)

    def test_zero_correlation_maps_to_zero(self):
        assert breach_icc_from_latent(0.0, copula="gaussian") == pytest.approx(0.0, abs=0.01)

    def test_an_unknown_copula_is_refused(self):
        with pytest.raises(ValueError, match="unknown copula"):
            breach_icc_from_latent(0.2, copula="clayton")


class TestBreachIcc:
    def test_it_reports_the_design_effect_that_defeats_the_naive_interval(self):
        """B1 in one number: the naive interval thinks it has 2800
        observations and it does not."""
        pit, breached, days = latent_world(14, 200, 0.30, seed=21)
        icc = estimate_breach_icc(pit, breached, days, np.random.default_rng(21), n_boot=200)
        assert icc.n_observations == 2_800
        assert icc.design_effect > 5
        assert icc.effective_sample_size < icc.n_observations / 5

    def test_the_direct_estimate_is_carried_but_not_used_for_sizing(self):
        pit, breached, days = latent_world(30, 200, 0.30, seed=22)
        icc = estimate_breach_icc(pit, breached, days, np.random.default_rng(22), n_boot=200)
        assert icc.direct_point is not None
        assert "not for sizing" in icc.summary()

    def test_the_interval_is_ordered_and_maps_through(self):
        pit, breached, days = latent_world(30, 200, 0.30, seed=23)
        icc = estimate_breach_icc(pit, breached, days, np.random.default_rng(23), n_boot=200)
        assert icc.ci_low <= icc.point <= icc.ci_high
        assert icc.point == pytest.approx(
            breach_icc_from_latent(icc.latent.point, copula="t"), abs=0.02
        )


class TestAssumptionFreeInterval:
    def test_it_covers_where_the_bootstrap_did_not(self):
        """Measured: the day-clustered percentile bootstrap covers 43% at 14
        days against a nominal 95%, and the normal-theory F interval 66%.
        Both fail low, which shortens the window. This one is built by
        inverting the test against the validated generator."""
        covered = 0
        trials = 20
        rng = np.random.default_rng(0)
        for s in range(trials):
            r = np.random.default_rng(4_000 + s)
            sums, counts = simulate_days(14, 200, NOMINAL_BREACH_RATE, 0.20, r)
            breached = np.concatenate([
                np.concatenate([np.ones(k), np.zeros(n - k)])
                for k, n in zip(sums, counts, strict=True)
            ])
            days = np.concatenate([np.full(n, d) for d, n in enumerate(counts)])
            lo, hi = breach_icc_confidence_set(breached, days, rng, n_sims=120)
            if lo <= 0.20 <= hi:
                covered += 1
        assert covered / trials >= 0.85

    def test_it_is_wide_enough_to_be_useless_for_sizing(self):
        """Not a defect — the finding. This is what the breaches alone
        support at pilot length, and it is why the PIT route exists."""
        rng = np.random.default_rng(1)
        sums, counts = simulate_days(14, 200, NOMINAL_BREACH_RATE, 0.20,
                                     np.random.default_rng(77))
        breached = np.concatenate([
            np.concatenate([np.ones(k), np.zeros(n - k)])
            for k, n in zip(sums, counts, strict=True)
        ])
        days = np.concatenate([np.full(n, d) for d, n in enumerate(counts)])
        lo, hi = breach_icc_confidence_set(breached, days, rng, n_sims=120)
        assert hi - lo > 0.4


class TestAuditFindings:
    """Regression tests for the adversarial audit, one per finding. Each of
    these was a reproduced defect; the test is the PoC turned assertion."""

    def test_f1_a_stricter_detect_target_changes_the_power_curve(self):
        """F-1: --detect was silently quantised to the 8%/10% columns, so a
        6% target — harder than 8% — returned the same window. Power must be
        simulated at the requested rate itself."""
        pit, breached, days = latent_world(30, 200, 0.15, seed=41)
        icc = estimate_breach_icc(pit, breached, days, np.random.default_rng(41), n_boot=200)
        grid = (21, 45, 90, 180)
        strict = recommend_window(icc, detect_rate=0.07, n_trials=80, n_boot=200,
                                  day_grid=grid)
        loose = recommend_window(icc, detect_rate=0.12, n_trials=80, n_boot=200,
                                 day_grid=grid)
        # A target closer to the nominal 5% is harder: at every searched
        # window its power is no higher, and the recommended window no shorter.
        for (_, p_strict), (_, p_loose) in zip(
            strict.searched, loose.searched[: len(strict.searched)], strict=False
        ):
            assert p_strict <= p_loose + 0.10
        if strict.days_required is not None and loose.days_required is not None:
            assert strict.days_required >= loose.days_required
        assert strict.searched != loose.searched

    def test_f1_a_detect_rate_at_or_below_nominal_is_refused(self):
        pit, breached, days = latent_world(14, 100, 0.15, seed=42)
        icc = estimate_breach_icc(pit, breached, days, np.random.default_rng(42), n_boot=150)
        with pytest.raises(ValueError, match="above the nominal"):
            recommend_window(icc, detect_rate=0.05, n_trials=10, n_boot=50)
        with pytest.raises(ValueError, match="above the nominal"):
            recommend_window(icc, detect_rate=0.03, n_trials=10, n_boot=50)

    def test_f4_the_interval_covers_on_lumpy_day_sizes(self):
        """F-4: simulating equal days at the mean size gave 80% coverage on
        20/400 alternating designs against a nominal 95%, failing low. The
        confidence set now simulates the pilot's actual day sizes."""
        def lumpy_world(n_days, sizes, rho, seed):
            rng = np.random.default_rng(seed)
            pit, days = [], []
            for d in range(n_days):
                n = sizes[d % len(sizes)]
                f = rng.standard_normal()
                e = rng.standard_normal(n)
                z = np.sqrt(rho) * f + np.sqrt(1 - rho) * e
                pit.append(stats.norm.cdf(z))
                days.append(np.full(n, d))
            return np.concatenate(pit), np.concatenate(days)

        covered = 0
        trials = 40
        for s in range(trials):
            pit, days = lumpy_world(14, [20, 400], 0.20, seed=1_000 + s)
            est = estimate_latent_correlation(pit, days, np.random.default_rng(s),
                                              n_boot=1_000)
            if est.ci_low <= 0.20 <= est.ci_high:
                covered += 1
        assert covered / trials >= 0.85

    def test_f3_the_t_map_floor_is_deliberate_and_matches_simulation(self):
        """F-3: at rho=0 the t map returns ~0.077, not 0. That is the shared
        chi-square talking — a fat-tailed day inflates everyone at once — and
        simulating the actual t world at rho=0 realises the same number, so
        the floor is the model's physics. This test states it on purpose;
        anyone flattening the floor to zero must argue with the simulation."""
        floor = breach_icc_from_latent(0.0, copula="t", n_mc=200_000, seed=5)
        assert 0.05 < floor < 0.11
        from risk_engine.shadow.clustering import _icc_from_counts

        rng = np.random.default_rng(6)
        realised = []
        for _ in range(4):
            # rho=0: no common factor at all -- pure idiosyncratic normals
            # under a shared per-day mixing draw.
            e = rng.standard_normal((200, 200))
            w = np.sqrt(4.0 / rng.chisquare(4.0, size=(200, 1)))
            z = e * w
            br = (z < stats.t.ppf(0.05, 4.0)).astype(float)
            sums, counts = br.sum(axis=1), np.full(200, 200)
            realised.append(_icc_from_counts(sums, counts)[0])
        assert floor == pytest.approx(float(np.mean(realised)), abs=0.03)

    def test_f3_the_gaussian_estimator_composes_unbiasedly_with_the_t_map(self):
        """F-3: the latent rho is measured by Gaussian-scores ANOVA and fed
        to a t-parameterised map — two different parameterisations, so the
        composition could have been biased either way. Measured against true
        shared-mixing t data (16 replications per rho) the mean bias is
        within +/-0.007, i.e. approximately unbiased; individual pilots
        scatter up to +/-0.05 at high rho, which is the interval's job, not
        the point's. This pins the mean: a refactor that introduces a
        systematic under-read would size windows short — the §10-forbidden
        direction — and must fail here.

        An earlier version of this test asserted the bias was systematically
        POSITIVE off four datasets. Sixteen showed that was seed noise. The
        assertion is now on the mean over eight, with a bound four standard
        errors wide, so it tests the estimator rather than the seeds.
        """
        from risk_engine.shadow.clustering import _icc_from_counts

        df = 4.0
        for rho, bound in ((0.10, 0.02), (0.30, 0.04)):
            biases = []
            for s in range(8):
                rng = np.random.default_rng(2_000 + s * 13 + int(rho * 100))
                f = rng.standard_normal((200, 1))
                e = rng.standard_normal((200, 200))
                g = np.sqrt(rho) * f + np.sqrt(1 - rho) * e
                w = np.sqrt(df / rng.chisquare(df, size=(200, 1)))
                z = g * w
                pit = stats.t.cdf(z, df).ravel()
                br = (z < stats.t.ppf(0.05, df)).astype(float)
                days = np.repeat(np.arange(200), 200)
                icc = estimate_breach_icc(pit, br.ravel(), days,
                                          np.random.default_rng(s), n_boot=100)
                empirical = _icc_from_counts(br.sum(axis=1), np.full(200, 200))[0]
                biases.append(icc.point - empirical)
            mean_bias = float(np.mean(biases))
            assert abs(mean_bias) < bound, (
                f"composition bias at rho={rho} is {mean_bias:+.4f}; systematic "
                "under-read sizes windows short, systematic over-read is dishonest"
            )

    def test_f2_degenerate_pits_are_counted_and_surfaced(self):
        """F-2: the docstring promised clipped PITs were 'counted and
        surfaced' while nothing counted them. A misfitted predictive
        distribution shows up as a pile of exact-0/1 PITs, and silence here
        would let that read as clustering."""
        rng = np.random.default_rng(7)
        pit = rng.random(4_000)
        pit[:600] = 0.0
        pit[600:800] = 1.0
        days = np.repeat(np.arange(20), 200)
        est = estimate_latent_correlation(pit, days, np.random.default_rng(7), n_boot=200)
        assert est.n_clipped == 800
        assert est.clipped_fraction == pytest.approx(0.2)
        assert "degenerate" in est.summary()

    def test_f2_clean_pits_report_zero_clipped_and_stay_quiet(self):
        pit, _, days = latent_world(10, 100, 0.1, seed=8)
        est = estimate_latent_correlation(pit, days, np.random.default_rng(8), n_boot=100)
        assert est.n_clipped == 0
        assert "degenerate" not in est.summary()


class TestWindowSizing:
    def test_it_sizes_off_the_upper_bound_by_default(self):
        pit, breached, days = latent_world(14, 200, 0.30, seed=31)
        icc = estimate_breach_icc(pit, breached, days, np.random.default_rng(31), n_boot=300)
        rec = recommend_window(icc, n_trials=60, n_boot=200, day_grid=(21, 60))
        assert rec.icc_used == icc.ci_high
        assert rec.icc_used >= icc.point

    def test_more_clustering_demands_a_longer_window(self):
        grid = (21, 45, 90, 180)
        quiet_pit, quiet_b, quiet_d = latent_world(60, 200, 0.05, seed=32)
        rough_pit, rough_b, rough_d = latent_world(60, 200, 0.60, seed=33)
        quiet = estimate_breach_icc(quiet_pit, quiet_b, quiet_d,
                                    np.random.default_rng(32), n_boot=150)
        rough = estimate_breach_icc(rough_pit, rough_b, rough_d,
                                    np.random.default_rng(33), n_boot=150)
        assert rough.point > quiet.point

        rec_quiet = recommend_window(quiet, n_trials=80, n_boot=200, day_grid=grid)
        rec_rough = recommend_window(rough, n_trials=80, n_boot=200, day_grid=grid)
        assert rec_quiet.days_required is not None
        assert (
            rec_rough.days_required is None
            or rec_rough.days_required >= rec_quiet.days_required
        )

    def test_an_unreachable_target_says_so_instead_of_returning_the_largest(self):
        """Returning 30 when 30 does not work would read as "wait a month and
        it is fine". It is not."""
        pit, breached, days = latent_world(60, 200, 0.85, seed=34)
        icc = estimate_breach_icc(pit, breached, days, np.random.default_rng(34), n_boot=150)
        rec = recommend_window(icc, n_trials=60, n_boot=200, day_grid=(21, 30))
        assert rec.days_required is None
        assert "not the right gate" in rec.summary()
