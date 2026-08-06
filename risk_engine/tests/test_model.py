"""§2 — the estimation layer."""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from risk_engine.model.copula import (
    assert_lower_tail_not_understated,
    diagnose_tail_asymmetry,
    fit_copula_df,
)
from risk_engine.model.correlation import MIN_HISTORY_HOURS, build_global_matrix
from risk_engine.model.drift import DriftConvention, log_drift_per_step
from risk_engine.model.ewma import effective_sample_size, ewma_moments, ewma_weights
from risk_engine.model.funding import FundingBounds, fit_ar1, simulate_funding
from risk_engine.model.marginals import DF_MAX, DF_MIN, fit_df, fit_marginal
from risk_engine.model.psd import is_positive_definite, project_to_correlation
from risk_engine.model.shrinkage import ledoit_wolf_constant_correlation
from risk_engine.observability.metrics import Metrics


class TestEwma:
    def test_weights_halve_over_the_half_life(self):
        w = ewma_weights(1000, half_life=100)
        assert w[-1] / w[-101] == pytest.approx(2.0)
        assert w.sum() == pytest.approx(1.0)

    def test_effective_sample_size_is_below_the_window(self):
        w = ewma_weights(2000, half_life=480)
        n_eff = effective_sample_size(w)
        assert 0 < n_eff < 2000
        # Equal weights would give exactly n; exponential weighting must not.
        # The infinite-window limit is (1+lambda)/(1-lambda) = 1385; a 2000-hour
        # window is short enough relative to a 480-hour half-life that
        # truncation still costs ~10% of that.
        assert n_eff == pytest.approx(1239, rel=0.01)

    def test_recovers_a_known_correlation(self):
        rng = np.random.default_rng(0)
        n = 20_000
        z = rng.standard_normal((n, 2))
        x = np.column_stack([z[:, 0], 0.7 * z[:, 0] + np.sqrt(1 - 0.49) * z[:, 1]]) * 0.01
        m = ewma_moments(x, ("A", "B"), half_life=100_000)
        assert m.corr[0, 1] == pytest.approx(0.7, abs=0.02)
        assert m.vol[0] == pytest.approx(0.01, rel=0.05)

    def test_rejects_unaligned_history(self):
        with pytest.raises(ValueError, match="NaN"):
            ewma_moments(np.array([[0.1, np.nan], [0.2, 0.3]]), ("A", "B"))


class TestShrinkage:
    def test_pulls_correlations_toward_the_average(self):
        rng = np.random.default_rng(1)
        # Few observations relative to dimension: exactly where shrinkage earns
        # its place, and where the unshrunk estimate is worst.
        x = rng.standard_normal((80, 12)) * 0.01
        w = ewma_weights(80, half_life=480)
        res = ledoit_wolf_constant_correlation(x, w, effective_sample_size(w))
        assert 0.0 < res.intensity <= 1.0

        sd = np.sqrt(np.diag(res.cov))
        shrunk_corr = res.cov / np.outer(sd, sd)
        sample = np.einsum("t,ti,tj->ij", w, x, x)
        s_sd = np.sqrt(np.diag(sample))
        raw_corr = sample / np.outer(s_sd, s_sd)
        off = ~np.eye(12, dtype=bool)
        assert np.abs(shrunk_corr[off]).mean() < np.abs(raw_corr[off]).mean()

    def test_variances_are_untouched(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((300, 5)) * 0.02
        w = ewma_weights(300, half_life=480)
        res = ledoit_wolf_constant_correlation(x, w, effective_sample_size(w))
        sample = np.einsum("t,ti,tj->ij", w, x, x)
        assert np.allclose(np.diag(res.cov), np.diag(sample))


class TestPsdProjection:
    def test_fixes_an_indefinite_matrix(self):
        bad = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
        assert not is_positive_definite(bad)
        metrics = Metrics()
        res = project_to_correlation(bad, metrics=metrics)
        assert res.corrected
        assert is_positive_definite(res.corr)
        assert np.allclose(np.diag(res.corr), 1.0)
        assert res.frobenius_correction > 0
        assert metrics.counters["psd_projection_corrections"] == 1
        assert metrics.psd_corrections[0]["min_eigenvalue_before"] < 0

    def test_leaves_a_good_matrix_alone(self):
        good = np.array([[1.0, 0.3], [0.3, 1.0]])
        metrics = Metrics()
        res = project_to_correlation(good, metrics=metrics)
        assert not res.corrected
        assert np.allclose(res.corr, good)
        assert not metrics.psd_corrections

    def test_result_is_strictly_pd_so_slices_factorise(self):
        """§2.1 leans on 'a slice of a PD matrix is PD'; PSD is not enough."""
        rng = np.random.default_rng(3)
        base = rng.standard_normal((4, 12))
        singular = np.corrcoef(base.T)  # rank 4 in 12 dimensions
        res = project_to_correlation(singular)
        assert res.min_eigenvalue_after > 0
        for cols in ([0, 5, 11], [1, 2], list(range(12))):
            np.linalg.cholesky(res.corr[np.ix_(cols, cols)])


class TestMarginals:
    def test_recovers_a_known_df(self):
        from scipy import stats

        x = stats.t(df=5.0).rvs(20_000, random_state=np.random.default_rng(4)) * 0.01
        _df, raw, clamped = fit_df(x)
        assert not clamped
        assert raw == pytest.approx(5.0, rel=0.2)

    def test_clamps_and_logs_a_gaussian_sample(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(20_000) * 0.01  # df -> infinity
        metrics = Metrics()
        spec = fit_marginal("GAUSS", x, step_vol=0.01, metrics=metrics)
        assert spec.df == DF_MAX
        assert spec.clamped
        assert metrics.counters["df_clamps"] == 1
        assert metrics.df_clamps[0]["df_raw"] > DF_MAX

    def test_clamp_bounds_are_the_ones_the_spec_names(self):
        assert (DF_MIN, DF_MAX) == (2.1, 30.0)

    def test_refuses_a_sample_too_short_to_estimate_a_tail(self):
        with pytest.raises(ValueError, match=">= 100"):
            fit_df(np.zeros(50))


class TestFunding:
    def test_ar1_recovers_its_parameters(self):
        rng = np.random.default_rng(6)
        mu, phi, sigma = 1e-5, 0.8, 2e-5
        r = np.empty(2000)
        r[0] = mu
        for i in range(1, r.size):
            r[i] = mu + phi * (r[i - 1] - mu) + sigma * rng.standard_normal()
        got = fit_ar1("X", r, FundingBounds.documented_default())
        assert got.phi == pytest.approx(phi, abs=0.05)
        assert got.sigma == pytest.approx(sigma, rel=0.1)

    def test_clamp_binds_in_simulation(self):
        """§1.5: without the clamp, a long path wanders to absurd rates."""
        bounds = FundingBounds(cap_per_hour=1e-4, source="test")
        model = fit_ar1("X", np.full(200, 5e-5) + 1e-6, bounds)
        model = type(model)(model.asset, 1e-3, 0.99, 1e-3, 5e-5, model.n_observations)
        out = simulate_funding([model], bounds, 500, 168, np.random.default_rng(7))
        assert np.abs(out).max() <= bounds.cap_per_hour + 1e-15
        assert np.abs(out).max() == pytest.approx(bounds.cap_per_hour, rel=1e-9)

    def test_a_stale_bound_fails_loudly_instead_of_truncating(self):
        bounds = FundingBounds(cap_per_hour=1e-5, source="stale")
        with pytest.raises(ValueError, match="exceeds the configured cap"):
            fit_ar1("X", np.full(200, 5e-3), bounds)

    def test_bounds_must_carry_provenance(self):
        with pytest.raises(ValueError, match="provenance"):
            FundingBounds(cap_per_hour=0.04, source="  ")


class TestDrift:
    def test_zero_log_return_is_the_default_convention(self):
        vol = np.array([0.01, 0.02])
        assert np.allclose(log_drift_per_step(DriftConvention.ZERO_LOG_RETURN, vol), 0.0)

    def test_convexity_convention_is_negative_and_small(self):
        vol = np.array([0.01])
        d = log_drift_per_step(DriftConvention.MEDIAN_PRESERVING_CONVEXITY, vol)
        assert d[0] == pytest.approx(-0.5e-4)


class TestGlobalMatrix:
    @staticmethod
    def _returns(rng, n_assets=6, n=MIN_HISTORY_HOURS * 2, rho=0.85):
        factor = rng.standard_normal(n)
        cols = {}
        names = ["BTC", "ETH", "SOL", "AVAX", "LINK", "OP"][:n_assets]
        for i, name in enumerate(names):
            idio = rng.standard_normal(n)
            cols[name] = (rho * factor + np.sqrt(1 - rho**2) * idio) * (0.01 + 0.002 * i)
        return cols

    def test_builds_and_slices(self):
        rng = np.random.default_rng(8)
        m = build_global_matrix(self._returns(rng), now=datetime.now(timezone.utc))
        assert {"BTC", "ETH"} <= set(m.assets)
        assert is_positive_definite(m.corr)
        sub = m.submatrix(("SOL", "BTC"))
        assert sub.shape == (2, 2)
        assert sub[0, 1] == pytest.approx(m.corr[m.assets.index("SOL"), 0])
        np.linalg.cholesky(sub)

    def test_cholesky_is_cached_per_universe(self):
        rng = np.random.default_rng(9)
        m = build_global_matrix(self._returns(rng))
        first = m.cholesky(("BTC", "ETH"))
        assert m.cholesky(("BTC", "ETH")) is first

    def test_young_asset_is_imputed_not_estimated(self):
        """§2.1's data gate. The imputed correlation must be high, and it must
        not depend on the young asset's own (short, noisy) sample."""
        rng = np.random.default_rng(10)
        series = self._returns(rng)
        # A new listing whose short history happens to look uncorrelated.
        series["NEWCOIN"] = rng.standard_normal(200) * 0.03
        m = build_global_matrix(series)
        assert m.diagnostics.imputed_assets == ("NEWCOIN",)

        i = m.assets.index("NEWCOIN")
        btc = m.assets.index("BTC")
        assert m.corr[i, btc] > 0.5
        assert m.corr[i, btc] == pytest.approx(m.diagnostics.gate_correlation, abs=0.05)

        sample_corr = np.corrcoef(series["NEWCOIN"], series["BTC"][-200:])[0, 1]
        assert abs(sample_corr) < 0.3  # what a naive estimate would have used
        assert m.corr[i, btc] > sample_corr + 0.3

    def test_imputed_row_does_not_depend_on_the_young_sample(self):
        """Two new listings with opposite short-sample correlations must get
        the same imputed row: the gate replaces the estimate, it does not
        blend with it."""
        rng = np.random.default_rng(21)
        base = self._returns(rng)
        anti = -base["BTC"][-200:] + rng.standard_normal(200) * 1e-4
        with_anti = build_global_matrix({**base, "NEWCOIN": anti})
        with_flat = build_global_matrix({**base, "NEWCOIN": rng.standard_normal(200) * 0.02})
        i = with_anti.assets.index("NEWCOIN")
        j = with_flat.assets.index("NEWCOIN")
        assert np.allclose(with_anti.corr[i], with_flat.corr[j])

    def test_gate_level_is_a_high_quantile_of_mature_correlations(self):
        rng = np.random.default_rng(11)
        series = self._returns(rng)
        series["NEWCOIN"] = rng.standard_normal(100) * 0.02
        m = build_global_matrix(series)
        btc = m.assets.index("BTC")
        n_mature = len(m.assets) - 1
        mature_rhos = [m.corr[btc, j] for j in range(n_mature) if j != btc]
        assert m.diagnostics.gate_correlation >= np.median(mature_rhos)

    def test_requires_btc_and_eth(self):
        rng = np.random.default_rng(12)
        series = self._returns(rng)
        del series["ETH"]
        with pytest.raises(ValueError, match="ETH"):
            build_global_matrix(series)


class TestConditionalTailFloor:
    """A11 option A, adopted 2026-08-05: `df* = min(df_ML, df_tail)`.

    The floor is conditional (only >= 2 sigma shortfalls are demands),
    one-sided (lambda_U is never a constraint), targets the demand's one-sided
    95% lower bound rather than its point estimate, and cannot raise the df.
    Each of those properties is a measured A11 finding; each gets its own
    assertion here so none can rot into prose.
    """

    RHO = 0.887          # backed out of the live 2026-08-05 ETH/SOL reading
    N_SIM = 100_000      # enough for these gaps; keeps the suite fast

    @staticmethod
    def _diag(lower: float, model: float, n: int = 108, upper: float = 0.685):
        from risk_engine.model.copula import TailDiagnostic

        return TailDiagnostic(
            pair=("ETH", "SOL"), threshold=0.05,
            empirical_lower=lower, empirical_upper=upper,
            model_at_threshold=model, model_asymptotic=0.593,
            n_lower_exceedances=n,
        )

    def _floor(self, diags, df_ml=6.5):
        from risk_engine.model.copula import tail_floor_df

        corr = np.array([[1.0, self.RHO], [self.RHO, 1.0]])
        return tail_floor_df(diags, ("ETH", "SOL"), corr, df_ml,
                             n_sim=self.N_SIM)

    def test_a_sub_threshold_pair_is_not_a_demand(self):
        """Finding 3's conditionality. A 1.1-sigma shortfall (one the OLD flat
        margin would have fired on) leaves the ML fit untouched: fitting to it
        would install a remedy on data indistinguishable from noise."""
        floor = self._floor([self._diag(lower=0.694, model=0.645)])
        assert not floor.floored
        assert floor.df == 6.5
        assert floor.demands == ()

    def test_every_pair_that_fires_the_gate_is_a_demand(self):
        """The stuck band 0.5.0 shipped with, closed and pinned.

        The gate's margin fires at 1.645*SE; the demand threshold was 2.0
        sigma. A pair between them held the gate lit -- recording mode, no
        gate-days -- while the floor never attempted a remedy, indefinitely.
        The two thresholds are now the same constant, and this asserts the
        implication that matters: understates => demand, so there is no
        reading that can keep the gate open without the floor at least
        trying to close it.
        """
        # 1.8 sigma: gap = 0.0797 over SE 0.0443 -- fires the 1.645*SE margin,
        # sat in the dead band under the old 2.0 demand threshold.
        d = self._diag(lower=0.694, model=0.694 - 1.8 * 0.0443)
        assert d.understates_lower_tail(), "the fixture must sit past the gate"
        floor = self._floor([d])
        assert d in floor.demands, (
            "a pair keeping the gate lit was not a demand; the stuck band is back"
        )

    def test_a_significant_demand_floors_to_the_largest_covering_df(self):
        """The live 2026-08-05 ETH/SOL reading: +2.7 sigma. The floor must
        land on the LARGEST grid df covering `empirical - 1.645*SE` -- not
        deeper (over-fitting the point) and not shallower (missing the bound).
        Asserted structurally rather than as a pinned constant, so a grid or
        estimator change re-derives the value instead of failing on it."""
        from risk_engine.model.copula import (
            COPULA_DF_GRID,
            model_tail_dependence_at_threshold,
        )

        d = self._diag(lower=0.759, model=0.648)
        floor = self._floor([d])
        assert floor.floored and floor.covered
        assert floor.demands == (d,)

        target = 0.759 - 1.645 * d.lower_standard_error
        assert model_tail_dependence_at_threshold(
            floor.df, self.RHO, 0.05, n_sim=self.N_SIM, seed=7
        ) >= target, "the chosen df must cover the one-sided bound"
        above = [float(g) for g in COPULA_DF_GRID
                 if floor.df < g <= floor.df_ml]
        if above:
            nxt = min(above)
            assert model_tail_dependence_at_threshold(
                nxt, self.RHO, 0.05, n_sim=self.N_SIM, seed=7
            ) < target, (
                "a larger grid df also covers the bound; the floor dug deeper "
                "than the data demanded"
            )

    def test_the_floored_bundle_passes_the_gate_it_was_floored_for(self):
        """The coherence property, which is the whole point of coupling the
        floor and the margin: after flooring, the residual shortfall is at
        most 1.645*SE, and the gate's margin is at least that -- so recording
        mode ends exactly when the floor covers the demands."""
        from risk_engine.model.copula import (
            model_tail_dependence_at_threshold,
        )

        d = self._diag(lower=0.759, model=0.648)
        floor = self._floor([d])
        model_at_floor = model_tail_dependence_at_threshold(
            floor.df, self.RHO, 0.05, n_sim=self.N_SIM, seed=7
        )
        refit = self._diag(lower=0.759, model=model_at_floor)
        assert not refit.understates_lower_tail(), (
            f"floored to df={floor.df} (model@q={model_at_floor:.3f}) yet the "
            f"gate still fires; the floor and the margin have decohered"
        )

    def test_the_floor_never_raises_the_df(self):
        """Thin-tailed data leave the ML fit alone: a demand whose bound the
        current fit already covers is not a demand at all, and an ML fit
        already below every covering df must stand."""
        floor = self._floor([self._diag(lower=0.660, model=0.648)])
        assert floor.df == 6.5

        deep = self._floor([self._diag(lower=0.759, model=0.648)], df_ml=2.5)
        assert deep.df == 2.5, "df_ml at the grid floor cannot be raised"

    def test_an_unreachable_demand_floors_to_the_grid_edge_and_says_so(self):
        """When even the heaviest grid df cannot reach the bound, the floor
        still applies (heavier is nearer the data) and reports covered=False,
        so the §2.3 gate stays lit and recording mode continues. Chasing
        coverage off the grid is foreclosed -- the family's measured reach
        says the point needs a near-Cauchy df."""
        from risk_engine.model.copula import COPULA_DF_GRID

        floor = self._floor([self._diag(lower=0.95, model=0.648)])
        assert floor.floored and not floor.covered
        assert floor.df == float(COPULA_DF_GRID[0])

    def test_lambda_U_is_never_a_constraint(self):
        """Finding 4's one-sidedness. A pair whose UPPER tail towers over the
        model must contribute nothing to the demand set: lambda_U is not
        monotone in the family's parameters and the observed BTC/ETH upper is
        unreachable at any admissible skew, so a two-sided demand would be
        unsatisfiable by construction."""
        floor = self._floor([self._diag(lower=0.650, model=0.648, upper=0.95)])
        assert not floor.floored
        assert floor.demands == ()


class TestConditionalRhoLift:
    """A11 option R+H, adopted 2026-08-05 evening: lift rho, then floor df.

    The first live firing measured the df lever short of the asymmetric
    pair's bound at the grid wall; the rho-lever covers it at the ML df with
    room to spare. The chain keeps every property the floor promised —
    conditional, one-sided, bound-targeting, recomputed per build — and adds
    two of its own: the lifted matrix is PD-projected, and coverage is
    re-verified on the PROJECTED entries rather than assumed from the search.
    Each property gets its own assertion so none can rot into prose.
    """

    RHO = 0.863          # backed out of the live 2026-08-05 banner asymptotics
    DF_ML = 5.0          # the live ML fit with HYPE in the joint window

    @staticmethod
    def _diag(lower: float, model: float, n: int = 108, upper: float = 0.685,
              pair: tuple[str, str] = ("ETH", "SOL")):
        from risk_engine.model.copula import TailDiagnostic

        return TailDiagnostic(
            pair=pair, threshold=0.05,
            empirical_lower=lower, empirical_upper=upper,
            model_at_threshold=model, model_asymptotic=0.593,
            n_lower_exceedances=n,
        )

    def _remedy(self, diags, df_ml=None, corr=None, assets=("ETH", "SOL")):
        from risk_engine.model.copula import tail_remedy_dependence

        if corr is None:
            corr = np.array([[1.0, self.RHO], [self.RHO, 1.0]])
        return tail_remedy_dependence(
            diags, tuple(assets), corr,
            self.DF_ML if df_ml is None else df_ml,
        )

    def test_no_demand_leaves_the_matrix_and_the_df_alone(self):
        """Conditionality, inherited from the floor: a 1.1-sigma shortfall is
        noise to re-test, and lifting the matrix for it would install a body
        distortion on data indistinguishable from luck."""
        remedy = self._remedy([self._diag(lower=0.694, model=0.645)])
        assert not remedy.lifted and not remedy.floored
        assert remedy.df == self.DF_ML
        assert remedy.demands == ()
        assert remedy.covered and remedy.uncovered_pairs == ()
        np.testing.assert_array_equal(
            remedy.corr, np.array([[1.0, self.RHO], [self.RHO, 1.0]])
        )

    def test_the_live_demand_lifts_to_the_smallest_covering_rho(self):
        """The 2026-08-05 evening reading: ETH/SOL lower 0.759 against
        model@q 0.652 at the ML df. The lift must cover the one-sided 95%
        bound at the ML df — where the df floor measurably could not — and
        must not overshoot it by more than the search resolution. Asserted
        structurally, not as a pinned 0.907, so an estimator change
        re-derives the value instead of failing on it."""
        from risk_engine.model.copula import model_tail_dependence_at_threshold

        d = self._diag(lower=0.759, model=0.652)
        remedy = self._remedy([d])
        assert remedy.lifted and remedy.covered and not remedy.floored
        assert remedy.df == self.DF_ML, "coverage at the ML df must keep it"

        (lift,) = remedy.lifts
        target = 0.759 - 1.645 * d.lower_standard_error
        assert lift.target == pytest.approx(target, abs=1e-9)
        rho_served = float(remedy.corr[0, 1])
        assert rho_served > self.RHO
        # The GATE's estimator for this pair (seed i*1000+j = 1, its default
        # n_sim): the lift promises coverage in the gate's own terms, so the
        # assertion must compute what the gate will compute.
        assert model_tail_dependence_at_threshold(
            self.DF_ML, rho_served, 0.05, n_sim=200_000, seed=1
        ) >= target, "the served rho must cover the bound, as the gate scores it"
        assert model_tail_dependence_at_threshold(
            self.DF_ML, rho_served - 0.01, 0.05, n_sim=200_000, seed=1
        ) < target, (
            "a materially smaller rho also covers the bound; the lift dug "
            "deeper into the body than the data demanded"
        )

    def test_the_lift_never_lowers_an_entry_and_touches_only_demand_pairs(self):
        """One-sidedness in the rho dimension: thin readings leave the EWMA
        estimate alone, and pairs that are not demands keep their measured
        correlation to the last decimal."""
        corr = np.array([
            [1.0, 0.863, 0.884],
            [0.863, 1.0, 0.876],
            [0.884, 0.876, 1.0],
        ])
        d = self._diag(lower=0.759, model=0.652)  # ETH/SOL only
        remedy = self._remedy([d], corr=corr, assets=("ETH", "SOL", "BTC"))
        assert remedy.lifted
        assert not remedy.projection_moved, (
            "this lift keeps the matrix PD outright; the projection must be idle"
        )
        assert float(remedy.corr[0, 1]) > 0.863
        assert float(remedy.corr[0, 2]) == pytest.approx(0.884, abs=1e-12)
        assert float(remedy.corr[1, 2]) == pytest.approx(0.876, abs=1e-12)
        assert (np.asarray(remedy.corr) >= corr - 1e-12).all(), (
            "no entry may move DOWN: the lift raises modelled co-crash or "
            "leaves it alone"
        )

    def test_an_unreachable_target_lifts_to_the_cap_and_composes_the_floor(self):
        """A pathological reading (lower 0.995 on a tight SE) exceeds what
        any rho below the cap can produce. The chain must then do everything
        it lawfully can — lift to the cap, floor the df — and still say
        covered=False so the §2.3 gate stays lit. And the served matrix must
        remain strictly inside the PD cone: a comonotone pair would kill
        every Cholesky downstream."""
        from risk_engine.model.copula import COPULA_DF_GRID, TAIL_RHO_CAP

        d = self._diag(lower=0.995, model=0.652, n=10_000)
        remedy = self._remedy([d])
        (lift,) = remedy.lifts
        assert not lift.covered
        assert lift.rho_to == pytest.approx(TAIL_RHO_CAP, abs=1e-9)
        assert remedy.floored and remedy.df == float(COPULA_DF_GRID[0])
        assert not remedy.covered, (
            "an unreachable bound must keep the gate lit, not pass silently"
        )
        assert ("ETH", "SOL") in remedy.uncovered_pairs
        assert float(np.max(np.abs(
            np.asarray(remedy.corr)[~np.eye(2, dtype=bool)]
        ))) < 1.0
        np.linalg.cholesky(remedy.corr)  # must not raise

    def test_coverage_is_reverified_on_the_projected_matrix(self, monkeypatch):
        """The PD projection may pull a lifted entry back down. Coverage must
        be judged on the entries that will actually be SERVED — a remedy that
        trusts its own search while the projection undid it would report a
        covered gate over an uncovered matrix, in the §10 direction."""
        import risk_engine.model.copula as copula_mod
        from risk_engine.model.psd import ProjectionResult

        def projection_pulls_the_lift_back(matrix, **_kwargs):
            return ProjectionResult(
                corr=np.array([[1.0, self.RHO], [self.RHO, 1.0]]),
                corrected=True, min_eigenvalue_before=0.0,
                min_eigenvalue_after=0.0, frobenius_correction=0.0,
                shift_applied=False,
            )

        monkeypatch.setattr(
            copula_mod, "project_to_correlation", projection_pulls_the_lift_back
        )
        d = self._diag(lower=0.759, model=0.652)
        remedy = self._remedy([d])
        assert remedy.projection_moved
        # At the un-lifted rho the bound is unreachable on the whole df grid
        # (the measured df-lever wall), so the honest outcome is a composed
        # floor that still cannot cover — never a quiet covered=True.
        assert remedy.floored
        assert not remedy.covered

    def test_a_measured_rho_past_the_cap_is_never_cut_down_to_it(self):
        """A pair can be MEASURED above TAIL_RHO_CAP (crash-regime BTC/ETH is
        the live candidate). The lift's only permitted direction is up:
        writing the cap over a higher measurement would serve an entry below
        the EWMA estimate — understating measured co-crash dependence, the
        §10 direction — to chase a bound the pair cannot meet anyway."""
        rho_measured = 0.985  # above the 0.98 cap
        corr = np.array([[1.0, rho_measured], [rho_measured, 1.0]])
        d = self._diag(lower=0.995, model=0.87, n=10_000)
        remedy = self._remedy([d], corr=corr)
        (lift,) = remedy.lifts
        assert lift.rho_to == pytest.approx(rho_measured, abs=1e-12), (
            "the entry was cut toward the cap; the lift may never lower"
        )
        assert float(remedy.corr[0, 1]) == pytest.approx(rho_measured, abs=1e-9)
        assert not remedy.covered, "the bound is unreachable; the gate stays lit"
        assert ("ETH", "SOL") in remedy.uncovered_pairs

    def test_an_unmeasurable_pair_is_uncovered_never_vacuously_covered(self):
        """A NaN lower tail (no joint exceedances) and a zero-SE reading
        (every exceedance joint, sigmas not finite) both fire the §2.3 gate
        while carrying no bound the remedy could target. They are excluded
        from the demand set — correctly, there is nothing to lift toward —
        but the remedy must then say covered=False: covered=True over a
        firing gate is the exact lie `covered` exists not to tell."""
        nan_pair = self._diag(lower=float("nan"), model=0.4)
        remedy = self._remedy([nan_pair])
        assert remedy.demands == ()
        assert not remedy.covered
        assert ("ETH", "SOL") in remedy.uncovered_pairs

        exact = self._diag(lower=1.0, model=0.652)  # SE = 0, sigmas NaN
        remedy = self._remedy([exact])
        assert remedy.demands == ()
        assert not remedy.covered
        assert ("ETH", "SOL") in remedy.uncovered_pairs

    def test_a_bystander_pair_degraded_by_the_projection_is_in_the_verdict(self):
        """The projection can pay for a lift by pulling down entries the
        search never touched. A bystander pair pushed past its own margin
        must appear in the verdict — the alternative is a bundle logged
        covered while the §2.3 gate fires on a pair the remedy never looked
        at, which is the 0.5.0 stuck band with extra steps."""
        corr = np.array([
            [1.0, 0.60, 0.95],
            [0.60, 1.0, 0.35],
            [0.95, 0.35, 1.0],
        ])  # PD, but tight: lifting (A,B) forces redistribution
        demand = self._diag(lower=0.62, model=0.3775, pair=("A", "B"))
        bystander = self._diag(lower=0.829, model=0.773, pair=("A", "C"))
        quiet = self._diag(lower=0.30, model=0.28, pair=("B", "C"))
        assert not bystander.understates_lower_tail(), (
            "the bystander must PASS the gate before the remedy runs"
        )
        remedy = self._remedy(
            [demand, bystander, quiet], corr=corr, assets=("A", "B", "C")
        )
        assert remedy.projection_moved, "this scenario exists to move it"
        assert float(remedy.corr[0, 2]) < 0.95, (
            "the projection should have pulled the bystander entry down"
        )
        assert not remedy.covered
        assert ("A", "C") in remedy.uncovered_pairs, (
            "the degraded bystander must be named, not just the demands"
        )

    def test_two_demands_are_both_lifted_and_both_covered(self):
        """The live banner reported up to three demand pairs; nothing about
        the chain may quietly assume one. Two demands on a loose matrix must
        both lift, both cover, and leave the third pair byte-identical."""
        corr = np.array([
            [1.0, 0.60, 0.55],
            [0.60, 1.0, 0.58],
            [0.55, 0.58, 1.0],
        ])
        d1 = self._diag(lower=0.55, model=0.45, pair=("A", "B"))
        d2 = self._diag(lower=0.55, model=0.45, pair=("B", "C"))
        remedy = self._remedy([d1, d2], corr=corr, assets=("A", "B", "C"))
        assert len(remedy.demands) == 2
        assert all(lift.lifted for lift in remedy.lifts), (
            "both demands must be lifted, not just the first"
        )
        assert remedy.covered and remedy.df == self.DF_ML
        assert float(remedy.corr[0, 1]) > 0.60
        assert float(remedy.corr[1, 2]) > 0.58
        assert float(remedy.corr[0, 2]) == pytest.approx(0.55, abs=1e-9), (
            "the pair with no demand must keep its measured value"
        )

    def test_the_embed_branch_serves_a_pd_matrix_with_imputed_rows(self):
        """When imputed young-asset rows exist, the lifted submatrix is
        embedded into the full matrix and the whole thing re-projected —
        served matrices are Cholesky-factorised per slice, so full-matrix
        PD is a hard invariant, and no test exercised this branch."""
        import risk_engine.service.state as state

        rng = np.random.default_rng(38)
        n = 40_000
        sub = np.array([[1.0, 0.6], [0.6, 1.0]])
        z = rng.standard_normal((n, 2)) @ np.linalg.cholesky(sub).T
        x = z / np.sqrt(rng.chisquare(6.0, size=(n, 1)) / 6.0)
        crash = rng.random(n) < 0.05
        x[crash, :] = -np.abs(x[crash, :]) - 3.0
        returns = {"A": x[:, 0], "B": x[:, 1], "C": rng.standard_normal(n)}
        full = np.array([
            [1.0, 0.6, 0.40],
            [0.6, 1.0, 0.35],
            [0.40, 0.35, 1.0],
        ])
        matrix = SimpleNamespace(
            assets=["A", "B", "C"], corr=full,
            diagnostics=SimpleNamespace(imputed_assets=("C",)),
        )

        remedied, _df = state._tail_remedied_dependence(returns, matrix, 6.0)
        served = np.asarray(remedied.corr)
        assert served.shape == (3, 3)
        assert float(served[0, 1]) > 0.6, "the estimated pair must be lifted"
        np.linalg.cholesky(served)  # the §2.1 invariant: PD or bust
        # The imputed row was §2.1's construction, not a measurement; the
        # remedy has no business rewriting it beyond what PD requires.
        assert float(served[0, 2]) == pytest.approx(0.40, abs=0.02)

    def test_the_builders_hand_the_gate_the_matrix_and_df_they_bundle(self, monkeypatch):
        """The wiring finding: nothing tested that the REMEDIED matrix and
        df are what reaches both the ModelBundle and the §2.3 gate. A slip
        that bundles the remedied pair while the gate judges the raw one
        (or vice versa) would serve an unguarded model with the gate dark."""
        import risk_engine.service.state as state

        marker = {}

        real_remedy = state._tail_remedied_dependence

        def spy_remedy(returns, matrix, df_ml, timestamps=None):
            out_matrix, out_df = real_remedy(returns, matrix, df_ml, timestamps)
            marker["matrix"], marker["df"] = out_matrix, out_df
            return out_matrix, out_df

        seen = {}
        real_checked = state._checked_tail_diagnostics

        def spy_checked(returns, matrix, copula_df, fatal=True, timestamps=None):
            seen["matrix"], seen["df"] = matrix, copula_df
            return real_checked(returns, matrix, copula_df,
                                fatal=fatal, timestamps=timestamps)

        monkeypatch.setattr(state, "_tail_remedied_dependence", spy_remedy)
        monkeypatch.setattr(state, "_checked_tail_diagnostics", spy_checked)
        bundle, _specs, _spot = state._build_fixture_bundle()

        assert bundle.matrix is marker["matrix"], (
            "the bundle must carry the remedy's matrix"
        )
        assert bundle.copula_df == marker["df"]
        assert seen["matrix"] is marker["matrix"], (
            "the gate must judge the same matrix the bundle serves"
        )
        assert seen["df"] == marker["df"]

    def test_gate_coherence_end_to_end_through_the_state_wrapper(self, caplog):
        """The property the whole chain exists for: a crash-together market
        that used to hold the §2.3 gate lit permanently is remedied into a
        bundle the gate PASSES — recording mode ends, gate-days accrue — with
        the lift covering at the ML df. Run through the real state wrapper so
        the wiring (submatrix, embed, final diagnostics) is what is tested,
        and the operator-visible trace (lift WARNING, counter) with it."""
        import logging

        import risk_engine.service.state as state
        from risk_engine.observability.metrics import METRICS

        rng = np.random.default_rng(38)
        n = 40_000
        corr = np.array([[1.0, 0.6], [0.6, 1.0]])
        z = rng.standard_normal((n, 2)) @ np.linalg.cholesky(corr).T
        x = z / np.sqrt(rng.chisquare(6.0, size=(n, 1)) / 6.0)
        crash = rng.random(n) < 0.05
        x[crash, :] = -np.abs(x[crash, :]) - 3.0
        returns = {"A": x[:, 0], "B": x[:, 1]}
        matrix = SimpleNamespace(assets=["A", "B"], corr=corr)

        # Unremedied, this market refuses (the pre-0.7.0 permanent state).
        with pytest.raises(ValueError, match="understates lower-tail"):
            state._checked_tail_diagnostics(returns, matrix, 6.0)

        METRICS.reset()
        with caplog.at_level(logging.WARNING, logger="risk_engine.service.state"):
            remedied, df = state._tail_remedied_dependence(returns, matrix, 6.0)
        assert float(remedied.corr[0, 1]) > 0.6, "the pair must be lifted"
        assert df == 6.0, "the lift covers at the ML df; no floor needed here"
        assert METRICS.counters.get("copula_rho_tail_lifted", 0) == 1
        lifted_warnings = [
            r for r in caplog.records if "copula rho lifted" in r.getMessage()
        ]
        assert lifted_warnings, "a served lift with no WARNING is invisible"
        assert "A/B" in lifted_warnings[-1].getMessage()
        # The remedied bundle passes the very gate that refused it (fatal
        # path, so a regression raises rather than records).
        diags = state._checked_tail_diagnostics(returns, remedied, df)
        assert diags, "the diagnostics must still be produced and recorded"


class TestCopula:
    """§2.3 — dependence fitting and the mandatory tail-asymmetry diagnostic."""

    @staticmethod
    def _t_copula_sample(n, corr, df, rng):
        chol = np.linalg.cholesky(corr)
        z = rng.standard_normal((n, corr.shape[0])) @ chol.T
        w = rng.chisquare(df, size=(n, 1)) / df
        return z / np.sqrt(w)

    def test_recovers_a_known_copula_df(self):
        rng = np.random.default_rng(30)
        corr = np.array([[1.0, 0.6], [0.6, 1.0]])
        x = self._t_copula_sample(40_000, corr, 4.0, rng)
        assert fit_copula_df(x, corr) == pytest.approx(4.0, abs=1.0)

    def test_prefers_a_high_df_for_gaussian_dependence(self):
        rng = np.random.default_rng(31)
        corr = np.array([[1.0, 0.6], [0.6, 1.0]])
        x = rng.standard_normal((40_000, 2)) @ np.linalg.cholesky(corr).T
        assert fit_copula_df(x, corr) >= 20.0

    def test_symmetric_data_produce_no_asymmetry_finding(self):
        rng = np.random.default_rng(32)
        corr = np.array([[1.0, 0.6], [0.6, 1.0]])
        x = self._t_copula_sample(40_000, corr, 4.0, rng)
        diags = diagnose_tail_asymmetry(x, ("A", "B"), corr, 4.0)
        assert abs(diags[0].asymmetry) < 0.05
        assert_lower_tail_not_understated(diags)  # must not raise

    def test_a_crash_together_market_is_reported_as_a_blocking_defect(self):
        """The failure mode §2.3 exists to catch: assets that fall together
        harder than they rise together. A symmetric copula cannot express it,
        so the diagnostic must refuse rather than quietly understate."""
        rng = np.random.default_rng(33)
        corr = np.array([[1.0, 0.6], [0.6, 1.0]])
        x = self._t_copula_sample(40_000, corr, 6.0, rng)
        # Force joint crashes into the lower tail only.
        crash = rng.random(40_000) < 0.05
        x[crash, :] = -np.abs(x[crash, :]) - 3.0

        diags = diagnose_tail_asymmetry(x, ("A", "B"), corr, 6.0)
        assert diags[0].asymmetry > 0.1
        assert diags[0].understates_lower_tail()
        with pytest.raises(ValueError, match="understates lower-tail"):
            assert_lower_tail_not_understated(diags)

    def test_a_symmetric_shortfall_is_not_reported_as_an_asymmetry(self):
        """§2.3 prescribes a skewed-t, and that is the wrong fix for half of
        what fires the gate.

        Observed live on mainnet 2026-08-03: BTC/ETH came back lower=0.694
        against upper=0.731 — the UPPER tail heavier — while still failing,
        because the model sat under both. A skewed-t buys one tail at the
        other's expense, so applying it there fits the lower tail by making
        the upper worse. The refusal has to name that case, or a reader
        following the message reaches for the wrong remedy.
        """
        from risk_engine.model.copula import TailDiagnostic

        live_btc_eth = TailDiagnostic(
            pair=("BTC", "ETH"), threshold=0.05,
            empirical_lower=0.694, empirical_upper=0.731,
            model_at_threshold=0.633, model_asymptotic=0.571,
            n_lower_exceedances=108,
        )
        # Under the 0.5.0 margin (max(0.05, 1.645*SE)) this reading no longer
        # fires at all: its shortfall is 1.4 sigma, and A11 finding 3 measured
        # what a margin without a null does (a remedy installed on zero-signal
        # data ~84% of the time). The gate not firing HERE is the designed
        # change, asserted rather than worked around.
        assert not live_btc_eth.understates_lower_tail(), (
            "a 1.4-sigma shortfall is noise under the null-calibrated margin"
        )
        assert live_btc_eth.asymmetry < 0, "the upper tail is the heavier one"
        assert live_btc_eth.upper_also_understated

        # The same shape with two more weeks of window: identical point
        # estimates at n=432 halve the SE twice over, the shortfall becomes
        # 2.8 sigma, and the gate fires -- now on signal. The NOT-AN-ASYMMETRY
        # note must name this case, or a reader reaches for the wrong remedy.
        matured = TailDiagnostic(
            pair=("BTC", "ETH"), threshold=0.05,
            empirical_lower=0.694, empirical_upper=0.731,
            model_at_threshold=0.633, model_asymptotic=0.571,
            n_lower_exceedances=432,
        )
        assert matured.understates_lower_tail()
        with pytest.raises(ValueError, match="NOT AN ASYMMETRY"):
            assert_lower_tail_not_understated([matured])

        # A genuinely asymmetric pair must NOT collect that note.
        live_eth_sol = TailDiagnostic(
            pair=("ETH", "SOL"), threshold=0.05,
            empirical_lower=0.750, empirical_upper=0.685,
            model_at_threshold=0.641, model_asymptotic=0.571,
            n_lower_exceedances=108,
        )
        assert not live_eth_sol.upper_also_understated
        with pytest.raises(ValueError) as exc:
            assert_lower_tail_not_understated([live_eth_sol])
        assert "NOT AN ASYMMETRY" not in str(exc.value)

    def test_the_refusal_reports_how_strong_the_signal_is(self):
        """The 0.05 margin is ~1.13 standard errors at the live sample size,
        so the gate fires readily on noise — intended (§10 makes a false alarm
        cheaper than a miss), but a refusal that does not separate a 2.6-sigma
        pair from a 1.4-sigma one invites dismissing all of them."""
        from risk_engine.model.copula import TailDiagnostic

        d = TailDiagnostic(
            pair=("ETH", "SOL"), threshold=0.05,
            empirical_lower=0.750, empirical_upper=0.685,
            model_at_threshold=0.641, model_asymptotic=0.571,
            n_lower_exceedances=108,
        )
        assert d.lower_standard_error == pytest.approx(0.0417, abs=1e-3)
        assert d.lower_excess_sigmas == pytest.approx(2.6, abs=0.1)
        assert "sigma" in str(d)

    def test_finite_threshold_dependence_exceeds_the_asymptotic_coefficient(self):
        """Why the diagnostic may not compare against the closed form: at any
        workable threshold the finite estimate sits well above the limit, so
        that comparison would flag a correct model as understating its tail."""
        from risk_engine.model.copula import model_tail_dependence_at_threshold
        from risk_engine.sim.paths import lower_tail_dependence

        at_q = model_tail_dependence_at_threshold(4.0, 0.6, 0.05, n_sim=200_000)
        assert at_q > lower_tail_dependence(4.0, 0.6) + 0.05

    def test_model_implied_tail_dependence_matches_theory(self):
        from risk_engine.sim.paths import lower_tail_dependence

        # Independence in the limit of a Gaussian copula: df -> large, rho 0.
        assert lower_tail_dependence(30.0, 0.0) < 0.02
        # Perfect dependence saturates at 1.
        assert lower_tail_dependence(4.0, 0.999999) == pytest.approx(1.0, abs=0.01)
        # Heavier copula tails mean more joint extremes at the same rho.
        assert lower_tail_dependence(3.0, 0.6) > lower_tail_dependence(20.0, 0.6)


class TestTheDiagnosticRunsOnTheShippedPath:
    """OPEN-QUESTIONS A10 — the check above ran nowhere.

    Every test in `TestCopula` calls `diagnose_tail_asymmetry` directly, so
    all of them passed while **no shipped code path invoked it**. The model
    was documented as guarded by an assertion that only tests ever reached,
    which is the worse failure: an unguarded model that says so is at least
    honest about it.

    These tests therefore assert the *wiring*, not the statistic. They are
    deliberately written against the real bundle builder rather than a stub,
    because a stub is exactly what let the gap exist — the thing being
    checked is whether production calls this at all.
    """

    def test_the_bundle_fits_its_copula_df_rather_than_hardcoding_it(self):
        """OPEN-QUESTIONS A9, the sibling defect to A10. `fit_copula_df` was
        implemented, tested and called from nowhere while both builders passed
        a literal 4.0 — and A9 described the IFM estimator in the present
        tense as though it were in use.

        The assertion is against the hardcoded value specifically, not against
        6.5: pinning the fitted number would make this a change-detector on
        the fixture's random seed. What must hold is that the number came from
        the data."""
        from risk_engine.model.copula import COPULA_DF_GRID
        from risk_engine.service.state import _build_fixture_bundle

        bundle, _, _ = _build_fixture_bundle()
        assert bundle.copula_df != 4.0, "still the pre-A9 hardcoded constant"
        assert COPULA_DF_GRID[0] <= bundle.copula_df <= COPULA_DF_GRID[-1]

    def test_the_diagnostic_judges_the_df_the_bundle_actually_uses(self):
        """A10 asks whether *this* copula understates the lower tail, so it
        must be handed the fitted df. Diagnosing 4.0 while simulating 6.5
        would clear a model that was never checked — and in the reassuring
        direction, because a fatter assumed tail makes the gap look smaller
        than it is.

        Checked by recomputing the model-implied tail dependence directly from
        the pair's rho at both dfs, and asserting the recorded value matches
        the fitted one."""
        from risk_engine.model.copula import model_tail_dependence_at_threshold
        from risk_engine.service.state import _build_fixture_bundle

        bundle, _, _ = _build_fixture_bundle()
        assets = list(bundle.matrix.assets)
        diag = bundle.tail_diagnostics[0]
        i, j = assets.index(diag.pair[0]), assets.index(diag.pair[1])
        rho = float(bundle.matrix.corr[i, j])

        at_fitted = model_tail_dependence_at_threshold(
            bundle.copula_df, rho, diag.threshold, n_sim=200_000, seed=i * 1000 + j)
        at_old = model_tail_dependence_at_threshold(
            4.0, rho, diag.threshold, n_sim=200_000, seed=i * 1000 + j)

        assert abs(at_fitted - at_old) > 0.01, (
            "the two dfs give indistinguishable tail dependence here, so this "
            "test cannot tell which one was diagnosed"
        )
        assert diag.model_at_threshold == pytest.approx(at_fitted, abs=1e-9)

    def test_building_the_fixture_bundle_runs_the_diagnostic(self):
        from risk_engine.observability.metrics import METRICS
        from risk_engine.service.state import _build_fixture_bundle

        METRICS.reset()
        bundle, _, _ = _build_fixture_bundle()

        assert METRICS.counters.get("tail_diagnostics_run", 0) >= 1
        assert METRICS.tail_diagnostics, "the measurement must survive a passing check"
        # 4 assets -> 6 unordered pairs, every one measured rather than a sample.
        assert len(bundle.tail_diagnostics) == 6
        assert not METRICS.counters.get("tail_understated", 0)

    @staticmethod
    def _crash_together_market(seed: int):
        """Two assets that fall together harder than they rise together.

        The one market shape a t-copula provably cannot represent, so it is
        what the §2.3 criterion must refuse. Returned in the shape the bundle
        builders hand to `_checked_tail_diagnostics`.
        """
        rng = np.random.default_rng(seed)
        n = 40_000
        corr = np.array([[1.0, 0.6], [0.6, 1.0]])
        z = rng.standard_normal((n, 2)) @ np.linalg.cholesky(corr).T
        x = z / np.sqrt(rng.chisquare(6.0, size=(n, 1)) / 6.0)
        crash = rng.random(n) < 0.05
        x[crash, :] = -np.abs(x[crash, :]) - 3.0
        return {"A": x[:, 0], "B": x[:, 1]}, SimpleNamespace(assets=["A", "B"], corr=corr)

    def test_the_refusal_is_scoped_by_consumer_not_softened(self):
        """The diagnostic fired on live mainnet and took down the §3.3 shadow
        harness with it — `shadow/cli.py` builds through the same
        `_build_live_bundle`. Refusing to MEASURE a model because it is
        unvalidated is circular, and it guarantees the defect is never
        characterised.

        So `fatal` splits by consumer. It is not an override flag: serving
        keeps refusing, and the recording path still runs the check, still
        records it, and still logs the refusal text."""
        import risk_engine.service.state as state
        from risk_engine.observability.metrics import METRICS

        returns, matrix = self._crash_together_market(38)

        # Serving: refuses, as before.
        with pytest.raises(ValueError, match="understates lower-tail"):
            state._checked_tail_diagnostics(returns, matrix, 6.0, fatal=True)

        # Recording: proceeds, and says so rather than passing silently.
        METRICS.reset()
        diagnostics = state._checked_tail_diagnostics(
            returns, matrix, 6.0, fatal=False)
        assert diagnostics, "the measurement must still be produced"
        assert state.understates_lower_tail(diagnostics)
        assert METRICS.counters.get("tail_understated_recorded_anyway", 0) >= 1
        # And the measurement itself is identical either way — `fatal` decides
        # what happens to it, never what it says.
        assert METRICS.tail_diagnostics[-1]["understates_lower_tail"] is True

    def test_serving_is_the_default_so_a_thoughtless_caller_gets_the_refusal(self):
        """A new caller that does not think about `fatal` must inherit the
        strict behaviour. The permissive path has to be asked for by name."""
        import inspect

        import risk_engine.service.state as state

        assert inspect.signature(
            state._checked_tail_diagnostics).parameters["fatal"].default is True
        assert inspect.signature(
            state._build_live_bundle).parameters["serving"].default is True

    def test_the_shadow_path_asks_for_the_recording_mode_explicitly(self):
        """Pins the wiring: the shadow harness must pass `serving=False`. If a
        refactor drops it, the harness starts refusing again and the §3.3
        window silently stops advancing — the exact failure this split fixes."""
        import inspect
        import re

        import risk_engine.shadow.cli as shadow_cli

        # Whichever function holds the call, `serving=False` must be on it.
        # A regex, not an exact call string: the pin is about the ARGUMENT
        # being passed explicitly, and an exact-string pin broke the first
        # time the call legitimately grew another keyword (C6's shared
        # budget) while the property it guards was untouched.
        builders = [shadow_cli._paced_bundle, shadow_cli._live_world]
        sources = {f.__name__: inspect.getsource(f) for f in builders}
        holding = [
            name for name, src in sources.items()
            if re.search(r"_build_live_bundle\(\s*serving=False", src)
        ]
        assert holding, (
            "no shadow entry point passes serving=False to _build_live_bundle; "
            "the harness would start refusing and the §3.3 window would stop "
            f"advancing. Searched: {list(sources)}"
        )

        # And the live path must reach it through the PACED builder. Calling
        # `_build_live_bundle` directly is what made a spent shared window
        # (C6) kill the run instead of pacing it, so this pins the route as
        # well as the argument.
        assert "_paced_bundle(" in sources["_live_world"], (
            "_live_world must build through _paced_bundle: an unpaced build "
            "turns a busy §5.3 window into a dead run rather than a wait"
        )

        # And it must make NO venue call of its own afterwards. Pinning the
        # paced route was not enough: the next line down re-fetched 90 days
        # of BTC candles for Baseline A -- the same 90 days the build had
        # just fetched -- unpaced, on a window the build had drained. A live
        # run died there with "weight 20 exceeds remaining 0 (323/300
        # spent)" immediately after successfully waiting its turn.
        #
        # Asserted as "no fetch at all" rather than "the fetch is paced",
        # because the data is already in `bundle.factor_returns`: pacing it
        # would have fixed the crash and kept the redundant round trip.
        for call in ("candle_snapshot(", "funding_history(", "clearinghouse_state("):
            assert call not in sources["_live_world"], (
                f"_live_world calls {call} directly. Everything it needs is on "
                "the bundle the paced build returned; a venue call here runs "
                "outside the pacing and duplicates weight already spent."
            )

    def test_a_crash_together_market_stops_the_bundle_from_building(self):
        """The behaviour §2.3 and §9 actually require. If this test can be
        made to pass by any change that lets the service start on a market
        whose lower tail the copula understates, that change is the defect."""
        import risk_engine.service.state as state

        returns, matrix = self._crash_together_market(34)
        with pytest.raises(ValueError, match="understates lower-tail"):
            state._checked_tail_diagnostics(returns, matrix, 6.0)

    def test_the_measurement_is_recorded_even_when_the_build_is_refused(self):
        """A refused bundle is the one whose numbers matter most. Recording
        after the assertion would leave exactly that case unmeasured."""
        import risk_engine.service.state as state
        from risk_engine.observability.metrics import METRICS

        returns, matrix = self._crash_together_market(35)
        METRICS.reset()
        with pytest.raises(ValueError):
            state._checked_tail_diagnostics(returns, matrix, 6.0)
        assert METRICS.counters.get("tail_understated", 0) >= 1
        assert METRICS.tail_diagnostics[-1]["understates_lower_tail"] is True

    def test_unequal_series_lengths_do_not_kill_the_bundle(self):
        """Audit F-2 (PoC-7). `build_global_matrix` aligns series on their
        common tail because live candles differ in length whenever one coin
        has a gap — E5.2 counts gaps precisely because they happen. The A9/A10
        wiring stacked the raw dict instead, so ONE missing candle on ONE coin
        over ninety days killed the live service at startup with a shape
        ValueError: an availability failure introduced by the code meant to
        guard the model, on a condition the matrix builder already survives."""
        import risk_engine.service.state as state

        returns, matrix = self._crash_together_market(37)
        returns["A"] = returns["A"][1:]  # one gap on one coin
        df = state._fitted_copula_df(returns, matrix)
        assert 2.0 <= df <= 31.0
        with pytest.raises(ValueError, match="understates lower-tail"):
            state._checked_tail_diagnostics(returns, matrix, 6.0)

    def test_provenance_records_the_copula_df_that_produced_the_number(self):
        """Audit F-7. Since A9 the df refits on every five-minute bundle
        rebuild, so `seed + model_version` no longer reproduces a number on
        their own — the same seed under 6.5 and 4.0 gives different tails.
        §2.5 calls provenance 'everything needed to reproduce'; the parameter
        now rides with every result, read off the SPEC so baseline B's
        Gaussian rows record None rather than the bundle's fitted value."""
        from risk_engine.sim.engine import MonteCarloEngine
        from risk_engine.service.state import _build_fixture_bundle
        from risk_engine.domain.types import Book, MarginMode, Position

        bundle, specs, spot = _build_fixture_bundle()
        book = Book("0x" + "c" * 40, 50_000.0, (
            Position("BTC", 0.1, spot["BTC"], MarginMode.CROSS, 20.0),
        ), datetime.now(timezone.utc))
        engine = MonteCarloEngine(bundle, specs)
        result = engine.run(book, spot, 24, n_paths=2_000, seed=7)
        assert f"copula_df={bundle.copula_df}" in result.provenance.notes
        # Baseline B simulates independently through the same engine: its rows
        # must record the Gaussian copula (None), not the bundle's fitted df.
        indep = engine.run(book, spot, 24, n_paths=2_000, seed=7, independent=True)
        assert "copula_df=None" in indep.provenance.notes

    def test_the_gaussian_baseline_is_not_held_to_the_t_copula_criterion(self):
        """`copula_df=None` is §3.2's deliberately naive baseline. Refusing it
        for being naive would block the comparator the model is scored against."""
        import risk_engine.service.state as state

        returns, matrix = self._crash_together_market(36)
        assert state._checked_tail_diagnostics(returns, matrix, None) == ()

    def test_every_pairs_tail_reading_is_logged_quiet_ones_included(self, caplog):
        """The 2026-08-05 HYPE probe had to infer "no demand on the new pairs"
        from their ABSENCE in the gate banner — and absence cannot distinguish
        a passing pair from one that never entered the fit, because the banner
        prints firing pairs only. The readings themselves are a probe's
        evidence, so the floor wrapper must log every pair on every build,
        including the ones with nothing to complain about."""
        import logging

        import risk_engine.service.state as state

        # A market the fitted copula genuinely represents: t-copula draws fed
        # back with their own correlation and df, so no pair fires and the
        # quiet-pair half of the claim is the one under test.
        rng = np.random.default_rng(41)
        n = 20_000
        corr = np.array([[1.0, 0.6], [0.6, 1.0]])
        z = rng.standard_normal((n, 2)) @ np.linalg.cholesky(corr).T
        x = z / np.sqrt(rng.chisquare(8.0, size=(n, 1)) / 8.0)
        returns = {"A": x[:, 0], "B": x[:, 1]}
        matrix = SimpleNamespace(assets=["A", "B"], corr=corr)

        with caplog.at_level(logging.INFO, logger="risk_engine.service.state"):
            out_matrix, df = state._tail_remedied_dependence(returns, matrix, 8.0)

        assert df == 8.0, "a market the model fits must not be floored"
        assert out_matrix is matrix, (
            "a market the model fits must not have its matrix touched"
        )
        readings = [r for r in caplog.records if "tail readings" in r.getMessage()]
        assert readings, "the per-pair readings line must be logged"
        assert "A/B" in readings[-1].getMessage(), (
            "the quiet pair must appear in the readings line by name"
        )


class TestShrinkageIntensityMagnitude:
    """Audit A-05. The previous suite checked only the *direction* of
    shrinkage (|shrunk| < |raw|) and that variances survive -- both true of a
    degenerate estimator that always returns intensity 1.0. A live mutation
    inverting `/n_eff` to `*n_eff`, which clips to full equicorrelation,
    survived the entire suite. These pin the magnitude.

    The fixture uses *heterogeneous* factor loadings on purpose: pairwise
    correlations then differ widely, so the constant-correlation target is
    genuinely wrong and the optimal intensity must decay with sample size.
    With independent columns the target is exactly right and intensity 1.0 is
    the correct answer -- which is why the naive fixture cannot tell a working
    estimator from a broken one.
    """

    @staticmethod
    def _hetero(n, a, seed):
        rng = np.random.default_rng(seed)
        f = rng.standard_normal(n)
        loads = np.linspace(0.15, 0.95, a)
        return np.column_stack([
            (loads[i] * f + np.sqrt(1 - loads[i] ** 2) * rng.standard_normal(n))
            * (0.01 + 0.001 * i)
            for i in range(a)
        ])

    @classmethod
    def _intensity(cls, n, a, seed):
        x = cls._hetero(n, a, seed)
        w = ewma_weights(n, half_life=10**9)  # ~flat weights: textbook LW regime
        return ledoit_wolf_constant_correlation(x, w, effective_sample_size(w)).intensity

    def test_intensity_vanishes_as_the_sample_grows(self):
        """LW intensity is O(1/n) against a misspecified target: with abundant
        data the sample estimate wins. A `*n_eff` mutant saturates at 1.0."""
        assert self._intensity(40_000, 8, 4001) < 0.01

    def test_intensity_is_interior_when_data_are_scarce(self):
        """Few observations relative to dimension: shrinkage must be doing
        real work, but must not have collapsed onto the target either."""
        got = self._intensity(80, 12, 4002)
        assert 0.02 < got < 0.95, got

    def test_intensity_falls_monotonically_with_sample_size(self):
        vals = [self._intensity(n, 8, 4003) for n in (100, 400, 1600, 6400)]
        assert all(b < a for a, b in pairwise(vals)), vals

    def test_intensity_saturates_when_the_target_is_exactly_right(self):
        """The other end of the scale, and the reason the old fixture was
        blind: for independent columns the equicorrelated target with r=0 IS
        the truth, so full shrinkage is optimal and expected."""
        rng = np.random.default_rng(4005)
        x = rng.standard_normal((40_000, 5)) * 0.01
        w = ewma_weights(40_000, half_life=10**9)
        got = ledoit_wolf_constant_correlation(x, w, effective_sample_size(w)).intensity
        assert got > 0.9

    def test_full_shrinkage_would_erase_a_known_correlation(self):
        """Guards the consequence, not just the coefficient: with plenty of
        data and a misspecified target, the shrunk correlation must still
        recover the true 0.7 rather than being pulled to the average."""
        rng = np.random.default_rng(4004)
        f = rng.standard_normal(30_000)
        cols = [f * 0.01]
        for load in (0.7, 0.15, 0.95):
            cols.append((load * f + np.sqrt(1 - load**2) * rng.standard_normal(30_000)) * 0.01)
        x = np.column_stack(cols)
        w = ewma_weights(30_000, half_life=10**9)
        res = ledoit_wolf_constant_correlation(x, w, effective_sample_size(w))
        sd = np.sqrt(np.diag(res.cov))
        assert res.cov[0, 1] / (sd[0] * sd[1]) == pytest.approx(0.7, abs=0.02)


class TestImputedRowShape:
    """Audit A-06. The gate's *level* was pinned but the *shape* of the
    imputed row was not: a mutant assigning the same gate correlation to
    every mature asset -- destroying the single-factor structure -- survived
    the whole suite."""

    @staticmethod
    def _series(rng, n=MIN_HISTORY_HOURS * 2):
        """One factor plus a deliberately anti-correlated asset, so that
        `gate * rho(anchor, j)` and a flat `gate` differ in sign, not just
        in magnitude."""
        factor = rng.standard_normal(n)
        out = {
            "BTC": factor * 0.01,
            "ETH": (0.9 * factor + 0.43 * rng.standard_normal(n)) * 0.012,
            "SOL": (0.8 * factor + 0.6 * rng.standard_normal(n)) * 0.015,
            # Moves against the market: a flat-gate imputation would claim a
            # young listing is positively correlated with it.
            "INVERSE": (-0.85 * factor + 0.52 * rng.standard_normal(n)) * 0.02,
        }
        return out

    def test_imputed_row_follows_the_single_factor_through_the_anchor(self):
        rng = np.random.default_rng(4101)
        series = self._series(rng)
        series["NEWCOIN"] = rng.standard_normal(300) * 0.03
        m = build_global_matrix(series)

        young = m.assets.index("NEWCOIN")
        anchor = m.assets.index("BTC")
        gate = m.diagnostics.gate_correlation
        for name in ("ETH", "SOL", "INVERSE"):
            j = m.assets.index(name)
            assert m.corr[young, j] == pytest.approx(gate * m.corr[anchor, j], abs=0.02), name

    def test_imputation_preserves_the_sign_of_the_anchor_relationship(self):
        """The sharp end of A-06: an asset that moves against BTC must not be
        handed a positive correlation with a new listing."""
        rng = np.random.default_rng(4102)
        series = self._series(rng)
        series["NEWCOIN"] = rng.standard_normal(300) * 0.03
        m = build_global_matrix(series)
        young = m.assets.index("NEWCOIN")
        inverse = m.assets.index("INVERSE")
        anchor = m.assets.index("BTC")
        assert m.corr[anchor, inverse] < 0
        assert m.corr[young, inverse] < 0, "flat-gate imputation would put this above zero"

    def test_two_young_assets_get_the_factor_product(self):
        rng = np.random.default_rng(4103)
        series = self._series(rng)
        series["NEW1"] = rng.standard_normal(300) * 0.03
        series["NEW2"] = rng.standard_normal(250) * 0.02
        m = build_global_matrix(series)
        i, j = m.assets.index("NEW1"), m.assets.index("NEW2")
        gate = m.diagnostics.gate_correlation
        assert m.corr[i, j] == pytest.approx(gate * gate, abs=0.02)
