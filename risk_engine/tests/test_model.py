"""§2 — the estimation layer."""

from __future__ import annotations

from datetime import datetime, timezone

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
