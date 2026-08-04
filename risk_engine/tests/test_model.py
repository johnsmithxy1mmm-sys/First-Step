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
        assert live_btc_eth.understates_lower_tail(), "it must still fail the gate"
        assert live_btc_eth.asymmetry < 0, "the upper tail is the heavier one"
        assert live_btc_eth.upper_also_understated

        with pytest.raises(ValueError, match="NOT AN ASYMMETRY"):
            assert_lower_tail_not_understated([live_btc_eth])

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
