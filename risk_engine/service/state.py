"""The offline half of §2.6, held warm and rebuilt on a schedule.

`EngineState` owns the global correlation matrix, the fitted marginals and
the funding models, and rebuilds them every five minutes (§2.1). A request
takes a submatrix slice of whatever is currently warm; it never estimates
anything itself, which is the whole reason the online path can meet the
300 ms budget.

Two properties matter for §6:

- the state knows how old its matrix is, and says so on `/health`. A matrix
  rebuilt on a five-minute cadence is *routinely* older than the 60-second
  staleness threshold that applies to book and price data, so the two clocks
  are reported separately and thresholded separately (OPEN-QUESTIONS D4).
- a failed rebuild does not replace the good state with a broken one, and it
  does not silently keep serving either: the failure is recorded, the age
  keeps climbing, and the backend's own thresholds take over. Serving a
  confidently stale number during a crash is the worst failure this product
  has (§6).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from risk_engine.domain.types import AssetSpec
from risk_engine.model.copula import (
    COPULA_DF_GRID,
    assert_lower_tail_not_understated,
    diagnose_tail_asymmetry,
    fit_copula_df,
)
from risk_engine.model.correlation import build_global_matrix
from risk_engine.model.funding import FundingBounds, fit_ar1
from risk_engine.model.marginals import fit_marginal
from risk_engine.observability.metrics import METRICS
from risk_engine.sim.engine import ModelBundle
from risk_engine.version import MODEL_VERSION

log = logging.getLogger("risk_engine.service.state")

#: Only reachable with a single-asset universe, where there is no pair and so
#: nothing to estimate. Named rather than written as a bare 4.0 so it cannot be
#: mistaken for the pre-A9 hardcoded default it replaces.
HL_FALLBACK_COPULA_DF = 4.0


def _fitted_copula_df(returns: dict, matrix) -> float:
    """The copula's degrees of freedom, estimated rather than assumed (A9).

    Both bundle builders passed a hardcoded `copula_df=4.0` while
    `fit_copula_df` sat implemented, tested and uncalled — and A9 described
    the two-stage IFM estimator in the present tense as though it were in use.
    On the fixture the fitted value is 6.5 against that 4.0: a materially
    thinner joint tail, not a rounding difference.

    Wiring it changes every number the model produces, which is why it is a
    MINOR version bump and resets the §3.3 shadow counter (see version.py).
    It was done before the clock started, when that costs nothing; afterwards
    it costs up to 21 days, and the alternative was validating a magic
    constant nobody could source.

    The df is profiled over `COPULA_DF_GRID`, so it is bounded by
    construction. Landing on either end is recorded rather than trusted: the
    grid's floor means "heavier joint tails than this grid can express" and
    its ceiling means "indistinguishable from Gaussian dependence", and both
    are statements about the data outrunning the model family — the same
    reason §2.2's marginal clamps are logged.
    """
    if len(matrix.assets) < 2:
        # No pair, no dependence to estimate. Cannot happen on the live path
        # (it requires BTC and ETH) but the fixture layout is editable.
        return HL_FALLBACK_COPULA_DF
    series = np.column_stack([returns[a] for a in matrix.assets])
    df = float(fit_copula_df(series, matrix.corr))
    lo, hi = float(COPULA_DF_GRID[0]), float(COPULA_DF_GRID[-1])
    if df <= lo or df >= hi:
        METRICS.incr("copula_df_at_grid_edge")
        METRICS.df_clamps.append({
            "which": "copula", "fitted": df, "grid_lo": lo, "grid_hi": hi,
            "meaning": ("joint tails heavier than the grid can express"
                        if df <= lo else "dependence indistinguishable from Gaussian"),
        })
        log.warning(
            "copula df fitted to the edge of its grid (%.2f, grid %.2f-%.2f): %s",
            df, lo, hi,
            "tails heavier than representable" if df <= lo else "≈ Gaussian dependence",
        )
    return df


def _checked_tail_diagnostics(returns: dict, matrix, copula_df: float | None) -> tuple:
    """Run §2.3's tail-asymmetry diagnostic and refuse if it fires.

    OPEN-QUESTIONS A10. `diagnose_tail_asymmetry` and
    `assert_lower_tail_not_understated` existed, were tested, and were called
    from **no shipped path** — while `model/copula.py` described the assertion
    in the present tense as something that "turns it into a hard failure". The
    mandated check was a function nobody invoked, which is worse than not
    having it: the docstring made the model look guarded.

    Why it refuses rather than warns. A t-copula's tail dependence is
    symmetric by construction. Crypto is not — assets crash together harder
    than they rally together. When the empirical lower tail exceeds what the
    fitted copula produces, the model understates the probability of the
    joint move that liquidates a leveraged book, which is the one direction
    §10 forbids simplifying in. §2.3 names the remedy (a skewed-t) and Phase 1
    does not implement it, so there is nothing to fall back to; §9 requires
    an unmet criterion to stop and be reported rather than worked around.
    Starting anyway would serve numbers that are wrong in the direction the
    product exists to protect against.

    There is deliberately no override flag. A flag would be used the first
    time it was inconvenient, and "the risk model understates crashes" is not
    a condition anyone should be able to click past.

    A `copula_df` of None means the Gaussian baseline (§3.2), which is a
    deliberately naive comparator rather than the shipped model; diagnosing it
    against §2.3's criterion would refuse the baseline for being what it is
    supposed to be.
    """
    if copula_df is None or len(matrix.assets) < 2:
        return ()
    series = np.column_stack([returns[a] for a in matrix.assets])
    diagnostics = diagnose_tail_asymmetry(
        series, tuple(matrix.assets), matrix.corr, copula_df
    )
    # Recorded BEFORE the assertion, so a bundle that is about to be refused
    # still leaves the measurement behind. Otherwise the one build whose
    # numbers matter most is the only one that reports nothing.
    METRICS.record_tail_diagnostics(diagnostics)
    assert_lower_tail_not_understated(diagnostics)
    return tuple(diagnostics)


class NotReady(RuntimeError):
    """No usable bundle yet, or the last one is unusable."""


@dataclass
class EngineState:
    _bundle: ModelBundle | None = None
    _specs: dict[str, AssetSpec] = field(default_factory=dict)
    _spot: dict[str, float] = field(default_factory=dict)
    _built_at: datetime | None = None
    _last_error: str | None = None
    _last_attempt_at: datetime | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _rebuild: object = field(default=None, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)

    # -- construction ---------------------------------------------------

    @classmethod
    def fixture(cls) -> EngineState:
        """A synthetic but fully-shaped bundle.

        Exists because `api.hyperliquid.xyz` is unreachable from some build
        environments (OPEN-QUESTIONS E5), and because the §6 degradation
        contract has to be exercisable without a live venue -- the whole
        point of that contract is what happens when data stops arriving.
        The numbers are synthetic and are labelled as such on `/health`, so
        nothing downstream can mistake this for live risk.
        """
        state = cls()
        state._rebuild = _build_fixture_bundle
        state.refresh()
        return state

    @classmethod
    def live(cls) -> EngineState:
        state = cls()
        state._rebuild = _build_live_bundle
        state.refresh()
        return state

    # -- access ---------------------------------------------------------

    def require_ready(self) -> tuple[ModelBundle, dict[str, AssetSpec], dict[str, float]]:
        with self._lock:
            if self._bundle is None:
                raise NotReady(self._last_error or "no model bundle has been built yet")
            return self._bundle, self._specs, dict(self._spot)

    def matrix_age_s(self) -> float | None:
        with self._lock:
            if self._built_at is None:
                return None
            return (datetime.now(timezone.utc) - self._built_at).total_seconds()

    def health(self) -> dict:
        with self._lock:
            built = self._built_at
            ready = self._bundle is not None
            assets = list(self._bundle.matrix.assets) if self._bundle else []
            diag = self._bundle.matrix.diagnostics if self._bundle else None
            error = self._last_error
            attempt = self._last_attempt_at
            synthetic = getattr(self._rebuild, "is_fixture", False)
        age = (datetime.now(timezone.utc) - built).total_seconds() if built else None
        return {
            "ready": ready,
            "model_version": MODEL_VERSION,
            "synthetic_data": bool(synthetic),
            # §6 requires the age of the last successful matrix rebuild to be
            # exposed. It is deliberately NOT compared against the 60s book
            # threshold here: the matrix rebuilds every 5 minutes by design.
            "matrix_age_s": age,
            "matrix_built_at": built.isoformat() if built else None,
            "last_attempt_at": attempt.isoformat() if attempt else None,
            "last_error": error,
            "assets": assets,
            "diagnostics": (
                {
                    "shrinkage_intensity": diag.shrinkage_intensity,
                    "mean_correlation": diag.mean_correlation,
                    "n_eff": diag.n_eff,
                    "window_hours": diag.window_hours,
                    "imputed_assets": list(diag.imputed_assets),
                    "gate_correlation": diag.gate_correlation,
                    "psd_corrected": diag.psd_corrected,
                    "min_eigenvalue": diag.min_eigenvalue,
                }
                if diag
                else None
            ),
        }

    # -- refresh --------------------------------------------------------

    def refresh(self) -> bool:
        """Rebuild the bundle. Returns whether it succeeded.

        A failure leaves the previous bundle in place and records the error.
        That is deliberate: the alternative -- dropping to no data -- would
        make a transient venue hiccup indistinguishable from a crash. What
        keeps a stale bundle from being served forever is that its age keeps
        climbing and the backend's §6 thresholds act on it.
        """
        attempt = datetime.now(timezone.utc)
        try:
            bundle, specs, spot = self._rebuild()  # type: ignore[misc]
        except Exception as exc:
            log.warning("matrix rebuild failed: %s", exc)
            METRICS.incr("matrix_rebuild_failures")
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._last_attempt_at = attempt
            return False
        with self._lock:
            self._bundle, self._specs, self._spot = bundle, specs, spot
            self._built_at = datetime.now(timezone.utc)
            self._last_attempt_at = attempt
            self._last_error = None
        METRICS.incr("matrix_rebuilds")
        log.info("matrix rebuilt over %d assets", len(bundle.matrix.assets))
        return True

    def start_refresh_loop(self, seconds: float = 300.0) -> threading.Thread:
        def loop() -> None:
            while not self._stop.wait(seconds):
                self.refresh()

        thread = threading.Thread(target=loop, daemon=True, name="matrix-refresh")
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()


def _build_fixture_bundle():
    """One-factor synthetic market, wide enough to exercise every code path."""
    from scipy import stats

    rng = np.random.default_rng(20260729)
    n = 30 * 24 * 3
    factor = stats.t(df=4.0).rvs(n, random_state=rng) / np.sqrt(2.0)
    layout = (
        ("BTC", 1.00, 0.008, 40.0, 100_000.0),
        ("ETH", 0.90, 0.011, 25.0, 4_000.0),
        ("SOL", 0.80, 0.015, 20.0, 200.0),
        ("AVAX", 0.75, 0.018, 10.0, 35.0),
    )
    returns, specs, spot = {}, {}, {}
    for name, load, vol, max_lev, px in layout:
        idio = stats.t(df=5.0).rvs(n, random_state=rng) / np.sqrt(5 / 3)
        returns[name] = (load * factor + np.sqrt(max(1 - load**2, 0.05)) * idio) * vol
        from risk_engine.domain.types import MarginTier

        specs[name] = AssetSpec(name, 4, max_lev, (MarginTier(0.0, max_lev),))
        spot[name] = px

    matrix = build_global_matrix(returns)
    marginals = {
        c: fit_marginal(c, r, float(matrix.step_vol[matrix.assets.index(c)]))
        for c, r in returns.items()
    }
    bounds = FundingBounds.documented_default()
    funding = {
        c: fit_ar1(c, 1e-5 + 2e-5 * rng.standard_normal(30 * 24), bounds)
        for c in returns
    }
    # The fixture is a symmetric one-factor market by construction, so the
    # diagnostic is expected to pass and is run anyway -- a check that only
    # runs on the path nobody exercises offline is a check that rots. It also
    # means the fixture asserts the diagnostic's own plumbing on every startup.
    copula_df = _fitted_copula_df(returns, matrix)
    bundle = ModelBundle(
        matrix=matrix, marginals=marginals, funding=funding,
        funding_bounds=bounds, copula_df=copula_df,
        tail_diagnostics=_checked_tail_diagnostics(returns, matrix, copula_df),
    )
    return bundle, specs, spot


_build_fixture_bundle.is_fixture = True  # type: ignore[attr-defined]


def _build_live_bundle():
    """Fetch from the Info API and fit (§5.1).

    NOT EXERCISED against the live API: `api.hyperliquid.xyz` is blocked at
    the proxy in the environment this was written in (OPEN-QUESTIONS E5), so
    this path is written against documented response shapes and has never
    executed end to end. It must be verified before it is trusted, which is
    why the fixture path exists and is what Phase 3's degradation test runs
    against.
    """
    from risk_engine.market.info import InfoClient
    from risk_engine.market.parse import (
        parse_candles_to_log_returns,
        parse_funding_history,
        parse_meta,
    )

    client = InfoClient()
    specs = parse_meta(client.meta())
    now_ms = int(time.time() * 1000)
    window_ms = 90 * 24 * 3600 * 1000

    universe = [c for c in ("BTC", "ETH", "SOL") if c in specs]
    if "BTC" not in universe or "ETH" not in universe:
        raise RuntimeError("BTC and ETH must be present as risk factors (§2.1)")

    returns, spot, funding_hist = {}, {}, {}
    for coin in universe:
        candles = client.candle_snapshot(coin, "1h", now_ms - window_ms, now_ms)
        _, rets = parse_candles_to_log_returns(candles)
        returns[coin] = rets
        spot[coin] = float(sorted(candles, key=lambda c: int(c["t"]))[-1]["c"])
        _, rates = parse_funding_history(
            client.funding_history(coin, now_ms - 30 * 24 * 3600 * 1000)
        )
        funding_hist[coin] = rates

    matrix = build_global_matrix(returns)
    marginals = {
        c: fit_marginal(c, r, float(matrix.step_vol[matrix.assets.index(c)]))
        for c, r in returns.items()
    }
    bounds = FundingBounds.documented_default()
    funding = {c: fit_ar1(c, r, bounds) for c, r in funding_hist.items()}
    # This is the call §2.3 is actually about, and the one that may refuse to
    # start the service. Ninety days of real hourly crypto returns is exactly
    # the data a symmetric copula is least able to represent, so a failure
    # here is a finding about the market and the model, not a bug -- and
    # finding it at startup is the point. See `_checked_tail_diagnostics`.
    #
    # The two are coupled and the direction is worth knowing before it
    # happens: A9's fitted df is thinner-tailed than the 4.0 it replaced
    # wherever the data say so, and a thinner model tail sits further below
    # the empirical one, which makes A10 MORE likely to fire. On the fixture
    # that moved the worst gap from -0.011 to +0.022 against a 0.05 margin.
    # If the live build starts refusing, that is the two working as specified
    # -- a fitted copula that cannot represent real crypto crashes is exactly
    # what §2.3 exists to catch -- not a regression to route around.
    copula_df = _fitted_copula_df(returns, matrix)
    bundle = ModelBundle(
        matrix=matrix, marginals=marginals, funding=funding,
        funding_bounds=bounds, copula_df=copula_df,
        tail_diagnostics=_checked_tail_diagnostics(returns, matrix, copula_df),
    )
    return bundle, {c: specs[c] for c in universe}, spot
