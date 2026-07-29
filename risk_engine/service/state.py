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
from risk_engine.model.correlation import build_global_matrix
from risk_engine.model.funding import FundingBounds, fit_ar1
from risk_engine.model.marginals import fit_marginal
from risk_engine.observability.metrics import METRICS
from risk_engine.sim.engine import ModelBundle
from risk_engine.version import MODEL_VERSION

log = logging.getLogger("risk_engine.service.state")


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
    bundle = ModelBundle(
        matrix=matrix, marginals=marginals, funding=funding,
        funding_bounds=bounds, copula_df=4.0,
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
    bundle = ModelBundle(
        matrix=matrix, marginals=marginals, funding=funding,
        funding_bounds=bounds, copula_df=4.0,
    )
    return bundle, {c: specs[c] for c in universe}, spot
