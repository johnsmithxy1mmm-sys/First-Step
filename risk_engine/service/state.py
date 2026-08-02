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
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from risk_engine.domain.types import AssetSpec, Book
from risk_engine.model.copula import (
    COPULA_DF_GRID,
    assert_lower_tail_not_understated,
    diagnose_tail_asymmetry,
    fit_copula_df,
)
from risk_engine.model.correlation import align_on_timestamps, build_global_matrix
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

#: How long a fetched book may be served from cache. Far inside §6's 60s
#: fresh tier; see `EngineState.fetch_book`.
BOOK_CACHE_TTL_S = 15.0


class BookUnavailable(RuntimeError):
    """The venue could not supply a book right now.

    Its message is written to be shown to the caller: the backend maps it to
    a non-500 status and the §6 contract renders it as `unavailable` with a
    reason, which is the honest answer -- distinct from a malformed request
    (400) and from a genuine engine bug (500, message withheld).
    """


def _estimated_assets(matrix) -> tuple[str, ...]:
    """The matrix assets whose correlation row was ESTIMATED, not imputed.

    A9's copula fit and A10's tail gate both score a fitted dependence
    structure against `matrix.corr`. For a young asset that row was never
    estimated — §2.1's gate imputes it through the anchor precisely because a
    three-week sample produces a correlation biased toward zero exactly when
    the asset is most likely to move with everything else. Scoring a copula
    against an imputed row measures the imputation, not the market.

    Worse, the old code took `min(size)` over ALL assets, so one young asset
    truncated the fit and the gate to its own short window: reproduced at a
    fitted copula df of 3.5 against 5.0 on the mature window, and a fatal
    §2.3 refusal fired off two joint observations. The young-asset gate exists
    to keep short samples out of dependence estimation; this routed them back
    in through the side door.
    """
    diagnostics = getattr(matrix, "diagnostics", None)
    # getattr, not attribute access: benchmarks and tests build minimal matrix
    # stubs that carry only `assets` and `corr`. A stub has no imputed rows by
    # construction, so "no diagnostics" correctly means "all estimated".
    imputed = set(getattr(diagnostics, "imputed_assets", ()) or ())
    return tuple(a for a in matrix.assets if a not in imputed)


def _corr_submatrix(matrix, assets) -> np.ndarray:
    """`assets`'s principal submatrix of `matrix.corr`, stub-tolerant."""
    if tuple(assets) == tuple(matrix.assets):
        return np.asarray(matrix.corr, dtype=np.float64)
    order = list(matrix.assets)
    idx = np.array([order.index(a) for a in assets], dtype=np.intp)
    return np.asarray(matrix.corr, dtype=np.float64)[np.ix_(idx, idx)]


def _aligned_series(returns: dict, assets, timestamps: dict | None = None) -> np.ndarray:
    """The return series stacked on their common tail (audit F-2).

    Aligns on shared candle timestamps when they are available (audit H5) and
    falls back to the common trailing index when they are not — the same
    ordering `build_global_matrix` uses, so the copula is fitted against the
    correlation matrix on the same rows rather than on a one-hour shift.

    `build_global_matrix` aligns on the common tail deliberately
    (`window = min(...)` in correlation.py) — live candle series differ in
    length whenever one coin has a gap, and E5.2 counts gaps precisely
    because they happen. The A9/A10 wiring stacked the RAW dict instead, so
    a single missing candle on a single coin over ninety days would have
    killed the live service at startup with a shape ValueError — an
    availability failure introduced by the very code meant to guard the
    model, on a data condition the matrix builder already survives.
    """
    if timestamps is not None and all(a in timestamps for a in assets):
        _, x = align_on_timestamps(returns, timestamps, list(assets))
        return x
    window = min(np.asarray(returns[a]).size for a in assets)
    return np.column_stack(
        [np.asarray(returns[a], dtype=np.float64)[-window:] for a in assets]
    )


def _fitted_copula_df(returns: dict, matrix, timestamps: dict | None = None) -> float:
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
    # Estimated rows only: a copula fitted against an imputed correlation row
    # is scoring the imputation (audit H5 / `_estimated_assets`).
    assets = _estimated_assets(matrix)
    if len(assets) < 2:
        # No pair, no dependence to estimate. Cannot happen on the live path
        # (it requires BTC and ETH, both mature) but the fixture layout is
        # editable and a universe of only-young assets is conceivable.
        return HL_FALLBACK_COPULA_DF
    series = _aligned_series(returns, assets, timestamps)
    df = float(fit_copula_df(series, _corr_submatrix(matrix, assets)))
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


def _checked_tail_diagnostics(returns: dict, matrix, copula_df: float | None,
                              fatal: bool = True, timestamps: dict | None = None) -> tuple:
    """Run §2.3's tail-asymmetry diagnostic; refuse when `fatal`.

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

    There is deliberately no override flag. `fatal` is NOT one, and the
    difference is the whole point of it existing.

    The diagnostic fired on live mainnet data on 2026-08-01, and the refusal
    took down something it was never aimed at. `shadow/cli.py:149` builds its
    bundle through the same `_build_live_bundle`, so the assertion blocked the
    §3.3 shadow harness as well as the serving path — and the shadow harness
    is the instrument that MEASURES whether an unvalidated model is any good.
    Refusing to measure a model because it is unvalidated is circular, and it
    is the one outcome that guarantees the defect is never characterised.

    So the refusal is scoped by CONSUMER, not softened:

      - a path that shows a number to a person keeps refusing (`fatal=True`).
        §10 is engaged there: nobody may be served a figure from a model known
        to understate the joint move that liquidates their book.
      - a path that only RECORDS observations runs (`fatal=False`). No one is
        told anything; the finding is stamped into the run's provenance and
        travels with every row it produces.

    What makes this worth doing rather than merely permissible: an adversarial
    review of the skewed-t remedy established that the **sign** of the error
    depends on the shape of the book -- `_any_liq` is a union over positions
    and `Position.size` is signed, so heavier joint downside does not move a
    hedged book's liquidation probability the same way it moves a long-only
    one. Nobody knows the magnitude or the direction on real books. The shadow
    window is what answers that, and skewed-t cannot be designed correctly
    without the answer.

    The days recorded this way do NOT count toward §3.3's gate: the remedy
    will bump MODEL_VERSION and reset the counter. They are diagnostic
    evidence, not gate-days, and calling them anything else would be the same
    laundering this file's other guards exist to prevent.

    A `copula_df` of None means the Gaussian baseline (§3.2), which is a
    deliberately naive comparator rather than the shipped model; diagnosing it
    against §2.3's criterion would refuse the baseline for being what it is
    supposed to be.
    """
    if copula_df is None or len(matrix.assets) < 2:
        return ()
    # Estimated rows only, for the same reason as the copula fit: the gate
    # compares an empirical tail against the model's tail at the pair's rho,
    # and an imputed rho is a §2.1 assumption rather than a measurement.
    assets = _estimated_assets(matrix)
    if len(assets) < 2:
        return ()
    series = _aligned_series(returns, assets, timestamps)
    diagnostics = diagnose_tail_asymmetry(
        series, tuple(assets), _corr_submatrix(matrix, assets), copula_df
    )
    # Recorded BEFORE the assertion, so a bundle that is about to be refused
    # still leaves the measurement behind. Otherwise the one build whose
    # numbers matter most is the only one that reports nothing.
    METRICS.record_tail_diagnostics(diagnostics)
    if fatal:
        assert_lower_tail_not_understated(diagnostics)
    else:
        # Not a silent pass. The same text the refusal would have carried is
        # logged at WARNING and counted, so a recording run says on every
        # rebuild what it is recording under.
        try:
            assert_lower_tail_not_understated(diagnostics)
        except ValueError as exc:
            METRICS.incr("tail_understated_recorded_anyway")
            log.warning(
                "RECORDING UNDER A KNOWN §2.3 DEFECT (no number is being served "
                "from this bundle): %s", exc,
            )
    return tuple(diagnostics)


def understates_lower_tail(diagnostics) -> bool:
    """Whether §2.3's criterion is currently violated.

    One predicate, so the serving path and the provenance stamp cannot drift
    apart in what they call a defect.
    """
    return any(d.understates_lower_tail() for d in diagnostics)


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
    #: Per-address book cache for the wallet path; (Book, monotonic stamp).
    _book_cache: dict = field(default_factory=dict, repr=False)
    _book_client: object = field(default=None, repr=False)

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

    @property
    def is_fixture(self) -> bool:
        return bool(getattr(self._rebuild, "is_fixture", False))

    def fetch_book(self, address: str) -> Book:
        """The live book behind `address`, for the wallet path (§4.1).

        The frontend used to send a hardcoded demo book; a connected wallet
        sends an address instead, and the engine fetches the book itself --
        the parsing, the §5.1 strictness and the §5.3 weight discipline all
        already live on this side of the wire, and `parse_clearinghouse_state`
        stamps `captured_at` at the fetch, which is exactly the timestamp the
        §6 book clock judges.

        Cached per address for `BOOK_CACHE_TTL_S`. The UI polls every 20
        seconds and each fetch is 20 §5.3 weight against serving's 900/minute
        -- the cache keeps one user at ~3 fetches/minute instead of every
        widget refresh paying full price, while staying far inside the 60s
        freshness tier.

        §5.1's named footgun applies and cannot be detected here: an AGENT
        address returns a well-formed empty state identical to a genuinely
        flat account's. The UI says "main account, not an agent address" at
        the input; a flat answer for a live trader is the symptom to check.
        """
        from risk_engine.domain.types import normalise_address

        if self.is_fixture:
            raise ValueError(
                "this engine runs on synthetic fixture data (see /health); it cannot "
                "fetch a real address's book. Send the book in the payload, or run "
                "the engine with --live."
            )
        address = normalise_address(address)
        now_mono = time.monotonic()
        with self._lock:
            hit = self._book_cache.get(address)
            if hit is not None and now_mono - hit[1] < BOOK_CACHE_TTL_S:
                return hit[0]
            client = self._book_client
        if client is None:
            from risk_engine.market.info import (
                SERVING_RESERVED_FRACTION,
                InfoClient,
                WeightBudget,
            )

            client = InfoClient(
                budget=WeightBudget(reserved_fraction=SERVING_RESERVED_FRACTION)
            )
            with self._lock:
                # Another request may have raced the construction; either
                # object is fine, but only one is kept.
                if self._book_client is None:
                    self._book_client = client
                client = self._book_client

        from risk_engine.market.info import RateLimitExceeded
        from risk_engine.market.parse import parse_clearinghouse_state

        try:
            state = client.clearinghouse_state(address)
        except RateLimitExceeded as exc:
            raise BookUnavailable(
                "the §5.3 weight window is spent; retry in a few seconds"
            ) from exc
        except Exception as exc:
            # The venue being unreachable is an availability answer the §6
            # contract wants to render, not an internal error to hide.
            raise BookUnavailable(
                f"could not fetch the book from the venue: {type(exc).__name__}"
            ) from exc
        book = parse_clearinghouse_state(state, address)
        with self._lock:
            self._book_cache[address] = (book, now_mono)
            # The cache is per polling user; a stale entry costs memory, not
            # correctness, and pruning on write keeps it bounded anyway.
            expired = [
                a for a, (_, t) in self._book_cache.items()
                if now_mono - t >= BOOK_CACHE_TTL_S
            ]
            for a in expired:
                del self._book_cache[a]
        return book

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

    def built_at(self) -> datetime | None:
        """When the warm bundle -- and therefore its prices -- was last built.

        §6's freshness contract has to be driven by the age of the DATA a
        number was computed from, not by when the arithmetic ran. The mark
        prices in `_spot` are written only by `refresh()`, so this is their
        observation time; the backend thresholds it on the matrix's own clock
        (OPEN-QUESTIONS D4) rather than the 60-second book clock.
        """
        with self._lock:
            return self._built_at

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


def _build_live_bundle(*, serving: bool = True, budget=None):
    """Fetch from the Info API and fit (§5.1).

    `serving` decides whether §2.3's tail-asymmetry criterion is fatal. It
    defaults to True so that every existing caller — and every future one that
    does not think about it — gets the refusal. The shadow harness passes
    False deliberately; see `_checked_tail_diagnostics` for why measuring is
    not serving.

    `budget` lets a shadow caller charge this build to the shared §5.3 pool
    (C6): the resolver rebuilds hourly and a bundle costs ~160 weight, which
    used to land on a private budget no other process could see. Left None,
    the build takes the serving allocation — correct for the engine, which is
    the interactive side the reserve protects.

    NOT EXERCISED against the live API: `api.hyperliquid.xyz` is blocked at
    the proxy in the environment this was written in (OPEN-QUESTIONS E5), so
    this path is written against documented response shapes and has never
    executed end to end. It must be verified before it is trusted, which is
    why the fixture path exists and is what Phase 3's degradation test runs
    against.
    """
    from risk_engine.market.info import (
        SERVING_RESERVED_FRACTION,
        InfoClient,
        WeightBudget,
    )
    from risk_engine.market.parse import (
        parse_candles_to_log_returns,
        parse_funding_history,
        parse_meta,
    )

    # Reserved rather than unbounded (C6): serving takes 75% of §5.3's window
    # and the shadow jobs share the other 25%, so the two together are the
    # venue's limit instead of 125% of it. Nothing on the request path spends
    # this -- a request slices the warm bundle -- so the cap constrains the
    # five-minute rebuild only, which needs about 140 weight.
    client = InfoClient(
        budget=budget
        if budget is not None
        else WeightBudget(reserved_fraction=SERVING_RESERVED_FRACTION)
    )
    specs = parse_meta(client.meta())
    now_ms = int(time.time() * 1000)
    window_ms = 90 * 24 * 3600 * 1000

    # Config, not code (OPEN-QUESTIONS B6). The 3-asset default silently
    # narrows the calibration cohort: an address whose positions are all
    # off-universe is skipped as "no open positions", byte-identical to a flat
    # account, and §3.3's 200-address count comes up short for a reason no
    # output names. The `calibration_sweeps` census measures that drop rate;
    # when it argues for widening, the widening is an env change here, one
    # more candle+funding fetch per coin per rebuild (~40 weight each against
    # serving's 900/min), and a B6 decision recorded in OPEN-QUESTIONS — not a
    # code edit. BTC and ETH stay mandatory (§2.1's risk factors, enforced
    # below); coins the venue does not list are dropped with the same
    # visibility as before.
    configured = [
        c.strip().upper()
        for c in os.environ.get("HL_UNIVERSE", "BTC,ETH,SOL").split(",")
        if c.strip()
    ]
    universe = [c for c in configured if c in specs]
    missing_coins = sorted(set(configured) - set(universe))
    if missing_coins:
        log.warning(
            "HL_UNIVERSE names coins the venue's meta does not list: %s "
            "(they are excluded; check the spelling against `meta.universe`)",
            missing_coins,
        )
    if "BTC" not in universe or "ETH" not in universe:
        raise RuntimeError("BTC and ETH must be present as risk factors (§2.1)")

    returns, spot, funding_hist, candle_times = {}, {}, {}, {}
    for coin in universe:
        candles = client.candle_snapshot(coin, "1h", now_ms - window_ms, now_ms)
        times, rets = parse_candles_to_log_returns(candles)
        returns[coin] = rets
        # Kept, not discarded (audit H5). Every caller used to throw these
        # away, which left `build_global_matrix` aligning coins on trailing
        # array index -- correct only while no coin has a candle gap.
        candle_times[coin] = times
        spot[coin] = float(sorted(candles, key=lambda c: int(c["t"]))[-1]["c"])
        _, rates = parse_funding_history(
            client.funding_history(coin, now_ms - 30 * 24 * 3600 * 1000)
        )
        funding_hist[coin] = rates

    matrix = build_global_matrix(returns, timestamps=candle_times)
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
    copula_df = _fitted_copula_df(returns, matrix, candle_times)
    bundle = ModelBundle(
        matrix=matrix, marginals=marginals, funding=funding,
        funding_bounds=bounds, copula_df=copula_df,
        tail_diagnostics=_checked_tail_diagnostics(
            returns, matrix, copula_df, fatal=serving, timestamps=candle_times),
    )
    return bundle, {c: specs[c] for c in universe}, spot
