"""The single global correlation matrix (§2.1).

One matrix over every tracked asset — top-50 by open interest plus every
asset appearing in an active user's book — rebuilt on a schedule. Individual
requests take a principal submatrix of it.

This is the architectural decision that makes the rest work:

  - a principal submatrix of a positive-definite matrix is positive definite,
    so the slice always factorises and no per-request projection is needed;
  - there is nothing to cache per user universe, so no combinatorial blowup;
  - `pre_trade_delta` on an asset the user does not yet hold costs a slice,
    not a cold start, which is what keeps §4.2 inside the 300 ms budget;
  - two users holding overlapping books get consistent numbers.

Young assets (§2.1's data gate) never receive an estimated row. A new
listing with three weeks of history produces a correlation estimate that is
biased toward zero exactly when the asset is most likely to move with
everything else, and underestimating a new listing's correlation is the most
dangerous single error this model can make. They get an imputed row instead,
anchored at a high quantile of the mature assets' correlation with BTC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from risk_engine.model.ewma import DEFAULT_HALF_LIFE_HOURS, effective_sample_size, ewma_weights
from risk_engine.model.psd import project_to_correlation
from risk_engine.model.shrinkage import ledoit_wolf_constant_correlation
from risk_engine.observability.metrics import METRICS, Metrics

#: §2.1's data gate: assets with less than 30 days of hourly history.
MIN_HISTORY_HOURS = 30 * 24
GATE_QUANTILE = 0.9
ANCHOR = "BTC"
#: §2.1 requires BTC and ETH to always be present as risk factors.
REQUIRED_FACTORS = ("BTC", "ETH")


@dataclass(frozen=True, slots=True)
class MatrixDiagnostics:
    shrinkage_intensity: float
    mean_correlation: float
    n_eff: float
    window_hours: int
    imputed_assets: tuple[str, ...]
    gate_correlation: float
    psd_corrected: bool
    psd_frobenius: float
    min_eigenvalue: float


@dataclass(frozen=True, slots=True)
class GlobalCorrelationMatrix:
    assets: tuple[str, ...]
    corr: np.ndarray
    step_vol: np.ndarray
    computed_at: datetime
    diagnostics: MatrixDiagnostics
    _index: dict[str, int] = field(init=False, repr=False, compare=False)
    _chol_cache: dict[tuple[str, ...], np.ndarray] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.computed_at.tzinfo is None:
            raise ValueError("computed_at must be timezone-aware")
        object.__setattr__(self, "_index", {a: i for i, a in enumerate(self.assets)})
        object.__setattr__(self, "_chol_cache", {})

    def age_seconds(self, now: datetime | None = None) -> float:
        return ((now or datetime.now(timezone.utc)) - self.computed_at).total_seconds()

    def covers(self, coins: tuple[str, ...]) -> bool:
        return all(c in self._index for c in coins)

    def submatrix(self, coins: tuple[str, ...]) -> np.ndarray:
        try:
            idx = np.array([self._index[c] for c in coins], dtype=np.intp)
        except KeyError as exc:  # pragma: no cover - guarded by covers()
            raise KeyError(f"{exc.args[0]} is not in the global matrix; add it and rebuild") from exc
        return self.corr[np.ix_(idx, idx)]

    def volatilities(self, coins: tuple[str, ...]) -> np.ndarray:
        return np.array([self.step_vol[self._index[c]] for c in coins], dtype=np.float64)

    def cholesky(self, coins: tuple[str, ...]) -> np.ndarray:
        """Lower Cholesky factor of the slice. Cached; the slice is PD by §2.1."""
        key = tuple(coins)
        hit = self._chol_cache.get(key)
        if hit is not None:
            return hit
        sub = self.submatrix(key)
        try:
            chol = np.linalg.cholesky(sub)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - would be a real bug
            raise RuntimeError(
                "a slice of the global matrix failed to factorise; the global matrix was "
                "not positive definite, which project_to_correlation is supposed to guarantee"
            ) from exc
        self._chol_cache[key] = chol
        return chol


def build_global_matrix(
    returns: dict[str, np.ndarray],
    now: datetime | None = None,
    half_life: float = DEFAULT_HALF_LIFE_HOURS,
    min_history_hours: int = MIN_HISTORY_HOURS,
    gate_quantile: float = GATE_QUANTILE,
    anchor: str = ANCHOR,
    metrics: Metrics | None = None,
) -> GlobalCorrelationMatrix:
    """Build the global matrix from per-asset hourly log-return series.

    Each series is oldest-first. Series may have different lengths; mature
    assets are aligned on their common tail, and the rest go through the gate.
    """
    metrics = metrics or METRICS
    now = now or datetime.now(timezone.utc)
    if anchor not in returns:
        raise ValueError(f"the anchor asset {anchor} must be tracked (§2.1)")
    for required in REQUIRED_FACTORS:
        if required not in returns:
            raise ValueError(f"{required} must always be in the matrix as a risk factor (§2.1)")

    series = {k: np.asarray(v, dtype=np.float64) for k, v in returns.items()}
    mature = sorted(k for k, v in series.items() if v.size >= min_history_hours)
    young = sorted(k for k in series if k not in mature)
    if anchor not in mature:
        raise ValueError(f"{anchor} has {series[anchor].size}h of history, below the gate")

    window = min(series[k].size for k in mature)
    x = np.column_stack([series[k][-window:] for k in mature])
    if not np.isfinite(x).all():
        raise ValueError("return history contains NaN/inf; clean the candle series first")

    w = ewma_weights(window, half_life)
    n_eff = effective_sample_size(w)
    shrunk = ledoit_wolf_constant_correlation(x, w, n_eff)
    sd = np.sqrt(np.diag(shrunk.cov))
    corr_mature = shrunk.cov / np.outer(sd, sd)
    np.fill_diagonal(corr_mature, 1.0)

    assets = tuple(mature) + tuple(young)
    n = len(assets)
    corr = np.eye(n)
    corr[: len(mature), : len(mature)] = corr_mature

    anchor_i = mature.index(anchor)
    # §2.1: the gate level is a high quantile of the mature assets' own
    # correlation with the anchor -- "assume a new listing behaves like the
    # most BTC-correlated things we already track".
    others = [i for i in range(len(mature)) if i != anchor_i]
    gate_rho = (
        float(np.quantile(corr_mature[anchor_i, others], gate_quantile)) if others else 0.9
    )
    gate_rho = float(np.clip(gate_rho, 0.0, 0.99))

    for a_pos, _name in enumerate(young, start=len(mature)):
        # Single-factor imputation through the anchor: rho(new, j) =
        # rho_gate * rho(anchor, j). Consistent by construction, and the PSD
        # projection below is the backstop.
        for j in range(len(mature)):
            rho = gate_rho if j == anchor_i else gate_rho * corr_mature[anchor_i, j]
            corr[a_pos, j] = corr[j, a_pos] = rho
    for a_pos in range(len(mature), n):
        for b_pos in range(a_pos + 1, n):
            corr[a_pos, b_pos] = corr[b_pos, a_pos] = gate_rho * gate_rho

    projected = project_to_correlation(corr, metrics=metrics, label="global")

    # Per-step volatility: EWMA for mature assets; for young ones the best
    # available estimate is their own short sample, which is noisy but is a
    # scale, not a dependency -- underestimating a *scale* is visible in
    # backtests, underestimating a correlation is not.
    vol = np.empty(n)
    vol[: len(mature)] = sd
    for a_pos, name in enumerate(young, start=len(mature)):
        v = series[name]
        if v.size < 24:
            raise ValueError(f"{name}: {v.size}h of history is too little to estimate any vol")
        vw = ewma_weights(v.size, min(half_life, max(v.size / 2.0, 2.0)))
        vol[a_pos] = float(np.sqrt(np.einsum("t,t,t->", vw, v, v)))

    metrics.incr("global_matrix_rebuilds")
    return GlobalCorrelationMatrix(
        assets=assets,
        corr=projected.corr,
        step_vol=vol,
        computed_at=now,
        diagnostics=MatrixDiagnostics(
            shrinkage_intensity=shrunk.intensity,
            mean_correlation=shrunk.mean_correlation,
            n_eff=n_eff,
            window_hours=window,
            imputed_assets=tuple(young),
            gate_correlation=gate_rho,
            psd_corrected=projected.corrected,
            psd_frobenius=projected.frobenius_correction,
            min_eigenvalue=projected.min_eigenvalue_after,
        ),
    )
