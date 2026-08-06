"""The two baselines of §3.2.

§3.2 defines them as probability predictors, but §0.2 and §3.3 score them by
CRPS, which needs a full predictive distribution of 24 h equity change. Both
are therefore implemented as distribution predictors, and the probability
falls out of the same draws.

Baseline B is the one that decides. If the full model does not beat an
identically-specified model with the correlations switched off, then the
copula, the shrinkage, the projection and the global matrix are not paying
for their complexity, and §3.2 says to report that and simplify rather than
build more on top. `run_baseline_b` is deliberately the *same* engine with
`independent=True`, so nothing but the dependence structure differs.

Baseline A collapses the book to one directional exposure driven by a
bootstrapped historical BTC return, with no per-asset volatility, no
correlation and no funding. It reuses the real liquidation model to turn
that draw into an outcome, so the comparison isolates the return model
rather than confounding it with margin arithmetic. §3.2's own wording
("historical unconditional frequency at that nominal leverage") does not
define a distribution; this reading is recorded in OPEN-QUESTIONS B3.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from risk_engine.domain.types import AssetSpec, Book
from risk_engine.liquidation.simulator import LiquidationSimulator
from risk_engine.sim.engine import ModelBundle, MonteCarloEngine, RiskResult
from risk_engine.sim.stats import PredictiveDistribution, wilson_interval


@dataclass(frozen=True, slots=True)
class BaselinePrediction:
    name: str
    equity_change: PredictiveDistribution
    p_liq: float
    p_liq_ci: tuple[float, float]
    start_equity: float
    n_draws: int
    seed: int
    computed_at: datetime


class NaiveBaseline:
    """Baseline A. One market factor, bootstrapped from history, no funding."""

    name = "baseline_a_naive"

    def __init__(self, factor_24h_log_returns: np.ndarray) -> None:
        r = np.asarray(factor_24h_log_returns, dtype=np.float64)
        r = r[np.isfinite(r)]
        if r.size < 100:
            raise ValueError(f"need >= 100 historical 24h returns, got {r.size}")
        self.history = r

    def predict(
        self,
        book: Book,
        spot: dict[str, float],
        specs: dict[str, AssetSpec],
        n_draws: int = 20_000,
        seed: int = 0,
        now: datetime | None = None,
    ) -> BaselinePrediction:
        now = now or datetime.now(timezone.utc)
        rng = np.random.default_rng(seed)
        coins = book.coins
        columns = {c: i for i, c in enumerate(coins)}
        draws = rng.choice(self.history, size=n_draws, replace=True)

        # Every asset takes the same factor move: this baseline has no notion
        # of a second source of risk. One step, evaluated at the horizon end
        # only -- no path, no intra-horizon monitoring.
        mult = np.exp(draws)[:, None]
        prices = np.empty((n_draws, 2, len(coins)))
        for c, i in columns.items():
            prices[:, 0, i] = spot[c]
            prices[:, 1, i] = spot[c] * mult[:, 0]

        out = LiquidationSimulator(book, specs, columns).run(prices)
        # The property, not a second copy of its body. Baseline A's p_liq is
        # compared against the model's, and the model's comes from
        # `engine.py::_any_liq`; two hand-written definitions of "liquidated"
        # on opposite sides of that comparison can drift into scoring different
        # events while both still look like probabilities.
        k = int(out.any_liquidated.sum())
        return BaselinePrediction(
            name=self.name,
            equity_change=PredictiveDistribution.from_samples(out.equity_change),
            p_liq=k / n_draws,
            p_liq_ci=wilson_interval(k, n_draws),
            start_equity=out.start_equity,
            n_draws=n_draws,
            seed=seed,
            computed_at=now,
        )


def run_baseline_b(
    bundle: ModelBundle,
    specs: dict[str, AssetSpec],
    book: Book,
    spot: dict[str, float],
    horizon_hours: int = 24,
    n_paths: int = 20_000,
    seed: int = 0,
    now: datetime | None = None,
) -> RiskResult:
    """Baseline B: identical marginals and liquidation model, zero correlation."""
    return MonteCarloEngine(bundle, specs).run(
        book, spot, horizon_hours, n_paths=n_paths, seed=seed, independent=True, now=now
    )


def historical_24h_log_returns(hourly_log_returns: np.ndarray) -> np.ndarray:
    """Overlapping 24 h sums of hourly log returns.

    Overlapping windows are autocorrelated by construction. That is fine for
    a bootstrap of the marginal distribution, which is all this baseline
    claims to be, and is noted so nobody later reads a confidence interval
    off these as if they were independent.
    """
    r = np.asarray(hourly_log_returns, dtype=np.float64)
    if r.size < 25:
        raise ValueError("need more than one day of hourly returns")
    csum = np.concatenate([[0.0], np.cumsum(r)])
    return csum[24:] - csum[:-24]
