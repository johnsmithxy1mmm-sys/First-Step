"""§4.1 `portfolio_risk` — the read-only view of one account.

Every number carries an interval, because the type it is returned in cannot
be built without one.

The "effective leverage" figure follows §4.1's definition -- the ratio of the
book's 24 h equity-return volatility to BTC's -- and the beta is returned
alongside it, because the ratio alone does not support the UI sentence §4.1
proposes. A volatility ratio is unsigned and direction-free: a delta-neutral
book with large idiosyncratic variance scores the same as an outright long,
and a short book scores the same as a long one. Whatever the UI says, it
should not claim a direction that this number does not carry
(OPEN-QUESTIONS D3).

The beta therefore carries an interval, like every other number here. It is
the figure the UI leads with when it wants to say which way the book leans,
and a directional claim read off a point estimate is the same mistake as
any other point estimate in this engine -- worse, because "your book is long
BTC" is a sentence a user acts on. `direction_detectable` reads the sign off
the *interval*: a book whose beta interval spans zero has no direction this
model can resolve, and saying so is the honest output.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from risk_engine.domain.types import AssetSpec, Book, RiskEstimate
from risk_engine.sim.engine import ModelBundle, MonteCarloEngine, RiskResult
from risk_engine.sim.stats import bootstrap_ci

DAY_HOURS = 24
WEEK_HOURS = 24 * 7


@dataclass(frozen=True, slots=True)
class PortfolioRisk:
    address: str
    effective_leverage: RiskEstimate
    factor_beta: RiskEstimate
    factor_coin: str
    p_liq_24h_any: RiskEstimate
    p_liq_24h_cross: RiskEstimate
    p_liq_24h_isolated: dict[str, RiskEstimate]
    p_liq_7d_any: RiskEstimate
    p_liq_7d_cross: RiskEstimate
    p_liq_7d_isolated: dict[str, RiskEstimate]
    cvar_95_24h_usd: RiskEstimate
    start_equity: float
    computed_at: datetime
    converged: bool
    result_24h: RiskResult

    @property
    def publishable(self) -> bool:
        """§2.5's flag. NOT a refusal — this object still carries every point
        estimate when it is False, and the service serves it at HTTP 200.

        What honours it is `services/backend/src/risk/client.ts`, which
        withholds the value the same way it withholds a stale one. A Python
        caller that reads `.p_liq_24h_any` without checking this gets an
        under-resolved number with no complaint.
        """
        return self.converged

    @property
    def direction_detectable(self) -> bool:
        """Whether the book leans a way this model can actually resolve.

        Read off the interval, not the point. A beta of +0.3 with an interval
        of [-1.4, +2.0] is not a long book; it is a book whose direction the
        sample does not determine, and "you are long BTC" is exactly the kind
        of sentence a user would act on.
        """
        return self.factor_beta.ci_low > 0.0 or self.factor_beta.ci_high < 0.0


def _resample_ratio(equity: np.ndarray, factor: np.ndarray, sample: np.ndarray) -> float:
    i = sample.astype(int)
    return float(np.std(equity[i])) / max(float(np.std(factor[i])), 1e-12)


def _resample_beta(equity: np.ndarray, factor: np.ndarray, sample: np.ndarray) -> float:
    i = sample.astype(int)
    var = float(np.var(factor[i]))
    if var <= 0.0:
        return 0.0
    return float(np.cov(equity[i], factor[i])[0, 1] / var)


def _effective_leverage(
    result: RiskResult, version: str, now: datetime
) -> tuple[RiskEstimate, RiskEstimate]:
    if result.factor_return.size == 0:
        raise ValueError("effective leverage needs the factor column; run with factor_coin set")
    equity_ret = result.raw_equity_change / result.start_equity
    factor_ret = result.factor_return
    factor_vol = float(np.std(factor_ret))
    if factor_vol <= 0:
        raise ValueError("factor volatility is zero; the ratio is undefined")

    ratio = float(np.std(equity_ret)) / factor_vol
    beta = float(np.cov(equity_ret, factor_ret)[0, 1] / np.var(factor_ret))

    # Both intervals come off the same resample -- same seed, so the same
    # draws -- and therefore describe the same resampled book rather than two
    # independently jittered ones. A user comparing the ratio against the beta
    # is entitled to have them refer to one thing.
    idx = np.arange(equity_ret.size, dtype=float)
    seed = result.provenance.seed ^ 0xBE7A
    lo, hi = bootstrap_ci(
        idx, lambda s: _resample_ratio(equity_ret, factor_ret, s),
        np.random.default_rng(seed), n_boot=200,
    )
    blo, bhi = bootstrap_ci(
        idx, lambda s: _resample_beta(equity_ret, factor_ret, s),
        np.random.default_rng(seed), n_boot=200,
    )
    return (
        RiskEstimate(ratio, min(lo, ratio), max(hi, ratio), version, now),
        RiskEstimate(beta, min(blo, beta), max(bhi, beta), version, now),
    )


def portfolio_risk(
    book: Book,
    spot: dict[str, float],
    bundle: ModelBundle,
    specs: dict[str, AssetSpec],
    n_paths: int = 20_000,
    seed: int | None = None,
    factor_coin: str = "BTC",
    now: datetime | None = None,
) -> PortfolioRisk:
    engine = MonteCarloEngine(bundle, specs)
    # One walk, two checkpoints. Reusing a seed across two separate runs does
    # NOT share the paths -- the (paths, steps, assets) draw shape differs, so
    # the streams diverge after the first path, and a user could be shown a 7d
    # probability below the 24h one (audit A-10). Checkpointing one walk makes
    # the ordering pathwise.
    results = engine.run_horizons(
        book, spot, (DAY_HOURS, WEEK_HOURS), n_paths=n_paths, seed=seed,
        factor_coin=factor_coin, now=now,
    )
    day, week = results[DAY_HOURS], results[WEEK_HOURS]

    version = bundle.model_version
    stamp = day.provenance.computed_at
    lev, beta = _effective_leverage(day, version, stamp)
    return PortfolioRisk(
        address=book.address,
        effective_leverage=lev,
        factor_beta=beta,
        factor_coin=factor_coin,
        p_liq_24h_any=day.p_liq_any,
        p_liq_24h_cross=day.p_liq_cross,
        p_liq_24h_isolated=day.p_liq_isolated,
        p_liq_7d_any=week.p_liq_any,
        p_liq_7d_cross=week.p_liq_cross,
        p_liq_7d_isolated=week.p_liq_isolated,
        cvar_95_24h_usd=day.cvar_95_usd,
        start_equity=day.start_equity,
        computed_at=stamp,
        converged=day.converged and week.converged,
        result_24h=day,
    )
