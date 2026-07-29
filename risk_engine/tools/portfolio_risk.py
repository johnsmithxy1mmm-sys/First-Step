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
    factor_beta: float
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
        """§2.5: an under-resolved probability is not shown to anyone."""
        return self.converged


def _effective_leverage(result: RiskResult, version: str, now: datetime) -> tuple[RiskEstimate, float]:
    if result.factor_return.size == 0:
        raise ValueError("effective leverage needs the factor column; run with factor_coin set")
    equity_ret = result.raw_equity_change / result.start_equity
    factor_ret = result.factor_return
    factor_vol = float(np.std(factor_ret))
    if factor_vol <= 0:
        raise ValueError("factor volatility is zero; the ratio is undefined")

    point = float(np.std(equity_ret)) / factor_vol
    idx = np.arange(equity_ret.size)
    lo, hi = bootstrap_ci(
        idx.astype(float),
        lambda sample: float(np.std(equity_ret[sample.astype(int)]))
        / max(float(np.std(factor_ret[sample.astype(int)])), 1e-12),
        np.random.default_rng(result.provenance.seed ^ 0xBE7A),
        n_boot=200,
    )
    beta = float(np.cov(equity_ret, factor_ret)[0, 1] / np.var(factor_ret))
    return (
        RiskEstimate(point, min(lo, point), max(hi, point), version, now),
        beta,
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
    day = engine.run(book, spot, DAY_HOURS, n_paths=n_paths, seed=seed,
                     factor_coin=factor_coin, now=now)
    # The 7-day horizon reuses the seed intentionally: the two horizons then
    # share their early steps, so a user cannot see a 7d probability below
    # the 24h one purely from sampling noise.
    week = engine.run(book, spot, WEEK_HOURS, n_paths=n_paths,
                      seed=day.provenance.seed, factor_coin=factor_coin, now=now)

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
