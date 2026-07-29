"""Fixtures.

The margin tables here mirror the *shape* of Hyperliquid's `meta` response
(a tiered table keyed on notional) with plausible numbers. They are fixtures,
not a source of truth: nothing outside tests may construct an `AssetSpec`
without going through `market.meta`, which parses the live table (§1.3).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from scipy import stats

from risk_engine.domain.types import AssetSpec, MarginTier
from risk_engine.model.correlation import build_global_matrix
from risk_engine.model.funding import FundingBounds, fit_ar1
from risk_engine.model.marginals import fit_marginal
from risk_engine.sim.engine import ModelBundle


def spec(name: str, max_leverage: float, tiers: list[tuple[float, float]] | None = None) -> AssetSpec:
    rows = tiers or [(0.0, max_leverage)]
    return AssetSpec(
        name=name,
        sz_decimals=5,
        max_leverage=max_leverage,
        tiers=tuple(MarginTier(lb, lev) for lb, lev in rows),
    )


@pytest.fixture
def btc() -> AssetSpec:
    # Two tiers, so the per-step tier lookup in §1.3 is actually exercised.
    return spec("BTC", 40.0, [(0.0, 40.0), (150_000_000.0, 20.0)])


@pytest.fixture
def eth() -> AssetSpec:
    return spec("ETH", 25.0, [(0.0, 25.0), (100_000_000.0, 15.0)])


@pytest.fixture
def sol() -> AssetSpec:
    return spec("SOL", 20.0)


@pytest.fixture
def specs(btc, eth, sol) -> dict[str, AssetSpec]:
    return {"BTC": btc, "ETH": eth, "SOL": sol}


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def spot() -> dict[str, float]:
    return {"BTC": 100_000.0, "ETH": 4_000.0, "SOL": 200.0}


@pytest.fixture(scope="session")
def synthetic_returns() -> dict[str, np.ndarray]:
    """A one-factor market with fat tails, long enough to clear the data gate."""
    rng = np.random.default_rng(1234)
    n = 30 * 24 * 3
    factor = stats.t(df=4.0).rvs(n, random_state=rng) / np.sqrt(2.0)
    out = {}
    for name, load, vol in (("BTC", 1.0, 0.008), ("ETH", 0.9, 0.011), ("SOL", 0.8, 0.015)):
        idio = stats.t(df=5.0).rvs(n, random_state=rng) / np.sqrt(5 / 3)
        out[name] = (load * factor + np.sqrt(max(1 - load**2, 0.05)) * idio) * vol
    return out


@pytest.fixture(scope="session")
def bundle(synthetic_returns):
    """A fully fitted offline half of §2.6: matrix, marginals, funding."""
    rng = np.random.default_rng(99)
    matrix = build_global_matrix(
        synthetic_returns, now=datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)
    )
    marginals = {
        coin: fit_marginal(coin, r, float(matrix.step_vol[matrix.assets.index(coin)]))
        for coin, r in synthetic_returns.items()
    }
    bounds = FundingBounds.documented_default()
    funding = {}
    for coin in synthetic_returns:
        rates = 1e-5 + 2e-5 * rng.standard_normal(30 * 24)
        funding[coin] = fit_ar1(coin, rates, bounds)
    return ModelBundle(
        matrix=matrix,
        marginals=marginals,
        funding=funding,
        funding_bounds=bounds,
        copula_df=4.0,
    )
