"""Funding as an AR(1) process, clamped to the protocol bound (§1.5).

Funding is charged hourly and over a 7-day horizon it erodes collateral
materially, so it is simulated on every step of every path rather than
subtracted at the end.

The clamp is mandatory. An unconstrained AR(1) fitted to a quiet month and
then run out 168 steps produces funding rates that the protocol cannot
charge, and those absurd values land in the tail of the loss distribution
where they matter most.

§1.5 forbids inventing the bound. `FundingBounds` therefore carries the
value *and* its provenance, and `validate_against_history` refuses to run if
observed history ever exceeded the configured bound — a stale or
misremembered bound then fails loudly instead of silently truncating
reality. The default value is attributed, not asserted; see
OPEN-QUESTIONS C1.

Known simplification: funding shocks are drawn independently of price
shocks. In reality funding tracks the perp-spot premium, which correlates
with recent returns, so a falling market tends to push funding negative.
The sign of the resulting bias differs between longs and shorts and is not
obviously conservative either way, so it is recorded as an open item
(OPEN-QUESTIONS A8) rather than claimed as safe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Hyperliquid's documented per-hour funding rate cap. MUST be re-verified
#: against the live API before Phase 4 (OPEN-QUESTIONS C1) -- this constant
#: exists so the value has one home, not so it can be trusted unchecked.
HL_DOCUMENTED_HOURLY_CAP = 0.04


@dataclass(frozen=True, slots=True)
class FundingBounds:
    cap_per_hour: float
    source: str

    def __post_init__(self) -> None:
        if self.cap_per_hour <= 0:
            raise ValueError("funding cap must be positive")
        if not self.source.strip():
            raise ValueError("funding bounds must carry their provenance (§1.5)")

    def validate_against_history(self, rates: np.ndarray, asset: str) -> None:
        r = np.asarray(rates, dtype=np.float64)
        worst = float(np.abs(r).max()) if r.size else 0.0
        if worst > self.cap_per_hour:
            raise ValueError(
                f"{asset}: observed funding {worst:.6f}/h exceeds the configured cap "
                f"{self.cap_per_hour:.6f}/h (source: {self.source}). The bound is stale "
                "or wrong; fix it rather than clamping reality away."
            )

    @classmethod
    def documented_default(cls) -> FundingBounds:
        return cls(
            cap_per_hour=HL_DOCUMENTED_HOURLY_CAP,
            source="Hyperliquid docs (unverified against live API; OPEN-QUESTIONS C1)",
        )


@dataclass(frozen=True, slots=True)
class Ar1Funding:
    asset: str
    mu: float
    phi: float
    sigma: float
    last_rate: float
    n_observations: int


def fit_ar1(
    asset: str,
    hourly_rates: np.ndarray,
    bounds: FundingBounds,
    max_phi: float = 0.999,
) -> Ar1Funding:
    """OLS AR(1) on the last 30 days of `fundingHistory` (§1.5)."""
    r = np.asarray(hourly_rates, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 48:
        raise ValueError(f"{asset}: need >= 48 hourly funding observations, got {r.size}")
    bounds.validate_against_history(r, asset)

    mu = float(r.mean())
    x, y = r[:-1] - mu, r[1:] - mu
    denom = float((x * x).sum())
    phi = 0.0 if denom == 0 else float((x * y).sum() / denom)
    # A unit root would let the simulated rate wander without limit; the
    # clamp would then be doing all the work.
    phi = float(np.clip(phi, -max_phi, max_phi))
    resid = y - phi * x
    sigma = float(np.sqrt((resid**2).sum() / max(resid.size - 1, 1)))
    return Ar1Funding(asset=asset, mu=mu, phi=phi, sigma=sigma,
                      last_rate=float(r[-1]), n_observations=int(r.size))


def simulate_funding(
    models: list[Ar1Funding | None],
    bounds: FundingBounds,
    n_paths: int,
    n_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """(n_paths, n_steps, n_assets) hourly funding rates.

    A `None` model means "no funding history for this column"; that column is
    zero, which understates cost. Callers must not pass None for an asset the
    book actually holds -- `tools/` enforces that.
    """
    # Deliberately a loop over assets, each walking a contiguous (n_paths,)
    # array. Vectorising across assets was tried and measured slower (24.6 ms
    # against 19.6 ms per 5k block at seven assets): the step slice
    # `x[:, s, :]` is strided on both read and write, and that costs more
    # than the interpreted loop it removes.
    out = np.zeros((n_paths, n_steps, len(models)), dtype=np.float64)
    cap = bounds.cap_per_hour
    for col, m in enumerate(models):
        if m is None:
            continue
        prev = np.full(n_paths, np.clip(m.last_rate, -cap, cap))
        shocks = rng.standard_normal((n_paths, n_steps)) * m.sigma
        for s in range(n_steps):
            prev = np.clip(m.mu + m.phi * (prev - m.mu) + shocks[:, s], -cap, cap)
            out[:, s, col] = prev
    return out
