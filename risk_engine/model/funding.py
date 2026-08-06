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

#: Hyperliquid's documented per-hour funding rate cap.
HL_DOCUMENTED_HOURLY_CAP = 0.04

#: Where that number was read, and when. C1 asked for a citation someone can
#: re-check rather than a value carried on trust; this is it.
HL_FUNDING_DOC_URL = "https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding"
HL_FUNDING_DOC_QUOTE = "Funding on Hyperliquid is capped at 4%/hour"
HL_FUNDING_DOC_READ_ON = "2026-08-03"

#: The trap C1 named in advance, recorded here because the page contains both
#: numbers and taking the wrong one is a 100x error toward understating cost
#: of carry (§10-forbidden). The published formula is
#:
#:     F = P + clamp(interest_rate - P, -0.0005, 0.0005)
#:
#: so ±0.0005 bounds the interest-rate TERM INSIDE the formula. It is not the
#: cap on the realised rate, which the same page states separately as 4%/hour.
#: `cap_per_hour` clips simulated funding paths and is therefore the latter.
HL_INTEREST_TERM_CLAMP = 0.0005


@dataclass(frozen=True, slots=True)
class FundingBounds:
    cap_per_hour: float
    source: str
    #: Whether `source` names a protocol reference someone actually read, as
    #: opposed to a value carried forward on trust.
    #:
    #: A separate field rather than a convention about the wording of `source`,
    #: because prose cannot be checked and this is the thing C1 turns on. It is
    #: also why `documented_default` cannot set it: a default that arrives
    #: pre-confirmed is a default nobody ever confirms.
    confirmed: bool = False

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
        """The value carried on trust. Kept, and no longer what ships.

        `hyperliquid_confirmed()` below replaced it at every shipped call site
        on 2026-08-03. This stays because it is the honest constructor for a
        number nobody has read, and the next bound added here will start life
        that way — deleting it would leave `from_protocol_source` as the only
        route and quietly invite a citation to be invented for it.
        """
        return cls(
            cap_per_hour=HL_DOCUMENTED_HOURLY_CAP,
            source="Hyperliquid docs (unverified against live API; OPEN-QUESTIONS C1)",
        )

    @classmethod
    def hyperliquid_confirmed(cls) -> FundingBounds:
        """The shipped bound, with the citation C1 spent its life asking for.

        Read from the primary source on 2026-08-03. Two things about the page
        are worth carrying here rather than only in OPEN-QUESTIONS, because
        both are ways to get this wrong later:

        **The page states two clamps and only one of them is this.** See
        `HL_INTEREST_TERM_CLAMP` — ±0.0005 bounds a term inside the formula,
        4%/hour bounds the realised rate. Recording the former here would clip
        simulated funding at 1/80th of the true bound, understating cost of
        carry, which §10 forbids.

        **The 8-hour/1-hour split does not put an 8x in this number.** The
        formula computes an 8-hour rate paid hourly at one eighth, and the cap
        is stated separately and explicitly per hour ("capped at 4%/hour"),
        which is the same basis `fundingHistory` reports and the same basis
        this bound clips. Had the reading been wrong in the other direction —
        the cap applying to the 8-hour rate, making the true hourly bound
        0.5% — this value would be 8x too permissive, which lets the model
        simulate funding more extreme than the protocol allows and OVERSTATES
        cost of carry. §10 permits that direction, so even the residual
        ambiguity fails safe.
        """
        return cls.from_protocol_source(
            HL_DOCUMENTED_HOURLY_CAP,
            f"{HL_FUNDING_DOC_URL} — {HL_FUNDING_DOC_QUOTE!r} "
            f"(read {HL_FUNDING_DOC_READ_ON})",
        )

    @classmethod
    def from_protocol_source(cls, cap_per_hour: float, source: str) -> FundingBounds:
        """A cap someone read out of the protocol's own documentation or code.

        This is the only way to reach `confirmed=True`, and it exists because
        C1 could not be closed at all before it. The check told an operator to
        "confirm the value from protocol documentation and record it as the
        `source` field" — and recording it changed nothing, because the check
        built its own `documented_default()` and returned INCONCLUSIVE
        whenever no observation breached the cap. The instruction was
        unactionable: following it exactly produced identical output. Same
        defect class as a counter that cannot fire and a diagnostic nothing
        calls.

        `source` must be specific enough to re-check: a URL, a doc section, a
        file and line in the protocol's source. "Hyperliquid docs" is not, and
        is refused, because the whole value of a confirmed bound is that the
        next person can confirm it again rather than inherit the belief.

        Note what confirmation does NOT do: it does not make the bound true.
        `validate_against_history` still refuses a rate that exceeds it, and a
        confirmed-but-wrong bound fails louder than an unconfirmed one, which
        is the correct ordering.
        """
        if len(source.strip()) < 12 or not any(ch.isdigit() for ch in source):
            raise ValueError(
                "a confirmed funding bound must cite something re-checkable — a "
                "URL, a dated doc section, or a file and line in the protocol's "
                "source. Got: " + repr(source) + ". An unspecific citation is how "
                "a value nobody verified becomes a value everybody trusts (§1.5, "
                "OPEN-QUESTIONS C1)."
            )
        return cls(cap_per_hour=cap_per_hour, source=source.strip(), confirmed=True)


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
