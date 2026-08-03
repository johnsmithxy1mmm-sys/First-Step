"""The largest order this book can take and stay under a risk threshold (§4.3).

§4.3 defines the answer as the largest size whose **upper CI bound** on
P(liq) stays under the threshold. The bound, not the point estimate: a point
that sits under the line with an interval straddling it is a number the
engine does not stand behind, and §2.5 forbids publishing those either way.

Three things this module refuses to do, each because getting it wrong points
in the direction §10 forbids.

**It does not binary-search.** OPEN-QUESTIONS D2 settles the design and names
Monte Carlo noise as the reason the search needs common random numbers. That
is true and is done here -- one draw, reused for every candidate -- but it is
not the whole problem, and the part D2 does not name is not a noise problem
at all. P(liq) is NOT monotone in order size on a hedged book: an order
opposite to the book's net exposure first REDUCES exposure, passes through a
minimum near flat, and only then raises risk on the other side. The safe set
is an interval only when the order adds to the existing exposure. Bisection
assumes "safe below, unsafe above"; run across that minimum it can return a
size that is not safe, which is an understatement of risk, silently. So the
grid is scanned first, the shape is measured rather than assumed, and
bisection runs ONLY inside a bracket already known to contain a crossing --
where it is valid.

**It does not offer a size beyond the first breach**, even when a larger one
measures safe again. On a U-shaped book that can happen; a recommendation
that requires the user to understand why 5 is safe, 10 is not, and 15 is safe
again is not a recommendation.

**It does not collapse "cannot tell" into an answer.** Three outcomes are
distinct: a safe size exists, no safe size exists (the smallest tradable
increment already breaches), or the scan could not separate safe from unsafe
because the interval straddles the threshold everywhere it looked. The third
is not the second. Returning zero for either would say "trade nothing", when
the truth in the third case is "this many paths cannot tell you".

Cost. Every candidate is a full book walk, but they all ride ONE set of price
paths -- `run_blocks` takes a sequence of books precisely so that several can
be walked over shared randomness, which is also what makes the comparison
across candidates pathwise instead of a race between independent samples.
Path generation therefore happens once per query, not once per candidate.
Even so this is far more work than §4.2's interactive path: it is not on
§2.6's 300 ms clock and must not be put there.

Gate. This computes a RECOMMENDATION, and §3.3 gates recommendations on
shadow validation exactly as it gates execution. The function exists, is
tested, and is deliberately not wired to any control a user acts on until the
validation window clears -- the same posture as the inert execute button.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from risk_engine.domain.types import AssetSpec, Book, MarginMode, RiskEstimate
from risk_engine.observability.metrics import METRICS, Metrics
from risk_engine.sim.engine import ModelBundle, _any_liq, run_blocks
from risk_engine.sim.stats import wilson_interval
from risk_engine.tools.pre_trade_delta import ProposedOrder

#: §4.3's default ceiling on P(liq) over the horizon.
DEFAULT_THRESHOLD = 0.05
DEFAULT_HORIZON_HOURS = 24
DEFAULT_PATHS = 20_000
#: Candidate sizes evaluated before any refinement. Coarse on purpose: the
#: grid exists to reveal the SHAPE of P(liq) against size, and a bracket for
#: bisection to work inside, not to be the answer.
DEFAULT_GRID = 12
#: Bisection steps inside the bracket. Each halves the remaining interval, so
#: 6 puts the answer within 1/64 of one grid cell -- well below the size
#: increment on any asset this would be used for.
DEFAULT_REFINE = 6


@dataclass(frozen=True, slots=True)
class SizeVerdict:
    """What the scan found at one candidate size."""

    size: float
    p_liq: RiskEstimate
    #: The §4.3 test: the UPPER bound, not the point, against the threshold.
    breaches: bool


@dataclass(frozen=True, slots=True)
class MaxSafeSize:
    """The answer, in a shape that can say "there isn't one".

    `size` is None for both the no-safe-size and the unresolved outcomes;
    `outcome` is what distinguishes them, and callers must branch on it rather
    than on `size is None`.
    """

    outcome: str  # "safe" | "none" | "unresolved"
    size: float | None
    p_liq_at_size: RiskEstimate | None
    threshold: float
    horizon_hours: int
    n_paths: int
    scanned: tuple[SizeVerdict, ...]
    #: True when nothing on the scanned range breached, so the answer is the
    #: top of what was LOOKED AT rather than where risk actually bites. The
    #: size is still safe -- this never overstates what is allowed -- but a
    #: caller presenting it as "your maximum" would be putting a scan
    #: artefact in front of a user.
    scan_bounded: bool
    reason: str
    model_version: str
    computed_at: datetime

    @property
    def has_answer(self) -> bool:
        return self.outcome == "safe"

    def summary(self) -> str:
        if self.outcome == "safe":
            assert self.size is not None and self.p_liq_at_size is not None
            tail = (
                " -- and nothing up to there breached, so this is the top of the "
                "scanned range, not a risk limit"
                if self.scan_bounded else ""
            )
            return (
                f"up to {self.size:g} keeps P(liq) within {self.threshold:.1%} over "
                f"{self.horizon_hours}h (upper bound "
                f"{self.p_liq_at_size.ci_high:.2%} at that size){tail}"
            )
        if self.outcome == "none":
            return f"no safe size: {self.reason}"
        return f"cannot tell at {self.n_paths} paths: {self.reason}"


def _rounded_down(size: float, increment: float) -> float:
    """Snap a size DOWN to the tradable increment.

    Down, never to nearest: rounding up offers a size whose risk was never
    evaluated and which may sit the wrong side of the threshold. `floor` on
    the quotient rather than `round`, and a nudge for the float error that
    makes e.g. 0.3/0.1 land at 2.9999999999999996.
    """
    if increment <= 0:
        return size
    q = size / increment
    n = math.floor(q + 1e-9)
    return max(0.0, n * increment)


def max_safe_size(
    book: Book,
    coin: str,
    spot: dict[str, float],
    bundle: ModelBundle,
    specs: dict[str, AssetSpec],
    *,
    direction: int = 1,
    leverage: float = 1.0,
    mode: MarginMode = MarginMode.CROSS,
    threshold: float = DEFAULT_THRESHOLD,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
    n_paths: int = DEFAULT_PATHS,
    max_size: float | None = None,
    grid: int = DEFAULT_GRID,
    refine: int = DEFAULT_REFINE,
    seed: int | None = None,
    now: datetime | None = None,
    metrics: Metrics | None = None,
    workers: int | None = None,
) -> MaxSafeSize:
    """Largest `direction`-signed size in `coin` whose P(liq) upper bound holds.

    `direction` is +1 for a buy and -1 for a sell; the sign matters because
    the answer is not symmetric on a book that already leans one way.
    """
    metrics = metrics or METRICS
    now = now or datetime.now(timezone.utc)
    if direction not in (1, -1):
        raise ValueError("direction must be +1 (buy) or -1 (sell)")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be a probability strictly inside (0, 1)")
    if coin not in specs:
        raise KeyError(f"no margin table for {coin} -- fetch `meta` first")
    if coin not in spot:
        raise KeyError(f"no spot price for {coin}")

    spec_asset = specs[coin]
    increment = spec_asset.size_increment
    price = float(spot[coin])

    if max_size is None:
        # Fresh capacity at the requested leverage, PLUS whatever it would
        # take to flatten an existing position on the other side.
        #
        # The second term is not padding. An order that offsets an open
        # position does not consume margin the way a new one does -- it
        # releases it -- so a ceiling of equity*leverage/price silently
        # truncates the answer on exactly the hedged books this tool is most
        # useful for. Measured: a book short 6 BTC on $60k equity scans safe
        # to 12 BTC long, while that ceiling stopped the scan at 6 and
        # reported 6 as the answer. Under-reporting the size is the safe
        # direction, but it is still a wrong number.
        equity = book.equity(spot)
        opposing = sum(
            abs(p.size) for p in book.positions
            if p.coin == coin and (p.size > 0) != (direction > 0)
        )
        max_size = max(increment, abs(equity) * leverage / price + opposing)

    # One candidate per grid point, plus the smallest tradable size, which is
    # the one that decides between "no safe size" and a real answer.
    raw = [increment] + [
        increment + (max_size - increment) * (i + 1) / grid for i in range(grid)
    ]
    sizes = _dedup_sizes(raw, increment)

    seed = seed if seed is not None else 0xB0FA
    verdicts = _evaluate(
        book, coin, sizes, direction, leverage, mode, spot, bundle, specs,
        horizon_hours, n_paths, threshold, seed, now, workers,
    )

    smallest = verdicts[0]
    if smallest.breaches:
        # Distinguish "already over" from "cannot tell at the smallest size":
        # a point estimate below the threshold with an interval above it is
        # the unresolved case, not a refusal.
        if smallest.p_liq.point >= threshold:
            metrics.incr("max_safe_size_none")
            return _result(
                "none", None, None, threshold, horizon_hours, n_paths, verdicts,
                f"the smallest tradable size ({increment:g}) already puts P(liq) at "
                f"{smallest.p_liq.point:.2%}, over the {threshold:.1%} threshold",
                bundle.model_version, now,
            )
        metrics.incr("max_safe_size_unresolved")
        return _result(
            "unresolved", None, None, threshold, horizon_hours, n_paths, verdicts,
            f"at the smallest tradable size the estimate is {smallest.p_liq.point:.2%} "
            f"with an interval to {smallest.p_liq.ci_high:.2%} that spans the "
            f"{threshold:.1%} threshold; more paths are needed to separate them",
            bundle.model_version, now,
        )

    # The CONTIGUOUS safe region starting at the smallest size. Anything past
    # the first breach is not offered even if it measures safe again.
    last_safe_i = 0
    for i, v in enumerate(verdicts):
        if v.breaches:
            break
        last_safe_i = i
    first_bad_i = last_safe_i + 1

    if first_bad_i >= len(verdicts):
        # Nothing on the grid breached. Report the top of the scanned range
        # rather than extrapolating past it -- the answer is bounded by what
        # was actually measured.
        top = verdicts[last_safe_i]
        metrics.incr("max_safe_size_at_scan_ceiling")
        return _result(
            "safe", top.size, top.p_liq, threshold, horizon_hours, n_paths, verdicts,
            f"no size up to the scanned ceiling {top.size:g} breaches; the answer is "
            "bounded by the scan, not by risk",
            bundle.model_version, now, scan_bounded=True,
        )

    # Bisect inside [safe, unsafe]. Valid here and only here: the bracket is
    # known to contain a crossing, so the objective is monotone across it in
    # the only sense bisection needs.
    lo, hi = verdicts[last_safe_i].size, verdicts[first_bad_i].size
    best = verdicts[last_safe_i]
    for _ in range(refine):
        mid = _rounded_down((lo + hi) / 2.0, increment)
        if mid <= lo or mid >= hi:
            break
        v = _evaluate(
            book, coin, [mid], direction, leverage, mode, spot, bundle, specs,
            horizon_hours, n_paths, threshold, seed, now, workers,
        )[0]
        verdicts = tuple(sorted((*verdicts, v), key=lambda x: x.size))
        if v.breaches:
            hi = mid
        else:
            lo, best = mid, v

    answer = _rounded_down(best.size, increment)
    if answer < increment:
        metrics.incr("max_safe_size_none")
        return _result(
            "none", None, None, threshold, horizon_hours, n_paths, verdicts,
            "the largest safe size rounds below the tradable increment",
            bundle.model_version, now,
        )
    metrics.incr("max_safe_size_answered")
    return _result(
        "safe", answer, best.p_liq, threshold, horizon_hours, n_paths, verdicts,
        f"largest scanned size whose P(liq) upper bound stays under {threshold:.1%}",
        bundle.model_version, now,
    )


def _dedup_sizes(raw: list[float], increment: float) -> list[float]:
    """Snap candidates to the increment and drop duplicates, ascending."""
    seen: dict[float, None] = {}
    for s in raw:
        snapped = _rounded_down(s, increment)
        if snapped >= increment:
            seen[snapped] = None
    return sorted(seen)


def _evaluate(
    book: Book, coin: str, sizes: list[float], direction: int, leverage: float,
    mode: MarginMode, spot: dict[str, float], bundle: ModelBundle,
    specs: dict[str, AssetSpec], horizon_hours: int, n_paths: int,
    threshold: float, seed: int, now: datetime, workers: int | None,
) -> tuple[SizeVerdict, ...]:
    """Walk every candidate book over ONE set of shared price paths.

    Shared paths are what make the candidates comparable to each other rather
    than to their own sampling noise, and they are why the whole scan costs
    one path generation instead of one per size.
    """
    price = float(spot[coin])
    books = []
    for s in sizes:
        order = ProposedOrder(
            coin=coin, size=direction * s, leverage=leverage, mode=mode
        )
        books.append(book.with_position(order.to_position(price)))

    coins = tuple(dict.fromkeys([c for b in books for c in b.coins]))
    missing = [c for c in coins if c not in spot]
    if missing:
        raise KeyError(f"no spot price for {missing}")
    spec = bundle.path_spec(coins)
    spot_vec = np.array([spot[c] for c in coins], dtype=np.float64)

    kwargs = dict(
        books=books, specs=specs, bundle=bundle, spec=spec, spot_vec=spot_vec,
        n_paths=n_paths, horizons=(horizon_hours,), seed=seed,
        n_iso=len(coins), include_funding=False, use_bridge=True,
    )
    if workers is not None:
        kwargs["workers"] = workers
    blocks = run_blocks(**kwargs)

    out = []
    for s, per_horizon in zip(sizes, blocks, strict=True):
        flags = _any_liq(per_horizon[horizon_hours])
        k, n = int(flags.sum()), flags.size
        lo, hi = wilson_interval(k, n)
        point = k / n
        est = RiskEstimate(point, min(lo, point), max(hi, point),
                           bundle.model_version, now)
        # >= not >: a bound exactly on the threshold is a breach. Overstating
        # risk is permitted, understating is not (§10).
        out.append(SizeVerdict(size=s, p_liq=est, breaches=est.ci_high >= threshold))
    return tuple(out)


def _result(outcome, size, est, threshold, horizon, n_paths, verdicts, reason,
            version, now, scan_bounded=False) -> MaxSafeSize:
    return MaxSafeSize(
        outcome=outcome, size=size, p_liq_at_size=est, threshold=threshold,
        horizon_hours=horizon, n_paths=n_paths,
        scanned=tuple(sorted(verdicts, key=lambda v: v.size)),
        scan_bounded=scan_bounded, reason=reason,
        model_version=version, computed_at=now,
    )
