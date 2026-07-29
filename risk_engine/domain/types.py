"""Value types shared by every layer of the engine.

Two of these carry design intent rather than just data:

`RiskEstimate` cannot be constructed without an interval. §4 requires that
returning a point estimate alone be *impossible*, not merely discouraged, so
the interval is positional and validated in `__post_init__`.

`AssetSpec` owns the margin-tier table (§1.3). The table comes from the
`meta` endpoint; nothing in this package hardcodes a maintenance margin rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import pairwise
from enum import Enum

import numpy as np


class MarginMode(str, Enum):
    CROSS = "cross"
    ISOLATED = "isolated"


@dataclass(frozen=True, slots=True)
class MarginTier:
    """One row of an asset's margin table.

    `lower_bound` is the position notional in USD at which this tier starts.
    Maintenance margin is half the initial margin at the tier's max leverage
    (§1.1), so mmr = 0.5 / max_leverage.
    """

    lower_bound: float
    max_leverage: float

    def __post_init__(self) -> None:
        if self.lower_bound < 0:
            raise ValueError(f"tier lower_bound must be >= 0, got {self.lower_bound}")
        if self.max_leverage <= 1:
            raise ValueError(f"tier max_leverage must be > 1, got {self.max_leverage}")

    @property
    def maintenance_margin_rate(self) -> float:
        return 0.5 / self.max_leverage


@dataclass(frozen=True, slots=True)
class AssetSpec:
    """Per-asset contract parameters, sourced from `meta` (§1.3, §5.1)."""

    name: str
    sz_decimals: int
    max_leverage: float
    tiers: tuple[MarginTier, ...]
    # Cached tier arrays for the vectorised lookup in the simulator.
    _bounds: np.ndarray = field(init=False, repr=False, compare=False)
    _mmrs: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.tiers:
            raise ValueError(f"{self.name}: margin table is empty; it must come from `meta`")
        bounds = [t.lower_bound for t in self.tiers]
        if bounds[0] != 0.0:
            raise ValueError(f"{self.name}: first margin tier must start at notional 0")
        if any(b <= a for a, b in pairwise(bounds)):
            raise ValueError(f"{self.name}: margin tiers must be strictly ascending: {bounds}")
        levs = [t.max_leverage for t in self.tiers]
        if any(b > a for a, b in pairwise(levs)):
            # Larger positions never get *more* leverage; if they appear to,
            # we have misread the table rather than found a generous venue.
            raise ValueError(f"{self.name}: max leverage must not increase with size: {levs}")
        object.__setattr__(self, "_bounds", np.asarray(bounds, dtype=np.float64))
        object.__setattr__(
            self, "_mmrs", np.asarray([t.maintenance_margin_rate for t in self.tiers], np.float64)
        )

    def maintenance_margin_rate(self, notional: float | np.ndarray) -> float | np.ndarray:
        """mmr for a position of this notional (absolute USD value).

        §1.3: the tier is chosen by the position's value *at the moment of
        evaluation*, so this is called on every step of every path rather
        than once at the start.
        """
        idx = np.searchsorted(self._bounds, np.abs(notional), side="right") - 1
        idx = np.maximum(idx, 0)
        out = self._mmrs[idx]
        return float(out) if np.isscalar(notional) or np.ndim(notional) == 0 else out

    @property
    def size_increment(self) -> float:
        return 10.0**-self.sz_decimals


@dataclass(frozen=True, slots=True)
class Position:
    """One open position.

    `size` is signed: positive long, negative short. `isolated_margin` is the
    collateral allocated to the position and is required for, and only for,
    isolated mode -- it is what makes an isolated liquidation independent of
    the cross pool (§1.1).
    """

    coin: str
    size: float
    entry_price: float
    mode: MarginMode
    leverage: float
    isolated_margin: float | None = None

    def __post_init__(self) -> None:
        if self.size == 0:
            raise ValueError(f"{self.coin}: zero-size position is not a position")
        if self.entry_price <= 0:
            raise ValueError(f"{self.coin}: entry price must be positive")
        if self.leverage <= 0:
            raise ValueError(f"{self.coin}: leverage must be positive")
        if self.mode is MarginMode.ISOLATED:
            if self.isolated_margin is None or self.isolated_margin <= 0:
                raise ValueError(f"{self.coin}: isolated position needs positive isolated_margin")
        elif self.isolated_margin is not None:
            raise ValueError(f"{self.coin}: cross position must not carry isolated_margin")

    @property
    def side(self) -> int:
        return 1 if self.size > 0 else -1

    def notional(self, price: float) -> float:
        return abs(self.size) * price

    def unrealised_pnl(self, price: float) -> float:
        return self.size * (price - self.entry_price)


@dataclass(frozen=True, slots=True)
class Book:
    """A snapshot of one account.

    `cross_collateral` is the USDC backing the cross pool *excluding*
    unrealised PnL -- i.e. `crossMarginSummary.accountValue` minus the
    unrealised PnL of the cross positions. Keeping the cash and the PnL
    separate is what lets the simulator move one and recompute the other.
    """

    address: str
    cross_collateral: float
    positions: tuple[Position, ...]
    captured_at: datetime

    def __post_init__(self) -> None:
        if self.captured_at.tzinfo is None:
            raise ValueError("captured_at must be timezone-aware")
        coins = [p.coin for p in self.positions]
        if len(coins) != len(set(coins)):
            raise ValueError(f"one position per coin (one-way mode); got {coins}")

    @property
    def cross_positions(self) -> tuple[Position, ...]:
        return tuple(p for p in self.positions if p.mode is MarginMode.CROSS)

    @property
    def isolated_positions(self) -> tuple[Position, ...]:
        return tuple(p for p in self.positions if p.mode is MarginMode.ISOLATED)

    @property
    def coins(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(p.coin for p in self.positions))

    def cross_account_value(self, prices: dict[str, float]) -> float:
        return self.cross_collateral + sum(
            p.unrealised_pnl(prices[p.coin]) for p in self.cross_positions
        )

    def isolated_equity(self, position: Position, prices: dict[str, float]) -> float:
        assert position.isolated_margin is not None
        return position.isolated_margin + position.unrealised_pnl(prices[position.coin])

    def equity(self, prices: dict[str, float]) -> float:
        """Total account equity: cross pool plus every isolated pocket."""
        return self.cross_account_value(prices) + sum(
            self.isolated_equity(p, prices) for p in self.isolated_positions
        )

    def with_position(self, position: Position) -> Book:
        """Return a copy with `position` added, or merged into an existing one.

        Used by `pre_trade_delta` (§4.2) and `max_safe_size` (§4.3).
        """
        existing = {p.coin: p for p in self.positions}
        prior = existing.get(position.coin)
        if prior is None:
            merged = position
        else:
            if prior.mode is not position.mode:
                raise ValueError(
                    f"{position.coin}: cannot mix {prior.mode.value} and {position.mode.value}"
                )
            size = prior.size + position.size
            if size == 0:
                remaining = tuple(p for p in self.positions if p.coin != position.coin)
                return Book(self.address, self.cross_collateral, remaining, self.captured_at)
            # Volume-weighted entry when adding in the same direction; when
            # reducing, the entry price of the remainder is unchanged.
            if prior.side == position.side:
                entry = (
                    prior.size * prior.entry_price + position.size * position.entry_price
                ) / size
            else:
                entry = prior.entry_price
            iso = None
            if position.mode is MarginMode.ISOLATED:
                iso = (prior.isolated_margin or 0.0) + (position.isolated_margin or 0.0)
            merged = Position(position.coin, size, entry, position.mode, position.leverage, iso)
        others = tuple(p for p in self.positions if p.coin != position.coin)
        return Book(self.address, self.cross_collateral, (*others, merged), self.captured_at)


@dataclass(frozen=True, slots=True)
class SimulationProvenance:
    """Everything needed to reproduce a number (§2.5: the seed is always logged)."""

    seed: int
    n_paths: int
    horizon_hours: int
    model_version: str
    computed_at: datetime
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.computed_at.tzinfo is None:
            raise ValueError("computed_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class RiskEstimate:
    """A number with its uncertainty, its model version and its age.

    §4: the type makes an interval-free answer unconstructible. §6: the type
    makes an age-free display unconstructible, because `computed_at` is
    mandatory and `age_seconds` is the only way to read freshness.

    §4.2: `overlaps` is how the UI is told two estimates are statistically
    indistinguishable. Drawing an arrow between overlapping intervals is
    false precision, which in a risk tool is worse than no tool.
    """

    point: float
    ci_low: float
    ci_high: float
    model_version: str
    computed_at: datetime

    def __post_init__(self) -> None:
        if self.computed_at.tzinfo is None:
            raise ValueError("computed_at must be timezone-aware")
        if not (self.ci_low <= self.point <= self.ci_high):
            raise ValueError(
                f"point {self.point} outside interval [{self.ci_low}, {self.ci_high}]"
            )

    @property
    def half_width(self) -> float:
        return 0.5 * (self.ci_high - self.ci_low)

    def age_seconds(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        return (now - self.computed_at).total_seconds()

    def overlaps(self, other: RiskEstimate) -> bool:
        return self.ci_low <= other.ci_high and other.ci_low <= self.ci_high

    def distinguishable_from(self, other: RiskEstimate) -> bool:
        return not self.overlaps(other)
