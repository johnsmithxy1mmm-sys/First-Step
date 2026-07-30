"""Value types shared by every layer of the engine.

Three of these carry design intent rather than just data:

`RiskEstimate` cannot be constructed without an interval. §4 requires that
returning a point estimate alone be *impossible*, not merely discouraged, so
the interval is positional and validated in `__post_init__`.

`AssetSpec` owns the margin-tier table (§1.3). The table comes from the
`meta` endpoint; nothing in this package hardcodes a maintenance margin rate.

`normalise_address` is the single definition of what an account address is.
It lives here, in the layer both the Info client and the calibration journal
already sit above, because an address is a domain identity rather than a
detail of either the transport or the storage -- and because a copy in each
of them would be two definitions that could drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from itertools import pairwise

import numpy as np

#: An account is 20 bytes, written as `0x` plus exactly 40 hex digits.
ADDRESS_HEX_DIGITS = 40
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def normalise_address(value: object) -> str:
    """The one canonical spelling of an account address: `0x` + 40 lowercase hex.

    Every address in this system ends up in one of two places -- the `user`
    field of an Info request, or the `address` column of the calibration
    journal -- and both punish a spelling difference silently rather than
    loudly. That is why this is a boundary that raises, not a convenience:

      - the venue answers an address it does not recognise with a well-formed
        *empty* state (§5.1), so a typo does not fail. It reads as "this
        account holds no positions", which for a risk tool is the most
        dangerous possible failure;
      - journal rows are written once and never updated (audit A-09), so
        anything written is permanent. `progress()` counts
        `DISTINCT address` towards §3.3's 200-account gate, `load_cohort`
        makes one row per address-day, and champion/challenger pairs runs on
        `(address, observation_day)`. One account admitted under two
        spellings therefore counts twice towards the gate, contributes the
        same realised outcome twice to the cohort, and can break the pairing
        the migration decision depends on -- unfixably, since the rows cannot
        be edited afterwards.

    Case is *folded*, not checked. An EIP-55 checksummed address -- what every
    block explorer displays, and so what an operator pastes -- is accepted and
    returns identical to its all-lowercase form, because the account is the 20
    bytes and the mixed case is only a checksum written over them. Verifying
    that checksum here would either reject the perfectly legal all-lowercase
    spelling or need keccak in a module that needs none, and would buy a
    refusal to measure a real account's risk.

    Whitespace is refused rather than stripped. An address list is
    hand-edited, so a stray tab or newline is a mistake worth showing the
    operator, and trimming it silently would make two entries that look
    different behave the same -- the same class of confusion this function
    exists to remove.

    Every rejection names the specific defect. A single regex would be
    shorter, but "invalid address" is not enough to fix a 42-character string
    by eye, and the operator holding a wrong address is exactly the person
    §5.1's empty-state footgun is waiting for.
    """
    if value is None:
        raise ValueError(
            "address is required, got None; expected 0x followed by "
            f"{ADDRESS_HEX_DIGITS} hex digits"
        )
    if not isinstance(value, str):
        raise ValueError(
            f"address must be a string, got {type(value).__name__}: {value!r}"
        )
    if not value:
        raise ValueError(
            f"address is empty; expected 0x followed by {ADDRESS_HEX_DIGITS} hex digits"
        )
    if any(ch.isspace() for ch in value):
        raise ValueError(
            f"address must not contain whitespace, got {value!r}; "
            "remove the surrounding spaces or newline rather than relying on a trim"
        )
    if value[:2] not in ("0x", "0X"):
        raise ValueError(
            f"address must start with '0x', got {value!r}"
        )
    digits = value[2:]
    if len(digits) != ADDRESS_HEX_DIGITS:
        raise ValueError(
            f"address must have exactly {ADDRESS_HEX_DIGITS} hex digits after '0x', "
            f"got {len(digits)} in {value!r}"
        )
    if unexpected := sorted(set(digits) - _HEX_DIGITS):
        raise ValueError(
            f"address must be hexadecimal after '0x', got {value!r} "
            f"containing {unexpected}"
        )
    return "0x" + digits.lower()


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
        # Finiteness first (audit A-07): NaN passes every `<=` comparison
        # below by evaluating False, and an infinite max leverage would give
        # a maintenance rate of zero -- both silent risk-understatements.
        if not (math.isfinite(self.lower_bound) and math.isfinite(self.max_leverage)):
            raise ValueError(
                f"tier fields must be finite, got ({self.lower_bound}, {self.max_leverage})"
            )
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
        # Finiteness before sign checks (audit A-07): NaN evaluates False in
        # every comparison below, so a NaN-poisoned position would otherwise
        # sail through validation and silently corrupt every downstream number.
        for name in ("size", "entry_price", "leverage"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{self.coin}: {name} must be finite, got {getattr(self, name)}")
        if self.isolated_margin is not None and not math.isfinite(self.isolated_margin):
            raise ValueError(f"{self.coin}: isolated_margin must be finite")
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
        if not math.isfinite(self.cross_collateral):
            raise ValueError(f"cross_collateral must be finite, got {self.cross_collateral}")
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
        """Return the book as it stands the instant `position`'s order fills.

        `position.entry_price` is read as the execution price of the
        hypothetical order. Used by `pre_trade_delta` (§4.2) and
        `max_safe_size` (§4.3).

        Conservation contract (audit A-01): an order moves value between the
        cross wallet, positions and isolated pockets, but never creates or
        destroys it, so evaluating the result at a spot equal to the
        execution price leaves `equity` unchanged. That requires two things
        the naive merge got wrong, fabricating $50k on a market-price flip:

        - the realized PnL of any closed portion is credited somewhere real:
          the cross wallet for cross positions, the pocket for a partial
          isolated close, the wallet for a full isolated close;
        - a flipped position opens at the execution price, not at the entry
          price of the side it replaced.

        An isolated order's `isolated_margin` is *transferred* from the cross
        wallet into the pocket -- the wallet is debited by it, including on
        reducing orders, where it models topping the pocket up.

        Raises when a reducing isolated order would leave its pocket
        non-positive: such a book is not constructible on the venue (the
        pocket would have been liquidated first), and returning it anyway
        would understate risk.
        """
        prior = next((p for p in self.positions if p.coin == position.coin), None)
        others = tuple(p for p in self.positions if p.coin != position.coin)
        cash = self.cross_collateral

        if prior is None:
            if position.mode is MarginMode.ISOLATED:
                cash -= position.isolated_margin or 0.0  # wallet -> new pocket
            return Book(self.address, cash, (*others, position), self.captured_at)

        if prior.mode is not position.mode:
            raise ValueError(
                f"{position.coin}: cannot mix {prior.mode.value} and {position.mode.value}"
            )

        exec_price = position.entry_price
        size = prior.size + position.size

        if prior.side == position.side:
            # Same-side add: a volume-weighted entry conserves equity by
            # construction, no cash movement beyond the pocket transfer.
            entry = (prior.size * prior.entry_price + position.size * exec_price) / size
            iso = None
            if position.mode is MarginMode.ISOLATED:
                iso = (prior.isolated_margin or 0.0) + (position.isolated_margin or 0.0)
                cash -= position.isolated_margin or 0.0
            merged = Position(position.coin, size, entry, position.mode, position.leverage, iso)
            return Book(self.address, cash, (*others, merged), self.captured_at)

        # Opposite sides: |closed| units of the prior position close at the
        # execution price and their PnL becomes real.
        closed = min(abs(prior.size), abs(position.size)) * prior.side
        realized = closed * (exec_price - prior.entry_price)

        if size == 0:
            # Full close: realized PnL and any pocket margin return to the wallet.
            cash += realized + (prior.isolated_margin or 0.0)
            return Book(self.address, cash, others, self.captured_at)

        flipped = (size > 0) != (prior.size > 0)
        if position.mode is MarginMode.ISOLATED:
            if flipped:
                # The old pocket closes entirely to the wallet; the order's
                # margin funds the new pocket on the other side.
                cash += (prior.isolated_margin or 0.0) + realized
                cash -= position.isolated_margin or 0.0
                iso = position.isolated_margin
            else:
                # Partial close: the venue realizes isolated PnL into the
                # pocket; the order's margin tops it up from the wallet.
                cash -= position.isolated_margin or 0.0
                iso = (prior.isolated_margin or 0.0) + (position.isolated_margin or 0.0) + realized
                if iso <= 0:
                    raise ValueError(
                        f"{position.coin}: hypothetical reduce leaves the isolated pocket "
                        f"at {iso:.2f} <= 0; the venue would have liquidated it first"
                    )
        else:
            cash += realized
            iso = None
        entry = exec_price if flipped else prior.entry_price
        merged = Position(position.coin, size, entry, position.mode, position.leverage, iso)
        return Book(self.address, cash, (*others, merged), self.captured_at)


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
        # Normalise numpy scalars to Python floats. Estimates are routinely
        # built from np.quantile and friends, and a np.float64 field makes
        # `overlaps` return np.bool_ -- which is not a bool subclass and is
        # not JSON serialisable, so the failure surfaces at the service
        # boundary as a 400 rather than here. Coercing at the type keeps
        # every consumer honest.
        # `type(...) is not float`, not `isinstance`: np.float64 IS a float
        # subclass, so an isinstance check passes it through unconverted and
        # every comparison on it still yields np.bool_.
        for name in ("point", "ci_low", "ci_high"):
            value = getattr(self, name)
            if type(value) is not float:
                object.__setattr__(self, name, float(value))
        if not math.isfinite(self.point + self.ci_low + self.ci_high):
            raise ValueError(
                f"estimate must be finite, got ({self.point}, {self.ci_low}, {self.ci_high})"
            )
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
