"""Closed-form margin and liquidation-price arithmetic (§1.1, §1.2, §1.3).

These functions are the display-side and test-side counterpart to
`simulator.py`. They must agree: `test_liquidation.py` asserts that the
simulator flips a position to liquidated at exactly the price these
functions return, which is the cheapest available check that the stepped
simulation implements the same condition as the algebra.

Two things here are deliberately not what §1.2 prints:

1. §1.2's isolated shortcut `entry * (1 - 1/L + mmr)` is a first-order
   expansion. The exact solution of the §1.1 condition is
   `entry * (1 - 1/L) / (1 - mmr)` for a long and
   `entry * (1 + 1/L) / (1 + mmr)` for a short. The expansion sits *above*
   the exact price in both cases, which is conservative for a long and
   anti-conservative for a short -- forbidden by §10. The exact form is used.

2. The general formula's `position_size` is read as `|size|`, not the signed
   size. With a signed size it returns a below-spot liquidation price for a
   short, which is unreachable. See docs/hl-risk/OPEN-QUESTIONS.md A2.

3. Both public functions solve for the maintenance-margin TIER they land in,
   rather than the tier the position sits in today. Holding the tier fixed is
   anti-conservative for a short -- it walks into heavier tiers as it loses,
   so the displayed price sat beyond the true one (126 bp on a 1490 BTC short,
   with the simulator liquidating inside the gap). §10 forbids exactly that
   direction. See `_tier_consistent`.
"""

from __future__ import annotations

import math

from risk_engine.domain.types import AssetSpec, Book, MarginMode, Position


def maintenance_margin(spec: AssetSpec, position: Position, price: float) -> float:
    """Maintenance margin required for `position` valued at `price` (§1.3)."""
    notional = position.notional(price)
    return float(spec.maintenance_margin_rate(notional)) * notional


def cross_maintenance_margin(
    book: Book, prices: dict[str, float], specs: dict[str, AssetSpec]
) -> float:
    return sum(maintenance_margin(specs[p.coin], p, prices[p.coin]) for p in book.cross_positions)


def cross_margin_available(
    book: Book, prices: dict[str, float], specs: dict[str, AssetSpec]
) -> float:
    """Cross account value minus the cross pool's maintenance requirement.

    This going negative *is* the cross liquidation condition (§1.1).
    """
    return book.cross_account_value(prices) - cross_maintenance_margin(book, prices, specs)


def isolated_margin_available(
    position: Position, price: float, spec: AssetSpec
) -> float:
    assert position.isolated_margin is not None
    equity = position.isolated_margin + position.unrealised_pnl(price)
    return equity - maintenance_margin(spec, position, price)


def _liq_from_margin_available(
    price: float, size: float, margin_available: float, mmr: float
) -> float:
    """Solve `equity(P) == maintenance_margin(P)` for P, holding mmr fixed.

    Derivation is in the module docstring of `simulator.py`; the result is
    Hyperliquid's published formula with `|size|` in the denominator.

    Holding mmr fixed ignores tier boundaries (§1.3) that the position might
    cross on its way to liquidation, and the direction of that error is NOT
    the same on both sides. `AssetSpec` enforces max-leverage non-increasing
    in size, so mmr is non-decreasing in notional, and therefore:

      - LONG: price falls toward liquidation, notional falls, the true mmr is
        at or below the fixed one, so the true liquidation price is at or
        below the displayed one. The display warns EARLY. Conservative.
      - SHORT: price rises toward liquidation, notional rises, the true mmr is
        at or above the fixed one, so the true liquidation price is again
        BELOW the displayed one -- which for a short means the display warns
        LATE. Anti-conservative, and the direction §10 forbids.

    Measured on the real BTC tier table (0->40x, 150M->20x): a 1490 BTC short
    displays 103,703.70 against a true tier-aware 102,440 -- 126 bp of entry
    beyond reality, and `LiquidationSimulator` does liquidate inside that gap.
    A randomized sweep over valid tier tables found 21/21 shorts optimistic
    and 0/29 longs optimistic.

    (An earlier version of this note had the two sides the wrong way round and
    named the long as the optimistic case. It is the short.)

    This function is the raw single-tier algebra and still holds mmr fixed --
    it is the inner step of the solve. `liquidation_price` and
    `isolated_liquidation_price` wrap it in `_tier_consistent`, which iterates
    until the assumed tier is the tier the answer lands in, so the numbers
    those two return are tier-aware and no longer carry the error above.
    """
    side = 1.0 if size > 0 else -1.0
    denom = 1.0 - mmr * side
    return price - side * margin_available / abs(size) / denom


def _tier_consistent(solve, spec: AssetSpec, abs_size: float, reference: float,
                     initial_mmr: float, max_iter: int = 12) -> float:
    """Iterate `solve(mmr)` until the tier it lands in is the tier it assumed.

    `solve` must take a maintenance-margin rate and return the liquidation
    price implied by it. The tier table is a step function of notional, so
    this reaches a consistent pair in a couple of passes or settles into a
    two-cycle straddling a boundary.

    When it does straddle, the conservative branch is taken: the candidate
    whose liquidation price is CLOSEST to the reference price, i.e. the one
    that warns earliest. §10 permits overstating risk and forbids the
    opposite, and a two-cycle means the true crossing is between the two
    candidates anyway.
    """
    mmr = initial_mmr
    candidates: list[float] = []
    for _ in range(max_iter):
        price = solve(mmr)
        if not math.isfinite(price) or price <= 0.0:
            # Unreachable liquidation (or a degenerate book); the single-tier
            # answer is as meaningful as any and the caller handles the sign.
            return price
        candidates.append(price)
        nxt = float(spec.maintenance_margin_rate(abs_size * price))
        if nxt == mmr:
            return price
        mmr = nxt
    return min(candidates, key=lambda p: abs(p - reference))


def isolated_liquidation_price(position: Position, spec: AssetSpec) -> float:
    """Isolated liquidation price (§1.2), consistent with the tier it lands in.

    Unlike cross, this genuinely depends on the leverage the user set,
    because leverage determines how much collateral was moved into the
    position's pocket.

    Not called "exact": it solves the §1.1 condition exactly for a single
    maintenance-margin rate and iterates that rate to the tier the answer
    falls in, which agrees with the stepped simulator at every tier boundary
    tested. What it does not model is anything the simulator adds beyond the
    §1.1 condition itself.
    """
    if position.mode is not MarginMode.ISOLATED:
        raise ValueError("isolated_liquidation_price called on a cross position")
    assert position.isolated_margin is not None
    entry = position.entry_price
    abs_size = abs(position.size)
    # margin_per_unit = isolated_margin / |size|, expressed as a fraction of
    # entry it is exactly 1/L when the margin was allocated at leverage L.
    m = position.isolated_margin / (abs_size * entry)
    long = position.side > 0

    def solve(mmr: float) -> float:
        if long:
            return entry * (1.0 - m) / (1.0 - mmr)
        return entry * (1.0 + m) / (1.0 + mmr)

    # Tier-consistent rather than fixed at the entry tier: a short walks INTO
    # heavier tiers as it loses, and pinning mmr at entry put the displayed
    # price beyond the true one -- 126 bp on a 1490 BTC short, in the
    # §10-forbidden direction. See `_liq_from_margin_available`.
    return _tier_consistent(
        solve, spec, abs_size, entry,
        float(spec.maintenance_margin_rate(position.notional(entry))),
    )


def liquidation_price(
    position: Position,
    book: Book,
    prices: dict[str, float],
    specs: dict[str, AssetSpec],
) -> float:
    """Price of `position.coin` at which `position` is liquidated.

    For a cross position this is the price at which the *whole cross pool*
    breaches, holding every other price fixed -- which is why it does not
    depend on the leverage the user set on this position (§1.2). Cross
    positions share one equity pool; the leverage slider only gates how much
    size may be opened, not where the pool breaks.
    """
    spec = specs[position.coin]
    price = prices[position.coin]
    abs_size = abs(position.size)
    initial_mmr = float(spec.maintenance_margin_rate(position.notional(price)))

    def solve(mmr: float) -> float:
        # `available` is recomputed for each candidate tier: it is defined as
        # equity minus the maintenance requirement, so it moves with mmr and
        # holding it fixed while varying mmr would solve a different equation.
        if position.mode is MarginMode.ISOLATED:
            assert position.isolated_margin is not None
            equity = position.isolated_margin + position.unrealised_pnl(price)
            available = equity - mmr * abs_size * price
        else:
            # Only THIS position's requirement depends on the tier being
            # solved for; the rest of the cross pool is held at its own tiers,
            # exactly as `cross_margin_available` computes it.
            available = cross_margin_available(book, prices, specs)
            available += maintenance_margin(spec, position, price)
            available -= mmr * abs_size * price
        return _liq_from_margin_available(price, position.size, available, mmr)

    return _tier_consistent(solve, spec, abs_size, price, initial_mmr)
