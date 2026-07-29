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
"""

from __future__ import annotations

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
    cross on its way to liquidation. Crossing a boundary downward raises mmr
    and therefore raises a long's true liquidation price: this closed form is
    an *optimistic* display value in that case. The simulator recomputes the
    tier on every step and is the number that drives every risk output; this
    function is for display and for tests.
    """
    side = 1.0 if size > 0 else -1.0
    denom = 1.0 - mmr * side
    return price - side * margin_available / abs(size) / denom


def isolated_liquidation_price(position: Position, spec: AssetSpec) -> float:
    """Exact isolated liquidation price (§1.2).

    Unlike cross, this genuinely depends on the leverage the user set,
    because leverage determines how much collateral was moved into the
    position's pocket.
    """
    if position.mode is not MarginMode.ISOLATED:
        raise ValueError("isolated_liquidation_price called on a cross position")
    assert position.isolated_margin is not None
    entry = position.entry_price
    mmr = float(spec.maintenance_margin_rate(position.notional(entry)))
    # margin_per_unit = isolated_margin / |size|, expressed as a fraction of
    # entry it is exactly 1/L when the margin was allocated at leverage L.
    m = position.isolated_margin / (abs(position.size) * entry)
    if position.side > 0:
        return entry * (1.0 - m) / (1.0 - mmr)
    return entry * (1.0 + m) / (1.0 + mmr)


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
    mmr = float(spec.maintenance_margin_rate(position.notional(price)))
    if position.mode is MarginMode.ISOLATED:
        available = isolated_margin_available(position, price, spec)
    else:
        available = cross_margin_available(book, prices, specs)
    return _liq_from_margin_available(price, position.size, available, mmr)
