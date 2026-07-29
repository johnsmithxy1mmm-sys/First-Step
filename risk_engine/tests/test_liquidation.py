"""§1 — the liquidation model.

The load-bearing test here is `test_simulator_agrees_with_closed_form`: the
stepped simulation and the algebra in `margin.py` are two independent
implementations of the same condition, and they must flip at the same price.
"""

from __future__ import annotations

import numpy as np
import pytest

from risk_engine.domain.types import Book, MarginMode, Position
from risk_engine.liquidation.margin import (
    cross_margin_available,
    isolated_liquidation_price,
    liquidation_price,
)
from risk_engine.liquidation.simulator import LiquidationSimulator


COLS = {"BTC": 0, "ETH": 1, "SOL": 2}


def ramp_paths(coin_prices: dict[str, float], target: dict[str, float], n_steps: int = 24):
    """One path that walks linearly from `coin_prices` to `target`."""
    out = np.empty((1, n_steps + 1, len(COLS)))
    for coin, col in COLS.items():
        a, b = coin_prices[coin], target.get(coin, coin_prices[coin])
        out[0, :, col] = np.linspace(a, b, n_steps + 1)
    return out


class TestClosedForm:
    def test_isolated_long_matches_condition(self, specs, now):
        """The exact formula solves equity == maintenance margin, by construction."""
        pos = Position("BTC", 1.0, 100_000.0, MarginMode.ISOLATED, 10.0, 10_000.0)
        liq = isolated_liquidation_price(pos, specs["BTC"])

        mmr = specs["BTC"].maintenance_margin_rate(abs(pos.size) * liq)
        equity = pos.isolated_margin + pos.unrealised_pnl(liq)
        assert equity == pytest.approx(mmr * abs(pos.size) * liq, rel=1e-12)

    def test_isolated_short_matches_condition(self, specs, now):
        pos = Position("BTC", -1.0, 100_000.0, MarginMode.ISOLATED, 10.0, 10_000.0)
        liq = isolated_liquidation_price(pos, specs["BTC"])
        assert liq > pos.entry_price  # a short is liquidated on the way up

        mmr = specs["BTC"].maintenance_margin_rate(abs(pos.size) * liq)
        equity = pos.isolated_margin + pos.unrealised_pnl(liq)
        assert equity == pytest.approx(mmr * abs(pos.size) * liq, rel=1e-12)

    def test_spec_shortcut_understates_short_risk(self, specs):
        """OPEN-QUESTIONS A2: why §1.2's shortcut is not used.

        The shortcut puts the short's liquidation price *above* the true one,
        i.e. further from spot, i.e. it makes the position look safer. §10
        forbids simplifications in that direction, so this asserts the gap
        exists rather than tolerating it.
        """
        pos = Position("BTC", -1.0, 100_000.0, MarginMode.ISOLATED, 20.0, 5_000.0)
        mmr = specs["BTC"].maintenance_margin_rate(100_000.0)
        shortcut = pos.entry_price * (1 + 1 / 20.0 - mmr)
        exact = isolated_liquidation_price(pos, specs["BTC"])
        assert shortcut > exact

    def test_cross_liquidation_ignores_the_leverage_slider(self, specs, now):
        """§1.2: cross positions share one equity pool."""
        prices = {"BTC": 100_000.0, "ETH": 4_000.0, "SOL": 200.0}
        low = Book("0x", 20_000.0, (Position("BTC", 1.0, 100_000.0, MarginMode.CROSS, 3.0),), now)
        high = Book("0x", 20_000.0, (Position("BTC", 1.0, 100_000.0, MarginMode.CROSS, 40.0),), now)
        a = liquidation_price(low.positions[0], low, prices, specs)
        b = liquidation_price(high.positions[0], high, prices, specs)
        assert a == pytest.approx(b)

    def test_isolated_liquidation_does_depend_on_leverage(self, specs):
        tight = Position("BTC", 1.0, 100_000.0, MarginMode.ISOLATED, 20.0, 5_000.0)
        loose = Position("BTC", 1.0, 100_000.0, MarginMode.ISOLATED, 5.0, 20_000.0)
        assert isolated_liquidation_price(tight, specs["BTC"]) > isolated_liquidation_price(
            loose, specs["BTC"]
        )


class TestSimulatorAgreesWithClosedForm:
    @pytest.mark.parametrize("size", [1.0, -1.0])
    @pytest.mark.parametrize("mode", [MarginMode.CROSS, MarginMode.ISOLATED])
    def test_flips_at_the_closed_form_price(self, specs, now, size, mode):
        entry = 100_000.0
        collateral = 12_000.0
        iso = collateral if mode is MarginMode.ISOLATED else None
        cash = 0.0 if mode is MarginMode.ISOLATED else collateral
        pos = Position("BTC", size, entry, mode, 10.0, iso)
        book = Book("0x", cash, (pos,), now)
        prices = {"BTC": entry, "ETH": 4_000.0, "SOL": 200.0}
        liq = liquidation_price(pos, book, prices, specs)

        sim = LiquidationSimulator(book, specs, COLS)
        # Two flat paths: one that stops just short of the liquidation price,
        # one that steps just past it. No bridge -- this is about the
        # step-close condition matching the algebra exactly.
        eps = liq * 1e-6
        just_short = liq + eps if size > 0 else liq - eps
        just_past = liq - eps if size > 0 else liq + eps

        for target, expected in ((just_short, False), (just_past, True)):
            paths = ramp_paths(prices, {"BTC": target}, n_steps=2)
            out = sim.run(paths)
            hit = out.cross_liquidated[0] if mode is MarginMode.CROSS else out.isolated_liquidated[0, 0]
            assert bool(hit) is expected, f"{mode} {size} at {target} vs liq {liq}"


class TestPoolIndependence:
    """§1.1: N+1 conditions, not one. A blown isolated pocket must not
    propagate into the cross pool, and vice versa."""

    def test_isolated_blowup_leaves_cross_alive(self, specs, now):
        positions = (
            Position("BTC", 0.05, 100_000.0, MarginMode.CROSS, 5.0),
            Position("SOL", 100.0, 200.0, MarginMode.ISOLATED, 20.0, 1_000.0),
        )
        book = Book("0x", 50_000.0, positions, now)
        sim = LiquidationSimulator(book, specs, COLS)
        prices = {"BTC": 100_000.0, "ETH": 4_000.0, "SOL": 200.0}
        # SOL halves (isolated pocket gone), BTC untouched.
        paths = ramp_paths(prices, {"SOL": 100.0}, n_steps=24)
        out = sim.run(paths)
        assert out.isolated_liquidated[0, 0]
        assert not out.cross_liquidated[0]
        # Cross equity survives intact; only the $1000 pocket is lost.
        assert out.terminal_equity[0] == pytest.approx(50_000.0, rel=1e-9)

    def test_cross_blowup_leaves_isolated_alive(self, specs, now):
        positions = (
            Position("BTC", 1.0, 100_000.0, MarginMode.CROSS, 10.0),
            Position("SOL", 10.0, 200.0, MarginMode.ISOLATED, 20.0, 1_500.0),
        )
        book = Book("0x", 8_000.0, positions, now)
        sim = LiquidationSimulator(book, specs, COLS)
        prices = {"BTC": 100_000.0, "ETH": 4_000.0, "SOL": 200.0}
        paths = ramp_paths(prices, {"BTC": 80_000.0}, n_steps=24)
        out = sim.run(paths)
        assert out.cross_liquidated[0]
        assert not out.isolated_liquidated[0, 0]
        # Only the isolated pocket's equity remains.
        assert out.terminal_equity[0] == pytest.approx(1_500.0, rel=1e-9)

    def test_two_isolated_positions_are_independent(self, specs, now):
        positions = (
            Position("SOL", 100.0, 200.0, MarginMode.ISOLATED, 20.0, 1_000.0),
            Position("ETH", 1.0, 4_000.0, MarginMode.ISOLATED, 5.0, 800.0),
        )
        book = Book("0x", 0.0, positions, now)
        sim = LiquidationSimulator(book, specs, COLS)
        prices = {"BTC": 100_000.0, "ETH": 4_000.0, "SOL": 200.0}
        paths = ramp_paths(prices, {"SOL": 100.0}, n_steps=24)
        out = sim.run(paths)
        sol_idx = out.isolated_coins.index("SOL")
        eth_idx = out.isolated_coins.index("ETH")
        assert out.isolated_liquidated[0, sol_idx]
        assert not out.isolated_liquidated[0, eth_idx]


class TestMarginTiers:
    def test_tier_is_rechecked_as_the_position_moves(self, btc):
        """§1.3: the tier is a function of notional at the moment of check."""
        small = btc.maintenance_margin_rate(10_000_000.0)
        large = btc.maintenance_margin_rate(200_000_000.0)
        assert large > small
        assert small == pytest.approx(0.5 / 40.0)
        assert large == pytest.approx(0.5 / 20.0)

    def test_vectorised_lookup_matches_scalar(self, btc):
        vals = np.array([0.0, 1e6, 149_999_999.0, 150_000_000.0, 1e12])
        got = btc.maintenance_margin_rate(vals)
        want = np.array([float(btc.maintenance_margin_rate(float(v))) for v in vals])
        assert np.allclose(got, want)


class TestFunding:
    """§1.5: funding is charged on every hourly step, not applied post-hoc."""

    def test_long_pays_positive_funding_from_cross_cash(self, specs, now):
        pos = Position("BTC", 1.0, 100_000.0, MarginMode.CROSS, 5.0)
        book = Book("0x", 30_000.0, (pos,), now)
        sim = LiquidationSimulator(book, specs, COLS)
        prices = {"BTC": 100_000.0, "ETH": 4_000.0, "SOL": 200.0}
        paths = ramp_paths(prices, {}, n_steps=24)
        funding = np.zeros((1, 24, len(COLS)))
        funding[:, :, COLS["BTC"]] = 1e-4  # 1 bp/hour
        out = sim.run(paths, funding_paths=funding)
        # 24 hours x 1bp x $100k notional
        assert out.funding_paid[0] == pytest.approx(24 * 1e-4 * 100_000.0)
        assert out.terminal_equity[0] == pytest.approx(30_000.0 - out.funding_paid[0])

    def test_short_receives_positive_funding(self, specs, now):
        pos = Position("BTC", -1.0, 100_000.0, MarginMode.CROSS, 5.0)
        book = Book("0x", 30_000.0, (pos,), now)
        sim = LiquidationSimulator(book, specs, COLS)
        prices = {"BTC": 100_000.0, "ETH": 4_000.0, "SOL": 200.0}
        paths = ramp_paths(prices, {}, n_steps=24)
        funding = np.zeros((1, 24, len(COLS)))
        funding[:, :, COLS["BTC"]] = 1e-4
        out = sim.run(paths, funding_paths=funding)
        assert out.funding_paid[0] == pytest.approx(-24 * 1e-4 * 100_000.0)

    def test_funding_alone_can_liquidate(self, specs, now):
        """The whole point of §1.5: on a flat price path, funding still kills."""
        pos = Position("BTC", 1.0, 100_000.0, MarginMode.CROSS, 20.0)
        book = Book("0x", 2_000.0, (pos,), now)  # thin cross pool
        sim = LiquidationSimulator(book, specs, COLS)
        prices = {"BTC": 100_000.0, "ETH": 4_000.0, "SOL": 200.0}
        paths = ramp_paths(prices, {}, n_steps=168)
        funding = np.zeros((1, 168, len(COLS)))
        funding[:, :, COLS["BTC"]] = 5e-4
        assert not sim.run(paths).cross_liquidated[0]
        assert sim.run(paths, funding_paths=funding).cross_liquidated[0]

    def test_isolated_funding_is_debited_from_the_pocket(self, specs, now):
        """OPEN-QUESTIONS C5 records that this needs protocol confirmation."""
        pos = Position("SOL", 100.0, 200.0, MarginMode.ISOLATED, 10.0, 2_000.0)
        book = Book("0x", 100_000.0, (pos,), now)
        sim = LiquidationSimulator(book, specs, COLS)
        prices = {"BTC": 100_000.0, "ETH": 4_000.0, "SOL": 200.0}
        paths = ramp_paths(prices, {}, n_steps=24)
        funding = np.zeros((1, 24, len(COLS)))
        funding[:, :, COLS["SOL"]] = 1e-3
        out = sim.run(paths, funding_paths=funding)
        paid = 24 * 1e-3 * 20_000.0
        assert out.funding_paid[0] == pytest.approx(paid)
        # The cross pool is untouched; the pocket absorbed all of it.
        assert out.terminal_equity[0] == pytest.approx(100_000.0 + 2_000.0 - paid)


class TestAlreadyLiquidatable:
    def test_book_below_maintenance_at_t0_is_reported_immediately(self, specs, now):
        pos = Position("BTC", 1.0, 100_000.0, MarginMode.CROSS, 40.0)
        book = Book("0x", 100.0, (pos,), now)  # $100 backing $100k of BTC
        prices = {"BTC": 100_000.0, "ETH": 4_000.0, "SOL": 200.0}
        assert cross_margin_available(book, prices, specs) < 0
        sim = LiquidationSimulator(book, specs, COLS)
        out = sim.run(ramp_paths(prices, {}, n_steps=2))
        assert out.cross_liquidated[0]


class TestWithPositionConservesValue:
    """Audit A-01. An order moves value between wallet, positions and
    pockets; it never creates any. Evaluated at a spot equal to the execution
    price, equity must be identical before and after. The original merge
    fabricated $50k on a market-price flip by dropping the realized PnL of
    the closed portion and letting the flipped side inherit the old entry."""

    PRICES = {"BTC": 90_000.0, "ETH": 4_000.0, "SOL": 200.0}

    @pytest.mark.parametrize("order_size", [-0.5, -1.0, -1.9, -2.0, -3.0, -5.0, 1.0])
    def test_cross_order_at_market_conserves_equity(self, now, order_size):
        book = Book("0x", 50_000.0,
                    (Position("BTC", 2.0, 100_000.0, MarginMode.CROSS, 20.0),), now)
        px = self.PRICES["BTC"]
        after = book.with_position(
            Position("BTC", order_size, px, MarginMode.CROSS, 20.0)
        )
        assert after.equity(self.PRICES) == pytest.approx(book.equity(self.PRICES), rel=1e-12)

    @pytest.mark.parametrize("order_size", [-0.5, -1.0, -2.0, -3.0, 1.0])
    def test_isolated_order_at_market_conserves_equity(self, now, order_size):
        book = Book("0x", 50_000.0,
                    (Position("BTC", 2.0, 100_000.0, MarginMode.ISOLATED, 5.0, 40_000.0),), now)
        px = self.PRICES["BTC"]
        iso = abs(order_size) * px / 5.0
        after = book.with_position(
            Position("BTC", order_size, px, MarginMode.ISOLATED, 5.0, iso)
        )
        assert after.equity(self.PRICES) == pytest.approx(book.equity(self.PRICES), rel=1e-12)

    def test_opening_a_fresh_position_conserves_equity(self, now):
        book = Book("0x", 50_000.0,
                    (Position("BTC", 1.0, 100_000.0, MarginMode.CROSS, 20.0),), now)
        for order in (
            Position("ETH", 5.0, 4_000.0, MarginMode.CROSS, 10.0),
            Position("SOL", -100.0, 200.0, MarginMode.ISOLATED, 4.0, 5_000.0),
        ):
            after = book.with_position(order)
            assert after.equity(self.PRICES) == pytest.approx(
                book.equity(self.PRICES), rel=1e-12
            ), order.coin

    def test_the_exact_case_the_audit_reproduced(self, now):
        """PoC from the audit: +$10k on a reduce, +$50k on a flip."""
        book = Book("0x", 50_000.0,
                    (Position("BTC", 2.0, 100_000.0, MarginMode.CROSS, 20.0),), now)
        assert book.equity(self.PRICES) == pytest.approx(30_000.0)
        reduced = book.with_position(Position("BTC", -1.0, 90_000.0, MarginMode.CROSS, 20.0))
        flipped = book.with_position(Position("BTC", -5.0, 90_000.0, MarginMode.CROSS, 20.0))
        assert reduced.equity(self.PRICES) == pytest.approx(30_000.0)
        assert flipped.equity(self.PRICES) == pytest.approx(30_000.0)
        # The flipped remainder opens at the execution price, not the old entry.
        assert flipped.positions[0].size == pytest.approx(-3.0)
        assert flipped.positions[0].entry_price == pytest.approx(90_000.0)

    def test_full_close_returns_pocket_and_pnl_to_the_wallet(self, now):
        book = Book("0x", 10_000.0,
                    (Position("SOL", 100.0, 200.0, MarginMode.ISOLATED, 4.0, 5_000.0),), now)
        px = 180.0
        after = book.with_position(Position("SOL", -100.0, px, MarginMode.ISOLATED, 4.0, 1.0))
        assert after.positions == ()
        # 5_000 pocket - 2_000 loss returns to the wallet on top of the 10k.
        assert after.cross_collateral == pytest.approx(13_000.0)
        assert after.equity({"SOL": px}) == pytest.approx(book.equity({"SOL": px}))

    def test_a_reduce_that_would_empty_the_pocket_is_refused(self, now):
        """Such a book is not constructible on the venue -- the pocket would
        have been liquidated first -- and returning it would understate risk."""
        book = Book("0x", 10_000.0,
                    (Position("SOL", 100.0, 200.0, MarginMode.ISOLATED, 20.0, 1_000.0),), now)
        # Closing 20 of the 100 units at half the entry realizes -2_000 into a
        # 1_000 pocket: the venue would have wiped it long before this price.
        with pytest.raises(ValueError, match="isolated pocket"):
            book.with_position(Position("SOL", -20.0, 100.0, MarginMode.ISOLATED, 20.0, 1.0))

    def test_same_side_add_uses_a_volume_weighted_entry(self, now):
        book = Book("0x", 50_000.0,
                    (Position("BTC", 1.0, 100_000.0, MarginMode.CROSS, 20.0),), now)
        after = book.with_position(Position("BTC", 1.0, 90_000.0, MarginMode.CROSS, 20.0))
        assert after.positions[0].entry_price == pytest.approx(95_000.0)


class TestNonFiniteInputsAreRefused:
    """Audit A-07: NaN evaluates False in every comparison, so an unguarded
    validator waves it straight through into the risk numbers."""

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_position_fields(self, bad):
        with pytest.raises(ValueError, match="finite"):
            Position("BTC", bad, 100.0, MarginMode.CROSS, 5.0)
        with pytest.raises(ValueError, match="finite"):
            Position("BTC", 1.0, bad, MarginMode.CROSS, 5.0)

    def test_cross_collateral(self, now):
        with pytest.raises(ValueError, match="finite"):
            Book("0x", float("nan"), (), now)


class TestCheckpointedHorizons:
    """Audit A-10: several horizons must come from one walk, so liquidation
    at a longer horizon is a pathwise superset of a shorter one."""

    def test_flags_are_nested_across_checkpoints(self, specs, now):
        book = Book("0x", 6_000.0,
                    (Position("BTC", 1.0, 100_000.0, MarginMode.CROSS, 20.0),), now)
        sim = LiquidationSimulator(book, specs, COLS)
        rng = np.random.default_rng(3)
        n, steps = 400, 48
        paths = np.empty((n, steps + 1, len(COLS)))
        for coin, col in COLS.items():
            base = {"BTC": 100_000.0, "ETH": 4_000.0, "SOL": 200.0}[coin]
            shocks = rng.standard_normal((n, steps)) * 0.02
            paths[:, 0, col] = base
            paths[:, 1:, col] = base * np.exp(np.cumsum(shocks, axis=1))

        outs = sim.run_checkpointed(paths, (12, 24, 48))
        early, mid, late = outs[12], outs[24], outs[48]
        assert np.all(late.cross_liquidated >= mid.cross_liquidated)
        assert np.all(mid.cross_liquidated >= early.cross_liquidated)
        assert late.cross_liquidated.mean() >= early.cross_liquidated.mean()

    def test_final_checkpoint_equals_a_plain_run(self, specs, now):
        book = Book("0x", 20_000.0,
                    (Position("BTC", 1.0, 100_000.0, MarginMode.CROSS, 20.0),), now)
        sim = LiquidationSimulator(book, specs, COLS)
        prices = {"BTC": 100_000.0, "ETH": 4_000.0, "SOL": 200.0}
        paths = ramp_paths(prices, {"BTC": 85_000.0}, n_steps=24)
        assert np.array_equal(
            sim.run(paths).cross_liquidated,
            sim.run_checkpointed(paths, (24,))[24].cross_liquidated,
        )

    def test_checkpoint_outside_the_walk_is_refused(self, specs, now):
        book = Book("0x", 20_000.0,
                    (Position("BTC", 1.0, 100_000.0, MarginMode.CROSS, 20.0),), now)
        sim = LiquidationSimulator(book, specs, COLS)
        prices = {"BTC": 100_000.0, "ETH": 4_000.0, "SOL": 200.0}
        with pytest.raises(ValueError, match="outside"):
            sim.run_checkpointed(ramp_paths(prices, {}, n_steps=10), (99,))
