"""§1 — the liquidation model.

The load-bearing test here is `test_simulator_agrees_with_closed_form`: the
stepped simulation and the algebra in `margin.py` are two independent
implementations of the same condition, and they must flip at the same price.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from datetime import datetime, timezone

from risk_engine.domain.types import (
    AssetSpec,
    Book,
    MarginMode,
    MarginTier,
    Position,
)
from risk_engine.liquidation.margin import (
    _tier_consistent,
    cross_margin_available,
    isolated_liquidation_price,
    liquidation_price,
)
from risk_engine.liquidation.simulator import BridgeContext, LiquidationSimulator


COLS = {"BTC": 0, "ETH": 1, "SOL": 2}

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
ADDR = "0x" + "a" * 40


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


class TestTheSnapshotIsJudgedByTheSameRuleAsEveryStep:
    """§1.1 at t=0, on both branches, at the boundary that separates them.

    The stepping loop kills a path with `dead = alive & (gap <= 0)`, so a gap
    of exactly zero -- equity precisely equal to the maintenance requirement --
    is liquidated. The t=0 guard is a SECOND expression of the same rule
    (`alive &= gap_prev > 0`), written once for cross and once for isolated,
    and nothing asserted the two agreed.

    What is NOT claimed here, because it was checked and is false: that the
    t=0 guard is load-bearing. Mutating either branch to `>= 0` leaves every
    observable identical -- `cross_liquidated`, `isolated_liquidated`,
    `terminal_equity` and `funding_paid` all match to the bit -- because a
    book at or below maintenance in a flat market is killed by the per-step
    rule at s=1 regardless. The guard is belt-and-braces over the loop, not a
    separate decision, so those mutants are equivalent rather than a blind
    spot. Measured, not assumed.

    These tests are therefore about the OBSERVABLE contract, which nothing
    pinned either way: a book whose equity exactly equals its maintenance
    requirement is reported liquidated, on both branches, and one a dollar
    above it is not. Reporting the boundary case as surviving would understate
    risk in the direction §10 forbids, whichever line produced the answer.

    Exactly-at-maintenance is reachable in float arithmetic here because the
    fixture rate is exact at these sizes; the assertions below check the
    construction landed before they check the verdict, so a fixture change
    turns this into a loud failure rather than a silent pass.
    """

    PRICES: ClassVar[dict[str, float]] = {"BTC": 100_000.0, "ETH": 4_000.0, "SOL": 200.0}

    @staticmethod
    def _flat_paths(n_steps: int = 2):
        """Prices that never move. Any liquidation is the snapshot's doing."""
        return ramp_paths(TestTheSnapshotIsJudgedByTheSameRuleAsEveryStep.PRICES, {},
                          n_steps=n_steps)

    def test_a_cross_pool_exactly_at_maintenance_is_already_gone(self, specs, now):
        size, price = 1.0, self.PRICES["BTC"]
        notional = size * price
        mmr = float(specs["BTC"].maintenance_margin_rate(notional))
        # Entry at spot, so unrealised PnL is zero and the whole gap is
        # collateral minus requirement.
        pos = Position("BTC", size, price, MarginMode.CROSS, 10.0)
        book = Book(ADDR, mmr * notional, (pos,), now)

        sim = LiquidationSimulator(book, specs, COLS)
        gap, _ = sim._initial_gaps(self._flat_paths()[:, 0, :], 1, 0)
        assert gap[0] == pytest.approx(0.0, abs=1e-9), (
            "the book was not built exactly at maintenance, so this test is "
            f"not standing on the boundary it claims to (gap={gap[0]})"
        )

        assert sim.run(self._flat_paths()).cross_liquidated[0], (
            "equity exactly equal to the maintenance requirement is liquidated "
            "by the per-step rule (gap <= 0); the t=0 guard must agree, or a "
            "book the venue can liquidate on the next tick is published with a "
            "positive survival probability"
        )

    def test_an_isolated_pocket_exactly_at_maintenance_is_already_gone(self, specs, now):
        size, price = 10.0, self.PRICES["SOL"]
        notional = size * price
        mmr = float(specs["SOL"].maintenance_margin_rate(notional))
        pos = Position("SOL", size, price, MarginMode.ISOLATED, 20.0, mmr * notional)
        book = Book(ADDR, 0.0, (pos,), now)

        sim = LiquidationSimulator(book, specs, COLS)
        _, gap = sim._initial_gaps(self._flat_paths()[:, 0, :], 0, 1)
        assert gap[0, 0] == pytest.approx(0.0, abs=1e-9), (
            f"the pocket was not built exactly at maintenance (gap={gap[0, 0]})"
        )

        assert sim.run(self._flat_paths()).isolated_liquidated[0, 0], (
            "the isolated branch's t=0 guard must use the same inequality as "
            "its own per-step rule, exactly as the cross branch does"
        )

    def test_a_hair_above_maintenance_survives_a_flat_market(self, specs, now):
        """The other side of the boundary, so the two tests above are pinning a
        boundary rather than asserting that everything liquidates."""
        size, price = 1.0, self.PRICES["BTC"]
        notional = size * price
        mmr = float(specs["BTC"].maintenance_margin_rate(notional))
        pos = Position("BTC", size, price, MarginMode.CROSS, 10.0)
        book = Book(ADDR, mmr * notional + 1.0, (pos,), now)

        out = LiquidationSimulator(book, specs, COLS).run(self._flat_paths())
        assert not out.cross_liquidated[0]


class TestTheWithinStepCorrection:
    """§2's Brownian-bridge correction, which had no test of its own at all.

    Before this class the string "bridge" appeared exactly once in the whole
    test tree, in a comment explaining that a test had switched it OFF. Yet
    `use_bridge` defaults to True everywhere in `engine.py`, so every number
    the product publishes carries it, and on a plain 10x BTC book it moves
    p_liq from 0.00395 to 0.00535 -- a 35% relative increase. The module
    docstring is explicit about why it exists: checking the condition only at
    step closes misses every excursion that breaches and recovers inside the
    hour, "which understates P(liq) -- the direction §10 forbids".

    So the largest single modelling choice in §2 after the copula was
    exercised only incidentally and asserted nowhere. Mutation found it by
    accident: both gap-variance expressions survive a `*` -> `/` swap on the
    `mmr * |size|` term, and the reason is not that the term does not matter
    but that no test looks at the correction at all.
    """

    def test_the_hit_probability_is_the_first_passage_formula(self):
        """P(hit) = exp(-2 g0 g1 / var), stated in the docstring, asserted here.

        Written against the closed form rather than against a recorded number,
        so it pins the identity and not a regression baseline.
        """
        g0 = np.array([500.0, 2_000.0, 50.0])
        g1 = np.array([300.0, 2_500.0, 10.0])
        var = np.array([1e6, 1e6, 1e5])
        got = LiquidationSimulator._bridge_hit_prob(g0, g1, var)

        assert np.allclose(got, np.exp(-2.0 * g0 * g1 / var))
        # Monotone in the obvious directions: further from the barrier is
        # safer, more volatile is riskier.
        assert got[0] > got[1], "a gap twice as far from zero cannot be riskier"
        assert LiquidationSimulator._bridge_hit_prob(
            g0, g1, var * 4.0
        )[0] > got[0], "quadrupling the step variance cannot lower P(hit)"

    def test_a_touched_barrier_is_certain_and_a_dead_variance_is_impossible(self):
        """The two degenerate ends. A zero endpoint means the gap reached the
        barrier, so the correction must return 1 rather than something that
        rounds to it; zero variance means the gap cannot move, and dividing by
        it must not leak a NaN into a published probability."""
        z = np.zeros(2)
        assert np.all(LiquidationSimulator._bridge_hit_prob(z, np.array([5.0, 9.0]),
                                                            np.array([1e6, 1e6])) == 1.0)
        out = LiquidationSimulator._bridge_hit_prob(
            np.array([100.0]), np.array([100.0]), np.array([0.0])
        )
        assert np.isfinite(out).all() and out[0] == 0.0

    def test_a_zero_variance_and_a_touched_barrier_together_stay_a_number(self):
        """0/0, which is where the guard actually earns its keep.

        A gap of zero AND a variance of zero -- an asset with no step
        volatility whose gap has reached the barrier -- makes the exponent
        `-0/0`, i.e. NaN, and NaN survives `np.clip`. `np.where(var > 0, ...)`
        replaces it; `np.where(var >= 0, ...)` does not, and that mutant
        survives every test that feeds a positive endpoint, because there the
        exponent is `-x/0 = -inf` and `exp(-inf)` is a perfectly good 0.0.
        A NaN here becomes `dead |= u < NaN`, silently False: a path that
        touched the barrier reported as surviving.
        """
        out = LiquidationSimulator._bridge_hit_prob(
            np.array([0.0, 100.0]), np.array([0.0, 100.0]), np.zeros(2)
        )
        assert np.isfinite(out).all(), f"NaN leaked into a hit probability: {out}"
        assert (out == 0.0).all()

    def test_a_negative_endpoint_cannot_overflow_into_a_probability(self):
        """Endpoints are floored before the exponential. A path already past
        the barrier has been caught by the direct check, and its negative
        product would otherwise send `exp` to +inf and clip to certainty on a
        path that is not being asked about."""
        out = LiquidationSimulator._bridge_hit_prob(
            np.array([-5_000.0]), np.array([4_000.0]), np.array([1.0])
        )
        assert np.isfinite(out).all()
        assert 0.0 <= out[0] <= 1.0

    def test_the_gap_variance_is_the_delta_approximation_it_claims_to_be(self, specs, now):
        """`var(dg)` against the formula the module docstring writes out.

        The docstring states `dg/dP_j = size_j - mmr_j * |size_j|` and
        `cov(dP_j, dP_k) = P_j P_k sigma_j sigma_k rho_jk`. Nothing checked
        that the code implements it, and the statistical test below cannot:
        a wrong variance shifts a first-passage probability by a couple of
        per cent, which moves no verdict on any finite sample. Mutation found
        it that way round -- `mmr * |size|` swapped to `mmr / |size|` survives
        the whole engine-facing suite.

        Two positions with |size| != 1 and != each other, and a correlation
        that is not the identity, so every factor in the expression is pinned:
        at unit size the `|size|` factor is invisible (multiplying and
        dividing by 1 agree), and at rho = I the cross terms vanish and the
        einsum degenerates into a sum of squares.
        """
        positions = (
            Position("BTC", 3.7, 100_000.0, MarginMode.CROSS, 10.0),
            Position("ETH", -12.5, 4_000.0, MarginMode.CROSS, 10.0),
        )
        book = Book(ADDR, 250_000.0, positions, now)
        sim = LiquidationSimulator(book, specs, COLS)

        step_vol = np.array([0.011, 0.017, 0.023])
        rho = 0.6
        corr = np.eye(3)
        corr[0, 1] = corr[1, 0] = rho
        bridge = BridgeContext(step_vol=step_vol, corr=corr)

        prices = np.array([[100_000.0, 4_000.0]])
        sizes = np.array([3.7, -12.5])
        mmr = np.array([[
            float(specs["BTC"].maintenance_margin_rate(3.7 * 100_000.0)),
            float(specs["ETH"].maintenance_margin_rate(12.5 * 4_000.0)),
        ]])

        d = sizes - mmr[0] * np.abs(sizes)
        v = d * prices[0] * step_vol[:2]
        expected = v[0] ** 2 + v[1] ** 2 + 2.0 * rho * v[0] * v[1]

        got = sim._cross_gap_var(prices, mmr, bridge)
        assert got[0] == pytest.approx(expected, rel=1e-12)
        # The |size| factor is what a unit-size fixture cannot see, so name the
        # wrong version and assert it is a different number.
        wrong = sizes - mmr[0] / np.abs(sizes)
        vw = wrong * prices[0] * step_vol[:2]
        assert got[0] != pytest.approx(
            vw[0] ** 2 + vw[1] ** 2 + 2.0 * rho * vw[0] * vw[1], rel=1e-9
        )

    def test_an_isolated_pockets_gap_variance_has_no_cross_terms(self, specs, now):
        """§1.1 again, at the variance: each pocket is its own single-asset
        problem, so `var(dg_i)` is exactly `(d_i P_i sigma_i)^2` with no
        correlation anywhere in it. A pocket that borrowed the cross pool's
        einsum would couple pockets that share no collateral."""
        positions = (
            Position("SOL", 40.0, 200.0, MarginMode.ISOLATED, 20.0, 2_000.0),
            Position("ETH", -2.5, 4_000.0, MarginMode.ISOLATED, 10.0, 1_500.0),
        )
        book = Book(ADDR, 0.0, positions, now)
        sim = LiquidationSimulator(book, specs, COLS)

        step_vol = np.array([0.011, 0.017, 0.023])
        corr = np.full((3, 3), 0.9)
        np.fill_diagonal(corr, 1.0)
        bridge = BridgeContext(step_vol=step_vol, corr=corr)

        iso_cols = sim._iso_cols
        prices = np.array([[{"SOL": 200.0, "ETH": 4_000.0}[c]
                            for c in (p.coin for p in book.isolated_positions)]])
        sizes = np.array([p.size for p in book.isolated_positions])
        mmr = np.array([[float(specs[p.coin].maintenance_margin_rate(abs(p.size) * px))
                         for p, px in zip(book.isolated_positions, prices[0], strict=True)]])

        got = sim._iso_gap_var(prices, mmr, bridge)
        d = sizes - mmr[0] * np.abs(sizes)
        expected = (d * prices[0] * step_vol[iso_cols]) ** 2
        assert np.allclose(got[0], expected, rtol=1e-12)
        assert got.shape == (1, 2), "one variance per pocket, not one for the book"

    @pytest.mark.parametrize("size", [1.0, 3.7])
    def test_the_correction_can_only_raise_p_liq_never_lower_it(self, specs, now, size):
        """The §10 direction, asserted as an inequality rather than a value.

        The bridge adds intra-step breaches to the step-close ones; it can
        never remove a liquidation that already happened at a close. A sign
        slip or a swapped endpoint would show up here as the correction making
        a book look SAFER, which is the one direction that is forbidden.

        This is the statistical half and it is deliberately weak: it pins the
        direction, not the magnitude. The magnitude is pinned by
        `test_the_gap_variance_is_the_delta_approximation_it_claims_to_be`,
        because a variance that is wrong by a few per cent moves no count on
        any sample this suite can afford.
        """
        pos = Position("BTC", size, 100_000.0, MarginMode.CROSS, 10.0)
        book = Book(ADDR, 100_000.0 * size * 0.11, (pos,), now)
        sim = LiquidationSimulator(book, specs, COLS)

        n_steps, n_paths = 24, 4_000
        rng = np.random.default_rng(11)
        shocks = rng.normal(0.0, 0.012, size=(n_paths, n_steps))
        paths = np.empty((n_paths, n_steps + 1, len(COLS)))
        for coin, col in COLS.items():
            base = {"BTC": 100_000.0, "ETH": 4_000.0, "SOL": 200.0}[coin]
            steps = shocks if coin == "BTC" else np.zeros_like(shocks)
            paths[:, :, col] = base * np.exp(
                np.concatenate([np.zeros((n_paths, 1)), steps.cumsum(axis=1)], axis=1)
            )

        closes_only = sim.run(paths).cross_liquidated
        bridge = BridgeContext(
            step_vol=np.full(len(COLS), 0.012), corr=np.eye(len(COLS)),
        )
        u = rng.random((n_paths, n_steps))
        corrected = sim.run(
            paths, bridge=bridge,
            bridge_uniforms=(u, rng.random((n_paths, n_steps, 0))),
        ).cross_liquidated

        assert not (closes_only & ~corrected).any(), (
            "the within-step correction removed a liquidation that had already "
            "happened at a step close; it may only ever add"
        )
        assert corrected.sum() > closes_only.sum(), (
            f"|size|={size}: the correction changed nothing on {n_paths} paths, "
            "so this test is not exercising it"
        )


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

    PRICES: ClassVar[dict[str, float]] = {"BTC": 90_000.0, "ETH": 4_000.0, "SOL": 200.0}

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


class TestTierConsistencyIsActuallyExercised:
    """The tier-aware closed form shipped with nothing that runs it.

    A mutation sweep over `margin.py` killed 17 of 25 mutants and left three
    alive, all inside `_tier_consistent`: the convergence test (`nxt == mmr`),
    the conservative tie-break on a two-cycle, and the degenerate-price guard.
    They survived the whole liquidation-facing suite, not just one module.

    The cause is one number. `conftest.spec` builds BTC with two tiers and the
    boundary at $150 000 000 notional, while every test book is $100k-$500k —
    five orders of magnitude below it. So `maintenance_margin_rate` returns
    the same rate whatever the price, `nxt == mmr` holds on the FIRST pass
    every time, and the loop, the straddle and the tie-break are dead code as
    far as the suite is concerned. The fixture comment says the two tiers mean
    "the per-step tier lookup in §1.3 is actually exercised", which is true of
    the LOOKUP and false of the iteration built on it.

    Live `meta` has 34 assets with more than one tier (E5, measured), so this
    machinery runs in production against books that do cross a boundary.
    """

    def _spec(self, boundary: float, second_leverage: float = 4.0) -> AssetSpec:
        """A boundary an ordinary book crosses, unlike the shipped fixture's."""
        return AssetSpec(
            name="BTC", sz_decimals=5, max_leverage=40.0,
            tiers=(MarginTier(0.0, 40.0), MarginTier(boundary, second_leverage)),
        )

    def test_the_answer_lands_in_the_tier_it_assumed(self):
        """What `_tier_consistent` exists for. The liquidation price implies a
        notional, the notional implies a tier, and the tier has to be the one
        the price was computed from — otherwise the number describes a margin
        requirement that does not apply at the price it names.

        Kills the mutant that flips the convergence test to `nxt != mmr`,
        which returns on the first iteration where the tier CHANGED.
        """
        spec = self._spec(300_000.0)
        book = Book(ADDR, 60_000.0,
                    (Position("BTC", 5.0, 100_000.0, MarginMode.CROSS, 20.0),), NOW)
        px = liquidation_price(book.positions[0], book, {"BTC": 100_000.0},
                               {"BTC": spec})
        assert px == pytest.approx(100_571.43, abs=0.01)
        # The notional at that price is past the boundary, so the SECOND
        # tier's rate is the one that must have produced it.
        assert 5.0 * px > 300_000.0
        assert spec.maintenance_margin_rate(5.0 * px) == pytest.approx(0.5 / 4.0)

    def test_a_straddle_takes_the_candidate_that_warns_earliest(self):
        """The two-cycle the conservative branch was written for, constructed
        rather than hoped for.

        A SHORT is what oscillates: raising the assumed rate moves the
        liquidation price toward spot, which for a short LOWERS it, which
        lowers the notional and the tier — a decreasing map, and a decreasing
        map is what can cycle. A long's map is increasing and converges.

        Here the two candidates sit on OPPOSITE SIDES of spot: 90 666.67 and
        100 740.74 against a spot of 100 000. A short is liquidated by a
        RISE, so only the second is reachable at all, and it is also the one
        closest to spot — §10 permits overstating risk and forbids the
        reverse. Kills the mutant that flips the tie-break's `p - reference`
        to `p + reference`.
        """
        spec = self._spec(454_000.0)
        book = Book(ADDR, 10_000.0,
                    (Position("BTC", -5.0, 100_000.0, MarginMode.CROSS, 20.0),), NOW)
        px = liquidation_price(book.positions[0], book, {"BTC": 100_000.0},
                               {"BTC": spec})
        assert px == pytest.approx(100_740.74, abs=0.01), (
            "the straddle picked the candidate 9% BELOW spot, which a short "
            "cannot reach by rising, over the one 0.7% above it"
        )

    def test_a_degenerate_price_is_returned_without_iterating(self):
        """`solve` is a parameter precisely so the helper can be driven, and
        this is the branch no book reaches by accident: a price at or below
        zero means the liquidation is unreachable or the book degenerate, and
        feeding it back through the tier table would ask for the margin rate
        of a negative notional.

        Kills the mutant that narrows `price <= 0.0` to `price < 0.0`, which
        no constructible book distinguishes — exactly 0.0 is not reachable
        through the public API by any input a test could choose.
        """
        spec = self._spec(300_000.0)
        calls = []

        def solve(mmr: float) -> float:
            calls.append(mmr)
            return 0.0

        # `initial_mmr` is the SECOND tier's rate while the rate at a zero
        # notional is the first tier's. That gap is what makes the guard
        # observable: skip it and the two disagree, so the loop takes another
        # pass. With a matching pair the mutant returns the same 0.0 after the
        # same single call and nothing distinguishes them — which is how a
        # first version of this test passed against both.
        out = _tier_consistent(solve, spec, abs_size=5.0, reference=100_000.0,
                               initial_mmr=0.5 / 4.0)
        assert out == 0.0
        assert spec.maintenance_margin_rate(0.0) != 0.5 / 4.0, (
            "the fixture no longer distinguishes the two branches"
        )
        assert len(calls) == 1, f"iterated {len(calls)} times on a zero price"
