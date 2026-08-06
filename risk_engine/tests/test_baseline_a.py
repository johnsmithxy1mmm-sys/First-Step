"""What Baseline A actually predicts (§3.2, OPEN-QUESTIONS B3).

B3 described this predictor as collapsing the book to its net notional
exposure with an equity change of `net_exposure x r`. Both halves were wrong,
and the tests here are the measurements that showed it — kept so the
description and the code cannot drift apart again.

The distinction is not cosmetic. "Collapsed to net notional" predicts one
P(liq) for any two books with the same net exposure; the code gives two
different ones, because isolated pockets fail independently.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from risk_engine.domain.types import Book, MarginMode, Position
from risk_engine.service.state import _build_fixture_bundle
from risk_engine.validation.baselines import NaiveBaseline, historical_24h_log_returns

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def world():
    bundle, specs, spot = _build_fixture_bundle()
    return bundle, specs, spot


@pytest.fixture(scope="module")
def coins(world):
    _, specs, spot = world
    return [c for c in spot if c in specs]


def _baseline(scale: float = 0.02) -> NaiveBaseline:
    rng = np.random.default_rng(3)
    prices = np.cumprod(1 + rng.normal(0, scale, 6000)) * 100.0
    return NaiveBaseline(historical_24h_log_returns(np.diff(np.log(prices))))


class TestTheEquityChangeCollapses:
    def test_it_is_net_notional_times_expm1_not_times_r(self, world, coins):
        """The draws are LOG returns and the code exponentiates them.

        `net_exposure x r` and `net_exposure x (e^r - 1)` differ enough at
        crypto's 24h scale to matter in the tail — which is the part that
        decides a liquidation.
        """
        _, specs, spot = world
        positions = (
            Position(coins[0], 2.0, spot[coins[0]], MarginMode.CROSS, 10.0),
            Position(coins[1], 30.0, spot[coins[1]], MarginMode.CROSS, 10.0),
        )
        book = Book("0x" + "11" * 20, 500_000.0, positions, NOW)
        net = sum(p.size * spot[p.coin] for p in positions)

        nb = _baseline(scale=0.008)
        pred = nb.predict(book, spot, specs, n_draws=20_000, seed=1, now=NOW)
        got = np.quantile(pred.equity_change.values, [0.05, 0.95])
        want = np.quantile(net * np.expm1(nb.history), [0.05, 0.95])
        assert got == pytest.approx(want, rel=0.02)

    def test_the_structure_does_not_change_it(self, world, coins):
        """One factor means one source of variation, so the spread depends on
        net exposure alone — this half of B3's description is right."""
        _, specs, spot = world
        n, coll = 1_000_000.0, 60_000.0
        cross = Book("0x" + "11" * 20, coll,
                     (Position(coins[0], n / spot[coins[0]], spot[coins[0]],
                               MarginMode.CROSS, 20.0),), NOW)
        iso = Book("0x" + "22" * 20, 0.0,
                   (Position(coins[0], n / spot[coins[0]], spot[coins[0]],
                             MarginMode.ISOLATED, 20.0, isolated_margin=coll),), NOW)
        nb = _baseline()
        a = nb.predict(cross, spot, specs, n_draws=20_000, seed=1, now=NOW)
        b = nb.predict(iso, spot, specs, n_draws=20_000, seed=1, now=NOW)
        assert a.equity_change.values.std() == pytest.approx(
            b.equity_change.values.std(), rel=0.01
        )


class TestPLiqDoesNotCollapse:
    def test_two_pockets_are_riskier_than_one_at_the_same_net_notional(
        self, world, coins
    ):
        """The half of B3's description that was wrong.

        Same net notional, same collateral — but isolated pockets fail
        INDEPENDENTLY and `any_liq` is a union over them, so splitting the
        book across two raises P(liq). A predictor that had really been
        collapsed to net notional could not show this.
        """
        _, specs, spot = world
        n, coll = 1_000_000.0, 60_000.0
        one = Book("0x" + "22" * 20, 0.0,
                   (Position(coins[0], n / spot[coins[0]], spot[coins[0]],
                             MarginMode.ISOLATED, 20.0, isolated_margin=coll),), NOW)
        two = Book("0x" + "33" * 20, 0.0,
                   (Position(coins[0], n / 2 / spot[coins[0]], spot[coins[0]],
                             MarginMode.ISOLATED, 20.0, isolated_margin=coll / 2),
                    Position(coins[1], n / 2 / spot[coins[1]], spot[coins[1]],
                             MarginMode.ISOLATED, 20.0, isolated_margin=coll / 2)), NOW)
        nb = _baseline()
        p_one = nb.predict(one, spot, specs, n_draws=20_000, seed=1, now=NOW).p_liq
        p_two = nb.predict(two, spot, specs, n_draws=20_000, seed=1, now=NOW).p_liq
        assert p_one > 0.0, "the fixture must reach liquidation for this to mean anything"
        assert p_two > p_one


class TestItIsNaiveInTheIntendedWay:
    def test_a_hedged_book_looks_almost_riskless_to_it(self, world, coins):
        """One factor means perfect correlation and no idiosyncratic risk.

        This is the naivety §3.2 wants, and it is why beating Baseline A is
        informative: the real model gives a hedged book genuine idiosyncratic
        risk that this predictor cannot see. A baseline nobody could beat, or
        one nobody could lose to, would make the comparison empty.
        """
        _, specs, spot = world
        hedged = Book("0x" + "44" * 20, 500_000.0, (
            Position(coins[0], 2.0, spot[coins[0]], MarginMode.CROSS, 10.0),
            Position(coins[1], -40.0, spot[coins[1]], MarginMode.CROSS, 10.0),
        ), NOW)
        outright = Book("0x" + "55" * 20, 500_000.0, (
            Position(coins[0], 2.0, spot[coins[0]], MarginMode.CROSS, 10.0),
        ), NOW)
        nb = _baseline()
        h = nb.predict(hedged, spot, specs, n_draws=20_000, seed=1, now=NOW)
        o = nb.predict(outright, spot, specs, n_draws=20_000, seed=1, now=NOW)
        # The offsetting legs cancel against a common factor, so the hedged
        # book's spread is a fraction of the outright one's.
        assert h.equity_change.values.std() < 0.5 * o.equity_change.values.std()

    def test_it_carries_no_funding_and_no_path(self, world, coins):
        """Documented naivety, asserted so it stays true: one step evaluated
        at the horizon end, so no intra-horizon monitoring and no funding."""
        import inspect

        src = inspect.getsource(NaiveBaseline.predict)
        assert "funding" not in src.lower()
        # Two price columns only: spot and the horizon end.
        assert "prices = np.empty((n_draws, 2," in src
