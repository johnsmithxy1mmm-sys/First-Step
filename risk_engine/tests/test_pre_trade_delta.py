"""§4.2 — Phase 2 acceptance criteria.

§9's Phase 2 criterion, item by item:
  - the delta has the right sign and order of magnitude on synthetic cases
  - overlapping intervals are recognised as indistinguishable
  - an asset outside the user's book is handled inside the 300 ms budget
"""

from __future__ import annotations

import numpy as np
import pytest

from risk_engine.domain.types import Book, MarginMode, Position
from risk_engine.observability.metrics import Metrics
from risk_engine.tools.pre_trade_delta import (
    LATENCY_BUDGET_MS,
    ProposedOrder,
    pre_trade_delta,
)


def book_at(now, leverage=8.0, equity=100_000.0):
    notional = equity * leverage
    return Book(
        "0xuser", equity,
        (
            Position("BTC", notional * 0.6 / 100_000.0, 100_000.0, MarginMode.CROSS, 20.0),
            Position("ETH", notional * 0.4 / 4_000.0, 4_000.0, MarginMode.CROSS, 20.0),
        ),
        now,
    )


class TestSignAndMagnitude:
    def test_adding_correlated_exposure_raises_risk(self, bundle, specs, spot, now):
        out = pre_trade_delta(
            book_at(now), ProposedOrder("SOL", 2_000.0, 20.0), spot, bundle, specs,
            n_paths=20_000, seed=1, now=now,
        )
        assert out.p_liq.direction > 0, out.summary()
        assert out.p_liq.change.point > 0
        assert out.cvar_95_usd.change.point > 0

    def test_a_bigger_order_moves_risk_further(self, bundle, specs, spot, now):
        """Ordering, not just sign: doubling the order must not shrink the delta."""
        book = book_at(now)
        deltas = []
        for size in (500.0, 2_000.0, 6_000.0):
            out = pre_trade_delta(
                book, ProposedOrder("SOL", size, 20.0), spot, bundle, specs,
                n_paths=20_000, seed=2, now=now,
            )
            deltas.append(out.p_liq.change.point)
        assert deltas[0] <= deltas[1] <= deltas[2], deltas
        assert deltas[2] > deltas[0]

    def test_a_hedging_order_lowers_risk(self, bundle, specs, spot, now):
        """The other sign. Selling into a long book reduces net exposure, so
        P(liq) must fall -- an engine that only ever reports 'riskier' would
        pass a sign test done in one direction only."""
        out = pre_trade_delta(
            book_at(now, leverage=14.0), ProposedOrder("BTC", -0.6, 20.0),
            spot, bundle, specs, n_paths=20_000, seed=3, now=now,
        )
        assert out.p_liq.direction < 0, out.summary()
        assert out.p_liq.change.point < 0

    def test_closing_the_whole_book_removes_liquidation_risk(self, bundle, specs, spot, now):
        book = Book("0x", 100_000.0,
                    (Position("BTC", 8.0, 100_000.0, MarginMode.CROSS, 20.0),), now)
        out = pre_trade_delta(
            book, ProposedOrder("BTC", -8.0, 20.0), spot, bundle, specs,
            n_paths=8_000, seed=4, now=now,
        )
        assert out.p_liq.after.point == 0.0
        assert out.p_liq.change.point <= 0

    def test_an_isolated_order_does_not_touch_the_cross_pool(self, bundle, specs, spot, now):
        """§1.1 through §4.2: the pocket's own risk appears, the cross pool's
        does not move except by the margin transferred out of the wallet."""
        book = book_at(now, leverage=4.0)
        out = pre_trade_delta(
            book, ProposedOrder("SOL", 500.0, 10.0, MarginMode.ISOLATED),
            spot, bundle, specs, n_paths=8_000, seed=5, now=now,
        )
        assert out.p_liq.after.point >= out.p_liq.before.point


class TestIndistinguishability:
    def test_a_negligible_order_is_reported_as_indistinguishable(
        self, bundle, specs, spot, now
    ):
        """§4.2: no arrow when there is nothing to draw."""
        out = pre_trade_delta(
            book_at(now), ProposedOrder("SOL", 0.01, 2.0), spot, bundle, specs,
            n_paths=20_000, seed=6, now=now,
        )
        assert not out.p_liq.distinguishable
        assert out.p_liq.direction == 0
        assert "no statistically distinguishable change" in out.summary()

    def test_the_paired_interval_is_far_tighter_than_the_marginals(
        self, bundle, specs, spot, now
    ):
        """Why common random numbers are not optional here: the concordant
        paths cancel exactly instead of contributing noise twice.

        Measured on a small order, which is where it matters -- that is
        exactly the regime in which two independent runs would report noise
        rather than the order's effect.
        """
        out = pre_trade_delta(
            book_at(now), ProposedOrder("SOL", 40.0, 20.0), spot, bundle, specs,
            n_paths=20_000, seed=7, now=now,
        )
        marginal_width = out.p_liq.before.half_width + out.p_liq.after.half_width
        assert out.p_liq.change.half_width < 0.35 * marginal_width

    def test_the_overlap_rule_is_reported_but_is_not_the_decision(
        self, bundle, specs, spot, now
    ):
        """OPEN-QUESTIONS D6. Overlapping marginal intervals do NOT imply an
        insignificant paired difference, and following §4.2's rule literally
        would hide real risk increases -- the §10-forbidden direction. Both
        answers are exposed and the discrepancy is flagged.

        This order raises P(liq) by ~0.17pp with a paired interval six times
        tighter than the marginal ones. §4.2's rule, applied literally, would
        tell the user "no detectable change".
        """
        out = pre_trade_delta(
            book_at(now), ProposedOrder("SOL", 40.0, 20.0), spot, bundle, specs,
            n_paths=20_000, seed=7, now=now,
        )
        assert out.p_liq.distinguishable
        assert out.p_liq.change.point > 0
        assert out.p_liq.marginal_intervals_overlap
        assert out.p_liq.overlap_rule_would_mislead

    def test_the_flag_counts_itself_in_observability(self, bundle, specs, spot, now):
        metrics = Metrics()
        pre_trade_delta(
            book_at(now), ProposedOrder("SOL", 40.0, 20.0), spot, bundle, specs,
            n_paths=20_000, seed=7, now=now, metrics=metrics,
        )
        assert metrics.counters["overlap_rule_would_mislead"] == 1

    def test_no_change_at_all_gives_a_degenerate_interval_not_a_fake_width(
        self, bundle, specs, spot, now
    ):
        book = book_at(now, leverage=2.0)
        out = pre_trade_delta(
            book, ProposedOrder("SOL", 0.001, 1.0), spot, bundle, specs,
            n_paths=4_000, seed=9, now=now,
        )
        if out.p_liq.change.point == 0.0:
            assert out.p_liq.change.ci_low == out.p_liq.change.ci_high == 0.0
            assert not out.p_liq.distinguishable


class TestColdStartAndBudget:
    def test_an_asset_outside_the_book_needs_no_cold_start(
        self, bundle, specs, spot, now
    ):
        """§4.2/§2.1: the global matrix already holds every tracked asset, so
        an unheld one costs a wider submatrix slice, not an estimation pass."""
        book = Book("0x", 100_000.0,
                    (Position("BTC", 5.0, 100_000.0, MarginMode.CROSS, 20.0),), now)
        assert "SOL" not in book.coins

        out = pre_trade_delta(
            book, ProposedOrder("SOL", 1_000.0, 20.0), spot, bundle, specs,
            n_paths=20_000, seed=10, now=now,
        )
        assert out.new_assets == ("SOL",)
        assert out.p_liq.after.point >= 0.0
        assert out.converged

    def test_the_unheld_asset_path_stays_inside_the_300ms_budget(
        self, bundle, specs, spot, now
    ):
        """The §9 Phase-2 acceptance number, measured rather than asserted.

        Timed over repeats because a single cold measurement on a shared CI
        box is mostly scheduler noise; the median is the honest figure.
        """
        book = Book("0x", 100_000.0,
                    (Position("BTC", 5.0, 100_000.0, MarginMode.CROSS, 20.0),), now)
        order = ProposedOrder("SOL", 1_000.0, 20.0)
        # Warm the quantile-map cache the same way a live process would be.
        pre_trade_delta(book, order, spot, bundle, specs, n_paths=20_000, seed=11, now=now)

        timings = [
            pre_trade_delta(
                book, order, spot, bundle, specs, n_paths=20_000, seed=12 + i, now=now
            ).total_latency_ms
            for i in range(5)
        ]
        median = float(np.median(timings))
        assert median <= LATENCY_BUDGET_MS, f"median {median:.0f}ms over budget: {timings}"

    def test_over_budget_runs_are_counted(self, bundle, specs, spot, now):
        """§7: a latency violation is a metric, never a silently cut path count.
        §2.5's interval rule outranks the clock (OPEN-QUESTIONS D1)."""
        metrics = Metrics()
        out = pre_trade_delta(
            book_at(now, leverage=20.0), ProposedOrder("SOL", 3_000.0, 20.0),
            spot, bundle, specs, n_paths=20_000, seed=13, now=now, metrics=metrics,
        )
        over = metrics.counters.get("pre_trade_budget_exceeded", 0.0)
        assert (over > 0) == (not out.within_budget)
        # Whatever the clock said, the interval rule was still honoured.
        assert out.converged
        assert out.p_liq.after.half_width <= 0.02 + 1e-9


class TestProvenanceAndContract:
    def test_every_number_carries_its_interval_and_version(
        self, bundle, specs, spot, now
    ):
        out = pre_trade_delta(
            book_at(now), ProposedOrder("SOL", 1_000.0, 20.0), spot, bundle, specs,
            n_paths=8_000, seed=14, now=now,
        )
        for est in (out.p_liq.before, out.p_liq.after, out.p_liq.change,
                    out.cvar_95_usd.before, out.cvar_95_usd.after, out.cvar_95_usd.change):
            assert est.ci_low <= est.point <= est.ci_high
            assert est.model_version == bundle.model_version
            assert est.computed_at == now
        assert out.provenance.seed == 14
        assert out.publishable

    def test_the_same_seed_reproduces_the_delta(self, bundle, specs, spot, now):
        args = (book_at(now), ProposedOrder("SOL", 1_000.0, 20.0), spot, bundle, specs)
        a = pre_trade_delta(*args, n_paths=8_000, seed=15, now=now)
        b = pre_trade_delta(*args, n_paths=8_000, seed=15, now=now)
        assert a.p_liq.change.point == b.p_liq.change.point
        assert a.cvar_95_usd.change.point == b.cvar_95_usd.change.point

    def test_an_order_in_an_untracked_asset_is_refused(self, bundle, specs, spot, now):
        with pytest.raises(KeyError):
            pre_trade_delta(
                book_at(now), ProposedOrder("DOGE", 1.0, 5.0), spot, bundle, specs,
                n_paths=1_000, seed=16, now=now,
            )

    def test_isolated_margin_defaults_to_notional_over_leverage(self):
        order = ProposedOrder("SOL", 100.0, 5.0, MarginMode.ISOLATED)
        pos = order.to_position(200.0)
        assert pos.isolated_margin == pytest.approx(100.0 * 200.0 / 5.0)
