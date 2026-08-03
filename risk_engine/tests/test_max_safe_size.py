"""§4.3 / OPEN-QUESTIONS D2.

The properties here are the ones that decide whether a recommended size is
safe to show a person. Most of them are about what the function REFUSES to
say.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from risk_engine.domain.types import Book, MarginMode, Position
from risk_engine.service.state import _build_fixture_bundle
from risk_engine.tools.max_safe_size import (
    MaxSafeSize,
    _evaluate,
    _rounded_down,
    max_safe_size,
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
PATHS = 4_000


@pytest.fixture(scope="module")
def world():
    bundle, specs, spot = _build_fixture_bundle()
    return bundle, specs, spot


@pytest.fixture(scope="module")
def coin(world):
    _, specs, spot = world
    return next(c for c in spot if c in specs)


def _book(world, coin, size, cash, address="0x" + "aa" * 20):
    _, _, spot = world
    positions = (
        (Position(coin, size, spot[coin], MarginMode.CROSS, 10.0),) if size else ()
    )
    return Book(address, cash, positions, NOW)


def _run(world, coin, book, **kw):
    bundle, specs, spot = world
    kw.setdefault("direction", 1)
    kw.setdefault("leverage", 10.0)
    kw.setdefault("threshold", 0.05)
    kw.setdefault("n_paths", PATHS)
    kw.setdefault("seed", 5)
    return max_safe_size(book, coin, spot, bundle, specs, now=NOW, workers=2, **kw)


class TestTheThreeOutcomes:
    """"No safe size" and "cannot tell" are different answers, and neither is 0."""

    def test_a_roomy_book_gets_a_size(self, world, coin):
        r = _run(world, coin, _book(world, coin, 0.2, 200_000.0))
        assert r.outcome == "safe"
        assert r.has_answer
        assert r.size is not None and r.size > 0

    def test_an_already_breached_book_is_told_there_is_no_safe_size(self, world, coin):
        """Not zero. Zero reads as "you may trade nothing extra"; the truth is
        that the account is already past the threshold before any order."""
        r = _run(world, coin, _book(world, coin, 3.0, 12_000.0))
        assert r.outcome == "none"
        assert r.size is None
        assert not r.has_answer
        assert "smallest tradable size" in r.reason

    def test_the_type_cannot_be_read_as_zero(self, world, coin):
        r = _run(world, coin, _book(world, coin, 3.0, 12_000.0))
        # A caller branching on `size` alone would treat this as "0 is safe".
        assert r.size is not None or r.outcome in ("none", "unresolved")
        assert r.summary().startswith("no safe size")


class TestTheUShape:
    """The reason this does not binary-search (D2 names only the noise half)."""

    @staticmethod
    def _hedged(world, coin):
        # Short and near the edge: a LONG order first OFFSETS the exposure,
        # so P(liq) falls before it rises.
        return _book(world, coin, -6.0, 60_000.0, "0x" + "77" * 20)

    def test_p_liq_is_genuinely_not_monotone_in_size(self, world, coin):
        """If this ever goes monotone the premise is gone and bisection would
        be admissible again — so it is asserted, not assumed."""
        bundle, specs, spot = world
        sizes = [0.0001, 1.0, 2.0, 5.0, 9.0, 12.0, 14.0]
        vs = _evaluate(
            self._hedged(world, coin), coin, sizes, +1, 10.0, MarginMode.CROSS,
            spot, bundle, specs, 24, PATHS, 0.05, 5, NOW, 2,
        )
        points = [v.p_liq.point for v in vs]
        assert points[0] > min(points), "adding an offsetting order should reduce P(liq)"
        assert points[-1] > min(points), "and enough of it should raise it again"

    def test_no_size_past_a_breach_is_ever_offered(self, world, coin):
        """A recommendation must not require the user to understand why 5 is
        safe, 10 is not, and 15 is safe again."""
        r = _run(world, coin, self._hedged(world, coin))
        if r.size is None:
            return
        first_breach = next((v.size for v in r.scanned if v.breaches), None)
        if first_breach is not None:
            assert r.size < first_breach

    def test_the_answer_is_a_size_that_actually_measured_safe(self, world, coin):
        r = _run(world, coin, self._hedged(world, coin))
        assert r.outcome == "safe"
        at = [v for v in r.scanned if v.size == r.size]
        assert at and not at[0].breaches


class TestTheThresholdTest:
    def test_the_upper_bound_is_what_is_tested_not_the_point(self, world, coin):
        """§4.3 says upper CI bound. A point under the line with an interval
        over it is not an answer the engine stands behind (§2.5)."""
        bundle, specs, spot = world
        vs = _evaluate(
            _book(world, coin, 0.5, 100_000.0), coin, [1.0], +1, 10.0,
            MarginMode.CROSS, spot, bundle, specs, 24, PATHS, 0.05, 5, NOW, 2,
        )
        v = vs[0]
        assert v.breaches == (v.p_liq.ci_high >= 0.05)

    def test_a_bound_exactly_on_the_threshold_counts_as_a_breach(self, world, coin):
        """§10 permits overstating risk and forbids the reverse, so the tie
        goes against the trade."""
        bundle, specs, spot = world
        vs = _evaluate(
            _book(world, coin, 0.5, 100_000.0), coin, [1.0], +1, 10.0,
            MarginMode.CROSS, spot, bundle, specs, 24, PATHS,
            # threshold set to exactly the measured upper bound
            _evaluate(_book(world, coin, 0.5, 100_000.0), coin, [1.0], +1, 10.0,
                      MarginMode.CROSS, spot, bundle, specs, 24, PATHS, 0.05, 5,
                      NOW, 2)[0].p_liq.ci_high,
            5, NOW, 2,
        )
        assert vs[0].breaches


class TestSizing:
    def test_sizes_are_snapped_down_never_up(self):
        """Rounding up would offer a size whose risk was never evaluated."""
        assert _rounded_down(0.37, 0.1) == pytest.approx(0.3)
        assert _rounded_down(0.3, 0.1) == pytest.approx(0.3)  # float 0.3/0.1 = 2.9999…
        assert _rounded_down(0.09, 0.1) == 0.0

    def test_the_answer_lands_on_a_tradable_increment(self, world, coin):
        _, specs, _ = world
        inc = specs[coin].size_increment
        r = _run(world, coin, _book(world, coin, 0.2, 200_000.0))
        assert r.size is not None
        assert abs(r.size / inc - round(r.size / inc)) < 1e-6

    def test_the_scan_ceiling_covers_flattening_an_opposing_position(self, world, coin):
        """Fresh capacity alone truncates the answer on exactly the hedged
        books this tool is for: an offsetting order releases margin rather
        than consuming it. Measured at 6 vs 12 before this was fixed."""
        r = _run(world, coin, _book(world, coin, -6.0, 60_000.0, "0x" + "77" * 20))
        assert max(v.size for v in r.scanned) > 6.0


class TestHonestyOfTheAnswer:
    def test_a_scan_bounded_answer_says_so(self, world, coin):
        r = _run(world, coin, _book(world, coin, 0.0, 5_000_000.0))
        if r.scan_bounded:
            assert "scanned range" in r.summary()

    def test_the_same_seed_reproduces(self, world, coin):
        """Common random numbers across candidates is what makes the scan a
        curve rather than a walk on noise; it must also make the whole query
        deterministic."""
        b = _book(world, coin, 0.2, 200_000.0)
        a, c = _run(world, coin, b), _run(world, coin, b)
        assert (a.outcome, a.size) == (c.outcome, c.size)

    def test_direction_is_not_symmetric_on_a_leaning_book(self, world, coin):
        """Buying and selling are different questions when the book already
        leans, and collapsing them would be wrong in one of the two."""
        b = _book(world, coin, -6.0, 60_000.0, "0x" + "77" * 20)
        long_r = _run(world, coin, b, direction=1)
        short_r = _run(world, coin, b, direction=-1)
        assert (long_r.outcome, long_r.size) != (short_r.outcome, short_r.size)


class TestRefusals:
    def test_an_unknown_coin_is_refused(self, world, coin):
        with pytest.raises(KeyError):
            _run(world, "NOSUCHCOIN", _book(world, coin, 0.2, 200_000.0))

    def test_a_nonsense_direction_is_refused(self, world, coin):
        with pytest.raises(ValueError, match="direction"):
            _run(world, coin, _book(world, coin, 0.2, 200_000.0), direction=0)

    def test_a_threshold_outside_zero_one_is_refused(self, world, coin):
        with pytest.raises(ValueError, match="threshold"):
            _run(world, coin, _book(world, coin, 0.2, 200_000.0), threshold=1.5)


def test_it_is_not_wired_to_a_serving_path_yet():
    """§3.3 gates recommendations exactly as it gates execution.

    `max_safe_size` is a Phase 4 deliverable and the shadow window has not
    cleared, so nothing a user acts on may call it. This is the same posture
    the inert execute button already has, and it is asserted rather than
    trusted because the whole point of the gate is that it does not depend on
    anyone remembering.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    serving = [root / "service" / "app.py", root / "service" / "state.py"]
    for path in serving:
        assert "max_safe_size" not in path.read_text(encoding="utf-8"), (
            f"{path.name} references max_safe_size; §4.3 is gate-closed until "
            "shadow validation clears (§3.3)"
        )
    assert isinstance(MaxSafeSize.__doc__, str)
