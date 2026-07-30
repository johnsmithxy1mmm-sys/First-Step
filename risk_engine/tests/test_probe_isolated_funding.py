"""The C5 probe, driven by a stub venue across a synthetic funding tick.

C5 is the one open question that can invalidate structure rather than shift
a number: if funding on an isolated position is debited from the cross pool,
§1.1's independence is false and the simulator has no term for it. A probe
that answers it has to be right about which balance moved, and — more
importantly — has to refuse to answer when something else moved the account.
"""

from __future__ import annotations

import time

import pytest

from risk_engine.market import probe_isolated_funding as probe_mod
from risk_engine.market.probe_isolated_funding import probe


class StubVenue:
    """Two scripted `clearinghouseState` reads plus a funding ledger."""

    def __init__(self, before: dict, after: dict, funding: list | Exception):
        self._states = [before, after]
        self._funding = funding
        self.calls = 0

    def clearinghouse_state(self, address, is_agent_address=False):
        state = self._states[min(self.calls, len(self._states) - 1)]
        self.calls += 1
        return state

    def user_funding(self, address, start_ms, end_ms=None):
        if isinstance(self._funding, Exception):
            raise self._funding
        return self._funding


def _state(cross: float, isolated: dict[str, float], sizes: dict[str, float] | None = None):
    sizes = sizes or dict.fromkeys(isolated, 1.0)
    positions = []
    for coin, margin in isolated.items():
        positions.append({
            "position": {
                "coin": coin, "szi": str(sizes[coin]), "entryPx": "100",
                "unrealizedPnl": "0",
                "leverage": {"type": "isolated", "value": 10, "rawUsd": str(margin)},
            }
        })
    return {"crossMarginSummary": {"accountValue": str(cross)}, "assetPositions": positions}


def _funding(coin: str, usdc: float, when_ms: int | None = None):
    # Stamped at "now": the probe only counts payments inside the window it
    # observed, so an epoch-0 timestamp would be filtered out and every
    # verdict would read INCONCLUSIVE for the wrong reason.
    when_ms = when_ms if when_ms is not None else int(time.time() * 1000)
    return [{"time": when_ms, "delta": {"coin": coin, "usdc": str(usdc)}}]


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Every test here supplies its own venue; none may reach the wire."""
    monkeypatch.setattr(probe_mod, "InfoClient", lambda **kw: pytest.fail(
        "the probe tried to open a real connection"
    ))


def _run(monkeypatch, venue, **kwargs):
    monkeypatch.setattr(probe_mod, "InfoClient", lambda **kw: venue)
    return probe("0xtest", wait=False, **kwargs)


class TestVerdicts:
    def test_isolated_margin_absorbing_the_payment_passes(self, monkeypatch):
        """§1.1 holds: the pocket paid its own funding, cross untouched."""
        venue = StubVenue(
            _state(50_000.0, {"SOL": 1_000.0}),
            _state(50_000.0, {"SOL": 990.0}),
            _funding("SOL", 10.0),
        )
        result = _run(monkeypatch, venue)
        assert result.status == "PASS"
        assert "independence holds" in result.detail
        assert result.deltas["SOL"] == pytest.approx(-10.0)

    def test_the_cross_pool_absorbing_it_fails_loudly(self, monkeypatch):
        """The finding that would invalidate the isolated/cross separation.
        It must be named as structural, not filed as a discrepancy."""
        venue = StubVenue(
            _state(50_000.0, {"SOL": 1_000.0}),
            _state(49_990.0, {"SOL": 1_000.0}),
            _funding("SOL", 10.0),
        )
        result = _run(monkeypatch, venue)
        assert result.status == "FAIL"
        assert "independence claim is false" in result.detail
        assert result.cross_delta == pytest.approx(-10.0)

    def test_a_received_payment_is_attributed_the_same_way(self, monkeypatch):
        """Shorts receive. The verdict uses magnitudes, so a credit to the
        isolated pocket must read as isolated accounting too."""
        venue = StubVenue(
            _state(50_000.0, {"SOL": 1_000.0}),
            _state(50_000.0, {"SOL": 1_010.0}),
            _funding("SOL", -10.0),
        )
        result = _run(monkeypatch, venue)
        assert result.status == "PASS"

    def test_neither_balance_moving_is_ambiguous_not_a_pass(self, monkeypatch):
        venue = StubVenue(
            _state(50_000.0, {"SOL": 1_000.0}),
            _state(50_000.0, {"SOL": 1_000.0}),
            _funding("SOL", 10.0),
        )
        result = _run(monkeypatch, venue)
        assert result.status == "INCONCLUSIVE"


class TestRefusals:
    def test_no_isolated_position_is_unchecked_not_a_pass(self, monkeypatch):
        """The address has to be able to answer the question. Reporting PASS
        because nothing contradicted the model would be the reassuring
        direction with no evidence behind it."""
        venue = StubVenue(_state(50_000.0, {}), _state(50_000.0, {}), [])
        result = _run(monkeypatch, venue)
        assert result.status == "UNCHECKABLE"
        assert "no isolated position" in result.detail

    def test_a_trade_during_the_window_aborts(self, monkeypatch):
        """A size change moves both balances for reasons unrelated to
        funding; attributing that to the venue's accounting would be the
        probe fooling itself."""
        venue = StubVenue(
            _state(50_000.0, {"SOL": 1_000.0}, sizes={"SOL": 1.0}),
            _state(49_000.0, {"SOL": 900.0}, sizes={"SOL": 2.0}),
            _funding("SOL", 10.0),
        )
        result = _run(monkeypatch, venue)
        assert result.status == "UNCHECKABLE"
        assert "book changed" in result.detail

    def test_a_payment_too_small_to_resolve_is_inconclusive(self, monkeypatch):
        """Sub-cent funding cannot be told from rounding in the reported
        balances, and a verdict either way would be arithmetic on noise."""
        venue = StubVenue(
            _state(50_000.0, {"SOL": 1_000.0}),
            _state(50_000.0, {"SOL": 999.999}),
            _funding("SOL", 0.001),
        )
        result = _run(monkeypatch, venue)
        assert result.status == "INCONCLUSIVE"
        assert "too small" in result.detail

    def test_a_missing_funding_endpoint_degrades_with_the_deltas_kept(self, monkeypatch):
        """`userFunding` is the ground truth. Without it there is no
        attribution, but the observed moves are still worth recording for a
        human to read."""
        venue = StubVenue(
            _state(50_000.0, {"SOL": 1_000.0}),
            _state(50_000.0, {"SOL": 990.0}),
            RuntimeError("404 unknown request type"),
        )
        result = _run(monkeypatch, venue)
        assert result.status == "UNCHECKABLE"
        assert "userFunding" in result.detail
        assert result.deltas["SOL"] == pytest.approx(-10.0)


class TestTickTiming:
    def test_the_wait_targets_the_top_of_the_hour(self):
        from risk_engine.market.probe_isolated_funding import _seconds_to_next_tick

        # 10 minutes past the hour -> 50 minutes to the next tick.
        assert _seconds_to_next_tick(3600.0 * 5 + 600.0) == pytest.approx(3000.0)
        assert 0 < _seconds_to_next_tick(3600.0 * 5 + 0.5) <= 3600.0

    def test_a_tick_beyond_the_limit_is_refused_rather_than_waited_out(self, monkeypatch):
        venue = StubVenue(
            _state(50_000.0, {"SOL": 1_000.0}),
            _state(50_000.0, {"SOL": 990.0}),
            _funding("SOL", 10.0),
        )
        monkeypatch.setattr(probe_mod, "InfoClient", lambda **kw: venue)
        monkeypatch.setattr(probe_mod, "_seconds_to_next_tick", lambda *a: 3_500.0)
        result = probe("0xtest", wait=True, max_wait_s=60.0)
        assert result.status == "UNCHECKABLE"
        assert "over the" in result.detail
