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
    """Two scripted `clearinghouseState` reads plus a funding ledger.

    `user_funding` filters by the REQUESTED range, exactly as the real API
    does. An earlier stub returned the whole ledger regardless, which is how
    the payment-window defect (audit P-1) survived its own tests: the probe
    over-widened the window by an hour and the stub could not show it.
    """

    def __init__(self, before: dict, after: dict, funding: list | Exception):
        self._states = [before, after]
        self._funding = funding
        self.calls = 0
        self.funding_requests: list[tuple[int, int | None]] = []

    def clearinghouse_state(self, address, is_agent_address=False):
        state = self._states[min(self.calls, len(self._states) - 1)]
        self.calls += 1
        return state

    def user_funding(self, address, start_ms, end_ms=None):
        if isinstance(self._funding, Exception):
            raise self._funding
        self.funding_requests.append((start_ms, end_ms))
        # A row stamped None is "the tick inside the observed window" -- the
        # venue materialises it mid-window at query time. Stamping such rows
        # with time.time() at fixture-construction instead races the probe's
        # own start_ms by microseconds and loses depending on test order.
        out = []
        for r in self._funding:
            t = r["time"]
            if t is None:
                t = (start_ms + (end_ms if end_ms is not None else start_ms)) // 2
            if start_ms <= t <= (end_ms if end_ms is not None else t):
                out.append({**r, "time": t})
        return out


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
    # None -> "the tick inside the observed window"; StubVenue materialises
    # it mid-window at query time. An explicit stamp is for payments that
    # deliberately sit OUTSIDE the window, like the prior tick in the P-1
    # regression.
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

    def test_cross_debit_is_detected_through_upnl_noise(self, monkeypatch):
        """Audit P-2, the PoC turned assertion. The cross account value
        carries every cross position's unrealised PnL, so over the probe's
        multi-minute window its drift dwarfs a funding payment. The earlier
        verdict demanded cross_delta ~= payment and therefore could never
        return FAIL on a real account -- the structurally dangerous world was
        the undetectable one. The verdict must read the pocket: the account
        paid, the pocket did not, and there is no third bucket."""
        venue = StubVenue(
            _state(50_000.0, {"SOL": 1_000.0}),
            # $10 cross-debited funding buried in -$400 of uPnL drift.
            _state(50_000.0 - 10.0 - 400.0, {"SOL": 1_000.0}),
            _funding("SOL", 10.0),
        )
        result = _run(monkeypatch, venue)
        assert result.status == "FAIL"
        assert "did not move" in result.detail or "not taken" in result.detail
        # The noisy cross figure is reported as corroboration, not hidden.
        assert "corroboration" in result.detail

    def test_the_previous_ticks_payment_does_not_poison_attribution(self, monkeypatch):
        """Audit P-1, the PoC turned assertion. Any account that can answer
        C5 held its isolated position through the PREVIOUS hourly tick, so a
        payment exists shortly before the probe starts -- and it is already
        inside the before-snapshot's balances. Widening the userFunding
        request beyond the observed window doubles the payment sum while the
        balance delta stays single, and every realistic run then reads
        ambiguous. The window must be [start, end], nothing more."""
        now = int(time.time() * 1000)
        venue = StubVenue(
            _state(50_000.0, {"SOL": 1_000.0}),
            _state(50_000.0, {"SOL": 990.0}),
            _funding("SOL", -10.0, when_ms=now - 35 * 60 * 1000)  # prior tick
            + _funding("SOL", -10.0),               # observed tick, mid-window
        )
        result = _run(monkeypatch, venue)
        assert result.status == "PASS", result.detail
        assert result.payments["SOL"] == pytest.approx(-10.0), (
            "the prior tick's payment leaked into the attribution window"
        )
        # And the request itself must not reach back an hour.
        (start_ms, _), = venue.funding_requests
        assert start_ms > now - 5 * 60 * 1000

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

    def test_a_silent_pocket_is_the_violation_itself(self, monkeypatch):
        """The account provably paid $10 (userFunding) and the pocket's
        collateral did not move at all. Wherever the payment lands in the
        cross side's noise, the pocket NOT paying is what §1.1 forbids --
        there is no third bucket. An earlier version filed this as
        inconclusive, which let the dangerous world hide behind the cross
        side's uPnL drift."""
        venue = StubVenue(
            _state(50_000.0, {"SOL": 1_000.0}),
            _state(50_000.0, {"SOL": 1_000.0}),
            _funding("SOL", 10.0),
        )
        result = _run(monkeypatch, venue)
        assert result.status == "FAIL"

    def test_a_partial_absorption_is_ambiguous_not_rounded_to_a_verdict(self, monkeypatch):
        """The pocket moved by half the payment: outside both bands. A margin
        transfer or venue rounding landed in the window, and resolving it in
        either direction would be arithmetic on a coincidence."""
        venue = StubVenue(
            _state(50_000.0, {"SOL": 1_000.0}),
            _state(50_000.0, {"SOL": 995.0}),
            _funding("SOL", 10.0),
        )
        result = _run(monkeypatch, venue)
        assert result.status == "INCONCLUSIVE"

    def test_a_two_cent_payment_is_still_resolvable(self, monkeypatch):
        """Kills the surviving MIN_RESOLVABLE_USD mutant: the threshold is a
        cent, and a payment of two cents that the pocket fully absorbed must
        produce a verdict, not fall through the resolvability filter."""
        venue = StubVenue(
            _state(50_000.0, {"SOL": 1_000.0}),
            _state(50_000.0, {"SOL": 999.98}),
            _funding("SOL", 0.02),
        )
        result = _run(monkeypatch, venue)
        assert result.status == "PASS"


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
