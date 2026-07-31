"""The verification harness, driven by a stub venue.

A checker that only ever runs against a live API is itself unverified code,
and it is unverified in the worst place: the one command whose job is to say
whether the model may be trusted. These tests drive it with responses whose
answer is known, including responses that should make it fail.

The property that matters most is the one it would be easiest to get wrong
in the reassuring direction: a clamp that nothing exceeded must not report
PASS. Absence of a counter-example is not confirmation, and on this
particular bound -- the one that truncates the funding tail -- laundering it
into confirmation is the §10-forbidden direction.
"""

from __future__ import annotations

import numpy as np
import pytest

from risk_engine.market import verify
from risk_engine.market.verify import FAIL, INCONCLUSIVE, PASS, UNCHECKABLE
from risk_engine.model.funding import FundingBounds

HOUR_MS = 3_600_000

#: A well-formed account address. `check_clearinghouse` refuses anything else
#: locally rather than letting `InfoClient`'s normalisation surface as a FAIL
#: about the venue, so the fixture has to be a real one.
ADDRESS = "0x" + "a" * 40

META = {
    "universe": [
        {"name": "BTC", "szDecimals": 5, "maxLeverage": 40, "marginTableId": 1},
        {"name": "ETH", "szDecimals": 4, "maxLeverage": 25, "marginTableId": 1},
    ],
    "marginTables": [
        [1, {"marginTiers": [
            {"lowerBound": "0", "maxLeverage": 40},
            {"lowerBound": "150000000", "maxLeverage": 20},
        ]}],
    ],
}


class StubClient:
    """Answers the four calls the harness makes, with whatever we choose."""

    def __init__(self, *, rates=None, basis=0.0, candles=True, meta=META,
                 ledger=None, ledger_error=None):
        self._meta = meta
        self._rates = rates if rates is not None else [1e-5, -2e-5, 3e-5]
        self._basis = basis
        self._candles = candles
        self._ledger = ledger if ledger is not None else []
        self._ledger_error = ledger_error

    def non_funding_ledger_updates(self, address, start_ms, end_ms=None):
        if self._ledger_error is not None:
            raise self._ledger_error
        return self._ledger

    def meta(self):
        return self._meta

    def candle_snapshot(self, coin, interval, start_ms, end_ms):
        if not self._candles:
            return []
        rng = np.random.default_rng(7)
        closes = 100_000.0 * np.exp(np.cumsum(rng.normal(0, 0.004, 500)))
        t0 = end_ms - 500 * HOUR_MS
        return [{"t": t0 + i * HOUR_MS, "c": f"{c:.4f}"} for i, c in enumerate(closes)]

    def funding_history(self, coin, start_ms, end_ms=None):
        return [
            {"time": start_ms + i * HOUR_MS, "fundingRate": str(r)}
            for i, r in enumerate(self._rates)
        ]

    def clearinghouse_state(self, address, is_agent_address=False):
        return {
            "marginSummary": {"accountValue": "100000.0", "totalRawUsd": "0"},
            "crossMarginSummary": {"accountValue": "100000.0"},
            "assetPositions": [],
        }

    def post(self, payload, weight=20):
        assert payload["type"] == "metaAndAssetCtxs"
        mid = 100_000.0
        mark = mid * (1.0 + self._basis)
        return [
            self._meta,
            [{"markPx": str(mark), "midPx": str(mid)},
             {"markPx": str(mark), "midPx": str(mid)}],
        ]


class TestFundingClamp:
    def test_a_quiet_history_is_inconclusive_not_a_pass(self):
        """The whole point. Nothing exceeded the clamp; that is not evidence
        the clamp is right, and reporting PASS would turn the absence of a
        counter-example into a confirmation."""
        check = verify.check_funding_clamp(StubClient(), ["BTC"], days=30)
        assert check.status == INCONCLUSIVE
        assert "confirms nothing" in check.detail
        assert check.evidence["observed_max_abs_rate"] == pytest.approx(3e-5)

    def test_an_observation_above_the_clamp_fails(self):
        cap = FundingBounds.documented_default().cap_per_hour
        check = verify.check_funding_clamp(
            StubClient(rates=[1e-5, cap * 1.5]), ["BTC"], days=30
        )
        assert check.status == FAIL
        assert check.evidence["breaches"][0]["observed"] == pytest.approx(cap * 1.5)
        # It must name the fix, and the fix is not "clamp reality away".
        assert "fix the bound" in check.detail

    def test_an_inconclusive_clamp_still_blocks(self):
        """C1 is a blocker, so "we could not tell" has to hold up the live
        path exactly as a failure would. A non-blocking maybe is how a
        blocker quietly stops blocking."""
        check = verify.check_funding_clamp(StubClient(), ["BTC"], days=30)
        assert check.blocking
        assert not check.satisfied


class TestBasis:
    def test_a_basis_above_a_quarter_of_an_hourly_move_fails(self):
        # §1.4's threshold is 25% of a typical hourly move; 0.004 hourly vol
        # makes that 0.001, so a 0.5% basis is five times over.
        check = verify.check_basis(
            StubClient(basis=0.005), ["BTC"], samples=2, interval_s=0.0,
            hourly_vol=0.004,
        )
        assert check.status == FAIL
        # The failure must say what the engine actually does -- feed a trade
        # price into a mark-price condition with no basis term -- rather than
        # name a guard to un-set. An earlier version pointed at an
        # `UnmeasuredBasis` class that exists nowhere in the engine, which
        # would send a reader looking for a switch instead of building the
        # missing term.
        assert "no basis term" in check.detail
        assert "silently" in check.detail

    def test_a_small_basis_over_minutes_is_still_inconclusive(self):
        """§1.4 asks for a median over a real window. Twelve samples over a
        minute is not that window, however clean the answer looks."""
        check = verify.check_basis(
            StubClient(basis=1e-5), ["BTC"], samples=2, interval_s=0.0,
            hourly_vol=0.004,
        )
        assert check.status == INCONCLUSIVE
        assert "not the window" in check.detail

    def test_without_a_volatility_reference_it_refuses_to_judge(self):
        check = verify.check_basis(
            StubClient(basis=1e-5), ["BTC"], samples=1, interval_s=0.0,
            hourly_vol=None,
        )
        assert check.status == INCONCLUSIVE

    def test_a_malformed_pair_is_a_failure_not_a_crash(self):
        class Broken(StubClient):
            def post(self, payload, weight=20):
                return {"unexpected": "shape"}

        check = verify.check_basis(Broken(), ["BTC"], 1, 0.0, 0.004)
        assert check.status == FAIL
        assert "documented" in check.detail

    def test_misaligned_universe_and_contexts_fail_rather_than_truncate(self):
        """Audit F-5: zip silently truncated to the shorter side, so a
        response with 3 universe entries and 1 context reported INCONCLUSIVE
        for BTC alone and dropped ETH and SOL without a trace. A checker that
        drops assets without saying so reports 'checked' for assets it never
        saw — the length mismatch IS the schema drift this harness hunts."""
        class Misaligned(StubClient):
            def post(self, payload, weight=20):
                return [
                    {"universe": [{"name": "BTC"}, {"name": "ETH"}, {"name": "SOL"}]},
                    [{"markPx": "100", "midPx": "100"}],
                ]

        check = verify.check_basis(Misaligned(), ["BTC", "ETH", "SOL"], 1, 0.0, 0.004)
        assert check.status == FAIL
        assert "misaligned" in check.detail
        assert check.evidence == {"n_universe": 3, "n_ctxs": 1}


class TestParsers:
    def test_meta_that_parses_passes(self):
        check, specs = verify.check_meta(StubClient())
        assert check.status == PASS
        assert set(specs) == {"BTC", "ETH"}
        assert check.evidence["n_tiered"] == 2

    def test_meta_that_does_not_parse_fails_with_the_reason(self):
        check, specs = verify.check_meta(StubClient(meta={"universe": "not a list"}))
        assert check.status == FAIL
        assert specs is None

    def test_an_empty_universe_fails_rather_than_passing_with_zero_assets(self):
        check, _ = verify.check_meta(StubClient(meta={"universe": [], "marginTables": []}))
        assert check.status == FAIL

    def test_candles_that_parse_report_the_measured_volatility(self):
        check, returns = verify.check_candles(StubClient(), "BTC")
        assert check.status == PASS
        assert returns is not None
        assert check.evidence["hourly_vol"] == pytest.approx(0.004, rel=0.2)

    def test_an_empty_candle_snapshot_fails(self):
        check, returns = verify.check_candles(StubClient(candles=False), "BTC")
        assert check.status == FAIL
        assert returns is None

    def test_the_book_parser_is_unchecked_without_an_address(self):
        """Not PASS. The parser the whole product reads a user's book through
        cannot be called verified because nobody supplied an address."""
        check = verify.check_clearinghouse(StubClient(), None)
        assert check.status == UNCHECKABLE
        assert not check.satisfied

    def test_the_book_parser_passes_on_a_real_response(self):
        # A well-formed address, because that is what the check now requires
        # and what a real run supplies. The old "0xabc" documented the
        # unguarded contract: against `StubClient`, which does not normalise,
        # it reported PASS -- so the test asserted that E5.3 could be *passed*
        # by an address the live client would refuse to send.
        check = verify.check_clearinghouse(StubClient(), ADDRESS)
        assert check.status == PASS

    def test_a_malformed_address_is_uncheckable_not_a_contradiction(self):
        """FAIL is reserved for the venue.

        `InfoClient` normalises before it sends, so a malformed `--address`
        raises inside `check_clearinghouse`'s try and used to come back as
        E5.3 FAIL, blocking -- which in this tool's own vocabulary means live
        data contradicted a documented shape and exits 2 with "the model is
        wrong today". Nothing was contradicted; no request was made. The
        assumption is unverified either way, so it stays blocking.
        """
        check = verify.check_clearinghouse(StubClient(), "0xabc")
        assert check.status == UNCHECKABLE
        assert check.blocking and not check.satisfied
        assert "40 hex digits" in check.detail
        assert "not a contradiction" in check.detail

    def test_a_malformed_address_never_reaches_the_client(self):
        """The guard sits ahead of the request, not around it."""
        class Exploding(StubClient):
            def clearinghouse_state(self, address, is_agent_address=False):
                raise AssertionError("a malformed address reached the client")

        check = verify.check_clearinghouse(Exploding(), "0xnot-an-address")
        assert check.status == UNCHECKABLE


class TestExternalFlow:
    """B2, and the reason it is checked BEFORE the shadow clock starts.

    Without a readable ledger the resolver fails every row, `_permanent_reason`
    classifies that as transient, and a live pilot accumulates fourteen days of
    snapshots against a gate that can never advance. This check exists to turn
    that into a one-command failure beforehand.
    """

    def _row(self, kind, usdc="1000", **extra):
        # Mid-window: the check asks for the last 90 days, so "now" is safe.
        import time as _t
        return {"time": int(_t.time() * 1000) - HOUR_MS,
                "delta": {"type": kind, "usdc": usdc, **extra}}

    def test_a_readable_ledger_passes_and_reports_the_net(self):
        client = StubClient(ledger=[
            self._row("deposit", "50000"), self._row("withdraw", "20000"),
        ])
        check = verify.check_external_flow(client, ADDRESS)
        assert check.status == PASS
        assert check.evidence["net_flow_usd"] == pytest.approx(30_000.0)

    def test_an_unknown_delta_type_fails_and_quotes_a_full_record(self):
        """The live finding this was written for: a real account returned a
        `send` type nobody here had seen. A count alone would say the type
        exists and nothing about its fields, and a sign guessed from a name
        already went wrong once -- `accountClassTransfer` needed a `toPerp`
        flag this repo could not have invented."""
        client = StubClient(ledger=[
            self._row("deposit", "1000"),
            self._row("someNewKind", "500", destination="0xabc"),
        ])
        check = verify.check_external_flow(client, ADDRESS)
        assert check.status == FAIL
        assert check.evidence["unknown_types"] == ["someNewKind"]
        example = check.evidence["examples"]["someNewKind"]
        # The whole record, not a summary -- the fields are the point.
        assert example["delta"]["destination"] == "0xabc"
        assert example["delta"]["usdc"] == "500"
        assert "evidence.examples" in check.detail

    def test_a_quiet_account_is_inconclusive_not_a_pass(self):
        """An empty list is what a genuinely quiet account and a wrong request
        type look like alike, so it cannot confirm the shape."""
        check = verify.check_external_flow(StubClient(ledger=[]), ADDRESS)
        assert check.status == INCONCLUSIVE
        assert "never exercised" in check.detail

    def test_an_unreadable_record_fails_rather_than_scoring_zero(self):
        """A directional transfer with no direction: returning 0.0 would hide
        a real transfer, which is the §10-forbidden direction here."""
        client = StubClient(ledger=[self._row("accountClassTransfer", "1000")])
        check = verify.check_external_flow(client, ADDRESS)
        assert check.status == FAIL
        assert "toPerp" in check.detail

    def test_an_endpoint_error_is_a_failure_naming_the_consequence(self):
        client = StubClient(ledger_error=RuntimeError("422 Unprocessable Entity"))
        check = verify.check_external_flow(client, ADDRESS)
        assert check.status == FAIL
        assert "cannot resolve any observation" in check.detail

    def test_without_an_address_it_is_unchecked_rather_than_passed(self):
        check = verify.check_external_flow(StubClient(), None)
        assert check.status == UNCHECKABLE
        assert not check.satisfied


class TestUncheckable:
    def test_webdata3_says_what_would_answer_it(self):
        check = verify.check_webdata3()
        assert check.status == UNCHECKABLE
        assert "webData3" in check.detail
        # Non-blocking: the shard planner works either way, and this must not
        # hold up a live path on its own.
        assert check.satisfied

    def test_isolated_funding_blocks_and_says_why_it_matters(self):
        check = verify.check_isolated_funding()
        assert check.status == UNCHECKABLE
        assert check.blocking and not check.satisfied
        assert "coupling term" in check.detail


class TestExitCodes:
    """The exit code is the whole interface for a CI job or a deploy script,
    and conflating "contradicted" with "unconfirmed" would let a wrong model
    through on a retry."""

    def _run(self, monkeypatch, checks):
        monkeypatch.setattr(verify, "run_all", lambda *a, **k: checks)
        return verify.main(["--coins", "BTC"])

    def test_contradiction_exits_two(self, monkeypatch):
        c = verify.Check("C1", "q", FAIL, "contradicted")
        assert self._run(monkeypatch, [c]) == 2

    def test_unconfirmed_exits_one(self, monkeypatch):
        c = verify.Check("C1", "q", INCONCLUSIVE, "cannot tell")
        assert self._run(monkeypatch, [c]) == 1

    def test_all_confirmed_exits_zero(self, monkeypatch):
        c = verify.Check("E5.1", "q", PASS, "held")
        assert self._run(monkeypatch, [c]) == 0

    def test_a_non_blocking_unchecked_item_does_not_hold_up_the_run(self, monkeypatch):
        checks = [verify.Check("E5.1", "q", PASS, "held"), verify.check_webdata3()]
        assert self._run(monkeypatch, checks) == 0


def test_the_harness_reaches_the_live_host_or_says_it_cannot():
    """E5 itself, exercised rather than asserted.

    This is the one test that touches the network. Where the API is
    reachable it is a real smoke test of the live path; where it is not --
    which is every environment this was written in -- it skips with the
    reason, so "the checker was never run against the venue" stays visible
    in the test output instead of being a fact only the docs remember.
    """
    from risk_engine.market.info import InfoClient

    client = InfoClient()
    try:
        check, specs = verify.check_meta(client)
    except Exception as exc:  # any transport failure means the same thing here
        pytest.skip(f"api.hyperliquid.xyz unreachable: {type(exc).__name__}: {exc}")
    if check.status != PASS:
        pytest.skip(f"live meta did not parse: {check.detail}")
    assert specs and len(specs) > 10
