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

import datetime as _dt

import numpy as np
import pytest

from risk_engine.market import info as verify_info
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
    @pytest.fixture
    def unconfirmed(self, monkeypatch):
        """The bound as it shipped before C1 closed.

        These tests are about what a QUIET HISTORY does or does not establish,
        and that question only has teeth while the citation is absent — with
        one, PASS is carried by the citation and the data merely fails to
        contradict it. Injecting the unconfirmed bound keeps them testing the
        §10-relevant half instead of quietly becoming duplicates of
        `test_the_shipped_bound_now_closes_it`.
        """
        monkeypatch.setattr(
            FundingBounds, "hyperliquid_confirmed",
            classmethod(lambda cls: FundingBounds.documented_default()),
        )

    def test_a_quiet_history_is_inconclusive_not_a_pass(self, unconfirmed):
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

    def test_an_inconclusive_clamp_still_blocks(self, unconfirmed):
        """C1 was a blocker, so "we could not tell" has to hold up the live
        path exactly as a failure would. A non-blocking maybe is how a
        blocker quietly stops blocking.

        Still asserted after C1 closed, because the property belongs to the
        INCONCLUSIVE verdict rather than to C1's status: the next bound added
        here starts unconfirmed, and it must block on the way in too.
        """
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

    def test_every_unreadable_type_is_reported_in_one_pass(self):
        """The harness used to stop at the first refusal, and that cost real
        time: two live rounds against the same account surfaced `send`, then
        `spotTransfer`, one per run — each needing a fix, a push, a pull and a
        re-run to reach the next. Ledger types are a long tail, and finding
        them one per round trip is the slowest possible way to find out.

        `net_external_flow` still raises on the first bad record, correctly: a
        resolver must not proceed on a partial read. This check has the
        opposite job — know everything before the clock starts."""
        client = StubClient(ledger=[
            self._row("deposit", "1000"),
            self._row("accountClassTransfer", "500"),   # directional, no toPerp
            self._row("internalTransfer", "700"),       # names neither side
        ])
        check = verify.check_external_flow(client, ADDRESS)
        assert check.status == FAIL
        assert check.evidence["unreadable_types"] == [
            "accountClassTransfer", "internalTransfer",
        ]
        # Both named in the text, not just counted -- the operator fixes from
        # this line, and a count sends them back to the report file.
        assert "accountClassTransfer" in check.detail
        assert "internalTransfer" in check.detail
        assert "single pass" in check.detail

    def test_a_classifiable_type_is_not_reported_as_unknown(self):
        """`internalTransfer` moved out of EXTERNAL_FLOW_SIGNS into its own
        table, and the `known` set here was not updated with it — so a type
        this build classifies correctly came back as "the venue returned a
        type this build cannot classify". Every table a type may live in has
        to be listed, and this asserts the list is complete."""
        from risk_engine.market.parse import (
            DEX_ROUTED_TYPES,
            EXTERNAL_FLOW_SIGNS,
            NON_FLOW_DELTA_TYPES,
            PERP_ADDRESS_ROUTED_TYPES,
        )

        every_table = (set(EXTERNAL_FLOW_SIGNS) | set(NON_FLOW_DELTA_TYPES)
                       | set(DEX_ROUTED_TYPES) | set(PERP_ADDRESS_ROUTED_TYPES))
        for kind in sorted(every_table):
            client = StubClient(ledger=[self._row(kind, "100")])
            check = verify.check_external_flow(client, ADDRESS)
            assert check.evidence.get("unknown_types") == [], (
                f"{kind} lives in a classification table but is reported unknown"
            )

    def test_one_bad_record_does_not_condemn_the_rest_of_its_type(self):
        """A type is only reported once, against the record that failed --
        and a well-formed record of a type that also has a bad one must not
        make the type look fine."""
        client = StubClient(ledger=[
            self._row("accountClassTransfer", "500", toPerp=True),   # readable
            self._row("accountClassTransfer", "900"),                # not
        ])
        check = verify.check_external_flow(client, ADDRESS)
        assert check.status == FAIL
        assert check.evidence["unreadable_types"] == ["accountClassTransfer"]
        # The FAILING record is the example, not the well-formed one that
        # happened to come first.
        assert "toPerp" not in check.evidence["examples"]["accountClassTransfer"]["delta"]

    def test_a_spot_transfer_no_longer_fails_the_check(self):
        """The 2026-07-31 live finding. An airdrop into a spot wallet is not a
        perp flow, so a ledger containing one is readable rather than a FAIL."""
        client = StubClient(ledger=[
            self._row("deposit", "1000"),
            {"time": self._row("deposit")["time"],
             "delta": {"type": "spotTransfer", "token": "UFART", "amount": "20.0",
                       "usdcValue": "4.9884", "user": "0x2000", "destination": ADDRESS}},
        ])
        check = verify.check_external_flow(client, ADDRESS)
        assert check.status == PASS
        assert check.evidence["net_flow_usd"] == pytest.approx(1000.0)

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
    def test_webdata3_unprobed_says_how_to_answer_it(self):
        check = verify.check_webdata3()
        assert check.status == UNCHECKABLE
        # It must name the flag, because "uncheckable" used to mean "this
        # harness cannot" and now means "you did not ask it to".
        assert "--probe-ws" in check.detail
        # And it must name --address too. Both subscriptions are keyed on
        # `user`; probed without one they are refused for that reason alone,
        # which reads exactly like "no such subscription" and did.
        assert "--address" in check.detail
        # Non-blocking, because nothing in this tree consumes either
        # subscription — not because of a planner that was never built.
        assert check.satisfied

    def test_isolated_funding_blocks_and_says_why_it_matters(self):
        check = verify.check_isolated_funding()
        assert check.status == UNCHECKABLE
        assert check.blocking and not check.satisfied
        assert "coupling term" in check.detail


class TestWebDataProbe:
    """C4 became a COMPARISON on 2026-08-03, and the reason is the finding.

    While it asked only "does `webData3` exist?", every answer was phrased
    against an assumption nobody had tested — "it does not matter, the shard
    planner uses `webData2`". Probing both found `webData2` refused by the
    live venue, with a well-formed `user`, in the exact payload shape
    `webData3` accepts. A one-sided question could not have found that.

    The probe is driven through a fake `_probe_subscription` here: the point
    under test is how the PAIR of results is read, and most of the
    combinations (one refused, both refused, silence, transport failure)
    cannot be produced on demand against a real venue.
    """

    ADDR = "0x" + "b" * 40

    @staticmethod
    def _patched(monkeypatch, results=None, raises=None):
        """Install a fake probe answering per subscription type.

        `results` maps a subscription type to `_probe_subscription`'s own
        return shape, `(accepted, detail)`, so the fake stands in for that
        function at its real contract rather than re-deriving it from frames.
        """
        asked: list = []

        def fake_probe(ws_url, sub_type, address, timeout_s):
            asked.append(sub_type)
            if raises is not None:
                return None, f"the probe itself failed: {type(raises).__name__}: {raises}"
            return (results or {}).get(sub_type, (None, "no stub for " + sub_type))

        monkeypatch.setattr(verify, "_probe_subscription", fake_probe)
        return asked

    def test_it_asks_about_both_not_just_the_new_one(self, monkeypatch):
        """The guard on the whole point of this rework."""
        asked = self._patched(monkeypatch, results={
            "webData3": (True, "acknowledged"), "webData2": (True, "acknowledged"),
        })
        verify.check_webdata3(address=self.ADDR, probe=True)
        assert sorted(asked) == ["webData2", "webData3"]

    def test_the_measured_result_reports_the_inversion(self, monkeypatch):
        """What the live venue actually said on 2026-08-03.

        §5.2 names `webData2`. The detail has to say plainly that building to
        that letter subscribes to something the venue rejects, because a PASS
        whose text merely says "webData3 exists" is how this went unnoticed.
        """
        self._patched(monkeypatch, results={
            "webData3": (True, "the venue acknowledged the webData3 subscription"),
            "webData2": (False, "the venue rejected it: 'Error parsing JSON'"),
        })
        check = verify.check_webdata3(address=self.ADDR, probe=True)
        assert check.status == PASS
        assert check.evidence["webData3_accepted"] is True
        assert check.evidence["webData2_accepted"] is False
        assert "§5.2" in check.detail and "rejects" in check.detail

    def test_without_an_address_it_refuses_to_conclude(self, monkeypatch):
        """The mistake this rework exists to prevent.

        Both subscriptions are keyed on `user`. Probed without one, both are
        refused for that reason alone — and a bare rejection reads exactly
        like "no such subscription". Reporting either way from that evidence
        is what produced a wrong refutation of C4 in the first place.
        """
        self._patched(monkeypatch, results={
            "webData3": (False, "the venue rejected it: 'Error parsing JSON'"),
            "webData2": (False, "the venue rejected it: 'Error parsing JSON'"),
        })
        check = verify.check_webdata3(probe=True)
        assert check.status == INCONCLUSIVE
        assert "missing" in check.detail and "`user`" in check.detail

    def test_both_refused_with_a_user_is_a_failure(self, monkeypatch):
        """Distinct from the case above, and the difference is the `user`.
        Both are documented; a venue refusing both WELL-FORMED subscribes
        means the documented shape moved, which is what FAIL is for."""
        self._patched(monkeypatch, results={
            "webData3": (False, "rejected"), "webData2": (False, "rejected"),
        })
        check = verify.check_webdata3(address=self.ADDR, probe=True)
        assert check.status == FAIL

    def test_the_spec_being_right_is_also_an_answer(self, monkeypatch):
        """Guard against a check that can only report the exciting result."""
        self._patched(monkeypatch, results={
            "webData3": (False, "rejected"),
            "webData2": (True, "the venue acknowledged the webData2 subscription"),
        })
        check = verify.check_webdata3(address=self.ADDR, probe=True)
        assert check.status == PASS
        assert "as §5.2 assumes" in check.detail

    def test_silence_is_inconclusive_not_a_rejection(self, monkeypatch):
        self._patched(monkeypatch, results={
            "webData3": (None, "neither acknowledged nor rejected"),
            "webData2": (True, "acknowledged"),
        })
        check = verify.check_webdata3(address=self.ADDR, probe=True)
        assert check.status == INCONCLUSIVE

    def test_a_transport_failure_does_not_masquerade_as_an_answer(self, monkeypatch):
        """DNS, TLS and a refused connection are all "we did not get an
        answer". Reporting any of them as "no such subscription" would record
        a fact about the venue that was never established."""
        self._patched(monkeypatch, raises=OSError("Name or service not known"))
        check = verify.check_webdata3(address=self.ADDR, probe=True)
        assert check.status == INCONCLUSIVE
        assert "probe itself failed" in check.detail

    def test_it_stays_non_blocking_however_it_answers(self, monkeypatch):
        """C4 must never hold up a live path — not because "the planner uses
        webData2", which was never a fact about code in this tree, but
        because nothing here consumes either subscription."""
        for res in (
            {"webData3": (True, "ok"), "webData2": (False, "no")},
            {"webData3": (False, "no"), "webData2": (False, "no")},
            {"webData3": (None, "quiet"), "webData2": (None, "quiet")},
        ):
            self._patched(monkeypatch, results=res)
            assert verify.check_webdata3(address=self.ADDR, probe=True).blocking is False

    def test_the_probe_is_off_unless_asked_for(self, monkeypatch):
        """A socket is a different kind of cost from a POST, and `websockets`
        is not an engine dependency. Default runs must not open one."""
        called: list = []
        monkeypatch.setattr(
            verify, "_probe_subscription",
            lambda *a, **k: called.append(a) or (True, "should not happen"),
        )
        verify.check_webdata3()
        assert called == []


class TestFrameLedgerSweep:
    """B2 across the sampling frame, because one address is not the cohort.

    The shadow window resolves 200-500 addresses daily for 21 days. An
    unclassifiable delta type on any one of them raises inside `resolve_due`,
    is classified TRANSIENT by name, and is retried forever — so it withholds
    that address's observations silently while the counter fails to advance.
    Found on day 6, it costs the window. Found by this sweep, it costs a
    minute.

    That this is not paranoia is the record: one real account produced five
    delta types, three of which had to be classified from live records, and
    two of those were found one run apart on the SAME address."""

    class _MultiClient(StubClient):
        def __init__(self, ledgers, errors=None, **kw):
            super().__init__(**kw)
            self._ledgers = ledgers
            self._errors = errors or {}

        def non_funding_ledger_updates(self, address, start_ms, end_ms=None):
            if address in self._errors:
                raise self._errors[address]
            return self._ledgers.get(address, [])

    @staticmethod
    def _row(kind, **extra):
        import time as _t
        return {"time": int(_t.time() * 1000) - HOUR_MS,
                "delta": {"type": kind, "usdc": "100", **extra}}

    def test_a_type_absent_from_the_first_address_is_still_found(self):
        """The whole reason this exists. Address A looks clean; the type that
        would break the window is on address B."""
        client = self._MultiClient({
            "0xa": [self._row("deposit")],
            "0xb": [self._row("brandNewKind")],
        })
        check = verify.check_frame_ledger_types(client, ["0xa", "0xb"], 50)
        assert check.status == FAIL
        assert check.evidence["unreadable_types"] == ["brandNewKind"]
        assert "brandNewKind" in check.evidence["examples"]

    def test_a_clean_frame_passes_and_names_what_it_did_not_exercise(self):
        """A PASS here is narrower than it looks: the known types this sample
        never contained are precisely the ones that surface on day 9. Naming
        them is the difference between 'checked' and 'checked, and here is
        what remains untested'."""
        client = self._MultiClient({
            "0xa": [self._row("deposit")], "0xb": [self._row("withdraw")],
        })
        check = verify.check_frame_ledger_types(client, ["0xa", "0xb"], 50)
        assert check.status == PASS
        unseen = check.evidence["known_types_not_exercised"]
        # The three that are filed as directional-needs-toPerp and have never
        # been seen live — the landmine this whole check is aimed at.
        assert "internalTransfer" in unseen
        assert "subAccountTransfer" in unseen
        assert "accountClassTransfer" in unseen
        assert "remain untested" in check.detail

    def test_one_dead_address_does_not_hide_the_rest(self):
        """Aborting on the first unreachable account would let a single dead
        address mask every type on the ones after it."""
        client = self._MultiClient(
            {"0xb": [self._row("brandNewKind")]},
            errors={"0xa": RuntimeError("422")},
        )
        check = verify.check_frame_ledger_types(client, ["0xa", "0xb"], 50)
        assert check.status == FAIL
        assert check.evidence["unreadable_types"] == ["brandNewKind"]
        assert check.evidence["addresses_unreachable"][0]["address"] == "0xa"
        assert check.evidence["addresses_read"] == 1

    def test_every_address_failing_is_reported_as_such(self):
        """Zero readable addresses is the endpoint or the list, not the
        classifier, and must not read as 'no bad types found'."""
        client = self._MultiClient({}, errors={"0xa": RuntimeError("422")})
        check = verify.check_frame_ledger_types(client, ["0xa"], 50)
        assert check.status == FAIL
        assert "not the classifier" in check.detail

    def test_the_sample_is_a_deterministic_prefix(self):
        """An operator re-running after a fix must see the same addresses. With
        a random draw, a type that vanished is indistinguishable from a type
        that was never sampled."""
        client = self._MultiClient({"0xa": [self._row("deposit")]})
        first = verify.check_frame_ledger_types(client, ["0xa", "0xb", "0xc"], 2)
        again = verify.check_frame_ledger_types(client, ["0xa", "0xb", "0xc"], 2)
        assert first.evidence["addresses_sampled"] == 2
        assert first.evidence["types_seen"] == again.evidence["types_seen"]

    def test_without_a_list_it_is_uncheckable_not_passed(self):
        check = verify.check_frame_ledger_types(self._MultiClient({}), [], 50)
        assert check.status == UNCHECKABLE
        assert not check.passed


class TestSelfInflictedLimitsNeverReadAsFail:
    """Audit F-3. `RateLimitExceeded` comes from our OWN WeightBudget before a
    byte leaves the process; only C2 and the frame sweep knew that. The other
    five venue-touching checks reported it through their generic `except → FAIL`
    — exit 2, "the model is wrong today", about a request that was never made.
    C1 escaped in the live run that exposed this only because the sweep's
    budget had partially refilled by the time it ran: luck, not code."""

    @staticmethod
    def _limited_client():
        from risk_engine.market.info import RateLimitExceeded

        class Limited(StubClient):
            def _boom(self, *a, **k):
                raise RateLimitExceeded("weight 20 exceeds remaining 0")
            meta = _boom
            candle_snapshot = _boom
            clearinghouse_state = _boom
            funding_history = _boom
            non_funding_ledger_updates = _boom
            post = _boom
        return Limited()

    def test_no_check_converts_a_self_limit_into_a_venue_verdict(self):
        client = self._limited_client()
        results = [
            verify.check_meta(client)[0],
            verify.check_candles(client, "BTC")[0],
            verify.check_clearinghouse(client, ADDRESS),
            verify.check_funding_clamp(client, ["BTC"], 30),
            verify.check_external_flow(client, ADDRESS),
            verify.check_basis(client, ["BTC"], 2, 0.0, 0.004),
        ]
        for check in results:
            assert check.status == UNCHECKABLE, f"{check.id}: {check.status}"
            assert not check.failed, check.id
            # The remedy is named, because "uncheckable" without a next move
            # is a dead end for the operator reading it. Asserted on the move
            # itself rather than on the word "budget": the message used to
            # say "re-run this check alone", which stopped being the remedy
            # once the pool became shared with the shadow jobs, and a
            # keyword-shaped assertion would not have noticed.
            assert "--frame-sample" in check.detail, check.id
            assert "docker compose ps" in check.detail, check.id

    def test_a_real_venue_error_still_fails(self):
        """The guard must not soften genuine failures: a 500 from the venue is
        exactly what FAIL exists for."""
        client = StubClient(ledger_error=RuntimeError("500 Internal Server Error"))
        assert verify.check_external_flow(client, ADDRESS).status == FAIL


class TestFundingClampCanActuallyClose:
    """C1 had two outcomes: FAIL and INCONCLUSIVE. There was no PASS.

    The INCONCLUSIVE text told an operator to "confirm the value from protocol
    documentation and record it as the `source` field" — and the check built
    its own `FundingBounds.documented_default()` and returned INCONCLUSIVE
    whenever nothing breached the cap. Recording the source changed nothing.
    The instruction was unactionable, and an assumption that cannot be closed
    is one that gets ignored rather than resolved.

    Same defect class as a counter pinned at zero and a diagnostic nothing
    calls: a mechanism whose success path does not exist.
    """

    def _client(self, rate=2.27e-05):
        # Realistic magnitude: the live worst over 30 days on this venue was
        # 2.27e-05/h, about 0.06% of the 0.04/h cap.
        return StubClient(rates=[rate, -rate, rate / 2])

    def test_an_unconfirmed_source_is_inconclusive_however_quiet_the_data(
        self, monkeypatch
    ):
        """Absence of a breach is not evidence for a protocol constant, and no
        volume of it becomes evidence.

        The bound shipped unconfirmed until 2026-08-03, so this used to hold
        with no setup. It now needs the unconfirmed bound injected — the
        principle is unchanged and still reachable, but a test that silently
        became a test of the *confirmed* path would have stopped guarding the
        thing it was written for.
        """
        from risk_engine.model.funding import FundingBounds

        monkeypatch.setattr(
            FundingBounds, "hyperliquid_confirmed",
            classmethod(lambda cls: FundingBounds.documented_default()),
        )
        check = verify.check_funding_clamp(self._client(), ["BTC"], 30)
        assert check.status == INCONCLUSIVE
        assert check.evidence["source_confirmed"] is False
        # It must name the call that closes it, not just ask for a "source".
        assert "from_protocol_source" in check.detail

    def test_the_shipped_bound_now_closes_it(self):
        """C1's success path, exercised on what actually ships.

        The two tests above drive injected bounds; this one drives none, so it
        fails if the shipped constructor ever loses its citation. That is the
        whole content of closing C1 — not that a confirmed bound *can* pass,
        which was already true, but that the one in the tree *is* one.
        """
        check = verify.check_funding_clamp(self._client(), ["BTC"], 30)
        assert check.status == PASS
        assert check.evidence["source_confirmed"] is True
        assert "hyperliquid.gitbook.io" in check.evidence["source"]

    def test_a_confirmed_source_plus_consistent_data_passes(self, monkeypatch):
        from risk_engine.model.funding import FundingBounds

        monkeypatch.setattr(
            FundingBounds, "hyperliquid_confirmed",
            classmethod(lambda cls: FundingBounds.from_protocol_source(
                0.04, "https://example.invalid/docs/funding#cap (read 2026-07-31)")),
        )
        check = verify.check_funding_clamp(self._client(), ["BTC"], 30)
        assert check.status == PASS
        assert check.evidence["source_confirmed"] is True

    def test_a_confirmed_source_does_not_excuse_a_breach(self, monkeypatch):
        """Confirmation says what the protocol specifies; it cannot say the
        venue obeys it. A confirmed-but-contradicted bound must fail LOUDER,
        not quieter — clamping reality away is the §10-forbidden direction."""
        from risk_engine.model.funding import FundingBounds

        monkeypatch.setattr(
            FundingBounds, "hyperliquid_confirmed",
            classmethod(lambda cls: FundingBounds.from_protocol_source(
                0.04, "https://example.invalid/docs/funding#cap (read 2026-07-31)")),
        )
        check = verify.check_funding_clamp(self._client(rate=0.5), ["BTC"], 30)
        assert check.status == FAIL
        assert "do not clamp" in check.detail

    def test_a_vague_citation_is_refused(self):
        """"Hyperliquid docs" is what the unconfirmed default already says.
        Accepting it as confirmation would let the flag be flipped without
        anyone reading anything."""
        from risk_engine.model.funding import FundingBounds

        for bad in ("Hyperliquid docs", "the docs", "confirmed", ""):
            with pytest.raises(ValueError, match=r"re-checkable|provenance"):
                FundingBounds.from_protocol_source(0.04, bad)

    def test_the_shipped_default_is_not_confirmed(self):
        """The value in the tree is carried on trust and must say so. This
        test fails the day someone flips it without a citation — which is the
        point of it existing."""
        from risk_engine.model.funding import FundingBounds

        assert FundingBounds.documented_default().confirmed is False


class TestRecordedFindings:
    """Verifications that happened outside a run, folded back in.

    The harness reported "3 assumption(s) still unconfirmed: C1, C2, C5" for
    days after C2 and C5 had been answered — by commands that cannot fit
    inside one run (C2 needs hours, C5 needs a funded account across a funding
    tick). A summary that overstates what is open invites redoing settled work
    and teaches an operator to discount the line, which is the same defect the
    READMEs had when they claimed the live API had never been reached.

    The danger is the opposite one: a recorded PASS is a claim that can outlive
    its evidence. These tests are mostly about the ways it must not be
    trusted."""

    @staticmethod
    def _finding(**over):
        from risk_engine.market.findings import RecordedFinding

        base = {
            "id": "C5", "status": PASS,
            "observed_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "command": "python -m risk_engine.market.probe_isolated_funding",
            "network": "testnet", "detail": "the pocket absorbed 100%",
            "evidence": {},
        }
        base.update(over)
        return RecordedFinding(**base)

    def test_a_recorded_result_fills_an_uncheckable_gap(self):
        merged = verify.apply_recorded(
            verify.check_isolated_funding(), self._finding()
        )
        assert merged.passed and merged.satisfied
        assert merged.evidence["recorded"]["network"] == "testnet"

    def test_a_recorded_pass_never_renders_as_a_live_pass(self):
        """The whole point. An operator skimming the output must be able to
        see, without opening the report, that this rests on history."""
        merged = verify.apply_recorded(
            verify.check_isolated_funding(), self._finding()
        )
        assert merged.status == "PASS (recorded)"
        assert merged.status != PASS
        assert "recorded 20" in merged.detail          # the date
        assert "probe_isolated_funding" in merged.detail  # the command
        assert "testnet" in merged.detail                 # what it is a claim about

    def test_live_data_outranks_a_recording(self):
        """A recording exists to fill a gap, not to answer a question the
        venue has already answered today."""
        live = verify.Check("C5", "q", PASS, "measured live just now")
        merged = verify.apply_recorded(live, self._finding())
        assert merged.status == PASS
        assert "recorded" not in merged.status

    def test_a_contradiction_is_surfaced_rather_than_hidden(self):
        """The one event this harness most needs to shout about: something
        that was true when recorded and is not true now."""
        live = verify.Check("C5", "q", FAIL, "the cross pool absorbed it")
        merged = verify.apply_recorded(live, self._finding(status=PASS))
        assert merged.failed
        assert "contradicts a recorded PASS" in merged.detail
        assert merged.evidence["contradicted_recording"] == PASS

    def test_a_stale_recording_vouches_for_nothing(self):
        """Venues change. A finding that cannot expire is a claim that becomes
        permanent by neglect."""
        from risk_engine.market.findings import STALE_AFTER_DAYS

        old = (_dt.datetime.now(_dt.timezone.utc)
               - _dt.timedelta(days=STALE_AFTER_DAYS + 1)).isoformat()
        merged = verify.apply_recorded(
            verify.check_isolated_funding(), self._finding(observed_utc=old)
        )
        assert merged.status == INCONCLUSIVE
        assert not merged.satisfied
        assert "no longer vouches" in merged.detail

    def test_a_recorded_fail_still_exits_two(self):
        """An out-of-band check that found the venue contradicting the model is
        exactly as disqualifying as an in-run one."""
        merged = verify.apply_recorded(
            verify.check_isolated_funding(), self._finding(status=FAIL)
        )
        assert merged.failed and not merged.satisfied

    def test_no_findings_file_is_normal_and_not_an_error(self, tmp_path):
        from risk_engine.market.findings import load_findings

        assert load_findings(tmp_path / "nope.json") == {}

    def test_a_finding_without_its_command_is_refused(self, tmp_path):
        """A verification whose reproduction is folklore is not evidence, and
        recording it would make it permanent."""
        import json as _json
        from risk_engine.market.findings import load_findings

        p = tmp_path / "v.json"
        p.write_text(_json.dumps({"C5": {
            "status": PASS, "observed_utc": "2026-07-31T00:00:00+00:00",
            "network": "testnet", "detail": "trust me",
        }}))
        with pytest.raises(ValueError, match="command"):
            load_findings(p)

    def test_a_typo_status_is_refused_not_silently_passed(self, tmp_path):
        """Audit F-6 (PoC-6): `Check.passed` matches by prefix so that
        'PASS (recorded)' counts — which meant 'PASSS', an operator's typo in
        a hand-edited file, satisfied a blocking check. Hand-edited means
        typos are the expected input, not the surprising one."""
        import json as _json
        from risk_engine.market.findings import load_findings

        for bad in ("PASSS", "pass", "ok", "PASSED"):
            p = tmp_path / f"v-{bad}.json"
            p.write_text(_json.dumps({"C5": {
                "status": bad, "observed_utc": "2026-07-31T00:00:00+00:00",
                "command": "x 1", "network": "testnet", "detail": "d",
            }}), encoding="utf-8")
            with pytest.raises(ValueError, match="not one of"):
                load_findings(p)

    def test_an_unparseable_date_is_refused(self, tmp_path):
        import json as _json
        from risk_engine.market.findings import load_findings

        p = tmp_path / "v.json"
        p.write_text(_json.dumps({"C5": {
            "status": PASS, "observed_utc": "last Tuesday", "command": "x",
            "network": "testnet", "detail": "d",
        }}))
        with pytest.raises(ValueError, match="ISO-8601"):
            load_findings(p)

    def test_the_shipped_file_loads_and_is_not_stale(self):
        """The file in the tree is read by every operator run, so a typo in it
        is a broken command for everyone. This also fails when a shipped
        finding ages out, which is the reminder to re-run it."""
        from risk_engine.market.findings import load_findings

        findings = load_findings()
        assert set(findings) >= {"C2", "C4", "C5"}
        for f in findings.values():
            assert not f.is_stale(), f"{f.id} recorded {f.observed_utc} is stale"
            assert f.command.strip() and f.network.strip()

    def test_the_shipped_c5_finding_states_it_is_testnet(self):
        """C5 was observed on testnet. That is evidence about testnet, and
        promoting it silently to a mainnet claim is the laundering this file
        exists to prevent."""
        from risk_engine.market.findings import load_findings

        c5 = load_findings()["C5"]
        assert c5.network == "testnet"
        assert "TESTNET" in c5.detail.upper()

    def test_the_shipped_c4_finding_records_both_halves(self):
        """C4's measurement is a COMPARISON, and the half nobody expected is
        the one that matters: `webData2` — the subscription §5.2 names — was
        rejected. A finding that recorded only "webData3 exists" would leave
        the next reader building against the broken one, which is exactly how
        this went unnoticed for as long as it did.
        """
        from risk_engine.market.findings import load_findings

        c4 = load_findings()["C4"]
        assert c4.network == "mainnet"
        assert "webData3" in c4.detail and "webData2" in c4.detail
        assert "§5.2" in c4.detail
        # Both directions in the evidence, not just the positive one.
        assert "accepted" in c4.evidence["webData3_with_user"]
        assert "rejected" in c4.evidence["webData2_with_user"]


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


class TestThisHarnessIsBackgroundTraffic:
    """§5.3 across the whole deployment, not per process (OPEN-QUESTIONS C6).

    C6 fixed the two shadow jobs and left this tool holding an interactive
    budget -- 1200/min, no reserve -- so a verification run during a sweep
    published a combined ceiling of 2400 against a venue limit of 1200. The
    ceiling was never reached, which is exactly why it survived: an
    accounting error nobody trips over is still an accounting error.
    """

    def test_it_takes_the_background_reserve_not_the_interactive_one(self, monkeypatch):
        from risk_engine.market.info import SHADOW_RESERVED_FRACTION

        monkeypatch.delenv("SHADOW_DSN", raising=False)
        budget = verify._verify_budget(None)
        assert budget.inner.reserved_fraction == SHADOW_RESERVED_FRACTION
        assert budget.available() == 300

    def test_a_postgres_journal_puts_it_in_the_shared_ledger(self, monkeypatch):
        """The point of the fix: same pool as the sweep and the resolver,
        rather than a third private window beside them."""
        seen = {}

        def _fake(target, reserved_fraction=0.0, actor="shadow"):
            seen.update(target=target, reserved_fraction=reserved_fraction, actor=actor)
            return verify_info.WeightBudget(reserved_fraction=reserved_fraction)

        import risk_engine.shadow.weight_ledger as ledger

        monkeypatch.setattr(ledger, "open_weight_budget", _fake)
        monkeypatch.setenv("SHADOW_DSN", "postgresql://u@db/shadow")
        verify._verify_budget(None)
        assert seen["target"] == "postgresql://u@db/shadow"
        assert seen["actor"] == "verify", (
            "the ledger records who spent what; 'shadow' would make a "
            "verification run indistinguishable from the cron in the table it "
            "shares with it"
        )

    def test_an_unreachable_ledger_verifies_anyway_and_says_so(self, monkeypatch, capsys):
        """A silent fallback would BE the C6 defect: believing you share a
        pool while holding a private one cannot be observed from inside."""
        import risk_engine.shadow.weight_ledger as ledger

        def _boom(*a, **k):
            raise OSError("could not translate host name 'db'")

        monkeypatch.setattr(ledger, "open_weight_budget", _boom)
        monkeypatch.setenv("SHADOW_DSN", "postgresql://u@db/shadow")
        budget = verify._verify_budget(None)
        out = capsys.readouterr().out
        assert budget.available() == 300, "still bounded, still background"
        assert "PRIVATE" in out and "db" in out

    def test_a_spent_window_is_waited_out_rather_than_reported(self):
        """One full run costs more than one minute's share -- a 50-address
        frame sweep alone is 1000 weight against 300/min -- so without
        pacing the reserve would turn most of a run into UNCHECKABLE results
        about requests that never left the process."""
        inner = verify_info.WeightBudget(reserved_fraction=0.75)
        inner.charge(300)
        paced = verify_info.PacedBudget(inner, max_wait_s=5.0, wait_s=0.0)
        # The window refills as the events age out; wait_s=0.0 spins instead
        # of sleeping, so this is fast without a fake clock.
        inner._events = [(t - 61.0, w) for t, w in inner._events]
        paced.charge(20)
        assert inner.spent() == 20

    def test_the_wait_has_a_ceiling(self):
        """A pool nobody releases is a real condition with an honest report
        (`_self_limited` → UNCHECKABLE), so the wait must end."""
        inner = verify_info.WeightBudget(reserved_fraction=0.75)
        inner.charge(300)
        paced = verify_info.PacedBudget(inner, max_wait_s=0.0, wait_s=0.0)
        with pytest.raises(verify_info.RateLimitExceeded):
            paced.charge(20)

    def test_a_frozen_clock_is_not_paced(self):
        """Deterministic replays and unit tests pass `now`; a window measured
        against a clock that does not move never refills, so waiting would
        spend the whole ceiling to reach the same refusal."""
        inner = verify_info.WeightBudget(reserved_fraction=0.75)
        paced = verify_info.PacedBudget(inner, max_wait_s=600.0, wait_s=600.0)
        paced.charge(300, now=1000.0)
        with pytest.raises(verify_info.RateLimitExceeded):
            paced.charge(20, now=1000.0)  # returns at once, or this test hangs

    def test_the_run_actually_uses_it(self, monkeypatch):
        """The budget only counts if it reaches the client that spends."""
        captured = {}

        class _Stub:
            def __init__(self, url=None, budget=None, **kw):
                captured["budget"] = budget

            def post(self, *a, **k):
                raise RuntimeError("stop here; the client is all this test wants")

        monkeypatch.delenv("SHADOW_DSN", raising=False)
        monkeypatch.setattr(verify, "InfoClient", _Stub)
        verify.run_all(None, ["BTC"], 1, 1, 0.0, False)
        assert isinstance(captured["budget"], verify_info.PacedBudget)
        assert captured["budget"].available() == 300


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


class TestItSaysWhatItIsDoing:
    """A run that shares one 300/min pool and waits out a spent window (C6)
    spends most of its wall clock asleep — a 200-address frame sweep is 4000
    weight, roughly twelve of its thirteen minutes waiting. Silent, that is
    indistinguishable from a hang, which is the confusion the shadow sweep
    had to be given progress output to fix. Same failure, same remedy.
    """

    def test_progress_goes_to_stderr_so_a_report_pipe_stays_clean(self, capsys):
        verify._step("E5.1 meta")
        out = capsys.readouterr()
        assert "E5.1 meta" in out.err
        assert out.out == ""

    def test_each_check_announces_itself_before_it_runs(self, monkeypatch, capsys):
        """Before, not after: the point is to name the step you are waiting
        on, and a line printed on completion arrives once the waiting is
        already over."""
        monkeypatch.delenv("SHADOW_DSN", raising=False)
        order: list[str] = []

        def _spy(label):
            order.append(label)

        monkeypatch.setattr(verify, "_step", _spy)
        monkeypatch.setattr(verify, "InfoClient", lambda **kw: StubClient())
        monkeypatch.setattr(verify, "check_meta", lambda c: (
            verify.Check("E5.1", "q", PASS, "ok"), {"BTC": object()}))
        monkeypatch.setattr(verify, "check_candles", lambda c, coin: (
            verify.Check("E5.2", "q", PASS, "ok"), np.zeros(10)))
        for name in ("check_clearinghouse", "check_external_flow",
                     "check_funding_clamp"):
            monkeypatch.setattr(verify, name,
                                lambda *a, **k: verify.Check("X", "q", PASS, "ok"))
        monkeypatch.setattr(verify, "check_basis",
                            lambda *a, **k: verify.Check("C2", "q", PASS, "ok"))

        verify.run_all(None, ["BTC"], 30, 12, 5.0, False)
        joined = " | ".join(order)
        for expected in ("E5.1", "E5.2", "E5.3", "C1", "C2"):
            assert expected in joined, joined

    def test_the_frame_sweep_reports_waiting_apart_from_elapsed(self):
        """Those two numbers are the diagnosis, and one combined number
        cannot give it: mostly-waiting is §5.3 working as designed, elapsed
        climbing while waiting does not is a stall."""
        import inspect

        src = inspect.getsource(verify.check_frame_ledger_types)
        assert "waited_s" in src
        assert "elapsed" in src and "waiting" in src

    def test_the_paced_budget_counts_what_it_slept(self):
        """The number the line above reports has to be real."""
        inner = verify_info.WeightBudget(limit_per_minute=100, reserved_fraction=0.0)
        inner.charge(100)
        paced = verify_info.PacedBudget(inner, max_wait_s=0.05, wait_s=0.01)
        with pytest.raises(verify_info.RateLimitExceeded):
            paced.charge(1)
        assert paced.waited_s > 0.0


class TestTheRecordedFindingsReachTheContainer:
    """The mechanism was built, tested, and inert where it is used.

    `market/verify.py` reads `docs/hl-risk/VERIFIED.json`, resolved relative
    to the package's own parent. The engine image copied `risk_engine/` and
    nothing else, so inside a container the file did not exist —
    `load_findings` returns {} for a missing file (correct: a fresh checkout
    has none) and every containerised run reported C2 and C5 as unconfirmed
    while the measurements that closed them sat in the repository.

    Observed 2026-08-04 in a live run whose C2 line read plain INCONCLUSIVE
    with no `(recorded)` suffix.
    """

    def test_the_image_copies_the_file_verify_reads(self):
        import pathlib as _p

        root = _p.Path(__file__).resolve().parents[2]
        dockerfile = (root / "deploy/Dockerfile.engine").read_text(encoding="utf-8")
        assert "COPY docs/hl-risk/VERIFIED.json" in dockerfile

    def test_the_copied_path_is_the_one_the_code_resolves(self):
        """Guard on the guard: copying the file to the wrong place inside the
        image would satisfy the test above and change nothing. `WORKDIR /app`
        plus `risk_engine/` at /app/risk_engine puts the default at
        /app/docs/hl-risk/VERIFIED.json, so the COPY destination must be the
        same repo-relative path."""
        import pathlib as _p

        from risk_engine.market.findings import DEFAULT_FINDINGS_PATH

        root = _p.Path(__file__).resolve().parents[2]
        assert DEFAULT_FINDINGS_PATH == root / "docs/hl-risk/VERIFIED.json"
        dockerfile = (root / "deploy/Dockerfile.engine").read_text(encoding="utf-8")
        assert "COPY docs/hl-risk/VERIFIED.json docs/hl-risk/VERIFIED.json" in dockerfile
        assert "WORKDIR /app" in dockerfile

    def test_an_empty_load_is_reported_rather_than_assumed(self, capsys, tmp_path):
        """A missing file must stay legal and stop being silent. Silence is
        what let this run for months: an empty result reads identically to
        'nothing was ever recorded'."""
        missing = tmp_path / "nope.json"
        rc = verify.main(["--coins", "BTC", "--findings", str(missing)])
        out = capsys.readouterr().out
        assert "no recorded findings" in out
        assert str(missing) in out
        assert rc in (0, 1, 2)  # the venue is unreachable here; the print is the point
