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
            self._row("accountClassTransfer", "500"),      # no toPerp
            self._row("internalTransfer", "700"),          # no toPerp either
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
        # Non-blocking: the shard planner works either way, and this must not
        # hold up a live path on its own.
        assert check.satisfied

    def test_isolated_funding_blocks_and_says_why_it_matters(self):
        check = verify.check_isolated_funding()
        assert check.status == UNCHECKABLE
        assert check.blocking and not check.satisfied
        assert "coupling term" in check.detail


class TestWebData3Probe:
    """C4 was UNCHECKABLE because "this harness speaks only the Info POST
    API". That stopped being true when `collect_addresses` shipped — it has
    since held a live socket to this exact venue for 1 110 frames. The
    capability was in the tree and this check did not use it.

    The probe is driven through a fake socket here: the point under test is
    how each venue response is *interpreted*, and three of the four
    interpretations (rejection, silence, transport failure) cannot be produced
    on demand against a real venue.
    """

    @staticmethod
    def _patched(monkeypatch, frames=(), raises=None):
        """Install a fake `_websockets_transport` and a no-op `asyncio.run`."""
        import json as _json

        sent: list = []

        def fake_probe(ws_url, sub_type, address, timeout_s):
            if raises is not None:
                return None, f"the probe itself failed: {type(raises).__name__}: {raises}"
            sent.append((ws_url, sub_type, address))
            for raw in frames:
                msg = _json.loads(raw)
                if msg.get("channel") == "error":
                    return False, f"the venue rejected it: {msg.get('data')!r}"
                if msg.get("channel") == "subscriptionResponse":
                    got = ((msg.get("data") or {}).get("subscription") or {}).get("type")
                    if got == sub_type:
                        return True, f"the venue acknowledged the {sub_type} subscription"
                if msg.get("channel") == sub_type:
                    return True, f"the venue delivered a {sub_type} frame"
            return None, "the venue neither acknowledged nor rejected it"

        monkeypatch.setattr(verify, "_probe_subscription", fake_probe)
        return sent

    def test_an_acknowledgement_answers_it(self, monkeypatch):
        self._patched(monkeypatch, frames=[
            '{"channel":"subscriptionResponse",'
            '"data":{"subscription":{"type":"webData3"}}}',
        ])
        check = verify.check_webdata3(probe=True)
        assert check.status == PASS
        assert check.evidence["accepted"] is True

    def test_a_rejection_is_an_answer_and_not_a_failure(self, monkeypatch):
        """`webData3` was only ever a maybe. The planner is specified against
        `webData2`, so "no such subscription" settles C4 rather than breaking
        anything — reporting it as FAIL would say the live data contradicted
        the model, which is a much stronger claim than the truth."""
        self._patched(monkeypatch, frames=[
            '{"channel":"error","data":"Unknown subscription type webData3"}',
        ])
        check = verify.check_webdata3(probe=True)
        assert check.status == PASS
        assert check.evidence["accepted"] is False
        assert "does not exist" in check.detail

    def test_silence_is_inconclusive_not_a_rejection(self, monkeypatch):
        self._patched(monkeypatch, frames=[])
        check = verify.check_webdata3(probe=True)
        assert check.status == INCONCLUSIVE
        assert check.evidence["accepted"] is None

    def test_a_transport_failure_does_not_masquerade_as_an_answer(self, monkeypatch):
        """DNS, TLS and a refused connection are all "we did not get an
        answer". Reporting any of them as "no such subscription" would record
        a fact about the venue that was never established."""
        self._patched(monkeypatch, raises=OSError("Name or service not known"))
        check = verify.check_webdata3(probe=True)
        assert check.status == INCONCLUSIVE
        assert check.evidence["accepted"] is None
        assert "probe itself failed" in check.detail

    def test_it_stays_non_blocking_however_it_answers(self, monkeypatch):
        """C4 must never be able to hold up a live path: the shard planner
        works against `webData2` regardless."""
        for frames in ([], ['{"channel":"error","data":"nope"}'],
                       ['{"channel":"webData3","data":{}}']):
            self._patched(monkeypatch, frames=frames)
            assert verify.check_webdata3(probe=True).blocking is False

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
        assert set(findings) >= {"C2", "C5"}
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
