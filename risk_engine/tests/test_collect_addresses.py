"""The B4 address collector, driven by a stub WebSocket. No network.

Nothing in this repository has ever seen a Hyperliquid trade frame, so the
collector's parser is written against an assumption rather than a fixture.
That makes these tests unusual in what they are for: they cannot establish
that the collector reads the real feed correctly -- only the live venue can do
that -- so they establish the next best thing, which is that every way of
being *wrong* about the feed produces a loud failure instead of a plausible
file.

The failure being defended against is specific. A collector that connects to
the wrong URL, or subscribes with a name the venue acknowledges and never
delivers on, or reads a field that does not exist, and then writes a valid,
empty, confidently-framed address list, is worse than one that crashes: the
file loads without complaint, the nightly sweep snapshots nothing, and the
gate simply never advances. So the assertions below are mostly about
refusals, and the one on the happy path that matters most is the round trip
through `FileAddressSource` -- the consumer this output exists for, which
until now had no test of its own at all.

The mirror-image failure gets the same treatment. A refusal is only correct if
it refuses the *right* thing: aborting a 300-address harvest over one odd
record, or over a transient "Websocket request timed out", destroys a good
sample and then misdiagnoses it -- the abort text accuses B4's assumption of
not holding when it had just held 300 times. `TestAnomaliesAfterTheShapeHolds`
pins the boundary: fatal while nothing has been collected, counted and
published afterwards.

**Every frame, counter, timestamp and address in this file is
STUB-GENERATED.** `StubSocket` replays scripted text, `FakeClock` and
`FakeUtcNow` invent the window, and `_addr` invents the accounts. The live
venue is 403 at this environment's proxy (OPEN-QUESTIONS E5), so no example
here -- and no example in any document or review of this collector -- is a
capture. Anything that reads like one (round half-hour windows, five-figure
record counts) came out of these helpers, and a reader deciding whether the
message shape is confirmed has to look at a live run instead.
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from risk_engine.market import collect_addresses as collect
from risk_engine.market.collect_addresses import UnexpectedFeedShape, harvest
from risk_engine.shadow.providers import FileAddressSource

WINDOW_START = datetime(2026, 7, 30, 11, 0, 0, tzinfo=timezone.utc)

#: Captured before the autouse no-network fixture replaces it, so the one test
#: that asserts on the lazy import can still reach the real function.
_REAL_WEBSOCKETS_TRANSPORT = collect._websockets_transport


class StubClosed(RuntimeError):
    """Stands in for `websockets.exceptions.ConnectionClosed`."""


class StubSocket:
    """Replays scripted frames, then goes quiet exactly as a real feed does.

    Going quiet -- awaiting a sleep the collector's `wait_for` cancels --
    rather than raising is the important half. A stub that raised on
    exhaustion would end every run through the closed-connection branch and
    would never exercise the time-budget path, which is the path an operator's
    30-minute run actually takes.
    """

    def __init__(self, frames, closed_error=None):
        self.pending = list(frames)
        self._closed_error = closed_error
        self.sent: list[str] = []

    async def send(self, payload: str) -> None:
        self.sent.append(payload)

    async def recv(self):
        if self.pending:
            return self.pending.pop(0)
        if self._closed_error is not None:
            raise self._closed_error
        await asyncio.sleep(3600)


class FakeClock:
    """A monotonic clock that advances a fixed step per reading.

    The deadline is read off this rather than `time.monotonic` so the
    time-budget test finishes instantly and deterministically. The suite
    already carries one wall-clock assertion that fails intermittently on a
    loaded box; a 30-minute test, or a sleeping one, would be more of the same.
    """

    def __init__(self, step: float = 5.0) -> None:
        self.step = step
        self.now = 0.0

    def __call__(self) -> float:
        self.now += self.step
        return self.now


class FakeUtcNow:
    """Two readings: the collection window's start and its end."""

    def __init__(self, start: datetime = WINDOW_START, minutes: float = 30.0) -> None:
        self._times = [start, start + timedelta(minutes=minutes)]

    def __call__(self) -> datetime:
        return self._times.pop(0) if len(self._times) > 1 else self._times[0]


def _addr(i: int) -> str:
    return "0x" + f"{i:040x}"


def _trade(users, coin: str = "BTC", **extra) -> dict:
    """A trade record shaped the way the collector assumes trades are shaped.

    Carries the incidental fields a real trade would (px, sz, side, hash, tid)
    so that a parser reading the wrong key has somewhere wrong to read from,
    rather than being handed a record whose only key is the right one.
    """
    record = {
        "coin": coin,
        "side": "B",
        "px": "118250.0",
        "sz": "0.0142",
        "time": 1_785_500_000_000,
        "hash": "0x" + "ab" * 32,
        "tid": 4_312_887,
    }
    if users is not None:
        record["users"] = users
    record.update(extra)
    return record


def _liquidation() -> dict:
    """The record from the review: a trades frame carrying a liquidation.

    Plausible as a real message on this feed and it names no participant at
    all, which makes it the cheapest way for a good harvest to meet something
    the assumed shape does not describe. Built fresh per call so no test can
    mutate another test's fixture.
    """
    return {"coin": "BTC", "px": "118250.0", "sz": "0.0142", "liquidation": True}


def _frame(records, channel: str = "trades") -> str:
    return json.dumps({"channel": channel, "data": records})


def _ack(coin: str = "BTC") -> str:
    return json.dumps({
        "channel": "subscriptionResponse",
        "data": {"method": "subscribe", "subscription": {"type": "trades", "coin": coin}},
    })


def _transport(frames, closed_error=None):
    socket = StubSocket(frames, closed_error)

    @asynccontextmanager
    async def connect():
        yield socket

    return collect.Transport(connect=connect, closed_errors=(StubClosed,)), socket


class RefusingConnect:
    """A connection attempt that fails at the handshake, as a wrong URL does.

    A class rather than an `asynccontextmanager` that raises before its yield,
    because the whole point is that `__aenter__` never succeeds and a generator
    with unreachable code after the raise reads like a mistake.
    """

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def __aenter__(self):
        raise self._exc

    async def __aexit__(self, *exc_info) -> bool:
        return False


def _unreachable_transport(exc: BaseException):
    return collect.Transport(connect=lambda: RefusingConnect(exc), closed_errors=(StubClosed,))


def _run(frames, *, closed_error=None, coins=("BTC",), clock_step=5.0, minutes=30.0, **kwargs):
    """Run one collection over scripted frames; returns (result, socket, lines)."""
    transport, socket = _transport(frames, closed_error)
    lines: list[str] = []
    result = harvest(
        coins=coins,
        minutes=minutes,
        transport=transport,
        clock=FakeClock(clock_step),
        utcnow=FakeUtcNow(minutes=minutes),
        emit=lines.append,
        poll_interval_s=0.01,
        **kwargs,
    )
    return result, socket, lines


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Every test supplies its own transport; none may reach the wire."""
    monkeypatch.setattr(
        collect,
        "_websockets_transport",
        lambda *a, **kw: pytest.fail("the collector tried to open a real connection"),
    )


class TestCollecting:
    def test_it_reaches_the_target_and_stops_there(self):
        """The target is a stopping condition, not a cap applied at the end:
        an operator's 30-minute budget is an upper bound they should not have
        to wait out once the sample is large enough."""
        frames = [
            _frame([_trade([_addr(1), _addr(2)])]),
            _frame([_trade([_addr(3), _addr(4)])]),
            _frame([_trade([_addr(5), _addr(6)])]),
        ]
        result, socket, _ = _run(frames, target=4)

        assert result.addresses == (_addr(1), _addr(2), _addr(3), _addr(4))
        assert "target" in result.stopped_because
        assert result.trade_records == 2
        assert result.address_field == "users"
        # The third frame was never read: the run stopped on the target rather
        # than draining everything and truncating afterwards.
        assert len(socket.pending) == 1

    def test_it_subscribes_once_per_coin(self):
        frames = [_frame([_trade([_addr(1), _addr(2)], coin="ETH")])]
        _, socket, _ = _run(frames, coins=("BTC", "ETH", "SOL"), target=2)

        assert [json.loads(s) for s in socket.sent] == [
            {"method": "subscribe", "subscription": {"type": "trades", "coin": c}}
            for c in ("BTC", "ETH", "SOL")
        ]

    def test_several_trades_may_arrive_in_one_frame(self):
        """The `data` array is plural in the assumed shape, and a busy venue
        batches. Reading only the first record would quietly discard most of
        the sample while still producing a file that looks fine."""
        frames = [_frame([_trade([_addr(i), _addr(i + 1)]) for i in (1, 3, 5)])]
        result, _, _ = _run(frames, target=6)

        assert len(result.addresses) == 6
        assert result.trade_records == 3

    def test_a_singular_user_field_is_accepted_and_recorded(self):
        """`users` is the assumed name and `user` the plausible alternative;
        whichever was really found has to reach the frame string, or a reader
        of the address list cannot tell which shape the venue produced."""
        frames = [
            json.dumps({"channel": "trades", "data": [
                {"coin": "BTC", "px": "1", "sz": "1", "user": _addr(9)},
            ]}),
        ]
        result, _, _ = _run(frames, target=1)

        assert result.addresses == (_addr(9),)
        assert result.address_field == "user"
        assert "'user' field" in collect.build_frame(result, required=1)

    def test_progress_is_printed_while_collecting(self):
        """A 30-minute run with no output is indistinguishable from a hung
        one, and an operator watching it needs to see the address count move
        before deciding to wait or to kill it."""
        frames = [_frame([_trade([_addr(i), _addr(i + 1)])]) for i in range(1, 40, 2)]
        _, _, lines = _run(frames, target=40, progress_every_s=1.0)

        assert any("subscribed to 'trades'" in line for line in lines)
        progress = [line for line in lines if "addresses" in line and "trades" in line]
        assert len(progress) >= 2
        assert "/40 addresses" in progress[-1]

    def test_acknowledgements_and_heartbeats_are_ignored_not_fatal(self):
        frames = [_ack(), json.dumps({"channel": "pong"}),
                  _frame([_trade([_addr(1), _addr(2)])])]
        result, _, _ = _run(frames, target=2)

        assert len(result.addresses) == 2
        assert result.frames == 3


class TestNormalisation:
    def test_mixed_case_and_duplicates_fold_to_one_address(self):
        """One account must not become two journal identities. The journal
        counts DISTINCT address towards §3.3's gate and its rows cannot be
        edited afterwards, so a duplicate admitted here inflates the gate
        permanently -- and a feed reporting checksummed addresses while an
        operator's hand-curated file holds lowercase ones is exactly how that
        happens."""
        same = "ab" * 20
        frames = [
            _frame([_trade(["0x" + same.upper()])]),
            _frame([_trade(["0x" + same])]),
            _frame([_trade(["0X" + same.title()])]),
            _frame([_trade([_addr(7)])]),
        ]
        result, _, _ = _run(frames, target=2)

        assert result.addresses == ("0x" + same, _addr(7))
        assert result.unparseable_addresses == 0

    def test_malformed_values_are_counted_and_skipped_not_fatal(self):
        """One junk value in a live feed is dirty data. Aborting the run over
        it would trade a whole 30-minute sample for one bad row -- but the
        count has to be reported, because silently dropping values is how a
        sample ends up smaller than its own frame text claims."""
        frames = [
            _frame([_trade(["0xnothex" + "0" * 33, "", _addr(1)])]),
            _frame([_trade([_addr(2), "0x1234", 42, None])]),
            _frame([_trade([" " + _addr(3)])]),
        ]
        # Target deliberately unreachable, so the run has to survive every bad
        # value and end on its time budget rather than on an abort.
        result, _, _ = _run(frames, target=3, minutes=0.5)

        assert result.addresses == (_addr(1), _addr(2))
        assert "time budget elapsed" in result.stopped_because
        # not-hex, empty, too-short, an int, a None, and a leading space.
        assert result.unparseable_addresses == 6
        assert f"{result.unparseable_addresses} values" in collect.build_frame(result, 1)

    def test_whitespace_is_refused_rather_than_trimmed(self):
        """`normalise_address` refuses whitespace by design; the collector must
        not paper over that with a `.strip()` of its own, which would make two
        spellings behave the same in the one place that is meant to be strict."""
        frames = [_frame([_trade([_addr(1) + "\n", _addr(2)])])]
        result, _, _ = _run(frames, target=1)

        assert result.addresses == (_addr(2),)
        assert result.unparseable_addresses == 1


class TestLoudFailures:
    def test_a_trade_record_without_addresses_fails_loudly(self):
        """The central B4 assumption -- that a public trade names its
        participants -- is unverified. If it is wrong, the only acceptable
        outcome is an abort quoting the record: collecting nothing and
        reporting success would put an empty, unaccountable sample behind a
        published calibration score."""
        frames = [_frame([_trade(None, px="118000.0")])]
        with pytest.raises(UnexpectedFeedShape) as exc:
            _run(frames, target=2)

        message = str(exc.value)
        assert "no account address" in message
        assert "['users', 'user']" in message          # what was expected
        assert "118000.0" in message                    # what was received, verbatim
        assert "'coin'" in message and "'tid'" in message  # the keys it did have
        assert "B4" in message

    def test_frames_that_are_never_trades_fail_loudly(self):
        """A subscription that is acknowledged and never delivered looks
        exactly like a quiet market. Waiting out the budget and reporting an
        empty list would hide a wrong subscription name behind the weather."""
        frames = [_ack(), _ack("ETH"), json.dumps({"channel": "pong"})]
        with pytest.raises(UnexpectedFeedShape) as exc:
            _run(frames, target=2, clock_step=0.1, first_trade_grace_s=1.0)

        message = str(exc.value)
        assert "without one recognisable trade record" in message
        assert "subscriptionResponse" in message   # the frames it did get
        assert '"channel": "trades"' in message    # the frames it wanted

    def test_no_frames_at_all_fails_loudly_and_names_the_url(self):
        """The URL itself is unconfirmed -- `verify.py` records it as
        UNCHECKABLE -- so silence is at least as likely to mean the endpoint
        is wrong as it is to mean the venue is idle."""
        with pytest.raises(UnexpectedFeedShape) as exc:
            _run([], target=2, first_frame_grace_s=1.0, ws_url="wss://example.invalid/ws")

        message = str(exc.value)
        assert "NOTHING AT ALL" in message
        assert "wss://example.invalid/ws" in message
        assert "C4" in message

    def test_an_error_frame_aborts_instead_of_being_ignored(self):
        """The channel name is inference from C4's webData3 snippet, so a
        rejected subscription is a likely outcome and the venue's own words
        about it are worth more than the silence that would otherwise follow."""
        frames = [json.dumps({"channel": "error", "data": "Invalid subscription trades"})]
        with pytest.raises(UnexpectedFeedShape) as exc:
            _run(frames, target=2)

        assert "Invalid subscription trades" in str(exc.value)

    def test_a_field_that_never_holds_an_address_aborts(self):
        """Present but always junk is the wrong field, not dirty data.
        Tolerating it row by row would spend the whole budget to produce an
        empty list -- the per-value skip is for a stray bad row, not for a
        systematic mismatch."""
        frames = [_frame([_trade([f"trader-{i}", f"trader-{i}b"])]) for i in range(15)]
        with pytest.raises(UnexpectedFeedShape) as exc:
            _run(frames, target=100)

        message = str(exc.value)
        assert "refused by normalise_address" in message
        assert "not one address was collected" in message
        assert "trader-0" in message

    def test_an_empty_address_field_cannot_report_success(self):
        """The last line of defence. `users: []` on every record trips none of
        the shape checks -- the field is there, nothing is malformed -- and
        would otherwise produce a clean, empty, confidently-framed file."""
        frames = [_frame([_trade([])]) for _ in range(3)]
        with pytest.raises(UnexpectedFeedShape) as exc:
            _run(frames, target=2, clock_step=5.0, minutes=0.5)

        assert "collected no addresses at all" in str(exc.value)

    def test_a_closed_connection_ends_the_run_with_a_stated_reason(self):
        """Not a shape fault: whatever was collected is real. But it must not
        be reported as a completed window, because whether this venue needs an
        application-level keepalive is unrecorded and a run that keeps dying
        early is the symptom."""
        frames = [_frame([_trade([_addr(1), _addr(2)])])]
        result, _, _ = _run(frames, target=100, closed_error=StubClosed("1006"))

        assert len(result.addresses) == 2
        assert "closed the connection" in result.stopped_because
        assert "keepalive" in result.stopped_because


class TestTheGateAndTheWrite:
    def test_a_run_short_of_the_gate_is_refused_and_writes_nothing(self, tmp_path):
        """§3.3 needs 200 distinct addresses. Writing 2 of them would produce a
        file that loads cleanly and a 21-day window that could never have
        cleared the gate -- discovered three weeks late."""
        frames = [_frame([_trade([_addr(1), _addr(2)])])]
        result, _, _ = _run(frames, target=500, minutes=0.5)
        assert "time budget elapsed" in result.stopped_because

        out = tmp_path / "addresses.json"
        with pytest.raises(ValueError) as exc:
            collect.write_address_list(result, out)

        message = str(exc.value)
        assert "found 2 distinct addresses" in message
        assert "needs 200" in message
        assert "--allow-short" in message
        assert not out.exists()

    def test_allow_short_writes_it_but_the_frame_says_so(self, tmp_path):
        frames = [_frame([_trade([_addr(1), _addr(2)])])]
        result, _, _ = _run(frames, target=500, minutes=0.5)

        out = tmp_path / "addresses.json"
        payload = collect.write_address_list(result, out, allow_short=True)

        assert payload["addresses"] == [_addr(1), _addr(2)]
        assert "SHORT: 2 addresses is below" in payload["frame"]
        assert "cannot open the Phase 4 gate" in payload["frame"]
        # And it still loads, because a short list is a judgement call rather
        # than a malformed file.
        assert FileAddressSource(out).addresses() == [_addr(1), _addr(2)]

    def test_an_empty_list_is_refused_even_with_allow_short(self, tmp_path):
        """`--allow-short` means "I know this sample is small", not "write a
        file that loads cleanly and sweeps nothing". `FileAddressSource`
        accepts `addresses: []` without complaint, so this is the only place
        it can be stopped."""
        frames = [_frame([_trade([_addr(1)])])]
        result, _, _ = _run(frames, target=1)
        empty = collect.HarvestResult(
            addresses=(), coins=result.coins, ws_url=result.ws_url,
            started_at=result.started_at, ended_at=result.ended_at,
            stopped_because=result.stopped_because, frames=result.frames,
            trade_records=result.trade_records, unparseable_addresses=0,
            address_field="users", target=1,
        )
        with pytest.raises(ValueError) as exc:
            collect.write_address_list(empty, tmp_path / "a.json", allow_short=True)

        assert "empty address list" in str(exc.value)
        assert "sweep against it would snapshot nothing" in str(exc.value)

    def test_it_will_not_clobber_an_existing_list_unasked(self, tmp_path):
        frames = [_frame([_trade([_addr(1)])])]
        result, _, _ = _run(frames, target=1)
        out = tmp_path / "addresses.json"
        out.write_text("{}")

        with pytest.raises(ValueError, match="--force"):
            collect.write_address_list(result, out, allow_short=True, required=1)
        assert out.read_text() == "{}"

    def test_the_required_count_is_read_off_the_gate_not_copied(self):
        """A literal 200 here would silently stop matching `ShadowProgress` the
        day the gate moves, and the collector would write a list that fails at
        the far end of a 21-day window."""
        from risk_engine.shadow.journal import ShadowProgress

        gate = ShadowProgress(
            distribution_version="x", distinct_days=0,
            distinct_addresses=0, resolved_observations=0,
        )
        assert collect.gate_required_addresses() == gate.required_addresses

    def test_a_full_list_round_trips_through_FileAddressSource(self, tmp_path):
        """The consumer this whole module exists to feed. `FileAddressSource`
        had no test of its own before this one, which is how the committed
        `deploy/addresses.json` drifted from the template it was generated
        from without anything noticing."""
        frames = [_frame([_trade([_addr(2 * i), _addr(2 * i + 1)])]) for i in range(120)]
        result, _, _ = _run(frames, target=240, progress_every_s=60.0)
        assert len(result.addresses) == 240

        out = tmp_path / "addresses.json"
        payload = collect.write_address_list(result, out)

        source = FileAddressSource(out)
        assert source.addresses() == list(result.addresses)
        assert source.frame == payload["frame"]
        # Written the way `shadow init-addresses` writes its template, so a
        # generated list and a hand-filled one diff without a whitespace storm.
        assert out.read_text().endswith("}\n")
        assert '\n  "frame":' in out.read_text()

    def test_the_provenance_block_records_what_the_frame_asserts(self, tmp_path):
        """The frame is prose a human reads; these are the same facts as data,
        so a reader does not have to take the sentence's word for the counts."""
        frames = [_frame([_trade([_addr(1), _addr(2)])])]
        result, _, _ = _run(frames, target=2, coins=("BTC",))
        payload = collect.build_payload(result, required=2)

        provenance = payload["_provenance"]
        assert provenance["address_field"] == "users"
        assert provenance["subscription_type"] == "trades"
        assert provenance["addresses_collected"] == 2
        assert provenance["gate_required_addresses"] == 2
        assert provenance["window_start_utc"] == WINDOW_START.isoformat()
        assert provenance["coins"] == ["BTC"]


class TestTheFrame:
    @pytest.fixture
    def frame(self):
        frames = [_frame([_trade([_addr(i), _addr(i + 1)], coin="ETH")])
                  for i in range(1, 500, 2)]
        result, _, _ = _run(frames, target=250, coins=("BTC", "ETH", "SOL"),
                            progress_every_s=60.0)
        return collect.build_frame(result, required=200)

    def test_it_states_the_source_and_how_the_addresses_were_read(self, frame):
        assert "public trades WebSocket feed" in frame
        assert "wss://api.hyperliquid.xyz/ws" in frame
        assert "subscription type 'trades'" in frame
        assert "'users' field" in frame
        assert "250 distinct accounts" in frame

    def test_it_states_the_exact_utc_window_and_the_coins(self, frame):
        assert "2026-07-30T11:00:00Z to 2026-07-30T11:30:00Z UTC" in frame
        assert "BTC, ETH, SOL" in frame

    def test_it_states_the_activity_bias_in_both_directions(self, frame):
        assert "selects on trading activity" in frame
        assert "over-represents accounts that trade frequently" in frame
        assert "holds a position without touching it" in frame
        # Both sides of a trade are taken, so market makers enter on the same
        # footing as one-off takers; a reader has to be told that.
        assert "makes markets" in frame

    def test_it_records_why_the_leaderboard_was_rejected(self, frame):
        """The reason is the whole argument for this frame, and it is the part
        most likely to be lost when the score is published somewhere else."""
        assert "leaderboard" in frame
        assert "realised performance" in frame
        assert "Activity bias is awkward; performance bias is circular" in frame
        assert "B4" in frame

    def test_it_records_the_B2_tension_that_weakens_it(self, frame):
        """The cohort the gate is read from discards the most active accounts
        -- precisely what this frame selects for. Omitting that would make the
        frame a sales pitch rather than a sampling frame."""
        assert "book-unchanged cohort" in frame
        assert "B2" in frame
        assert "shrinks in a way correlated with how it was drawn" in frame
        assert "materially smaller than 250" in frame

    def test_it_admits_what_is_still_unverified(self, frame):
        assert "NOT VERIFIED" in frame
        assert "asserted at runtime" in frame
        assert "blocked at the proxy" in frame

    def test_it_says_the_runtime_assertion_is_a_floor_and_not_a_proof(self, frame):
        """Without this the NOT VERIFIED clause reads as though every frame was
        checked and passed, when what actually happened is that the check stops
        being fatal once the first address is read."""
        assert "a floor and not a proof" in frame
        assert "treated as confirmed" in frame

    def test_it_labels_the_repository_examples_as_stub_generated(self, frame):
        """A reader of a published score may well go looking for the example
        frames in this repository as evidence that the shape is right. They are
        not evidence -- the venue is 403 here and every one of them came out of
        this test file's helpers -- and the frame is the only place that reader
        is guaranteed to look."""
        assert "stub-generated" in frame
        assert "nothing there can be cited as evidence" in frame

    def test_thin_headroom_is_called_out_at_the_gate_boundary(self):
        """Exactly 200 addresses cannot clear a 200-address gate: the sweep
        drops flat and zero-equity accounts before any of them count."""
        frames = [_frame([_trade([_addr(2 * i), _addr(2 * i + 1)])]) for i in range(105)]
        result, _, _ = _run(frames, target=210, progress_every_s=60.0)

        frame = collect.build_frame(result, required=200)
        assert "THIN HEADROOM" in frame
        assert "may not yield 200" in frame

    def test_it_is_accepted_by_the_loader_that_refuses_a_blank_frame(self, tmp_path):
        """`FileAddressSource` refuses a list whose frame is blank, and that
        refusal is the reason the frame is generated rather than left to an
        operator who would type 'trades feed' and satisfy the check while
        stating nothing."""
        frames = [_frame([_trade([_addr(1), _addr(2)])])]
        result, _, _ = _run(frames, target=2)
        out = tmp_path / "a.json"
        collect.write_address_list(result, out, allow_short=True)

        loaded = json.loads(out.read_text())
        assert loaded["frame"].strip() == loaded["frame"]
        assert len(loaded["frame"]) > 500, "a frame this short is not a sampling frame"
        assert FileAddressSource(out).frame == loaded["frame"]


class TestTheDependencyAndTheCLI:
    def test_websockets_is_not_imported_until_it_is_needed(self, monkeypatch):
        """`requirements.txt` is numpy, scipy and pytest under the header
        'deliberately thin: numerics only', and `deploy/Dockerfile.engine`
        installs exactly that. A module-level import would make one
        operator-run command a hard dependency of the whole engine and would
        break the container image rather than this one job."""
        assert "websockets" not in sys.modules, (
            "the collector must not import websockets at module scope"
        )
        # None in sys.modules makes `import websockets` raise, whether or not
        # the package happens to be installed on the machine running the suite.
        monkeypatch.setitem(sys.modules, "websockets", None)
        with pytest.raises(ImportError) as exc:
            _REAL_WEBSOCKETS_TRANSPORT("wss://example.invalid/ws")

        message = str(exc.value)
        assert "pip install 'websockets>=12.0'" in message
        assert "Dockerfile.engine" in message

    def test_the_output_path_is_checked_before_the_collection_window(
        self, tmp_path, monkeypatch, capsys
    ):
        """An operator who mistypes --out or forgets --force must learn that in
        the first second, not after standing over a 30-minute run that then
        refuses to write."""
        monkeypatch.setattr(collect, "harvest", lambda **kw: pytest.fail(
            "the collector connected before validating --out"
        ))
        existing = tmp_path / "addresses.json"
        existing.write_text("{}")

        with pytest.raises(SystemExit):
            collect.main(["--out", str(existing), "--minutes", "30"])
        assert "pass --force" in capsys.readouterr().err

    def test_a_shape_mismatch_exits_two_and_a_short_run_exits_one(
        self, tmp_path, monkeypatch, capsys
    ):
        """Different exit codes because they need different responses: a short
        run wants a longer window, a shape mismatch means no run of this
        collector is currently trustworthy."""
        def _shape_fault(**kwargs):
            raise UnexpectedFeedShape("a trade record carried no account address")

        monkeypatch.setattr(collect, "harvest", _shape_fault)
        assert collect.main(["--out", str(tmp_path / "a.json")]) == 2
        assert "FEED SHAPE MISMATCH" in capsys.readouterr().out

        frames = [_frame([_trade([_addr(1), _addr(2)])])]
        short, _, _ = _run(frames, target=500, minutes=0.5)
        monkeypatch.setattr(collect, "harvest", lambda **kw: short)
        assert collect.main(["--out", str(tmp_path / "b.json")]) == 1
        assert "REFUSED" in capsys.readouterr().out
        assert not (tmp_path / "b.json").exists()

    def test_a_successful_run_writes_the_file_and_prints_the_frame(
        self, tmp_path, monkeypatch, capsys
    ):
        frames = [_frame([_trade([_addr(2 * i), _addr(2 * i + 1)])]) for i in range(101)]
        result, _, _ = _run(frames, target=202, progress_every_s=60.0)
        monkeypatch.setattr(collect, "harvest", lambda **kw: result)

        out = tmp_path / "addresses.json"
        assert collect.main(["--out", str(out), "--minutes", "30"]) == 0

        printed = capsys.readouterr().out
        assert "sampling frame: 202 distinct accounts" in printed
        assert FileAddressSource(out).addresses() == list(result.addresses)

    def test_the_cli_rejects_a_window_that_cannot_collect_anything(self, tmp_path):
        for argv in (["--minutes", "0"], ["--target", "0"], ["--coins", " ,, "]):
            with pytest.raises((SystemExit, ValueError)):
                collect.main(["--out", str(tmp_path / "a.json"), *argv])


class TestAnomaliesAfterTheShapeHolds:
    """Fatal while nothing has been collected; counted and published after.

    The asymmetry is the whole finding. Before the first address, an odd record
    is evidence that the assumed shape is wrong and aborting is the only safe
    move. After 300 addresses have come out of the assumed field, the venue has
    *demonstrated* the shape, and an abort both throws the sample away and
    tells the operator a falsehood about why.
    """

    def _three_hundred_then(self, extra_frames, **kwargs):
        """300 good addresses, then whatever oddity is being tested, then 2 more.

        Ending on two more good addresses rather than on the time budget so the
        run stops on its target deterministically -- and so the test proves the
        loop kept *collecting* after the anomaly, not merely that it survived it.
        """
        frames = [_frame([_trade([_addr(2 * i), _addr(2 * i + 1)])]) for i in range(1, 151)]
        frames += list(extra_frames)
        frames += [_frame([_trade([_addr(900), _addr(901)])])]
        return _run(frames, target=302, progress_every_s=600.0, **kwargs)

    def test_one_record_without_an_address_does_not_destroy_the_harvest(self):
        """300 addresses had already been read from 'users' when a single record
        arrived without it. Aborting there loses a good 30-minute sample and
        reports that B4's assumption "did not hold" -- which is false, it held
        300 times, and it sends an operator to edit TRADE_ADDRESS_FIELDS on
        evidence that says nothing of the kind."""
        result, _, lines = self._three_hundred_then([_frame([_liquidation()])])

        assert len(result.addresses) == 302
        assert "target" in result.stopped_because
        assert result.anomalies.records_without_address == 1
        assert result.anomalies.total == 1
        # Counted, never silent: the operator watching the run is warned as it
        # happens, and the published frame carries it as a named shortfall.
        assert any("WARNING: a trade record carried none of" in line for line in lines)
        frame = collect.build_frame(result, required=200)
        assert "ANOMALIES: 1 frame(s)/record(s)" in frame
        assert "liquidation" in frame            # the record, verbatim
        assert "lower bound" in frame

    def test_a_transient_error_frame_mid_run_does_not_destroy_the_harvest(self):
        """'Websocket request timed out' is a real Hyperliquid WS error string.
        Arriving at minute 29 of a delivering run it used to abort and report a
        rejected subscription -- an accusation the 41 000 trades already
        collected flatly contradict."""
        error = json.dumps({"channel": "error", "data": "Websocket request timed out"})
        result, _, lines = self._three_hundred_then([error])

        assert len(result.addresses) == 302
        assert result.anomalies.error_frames == 1
        assert any("WARNING: error frame from the feed" in line for line in lines)
        frame = collect.build_frame(result, required=200)
        assert "Websocket request timed out" in frame
        assert "1 error frame from the venue" in frame

    def test_the_same_error_frame_before_any_address_is_still_fatal(self):
        """The dangerous case is preserved exactly: an error frame with nothing
        collected is the most likely way this collector is wrong, because the
        channel name is inference from C4's webData3 snippet."""
        error = json.dumps({"channel": "error", "data": "Invalid subscription trades"})
        with pytest.raises(UnexpectedFeedShape) as exc:
            _run([error, _frame([_trade([_addr(1)])])], target=2)

        assert "before one address had been collected" in str(exc.value)

    def test_the_same_odd_record_before_any_address_is_still_fatal(self):
        with pytest.raises(UnexpectedFeedShape) as exc:
            _run([_frame([_liquidation()])], target=2)

        message = str(exc.value)
        assert "no account address" in message
        assert "none had been collected yet" in message
        assert "B4" in message

    def test_the_anomaly_share_of_records_is_stated_not_just_the_count(self):
        """A stray record and a second record layout on the same feed produce
        identical address lists and are not the same sample. If most records
        carry no address field, this list is a sample of the records the parser
        could read -- §10 forbids publishing that without saying so."""
        odd = [_frame([_liquidation()]) for _ in range(20)]
        result, _, _ = self._three_hundred_then(odd)

        assert result.anomalies.records_without_address == 20
        frame = collect.build_frame(result, required=200)
        assert "% of the" in frame
        assert "second record layout on the same feed" in frame

    def test_the_anomaly_counts_reach_the_provenance_as_data(self):
        """The frame is prose. A reader diffing two address lists needs the
        counts as numbers, without parsing English."""
        result, _, _ = self._three_hundred_then([_frame([_liquidation()])])
        anomalies = collect.build_payload(result, required=200)["_provenance"]["anomalies"]

        assert anomalies["total"] == 1
        assert anomalies["trade_records_without_address_field"] == 1
        assert anomalies["records_without_address_examples"], "no evidence retained"
        assert "liquidation" in anomalies["records_without_address_examples"][0]

    def test_the_progress_line_reports_anomalies_while_there_is_time_to_react(self):
        """The operator standing over the window is the only person who can
        still act on a count that climbs with every frame."""
        frames = [_frame([_trade([_addr(1), _addr(2)])])]
        frames += [_frame([_liquidation()]) for _ in range(4)]
        frames += [_frame([_trade([_addr(i), _addr(i + 1)])]) for i in range(11, 40, 2)]
        _, _, lines = _run(frames, target=40, progress_every_s=1.0)

        progress = [line for line in lines if "anomalies" in line and "frames" in line]
        assert progress, "the progress line does not mention anomalies at all"
        assert "4 anomalies" in progress[-1]


class TestTheEnvelopeIsTheTolerantHalf:
    """A guessed envelope must not feed a fatal check on the contents.

    `{"channel": "trades", ...}` is the venue labelling the frame; a bare array
    is this module inferring. The inferred half is the one that has to be
    tolerant, because a fatal reading of an inference produces an abort about a
    frame that was never evidence either way.
    """

    def test_a_bare_array_that_is_not_trades_is_not_a_shape_fault(self):
        """`[{"px": "1", "sz": "2"}]` has no channel and no address field. It
        used to become a fatal 'a trade record carried no account address ...
        B4 ... did not hold' -- an accusation about trades, made on a frame with
        no claim to be one."""
        frames = [json.dumps([{"px": "1", "sz": "2"}])]
        with pytest.raises(UnexpectedFeedShape) as exc:
            _run(frames, target=2, clock_step=0.1, first_trade_grace_s=1.0)

        message = str(exc.value)
        assert "without one recognisable trade record" in message
        assert "no account address" not in message
        assert "B4" not in message
        # Still quoted, because it is the only evidence about what did arrive.
        assert '"px": "1"' in message

    def test_a_bare_array_carrying_the_address_field_is_still_accepted(self):
        """Tolerance is not indifference. Where the frame itself supports the
        inference -- a dict with 'users' in it -- the bare envelope is read,
        because a venue can change its wrapper without changing what a trade is."""
        frames = [json.dumps([_trade([_addr(1), _addr(2)])])]
        result, _, _ = _run(frames, target=2)

        assert result.addresses == (_addr(1), _addr(2))
        assert result.address_field == "users"

    def test_a_labelled_trades_frame_without_the_field_is_still_a_shape_fault(self):
        """The other side of the same rule: when the venue calls it a trade, a
        missing address field really is B4's assumption failing."""
        with pytest.raises(UnexpectedFeedShape) as exc:
            _run([_frame([{"px": "1", "sz": "2"}])], target=2)

        assert "no account address" in str(exc.value)


class TestEvidenceForTheLoudAborts:
    def test_a_trades_frame_whose_data_holds_no_records_is_kept_as_evidence(self):
        """Nine decodable `{"channel": "trades", "data": ["0x.."]}` frames used
        to produce an abort whose RECEIVED section read 'nothing decodable':
        `[]` records is falsy but not None, so the frame was neither counted nor
        retained. A venue naming participants directly in `data` -- plausible,
        the field name is a guess -- lands exactly here, and the one thing worth
        reading was the thing discarded."""
        frames = [json.dumps({"channel": "trades", "data": [_addr(3), _addr(4)]})
                  for _ in range(9)]
        with pytest.raises(UnexpectedFeedShape) as exc:
            _run(frames, target=2, clock_step=0.1, first_trade_grace_s=1.0)

        message = str(exc.value)
        assert "nothing decodable" not in message
        assert _addr(3) in message                       # verbatim, quotable
        assert "entries inside a 'trades' frame that were not records" in message

    def test_undecodable_frames_reach_the_published_artifact(self):
        """`counters.undecodable` was incremented and read by nothing at all, so
        a run could take non-JSON off the wire all window and publish no hint of
        it."""
        frames = ["<html>403 Forbidden</html>", _frame([_trade([_addr(1), _addr(2)])])]
        result, _, _ = _run(frames, target=2)

        assert result.anomalies.undecodable_frames == 1
        payload = collect.build_payload(result, required=2)
        assert payload["_provenance"]["anomalies"]["undecodable_frames"] == 1
        assert "1 frame(s) that were not decodable JSON" in payload["frame"]

    def test_the_no_frames_abort_still_names_the_counts_it_has(self):
        """Belt and braces on the same failure: the abort quotes its counters
        even when it retained no example, so 'RECEIVED: nothing' is always
        accompanied by how many of what."""
        frames = [_ack(), json.dumps({"channel": "pong"})]
        with pytest.raises(UnexpectedFeedShape) as exc:
            _run(frames, target=2, clock_step=0.1, first_trade_grace_s=1.0)

        assert "2 frames that yielded no trade record" in str(exc.value)


class TestObservedVersusSubscribedCoins:
    def test_the_frame_states_the_coins_that_delivered_not_the_ones_asked_for(self):
        """`result.coins` is the --coins argument. Publishing it as observed can
        only ever overstate the breadth of the sample: subscribe BTC, ETH, SOL,
        have only BTC deliver, and the frame claimed accounts trading all three.
        Every record carries `coin`, so the honest set costs nothing."""
        frames = [_frame([_trade([_addr(2 * i), _addr(2 * i + 1)], coin="BTC")])
                  for i in range(1, 4)]
        result, _, _ = _run(frames, target=6, coins=("BTC", "ETH", "SOL"))

        assert result.coins == ("BTC", "ETH", "SOL")      # subscribed
        assert result.coins_observed == ("BTC",)          # observed
        frame = collect.build_frame(result, required=6)
        assert "6 distinct accounts observed trading BTC on" in frame
        assert "observed trading BTC, ETH, SOL" not in frame
        assert "subscriptions were sent for BTC, ETH, SOL" in frame
        assert "Nothing arrived for ETH, SOL" in frame

    def test_the_per_coin_record_counts_are_published(self):
        frames = [_frame([_trade([_addr(1), _addr(2)], coin="BTC"),
                          _trade([_addr(3), _addr(4)], coin="ETH"),
                          _trade([_addr(5), _addr(6)], coin="ETH")])]
        result, _, _ = _run(frames, target=6, coins=("BTC", "ETH"))

        assert dict(result.records_by_coin) == {"BTC": 1, "ETH": 2}
        provenance = collect.build_payload(result, required=6)["_provenance"]
        assert provenance["coins"] == ["BTC", "ETH"]                 # subscribed
        assert provenance["coins_observed"] == ["BTC", "ETH"]
        assert provenance["trade_records_by_coin"] == {"BTC": 1, "ETH": 2}
        assert "BTC 1, ETH 2 records" in collect.build_frame(result, required=6)

    def test_records_with_no_coin_field_make_the_counts_a_stated_lower_bound(self):
        """The coin field is as much an assumption as anything else here. If it
        is absent the observed set is unknown, and the frame must say unknown
        rather than fall back to the subscription list."""
        frames = [json.dumps({"channel": "trades", "data": [{"users": [_addr(1)]}]})]
        result, _, _ = _run(frames, target=1, coins=("BTC", "ETH"))

        assert result.coins_observed == ()
        assert result.records_without_coin == 1
        frame = collect.build_frame(result, required=1)
        assert "the coin per record was not readable" in frame
        assert "which of them actually delivered is UNKNOWN" in frame


class TestTheGateFloorAndTheRescueCopy:
    def _two_addresses(self):
        frames = [_frame([_trade([_addr(1), _addr(2)])])]
        result, _, _ = _run(frames, target=500, minutes=0.5)
        return result

    def test_required_cannot_be_used_to_lower_the_gate(self, tmp_path):
        """`required=2` used to do two damaging things at once: skip the refusal
        for a 2-address harvest, and make the published frame assert 'THIN
        HEADROOM: the gate counts 2 distinct addresses' while §3.3's gate is
        200. A caller may tighten this bar; lowering §3.3 from a keyword
        argument is what --allow-short is for."""
        out = tmp_path / "addresses.json"
        with pytest.raises(ValueError) as exc:
            collect.write_address_list(self._two_addresses(), out, required=2)

        assert "needs 200" in str(exc.value)
        assert not out.exists()

    def test_a_lowered_required_cannot_corrupt_the_published_frame(self, tmp_path):
        out = tmp_path / "addresses.json"
        payload = collect.write_address_list(
            self._two_addresses(), out, allow_short=True, required=2
        )

        assert "SHORT: 2 addresses is below §3.3's requirement of 200" in payload["frame"]
        assert "THIN HEADROOM" not in payload["frame"]
        assert payload["_provenance"]["gate_required_addresses"] == 200

    def test_required_may_still_be_raised_above_the_gate(self, tmp_path):
        """The floor only ever tightens. A caller who wants more headroom than
        §3.3 demands is asking for something reasonable."""
        frames = [_frame([_trade([_addr(2 * i), _addr(2 * i + 1)])]) for i in range(105)]
        result, _, _ = _run(frames, target=210, progress_every_s=60.0)

        with pytest.raises(ValueError, match="needs 400"):
            collect.write_address_list(result, tmp_path / "a.json", required=400)

    def test_a_refused_harvest_is_parked_rather_than_discarded(self, tmp_path):
        """A 199-address harvest used to appear in no file anywhere: the refusal
        printed counts and wrote nothing, so the fix was another 30-minute
        window. That is the same 'discover it after standing over the window'
        failure the pre-flight checks exist to prevent."""
        out = tmp_path / "addresses.json"
        with pytest.raises(ValueError) as exc:
            collect.write_address_list(self._two_addresses(), out)

        rescued = list(tmp_path.glob("addresses.json.refused-*"))
        assert len(rescued) == 1
        assert str(rescued[0]) in str(exc.value)
        assert not out.exists(), "the requested path must stay untouched"
        # Not a .json: it must not be swept up by a glob, or mistaken for the
        # file that was asked for, but it IS a loadable list -- being usable is
        # the entire point.
        assert not rescued[0].name.endswith(".json")
        assert FileAddressSource(rescued[0]).addresses() == [_addr(1), _addr(2)]

    def test_the_rescue_copy_says_in_its_frame_that_it_was_refused(self, tmp_path):
        """It loads cleanly, so a reader who finds it has to be told from the
        frame itself that nothing accepted this list."""
        out = tmp_path / "addresses.json"
        with pytest.raises(ValueError):
            collect.write_address_list(self._two_addresses(), out)

        frame = FileAddressSource(next(tmp_path.glob("*.refused-*"))).frame
        assert "NOT THE FILE THAT WAS ASKED FOR" in frame
        assert "2 addresses against §3.3's requirement of 200" in frame
        assert "SHORT: 2 addresses is below" in frame

    def test_two_refused_runs_do_not_overwrite_each_others_rescue(self, tmp_path):
        """The stamp is the collection window's start, so a second refused run
        cannot destroy the first one's addresses -- which would resurrect the
        bug in a subtler form."""
        first = self._two_addresses()
        second = replace(
            first,
            addresses=(_addr(7), _addr(8)),
            started_at=first.started_at + timedelta(hours=1),
        )
        out = tmp_path / "addresses.json"
        for result in (first, second):
            with pytest.raises(ValueError):
                collect.write_address_list(result, out)

        rescued = sorted(tmp_path.glob("addresses.json.refused-*"))
        assert len(rescued) == 2
        assert FileAddressSource(rescued[0]).addresses() == [_addr(1), _addr(2)]
        assert FileAddressSource(rescued[1]).addresses() == [_addr(7), _addr(8)]

    def test_an_existing_out_also_parks_the_harvest_before_refusing(self, tmp_path):
        """The CLI pre-flights this, but a library caller does not, and the
        addresses are just as real either way."""
        out = tmp_path / "addresses.json"
        out.write_text("{}")
        with pytest.raises(ValueError, match="--force"):
            collect.write_address_list(self._two_addresses(), out, allow_short=True)

        assert out.read_text() == "{}"
        assert len(list(tmp_path.glob("addresses.json.refused-*"))) == 1

    def test_a_rescue_that_cannot_be_written_still_reports_the_refusal(
        self, tmp_path, monkeypatch
    ):
        """The rescue runs on the way to raising, so a failure writing it must
        not replace 'you are 198 short of the gate' with an unrelated errno.
        The refusal is the more important of the two messages."""
        real_write = Path.write_text

        def _fail_the_rescue(self, *args, **kwargs):
            if ".refused-" in self.name:
                raise PermissionError(13, "Permission denied")
            return real_write(self, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", _fail_the_rescue)
        with pytest.raises(ValueError) as exc:
            collect.write_address_list(self._two_addresses(), tmp_path / "addresses.json")

        message = str(exc.value)
        assert "needs 200" in message
        assert "could not be parked" in message
        assert "Permission denied" in message

    def test_an_empty_result_has_nothing_to_rescue(self, tmp_path):
        """No file at all for a zero-address run: a loadable empty list is the
        exact artifact this module exists to keep out of the world."""
        result = self._two_addresses()
        empty = replace(result, addresses=())
        with pytest.raises(ValueError, match="empty address list"):
            collect.write_address_list(empty, tmp_path / "a.json", allow_short=True)

        assert list(tmp_path.iterdir()) == []


class TestNormalisationAtTheWriteBoundary:
    def test_a_hand_built_result_cannot_publish_a_count_the_loader_disagrees_with(
        self, tmp_path
    ):
        """`write_address_list` used to publish whatever it was handed, so a
        result holding one account under two spellings wrote
        `addresses_collected: 3` into a file `FileAddressSource` loads as 2 --
        and because that loader dedups on the same key, the disagreement would
        never surface as an error, only as a frame whose arithmetic is wrong."""
        same = "cd" * 20
        result = collect.HarvestResult(
            addresses=("0x" + same.upper(), "0x" + same, _addr(5)),
            coins=("BTC",), ws_url="wss://example.invalid/ws",
            started_at=WINDOW_START, ended_at=WINDOW_START + timedelta(minutes=30),
            stopped_because="a test built it by hand", frames=1, trade_records=1,
            unparseable_addresses=0, address_field="users", target=1,
        )

        assert result.addresses == ("0x" + same, _addr(5))
        out = tmp_path / "a.json"
        payload = collect.write_address_list(result, out, allow_short=True)
        assert payload["_provenance"]["addresses_collected"] == 2
        assert len(FileAddressSource(out).addresses()) == 2

    def test_a_malformed_address_is_refused_where_the_operator_is_standing(self):
        """Same boundary `normalise_address` already draws, moved as early as it
        will go: at construction, rather than at load time three weeks into a
        window."""
        with pytest.raises(ValueError, match="hex"):
            collect.HarvestResult(
                addresses=("0xnot-an-address",), coins=("BTC",), ws_url="wss://x/ws",
                started_at=WINDOW_START, ended_at=WINDOW_START, stopped_because="",
                frames=0, trade_records=0, unparseable_addresses=0,
                address_field="users", target=1,
            )


class TestExitCodesDoNotCollide:
    """Each code is a different next action, so two failures sharing one is a
    wrong diagnosis handed to whoever reads the shell."""

    def test_a_refused_connection_raises_rather_than_tracebacking(self):
        """The URL is this module's most-doubted fact. It spends a paragraph on
        the silent way of being wrong -- a host that accepts and never delivers
        -- and left the loud way as an unhandled OSError."""
        transport = _unreachable_transport(ConnectionRefusedError(111, "Connection refused"))
        with pytest.raises(collect.FeedUnreachable) as exc:
            harvest(coins=("BTC",), minutes=1.0, transport=transport,
                    clock=FakeClock(), utcnow=FakeUtcNow(), emit=lambda line: None,
                    ws_url="wss://api.hyperliquid.invalid/ws")

        message = str(exc.value)
        assert "could not open a WebSocket connection" in message
        assert "wss://api.hyperliquid.invalid/ws" in message
        assert "--ws-url" in message
        assert "UNCHECKABLE" in message

    def test_a_wrong_url_exits_three_not_one(self, tmp_path, monkeypatch, capsys):
        """Exit 1 means 'collected a list, refused to publish it', whose fix is
        a longer window -- useless advice for a collector that never opened a
        socket."""
        def _unreachable(**kwargs):
            raise collect.FeedUnreachable("could not open a WebSocket connection to wss://x")

        monkeypatch.setattr(collect, "harvest", _unreachable)
        assert collect.main(["--out", str(tmp_path / "a.json")]) == collect.EXIT_UNREACHABLE
        assert "COULD NOT CONNECT" in capsys.readouterr().out

    def test_a_missing_websockets_package_exits_three_with_the_install_line(
        self, tmp_path, monkeypatch, capsys
    ):
        def _missing(**kwargs):
            raise ImportError("... pip install 'websockets>=12.0' ...")

        monkeypatch.setattr(collect, "harvest", _missing)
        assert collect.main(["--out", str(tmp_path / "a.json")]) == collect.EXIT_UNREACHABLE
        assert "pip install 'websockets>=12.0'" in capsys.readouterr().out

    def test_a_failed_write_prints_the_harvest_instead_of_losing_it(
        self, tmp_path, monkeypatch, capsys
    ):
        """The same defect as F7, reached by a different door.

        A read-only mount, a full disk or a dangling symlink all pass every
        pre-flight check -- the pre-flight cannot know a write will succeed
        until it tries one -- and they surface as OSError from `write_text`,
        after the collection window has already been stood over. Only
        ValueError was caught, so the harvest died to a traceback with the
        addresses in memory and nothing on disk.

        Exit 5, not 1: EXIT_REFUSED tells an operator the sample was not good
        enough and to collect for longer, which is precisely the wrong advice
        when the sample is fine and sitting on stdout.
        """
        frames = [_frame([_trade([_addr(2 * i), _addr(2 * i + 1)])]) for i in range(101)]
        result, _, _ = _run(frames, target=202, progress_every_s=60.0)
        monkeypatch.setattr(collect, "harvest", lambda **kw: result)

        out = tmp_path / "addresses.json"
        real_write_text = Path.write_text

        def _read_only(self, *args, **kwargs):
            if self == out:
                raise OSError(30, "Read-only file system")
            return real_write_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", _read_only)
        assert collect.main(["--out", str(out), "--minutes", "30"]) == collect.EXIT_WRITE_FAILED

        printed = capsys.readouterr().out
        assert "COULD NOT WRITE" in printed
        assert "Read-only file system" in printed
        assert "NOT lost" in printed
        # The whole harvest is recoverable from stdout, not just a count of it.
        dumped = json.loads(printed[printed.index("{"):printed.rindex("}") + 1])
        assert dumped["addresses"] == list(result.addresses)
        assert dumped["frame"]
        assert not out.exists()

    def test_a_zero_length_window_is_a_usage_error_not_a_short_run(
        self, tmp_path, monkeypatch, capsys
    ):
        """--minutes 0 used to traceback out of `harvest` and exit 1, sending an
        operator to lengthen a window they had just set to zero."""
        monkeypatch.setattr(collect, "harvest", lambda **kw: pytest.fail(
            "the collector connected on a window that cannot collect anything"
        ))
        with pytest.raises(SystemExit) as exc:
            collect.main(["--out", str(tmp_path / "a.json"), "--minutes", "0"])

        assert exc.value.code == collect.EXIT_USAGE
        assert "--minutes must be positive" in capsys.readouterr().err

    def test_a_usage_error_does_not_claim_a_feed_shape_mismatch(self, tmp_path, capsys):
        """argparse exits 2 for everything, and 2 is 'the feed did not look the
        way I assume'. Left alone, a typo in a flag name sends an operator to
        TRADE_ADDRESS_FIELDS."""
        with pytest.raises(SystemExit) as exc:
            collect.main(["--out", str(tmp_path / "a.json"), "--not-a-flag"])

        assert exc.value.code == collect.EXIT_USAGE
        assert exc.value.code != collect.EXIT_FEED_SHAPE
        assert "unrecognized arguments" in capsys.readouterr().err

    def test_the_pre_flight_path_refusal_is_a_usage_error_too(self, tmp_path, capsys):
        existing = tmp_path / "addresses.json"
        existing.write_text("{}")
        with pytest.raises(SystemExit) as exc:
            collect.main(["--out", str(existing)])

        assert exc.value.code == collect.EXIT_USAGE
        assert "pass --force" in capsys.readouterr().err

    def test_a_directory_as_out_is_caught_before_the_collection_window(
        self, tmp_path, monkeypatch, capsys
    ):
        """--out <dir> --force passed validation, collected the full window and
        then died on an uncaught IsADirectoryError with the harvest in memory
        and nowhere to put it."""
        monkeypatch.setattr(collect, "harvest", lambda **kw: pytest.fail(
            "the collector connected with a directory as --out"
        ))
        with pytest.raises(SystemExit) as exc:
            collect.main(["--out", str(tmp_path), "--force"])

        assert exc.value.code == collect.EXIT_USAGE
        assert "is a directory" in capsys.readouterr().err

    def test_a_missing_parent_is_still_caught_and_names_the_directory(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            collect.main(["--out", str(tmp_path / "nope" / "a.json")])

        assert exc.value.code == collect.EXIT_USAGE
        assert "not an existing directory" in capsys.readouterr().err

    def test_the_summary_line_names_the_anomalies_the_run_counted(
        self, tmp_path, monkeypatch, capsys
    ):
        """The frame is long; this is the line an operator reads."""
        frames = [_frame([_trade([_addr(2 * i), _addr(2 * i + 1)])]) for i in range(101)]
        frames.insert(50, _frame([{"coin": "BTC", "px": "1", "liquidation": True}]))
        result, _, _ = _run(frames, target=202, progress_every_s=600.0)
        monkeypatch.setattr(collect, "harvest", lambda **kw: result)

        assert collect.main(["--out", str(tmp_path / "a.json")]) == collect.EXIT_OK
        printed = capsys.readouterr().out
        assert "NOTE: 1 anomaly was counted rather than aborted on" in printed
        assert "anomalies: 1 trade record carrying none of" in printed  # result.summary()


def test_the_module_is_importable_without_a_websocket_library():
    """Belt and braces on the lazy import: importing the collector must not
    require the dependency, because `python -m risk_engine.shadow` and the
    engine's own tests share this package."""
    assert Path(collect.__file__).name == "collect_addresses.py"
    assert "websockets" not in sys.modules
