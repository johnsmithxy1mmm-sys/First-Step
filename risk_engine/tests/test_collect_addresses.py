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
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
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


def test_the_module_is_importable_without_a_websocket_library():
    """Belt and braces on the lazy import: importing the collector must not
    require the dependency, because `python -m risk_engine.shadow` and the
    engine's own tests share this package."""
    assert Path(collect.__file__).name == "collect_addresses.py"
    assert "websockets" not in sys.modules
