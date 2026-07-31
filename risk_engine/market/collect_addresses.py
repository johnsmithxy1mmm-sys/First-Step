"""Harvest the shadow address list from the public trades feed (OPEN-QUESTIONS B4).

    python -m risk_engine.market.collect_addresses --minutes 30 --out addresses.json

§3.3 needs 200-500 active public addresses to snapshot, and the Info API
enumerates none of them: it reads any address handed to it and returns no
endpoint that lists accounts. B4 named two candidate sources. The product
owner chose this one, the public trades feed, and rejected the leaderboard
for a specific reason worth restating because it is the whole argument: the
leaderboard ranks on realised performance, and realised performance is the
variable the calibration score measures. Sampling on the outcome would make
the model look badly calibrated on whichever tail the sample excluded, and no
amount of care downstream recovers from that.

The trades feed is not neutral either -- nothing available is -- it is biased
on a variable that is merely *awkward* rather than circular. It selects on
trading activity, and §3.3's gate is read off the book-unchanged cohort
(OPEN-QUESTIONS B2), which discards accounts whose book moved during the
observation day. So this frame selects for the accounts that cohort filter
then throws away. That tension is real, it is not fixable here, and it is
written into every frame string this collector generates rather than left for
a reader of the published score to discover.

**The message shape is an assumption, and it is checked at runtime.** Nothing
in this repository has ever spoken Hyperliquid's WebSocket protocol. The URL
comes from one UNCHECKABLE note in `verify.py`, the subscribe envelope from a
snippet in OPEN-QUESTIONS C4 that has never been executed here (the API is
403 at this environment's proxy), and the claim that a public trade carries
its participants' addresses is exactly what B4 says "need[s] to be verified".
So every one of those facts is asserted while collecting, and a mismatch
raises `UnexpectedFeedShape` quoting the frame verbatim.

**Those assertions are fatal only until the first address is read.** This is
the one asymmetry in the module worth stating twice, because getting it wrong
in either direction is a real failure. Before anything has been collected, an
odd record or an error frame is evidence that the assumed shape is wrong, and
aborting is the only safe response. *After* an address has been read out of
one of `TRADE_ADDRESS_FIELDS`, the assumption is no longer an assumption: the
venue has demonstrated it, and a later anomaly is one bad record on a feed
whose shape has been confirmed hundreds of times. Aborting there destroys a
good harvest and -- worse -- misdiagnoses it, telling an operator that "the
B4 assumption did not hold" when it had just held 300 times and sending them
to edit `TRADE_ADDRESS_FIELDS` on evidence that says nothing of the kind. So
past that point anomalies are counted, warned about on the progress stream as
they happen, and published in the frame and the provenance as a named
shortfall of the sample. They are never dropped silently: an anomaly is a
trade this list does not contain.

The failure this module is built to make impossible is the quiet one: a
collector that connects to the wrong URL, or subscribes with a key the venue
acknowledges and never delivers on, or reads an address field that does not
exist, and then writes a valid, empty, confidently-framed address list. That
file loads without complaint (`FileAddressSource` has no minimum count), the
nightly sweep snapshots nothing, `progress()` shows a gate that never moves,
and the reason sits three layers away. Every path that reaches zero addresses
here raises instead, and the write refuses an empty list even when the
operator has passed `--allow-short`.

The mirror-image failure -- refusing to publish a harvest and then discarding
it -- is guarded too. A run that comes back short of §3.3's gate is refused,
but the addresses are written to a timestamped rescue file first, because the
alternative is telling an operator who has just stood over a 30-minute window
that their 199 addresses exist in no file anywhere.

Exit codes are distinct because the four failures need different responses,
and a collision between any two of them is a wrong diagnosis in an operator's
hands (`EXIT_*` constants below):

    0  wrote the list
    1  collected, then refused to publish it (short of the gate, path exists)
    2  the feed did not look the way this module assumes -- code, not weather
    3  never got a usable connection: --ws-url, DNS, TLS, refusal, missing dep
    4  bad invocation, caught before anything connects

**No example output anywhere in this repository was captured from the live
feed.** The API is 403 at this environment's proxy (OPEN-QUESTIONS E5), so
every frame, counter, timestamp and address quoted in a docstring, a document
or a test is stub-generated -- the test suite drives this module through a
scripted socket and a fake clock. Anything that looks like a capture is not
one, and a reader deciding whether the shape is confirmed must go to a live
run, not to an example.

`websockets` is not a dependency of the risk engine and must not become one
(`requirements.txt` is numerics only). It is imported lazily, inside the one
function that needs it, with an ImportError that says what to install.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

from risk_engine.domain.types import normalise_address

#: The only WebSocket URL recorded anywhere in this repository, and it is
#: CONFIRMED 2026-07-30 by `--dry-run` from an operator's machine: the
#: connection opened and the subscription was acknowledged and delivered. It
#: stays blocked at this environment's proxy, so `verify.py`'s
#: `check_webdata3` still returns UNCHECKABLE from here — that is a fact about
#: this sandbox, not about the URL. There is deliberately no testnet constant to match
#: `info.TESTNET_URL`: inferring `wss://api.hyperliquid-testnet.xyz/ws` from
#: the Info host convention is a guess, and a guessed URL that happens to
#: resolve to something is worse than `--ws-url` typed by an operator who
#: knows what they are connecting to.
MAINNET_WS_URL = "wss://api.hyperliquid.xyz/ws"

#: The subscription and channel name for public trades. CONFIRMED 2026-07-30:
#: the envelope was inferred -- the outer `method`/`subscription` shape
#: borrowed from C4's `webData3` snippet, the inner `type`/`coin` pair pure
#: inference -- and a live `--dry-run` had it acknowledged and delivering
#: within seconds. The two failure modes it was written against (rejected,
#: surfacing as an error frame; or acknowledged and never delivered,
#: surfacing as the no-trade-records abort) both stay guarded, because
#: neither can be told from a quiet market without asserting, and a venue
#: that renames a channel does not announce it.
TRADES_CHANNEL = "trades"

#: Where a trade record carries the accounts that traded. This was THE
#: unverified assumption of the whole module -- B4's own words were that a
#: public trade naming its participants "needs to be verified", and nothing
#: in this repository had ever spoken the protocol.
#:
#: CONFIRMED 2026-07-30, mainnet, 60-second `--dry-run`: 36 distinct addresses
#: from 30 trade records, read from `users`. The plural won, as guessed, and
#: for the guessed reason -- a trade has two sides. `user` stays in the tuple
#: as a fallback rather than being pruned to the observed value: one minute of
#: one venue on one day is thin evidence for deleting a branch, and the
#: singular is what the webData2 subscription is conventionally keyed on.
#:
#: Whichever field is actually found is recorded in the result and stated in
#: the generated frame, so a reader of the address list can tell which shape
#: the venue really produced on the day it was collected. Extend this tuple if
#: the shape changes -- do not paper over a different field name further
#: downstream, where it becomes an address list nobody can account for.
TRADE_ADDRESS_FIELDS = ("users", "user")

#: The coins to watch. Defaulted to `LiveSnapshotProvider`'s default universe
#: on purpose: the sweep simulates only the assets in its universe, so an
#: account harvested from a trade in some other coin may hold nothing the
#: sweep can price and will be skipped for "no open positions" -- a harvested
#: address that silently never reaches the journal. Widen both together or
#: neither.
DEFAULT_COINS = ("BTC", "ETH", "SOL")

#: §3.3 asks for 200-500 addresses and the gate needs 200 *resolved* ones.
#: Targeting the top of the range rather than the bottom is deliberate: the
#: sweep drops accounts that are flat or have non-positive equity, resolution
#: can go stale, and the book-unchanged cohort then discards the active
#: accounts this very frame selects for. A list of exactly 200 cannot clear a
#: 200-address gate; only headroom can.
DEFAULT_TARGET = 500

#: How long to wait for the first frame of any kind before concluding the URL
#: or the subscribe envelope is wrong. A liquid venue acknowledges a
#: subscription immediately; silence here is a wiring fault, not a quiet
#: market, and finding out in 30 seconds beats finding out in 30 minutes.
FIRST_FRAME_GRACE_S = 30.0

#: How long to tolerate frames that are not trade records before concluding
#: the subscription was acknowledged and is not delivering. Longer than the
#: first-frame grace because this one can in principle be a genuinely quiet
#: market -- but not on BTC for five minutes.
FIRST_TRADE_GRACE_S = 300.0

#: Values found in the address field that `normalise_address` refuses are
#: counted and skipped, because one malformed entry in a live feed is dirty
#: data. This many with not one address accepted is not dirty data, it is the
#: wrong field, and continuing would spend the whole time budget to produce an
#: empty list.
MALFORMED_ABORT_THRESHOLD = 20

#: Per-`recv` wait. The deadline and the progress line are both checked at the
#: top of the loop, so this only bounds how late either can be, and it is
#: deliberately read in real seconds rather than off the injectable clock
#: below -- mixing a simulated clock into an awaited timeout is how a test
#: ends up hanging on a real 30-minute wait.
POLL_INTERVAL_S = 0.5

#: WebSocket-level keepalive. Whether Hyperliquid also expects an
#: application-level ping is unrecorded (the Polymarket feed in
#: `polymarket_bot/` does, but that is a different venue and transfers
#: nothing), so a connection that closes mid-run says so in
#: `stopped_because` and names this as a candidate cause instead of silently
#: reporting a short list as a complete one.
PING_INTERVAL_S = 20.0

_EXAMPLE_LIMIT = 3
_EXAMPLE_CHARS = 300

#: Exit codes. Every one of these is a different action for the operator, and
#: two failures sharing a code is a wrong diagnosis handed to whoever is
#: reading the shell: a refused connection reported as "short run" sends them
#: to lengthen `--minutes` on a collector that never opened a socket, and a
#: mistyped flag reported as "shape mismatch" sends them to edit
#: `TRADE_ADDRESS_FIELDS`. Note EXIT_USAGE is 4 rather than argparse's default
#: of 2: `_ExitCodeParser` below re-routes argparse's own errors onto it,
#: because 2 belongs to the shape mismatch and argparse would otherwise hand
#: out that meaning for a typo in a flag name.
EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_FEED_SHAPE = 2
EXIT_UNREACHABLE = 3
EXIT_USAGE = 4
#: The harvest succeeded and only the write failed -- read-only mount, dangling
#: symlink, full disk. Distinct from EXIT_REFUSED because the two call for
#: opposite responses: a refusal means the sample is not good enough and the
#: operator needs another window, whereas this means the sample IS good and is
#: sitting on stdout waiting to be redirected somewhere writable. Collapsing
#: them would send an operator to spend thirty minutes re-collecting addresses
#: they already have.
EXIT_WRITE_FAILED = 5


class FeedFault(RuntimeError):
    """Base for the two ways this collector declares the feed unusable.

    Shared so a library caller can catch both with one clause, kept as two
    subclasses because the CLI must not: they exit differently and they send
    an operator to different places (a URL, versus this module's parser).
    """


class UnexpectedFeedShape(FeedFault):
    """The feed was not shaped the way this collector assumed.

    A distinct type rather than a bare RuntimeError because the caller has to
    treat it differently from a short run: a short run is a judgement call an
    operator can override with `--allow-short`, while this means the frame on
    the wire does not match what was parsed and no address list from this run
    is trustworthy at any length.

    Raised only while nothing has been collected. Once an address has been
    read from an expected field the shape is confirmed by observation, and a
    later contradiction is recorded in `FeedAnomalies` instead -- see the
    module docstring for why the abort would be both destructive and wrong.
    """


class FeedUnreachable(FeedFault):
    """No usable connection was ever opened, so nothing was observed at all.

    Separate from `UnexpectedFeedShape` because the module's most-doubted fact
    is its URL, and the two ways a wrong URL shows up need different words. It
    spends a paragraph on the *silent* case -- a host that accepts the
    connection and never delivers trades -- and the *loud* case used to fall
    out as an unhandled `OSError`, which tracebacks and exits 1, the code
    reserved for "collected, but short". An operator reading that goes and
    lengthens the window on a collector that never opened a socket.
    """


@dataclass(frozen=True, slots=True)
class Transport:
    """Everything the collector needs from a WebSocket library.

    Injected rather than imported so the tests can drive the whole collection
    loop -- including the shape assertions, which are the part most worth
    testing -- against scripted frames with no network and no dependency. It
    carries `closed_errors` because "the peer closed the connection" is a
    normal end to a run that has to be told apart from a shape fault, and the
    exception class that means it lives in whichever library is in use.
    """

    connect: Callable[[], Any]
    closed_errors: tuple[type[BaseException], ...] = ()


@dataclass
class _Counters:
    frames: int = 0
    trade_records: int = 0
    other_frames: int = 0
    undecodable: int = 0
    unparseable_addresses: int = 0
    error_frames: int = 0
    records_without_address: int = 0
    unreadable_trade_entries: int = 0
    records_without_coin: int = 0
    address_field: str = ""
    #: Per-coin record counts in arrival order. The *observed* coins, which are
    #: not the subscribed ones and can only be a subset of them.
    coin_counts: dict[str, int] = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)
    bad_address_examples: list[str] = field(default_factory=list)
    error_examples: list[str] = field(default_factory=list)
    no_address_examples: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class FeedAnomalies:
    """Frames that contradicted the assumed shape *after* it had already held.

    Every field here is a trade this address list does not contain. They are
    counted rather than fatal because the shape assumption was confirmed by
    observation before they arrived (see `UnexpectedFeedShape`), and they are
    published rather than swallowed because a count of zero and a count of
    nine thousand describe very different samples while producing identical
    address lists. §10 forbids understating what a number rests on, so these
    reach the generated frame and the provenance block, not just a log line
    that scrolls off an operator's terminal.

    `undecodable_frames` is in here rather than beside `frames` on the result
    for one reason: it used to be incremented and read by nothing at all, so a
    run could take thousands of non-JSON frames off the wire and publish no
    hint of it.
    """

    error_frames: int = 0
    records_without_address: int = 0
    unreadable_trade_entries: int = 0
    undecodable_frames: int = 0
    error_examples: tuple[str, ...] = ()
    no_address_examples: tuple[str, ...] = ()

    @property
    def total(self) -> int:
        return (
            self.error_frames
            + self.records_without_address
            + self.unreadable_trade_entries
            + self.undecodable_frames
        )

    def clauses(self) -> list[str]:
        """One phrase per kind of contradiction, for the frame and the summary."""
        parts = []
        if self.error_frames:
            parts.append(
                f"{self.error_frames} error frame{'' if self.error_frames == 1 else 's'} "
                f"from the venue (verbatim: {list(self.error_examples)})"
            )
        if self.records_without_address:
            parts.append(
                f"{self.records_without_address} trade record"
                f"{'' if self.records_without_address == 1 else 's'} carrying none of "
                f"{list(TRADE_ADDRESS_FIELDS)} (verbatim: {list(self.no_address_examples)})"
            )
        if self.unreadable_trade_entries:
            parts.append(
                f"{self.unreadable_trade_entries} entr"
                f"{'y' if self.unreadable_trade_entries == 1 else 'ies'} inside a "
                f"'{TRADES_CHANNEL}' frame's data that was not a trade record"
            )
        if self.undecodable_frames:
            parts.append(f"{self.undecodable_frames} frame(s) that were not decodable JSON")
        return parts


@dataclass(frozen=True, slots=True)
class HarvestResult:
    """What one collection run saw, in enough detail to be published.

    Deliberately keeps the counts, not just the addresses. The addresses alone
    cannot be audited: 200 addresses harvested from 40 000 trades over half an
    hour and 200 harvested from 210 trades in the first four seconds are
    different samples of different things, and a reader of the calibration
    score has no way to tell them apart afterwards unless the run says so.

    `coins` is what was SUBSCRIBED. What was observed is `records_by_coin`,
    derived from the `coin` field of the records themselves, and the two are
    not the same fact: subscribing BTC, ETH and SOL while only BTC delivers is
    an ordinary outcome, and the difference can only ever overstate the
    breadth of the sample, never understate it. Both are kept because a coin
    that was asked for and never arrived is itself a finding.

    Every field added after `target` carries a default, so a caller
    constructing a result by hand -- a test, or a script fixing up a rescued
    run -- keeps working and simply publishes zeroes for what it does not know.
    """

    addresses: tuple[str, ...]
    coins: tuple[str, ...]
    ws_url: str
    started_at: datetime
    ended_at: datetime
    stopped_because: str
    frames: int
    trade_records: int
    unparseable_addresses: int
    address_field: str
    target: int
    records_by_coin: tuple[tuple[str, int], ...] = ()
    records_without_coin: int = 0
    other_frames: int = 0
    anomalies: FeedAnomalies = FeedAnomalies()

    def __post_init__(self) -> None:
        # Canonicalise and dedup HERE, not only in `_collect`. This type is the
        # only thing `build_payload` and `write_address_list` see, and both
        # publish `len(self.addresses)` as the count a reader of the
        # calibration score is asked to trust. A hand-built result holding the
        # same account twice under two spellings would otherwise write
        # `addresses_collected: 3` into a file that `FileAddressSource` loads
        # as 2 -- and that loader dedups on exactly this key, so the
        # disagreement would never surface as an error, only as a frame whose
        # arithmetic is wrong. Raising on a malformed entry is the same
        # boundary `normalise_address` already draws: better at construction,
        # where the operator is standing here, than at load time three weeks in.
        canonical: dict[str, None] = {}
        for a in self.addresses:
            canonical.setdefault(normalise_address(a), None)
        if tuple(canonical) != self.addresses:
            object.__setattr__(self, "addresses", tuple(canonical))

    @property
    def window_seconds(self) -> float:
        return (self.ended_at - self.started_at).total_seconds()

    @property
    def coins_observed(self) -> tuple[str, ...]:
        """The coins trade records actually arrived for, in arrival order."""
        return tuple(coin for coin, _ in self.records_by_coin)

    def summary(self) -> str:
        observed = ", ".join(self.coins_observed) or "no coin field on any record"
        anomalies = self.anomalies.clauses()
        return (
            f"{len(self.addresses)} distinct addresses from {self.trade_records} trade "
            f"records over {self.frames} frames in {self.window_seconds / 60:.1f} min "
            f"(subscribed {', '.join(self.coins)}; records arrived for {observed}); "
            f"addresses read from '{self.address_field}'; "
            f"{_values(self.unparseable_addresses)} refused by normalise_address; "
            f"anomalies: {'; '.join(anomalies) if anomalies else 'none'}; "
            f"stopped because {self.stopped_because}"
        )


def _values(n: int) -> str:
    return f"{n} value" if n == 1 else f"{n} values"


def _truncate(raw: object, limit: int = _EXAMPLE_CHARS) -> str:
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [{len(text)} chars total]"


def _remember(bucket: list[str], raw: object) -> None:
    """Keep the first few oddities verbatim for the abort message.

    The first ones rather than the last: when a subscription is wrong, the
    error or the ack that explains it arrives immediately and would be pushed
    out by thousands of later frames.
    """
    if len(bucket) < _EXAMPLE_LIMIT:
        bucket.append(_truncate(raw))


def _anomalies(counters: _Counters) -> FeedAnomalies:
    """Freeze the anomaly counters for publication."""
    return FeedAnomalies(
        error_frames=counters.error_frames,
        records_without_address=counters.records_without_address,
        unreadable_trade_entries=counters.unreadable_trade_entries,
        undecodable_frames=counters.undecodable,
        error_examples=tuple(counters.error_examples),
        no_address_examples=tuple(counters.no_address_examples),
    )


def _anomaly_total(counters: _Counters) -> int:
    """The running anomaly count, for the progress line.

    On the progress stream as well as in the published frame because an
    operator watching a 30-minute window is the only person who can still act
    on it -- a count that climbs with every frame means the shape is only
    accidentally right, and killing the run costs less than publishing it.
    """
    return _anomalies(counters).total


def _error_text(payload: object) -> str | None:
    """The feed's own complaint, if this frame is one.

    Checked before anything else is attempted on a frame. A rejected
    subscription is the single most likely way this collector is wrong -- the
    channel name is inference -- and the venue's own words about it are worth
    more than any inference this module could make from the silence that
    follows.
    """
    if not isinstance(payload, dict):
        return None
    if payload.get("channel") == "error":
        return _truncate(payload.get("data", payload))
    if "error" in payload and isinstance(payload["error"], str):
        return _truncate(payload["error"])
    return None


def _trade_records(payload: object) -> tuple[list[dict], list[object]] | None:
    """The trade records in one frame, and the entries that were not records.

    Returns None when this frame is not trades at all, which the caller counts
    as an ordinary non-trade frame (an ack, a heartbeat, something unknown)
    and retains as evidence.

    **Envelope tolerance is calibrated to how much the envelope was guessed.**
    The pairing this function used to have was backwards: it accepted a bare
    JSON array of dicts as trade records on inference alone, and then handed
    the contents to a check that was fatal. So `[{"px": "1", "sz": "2"}]` --
    a frame that was never trades and says so by having no channel and no
    address field -- became a loud abort asserting that B4's "a trade names
    its participants" assumption "did not hold", on a frame that is no
    evidence about trades either way. Where a fact is inferred, the reading of
    it has to be the tolerant half.

    So there are two envelopes with two confidence levels:

      - `{"channel": "trades", "data": [...]}` -- the venue has *labelled*
        the frame. Every dict in `data` is a trade record, and one carrying no
        address field really is the B4 assumption failing, because the venue
        called it a trade. Non-dict entries in `data` are returned separately
        rather than dropped: a venue that puts addresses directly in `data` is
        plausible (the field name is a guess), and silently discarding those
        entries is how nine decodable trades frames produced an abort message
        whose evidence section read "nothing decodable".
      - a bare array -- inference. Accepted only if at least one dict in it
        actually carries one of `TRADE_ADDRESS_FIELDS`, i.e. only if the frame
        itself supports the guess. Otherwise it is not treated as trades, and
        the first-trade grace decides whether the run was ever delivering.
    """
    if isinstance(payload, dict):
        if payload.get("channel") != TRADES_CHANNEL:
            return None
        data = payload.get("data")
        if isinstance(data, dict):
            return [data], []
        if isinstance(data, list):
            return (
                [r for r in data if isinstance(r, dict)],
                [r for r in data if not isinstance(r, dict)],
            )
        # Labelled trades, but `data` is neither a record nor a list of them.
        # Not treated as trades, and returned as an unknown frame so the caller
        # keeps it verbatim for whatever abort message needs the evidence.
        return None
    if isinstance(payload, list):
        records = [r for r in payload if isinstance(r, dict)]
        if not any(_address_values(r) is not None for r in records):
            return None
        return records, [r for r in payload if not isinstance(r, dict)]
    return None


def _address_values(record: dict) -> tuple[str, list[object]] | None:
    """The account addresses on a trade record, with the field they came from.

    Returns None when no expected field is present, which the caller turns
    into a loud abort while nothing has been collected and into a counted
    anomaly afterwards. A scalar field is read as one address and a sequence as
    many; the *contents* are not inspected here on purpose, so that a single
    junk entry inside an otherwise good list travels to `normalise_address`
    and is counted as one refused value rather than condemning the whole
    record as the wrong shape. The distinction the two paths draw is between
    "this field does not exist" (a shape fault) and "this field holds
    something that is not an address" (dirty data, counted).

    Anything that is neither a string nor a sequence -- a nested object, a
    number -- is treated as absent, because the assumption under test is not
    that the key exists but that it holds account addresses.
    """
    for name in TRADE_ADDRESS_FIELDS:
        if name not in record:
            continue
        value = record[name]
        if isinstance(value, str):
            return name, [value]
        if isinstance(value, (list, tuple)):
            return name, list(value)
    return None


def _subscribe_payload(coin: str) -> str:
    return json.dumps({"method": "subscribe", "subscription": {"type": TRADES_CHANNEL, "coin": coin}})


async def _collect(
    transport: Transport,
    coins: Sequence[str],
    *,
    ws_url: str,
    budget_s: float,
    target: int,
    clock: Callable[[], float],
    emit: Callable[[str], None],
    progress_every_s: float,
    first_frame_grace_s: float,
    first_trade_grace_s: float,
    poll_interval_s: float,
) -> tuple[list[str], _Counters, str]:
    seen: dict[str, None] = {}
    counters = _Counters()
    started = clock()
    deadline = started + budget_s
    last_progress = started
    first_frame_at: float | None = None
    first_trade_at: float | None = None
    # One warning per kind of anomaly, not one per occurrence: a feed with a
    # second record layout produces thousands of them, and a progress stream
    # nobody can read is the same as no progress stream. The running total goes
    # on the progress line instead.
    warned_unreadable = False
    stopped = ""

    async with AsyncExitStack() as stack:
        # Only the handshake is wrapped, not the loop below it. A refused
        # connection, an unresolvable host, a TLS failure or a URL the library
        # rejects outright all land here, and all four used to escape as an
        # unhandled OSError -- a traceback whose exit code (1) means "collected
        # a short list" to anyone reading it. The URL is this module's
        # most-doubted fact; the loud way of it being wrong deserves at least
        # as clear a diagnosis as the silent way.
        try:
            ws = await stack.enter_async_context(transport.connect())
        except Exception as exc:
            raise FeedUnreachable(
                f"could not open a WebSocket connection to {ws_url}: "
                f"{type(exc).__name__}: {exc}. Nothing was collected and nothing will be "
                f"written. This URL has never been reached from this repository -- "
                f"`market/verify.py` records it as UNCHECKABLE and the host is 403 at this "
                f"environment's proxy (OPEN-QUESTIONS C4, E5) -- so a refusal here is as "
                f"likely to mean the URL is wrong as it is to mean the network is. Check "
                f"--ws-url, then check whether this host is reachable at all from where "
                f"the collector is running."
            ) from exc

        for coin in coins:
            await ws.send(_subscribe_payload(coin))
        emit(
            f"subscribed to '{TRADES_CHANNEL}' for {', '.join(coins)} on {ws_url}; "
            f"collecting for up to {budget_s / 60:.0f} min or {target} addresses"
        )

        while True:
            now = clock()
            elapsed = now - started

            # The shape assertions run at the top of every iteration, not only
            # when a recv times out. A subscription that is acknowledged and
            # then streams heartbeats forever never produces a timeout, and
            # that is precisely the failure mode this collector exists to
            # refuse: frames arriving, nothing collected, success reported.
            if first_frame_at is None and elapsed >= first_frame_grace_s:
                raise UnexpectedFeedShape(
                    f"connected to {ws_url} and subscribed to '{TRADES_CHANNEL}' for "
                    f"{list(coins)}, then received NOTHING AT ALL for {elapsed:.0f}s. "
                    f"EXPECTED: a subscription acknowledgement within a second, and trade "
                    f"frames within seconds on a liquid coin. RECEIVED: no frames. Neither "
                    f"this URL nor this subscribe envelope has ever been exercised from "
                    f"this repository (OPEN-QUESTIONS C4 records the URL as UNCHECKABLE), "
                    f"so treat both as unconfirmed: check --ws-url, and check that the "
                    f"venue really names the subscription '{TRADES_CHANNEL}'. Nothing was "
                    f"collected and nothing will be written."
                )
            if first_trade_at is None and elapsed >= first_trade_grace_s:
                raise UnexpectedFeedShape(
                    f"received {counters.frames} frames from {ws_url} in {elapsed:.0f}s "
                    f"without one recognisable trade record. EXPECTED: frames shaped "
                    f"{{\"channel\": \"{TRADES_CHANNEL}\", \"data\": [ <trade record>, ... ]}}, "
                    f"or a bare array of trade records carrying "
                    f"{list(TRADE_ADDRESS_FIELDS)}. RECEIVED, verbatim: "
                    f"{counters.examples or 'nothing decodable'} "
                    f"({counters.other_frames} frames that yielded no trade record, "
                    f"{counters.undecodable} undecodable, "
                    f"{counters.unreadable_trade_entries} entries inside a "
                    f"'{TRADES_CHANNEL}' frame that were not records). A subscription that is "
                    f"acknowledged and never delivered is the shape of a wrong "
                    f"'subscription.type' or a coin the venue does not list -- both of "
                    f"which look identical to a quiet market, which is why this aborts "
                    f"rather than waiting out the budget and reporting an empty list."
                )

            if len(seen) >= target:
                stopped = f"it reached the {target}-address target"
                break
            if now >= deadline:
                stopped = f"the {budget_s / 60:.1f}-minute time budget elapsed"
                break
            if now - last_progress >= progress_every_s:
                emit(
                    f"  {elapsed / 60:6.1f} min  {counters.frames:7d} frames  "
                    f"{counters.trade_records:7d} trades  {len(seen):5d}/{target} addresses  "
                    f"{counters.unparseable_addresses} unparseable  "
                    f"{_anomaly_total(counters)} anomalies"
                )
                last_progress = now

            try:
                raw = await asyncio.wait_for(ws.recv(), poll_interval_s)
            except TimeoutError:
                # A quiet moment on the feed, which is normal. The graces above
                # decide whether quiet has become evidence of a fault.
                #
                # This clause must stay ABOVE the closed-connection one:
                # `TimeoutError` is a subclass of `OSError`, which the real
                # transport lists as a closed error, so reversing these two
                # would end every run at its first quiet half-second and
                # report the truncated sample as a finished one.
                continue
            except transport.closed_errors as exc:
                stopped = (
                    f"the feed closed the connection ({type(exc).__name__}: {exc}) after "
                    f"{elapsed / 60:.1f} min. Whether this venue requires an "
                    "application-level keepalive on top of WebSocket pings is unrecorded, "
                    "so that is a candidate cause as much as the network is"
                )
                break

            counters.frames += 1
            if first_frame_at is None:
                first_frame_at = now

            try:
                payload = json.loads(raw)
            except (TypeError, ValueError):
                counters.undecodable += 1
                _remember(counters.examples, raw)
                continue

            if error := _error_text(payload):
                # Fatal only while nothing has been collected. "Websocket
                # request timed out" is a real Hyperliquid error string, and a
                # transient one arriving at minute 29 of a delivering run used
                # to abort it and report a REJECTED SUBSCRIPTION -- an
                # accusation flatly contradicted by the trades that had been
                # flowing for 29 minutes.
                counters.error_frames += 1
                _remember(counters.error_examples, error)
                if not seen:
                    raise UnexpectedFeedShape(
                        f"the feed answered with an error frame before one address had been "
                        f"collected: {error}. The subscription sent was "
                        f"{_subscribe_payload(coins[0])} (one per coin). EXPECTED: "
                        f"an acknowledgement, then trade frames. The channel name "
                        f"'{TRADES_CHANNEL}' and the 'coin' key are inference from "
                        f"OPEN-QUESTIONS C4's webData3 snippet, not documentation, so the "
                        f"venue rejecting them is a likely outcome and is reported as a fault "
                        f"in this collector rather than as an empty sample."
                    )
                if counters.error_frames == 1:
                    emit(
                        f"  WARNING: error frame from the feed after {len(seen)} addresses; "
                        f"counted, not fatal, and recorded in the sampling frame: {error}"
                    )
                continue

            classified = _trade_records(payload)
            if classified is None:
                counters.other_frames += 1
                _remember(counters.examples, raw)
                continue

            records, unreadable = classified
            if unreadable:
                # A labelled trades frame carrying entries that are not records
                # at all. Kept as evidence: `[]` here used to be indistinguishable
                # from "no such frame arrived", so nine decodable
                # {"channel": "trades", "data": ["0x.."]} frames produced an
                # abort whose RECEIVED section read 'nothing decodable' -- and
                # a venue naming its participants directly in `data` lands
                # exactly here, with the one thing worth reading discarded.
                counters.unreadable_trade_entries += len(unreadable)
                _remember(counters.examples, raw)
                if not warned_unreadable:
                    warned_unreadable = True
                    emit(
                        f"  WARNING: a '{TRADES_CHANNEL}' frame's data held "
                        f"{len(unreadable)} entr{'y' if len(unreadable) == 1 else 'ies'} that "
                        f"was not a trade record: {_truncate(raw)}"
                    )
            if not records:
                # Labelled trades, nothing readable in it. Counted as a
                # non-delivering frame so the first-trade grace still fires on a
                # run of these rather than waiting out the whole budget.
                counters.other_frames += 1
                continue

            for record in records:
                counters.trade_records += 1
                if first_trade_at is None:
                    first_trade_at = now

                # The observed coin, per record. `result.coins` is the
                # subscription list, which is what was ASKED FOR; publishing it
                # as what was seen can only overstate the breadth of the sample
                # (subscribe BTC, ETH, SOL, have only BTC deliver, and the frame
                # claims accounts trading all three). Every record carries the
                # coin, so the honest set costs one dict update.
                coin = record.get("coin")
                if isinstance(coin, str) and coin:
                    counters.coin_counts[coin] = counters.coin_counts.get(coin, 0) + 1
                else:
                    counters.records_without_coin += 1

                found = _address_values(record)
                if found is None:
                    counters.records_without_address += 1
                    _remember(
                        counters.no_address_examples,
                        f"{_truncate(json.dumps(record))} (keys: {sorted(record)})",
                    )
                    if not seen:
                        raise UnexpectedFeedShape(
                            f"a trade record carried no account address, and none had been "
                            f"collected yet. EXPECTED one of the "
                            f"fields {list(TRADE_ADDRESS_FIELDS)}, holding an address or a "
                            f"list of them. RECEIVED: {_truncate(json.dumps(record))} "
                            f"(keys: {sorted(record)}). This is exactly the assumption "
                            f"OPEN-QUESTIONS B4 flags as needing verification -- that a "
                            f"public trade names its participants -- and it has not held "
                            f"once. If the venue names the field differently, add it to "
                            f"TRADE_ADDRESS_FIELDS here, where the frame string can record "
                            f"which field the addresses came from; do not translate it "
                            f"downstream, where the published calibration score would carry "
                            f"a sample nobody can account for."
                        )
                    if counters.records_without_address == 1:
                        emit(
                            f"  WARNING: a trade record carried none of "
                            f"{list(TRADE_ADDRESS_FIELDS)} after {len(seen)} addresses had "
                            f"been read from '{counters.address_field}'. The field exists on "
                            f"this feed, so this is one odd record rather than a wrong "
                            f"assumption: counted, and recorded in the sampling frame as a "
                            f"trade this list does not contain. Verbatim: "
                            f"{_truncate(json.dumps(record))}"
                        )
                    continue

                name, values = found
                counters.address_field = name
                for value in values:
                    try:
                        seen.setdefault(normalise_address(value), None)
                    except ValueError as exc:
                        # Counted and skipped, not fatal: one malformed value in
                        # a live feed is dirty data, and aborting a 30-minute run
                        # over it would trade a whole sample for one bad row.
                        counters.unparseable_addresses += 1
                        _remember(counters.bad_address_examples, f"{value!r}: {exc}")

            if counters.unparseable_addresses >= MALFORMED_ABORT_THRESHOLD and not seen:
                raise UnexpectedFeedShape(
                    f"all {counters.unparseable_addresses} values taken from the "
                    f"'{counters.address_field}' field of {counters.trade_records} trade "
                    f"records were refused by normalise_address, and not one address was "
                    f"collected. EXPECTED: '0x' followed by 40 hex digits. RECEIVED: "
                    f"{counters.bad_address_examples}. A field that is present and never "
                    f"holds an address is the wrong field, not dirty data, so this aborts "
                    f"instead of spending the rest of the budget to produce an empty list."
                )

    return list(seen), counters, stopped


def harvest(
    *,
    coins: Sequence[str] = DEFAULT_COINS,
    minutes: float = 30.0,
    target: int = DEFAULT_TARGET,
    ws_url: str = MAINNET_WS_URL,
    transport: Transport | None = None,
    clock: Callable[[], float] = time.monotonic,
    utcnow: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    emit: Callable[[str], None] = lambda line: print(line, flush=True),
    progress_every_s: float = 15.0,
    first_frame_grace_s: float = FIRST_FRAME_GRACE_S,
    first_trade_grace_s: float = FIRST_TRADE_GRACE_S,
    poll_interval_s: float = POLL_INTERVAL_S,
) -> HarvestResult:
    """Collect addresses from the trades feed until the target or the budget.

    Raises `UnexpectedFeedShape` rather than returning a short or empty
    result whenever the feed did not look the way this module assumes. That
    asymmetry is the point of the module: a short list is a fact about the
    market that an operator can weigh, an empty list is almost always a fact
    about this code being wrong, and the two must not arrive looking alike.

    Anomalies that arrive *after* the first address are counted into
    `HarvestResult.anomalies` instead of raising, because by then the shape is
    confirmed rather than assumed. Everything counted there is published.

    `clock` and `utcnow` are injectable so the time-budget path and the
    generated frame's window are testable without a wall-clock wait. The
    project has one test already failing intermittently on a shared box
    because it measures elapsed time; adding a 30-minute one, or a sleeping
    one, would be adding to that problem.
    """
    if complaint := _window_complaint(coins, minutes, target):
        raise ValueError(complaint)

    transport = transport or _websockets_transport(ws_url)
    started_at = utcnow()
    addresses, counters, stopped = asyncio.run(
        _collect(
            transport,
            tuple(coins),
            ws_url=ws_url,
            budget_s=minutes * 60.0,
            target=target,
            clock=clock,
            emit=emit,
            progress_every_s=progress_every_s,
            first_frame_grace_s=first_frame_grace_s,
            first_trade_grace_s=first_trade_grace_s,
            poll_interval_s=poll_interval_s,
        )
    )
    ended_at = utcnow()

    # The last line of defence against the failure this module exists to
    # prevent. Every known way of collecting nothing raises above; this catches
    # the unknown ones, because an empty list that reaches the writer becomes a
    # file that loads cleanly and sweeps nothing.
    if not addresses:
        raise UnexpectedFeedShape(
            f"collected no addresses at all from {counters.trade_records} trade records "
            f"over {counters.frames} frames, without any of the shape checks firing. "
            f"That combination is not understood and must not be reported as an empty "
            f"sample: frames seen were {counters.examples or 'none retained'}, refused "
            f"address values were {counters.bad_address_examples or 'none'}, anomalies were "
            f"{_anomalies(counters).clauses() or 'none'}."
        )

    return HarvestResult(
        addresses=tuple(addresses),
        coins=tuple(coins),
        ws_url=ws_url,
        started_at=started_at,
        ended_at=ended_at,
        stopped_because=stopped,
        frames=counters.frames,
        trade_records=counters.trade_records,
        unparseable_addresses=counters.unparseable_addresses,
        address_field=counters.address_field,
        target=target,
        records_by_coin=tuple(counters.coin_counts.items()),
        records_without_coin=counters.records_without_coin,
        other_frames=counters.other_frames,
        anomalies=_anomalies(counters),
    )


def _window_complaint(coins: Sequence[str], minutes: float, target: int) -> str | None:
    """Why this collection window cannot collect anything, if it cannot.

    Shared between `harvest` and `main` so the CLI can refuse before it
    connects and still say the same thing the library says. Split out for the
    exit code rather than for the DRY: `--minutes 0` used to reach `harvest`,
    raise an uncaught ValueError, traceback, and exit 1 -- the code that means
    "collected a list, refused to publish it". The operator's next move for
    that code is a longer window, which is precisely the thing they had just
    set to zero.
    """
    if not coins:
        return "at least one coin is required; the feed is subscribed per coin"
    if minutes <= 0:
        return f"--minutes must be positive, got {minutes}"
    if target <= 0:
        return f"--target must be positive, got {target}"
    return None


def _websockets_transport(url: str, ping_interval_s: float = PING_INTERVAL_S) -> Transport:
    """Bind the collector to `websockets`, lazily.

    Lazy on purpose and not as a style choice. `risk_engine/requirements.txt`
    is numpy, scipy and pytest under the header "deliberately thin: numerics
    only", `deploy/Dockerfile.engine` installs exactly that file, and nothing
    else in the engine speaks WebSocket. A module-level import would make one
    operator-run collection job a hard dependency of the whole engine, and
    would break the container image rather than this one command.
    """
    try:
        import websockets
    except ImportError as exc:
        raise ImportError(
            "collecting addresses from the trades feed needs the 'websockets' package, "
            "which the risk engine deliberately does not depend on (requirements.txt is "
            "numerics only, and this is the only command in the engine that speaks "
            "WebSocket). Install it for this job:\n"
            "    pip install 'websockets>=12.0'\n"
            "and add the same line to deploy/Dockerfile.engine if the collector is to run "
            "inside the container rather than from an operator's shell."
        ) from exc

    from websockets.exceptions import ConnectionClosed

    def connect() -> Any:
        return websockets.connect(url, ping_interval=ping_interval_s)

    # OSError alongside ConnectionClosed so a mid-run network drop ends the run
    # with a stated reason and whatever was collected, rather than a traceback
    # that discards half an hour of harvesting.
    return Transport(connect=connect, closed_errors=(ConnectionClosed, OSError))


def gate_required_addresses() -> int:
    """§3.3's address count, read off the gate rather than copied.

    Imported here rather than at module scope for two reasons: `market/` sits
    *below* `shadow/` in this package's layering (shadow/providers.py imports
    market/info, never the reverse), and the number is wanted once, in a CLI,
    not at import time. Copying the literal 200 into this file would be the
    cheaper option and would silently stop matching `ShadowProgress` the day
    the gate moves -- and a collector that writes 200 addresses against a gate
    that now wants 300 fails at the far end of a 21-day window.
    """
    from risk_engine.shadow.journal import ShadowProgress

    return ShadowProgress(
        distribution_version="",
        distinct_days=0,
        distinct_addresses=0,
        resolved_observations=0,
    ).required_addresses


def _observed_clause(result: HarvestResult) -> str:
    """What was subscribed against what actually delivered.

    Two facts that the first sentence of this frame used to conflate, in the
    direction that flatters the sample: `result.coins` is the `--coins`
    argument, so subscribing BTC, ETH and SOL while only BTC ever delivers
    published "N distinct accounts observed trading BTC, ETH, SOL". The error
    is one-sided -- the observed set is always a subset of the subscribed one
    -- so conflating them can only ever overstate the breadth of the sample,
    and this artifact's whole purpose is to be the one place that does not do
    that.
    """
    subscribed = ", ".join(result.coins)
    if not result.records_by_coin:
        return (
            f"SUBSCRIBED vs OBSERVED: subscriptions were sent for {subscribed}, but no trade "
            "record carried a 'coin' field, so which of them actually delivered is UNKNOWN "
            "and the subscription list must not be read as the observed one."
        )
    per_coin = ", ".join(f"{coin} {n}" for coin, n in result.records_by_coin)
    missing = [c for c in result.coins if c not in result.coins_observed]
    clause = (
        f"SUBSCRIBED vs OBSERVED: subscriptions were sent for {subscribed}; trade records "
        f"actually arrived for {', '.join(result.coins_observed)} ({per_coin} records). "
    )
    if missing:
        clause += (
            f"Nothing arrived for {', '.join(missing)}, which therefore contribute no "
            "accounts to this list -- a subscription the venue never delivered on is "
            "indistinguishable here from a coin that did not trade, and either way naming "
            "it as observed would overstate this sample. "
        )
    if result.records_without_coin:
        clause += (
            f"{result.records_without_coin} record(s) carried no readable 'coin' field, so "
            "the per-coin counts are a lower bound. "
        )
    return clause.rstrip()


def _anomaly_clause(result: HarvestResult) -> str:
    """The contradictions that were counted rather than aborted on.

    Published because the alternative is a file that cannot be told apart from
    one collected off a feed that matched the assumption exactly. §10 forbids
    understating what a number rests on; a demoted abort that reached no
    artifact would be exactly that.
    """
    anomalies = result.anomalies
    lower_bound = (
        "Each one is a trade this list does NOT contain, so treat the address count as a "
        "lower bound on the accounts that traded and the message shape as only partly "
        "understood."
    )
    share = ""
    if anomalies.records_without_address and result.trade_records:
        pct = 100.0 * anomalies.records_without_address / result.trade_records
        share = (
            f" That is {pct:.1f}% of the {result.trade_records} records read; above a few "
            "per cent this is no longer a stray record but a second record layout on the "
            "same feed, and this list is then a sample of the records this parser could "
            "read rather than of the trades that occurred."
        )
    return (
        f"ANOMALIES: {anomalies.total} frame(s)/record(s) on this feed did not match the "
        f"assumed message shape: {'; '.join(anomalies.clauses())}. They were counted rather "
        f"than aborted on, because this collector's shape assertions are fatal only until "
        f"the first address is read out of one of {list(TRADE_ADDRESS_FIELDS)} -- and this "
        f"run read {len(result.addresses)} from '{result.address_field}', which confirms the "
        f"assumption these anomalies would otherwise be read as refuting.{share} "
        f"{lower_bound}"
    )


def build_frame(result: HarvestResult, required: int, *, refusal: str = "") -> str:
    """The sampling frame this run represents, as prose fit to publish.

    `FileAddressSource` refuses a list whose frame is blank because an
    unstated frame becomes an unstated bias in the published calibration
    score. Generating the frame rather than asking the operator to write one
    is the only way that requirement survives automation: an operator handed
    an address list and a blank field writes "trades feed", which satisfies
    the check and states nothing. Everything a reader needs to discount this
    sample is therefore assembled here -- what was watched, when, how it was
    selected, what that selection does to the gate, and what remains
    unverified.

    `required` is used verbatim, not floored: this function formats what it is
    told. The floor belongs at the write boundary, where the decision to
    publish is actually taken -- see `write_address_list`.

    `refusal`, when set, marks the frame as belonging to a rescue copy of a
    file the writer refused to publish. It matters because that file is a
    valid, loadable address list sitting next to the one that was asked for,
    and a reader who finds it has to be told it was never accepted.
    """
    n = len(result.addresses)
    window = (
        f"{result.started_at.strftime('%Y-%m-%dT%H:%M:%SZ')} to "
        f"{result.ended_at.strftime('%Y-%m-%dT%H:%M:%SZ')} UTC "
        f"({result.window_seconds / 60:.1f} min)"
    )
    # What was OBSERVED, never what was subscribed. See `_observed_clause`.
    traded = (
        f" trading {', '.join(result.coins_observed)}" if result.coins_observed
        else " (the coin per record was not readable -- see SUBSCRIBED vs OBSERVED)"
    )
    parts = [
        f"{n} distinct accounts observed{traded} on "
        f"Hyperliquid's public trades WebSocket feed ({result.ws_url}, subscription "
        f"type '{TRADES_CHANNEL}', addresses read from each trade record's "
        f"'{result.address_field}' field), {window}, from {result.trade_records} trade "
        f"records over {result.frames} frames; collection stopped because "
        f"{result.stopped_because}.",
        _observed_clause(result),
        "SELECTION: an account is in this list because it traded inside that window, so "
        "the frame selects on trading activity. It over-represents accounts that trade "
        "frequently, omits every account that holds a position without touching it, and "
        "is conditioned on whatever the market happened to be doing over those minutes. "
        "Both sides of each trade are taken where the feed reports them, so an account "
        "that makes markets and appears in a large share of trades enters the list on "
        "the same footing as a one-off taker.",
        "WHY THIS FRAME AND NOT THE LEADERBOARD: the Info API enumerates no addresses, "
        "so the alternative enumerable source is the leaderboard, which ranks on "
        "realised performance -- the very variable a calibration score measures. "
        "Sampling on the outcome would make the model appear mis-calibrated in whichever "
        "direction the sample was skewed, and nothing downstream recovers from that. "
        "Activity bias is awkward; performance bias is circular (OPEN-QUESTIONS B4).",
        "KNOWN TENSION, stated because it weakens this list: §3.3's gate is read off the "
        "book-unchanged cohort (OPEN-QUESTIONS B2), which discards any account whose book "
        "moved during the observation day -- that is, precisely the most active accounts "
        "this frame selects for. The usable sample shrinks in a way correlated with how it "
        f"was drawn, so the effective n behind any score computed on that cohort will be "
        f"materially smaller than {n}.",
        "NOT VERIFIED: the feed's message shape was asserted at runtime while collecting "
        "-- this collector refuses to report success unless it finds addresses where it "
        "expects them -- but the assertion is a floor and not a proof: once the first "
        "address had been read the shape was treated as confirmed, and later contradictions "
        "were counted (see ANOMALIES if that section is present) rather than aborting a "
        "window that had already produced a sample. The URL, the subscription name and the "
        "record layout are not confirmed against Hyperliquid documentation from the "
        "environment this was built in, where the API is blocked at the proxy "
        "(OPEN-QUESTIONS C4, E5). No example trade frame in the collector's own repository "
        "was captured from the live venue either -- every one is stub-generated, so nothing "
        "there can be cited as evidence that this shape is right.",
    ]
    if result.anomalies.total:
        parts.append(_anomaly_clause(result))
    if refusal:
        parts.append(
            f"NOT THE FILE THAT WAS ASKED FOR: this is a rescue copy, written because the "
            f"requested output was refused -- {refusal}. The requested path was left "
            f"untouched. It exists so that a collection window is never spent and then "
            f"thrown away, and it is a loadable address list, so pointing a sweep at it is "
            f"a deliberate act with the consequences stated in this frame."
        )
    if n < required:
        parts.append(
            f"SHORT: {n} addresses is below §3.3's requirement of {required} distinct "
            "addresses. It cannot open the Phase 4 gate on its own, and a list this short "
            "is only ever published because someone asked for it explicitly (--allow-short) "
            "or as the rescue copy of a refused write."
        )
    elif n < required * 1.25:
        parts.append(
            f"THIN HEADROOM: the gate counts {required} distinct addresses with a resolved, "
            "non-stale observation, and the sweep drops accounts that are flat or hold "
            f"non-positive equity before that point, so {n} entries may not yield "
            f"{required} counted ones."
        )
    if result.unparseable_addresses:
        parts.append(
            f"{_values(result.unparseable_addresses)} in the '{result.address_field}' field "
            f"{'was' if result.unparseable_addresses == 1 else 'were'} refused by "
            "normalise_address and skipped; they are excluded from the count above."
        )
    return " ".join(parts)


def build_payload(result: HarvestResult, required: int, *, refusal: str = "") -> dict:
    """The file `FileAddressSource` reads, plus a record of how it was made.

    Two load-bearing keys and one extra. `frame` and `addresses` are what the
    loader validates; `_provenance` is ignored by it and is here because the
    template written by `shadow init-addresses` carries `_frame_help` and
    `_addresses_help` prose telling a human what to fill in, and copying
    fill-me-in instructions into an already-filled generated file would leave
    a reader unable to tell whether a human curated this list. The counts a
    reader would otherwise have to take on trust from the frame sentence go
    here as data.

    `addresses_collected` is `len(result.addresses)` and `HarvestResult`
    canonicalises in `__post_init__`, so this count is the count
    `FileAddressSource` will load -- the two cannot disagree.
    """
    return {
        "frame": build_frame(result, required, refusal=refusal),
        "_provenance": {
            "collector": "risk_engine.market.collect_addresses",
            "source": "hyperliquid public trades websocket",
            "ws_url": result.ws_url,
            "subscription_type": TRADES_CHANNEL,
            "address_field": result.address_field,
            # `coins` is what was SUBSCRIBED, and keeps that key because it is
            # what this file has always carried. What the feed actually
            # delivered is `coins_observed`, which is a subset and is the only
            # one of the two that can honestly be called observed.
            "coins": list(result.coins),
            "coins_observed": list(result.coins_observed),
            "trade_records_by_coin": dict(result.records_by_coin),
            "records_without_coin_field": result.records_without_coin,
            "window_start_utc": result.started_at.isoformat(),
            "window_end_utc": result.ended_at.isoformat(),
            "frames": result.frames,
            # Acks, heartbeats, unknown frames, and frames the venue labelled
            # 'trades' that held nothing readable. Not called "non-trade" because
            # the last of those was labelled a trade and is the interesting case.
            "frames_yielding_no_trade_record": result.other_frames,
            "trade_records": result.trade_records,
            "addresses_collected": len(result.addresses),
            "unparseable_values_skipped": result.unparseable_addresses,
            # The anomaly counts as data, not only as prose in the frame. A
            # reader comparing two address lists needs to be able to diff these
            # without parsing English.
            "anomalies": {
                "total": result.anomalies.total,
                "error_frames": result.anomalies.error_frames,
                "trade_records_without_address_field": (
                    result.anomalies.records_without_address
                ),
                "unreadable_entries_in_trades_frames": (
                    result.anomalies.unreadable_trade_entries
                ),
                "undecodable_frames": result.anomalies.undecodable_frames,
                "error_examples": list(result.anomalies.error_examples),
                "records_without_address_examples": list(
                    result.anomalies.no_address_examples
                ),
            },
            "target": result.target,
            "gate_required_addresses": required,
            "refused_write": refusal,
            "stopped_because": result.stopped_because,
        },
        "addresses": list(result.addresses),
    }


def write_address_list(
    result: HarvestResult,
    path: Path,
    *,
    allow_short: bool = False,
    force: bool = False,
    required: int | None = None,
) -> dict:
    """Write the list, refusing the two files that would mislead a sweep.

    The gate check lives here rather than in the CLI so a library caller
    cannot skip it: the danger is not an operator typing the wrong flag, it is
    a short list becoming a 21-day window that turns out never to have been
    able to clear the gate.

    An empty list is refused unconditionally, `--allow-short` or not.
    `--allow-short` is an operator saying "I know this sample is small";
    nobody means "write a file that loads cleanly and sweeps nothing" by it,
    and `FileAddressSource` accepts `addresses: []` without complaint.

    `required` can only ever make the check STRICTER. It is floored at
    `gate_required_addresses()` because as a plain override it did two
    damaging things at once: `required=2` both skipped the refusal for a
    2-address harvest and made the published frame say "THIN HEADROOM: the
    gate counts 2 distinct addresses" -- a false statement about §3.3 in the
    one artifact whose job is to be honest about the sample. A caller
    tightening the bar is a legitimate thing to want; a caller lowering §3.3's
    gate from a keyword argument is not, and `--allow-short` already exists
    for the case where someone genuinely means to publish a short list.

    Every refusal that has addresses to lose writes a timestamped rescue copy
    first. A refusal that also discards the harvest makes an operator find out
    about a problem after standing over a collection window -- the exact thing
    the pre-flight checks in `main` exist to prevent -- and 199 addresses
    appearing in no file anywhere costs another 30 minutes to fix.
    """
    gate = gate_required_addresses()
    required = gate if required is None else max(required, gate)
    n = len(result.addresses)
    if n == 0:
        raise ValueError(
            "refusing to write an empty address list. It would load without complaint "
            "and every sweep against it would snapshot nothing, showing up as a gate "
            "that never advances rather than as a collector that failed."
        )
    if n < required and not allow_short:
        rescued = _write_rescue_copy(
            result, path, required, f"{n} addresses against §3.3's requirement of {required}"
        )
        raise ValueError(
            f"found {n} distinct addresses, §3.3 needs {required} -- nothing written to "
            f"{path}. The gate counts addresses with a resolved observation, and the sweep "
            f"drops flat and zero-equity accounts before that, so aim above {required} "
            f"rather than at it ({DEFAULT_TARGET} is the top of §3.3's range). Re-run with a "
            f"longer --minutes or more --coins, or pass --allow-short to write it anyway "
            f"-- the frame will record that it is short and cannot open the gate. {rescued} "
            f"Nothing reads that path by itself."
        )
    if path.exists() and not force:
        rescued = _write_rescue_copy(result, path, required, f"{path} already exists")
        raise ValueError(f"{path} exists; pass --force to overwrite. {rescued}")

    payload = build_payload(result, required)
    # indent=2 with a trailing newline, matching `shadow init-addresses`, so a
    # generated list and a hand-filled template diff against each other
    # without a whitespace storm hiding the change that matters.
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _write_rescue_copy(result: HarvestResult, path: Path, required: int, refusal: str) -> str:
    """Park a refused harvest next to where it was going; report where it went.

    Three deliberate choices in the filename. It keeps the requested name as a
    prefix so the two are obviously related; it ends in the collection
    window's UTC start rather than anything reusable, so no rescue file can
    ever overwrite another one and no addresses are lost to a second refused
    run; and it does NOT end in `.json`, so it cannot be swept up by a glob or
    mistaken for the list an operator asked for. The contents are a valid
    address list -- deliberately, since the point is to be usable -- carrying a
    frame that opens by saying it was refused.

    Returns a sentence rather than a path, and swallows OSError into that
    sentence, because this runs on the way to raising the refusal. A rescue
    that failed on a read-only directory must not replace the operator's "you
    are 199 short" message with an unrelated errno; the refusal is the more
    important of the two and has to survive.
    """
    n = len(result.addresses)
    stamp = result.started_at.strftime("%Y%m%dT%H%M%SZ")
    rescue = path.parent / f"{path.name}.refused-{stamp}"
    try:
        rescue.write_text(
            json.dumps(build_payload(result, required, refusal=refusal), indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        return (
            f"WARNING: the {n} addresses collected could not be parked beside {path} either "
            f"({type(exc).__name__}: {exc}), so this window's harvest exists nowhere on "
            f"disk. Fix that path before re-running, or the next 30 minutes go the same way."
        )
    return (
        f"The {n} addresses collected are NOT lost: they are in {rescue}, whose frame states "
        f"that it was refused and why."
    )


def _parse_coins(raw: str) -> tuple[str, ...]:
    coins = tuple(c.strip().upper() for c in raw.split(",") if c.strip())
    if not coins:
        raise argparse.ArgumentTypeError("--coins needs at least one coin name")
    return coins


class _ExitCodeParser(argparse.ArgumentParser):
    """argparse, with its exit code moved off the one that means "shape".

    argparse exits 2 for every usage error, and 2 is this module's "the feed
    did not look the way I assume". Left alone, a typo in a flag name reports
    itself as a feed-shape mismatch and sends an operator to
    `TRADE_ADDRESS_FIELDS` -- a wrong diagnosis of a wrong diagnosis. The
    message and its destination (stderr) are unchanged; only the number moves.
    """

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE)


def _dry_run(args, parser) -> int:
    """Probe the feed briefly and report, writing nothing.

    Every fact this module relies on about the venue is unverified here: the
    WebSocket URL, the subscribe envelope, the channel name, and the claim
    that a public trade names its participants. All four come from one
    UNCHECKABLE note and a snippet nobody has executed, because the API is 403
    at this environment's proxy. The full command spends thirty minutes before
    an operator learns whether any of them hold; this spends one.

    It deliberately reuses `harvest` rather than re-implementing a lighter
    probe. A separate code path would be testing a different set of
    assumptions from the one the real run makes -- which is precisely the
    failure mode this whole module is built around -- so the dry run is the
    real collector with a short window and no write, and its verdict
    transfers exactly.

    A `target` of one address is deliberate: this asks whether the wiring
    works, not whether the sample is big enough. Sample size is what
    `--minutes` buys, and it is a different question with a different answer.
    """
    seconds = max(args.dry_run_seconds, 1.0)
    print(
        f"DRY RUN: {seconds:.0f}s against {args.ws_url}, coins {', '.join(args.coins)}. "
        "Nothing will be written.\n"
    )
    try:
        result = harvest(
            coins=args.coins,
            minutes=seconds / 60.0,
            target=1,
            ws_url=args.ws_url,
            progress_every_s=max(seconds / 4.0, 5.0),
            # The graces are what turn "acknowledged but silent" into a loud
            # fault, and they are sized for a 30-minute run. Cap them at the
            # window so a 60-second probe cannot spend its whole budget
            # waiting for a grace that was never going to expire in time.
            first_frame_grace_s=min(FIRST_FRAME_GRACE_S, seconds),
            first_trade_grace_s=min(FIRST_TRADE_GRACE_S, seconds),
        )
    except FeedUnreachable as exc:
        print(f"COULD NOT CONNECT: {exc}")
        print(
            "\nVERDICT: the URL or the network is wrong, and nothing was learned about "
            "the message shape. A real run would fail the same way in its first seconds."
        )
        return EXIT_UNREACHABLE
    except ImportError as exc:
        print(str(exc))
        return EXIT_UNREACHABLE
    except UnexpectedFeedShape as exc:
        print(f"FEED SHAPE MISMATCH: {exc}")
        print(
            "\nVERDICT: the connection works and the feed is not what this module "
            "assumes. This is the finding OPEN-QUESTIONS B4 and C4 said needed "
            "verification, and a real run would have wasted 30 minutes to reach it. "
            "The quoted frame above is the evidence for whichever constant needs "
            "changing -- TRADES_CHANNEL, the subscribe envelope, or "
            "TRADE_ADDRESS_FIELDS."
        )
        return EXIT_FEED_SHAPE

    n = len(result.addresses)
    print(
        f"\nVERDICT: the feed behaves as assumed. {n} distinct "
        f"address{'' if n == 1 else 'es'} from {result.trade_records} trade "
        f"record{'' if result.trade_records == 1 else 's'} in {seconds:.0f}s, "
        f"read from the '{result.address_field}' field."
    )
    if result.anomalies.total:
        print(
            f"  {result.anomalies.total} anomaly/anomalies were counted rather than "
            "aborted on, because addresses were already arriving. A real run publishes "
            "these in the frame; read them before trusting the sample."
        )
    rate = n / seconds * 60.0 if seconds else 0.0
    if rate > 0:
        needed = gate_required_addresses()
        print(
            f"  Roughly {rate:.0f} new addresses/minute at this moment, so §3.3's "
            f"{needed} would take on the order of {needed / rate:.0f} minutes if the "
            "rate held. It will not hold -- distinct addresses saturate as the active "
            "traders repeat -- so treat that as a floor on the window, not an estimate "
            f"of it, and prefer --minutes 30 over the arithmetic."
        )
    print("\nReady to collect:\n    python -m risk_engine.market.collect_addresses "
          "--minutes 30 --out addresses.json")
    return EXIT_OK


def main(argv: Iterable[str] | None = None) -> int:
    parser = _ExitCodeParser(
        prog="risk_engine.market.collect_addresses",
        description=(
            "harvest a §3.3 shadow address list from Hyperliquid's public trades feed "
            "(OPEN-QUESTIONS B4)"
        ),
        epilog=(
            "exit codes: 0 wrote the list; 1 collected but refused to publish "
            "(short of the gate, or --out exists) -- the addresses are kept in a "
            "'.refused-<window>' file beside --out; 2 the feed did not match this "
            "collector's assumed message shape; 3 no usable connection (--ws-url, DNS, "
            "TLS, refusal, or the missing 'websockets' package); 4 bad invocation."
        ),
    )
    parser.add_argument("--out", help="where to write the address list "
                                      "(not needed with --dry-run)")
    parser.add_argument(
        "--dry-run", dest="dry_run", action="store_true",
        help="connect, subscribe and read for --dry-run-seconds, report what the feed "
             "actually looks like, write nothing. Run this BEFORE a real collection: "
             "every assumption this module makes about the venue is unverified, and a "
             "minute now is cheaper than finding out at minute 30",
    )
    parser.add_argument("--dry-run-seconds", dest="dry_run_seconds", type=float, default=60.0)
    parser.add_argument("--minutes", type=float, default=30.0,
                        help="time budget for the collection window (default 30)")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET,
                        help=f"stop once this many distinct addresses are seen "
                             f"(default {DEFAULT_TARGET}, the top of §3.3's 200-500 range)")
    parser.add_argument("--coins", type=_parse_coins, default=DEFAULT_COINS,
                        help="comma-separated coins to watch; default matches the shadow "
                             "sweep's universe, and widening one without the other "
                             "harvests accounts the sweep cannot price")
    parser.add_argument("--ws-url", dest="ws_url", default=MAINNET_WS_URL,
                        help=f"WebSocket endpoint (default {MAINNET_WS_URL}, itself "
                             "unconfirmed -- OPEN-QUESTIONS C4)")
    parser.add_argument("--allow-short", dest="allow_short", action="store_true",
                        help="write a list below §3.3's address count; the frame records it")
    parser.add_argument("--force", action="store_true", help="overwrite an existing --out")
    parser.add_argument("--progress-seconds", dest="progress_seconds", type=float, default=15.0)
    args = parser.parse_args(argv)

    if args.dry_run:
        return _dry_run(args, parser)
    if not args.out:
        parser.error("--out is required (or pass --dry-run to probe the feed first)")

    out = Path(args.out)
    required = gate_required_addresses()

    # Every check BEFORE connecting. An operator who mistypes the output path,
    # forgets --force or asks for a zero-length window should learn that in the
    # first second, not after standing over a 30-minute collection window that
    # then refuses to write -- or, worse, dies on an uncaught IsADirectoryError
    # with the harvest already in memory and no file to put it in.
    if out.is_dir():
        parser.error(
            f"--out {out} is a directory. --force would have got past the exists check "
            f"below and the write would then have failed with IsADirectoryError after the "
            f"whole collection window (checked before collecting)"
        )
    if out.exists() and not args.force:
        parser.error(f"{out} exists; pass --force to overwrite (checked before collecting)")
    if not out.parent.is_dir():
        parser.error(
            f"{out.parent} is not an existing directory (checked before collecting)"
        )
    if complaint := _window_complaint(args.coins, args.minutes, args.target):
        # Validated here as well as in `harvest` so it exits 4 rather than
        # tracebacking out of `harvest` with exit 1, the code that means
        # "collected a list and refused to publish it".
        parser.error(f"{complaint} (checked before collecting)")

    try:
        result = harvest(
            coins=args.coins,
            minutes=args.minutes,
            target=args.target,
            ws_url=args.ws_url,
            progress_every_s=args.progress_seconds,
        )
    except FeedUnreachable as exc:
        # Exit 3. Distinct from the shape mismatch because nothing was observed
        # at all: there is no evidence here about the message shape, only about
        # the URL and the network, and sending an operator to this module's
        # parser on that evidence wastes their time.
        print(f"COULD NOT CONNECT -- nothing collected, nothing written.\n\n{exc}")
        return EXIT_UNREACHABLE
    except ImportError as exc:
        # The lazy `websockets` import. Also exit 3: from the operator's side it
        # is the same class of problem -- no connection was ever attempted --
        # and the message says exactly what to install.
        print(f"COULD NOT CONNECT -- the transport is missing.\n\n{exc}")
        return EXIT_UNREACHABLE
    except UnexpectedFeedShape as exc:
        # Exit 2, distinct from the short-list refusal below: this says the
        # collector's assumptions about the feed are wrong and no run of it is
        # currently trustworthy, which is a code-and-documentation problem
        # rather than something a longer window fixes.
        print(f"FEED SHAPE MISMATCH -- nothing collected, nothing written.\n\n{exc}")
        return EXIT_FEED_SHAPE

    print(result.summary())
    try:
        payload = write_address_list(
            result, out, allow_short=args.allow_short, force=args.force, required=required
        )
    except ValueError as exc:
        print(f"REFUSED: {exc}")
        return EXIT_REFUSED
    except OSError as exc:
        # The harvest is already in memory and the operator has already stood
        # over the collection window; losing it to a filesystem error would be
        # the same defect as F7 (a refusal that discards what it took thirty
        # minutes to gather), just with a different cause. Read-only mount and
        # dangling symlink both land here and both pass every pre-flight check,
        # because the pre-flight cannot know whether a write will succeed until
        # it tries one.
        #
        # Dump to stdout rather than to a second guessed path: any fallback
        # location is another write that can fail the same way, and an operator
        # who can see the addresses can save them. Exit distinctly so this is
        # not confused with a short run.
        print(f"COULD NOT WRITE {out}: {type(exc).__name__}: {exc}")
        print(
            f"The harvest is NOT lost -- all {len(result.addresses)} addresses follow "
            "as the exact file that was about to be written. Redirect this output to a "
            "path that is writable, or paste it into an address file; another "
            "collection window is not needed."
        )
        print(json.dumps(build_payload(result, required), indent=2))
        return EXIT_WRITE_FAILED

    print(f"wrote {out} with {len(result.addresses)} addresses")
    if result.anomalies.total:
        # Repeated outside the frame text because the frame is long and this is
        # the line an operator reads. An anomaly count is the difference
        # between a shape that is confirmed and one that is only mostly right.
        one = result.anomalies.total == 1
        print(
            f"NOTE: {result.anomalies.total} anomal{'y' if one else 'ies'} "
            f"{'was' if one else 'were'} counted rather than aborted on, because addresses "
            f"had already been read from '{result.address_field}': "
            f"{'; '.join(result.anomalies.clauses())}"
        )
    print(f"sampling frame: {payload['frame']}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
