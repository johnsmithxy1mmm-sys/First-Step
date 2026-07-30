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

The failure this module is built to make impossible is the quiet one: a
collector that connects to the wrong URL, or subscribes with a key the venue
acknowledges and never delivers on, or reads an address field that does not
exist, and then writes a valid, empty, confidently-framed address list. That
file loads without complaint (`FileAddressSource` has no minimum count), the
nightly sweep snapshots nothing, `progress()` shows a gate that never moves,
and the reason sits three layers away. Every path that reaches zero addresses
here raises instead, and the write refuses an empty list even when the
operator has passed `--allow-short`.

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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from risk_engine.domain.types import normalise_address

#: The only WebSocket URL recorded anywhere in this repository, and it is
#: recorded as an instruction nobody has been able to follow: `verify.py`'s
#: `check_webdata3` returns UNCHECKABLE because the harness speaks only the
#: Info POST API and the host is blocked at this environment's proxy. Treat it
#: as unconfirmed. There is deliberately no testnet constant to match
#: `info.TESTNET_URL`: inferring `wss://api.hyperliquid-testnet.xyz/ws` from
#: the Info host convention is a guess, and a guessed URL that happens to
#: resolve to something is worse than `--ws-url` typed by an operator who
#: knows what they are connecting to.
MAINNET_WS_URL = "wss://api.hyperliquid.xyz/ws"

#: The subscription and channel name for public trades. ASSUMED. C4 records
#: `{"method": "subscribe", "subscription": {"type": "webData3"}}` as the only
#: envelope anyone has written down here, so the outer `method`/`subscription`
#: shape is borrowed from it and the inner `type`/`coin` pair is inference. If
#: the venue names either differently the subscription is either rejected --
#: which surfaces as an error frame and a loud abort -- or acknowledged and
#: never delivered, which surfaces as the no-trade-records abort below.
TRADES_CHANNEL = "trades"

#: Where a trade record is expected to carry the accounts that traded. This is
#: THE unverified assumption of the whole module: `git grep -E '\busers\b'`
#: over this repository finds only English prose, never a JSON field, so there
#: is no fixture, no documentation and no observation behind this tuple.
#: `users` first because a trade has two sides; `user` accepted because the
#: webData2 subscription is conventionally keyed on a singular `user` and a
#: trade record may well mirror it. Whichever one is actually found is
#: recorded in the result and stated in the generated frame, so a reader of
#: the address list can tell which shape the venue really produced. Extend
#: this tuple when the live shape is known -- do not paper over a different
#: field name further downstream, where it becomes an address list nobody can
#: account for.
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


class UnexpectedFeedShape(RuntimeError):
    """The feed was not shaped the way this collector assumed.

    A distinct type rather than a bare RuntimeError because the caller has to
    treat it differently from a short run: a short run is a judgement call an
    operator can override with `--allow-short`, while this means the frame on
    the wire does not match what was parsed and no address list from this run
    is trustworthy at any length.
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
    address_field: str = ""
    examples: list[str] = field(default_factory=list)
    bad_address_examples: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class HarvestResult:
    """What one collection run saw, in enough detail to be published.

    Deliberately keeps the counts, not just the addresses. The addresses alone
    cannot be audited: 200 addresses harvested from 40 000 trades over half an
    hour and 200 harvested from 210 trades in the first four seconds are
    different samples of different things, and a reader of the calibration
    score has no way to tell them apart afterwards unless the run says so.
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

    @property
    def window_seconds(self) -> float:
        return (self.ended_at - self.started_at).total_seconds()

    def summary(self) -> str:
        return (
            f"{len(self.addresses)} distinct addresses from {self.trade_records} trade "
            f"records over {self.frames} frames in {self.window_seconds / 60:.1f} min "
            f"({', '.join(self.coins)}); addresses read from '{self.address_field}'; "
            f"{_values(self.unparseable_addresses)} refused by normalise_address; "
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


def _trade_records(payload: object) -> list[dict] | None:
    """The trade records in one frame, or None if this frame is not trades.

    Permissive about the envelope, strict about the contents. The outer
    wrapper is a transport detail that a venue can change without changing
    what a trade *is*, and the repo has no record of it either way, so both
    the documented-by-convention `{"channel": ..., "data": [...]}` form and a
    bare array of records are accepted. The address field inside a record is
    the load-bearing fact and is not guessed at all -- see
    `_address_values`.
    """
    if isinstance(payload, dict):
        if payload.get("channel") != TRADES_CHANNEL:
            return None
        data = payload.get("data")
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        return None
    if isinstance(payload, list):
        records = [r for r in payload if isinstance(r, dict)]
        return records or None
    return None


def _address_values(record: dict) -> tuple[str, list[object]] | None:
    """The account addresses on a trade record, with the field they came from.

    Returns None when no expected field is present, which the caller turns
    into a loud abort. A scalar field is read as one address and a sequence as
    many; the *contents* are not inspected here on purpose, so that a single
    junk entry inside an otherwise good list travels to `normalise_address`
    and is counted as one refused value rather than condemning the whole
    record as the wrong shape. The distinction the two paths draw is between
    "this field does not exist" (a shape fault, fatal) and "this field holds
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
    stopped = ""

    async with transport.connect() as ws:
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
                    f"or a bare array of trade records. RECEIVED, verbatim: "
                    f"{counters.examples or 'nothing decodable'}. A subscription that is "
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
                    f"{counters.unparseable_addresses} unparseable"
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
                raise UnexpectedFeedShape(
                    f"the feed answered with an error frame: {error}. The subscription "
                    f"sent was {_subscribe_payload(coins[0])} (one per coin). EXPECTED: "
                    f"an acknowledgement, then trade frames. The channel name "
                    f"'{TRADES_CHANNEL}' and the 'coin' key are inference from "
                    f"OPEN-QUESTIONS C4's webData3 snippet, not documentation, so the "
                    f"venue rejecting them is a likely outcome and is reported as a fault "
                    f"in this collector rather than as an empty sample."
                )

            records = _trade_records(payload)
            if records is None:
                counters.other_frames += 1
                _remember(counters.examples, raw)
                continue

            for record in records:
                counters.trade_records += 1
                if first_trade_at is None:
                    first_trade_at = now

                found = _address_values(record)
                if found is None:
                    raise UnexpectedFeedShape(
                        f"a trade record carried no account address. EXPECTED one of the "
                        f"fields {list(TRADE_ADDRESS_FIELDS)}, holding an address or a "
                        f"list of them. RECEIVED: {_truncate(json.dumps(record))} "
                        f"(keys: {sorted(record)}). This is exactly the assumption "
                        f"OPEN-QUESTIONS B4 flags as needing verification -- that a public "
                        f"trade names its participants -- and it did not hold. If the venue "
                        f"names the field differently, add it to TRADE_ADDRESS_FIELDS here, "
                        f"where the frame string can record which field the addresses came "
                        f"from; do not translate it downstream, where the published "
                        f"calibration score would carry a sample nobody can account for."
                    )

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

    `clock` and `utcnow` are injectable so the time-budget path and the
    generated frame's window are testable without a wall-clock wait. The
    project has one test already failing intermittently on a shared box
    because it measures elapsed time; adding a 30-minute one, or a sleeping
    one, would be adding to that problem.
    """
    if not coins:
        raise ValueError("at least one coin is required; the feed is subscribed per coin")
    if minutes <= 0:
        raise ValueError(f"--minutes must be positive, got {minutes}")
    if target <= 0:
        raise ValueError(f"--target must be positive, got {target}")

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
            f"address values were {counters.bad_address_examples or 'none'}."
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
    )


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


def build_frame(result: HarvestResult, required: int) -> str:
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
    """
    n = len(result.addresses)
    window = (
        f"{result.started_at.strftime('%Y-%m-%dT%H:%M:%SZ')} to "
        f"{result.ended_at.strftime('%Y-%m-%dT%H:%M:%SZ')} UTC "
        f"({result.window_seconds / 60:.1f} min)"
    )
    parts = [
        f"{n} distinct accounts observed trading {', '.join(result.coins)} on "
        f"Hyperliquid's public trades WebSocket feed ({result.ws_url}, subscription "
        f"type '{TRADES_CHANNEL}', addresses read from each trade record's "
        f"'{result.address_field}' field), {window}, from {result.trade_records} trade "
        f"records over {result.frames} frames; collection stopped because "
        f"{result.stopped_because}.",
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
        "expects them -- but the URL, the subscription name and the record layout are not "
        "confirmed against Hyperliquid documentation from the environment this was built "
        "in, where the API is blocked at the proxy (OPEN-QUESTIONS C4, E5).",
    ]
    if n < required:
        parts.append(
            f"SHORT: {n} addresses is below §3.3's requirement of {required} distinct "
            "addresses, and this file was written with --allow-short. It cannot open the "
            "Phase 4 gate on its own."
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


def build_payload(result: HarvestResult, required: int) -> dict:
    """The file `FileAddressSource` reads, plus a record of how it was made.

    Two load-bearing keys and one extra. `frame` and `addresses` are what the
    loader validates; `_provenance` is ignored by it and is here because the
    template written by `shadow init-addresses` carries `_frame_help` and
    `_addresses_help` prose telling a human what to fill in, and copying
    fill-me-in instructions into an already-filled generated file would leave
    a reader unable to tell whether a human curated this list. The counts a
    reader would otherwise have to take on trust from the frame sentence go
    here as data.
    """
    return {
        "frame": build_frame(result, required),
        "_provenance": {
            "collector": "risk_engine.market.collect_addresses",
            "source": "hyperliquid public trades websocket",
            "ws_url": result.ws_url,
            "subscription_type": TRADES_CHANNEL,
            "address_field": result.address_field,
            "coins": list(result.coins),
            "window_start_utc": result.started_at.isoformat(),
            "window_end_utc": result.ended_at.isoformat(),
            "frames": result.frames,
            "trade_records": result.trade_records,
            "addresses_collected": len(result.addresses),
            "unparseable_values_skipped": result.unparseable_addresses,
            "target": result.target,
            "gate_required_addresses": required,
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
    """
    required = gate_required_addresses() if required is None else required
    n = len(result.addresses)
    if n == 0:
        raise ValueError(
            "refusing to write an empty address list. It would load without complaint "
            "and every sweep against it would snapshot nothing, showing up as a gate "
            "that never advances rather than as a collector that failed."
        )
    if n < required and not allow_short:
        raise ValueError(
            f"found {n} distinct addresses, §3.3 needs {required} -- nothing written. "
            f"The gate counts addresses with a resolved observation, and the sweep drops "
            f"flat and zero-equity accounts before that, so aim above {required} rather "
            f"than at it ({DEFAULT_TARGET} is the top of §3.3's range). Re-run with a "
            f"longer --minutes or more --coins, or pass --allow-short to write it anyway "
            f"-- the frame will record that it is short and cannot open the gate."
        )
    if path.exists() and not force:
        raise ValueError(f"{path} exists; pass --force to overwrite")

    payload = build_payload(result, required)
    # indent=2 with a trailing newline, matching `shadow init-addresses`, so a
    # generated list and a hand-filled template diff against each other
    # without a whitespace storm hiding the change that matters.
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _parse_coins(raw: str) -> tuple[str, ...]:
    coins = tuple(c.strip().upper() for c in raw.split(",") if c.strip())
    if not coins:
        raise argparse.ArgumentTypeError("--coins needs at least one coin name")
    return coins


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="risk_engine.market.collect_addresses",
        description=(
            "harvest a §3.3 shadow address list from Hyperliquid's public trades feed "
            "(OPEN-QUESTIONS B4)"
        ),
    )
    parser.add_argument("--out", required=True, help="where to write the address list")
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

    out = Path(args.out)
    required = gate_required_addresses()

    # Both checks BEFORE connecting. An operator who mistypes the output path
    # or forgets --force should learn that in the first second, not after
    # standing over a 30-minute collection window that then refuses to write.
    if out.exists() and not args.force:
        parser.error(f"{out} exists; pass --force to overwrite (checked before collecting)")
    if not out.parent.exists():
        parser.error(f"{out.parent} does not exist (checked before collecting)")

    try:
        result = harvest(
            coins=args.coins,
            minutes=args.minutes,
            target=args.target,
            ws_url=args.ws_url,
            progress_every_s=args.progress_seconds,
        )
    except UnexpectedFeedShape as exc:
        # Exit 2, distinct from the short-list refusal below: this says the
        # collector's assumptions about the feed are wrong and no run of it is
        # currently trustworthy, which is a code-and-documentation problem
        # rather than something a longer window fixes.
        print(f"FEED SHAPE MISMATCH -- nothing collected, nothing written.\n\n{exc}")
        return 2

    print(result.summary())
    try:
        payload = write_address_list(
            result, out, allow_short=args.allow_short, force=args.force, required=required
        )
    except ValueError as exc:
        print(f"REFUSED: {exc}")
        return 1

    print(f"wrote {out} with {len(result.addresses)} addresses")
    print(f"sampling frame: {payload['frame']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
