"""Where the shadow harness gets its addresses and its books (§3.3).

Two concerns live here, and only one of them is a plumbing problem.

The plumbing: `LiveSnapshotProvider` reads books and prices through the Info
API, and charges every request against a shared weight budget so the
background sweep yields to live users (§5.3, OPEN-QUESTIONS C6).

The awkward one: **where the address list comes from** (OPEN-QUESTIONS B4).
The Info API reads any address but enumerates none — there is no endpoint
that returns a list of accounts. Every available source is biased, and the
bias lands squarely on the variable being calibrated:

  - the leaderboard selects on realised performance, which is precisely the
    outcome the shadow window measures. Calibrating against winners would
    make the model look badly calibrated on losers and vice versa;
  - harvesting from the public trades feed selects on trading frequency, and
    active traders are exactly the cohort §3.3's book-unchanged filter
    discards, so the usable sample would shrink in a way correlated with the
    sampling itself;
  - a hand-curated list selects on whatever the curator noticed.

None of these is neutral, and no amount of code makes them so. What the code
can do is force the choice to be explicit and recorded, so the eventual
calibration score can state its sampling frame instead of implying it had
none. Every source therefore carries a `frame` describing its bias, and the
cron writes that into the run so it is attached to the numbers forever.

The choice has since been made: **the trades feed**, on the grounds that its
bias is awkward rather than circular — activity is not the quantity the score
measures, whereas leaderboard rank is. `risk_engine.market.collect_addresses`
harvests it and writes a file this module's `FileAddressSource` reads, with
the frame generated rather than left to whoever runs it. That does not make
`FileAddressSource` a legacy path: a hand-curated list remains the right
thing for a one-off investigation, and the collector's output is one of these
files rather than a separate kind of source. What is decided is the frame for
the §3.3 window, not the only way an address may enter this system.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from risk_engine.domain.types import AssetSpec, Book, normalise_address
from risk_engine.market.info import InfoClient, WeightBudget
from risk_engine.market.parse import (
    net_external_flow,
    parse_clearinghouse_state,
    parse_meta,
)


class AddressSource(Protocol):
    """A sampling frame, not just a list.

    `frame` is mandatory and is written into every run's provenance. A
    calibration score whose sampling frame is unstated is not a calibration
    score, it is a number.

    `addresses` returns canonical addresses (`normalise_address`), deduped.
    Stating it in the Protocol is what stops two implementations from
    disagreeing about it: they feed the same journal, and an account that
    arrives lowercase from one source and checksummed from another becomes two
    permanent identities in it.
    """

    @property
    def frame(self) -> str: ...

    def addresses(self) -> list[str]: ...


@dataclass(frozen=True, slots=True)
class FileAddressSource:
    """Addresses from a JSON file, with the frame stated by whoever made it.

    The honest default. It does not pretend to solve B4 — it makes the
    operator write down what their list is a sample *of*, and refuses to run
    without that.
    """

    path: Path
    _frame: str = ""

    @property
    def frame(self) -> str:
        return self._frame or self._load()["frame"]

    def _load(self) -> dict:
        payload = json.loads(Path(self.path).read_text())
        if "addresses" not in payload:
            raise ValueError(f"{self.path}: expected an 'addresses' key")
        frame = payload.get("frame", "").strip()
        if not frame:
            raise ValueError(
                f"{self.path}: an address list must state the sampling frame it "
                "represents ('frame' key). An unstated frame becomes an unstated "
                "bias in the published calibration score (OPEN-QUESTIONS B4)."
            )
        return payload

    def addresses(self) -> list[str]:
        payload = self._load()
        # Canonical form as the dedup key: an operator who lists one account
        # twice -- once lowercase, once EIP-55 as a block explorer shows it --
        # gets one address, not two identities in the journal.
        #
        # Load time is the right place to refuse a malformed entry, and the
        # index is what makes the refusal actionable: "invalid address" sends
        # an operator hunting through 200 lines by eye.
        #
        # A typo caught here costs no §5.3 weight -- but only because the two
        # callers were changed to make that true, so both are named here and
        # the claim breaks if either is reordered. `ShadowCron.run_once` reads
        # this list before it calls `specs()` or `spot()`, and `cli._live_world`
        # loads it before `_build_live_bundle`. Neither ordering was the
        # original: the CLI charged 40 weight (meta plus a candle fetch, on a
        # 1-coin universe) before the file was opened at all.
        #
        # Refusing the list refuses the whole sweep, which `run_once` reports
        # as a run-level failure rather than as one more skip line. That is
        # the deliberate choice: a list with a non-address in it is not the
        # sampling frame its `frame` field claims to describe, and sweeping
        # the entries that happen to parse would publish a calibration score
        # against a frame nobody wrote down (OPEN-QUESTIONS B4).
        seen: dict[str, None] = {}
        for i, a in enumerate(payload["addresses"]):
            try:
                seen.setdefault(normalise_address(a), None)
            except ValueError as exc:
                raise ValueError(f"{self.path}: addresses[{i}]: {exc}") from exc
        return list(seen)


@dataclass(frozen=True, slots=True)
class StaticAddressSource:
    """In-memory list, for tests and for one-off runs."""

    _addresses: tuple[str, ...]
    _frame: str

    @property
    def frame(self) -> str:
        if not self._frame.strip():
            raise ValueError("an address source must state its sampling frame")
        return self._frame

    def addresses(self) -> list[str]:
        # Canonicalised and deduped exactly as `FileAddressSource` does. The
        # two used to disagree, and that asymmetry was the realistic way one
        # account acquired a second identity: this class is documented for
        # one-off runs, a one-off run writes to the same permanent journal as
        # the cron, and the address an operator has to hand for a one-off run
        # is the checksummed one they just copied out of a block explorer.
        #
        # The index is reported for the same reason `FileAddressSource` reports
        # it: `run_once` surfaces this message as the reason the whole sweep
        # was refused, and "one of these is not an address" is not a message
        # anyone can act on.
        seen: dict[str, None] = {}
        for i, a in enumerate(self._addresses):
            try:
                seen.setdefault(normalise_address(a), None)
            except ValueError as exc:
                raise ValueError(f"addresses[{i}]: {exc}") from exc
        return list(seen)


class LiveSnapshotProvider:
    """Books and prices from the Info API, under a weight budget (§5.3).

    NOT EXERCISED AGAINST THE LIVE API. `api.hyperliquid.xyz` is blocked at
    the proxy in the environment this was written in (OPEN-QUESTIONS E5), so
    the request shapes come from documentation and this class has never made
    a real call. The fixture path is what the tests drive.

    §5.1's named footgun is NOT enforced by `InfoClient`, contrary to what
    this docstring said before it was checked. An agent address returns a
    well-formed *empty* state that reads as "no positions", and nothing
    detects it — `is_agent_address` is a flag the caller asserts and this
    provider never passes. Nor can it be detected: the venue's response is
    identical to a genuinely flat account's.

    What actually happens to such an address here is worth knowing, because
    it is not the failure the old wording implied. `ShadowCron.run_once`
    skips books with no open positions, so an agent address does not fill the
    journal with flat books — it silently *leaves the sample*, reported as
    "no open positions", byte-identical to a real account that closed out.
    §3.3's 200-address count then comes up short for a reason that appears in
    no output. Addresses from `collect_addresses` are accounts that traded,
    which narrows this; a hand-assembled list does not.
    """

    def __init__(
        self,
        source: AddressSource,
        client: InfoClient | None = None,
        universe: tuple[str, ...] = ("BTC", "ETH", "SOL"),
        budget: WeightBudget | None = None,
    ) -> None:
        self.source = source
        self.client = client or InfoClient(budget=budget)
        self.universe = universe
        self._specs: dict[str, AssetSpec] | None = None
        self._spot: dict[str, float] = {}
        self._spot_at: float = 0.0

    @property
    def frame(self) -> str:
        return self.source.frame

    def addresses(self) -> list[str]:
        return self.source.addresses()

    def specs(self) -> dict[str, AssetSpec]:
        if self._specs is None:
            self._specs = parse_meta(self.client.meta())
        return self._specs

    def spot(self, max_age_s: float = 60.0) -> dict[str, float]:
        """Latest hourly close per asset.

        Cached briefly: a sweep over hundreds of addresses must not re-fetch
        prices per address, and a minute-old price is well inside what a
        24-hour prediction is sensitive to.
        """
        if self._spot and (time.monotonic() - self._spot_at) < max_age_s:
            return dict(self._spot)
        now_ms = int(time.time() * 1000)
        out: dict[str, float] = {}
        for coin in self.universe:
            candles = self.client.candle_snapshot(coin, "1h", now_ms - 6 * 3600 * 1000, now_ms)
            if not candles:
                raise RuntimeError(f"no recent candles for {coin}")
            out[coin] = float(sorted(candles, key=lambda c: int(c["t"]))[-1]["c"])
        self._spot, self._spot_at = out, time.monotonic()
        return dict(out)

    def book(self, address: str) -> Book:
        # Normalised once here so the request and the Book carry the same
        # string. `parse_clearinghouse_state` echoes the address it is handed
        # into `Book.address` and reads none from the response, so taking the
        # raw argument would put the canonical spelling on the wire and a
        # different one on the book the cron then journals.
        address = normalise_address(address)
        state = self.client.clearinghouse_state(address)
        return parse_clearinghouse_state(state, address, datetime.now(timezone.utc))

    def external_flow(self, address: str, since: datetime, until: datetime) -> float:
        """Net deposits minus withdrawals over the window, in USD.

        Refuses rather than returning zero (audit A-04's neighbour): a silent
        zero turns a $50k deposit into a spectacular model failure in the
        calibration score.

        This used to raise `NotImplementedError` unconditionally, and that was
        worse than the silent zero it avoided. `resolve_due` catches per-row
        failures and `_permanent_reason` classifies an unimplemented
        `external_flow` as TRANSIENT by name, so a live shadow run resolved
        zero observations and retried them forever: fourteen days of snapshots
        accumulating against a gate that could never advance, with the reason
        visible only as a repeated traceback in a container log.

        So it is wired, from `userNonFundingLedgerUpdates`, and the shape is
        ASSERTED rather than trusted — the same discipline the trades-feed
        collector uses, for the same reason. A record this cannot read raises
        with the record quoted, which surfaces as a resolver failure naming
        the prediction; it never becomes a zero. `verify --address` checks the
        endpoint's shape in one command, and doing that before starting the
        clock is the difference between finding out now and finding out in
        three weeks.
        """
        raw = self.client.non_funding_ledger_updates(
            normalise_address(address),
            int(since.timestamp() * 1000),
            int(until.timestamp() * 1000),
        )
        return net_external_flow(raw, since, until, address)
