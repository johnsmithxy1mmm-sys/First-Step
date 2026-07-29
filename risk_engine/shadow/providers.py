"""Where the shadow harness gets its addresses and its books (§3.3).

Two concerns live here, and only one of them is a plumbing problem.

The plumbing: `LiveSnapshotProvider` reads books and prices through the Info
API, and charges every request against a shared weight budget so the
background sweep yields to live users (§5.3, OPEN-QUESTIONS C6).

The unsolved one: **where the address list comes from** (OPEN-QUESTIONS B4).
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
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from risk_engine.domain.types import AssetSpec, Book
from risk_engine.market.info import InfoClient, WeightBudget
from risk_engine.market.parse import parse_clearinghouse_state, parse_meta


class AddressSource(Protocol):
    """A sampling frame, not just a list.

    `frame` is mandatory and is written into every run's provenance. A
    calibration score whose sampling frame is unstated is not a calibration
    score, it is a number.
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
        seen: dict[str, None] = {}
        for a in payload["addresses"]:
            seen.setdefault(str(a).lower(), None)
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
        return list(self._addresses)


class LiveSnapshotProvider:
    """Books and prices from the Info API, under a weight budget (§5.3).

    NOT EXERCISED AGAINST THE LIVE API. `api.hyperliquid.xyz` is blocked at
    the proxy in the environment this was written in (OPEN-QUESTIONS E5), so
    the request shapes come from documentation and this class has never made
    a real call. The fixture path is what the tests drive.

    §5.1's named footgun is enforced by `InfoClient`: an agent address
    returns a well-formed *empty* state, which reads as "no positions". For a
    shadow sweep that would silently fill the calibration journal with
    flat books.
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
        state = self.client.clearinghouse_state(address)
        return parse_clearinghouse_state(state, address, datetime.now(timezone.utc))

    def external_flow(self, address: str, since: datetime, until: datetime) -> float:
        """Net deposits minus withdrawals over the window, in USD.

        Refuses rather than returning zero (audit A-04's neighbour): a silent
        zero turns a $50k deposit into a spectacular model failure in the
        calibration score. `userNonFundingLedgerUpdates` is the documented
        source; it is not wired because it has never been seen, and guessing
        its shape here would produce exactly the silent zero this is meant to
        prevent.
        """
        raise NotImplementedError(
            "external_flow needs userNonFundingLedgerUpdates, whose response shape has "
            "not been verified against the live API (OPEN-QUESTIONS E5). Returning 0.0 "
            "instead would score every deposit as a model error."
        )
