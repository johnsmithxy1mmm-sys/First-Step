"""Hyperliquid Info endpoint client (§5.1) with weight accounting (§5.3).

Read-only, unsigned. Standard library only: this issues one shape of request
(POST JSON to one host) and the weight budget has to be hand-rolled either
way, so an HTTP dependency would buy nothing.

NOT EXERCISED AGAINST THE LIVE API. `api.hyperliquid.xyz` is blocked at the
proxy in the environment this was written in (403), so every parser here is
built against recorded fixtures and the response shapes are taken from the
documentation rather than observed. This is OPEN-QUESTIONS E5 and must be
re-verified where the API is reachable before anything downstream is trusted.

One thing §5.1 flags that IS enforced here rather than left to the caller:
the margin tier table comes from `meta`, never from a constant.

The other one is NOT, and the distinction matters enough to state at length
because this docstring used to claim otherwise. §5.1 warns that
`clearinghouseState` must be queried with the REAL account address, not an
agent address: an agent address returns a well-formed *empty* state, which
reads as "this user has no positions" — the most dangerous possible failure
for a risk tool, since a flat book and an unreadable one are indistinguishable
downstream.

`clearinghouse_state` takes an `is_agent_address` flag and refuses when it is
set, but **nothing detects the case**. The flag is an assertion the caller
makes, it defaults to False, and no shipped caller passes it — the shadow
provider, `verify` and the C5 probe all use the one-argument form. So the
footgun is open on every live path.

It is open because it cannot be closed here. The venue returns byte-identical
responses for an agent address and a genuinely flat account; there is no
read-only signal to branch on, and inventing one would mean guessing. What
narrows it instead is where addresses come from: `collect_addresses` harvests
them from the public trades feed, so every address in a generated list is an
account that *traded*, which an agent address does not do on its own behalf.
A hand-assembled list carries the full risk, and `--allow-short` plus a
hand-written `frame` is exactly the path that skips the collector.

A third, for the same reason as the first: no address reaches the wire in a
spelling this module has not canonicalised. `normalise_address` runs in every
wrapper that takes one, and again in `post` for any `user` field, because an
unrecognised address is answered with that same well-formed empty state
rather than an error -- the venue will never tell us the address was wrong.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from risk_engine.domain.types import normalise_address

MAINNET_URL = "https://api.hyperliquid.xyz/info"
TESTNET_URL = "https://api.hyperliquid-testnet.xyz/info"

#: §5.3, from the published rate-limit page, read 2026-08-03:
#: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/rate-limits-and-user-limits
#:
#:     REST requests share an aggregated weight limit of 1200 per minute.
WEIGHT_BUDGET_PER_MINUTE = 1200

#: The default for documented info requests, per the same page.
INFO_REQUEST_WEIGHT = 20

#: The endpoints that are NOT the default. Until 2026-08-03 this file charged
#: a flat 20 for everything and called the error safe, on the grounds that
#: over-charging self-limits harder than the venue asks. Reading the actual
#: table showed the flat rate is wrong in BOTH directions, and only one of
#: them is the safe one:
#:
#:   - `clearinghouseState` is weight 2, not 20. It is one request per
#:     address and the dominant cost of the shadow sweep, so the sweep was
#:     paying 10x its real cost -- safe, but it is why §3.3's 200-address
#:     floor looked like 27 minutes of budget when it is nearer two.
#:   - `candleSnapshot` and `fundingHistory` bill an EXTRA weight unit per
#:     block of items returned, on top of their base 20. A 90-day hourly
#:     candle fetch is 2160 items and therefore ~56 weight, not 20. That
#:     direction is the dangerous one: the bundle build spends more than it
#:     records, so the reserve §5.3 promises interactive users was being
#:     eaten by an amount nothing could observe.
INFO_REQUEST_WEIGHTS = {
    "l2Book": 2,
    "allMids": 2,
    "clearinghouseState": 2,
    "orderStatus": 2,
    "spotClearinghouseState": 2,
    "exchangeStatus": 2,
    "userRole": 60,
}

#: Endpoints charging one extra weight unit per N items in the RESPONSE, and
#: N. The page words it as "an additional rate limit weight per 20 items
#: returned", which could be read as +1 per 20 items or +20 per 20 items.
#:
#: It is +1, and the deployment itself is the evidence. Under the other
#: reading a single 90-day hourly `candleSnapshot` -- 2160 items -- would
#: cost 2160 weight against a 1200/minute limit, so it could never succeed
#: even from an otherwise idle process. This build has made that exact call
#: on every bundle rebuild for weeks without a 429.
#:
#: Note `userNonFundingLedgerUpdates` is deliberately absent. The page lists
#: `nonUserFundingUpdates`, which is a different endpoint with a confusingly
#: similar name, and reading one for the other would invent a surcharge on
#: the call B2's resolver makes per address.
INFO_ITEMS_PER_EXTRA_WEIGHT = {
    "recentTrades": 20,
    "historicalOrders": 20,
    "userFills": 20,
    "userFillsByTime": 20,
    "fundingHistory": 20,
    "userFunding": 20,
    "nonUserFundingUpdates": 20,
    "twapHistory": 20,
    "userTwapSliceFills": 20,
    "userTwapSliceFillsByTime": 20,
    "delegatorHistory": 20,
    "delegatorRewards": 20,
    "validatorStats": 20,
    "candleSnapshot": 60,
}


def info_request_weight(request_type: str) -> int:
    """The weight charged BEFORE a request, from its type alone."""
    return INFO_REQUEST_WEIGHTS.get(request_type, INFO_REQUEST_WEIGHT)


def info_response_surcharge(request_type: str, response: object) -> int:
    """The extra weight a response's own length incurred.

    Charged after the fact because it cannot be known before: it is a
    function of how many items came back. Returns 0 for every endpoint that
    does not bill this way, which is most of them.
    """
    per = INFO_ITEMS_PER_EXTRA_WEIGHT.get(request_type)
    if per is None or not isinstance(response, list):
        return 0
    return len(response) // per


#: How the 1200 is divided, as reserved fractions. These are complements on
#: purpose: 25% background plus 75% interactive is exactly the limit, so the
#: deployment's ceiling matches what the venue will actually serve.
#:
#: The serving engine used to take a budget with NO reserve, i.e. all 1200,
#: while the shadow jobs took 300 each on top -- a combined ceiling of 1500
#: against a limit of 1200 (OPEN-QUESTIONS C6). Its realised traffic is far
#: below its ceiling (the five-minute rebuild, nothing on the request path),
#: which is why this was never observed; a ceiling nobody reaches is still
#: the wrong ceiling to publish.
#:
#: The two shadow jobs share ONE 300/minute pool rather than taking 300 each
#: -- see `shadow/weight_ledger.py` for why the pool is shared and not
#: divided.
SHADOW_RESERVED_FRACTION = 0.75
SERVING_RESERVED_FRACTION = 0.25


class RateLimitExceeded(RuntimeError):
    pass


@dataclass
class WeightBudget:
    """A sliding one-minute weight window with reserved headroom.

    §5.3 requires the shadow cron to yield to live users. It does that by
    taking a budget with `reserved_fraction` set high: background work is
    refused once it would eat into the reserve, while the interactive path
    takes a budget with no reserve.
    """

    limit_per_minute: int = WEIGHT_BUDGET_PER_MINUTE
    reserved_fraction: float = 0.0
    _events: list[tuple[float, int]] = field(default_factory=list, repr=False)

    def _prune(self, now: float) -> None:
        cutoff = now - 60.0
        self._events = [(t, w) for t, w in self._events if t > cutoff]

    def spent(self, now: float | None = None) -> int:
        now = now if now is not None else time.monotonic()
        self._prune(now)
        return sum(w for _, w in self._events)

    def available(self, now: float | None = None) -> int:
        usable = int(self.limit_per_minute * (1.0 - self.reserved_fraction))
        return max(0, usable - self.spent(now))

    def charge(self, weight: int, now: float | None = None) -> None:
        now = now if now is not None else time.monotonic()
        if weight > self.available(now):
            raise RateLimitExceeded(
                f"weight {weight} exceeds remaining {self.available(now)} "
                f"(limit {self.limit_per_minute}/min, reserved {self.reserved_fraction:.0%})"
            )
        self._events.append((now, weight))

    def charge_incurred(self, weight: int, now: float | None = None) -> None:
        """Record weight the venue has already counted. Never refuses.

        Some endpoints bill per item RETURNED, so their true cost is not
        knowable until the response is in hand. Refusing at that point would
        throw away a response already paid for, and pretending it was free
        would understate the window — which on this side of the accounting
        means eating the reserve §5.3 sets aside for interactive users.

        So it is recorded and allowed to overshoot. The window then reads
        fuller than `usable`, `available()` floors at zero, and the NEXT
        charge waits — which is the correct consequence, one request late.
        """
        now = now if now is not None else time.monotonic()
        self._events.append((now, weight))


#: How a paced charge waits out a spent window: poll interval, and the
#: ceiling past which a full window stops being a pace and starts being a
#: fault worth reporting.
PACED_BUDGET_WAIT_S = 5.0
PACED_BUDGET_MAX_WAIT_S = 10 * 60.0


class PacedBudget:
    """Any weight budget, waiting out a spent window rather than raising.

    "A rate limit is a pace, not an error" is settled for the background
    jobs -- the sweep, the resolver and the bundle build all wait -- but it
    was settled three separate times, once per caller, each time after the
    unpaced version had already killed a run. Wrapping the budget puts the
    wait where the charge is, so the next caller inherits it instead of
    rediscovering it.

    The ceiling stays. A window that has not refilled in ten minutes is not
    congestion, it is a pool somebody else is holding, and the callers have
    honest ways to say so; an unbounded wait would turn that into a process
    asleep forever.

    Deliberately duck-typed rather than a `WeightBudget` subclass, for the
    same reason `SharedWeightBudget` is: it must be able to wrap either, and
    inheriting a `_events` list that stays empty would look authoritative.
    """

    def __init__(
        self,
        inner: "WeightBudget | PacedBudget",
        max_wait_s: float = PACED_BUDGET_MAX_WAIT_S,
        wait_s: float = PACED_BUDGET_WAIT_S,
    ) -> None:
        self.inner = inner
        self.max_wait_s = max_wait_s
        self.wait_s = wait_s

    def spent(self, now: float | None = None) -> int:
        return self.inner.spent(now)

    def available(self, now: float | None = None) -> int:
        return self.inner.available(now)

    def charge_incurred(self, weight: int, now: float | None = None) -> None:
        # Never refuses, so there is nothing to pace.
        self.inner.charge_incurred(weight, now)

    def charge(self, weight: int, now: float | None = None) -> None:
        # An explicit `now` is a frozen clock -- a test or a deterministic
        # replay -- and a window measured against it never refills. Waiting
        # would spend the whole ceiling to arrive at the same refusal, so
        # such a caller gets the unpaced answer it asked for.
        if now is not None:
            self.inner.charge(weight, now)
            return
        deadline = time.monotonic() + self.max_wait_s
        while True:
            try:
                self.inner.charge(weight)
                return
            except RateLimitExceeded:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(self.wait_s)


class InfoClient:
    def __init__(
        self,
        url: str = MAINNET_URL,
        budget: WeightBudget | None = None,
        timeout: float = 10.0,
        max_retries: int = 3,
    ) -> None:
        self.url = url
        self.budget = budget or WeightBudget()
        self.timeout = timeout
        self.max_retries = max_retries

    def post(self, payload: dict, weight: int | None = None) -> dict | list:
        """POST one Info request, charging §5.3's published weight for it.

        `weight` defaults to the type's own published weight rather than to a
        flat 20 -- see `INFO_REQUEST_WEIGHTS`. An explicit value still wins,
        for a caller that knows better than the table.
        """
        # Backstop, and deliberately *before* `charge`: a malformed address
        # must cost no weight, or the shadow sweep pays §5.3 budget for a
        # request it was never going to be able to make (cron.py charges per
        # address and truncates the sweep when the budget runs out).
        #
        # The typed wrappers below already normalise, so for every method that
        # exists today this is redundant. It is here for the next wrapper
        # someone adds and for callers that assemble a payload by hand:
        # `user` is the only address-carrying field in the Info request
        # shapes, so one check covers all of them. Rebuilt rather than mutated
        # -- the caller's dict is theirs.
        if "user" in payload:
            payload = {**payload, "user": normalise_address(payload["user"])}
        request_type = str(payload.get("type", ""))
        if weight is None:
            weight = info_request_weight(request_type)
        self.budget.charge(weight)
        body = json.dumps(payload).encode()
        # S310: the scheme is fixed by MAINNET_URL / TESTNET_URL, which are
        # module constants. No caller-supplied URL reaches this.
        req = urllib.request.Request(  # noqa: S310
            self.url, data=body, headers={"Content-Type": "application/json"}
        )
        last: Exception | None = None
        for attempt in range(self.max_retries):
            if attempt:
                # Every retry is another request ON THE WIRE, and the venue
                # counts requests, not intentions. Charging once before the
                # loop meant up to `max_retries` real requests per 20-weight
                # charge -- under network flakiness the shadow sweep's promised
                # 300/min became as much as 900/min, eating the very reserve
                # §5.3 sets aside for interactive users. A retry that cannot
                # be paid for waits for the window rather than being sent.
                #
                # Note this makes `RateLimitExceeded` reachable from inside a
                # retry, which is correct: the paced callers treat it as "wait
                # and try again", which is what a spent window means.
                self.budget.charge(weight)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                    payload_out = json.load(resp)
                # Some endpoints bill per item returned, which is knowable
                # only now. Recorded rather than refused: the request is
                # already on the wire and the venue has already counted it.
                extra = info_response_surcharge(request_type, payload_out)
                if extra:
                    self.budget.charge_incurred(extra)
                return payload_out
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2.0**attempt)
        raise RuntimeError(f"info request {payload.get('type')} failed: {last}") from last

    # -- typed wrappers -------------------------------------------------

    def meta(self) -> dict:
        return self.post({"type": "meta"})  # type: ignore[return-value]

    def clearinghouse_state(self, address: str, is_agent_address: bool = False) -> dict:
        # The agent refusal stays ahead of the format check on purpose: an
        # agent address is a *well-formed* address, so normalising first would
        # answer the caller who has the right format and the wrong account
        # with a generic complaint instead of §5.1's specific one. Neither
        # path reaches `post`, so neither spends weight.
        if is_agent_address:
            raise ValueError(
                "clearinghouseState must be queried with the real account address; "
                "an agent address returns an empty state that reads as 'no positions' (§5.1)"
            )
        address = normalise_address(address)
        return self.post({"type": "clearinghouseState", "user": address})  # type: ignore[return-value]

    def open_orders(self, address: str) -> list:
        address = normalise_address(address)
        return self.post({"type": "openOrders", "user": address})  # type: ignore[return-value]

    def candle_snapshot(self, coin: str, interval: str, start_ms: int, end_ms: int) -> list:
        return self.post(  # type: ignore[return-value]
            {
                "type": "candleSnapshot",
                "req": {"coin": coin, "interval": interval,
                        "startTime": start_ms, "endTime": end_ms},
            }
        )

    def user_funding(self, address: str, start_ms: int, end_ms: int | None = None) -> list:
        """Per-user funding payments. Ground truth for OPEN-QUESTIONS C5.

        `fundingHistory` gives the venue-wide rate; this gives what a
        specific account actually paid or received, which is what makes the
        isolated-vs-cross accounting question answerable by observation
        instead of by opening a position.
        """
        address = normalise_address(address)
        req: dict = {"type": "userFunding", "user": address, "startTime": start_ms}
        if end_ms is not None:
            req["endTime"] = end_ms
        return self.post(req)  # type: ignore[return-value]

    def non_funding_ledger_updates(
        self, address: str, start_ms: int, end_ms: int | None = None
    ) -> list:
        """Deposits, withdrawals and transfers — everything that moves equity
        for a reason the model does not predict (OPEN-QUESTIONS B2).

        Named `userNonFundingLedgerUpdates` by the venue, and the "non
        funding" is the point: `user_funding` above covers the payments the
        model DOES predict, and this covers the ones it must not be scored
        on. A $50k deposit landing inside a 24-hour observation window is a
        spectacular apparent model failure if it is scored as an equity move.

        Shape is written from documentation and asserted by the caller rather
        than trusted here — see `LiveSnapshotProvider.external_flow`, which
        refuses a record it cannot read instead of treating it as zero.
        """
        address = normalise_address(address)
        req: dict = {
            "type": "userNonFundingLedgerUpdates",
            "user": address,
            "startTime": start_ms,
        }
        if end_ms is not None:
            req["endTime"] = end_ms
        return self.post(req)  # type: ignore[return-value]

    def funding_history(self, coin: str, start_ms: int, end_ms: int | None = None) -> list:
        req = {"type": "fundingHistory", "coin": coin, "startTime": start_ms}
        if end_ms is not None:
            req["endTime"] = end_ms
        return self.post(req)  # type: ignore[return-value]
