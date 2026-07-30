"""Hyperliquid Info endpoint client (§5.1) with weight accounting (§5.3).

Read-only, unsigned. Standard library only: this issues one shape of request
(POST JSON to one host) and the weight budget has to be hand-rolled either
way, so an HTTP dependency would buy nothing.

NOT EXERCISED AGAINST THE LIVE API. `api.hyperliquid.xyz` is blocked at the
proxy in the environment this was written in (403), so every parser here is
built against recorded fixtures and the response shapes are taken from the
documentation rather than observed. This is OPEN-QUESTIONS E5 and must be
re-verified where the API is reachable before anything downstream is trusted.

Two things §5.1 flags that are enforced here rather than left to the caller:

  - `clearinghouseState` must be queried with the REAL account address, not
    the agent address. An agent address returns a well-formed *empty* state,
    which silently reads as "this user has no positions" -- the most
    dangerous possible failure for a risk tool. `fetch_clearinghouse_state`
    refuses an address flagged as an agent.
  - the margin tier table comes from `meta`, never from a constant.

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

#: §5.3. The published budget is 1200 weight/minute per IP; info requests
#: cost about 20. These are the numbers the governor is built around and,
#: like everything else here, need confirming against the live API.
WEIGHT_BUDGET_PER_MINUTE = 1200
INFO_REQUEST_WEIGHT = 20


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

    def post(self, payload: dict, weight: int = INFO_REQUEST_WEIGHT) -> dict | list:
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
        self.budget.charge(weight)
        body = json.dumps(payload).encode()
        # S310: the scheme is fixed by MAINNET_URL / TESTNET_URL, which are
        # module constants. No caller-supplied URL reaches this.
        req = urllib.request.Request(  # noqa: S310
            self.url, data=body, headers={"Content-Type": "application/json"}
        )
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                    return json.load(resp)
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

    def funding_history(self, coin: str, start_ms: int, end_ms: int | None = None) -> list:
        req = {"type": "fundingHistory", "coin": coin, "startTime": start_ms}
        if end_ms is not None:
            req["endTime"] = end_ms
        return self.post(req)  # type: ignore[return-value]
