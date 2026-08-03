"""The control experiment for OPEN-QUESTIONS C4.

    pip install "websockets>=12.0"
    python scripts/probe_webdata.py 0x<any-real-account>

A bare `{"type": "webData3"}` subscribe returns

    Error parsing JSON into valid websocket request: ...

which was once read as "webData3 does not exist". It does not establish that.
The venue is complaining that the REQUEST did not parse, and a missing
required field produces the same complaint as an unknown type -- `webData2`
is keyed on `user`, and that payload carries no `user`.

So this sends three, and the first is the control:

  1. `webData2` with no user   -- the subscription KNOWN to exist, malformed.
                                  If this errors too, the error is about the
                                  payload and says nothing about any type.
  2. `webData3` with no user   -- what was run before, kept for comparison.
  3. `webData3` with a user    -- the well-formed question C4 actually asks.

Read row 1 first. If it errors, rows 2 and 3 are the whole answer: 3 tells
you whether `webData3` exists, and 2 is noise. If row 1 succeeds, then a bare
subscribe IS well-formed here, row 2 is meaningful on its own, and rows 2
and 3 should agree.
"""

import asyncio
import json
import sys

URL = "wss://api.hyperliquid.xyz/ws"
TIMEOUT_S = 10.0


async def probe(sub: dict) -> str:
    import websockets

    async with websockets.connect(URL) as ws:
        await ws.send(json.dumps({"method": "subscribe", "subscription": sub}))
        try:
            raw = await asyncio.wait_for(ws.recv(), TIMEOUT_S)
        except TimeoutError:
            return f"SILENCE for {TIMEOUT_S:.0f}s (not a refusal)"
    msg = json.loads(raw)
    if msg.get("channel") == "error":
        return f"REJECTED  {msg.get('data')!r}"
    return f"ACCEPTED  {raw[:200]}"


async def main(address: str | None) -> None:
    cases = [
        ("control: webData2, no user", {"type": "webData2"}),
        ("webData3, no user", {"type": "webData3"}),
    ]
    if address:
        cases.append(("webData3, with user", {"type": "webData3", "user": address}))
        cases.append(("control: webData2, with user", {"type": "webData2", "user": address}))
    else:
        print("no address given: the two rows that matter most are skipped.\n"
              "    python scripts/probe_webdata.py 0x<any-real-account>\n")
    for label, sub in cases:
        try:
            result = await probe(sub)
        except Exception as exc:  # a transport failure is not evidence either way
            result = f"PROBE FAILED  {type(exc).__name__}: {exc}"
        print(f"{label:32} {result}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else None))
