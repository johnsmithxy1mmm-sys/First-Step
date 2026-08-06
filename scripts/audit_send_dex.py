"""Decide what `dex: ""` means, from the venue's own records (OPEN-QUESTIONS B2).

    python scripts/audit_send_dex.py 0x<address>

`send` records name a `sourceDex` and a `destinationDex`, and one of the values
they carry is the empty string. Reading it as the primary perp account rather
than as spot moved one real account's 90-day net external flow from
-$18,472,046 to +$21,154,705 — a $39.6M swing that flips the sign. A number
that large is the correction working or the correction inventing itself, and
nothing in the totals distinguishes those two.

**The decisive test is the token.** A perpetuals account is margined in USDC
and holds nothing else; there is no way to send HYPE *into* a perp dex. So:

  - if every record naming `""` carries `token: "USDC"`, `""` is consistent
    with being a USDC-margined perp venue;
  - if any record sends a non-USDC token into `""`, then `""` cannot be a perp
    account, the primary-perp reading is wrong, and the $39.6M is fabricated.

That is a falsification test rather than a confirmation: passing it does not
prove `""` is the primary perp, because a USDC-only spot transfer would pass
it too. It can only catch the error, which is the direction that matters —
this build currently ACTS on the primary-perp reading, so the useful question
is whether the venue contradicts it.

Prints the dex pairs by USD volume so the shape of the account's flows is
visible alongside the verdict, because "40M arrived from nowhere" and "40M
arrived from this account's own spot balance" look identical in a total.
"""

from __future__ import annotations

import pathlib
import sys
import time
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from risk_engine.domain.types import normalise_address
from risk_engine.market.info import InfoClient

HOUR_MS = 3_600_000
WINDOW_DAYS = 90


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python scripts/audit_send_dex.py 0x<address>")
        return 2
    address = normalise_address(args[0])

    now_ms = int(time.time() * 1000)
    rows = InfoClient().non_funding_ledger_updates(
        address, now_ms - WINDOW_DAYS * 24 * HOUR_MS, now_ms) or []

    pairs: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "usd": 0.0, "tokens": set()})
    non_usdc_into_empty: list[dict] = []
    non_usdc_out_of_empty: list[dict] = []

    for row in rows:
        d = row.get("delta") or {}
        if d.get("type") != "send":
            continue
        # Audit F-11: `.get(..., "")` conflated a MISSING field with the
        # explicit empty string — but the empty string is exactly the value
        # under audit ("" = primary perp). The classifier in parse.py raises
        # on a missing dex; this tool, which exists to audit that classifier,
        # was silently bucketing the missing case into the primary-perp bin,
        # i.e. it was looser than the code it judges. A missing field is its
        # own bucket, and the decisive test skips it rather than counting it
        # for either side.
        raw_src, raw_dst = d.get("sourceDex"), d.get("destinationDex")
        src = "<missing>" if raw_src is None else str(raw_src).strip().lower()
        dst = "<missing>" if raw_dst is None else str(raw_dst).strip().lower()
        token = str(d.get("token") or "?")
        try:
            usd = abs(float(d.get("usdcValue") or 0.0))
        except (TypeError, ValueError):
            usd = 0.0
        side = "sent" if str(d.get("user", "")).lower() == address.lower() else "received"

        key = (src if src else "<empty>", dst if dst else "<empty>", side)
        pairs[key]["n"] += 1
        pairs[key]["usd"] += usd
        pairs[key]["tokens"].add(token)

        if token.upper() != "USDC":
            if dst == "":
                non_usdc_into_empty.append(row)
            if src == "":
                non_usdc_out_of_empty.append(row)

    if not pairs:
        print(f"no `send` records for {address} in {WINDOW_DAYS}d — nothing to decide from")
        return 1

    print(f"`send` records for {address}, last {WINDOW_DAYS}d\n")
    print(f"{'sourceDex':<12} {'destDex':<12} {'side':<9} {'n':>5} {'USD':>16}  tokens")
    for key in sorted(pairs, key=lambda k: -pairs[k]["usd"]):
        src, dst, side = key
        v = pairs[key]
        toks = ",".join(sorted(v["tokens"]))[:30]
        print(f"{src:<12} {dst:<12} {side:<9} {v['n']:>5} {v['usd']:>16,.2f}  {toks}")

    print("\n--- the decisive test ---")
    bad = non_usdc_into_empty + non_usdc_out_of_empty
    if bad:
        print(
            f"FALSIFIED: {len(bad)} record(s) move a NON-USDC token in or out of "
            f'dex "". A perp account is margined in USDC and can hold nothing '
            f'else, so "" is not the primary perp account. The primary-perp '
            f"reading in PERP_DEX_VALUES is wrong and the flow it produces is "
            f"fabricated. Example:"
        )
        print(f"  {bad[0]!r}")
        return 2

    empty_seen = any(k[0] == "<empty>" or k[1] == "<empty>" for k in pairs)
    if not empty_seen:
        print('inconclusive: no record names dex "" at all, so this account '
              "cannot settle the question either way.")
        return 1

    print(
        'NOT FALSIFIED: every record naming dex "" moves USDC, which is what a '
        "USDC-margined perp venue would look like. This does NOT prove the "
        "reading — a USDC-only spot transfer looks identical — it establishes "
        "that the venue does not contradict it on this account. Re-run against "
        "another address before treating it as settled."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
