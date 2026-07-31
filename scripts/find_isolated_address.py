"""Find an address in the shadow list that holds an isolated position (C5).

    python scripts/find_isolated_address.py deploy/addresses.json

C5 needs one account holding at least one isolated position across an hourly
funding tick. The obvious reading of that is "open an isolated position" —
capital at risk, on mainnet, to answer a bookkeeping question.

It is not necessary. `probe_isolated_funding` is entirely read-only: it
snapshots `clearinghouseState` on both sides of a tick and asks `userFunding`
what was actually paid. Any address holding an isolated position will do, and
the collector already harvested hundreds of them. This scans that list and
prints the ones that qualify.

Stops at the first few matches rather than scanning everything: each address
costs 20 weight against §5.3's budget, the answer needs exactly one, and
scanning 500 to report 40 is 10 000 weight spent to no purpose.
"""

from __future__ import annotations

import pathlib
import sys

# Running this by path puts `scripts/` on sys.path, not the repo root, so the
# invocation in the docstring above would fail on the first import. Prepending
# the root here rather than telling the operator to set PYTHONPATH: a command
# that only works with an undocumented environment variable is a command that
# gets reported as broken.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from risk_engine.market.info import InfoClient
from risk_engine.market.parse import parse_clearinghouse_state
from risk_engine.shadow.providers import FileAddressSource

WANTED = 3
SCAN_LIMIT = 120


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__.strip().splitlines()[2].strip())
        return 2

    from datetime import datetime, timezone

    # The list is loaded through the same source the cron uses, so its
    # refusals are the cron's refusals -- including the one an operator hits
    # first, pointing this at the unfilled `deploy/addresses.json` template.
    # Reported as a message rather than a traceback: the refusal is the
    # answer here, not a crash.
    try:
        addresses = FileAddressSource(args[0]).addresses()
    except (OSError, ValueError) as exc:
        print(f"cannot use {args[0]}: {exc}")
        return 2
    if not addresses:
        print(f"{args[0]} lists no addresses.")
        return 2

    client = InfoClient()
    found = 0
    print(f"scanning up to {SCAN_LIMIT} of {len(addresses)} addresses "
          f"for isolated positions (20 weight each)...")

    for i, addr in enumerate(addresses[:SCAN_LIMIT], 1):
        try:
            book = parse_clearinghouse_state(
                client.clearinghouse_state(addr), addr, datetime.now(timezone.utc))
        except Exception as exc:
            print(f"  [{i:3}] {addr}  unreadable: {type(exc).__name__}")
            continue
        iso = book.isolated_positions
        if not iso:
            continue
        found += 1
        coins = ", ".join(f"{p.coin} {p.size:+.4g}" for p in iso)
        print(f"  [{i:3}] {addr}  {len(iso)} isolated: {coins}")
        if found >= WANTED:
            break

    if not found:
        print(
            f"\nNo isolated positions in the first {SCAN_LIMIT}. Isolated margin is "
            f"the less common mode, so this is a real possibility rather than a "
            f"failure — raise SCAN_LIMIT, or use any address you know holds one."
        )
        return 1

    print(
        f"\nFound {found}. Run the probe against one of them — it is read-only and "
        f"needs no permission from the account holder:\n"
        f"    python -m risk_engine.market.probe_isolated_funding --address <addr> "
        f"--report c5-mainnet.json\n"
        f"It waits for the next hourly funding tick, so allow up to an hour. The "
        f"probe aborts if the position size changes during the window: someone "
        f"else's account can trade mid-observation, and a size change moves both "
        f"balances for reasons that have nothing to do with funding."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
