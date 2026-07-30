"""Answer OPEN-QUESTIONS C5 by observation instead of by opening a position.

    python -m risk_engine.market.probe_isolated_funding --address 0x...

C5 asks where funding on an *isolated* position is debited from. The model
takes it from that position's own margin, which is what makes isolated
liquidations independent of the cross pool (§1.1). If the venue instead
debits the cross pool, that independence claim is false and the simulator
needs a coupling term it does not have -- so this is the one unresolved
question that can invalidate a structural part of the model rather than
shift a number.

The `verify` harness reports C5 as UNCHECKABLE because no *single* read
distinguishes the two. A pair of reads does. `clearinghouseState` exposes
`leverage.rawUsd` per isolated position -- the collateral sitting in that
pocket -- alongside `crossMarginSummary.accountValue`. Funding is charged
hourly. So: snapshot both sides of a funding tick, ask `userFunding` what
was actually paid, and see which balance moved by that amount. No position
to open, no capital at risk, and it works on any address that happens to
hold an isolated position.

What this cannot do is invent an address. It needs one holding at least one
isolated position across the tick, and it says so rather than guessing when
it does not get one.

Confounds are checked rather than hoped away. A user who trades during the
window moves both balances for reasons that have nothing to do with funding,
so the probe aborts on any change in position size. A funding payment too
small to distinguish from the residual noise in the balances is reported as
inconclusive rather than resolved in whichever direction the arithmetic
happened to land.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from risk_engine.market.info import MAINNET_URL, TESTNET_URL, InfoClient

#: Funding is charged on the hour. Sample a little after so the venue has
#: settled it, rather than racing the tick.
SETTLE_GRACE_S = 120.0
#: Below this, a payment cannot be told from rounding in the reported
#: balances, and a verdict either way would be arithmetic on noise.
MIN_RESOLVABLE_USD = 0.01
#: Fraction of the funding payment the moving balance must account for.
ATTRIBUTION_TOLERANCE = 0.25


@dataclass
class Snapshot:
    at: str
    cross_account_value: float
    #: coin -> isolated pocket collateral (`leverage.rawUsd`)
    isolated_margin: dict[str, float]
    #: coin -> signed position size, for the did-they-trade check
    sizes: dict[str, float]


@dataclass
class ProbeResult:
    status: str
    detail: str
    before: Snapshot | None = None
    after: Snapshot | None = None
    #: coin -> USD funding paid (positive) or received (negative)
    payments: dict[str, float] = field(default_factory=dict)
    deltas: dict[str, float] = field(default_factory=dict)
    cross_delta: float = 0.0

    def render(self) -> str:
        return f"[{self.status}] C5\n  {self.detail}"


def _snapshot(client: InfoClient, address: str) -> Snapshot:
    state = client.clearinghouse_state(address)
    cross = state.get("crossMarginSummary") or {}
    isolated: dict[str, float] = {}
    sizes: dict[str, float] = {}
    for entry in state.get("assetPositions") or []:
        p = entry.get("position") or {}
        coin, size = p.get("coin"), float(p.get("szi") or 0.0)
        if not coin or size == 0.0:
            continue
        sizes[coin] = size
        lev = p.get("leverage") or {}
        if lev.get("type") == "isolated":
            raw = lev.get("rawUsd", p.get("marginUsed"))
            isolated[coin] = float(raw)
    return Snapshot(
        at=datetime.now(timezone.utc).isoformat(),
        cross_account_value=float(cross.get("accountValue") or 0.0),
        isolated_margin=isolated,
        sizes=sizes,
    )


def _seconds_to_next_tick(now: float | None = None) -> float:
    now = now if now is not None else time.time()
    return 3600.0 - (now % 3600.0)


def probe(
    address: str,
    testnet: bool = False,
    wait: bool = True,
    max_wait_s: float = 4_000.0,
) -> ProbeResult:
    client = InfoClient(url=TESTNET_URL if testnet else MAINNET_URL)

    before = _snapshot(client, address)
    if not before.isolated_margin:
        return ProbeResult(
            "UNCHECKABLE",
            "this address holds no isolated position, so it cannot answer C5. "
            "Find an address with at least one isolated position (or open a "
            "minimal one on testnet with --testnet) and re-run. Nothing about "
            "the model is established or refuted by this run.",
            before=before,
        )

    start_ms = int(time.time() * 1000)
    if wait:
        delay = _seconds_to_next_tick() + SETTLE_GRACE_S
        if delay > max_wait_s:
            return ProbeResult(
                "UNCHECKABLE",
                f"the next funding tick is {delay / 60:.0f} min away, over the "
                f"{max_wait_s / 60:.0f} min limit. Re-run closer to the hour or "
                "raise --max-wait-minutes.",
                before=before,
            )
        print(
            f"  isolated positions: {sorted(before.isolated_margin)}\n"
            f"  waiting {delay / 60:.1f} min for the funding tick to settle...",
            flush=True,
        )
        time.sleep(delay)

    after = _snapshot(client, address)
    end_ms = int(time.time() * 1000)

    # A trade during the window moves both balances for reasons unrelated to
    # funding. Nothing can be concluded, and pretending otherwise would be
    # attributing a position change to the venue's funding accounting.
    changed = [c for c in set(before.sizes) | set(after.sizes)
               if abs(before.sizes.get(c, 0.0) - after.sizes.get(c, 0.0)) > 1e-12]
    if changed:
        return ProbeResult(
            "UNCHECKABLE",
            f"the book changed during the window ({changed}); a trade moves both "
            "balances independently of funding, so this window proves nothing. "
            "Re-run on an idle account.",
            before=before, after=after,
        )

    try:
        raw_payments = client.user_funding(address, start_ms - 3_600_000, end_ms)
    except Exception as exc:
        return ProbeResult(
            "UNCHECKABLE",
            f"could not read userFunding ({type(exc).__name__}: {exc}), so there "
            "is no ground truth to attribute the balance moves to. The deltas "
            "are recorded in the report for manual inspection.",
            before=before, after=after,
            deltas={c: after.isolated_margin.get(c, 0.0) - before.isolated_margin[c]
                    for c in before.isolated_margin},
            cross_delta=after.cross_account_value - before.cross_account_value,
        )

    payments: dict[str, float] = {}
    for row in raw_payments or []:
        delta = row.get("delta") or row
        coin = delta.get("coin")
        usdc = delta.get("usdc")
        if coin is None or usdc is None:
            continue
        if not (start_ms - 3_600_000 <= int(row.get("time", 0)) <= end_ms):
            continue
        # The venue reports what the account paid as a signed cash flow; the
        # sign convention is recorded rather than assumed, since the verdict
        # below only uses magnitudes.
        payments[coin] = payments.get(coin, 0.0) + float(usdc)

    deltas = {
        c: after.isolated_margin.get(c, 0.0) - before.isolated_margin[c]
        for c in before.isolated_margin
    }
    cross_delta = after.cross_account_value - before.cross_account_value

    resolvable = {
        c: p for c, p in payments.items()
        if c in before.isolated_margin and abs(p) >= MIN_RESOLVABLE_USD
    }
    if not resolvable:
        return ProbeResult(
            "INCONCLUSIVE",
            "no isolated position took a funding payment large enough to "
            f"attribute (threshold ${MIN_RESOLVABLE_USD}). Funding was either "
            "zero this hour or too small to separate from rounding in the "
            "reported balances. Re-run on a larger isolated position.",
            before=before, after=after, payments=payments,
            deltas=deltas, cross_delta=cross_delta,
        )

    verdicts = []
    for coin, payment in resolvable.items():
        moved_iso = abs(deltas.get(coin, 0.0))
        share_iso = moved_iso / abs(payment)
        share_cross = abs(cross_delta) / abs(payment)
        if abs(share_iso - 1.0) <= ATTRIBUTION_TOLERANCE:
            verdicts.append((coin, "isolated", share_iso, share_cross))
        elif abs(share_cross - 1.0) <= ATTRIBUTION_TOLERANCE and share_iso < 0.25:
            verdicts.append((coin, "cross", share_iso, share_cross))
        else:
            verdicts.append((coin, "ambiguous", share_iso, share_cross))

    lines = [
        f"{c}: funding ${p:+.4f}, isolated margin moved "
        f"{s_i:.0%} of it, cross value moved {s_c:.0%} -> {v}"
        for (c, v, s_i, s_c), p in zip(verdicts, resolvable.values(), strict=True)
    ]
    kinds = {v for _, v, _, _ in verdicts}

    if kinds == {"isolated"}:
        status, detail = "PASS", (
            "funding on isolated positions is debited from the position's own "
            "margin, so §1.1's independence holds and the simulator needs no "
            "coupling term.\n  " + "\n  ".join(lines)
        )
    elif "cross" in kinds:
        status, detail = "FAIL", (
            "funding on an isolated position moved the CROSS balance. §1.1's "
            "independence claim is false: an isolated position can drain the "
            "cross pool through funding, and the simulator has no term for "
            "that. This invalidates the isolated/cross separation, not just a "
            "number.\n  " + "\n  ".join(lines)
        )
    else:
        status, detail = "INCONCLUSIVE", (
            "the balance moves do not cleanly attribute to the funding "
            "payments -- something else moved the account in the same window.\n  "
            + "\n  ".join(lines)
        )
    return ProbeResult(status, detail, before=before, after=after,
                       payments=payments, deltas=deltas, cross_delta=cross_delta)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="risk_engine.market.probe_isolated_funding",
        description="resolve OPEN-QUESTIONS C5 by watching a funding tick",
    )
    parser.add_argument("--address", required=True,
                        help="an account holding at least one isolated position")
    parser.add_argument("--testnet", action="store_true")
    parser.add_argument("--no-wait", dest="wait", action="store_false",
                        help="snapshot twice immediately; for wiring checks only")
    parser.add_argument("--max-wait-minutes", type=float, default=67.0)
    parser.add_argument("--report", help="write the full observation as JSON")
    args = parser.parse_args(argv)

    result = probe(args.address, testnet=args.testnet, wait=args.wait,
                   max_wait_s=args.max_wait_minutes * 60.0)
    print(result.render())

    if args.report:
        with open(args.report, "w") as fh:
            json.dump(asdict(result), fh, indent=2, default=str)
        print(f"wrote {args.report}")

    return {"PASS": 0, "FAIL": 2}.get(result.status, 1)


if __name__ == "__main__":
    sys.exit(main())
