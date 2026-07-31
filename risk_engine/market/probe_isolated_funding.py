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
it does not get one. An address that is not an address gets the same
treatment -- UNCHECKABLE, before any connection is opened. FAIL from this
probe means §1.1's independence claim is false and the simulator is missing
a coupling term; a typo must not be able to reach a verdict that strong.

Confounds are checked rather than hoped away. A user who trades during the
window moves both balances for reasons that have nothing to do with funding,
so the probe aborts on any change in position size. A funding payment too
small to distinguish from the residual noise in the balances is reported as
inconclusive rather than resolved in whichever direction the arithmetic
happened to land.

Attribution reads the ISOLATED side in both directions, never the cross
side. The pocket's `rawUsd` is ledger collateral — it moves on funding and
on explicit margin transfers, not on mark prices — while the cross account
value carries the unrealised PnL of every cross position, whose drift over
the probe's multi-minute window dwarfs a funding payment. An earlier draft
required `cross_delta ~= payment` for the cross-debit verdict; the audit
showed that condition is unreachable under normal uPnL noise, which made the
structurally dangerous world the one the probe could not detect. The pocket
not paying IS the violation: the account demonstrably paid (userFunding) and
the only other bucket is cross, however invisibly the payment lands in its
noise. The stated assumption — checked against the venue's docs, not proven
— is that isolated funding settles into pocket collateral rather than into
the pocket's own unrealised PnL; the FAIL text names that assumption.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from risk_engine.domain.types import normalise_address
from risk_engine.market.info import MAINNET_URL, TESTNET_URL, InfoClient

#: Funding is charged on the hour. Sample a little after so the venue has
#: settled it, rather than racing the tick.
SETTLE_GRACE_S = 120.0
#: Below this, a payment cannot be told from rounding in the reported
#: balances, and a verdict either way would be arithmetic on noise.
MIN_RESOLVABLE_USD = 0.01
#: The pocket paid its own funding: |pocket move| within this of the payment.
ATTRIBUTION_TOLERANCE = 0.25
#: The pocket demonstrably did NOT pay: it absorbed under this fraction of a
#: payment the account provably made. The gap between the two bands reads as
#: ambiguous rather than being rounded toward either verdict.
ISOLATED_SILENT_BELOW = 0.25


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
            if raw is None:
                # Refuse at the BEFORE snapshot, loudly, rather than crashing
                # on float(None) at the after snapshot -- an hour of waiting
                # later (audit P-3).
                raise ValueError(
                    f"{coin}: isolated position exposes neither leverage.rawUsd "
                    "nor marginUsed; the pocket's collateral cannot be observed"
                )
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
    # Before the client exists, so no connection is opened and no §5.3 weight
    # is charged for a request that could not be made. `main` normalises too,
    # but this function is importable and is what the tests drive, and the
    # vocabulary has to hold wherever it is entered from.
    #
    # UNCHECKABLE, not FAIL. FAIL out of this probe is the loudest verdict in
    # the repository -- it says §1.1's independence claim is false, that an
    # isolated position can drain the cross pool through funding, and that the
    # simulator is missing a term; `main` maps it to exit code 2. A typo'd
    # address must not be able to reach that verdict. Before this guard the
    # format error surfaced as an unhandled ValueError out of the first
    # `_snapshot`, which is not FAIL but is not a verdict either: the caller
    # got a traceback where the type says it gets a `ProbeResult`, and a
    # `--report` file that was never written.
    try:
        address = normalise_address(address)
    except ValueError as exc:
        return ProbeResult(
            "UNCHECKABLE",
            f"'{address}' is not an account address ({exc}), so nothing was "
            "read and C5 is exactly as open as it was. This says nothing about "
            "where the venue debits isolated funding from.",
        )

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
        # STRICTLY the observed window. The tick lands inside it by
        # construction (start < top-of-hour < end). Widening the request by
        # an hour -- an earlier draft did -- pulls in the PREVIOUS tick's
        # payment, which the before-snapshot already contains: the payment
        # sum doubles while the balance delta does not, and every realistic
        # run reads as ambiguous (audit P-1).
        raw_payments = client.user_funding(address, start_ms, end_ms)
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
        if not (start_ms <= int(row.get("time", 0)) <= end_ms):
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

    # Attribution reads the pocket, in BOTH directions. The pocket's rawUsd
    # is ledger collateral -- funding and explicit transfers move it, mark
    # prices do not -- while cross account value drifts with every cross
    # position's unrealised PnL, drowning a funding-sized move within
    # minutes. An earlier draft demanded cross_delta ~= payment for the
    # cross verdict; under normal uPnL noise that condition is unreachable,
    # which made the structurally dangerous world the undetectable one
    # (audit P-2). The pocket refusing to pay IS the finding: the account
    # provably paid (userFunding) and there is no third bucket.
    verdicts = []
    for coin, payment in resolvable.items():
        share_iso = abs(deltas.get(coin, 0.0)) / abs(payment)
        if abs(share_iso - 1.0) <= ATTRIBUTION_TOLERANCE:
            verdicts.append((coin, "isolated", share_iso))
        elif share_iso < ISOLATED_SILENT_BELOW:
            verdicts.append((coin, "cross", share_iso))
        else:
            verdicts.append((coin, "ambiguous", share_iso))

    lines = [
        f"{c}: funding ${p:+.4f}, the pocket absorbed {s_i:.0%} of it -> {v}"
        for (c, v, s_i), p in zip(verdicts, resolvable.values(), strict=True)
    ]
    lines.append(
        f"cross account value moved ${cross_delta:+.2f} over the window "
        "(uPnL drift included; corroboration only, never the verdict)"
    )
    kinds = {v for _, v, _ in verdicts}

    if kinds == {"isolated"}:
        status, detail = "PASS", (
            "funding on isolated positions is debited from the position's own "
            "margin, so §1.1's independence holds and the simulator needs no "
            "coupling term.\n  " + "\n  ".join(lines)
        )
    elif "cross" in kinds:
        status, detail = "FAIL", (
            "an isolated position's funding was NOT taken from its own margin: "
            "the account paid (userFunding) while the pocket's collateral did "
            "not move. The only other bucket is the cross pool, so §1.1's "
            "independence claim is false -- an isolated position can drain "
            "cross through funding, and the simulator has no term for that. "
            "Stated assumption: the venue settles isolated funding into pocket "
            "collateral (rawUsd); if it settles into the pocket's own uPnL "
            "instead, confirm on testnet before treating this as final.\n  "
            + "\n  ".join(lines)
        )
    else:
        status, detail = "INCONCLUSIVE", (
            "the pocket absorbed part of the payment but not within tolerance "
            "of it -- an explicit margin transfer or venue rounding landed in "
            "the same window.\n  " + "\n  ".join(lines)
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

    # Refused as a usage error here as well as inside `probe`, and the two are
    # not redundant. `probe` returns UNCHECKABLE with the format complaint,
    # which is the right *verdict* for a library caller and lands in
    # `--report`; argparse gives the operator the standard usage error and a
    # distinct exit code, so a typo at the terminal does not read as "the
    # probe ran and could not tell", which is what an UNCHECKABLE line looks
    # like next to the several genuine ways this probe cannot tell.
    try:
        address = normalise_address(args.address)
    except ValueError as exc:
        parser.error(str(exc))

    result = probe(address, testnet=args.testnet, wait=args.wait,
                   max_wait_s=args.max_wait_minutes * 60.0)
    print(result.render())

    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(asdict(result), fh, indent=2, default=str)
        print(f"wrote {args.report}")

    return {"PASS": 0, "FAIL": 2}.get(result.status, 1)


if __name__ == "__main__":
    sys.exit(main())
