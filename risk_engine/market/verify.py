"""Check the documented-but-unverified assumptions against the live API.

    python -m risk_engine.market.verify --report verify.json

Everything in `market/` is written against documented response shapes and
recorded fixtures. `api.hyperliquid.xyz` was blocked at the proxy in the
environment this was built in (403), so not one parser has seen a live
response (OPEN-QUESTIONS E5), and four more questions -- C1, C2, C4, C5 --
are blocked on the same access.

This exists so that closing them is running a command rather than
reconstructing what needed checking. It reports per assumption, exits
non-zero while any blocking one is unconfirmed, and distinguishes four
outcomes, because collapsing them is how "we ran the checker" becomes
"the model is verified":

  PASS          the assumption held against live data
  FAIL          live data contradicts it -- the model is wrong today
  INCONCLUSIVE  the check ran and cannot decide (usually: absence of a
                counter-example is not confirmation)
  UNCHECKABLE   this harness cannot decide it at all; the entry says what can

FAIL is reserved for the venue. It means a live response contradicted a
documented shape, it exits 2, and it says the model is wrong today. A fault
in this harness's own input -- an `--address` that is not an address -- is
UNCHECKABLE instead, whatever layer notices it: a checker that reports an
operator's typo as the venue contradicting the model is a checker whose
FAILs stop being read.

The distinction between PASS and INCONCLUSIVE carries most of the weight.
A clamp no observation exceeded is not a verified clamp -- it is a clamp
nothing has contradicted yet, over whatever window the venue serves. Saying
"PASS" there would launder the absence of evidence into evidence, which is
the §10-forbidden direction on the one bound that truncates tail risk.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from risk_engine.domain.types import normalise_address
from risk_engine.market.info import InfoClient
from risk_engine.market.parse import (
    parse_candles_to_log_returns,
    parse_clearinghouse_state,
    parse_funding_history,
    parse_meta,
)
from risk_engine.model.funding import FundingBounds

PASS = "PASS"  # noqa: S105 — a check outcome, not a credential
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"
UNCHECKABLE = "UNCHECKABLE"

HOUR_MS = 3_600_000


@dataclass
class Check:
    id: str
    question: str
    status: str
    detail: str
    #: Whether an unconfirmed result should hold up the live path.
    blocking: bool = True
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def satisfied(self) -> bool:
        return self.status == PASS or not self.blocking

    def render(self) -> str:
        return f"[{self.status:12}] {self.id:6} {self.question}\n               {self.detail}"


# -- E5: the parsers have never seen a live response ---------------------


def check_meta(client: InfoClient) -> tuple[Check, dict | None]:
    try:
        raw = client.meta()
        specs = parse_meta(raw)
    except Exception as exc:
        return Check(
            "E5.1", "does `meta` parse into margin tiers?", FAIL,
            f"{type(exc).__name__}: {exc}",
        ), None
    if not specs:
        return Check("E5.1", "does `meta` parse into margin tiers?", FAIL,
                     "parsed, but the universe is empty"), None
    tiered = [s for s in specs.values() if len(s.tiers) > 1]
    return Check(
        "E5.1", "does `meta` parse into margin tiers?", PASS,
        f"{len(specs)} assets parsed, {len(tiered)} with more than one margin tier",
        evidence={"n_assets": len(specs), "n_tiered": len(tiered),
                  "sample": sorted(specs)[:8]},
    ), specs


def check_candles(client: InfoClient, coin: str) -> tuple[Check, np.ndarray | None]:
    now_ms = int(time.time() * 1000)
    try:
        candles = client.candle_snapshot(coin, "1h", now_ms - 30 * 24 * HOUR_MS, now_ms)
        times, returns = parse_candles_to_log_returns(candles)
    except Exception as exc:
        return Check("E5.2", f"does `candleSnapshot` parse for {coin}?", FAIL,
                     f"{type(exc).__name__}: {exc}"), None
    if not np.isfinite(returns).all():
        return Check("E5.2", f"does `candleSnapshot` parse for {coin}?", FAIL,
                     "parsed, but produced non-finite log returns"), None
    gaps = int((np.diff(times) > HOUR_MS * 1.5).sum())
    return Check(
        "E5.2", f"does `candleSnapshot` parse for {coin}?", PASS,
        f"{returns.size} hourly returns, {gaps} gaps, "
        f"hourly vol {float(np.std(returns)):.5f}",
        evidence={"n": int(returns.size), "gaps": gaps,
                  "hourly_vol": float(np.std(returns))},
    ), returns


def check_clearinghouse(client: InfoClient, address: str | None) -> Check:
    if not address:
        return Check(
            "E5.3", "does `clearinghouseState` parse into a Book?", UNCHECKABLE,
            "no --address given. This is the parser the whole product reads a "
            "user's book through; pass any address holding perps.",
        )
    # A local format error is not evidence about the venue, and this function
    # is where that has to be enforced rather than only at the argparse layer.
    # `InfoClient` normalises on the way out (§5.1), so a malformed address
    # raises ValueError before a byte leaves the process -- and the `except`
    # below would file that under FAIL, which in this file's published
    # vocabulary means "live data contradicts it, the model is wrong today"
    # (see the module docstring) and exits 2 with "the model is wrong today;
    # fix it before the live path runs". Nothing was contradicted, because
    # nothing was asked. `main` does refuse a bad `--address`, but `run_all`
    # and this function are both importable and this is the one the tests
    # drive, so the vocabulary has to hold here on its own.
    #
    # UNCHECKABLE, still blocking, and deliberately not softer than that: a
    # typo leaves E5.3 exactly as unverified as no address at all did, and the
    # parser the whole product reads a user's book through must not come out
    # of a run looking checked.
    try:
        address = normalise_address(address)
    except ValueError as exc:
        return Check(
            "E5.3", "does `clearinghouseState` parse into a Book?", UNCHECKABLE,
            f"--address is not an account address ({exc}), so no request was "
            "made and nothing was learned about the venue. This is a local "
            "format error, not a contradiction: fix the address and re-run.",
            evidence={"rejected_address": address},
        )
    try:
        raw = client.clearinghouse_state(address)
        book = parse_clearinghouse_state(raw, address)
    except Exception as exc:
        return Check("E5.3", "does `clearinghouseState` parse into a Book?", FAIL,
                     f"{type(exc).__name__}: {exc}")
    return Check(
        "E5.3", "does `clearinghouseState` parse into a Book?", PASS,
        f"{len(book.positions)} positions, cross collateral "
        f"{book.cross_collateral:,.2f}",
        evidence={"n_positions": len(book.positions),
                  "coins": [p.coin for p in book.positions],
                  "n_isolated": len(book.isolated_positions)},
    )


# -- C1: the funding clamp ------------------------------------------------


def check_funding_clamp(client: InfoClient, coins: list[str], days: int) -> Check:
    bounds = FundingBounds.documented_default()
    now_ms = int(time.time() * 1000)
    worst = 0.0
    worst_coin = ""
    total = 0
    breaches: list[dict] = []
    for coin in coins:
        try:
            history = client.funding_history(coin, now_ms - days * 24 * HOUR_MS, now_ms)
            _, rates = parse_funding_history(history)
        except Exception as exc:
            return Check("C1", "is the documented funding clamp real?", FAIL,
                         f"could not read funding history for {coin}: "
                         f"{type(exc).__name__}: {exc}")
        total += rates.size
        if rates.size:
            local = float(np.abs(rates).max())
            if local > worst:
                worst, worst_coin = local, coin
            if local > bounds.cap_per_hour:
                breaches.append({"coin": coin, "observed": local})

    evidence = {
        "configured_cap_per_hour": bounds.cap_per_hour,
        "source": bounds.source,
        "observed_max_abs_rate": worst,
        "worst_coin": worst_coin,
        "n_observations": total,
        "window_days": days,
        "headroom_ratio": (worst / bounds.cap_per_hour) if bounds.cap_per_hour else None,
    }
    if breaches:
        return Check(
            "C1", "is the documented funding clamp real?", FAIL,
            f"observed |rate| {worst:.6g}/h on {worst_coin} exceeds the configured "
            f"cap {bounds.cap_per_hour:.6g}/h. The bound is wrong, and the engine "
            "refuses to fit an AR(1) against it — fix the bound, do not clamp "
            "reality away (§1.5).",
            evidence=evidence | {"breaches": breaches},
        )
    return Check(
        "C1", "is the documented funding clamp real?", INCONCLUSIVE,
        f"no breach in {total} observations over {days}d; worst was {worst:.6g}/h on "
        f"{worst_coin}, {worst / bounds.cap_per_hour:.1%} of the configured cap. "
        "This falsifies nothing and confirms nothing: a clamp is a protocol "
        "constant and no sample of realised rates can establish it. Confirm the "
        "value from protocol documentation or source and record it as the "
        "`source` field.",
        evidence=evidence,
    )


# -- C2: mark versus trade basis -----------------------------------------


def check_basis(client: InfoClient, coins: list[str], samples: int,
                interval_s: float, hourly_vol: float | None) -> Check:
    """§1.4: median |mark - mid| / mid against a typical hourly move.

    One snapshot cannot answer this -- the basis is a time series and §1.4
    asks for its median -- so this samples repeatedly. It is still a much
    shorter window than §1.4 intends, which is why a clean result here is
    INCONCLUSIVE rather than PASS.
    """
    observed: dict[str, list[float]] = {c: [] for c in coins}
    for i in range(samples):
        try:
            payload = client.post({"type": "metaAndAssetCtxs"})
        except Exception as exc:
            return Check("C2", "is mark ≈ mid, per §1.4's threshold?", FAIL,
                         f"metaAndAssetCtxs failed: {type(exc).__name__}: {exc}")
        try:
            universe = [a["name"] for a in payload[0]["universe"]]
            ctxs = payload[1]
        except (KeyError, IndexError, TypeError) as exc:
            return Check(
                "C2", "is mark ≈ mid, per §1.4's threshold?", FAIL,
                f"metaAndAssetCtxs is not the documented [meta, ctxs] pair: "
                f"{type(exc).__name__}: {exc}",
            )
        if len(universe) != len(ctxs):
            # zip would silently truncate to the shorter side, and a checker
            # that drops assets without saying so reports "checked" for assets
            # it never saw. A length mismatch IS the schema drift this
            # harness exists to catch.
            return Check(
                "C2", "is mark ≈ mid, per §1.4's threshold?", FAIL,
                f"metaAndAssetCtxs is misaligned: {len(universe)} universe entries "
                f"against {len(ctxs)} contexts. The documented contract is one "
                "context per universe entry, in order; a mismatch means the pairing "
                "cannot be trusted for any asset.",
                evidence={"n_universe": len(universe), "n_ctxs": len(ctxs)},
            )
        for name, ctx in zip(universe, ctxs, strict=True):
            if name not in observed:
                continue
            mark, mid = ctx.get("markPx"), ctx.get("midPx")
            if mark is None or mid is None:
                continue
            mid_f = float(mid)
            if mid_f > 0:
                observed[name].append(abs(float(mark) - mid_f) / mid_f)
        if i + 1 < samples:
            time.sleep(interval_s)

    medians = {c: statistics.median(v) for c, v in observed.items() if v}
    if not medians:
        return Check("C2", "is mark ≈ mid, per §1.4's threshold?", FAIL,
                     "metaAndAssetCtxs carried no markPx/midPx pair for any "
                     "requested coin")
    worst_coin = max(medians, key=lambda c: medians[c])
    worst = medians[worst_coin]
    evidence = {"median_abs_basis": medians, "n_samples": samples,
                "interval_s": interval_s, "hourly_vol": hourly_vol}

    if hourly_vol is None:
        return Check("C2", "is mark ≈ mid, per §1.4's threshold?", INCONCLUSIVE,
                     f"median |basis| up to {worst:.6g} ({worst_coin}), but no "
                     "hourly volatility was measured to compare it against",
                     evidence=evidence)
    threshold = 0.25 * hourly_vol
    evidence["threshold"] = threshold
    if worst > threshold:
        return Check(
            "C2", "is mark ≈ mid, per §1.4's threshold?", FAIL,
            f"median |basis| {worst:.6g} on {worst_coin} exceeds 25% of a typical "
            f"hourly move ({threshold:.6g}). §1.4's own condition for treating "
            "mark as trade price does NOT hold. Note what that means concretely: "
            "the engine is fed the last hourly candle close (a trade price) and "
            "checks §1.1's margin condition against it as though it were the mark "
            "price, with no basis term anywhere — so this is not a guard to "
            "un-set, it is an approximation the simulator makes silently and "
            "which this result has just falsified. A basis term has to be built "
            "before the liquidation numbers can be trusted (OPEN-QUESTIONS C2).",
            evidence=evidence,
        )
    return Check(
        "C2", "is mark ≈ mid, per §1.4's threshold?", INCONCLUSIVE,
        f"median |basis| {worst:.6g} on {worst_coin}, under the {threshold:.6g} "
        f"threshold — but measured over {samples} samples spanning "
        f"{samples * interval_s / 60:.0f} minutes, which is not the window §1.4 "
        "intends. Promote to PASS from a shadow-mode series, not from this.",
        evidence=evidence,
    )


# -- C4, C5: not decidable from the Info API -----------------------------


def check_external_flow(client: InfoClient, address: str | None) -> Check:
    """B2's correction: can deposits and withdrawals actually be read?

    The shadow harness cannot resolve a single observation without this. An
    equity change is only model error once the flows the model does not
    predict have been subtracted, so `resolve_due` calls `external_flow` on
    every row — and a provider that cannot answer fails every row, is
    classified transient, and retries forever. That is fourteen days of
    snapshots against a gate that never advances, which is why this is
    checked before the clock starts rather than discovered during it.
    """
    from risk_engine.market.parse import (
        DEX_ROUTED_TYPES,
        EXTERNAL_FLOW_SIGNS,
        NON_FLOW_DELTA_TYPES,
        net_external_flow,
    )

    if not address:
        return Check(
            "B2", "can external flows be read and classified?", UNCHECKABLE,
            "no --address given. Without this the shadow harness resolves nothing, "
            "so it is worth passing an address that has deposited or withdrawn.",
        )
    now_ms = int(time.time() * 1000)
    window_days = 90
    try:
        raw = client.non_funding_ledger_updates(
            address, now_ms - window_days * 24 * HOUR_MS, now_ms
        )
    except Exception as exc:
        return Check(
            "B2", "can external flows be read and classified?", FAIL,
            f"userNonFundingLedgerUpdates failed: {type(exc).__name__}: {exc}. The "
            "shadow harness cannot resolve any observation without it.",
        )

    rows = list(raw or [])
    kinds: dict[str, int] = {}
    examples: dict[str, dict] = {}
    for row in rows:
        kind = ((row.get("delta") or {}).get("type")) or "<no delta.type>"
        kinds[kind] = kinds.get(kind, 0) + 1
        # One full record per type, verbatim. A count says a type exists; it
        # says nothing about its fields, and guessing a sign from a name
        # already went wrong once here -- `accountClassTransfer` needed a
        # live `toPerp` flag this repo could not have invented. Public
        # on-chain data, so nothing here needs redacting.
        examples.setdefault(kind, row)
    known = set(EXTERNAL_FLOW_SIGNS) | set(NON_FLOW_DELTA_TYPES) | set(DEX_ROUTED_TYPES)
    unknown = sorted(k for k in kinds if k not in known)
    evidence = {"n_records": len(rows), "window_days": window_days,
                "types_seen": kinds, "unknown_types": unknown,
                "examples": {k: examples[k] for k in unknown}}

    if unknown:
        return Check(
            "B2", "can external flows be read and classified?", FAIL,
            f"the venue returned delta types this build cannot classify: {unknown}. "
            "Each one is either an external flow or it is not, and guessing either "
            "way corrupts the calibration record — an unclassified transfer scored "
            "as model error, or a real deposit hidden. A full example record for "
            "each is in this check's `evidence.examples` (--report to see it) — "
            "read the fields before adding it to EXTERNAL_FLOW_SIGNS or "
            "NON_FLOW_DELTA_TYPES in market/parse.py (OPEN-QUESTIONS B2).",
            evidence=evidence,
        )
    try:
        from datetime import datetime as _dt
        from datetime import timezone as _tz

        total = net_external_flow(
            rows,
            _dt.fromtimestamp((now_ms - window_days * 24 * HOUR_MS) / 1000, tz=_tz.utc),
            _dt.fromtimestamp(now_ms / 1000, tz=_tz.utc),
            address,
        )
    except ValueError as exc:
        return Check(
            "B2", "can external flows be read and classified?", FAIL,
            f"a ledger record could not be read: {exc}", evidence=evidence,
        )

    if not rows:
        return Check(
            "B2", "can external flows be read and classified?", INCONCLUSIVE,
            f"the endpoint answered and this account has no ledger activity in "
            f"{window_days}d, so the response shape was never exercised. An empty "
            "list is what a quiet account and a wrong request type look like alike. "
            "Re-run against an address that has deposited or withdrawn.",
            evidence=evidence,
        )
    return Check(
        "B2", "can external flows be read and classified?", PASS,
        f"{len(rows)} ledger records over {window_days}d, every delta type "
        f"recognised ({', '.join(sorted(kinds))}); net flow {total:+,.2f} USD.",
        evidence=evidence | {"net_flow_usd": total},
    )


def check_webdata3() -> Check:
    return Check(
        "C4", "does the `webData3` subscription exist?", UNCHECKABLE,
        "a WebSocket question, and this harness speaks only the Info POST API. "
        "Subscribe to {\"method\":\"subscribe\",\"subscription\":{\"type\":\"webData3\"}} "
        "on wss://api.hyperliquid.xyz/ws and see whether it errors; `webData2` is "
        "the documented one and the shard planner is agnostic either way.",
        blocking=False,
    )


def check_isolated_funding() -> Check:
    return Check(
        "C5", "is isolated-position funding debited from isolated margin?", UNCHECKABLE,
        "needs a funded account observed across an hourly funding tick: open one "
        "isolated position, record isolated margin and cross balance before and "
        "after the tick, and see which moved. No read-only endpoint distinguishes "
        "them. This matters beyond bookkeeping — if funding on an isolated "
        "position is debited from the cross pool, §1.1's independence claim is "
        "false and the simulator needs a coupling term it does not have.",
    )


# -- driver ---------------------------------------------------------------


def run_all(address: str | None, coins: list[str], days: int, samples: int,
            interval_s: float, testnet: bool) -> list[Check]:
    from risk_engine.market.info import MAINNET_URL, TESTNET_URL

    client = InfoClient(url=TESTNET_URL if testnet else MAINNET_URL)
    checks: list[Check] = []

    meta_check, specs = check_meta(client)
    checks.append(meta_check)
    if specs:
        known = [c for c in coins if c in specs]
        missing = [c for c in coins if c not in specs]
        if missing:
            checks.append(Check(
                "E5.0", "are the requested coins in the live universe?", FAIL,
                f"not listed by `meta`: {missing}", evidence={"missing": missing},
            ))
        coins = known or coins

    candle_check, returns = check_candles(client, coins[0])
    checks.append(candle_check)
    hourly_vol = float(np.std(returns)) if returns is not None else None

    checks.append(check_clearinghouse(client, address))
    checks.append(check_external_flow(client, address))
    checks.append(check_funding_clamp(client, coins, days))
    checks.append(check_basis(client, coins, samples, interval_s, hourly_vol))
    checks.append(check_webdata3())
    checks.append(check_isolated_funding())
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="risk_engine.market.verify",
        description="check the unverified live-API assumptions (E5, C1, C2, C4, C5)",
    )
    parser.add_argument("--address", help="an account holding perps, for E5.3")
    parser.add_argument("--coins", default="BTC,ETH,SOL",
                        help="comma-separated; the first is used for the vol reference")
    parser.add_argument("--days", type=int, default=30,
                        help="funding-history window for C1")
    parser.add_argument("--samples", type=int, default=12,
                        help="mark/mid snapshots for C2")
    parser.add_argument("--interval-s", type=float, default=5.0,
                        help="seconds between C2 snapshots")
    parser.add_argument("--testnet", action="store_true")
    parser.add_argument("--report", help="write the full result as JSON")
    args = parser.parse_args(argv)

    coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    # The address gets the same treatment as the coins on the line above. It
    # is normalised again inside `InfoClient`, so this is not about what
    # reaches the venue -- it is about *where the operator hears about it*: an
    # unnormalisable address surfaces from `run_all` as E5.3 FAIL, which in
    # this tool's vocabulary means the live API contradicted a documented
    # shape. A typo must not be reportable as evidence about the venue.
    address: str | None = None
    if args.address:
        try:
            address = normalise_address(args.address)
        except ValueError as exc:
            parser.error(str(exc))
    checks = run_all(address, coins, args.days, args.samples,
                     args.interval_s, args.testnet)

    print(f"live-API verification against {'testnet' if args.testnet else 'mainnet'}\n")
    for check in checks:
        print(check.render())
        print()

    if args.report:
        with open(args.report, "w") as fh:
            json.dump([asdict(c) for c in checks], fh, indent=2, default=str)
        print(f"wrote {args.report}")

    failed = [c for c in checks if c.status == FAIL]
    unconfirmed = [c for c in checks if not c.satisfied and c.status != FAIL]
    if failed:
        print(f"\n{len(failed)} assumption(s) CONTRADICTED by live data: "
              f"{', '.join(c.id for c in failed)}. The model is wrong today; "
              "fix it before the live path runs.")
        return 2
    if unconfirmed:
        print(f"\n{len(unconfirmed)} assumption(s) still unconfirmed: "
              f"{', '.join(c.id for c in unconfirmed)}. Nothing here contradicts "
              "the model, and nothing here establishes it either.")
        return 1
    print("\nEvery blocking assumption confirmed against live data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
