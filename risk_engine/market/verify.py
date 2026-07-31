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
from risk_engine.market.findings import STALE_AFTER_DAYS, load_findings
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
    def passed(self) -> bool:
        """PASS whether reached live or carried over from a recorded finding.

        The suffix is deliberately part of the status string rather than a
        separate flag: it renders everywhere the status renders, so there is
        no display path on which a recorded result can be mistaken for a live
        one. That makes an exact `== PASS` comparison wrong, and this property
        is what every caller must use instead.
        """
        return self.status.startswith(PASS)

    @property
    def failed(self) -> bool:
        """FAIL live, or carried over from a recorded out-of-band check.

        A recorded FAIL still exits 2. An out-of-band check that found the
        venue contradicting the model is exactly as disqualifying as an
        in-run one; the only difference is where it was measured.
        """
        return self.status.startswith(FAIL)

    @property
    def satisfied(self) -> bool:
        return self.passed or not self.blocking

    def render(self) -> str:
        return f"[{self.status:15}] {self.id:6} {self.question}\n                  {self.detail}"


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
        "source_confirmed": bounds.confirmed,
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
    sample = (f"no breach in {total} observations over {days}d; worst was "
              f"{worst:.6g}/h on {worst_coin}, "
              f"{worst / bounds.cap_per_hour:.1%} of the configured cap")

    if bounds.confirmed:
        # Two independent things, and PASS needs both. The citation says the
        # constant is what the protocol specifies; the sample says the venue
        # has not been observed contradicting it. Either alone is weaker than
        # it looks — a citation can be stale, and a quiet month proves nothing
        # about a bound nothing approached.
        return Check(
            "C1", "is the documented funding clamp real?", PASS,
            f"the cap is confirmed against a protocol reference — {bounds.source} — "
            f"and live data is consistent with it: {sample}. The sample does not "
            f"establish the bound and is not asked to; it would have falsified it.",
            evidence=evidence,
        )

    return Check(
        "C1", "is the documented funding clamp real?", INCONCLUSIVE,
        f"{sample}. This falsifies nothing and confirms nothing: a clamp is a "
        f"protocol constant and no sample of realised rates can establish it — "
        f"the largest rate seen is {worst / bounds.cap_per_hour:.2%} of the cap, so "
        f"the venue has never been near it. Read the value out of Hyperliquid's "
        f"documentation or source and record it with "
        f"`FundingBounds.from_protocol_source(cap, '<url or file:line>')`, which "
        f"is what makes this check able to PASS. Until 2026-07-31 it could not: "
        f"it built its own unconfirmed default, so following this instruction "
        f"changed nothing about its output. Current source: {bounds.source}",
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
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    since = _dt.fromtimestamp((now_ms - window_days * 24 * HOUR_MS) / 1000, tz=_tz.utc)
    until = _dt.fromtimestamp(now_ms / 1000, tz=_tz.utc)

    # Every unreadable record, not the first one. `net_external_flow` raises on
    # the first refusal by design -- a resolver must not proceed on a partial
    # read -- but this harness is the opposite job: it exists so the whole
    # ledger's worth of surprises is known BEFORE the clock starts.
    #
    # This was a real cost, not a hypothetical one. Two live rounds against the
    # same account surfaced `send`, then `spotTransfer`, one per run, each
    # needing a fix, a push, a pull and a re-run to reach the next. Ledger
    # types are a long tail; discovering them one per round trip is the slowest
    # possible way to find out, and it is the operator's time being spent.
    #
    # Each type is probed on its OWN records so one bad type cannot mask
    # another, and every record of a type is tried before it is called good --
    # `spotTransfer` was well-formed in some rows and not others.
    per_type_failures: dict[str, str] = {}
    for kind in sorted(kinds):
        of_kind = [r for r in rows if ((r.get("delta") or {}).get("type")) == kind]
        for row in of_kind:
            try:
                net_external_flow([row], since, until, address)
            except ValueError as exc:
                per_type_failures[kind] = str(exc)
                # The FAILING record, overwriting any earlier example of this
                # type: a well-formed row of the same type is what makes a
                # refusal look inexplicable.
                evidence["examples"][kind] = row
                break

    if per_type_failures:
        detail = "; ".join(f"{k}: {v}" for k, v in per_type_failures.items())
        return Check(
            "B2", "can external flows be read and classified?", FAIL,
            f"{len(per_type_failures)} of {len(kinds)} delta type(s) could not be "
            f"read. ALL of them are listed here rather than one per run, so this "
            f"can be fixed in a single pass — {detail}",
            evidence=evidence | {"unreadable_types": sorted(per_type_failures)},
        )

    try:
        total = net_external_flow(rows, since, until, address)
    except ValueError as exc:
        # Unreachable if the per-record probe above is faithful, which is
        # exactly why it is worth catching: reaching here means whole-ledger
        # evaluation refuses something no single record does, and reporting
        # that as a traceback would hide a real defect in this harness.
        return Check(
            "B2", "can external flows be read and classified?", FAIL,
            f"every record read individually, but the ledger as a whole did not: "
            f"{exc}. That is a defect in this check rather than in the data.",
            evidence=evidence,
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

    # Net flow per type, not just the total. A single aggregate is unfalsifiable:
    # a sign error on the dominant type produces a large plausible-looking number,
    # and "-18,472,046.51" reads the same whether it is right or inverted.
    #
    # The breakdown is what makes it checkable, because the types differ in how
    # much of this build's judgement they carry. `deposit`/`withdraw` are
    # fixed-sign and were never in doubt; `send` is routed by dex and by which
    # side the account was on, which is new logic decided from a single live
    # record. Seeing which type dominates tells an operator how much of the
    # total rests on the part most likely to be wrong.
    by_type: dict[str, float] = {}
    for kind in sorted(kinds):
        of_kind = [r for r in rows if ((r.get("delta") or {}).get("type")) == kind]
        by_type[kind] = net_external_flow(of_kind, since, until, address)
    contributing = {k: v for k, v in by_type.items() if v}
    # Per-type sums must reconstruct the total. They are computed by the same
    # function over a partition of the same rows, so a mismatch means the
    # accumulator is order-dependent -- worth catching here rather than in a
    # calibration score three weeks later.
    drift = abs(sum(by_type.values()) - total)
    scale = max(1.0, abs(total))
    if drift / scale > 1e-9:
        return Check(
            "B2", "can external flows be read and classified?", FAIL,
            f"the per-type breakdown does not reconstruct the total "
            f"({sum(by_type.values()):+,.2f} vs {total:+,.2f}). Flow is being "
            f"double-counted or dropped depending on how rows are grouped.",
            evidence=evidence | {"net_flow_usd": total, "net_flow_by_type": by_type},
        )

    parts = ", ".join(f"{k} {v:+,.2f}" for k, v in
                      sorted(contributing.items(), key=lambda kv: -abs(kv[1])))
    return Check(
        "B2", "can external flows be read and classified?", PASS,
        f"{len(rows)} ledger records over {window_days}d, every delta type "
        f"recognised ({', '.join(sorted(kinds))}); net flow {total:+,.2f} USD "
        f"[{parts or 'no type moved perp equity'}].",
        evidence=evidence | {"net_flow_usd": total, "net_flow_by_type": by_type},
    )


#: How long to wait for the venue to answer a subscribe. Long enough that a
#: slow ack is not read as a rejection, short enough that a `verify` run does
#: not stall: silence is the *inconclusive* outcome here, not a failure, so
#: erring long costs only time.
WEBDATA_PROBE_TIMEOUT_S = 8.0


def _probe_subscription(ws_url: str, sub_type: str, address: str | None,
                        timeout_s: float) -> tuple[bool | None, str]:
    """Subscribe to one channel and report whether the venue accepted it.

    Returns `(accepted, detail)`, where `accepted is None` means the venue
    said nothing either way inside the timeout — genuinely inconclusive, and
    distinct from a refusal.

    `websockets` is imported through the collector's own lazy binding rather
    than at module scope, for the reason stated there: it is not an engine
    dependency and must not become one by way of this file.
    """
    import asyncio
    import json as _json

    from risk_engine.market.collect_addresses import _websockets_transport

    payload: dict = {"method": "subscribe", "subscription": {"type": sub_type}}
    if address:
        payload["subscription"]["user"] = address

    async def _run() -> tuple[bool | None, str]:
        transport = _websockets_transport(ws_url)
        async with transport.connect() as ws:
            await ws.send(_json.dumps(payload))
            deadline = asyncio.get_running_loop().time() + timeout_s
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    return None, (
                        f"connected and sent the {sub_type} subscribe, and the venue "
                        f"neither acknowledged nor rejected it within {timeout_s:.0f}s. "
                        f"Silence is not a refusal."
                    )
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                msg = _json.loads(raw)
                channel = msg.get("channel")
                # An explicit error naming the subscription is the clearest
                # possible answer, and the venue gives one for unknown types.
                if channel == "error":
                    return False, f"the venue rejected it: {msg.get('data')!r}"
                if channel == "subscriptionResponse":
                    got = ((msg.get("data") or {}).get("subscription") or {}).get("type")
                    if got == sub_type:
                        return True, f"the venue acknowledged the {sub_type} subscription"
                # Data on the channel itself is acceptance by demonstration.
                if channel == sub_type:
                    return True, f"the venue delivered a {sub_type} frame"

    try:
        return asyncio.run(asyncio.wait_for(_run(), timeout=timeout_s + 10.0))
    except ImportError as exc:
        return None, f"cannot probe: {exc}"
    except Exception as exc:
        # Broad on purpose: a DNS failure, a TLS refusal and a protocol error
        # are all "we did not get an answer", which is inconclusive rather
        # than evidence about the subscription. Reported with its type, never
        # swallowed -- and C4 is non-blocking, so a probe that cannot run must
        # not take the whole verification down with it.
        return None, f"the probe itself failed: {type(exc).__name__}: {exc}"


def check_webdata3(ws_url: str | None = None, address: str | None = None,
                   probe: bool = False) -> Check:
    """C4, now answerable rather than merely described.

    This returned UNCHECKABLE with the reason "this harness speaks only the
    Info POST API". That was true when written and stopped being true when
    `collect_addresses` shipped: the collector has since held a live
    WebSocket session against this exact venue for 1 110 frames. The
    capability existed in the tree and this check did not use it, which is the
    same defect class as A10 — a stated limitation that had quietly become
    false.

    Off by default (`--probe-ws`) because it needs the `websockets` package,
    which the engine deliberately does not depend on, and because a socket is
    a different kind of cost from a POST. C4 is non-blocking either way: the
    shard planner works against `webData2` regardless of the answer.
    """
    if not probe:
        return Check(
            "C4", "does the `webData3` subscription exist?", UNCHECKABLE,
            "not probed. Pass --probe-ws to answer it: this harness can now open "
            "the socket (the trades collector does, against the same venue), it "
            "just needs `pip install 'websockets>=12.0'`, which the engine does "
            "not depend on. `webData2` is the documented one and the shard "
            "planner is agnostic either way.",
            blocking=False,
        )

    from risk_engine.market.collect_addresses import MAINNET_WS_URL

    url = ws_url or MAINNET_WS_URL
    accepted, detail = _probe_subscription(
        url, "webData3", address, WEBDATA_PROBE_TIMEOUT_S
    )
    evidence = {"ws_url": url, "subscription": "webData3", "accepted": accepted}
    if accepted is True:
        return Check(
            "C4", "does the `webData3` subscription exist?", PASS,
            f"{detail}. The shard planner may use it; it is not required to.",
            blocking=False, evidence=evidence,
        )
    if accepted is False:
        # A rejection is a real answer and NOT a failure of the model: `webData3`
        # was only ever a maybe, and the planner is specified against `webData2`.
        return Check(
            "C4", "does the `webData3` subscription exist?", PASS,
            f"answered: `webData3` does not exist on this venue — {detail}. "
            f"The shard planner uses `webData2`, so nothing depends on it.",
            blocking=False, evidence=evidence,
        )
    return Check(
        "C4", "does the `webData3` subscription exist?", INCONCLUSIVE,
        f"{detail}", blocking=False, evidence=evidence,
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


# -- recorded findings ----------------------------------------------------


#: How a recorded status renders, so it can never be read as a live result.
RECORDED_SUFFIX = " (recorded)"


def apply_recorded(check: Check, finding, now=None) -> Check:
    """Fold an out-of-band verification into a live check's result.

    The rules, in the order they matter:

    1. **A live verdict wins.** If this run reached PASS or FAIL, the venue
       answered today and history is not needed. When the two *disagree*, the
       disagreement is the finding — reported on the live verdict rather than
       hidden behind it, because a recorded PASS now contradicted is the exact
       event this whole harness exists to surface.
    2. **A recording only fills a gap.** It is consulted when the live check
       came back INCONCLUSIVE or UNCHECKABLE, which is the case it was written
       for: C2 needs hours, C5 needs a funded account across a funding tick,
       and neither fits inside one CLI run.
    3. **A stale recording vouches for nothing.** Past `STALE_AFTER_DAYS` it
       degrades to INCONCLUSIVE with its age stated, rather than standing in
       for a fact that may have moved.
    """
    if finding is None:
        return check

    prov = finding.provenance(now)

    if check.status in (PASS, FAIL):
        if finding.status != check.status:
            return Check(
                check.id, check.question, check.status,
                f"{check.detail} NOTE: this contradicts a recorded "
                f"{finding.status} — {prov}. The live venue is the authority; "
                f"the recording is stale, wrong, or the venue changed.",
                blocking=check.blocking,
                evidence=check.evidence | {"contradicted_recording": finding.status},
            )
        return check

    if finding.is_stale(now):
        return Check(
            check.id, check.question, INCONCLUSIVE,
            f"{check.detail} A recorded {finding.status} exists but is older than "
            f"{STALE_AFTER_DAYS} days and no longer vouches for anything — "
            f"{prov}. Re-run it.",
            blocking=check.blocking,
            evidence=check.evidence | {"stale_recording": finding.status},
        )

    return Check(
        check.id, check.question, finding.status + RECORDED_SUFFIX,
        f"{finding.detail} [{prov}] — not checked by this run: {check.detail}",
        blocking=check.blocking,
        evidence=check.evidence | {"recorded": {
            "status": finding.status, "observed_utc": finding.observed_utc,
            "command": finding.command, "network": finding.network,
            "age_days": round(finding.age_days(now), 1),
            **({"evidence": finding.evidence} if finding.evidence else {}),
        }},
    )


# -- driver ---------------------------------------------------------------


def run_all(address: str | None, coins: list[str], days: int, samples: int,
            interval_s: float, testnet: bool, probe_ws: bool = False,
            findings: dict | None = None) -> list[Check]:
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
    # Testnet has its own socket. Inferring the URL was called out as a
    # guess in `collect_addresses`, so it is not inferred here either --
    # probing testnet needs the URL passed explicitly.
    checks.append(check_webdata3(address=address, probe=probe_ws and not testnet))
    checks.append(check_isolated_funding())

    # Folded in last, over the finished list, so every check is written and
    # tested as a pure live check that knows nothing about recorded history.
    # A check that consulted the file itself could not be tested for what it
    # does when the file disagrees with it.
    recorded = findings if findings is not None else {}
    return [apply_recorded(c, recorded.get(c.id)) for c in checks]


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
    parser.add_argument("--probe-ws", dest="probe_ws", action="store_true",
                        help="answer C4 by opening a WebSocket and subscribing to "
                             "webData3. Needs `pip install 'websockets>=12.0'`, which "
                             "the engine deliberately does not depend on. Mainnet only "
                             "-- the testnet socket URL is a guess this build will not "
                             "make silently.")
    parser.add_argument("--report", help="write the full result as JSON")
    parser.add_argument("--findings", default=None,
                        help="JSON of verifications made outside this harness "
                             "(default docs/hl-risk/VERIFIED.json). C2 needs hours "
                             "and C5 needs a funded account across a funding tick, "
                             "so neither fits in one run; without this the summary "
                             "reports settled questions as open forever.")
    parser.add_argument("--no-findings", dest="no_findings", action="store_true",
                        help="ignore recorded findings and report only what this "
                             "run established")
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
    if args.probe_ws and args.testnet:
        parser.error(
            "--probe-ws is mainnet only: the testnet WebSocket URL is not "
            "documented here, and guessing it would report 'no such subscription' "
            "for an endpoint that was simply never reached."
        )
    findings: dict = {}
    if not args.no_findings:
        try:
            findings = load_findings(args.findings)
        except ValueError as exc:
            # A malformed findings file is the operator's own input, so it is
            # a usage error and not evidence about the venue -- the same
            # reasoning that makes a bad `--address` a usage error rather than
            # an E5.3 FAIL.
            parser.error(str(exc))

    checks = run_all(address, coins, args.days, args.samples,
                     args.interval_s, args.testnet, args.probe_ws, findings)

    print(f"live-API verification against {'testnet' if args.testnet else 'mainnet'}\n")
    for check in checks:
        print(check.render())
        print()

    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump([asdict(c) for c in checks], fh, indent=2, default=str)
        print(f"wrote {args.report}")

    failed = [c for c in checks if c.failed]
    unconfirmed = [c for c in checks if not c.satisfied and not c.failed]
    carried = [c for c in checks if c.status.endswith(RECORDED_SUFFIX)]
    if failed:
        print(f"\n{len(failed)} assumption(s) CONTRADICTED by live data: "
              f"{', '.join(c.id for c in failed)}. The model is wrong today; "
              "fix it before the live path runs.")
        return 2
    if unconfirmed:
        print(f"\n{len(unconfirmed)} assumption(s) still unconfirmed: "
              f"{', '.join(c.id for c in unconfirmed)}. Nothing here contradicts "
              "the model, and nothing here establishes it either.")
        if carried:
            # Said even on the failure path: an operator reading "1 unconfirmed"
            # needs to know which of the satisfied ones rest on history rather
            # than on this run, or the line understates what is being assumed.
            print(f"{len(carried)} rest(s) on a recorded finding rather than on "
                  f"this run: {', '.join(c.id for c in carried)}. Re-run those "
                  f"commands if anything about the venue may have changed.")
        return 1
    if carried:
        print(f"\nEvery blocking assumption satisfied, but {len(carried)} of them "
              f"({', '.join(c.id for c in carried)}) rest on a recorded finding "
              f"rather than on this run. That is not the same as verified today.")
        return 0
    print("\nEvery blocking assumption confirmed against live data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
