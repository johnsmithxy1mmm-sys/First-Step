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
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from risk_engine.domain.types import normalise_address
from risk_engine.market.findings import (
    DEFAULT_FINDINGS_PATH,
    STALE_AFTER_DAYS,
    load_findings,
)
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

#: How often the frame sweep says where it is. Matches the shadow sweep's
#: cadence, and for the same reason: both spend most of their time asleep on
#: the §5.3 window, so both are outwardly indistinguishable from a hang.
PROGRESS_EVERY_S = 30.0


def _is_self_inflicted(exc: BaseException) -> bool:
    """Whether a failure is this harness's own doing rather than the venue's.

    `RateLimitExceeded` is raised by our OWN `WeightBudget` before a request
    leaves the process: it means this run has spent its §5.3 allowance, not
    that the venue said anything. Reporting it as FAIL claims live data
    contradicted the model — the strongest verdict this tool has, worth exit
    code 2 and "the model is wrong today" — on the strength of a self-imposed
    limit.

    Observed exactly that way: a 50-address frame sweep spent the minute's
    budget, C2's next request was refused locally, and C2 came back FAIL
    "contradicts a recorded PASS". Nothing about the basis had changed. The
    module docstring already draws this line for a malformed `--address`;
    a self-imposed rate limit is the same category and was not covered.

    Since `_verify_budget` paces, this is no longer reachable by simply
    spending the minute's allowance -- that now waits. What remains reachable
    is the ceiling on the wait, which means a pool held by something else for
    ten minutes. Still self-inflicted in the sense that matters here: the
    venue never answered, so no verdict about the venue is available.
    """
    from risk_engine.market.info import RateLimitExceeded

    return isinstance(exc, RateLimitExceeded)


def _self_limited(check_id: str, question: str, exc: BaseException) -> "Check | None":
    """The UNCHECKABLE result for a self-inflicted failure, or None.

    Audit F-3: only C2 and the frame sweep had this guard; `check_meta`,
    `check_candles`, `check_clearinghouse`, `check_funding_clamp` and
    `check_external_flow` all reported a self-imposed rate limit through
    their generic `except → FAIL` — exit 2, "the model is wrong today",
    about a request that never left the process. C1 escaped that fate in
    the live run that exposed this only because the sweep's budget had
    partially refilled by the time it ran: luck, not code. One helper,
    called first from every venue-touching except site, so the next check
    added cannot quietly reintroduce the hole.
    """
    if not _is_self_inflicted(exc):
        return None
    return Check(
        check_id, question, UNCHECKABLE,
        f"the §5.3 weight budget stayed spent for the whole wait ceiling, so "
        f"this request was never made ({exc}). That is a rate limit on our "
        f"side, not the venue answering — this run shares its budget with the "
        f"shadow jobs, so check whether one of them is holding the pool "
        f"(`docker compose ps`), or lower --frame-sample. FAIL here would "
        f"claim live data contradicted the model on the strength of a "
        f"self-imposed limit.",
    )


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
        # 23 = len("INCONCLUSIVE (recorded)"), the widest constructible status
        # (F-6 pins the recordable set to PASS/FAIL/INCONCLUSIVE). Sized to
        # the widest rather than the common case so the columns hold whatever
        # combination a run produces.
        return f"[{self.status:23}] {self.id:6} {self.question}\n" + " " * 26 + self.detail


# -- E5: the parsers have never seen a live response ---------------------


def check_meta(client: InfoClient) -> tuple[Check, dict | None]:
    try:
        raw = client.meta()
        specs = parse_meta(raw)
    except Exception as exc:
        if (c := _self_limited("E5.1", "does `meta` parse into margin tiers?", exc)):
            return c, None
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
        if (c := _self_limited("E5.2", f"does `candleSnapshot` parse for {coin}?", exc)):
            return c, None
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
        if (c := _self_limited("E5.3", "does `clearinghouseState` parse into a Book?", exc)):
            return c
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
    bounds = FundingBounds.hyperliquid_confirmed()
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
            if (c := _self_limited("C1", "is the documented funding clamp real?", exc)):
                return c
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
            if _is_self_inflicted(exc):
                return Check(
                    "C2", "is mark ≈ mid, per §1.4's threshold?", UNCHECKABLE,
                    f"the §5.3 weight budget stayed spent for the whole wait "
                    f"ceiling, so C2 could not sample ({exc}). That is a rate "
                    f"limit on our side, not the venue answering — this run "
                    f"shares its budget with the shadow jobs, so check whether "
                    f"one of them is holding the pool (`docker compose ps`), or "
                    f"lower --frame-sample. Reporting it as FAIL would claim live "
                    f"data contradicted the model on the strength of a "
                    f"self-imposed limit.",
                    evidence={"samples_taken": i},
                )
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


def check_frame_ledger_types(client: InfoClient, addresses: list[str],
                             sample: int, window_days: int = 90) -> Check:
    """B2 across the actual sampling frame, not one account (§3.3 pre-flight).

    `check_external_flow` proves the ledger of *one* address is readable. The
    shadow window resolves 200-500 of them, every day, for 21 days, and the
    types a single account happens to have are not the types the cohort has.
    One real account produced five (`deposit`, `withdraw`, `send`,
    `spotTransfer`, `spotGenesis`) and each of the last three had to be
    classified from a live record — two of them found one run apart, on the
    same address.

    What makes this worth a command rather than a note. `resolve_due` calls
    `external_flow` per row; an unclassifiable type raises; `_permanent_reason`
    classifies the failure by name and an unrecognised one lands in TRANSIENT;
    a TRANSIENT failure is retried forever. So a single unknown delta type on
    a single address, encountered on day 6, does not stop the run and does not
    announce itself — it quietly withholds that address's observations while
    the counter fails to advance, visible only as a repeating traceback in a
    container log. The whole 21 days can be spent accumulating snapshots that
    never resolve.

    Three types are known-unproven and are the reason this is not paranoia:
    `internalTransfer`, `subAccountTransfer` and `accountClassTransfer` are
    all filed as directional-needs-`toPerp`, and the two dex-routed types that
    HAVE been seen live (`send`, `spotTransfer`) both carried
    `user`/`destination` instead. If those three follow the same shape they
    will raise on first contact. None appeared on the one address checked so
    far; across hundreds of accounts they are near-certain.

    Costs `sample * 20` weight — about a minute for 50 addresses, against 21
    days of window it would otherwise risk.
    """
    from risk_engine.market.parse import (
        DEX_ROUTED_TYPES,
        EXTERNAL_FLOW_SIGNS,
        NON_FLOW_DELTA_TYPES,
        net_external_flow,
    )
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    if not addresses:
        return Check(
            "B2.frame", "can every ledger type in the sampling frame be read?",
            UNCHECKABLE,
            "no --addresses file given. B2 proves one account is readable; the "
            "shadow window resolves hundreds, and an unknown type on any one of "
            "them is retried forever rather than reported (§3.3).",
        )

    # Deterministic, and the first N rather than a random draw: an operator who
    # re-runs after a fix must see the same addresses, or a type that vanished
    # is indistinguishable from a type that was never sampled.
    chosen = addresses[:sample]
    now_ms = int(time.time() * 1000)
    since = _dt.fromtimestamp((now_ms - window_days * 24 * HOUR_MS) / 1000, tz=_tz.utc)
    until = _dt.fromtimestamp(now_ms / 1000, tz=_tz.utc)

    kinds: dict[str, int] = {}
    examples: dict[str, dict] = {}
    failures: dict[str, str] = {}
    unreachable: list[dict] = []
    n_records = 0
    n_read = 0

    budget_stopped = False
    started = time.monotonic()
    last_progress = started
    for i, addr in enumerate(chosen, 1):
        # This loop is where a verification run spends nearly all its wall
        # clock, and nearly all of THAT is asleep: 200 addresses is 4000
        # weight against a 300/min shared pool, so roughly twelve of the
        # thirteen minutes are waiting for the window. Silence here reads as
        # a hang. Elapsed and waiting are reported separately for the same
        # reason the shadow sweep separates them -- mostly-waiting is §5.3
        # working as designed, elapsed climbing while waiting does not is a
        # stall worth acting on.
        if time.monotonic() - last_progress >= PROGRESS_EVERY_S:
            last_progress = time.monotonic()
            waited = _waited_s(client)
            _step(f"B2.frame {i}/{len(chosen)} addresses, {n_records} records, "
                  f"{len(kinds)} delta types, {time.monotonic() - started:.0f}s "
                  f"elapsed ({waited:.0f}s waiting)")
        try:
            rows = list(client.non_funding_ledger_updates(
                addr, now_ms - window_days * 24 * HOUR_MS, now_ms) or [])
        except Exception as exc:
            if _is_self_inflicted(exc):
                # Stop the sweep rather than burn through the rest as
                # "unreachable", and stop it HERE rather than starving every
                # check that runs after this one. A 50-address scan is 1000 of
                # the 1200 weight a minute allows, so the first version of this
                # check emptied the budget and C2 came back FAIL against a
                # request that never left the process.
                budget_stopped = True
                break
            # One address failing is a fact about that address, not about the
            # frame. Recorded and stepped over: aborting here would make a
            # single dead account hide every type on the remaining ones.
            unreachable.append({"address": addr, "error": f"{type(exc).__name__}: {exc}"})
            continue
        n_read += 1
        n_records += len(rows)
        for row in rows:
            kind = ((row.get("delta") or {}).get("type")) or "<no delta.type>"
            kinds[kind] = kinds.get(kind, 0) + 1
            if kind in failures:
                continue
            try:
                net_external_flow([row], since, until, addr)
            except ValueError as exc:
                failures[kind] = str(exc)
                examples[kind] = row

    from risk_engine.market.parse import (
        PERP_ADDRESS_ROUTED_TYPES,
        UNRECOGNISED_DEX_NAMES,
    )

    known = (set(EXTERNAL_FLOW_SIGNS) | set(NON_FLOW_DELTA_TYPES)
             | set(DEX_ROUTED_TYPES) | set(PERP_ADDRESS_ROUTED_TYPES))
    unseen = sorted(k for k in known if k not in kinds)
    evidence = {
        "addresses_sampled": len(chosen), "addresses_read": n_read,
        "addresses_unreachable": unreachable, "n_records": n_records,
        "types_seen": kinds, "unreadable_types": sorted(failures),
        "known_types_not_exercised": unseen, "examples": examples,
        "window_days": window_days, "budget_stopped": budget_stopped,
        # Dex names treated as builder-deployed venues rather than the primary
        # perp account. Reported because the one way that reading goes wrong is
        # a future ALIAS of the primary dex being read as a builder one, which
        # would hide a real outflow.
        "unrecognised_dex_names": dict(UNRECOGNISED_DEX_NAMES),
    }

    if failures:
        detail = "; ".join(f"{k}: {v}" for k, v in failures.items())
        return Check(
            "B2.frame", "can every ledger type in the sampling frame be read?", FAIL,
            f"{len(failures)} delta type(s) across {n_read} sampled addresses cannot "
            f"be read. Each one would fail its address's resolution silently and be "
            f"retried forever, so the §3.3 counter would not advance and nothing "
            f"would say why — {detail}",
            evidence=evidence,
        )

    if n_read == 0:
        return Check(
            "B2.frame", "can every ledger type in the sampling frame be read?", FAIL,
            f"none of the {len(chosen)} sampled addresses could be read at all. "
            f"That is the endpoint or the address list, not the classifier.",
            evidence=evidence,
        )

    # Not a PASS with a footnote: the types this sample did NOT contain are
    # exactly the ones that will surface on day 9 of the window. Naming them is
    # the difference between "checked" and "checked, and here is what remains
    # untested".
    note = (f" {len(unseen)} known type(s) never appeared and remain untested "
            f"against live data: {', '.join(unseen)}." if unseen else "")
    if budget_stopped:
        note += (f" Stopped at {n_read} addresses: this run spent its §5.3 weight "
                 f"budget. The types below are what {n_read} addresses hold, not "
                 f"what {len(chosen)} do — re-run with a smaller --frame-sample, "
                 f"or a minute later.")
    if UNRECOGNISED_DEX_NAMES:
        note += (f" Dex names read as builder-deployed venues rather than the "
                 f"primary perp account: {sorted(UNRECOGNISED_DEX_NAMES)}. Transfers "
                 f"to them count as outflows; if any of these is in fact an alias "
                 f"for the primary dex, add it to PERP_DEX_VALUES.")
    return Check(
        "B2.frame", "can every ledger type in the sampling frame be read?", PASS,
        f"{n_records} records across {n_read} of {len(chosen)} sampled addresses, "
        f"{len(kinds)} distinct delta types, all readable "
        f"({', '.join(sorted(kinds))}).{note}",
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
        PERP_ADDRESS_ROUTED_TYPES,
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
        if (c := _self_limited("B2", "can external flows be read and classified?", exc)):
            return c
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
    # Every table a type can legitimately live in. Missing one here reports a
    # perfectly classifiable type as unknown -- which is how `internalTransfer`
    # came back as "the venue returned a type this build cannot classify" the
    # moment it was moved out of EXTERNAL_FLOW_SIGNS and into its own.
    known = (set(EXTERNAL_FLOW_SIGNS) | set(NON_FLOW_DELTA_TYPES)
             | set(DEX_ROUTED_TYPES) | set(PERP_ADDRESS_ROUTED_TYPES))
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
    a different kind of cost from a POST.

    **Both subscriptions are probed, and that is the point.** Asking only
    about `webData3` produced a year of answers phrased as "it does not
    matter, the planner uses `webData2`" — and on 2026-08-03 a probe of both
    found that `webData2` is REFUSED by this venue, with a well-formed `user`,
    in the exact payload shape `webData3` accepts. A one-sided question could
    not have found that, because the assumed-good alternative was never asked
    about. C4 is now the comparison rather than the existence check.

    Non-blocking still, and for a better reason than the one given before:
    nothing in this tree consumes either subscription. The "shard planner"
    every previous version of this comment deferred to is a §5.2 concept with
    no implementation here — `grep -rn shard` finds only these comments — so
    "the planner uses `webData2`" was never a fact about running code.
    """
    if not probe:
        return Check(
            "C4", "which of `webData2` / `webData3` does the venue accept?",
            UNCHECKABLE,
            "not probed. Pass --probe-ws to answer it: this harness can now open "
            "the socket (the trades collector does, against the same venue), it "
            "just needs `pip install 'websockets>=12.0'`, which the engine does "
            "not depend on. Pass --address too: both subscriptions are keyed on "
            "`user`, and without one they are refused for that reason alone, "
            "which reads exactly like 'no such subscription'.",
            blocking=False,
        )

    from risk_engine.market.collect_addresses import MAINNET_WS_URL

    url = ws_url or MAINNET_WS_URL
    three, three_detail = _probe_subscription(
        url, "webData3", address, WEBDATA_PROBE_TIMEOUT_S
    )
    two, two_detail = _probe_subscription(
        url, "webData2", address, WEBDATA_PROBE_TIMEOUT_S
    )
    evidence = {
        "ws_url": url,
        "webData3_accepted": three,
        "webData3_detail": three_detail,
        "webData2_accepted": two,
        "webData2_detail": two_detail,
        "probed_with_user": bool(address),
    }
    question = "which of `webData2` / `webData3` does the venue accept?"

    if not address:
        # Without a `user` both are refused for the same uninformative reason,
        # and reporting that as an answer is how the earlier reading went
        # wrong. Say what is missing instead of resolving it.
        return Check(
            "C4", question, INCONCLUSIVE,
            f"probed without --address, so both were refused for a missing "
            f"`user` rather than for anything about the subscription. "
            f"webData3: {three_detail}; webData2: {two_detail}.",
            blocking=False, evidence=evidence,
        )
    if three is None or two is None:
        return Check(
            "C4", question, INCONCLUSIVE,
            f"webData3: {three_detail}; webData2: {two_detail}",
            blocking=False, evidence=evidence,
        )
    if three and not two:
        return Check(
            "C4", question, PASS,
            f"`webData3` is the live one and `webData2` is NOT: {three_detail}, "
            f"while webData2 in the same payload shape was refused — "
            f"{two_detail}. §5.2 names `webData2`; anything built to that "
            f"letter would subscribe to something this venue rejects.",
            blocking=False, evidence=evidence,
        )
    if two and not three:
        return Check(
            "C4", question, PASS,
            f"`webData2` is the live one, as §5.2 assumes, and `webData3` is "
            f"not: {two_detail}, while webData3 was refused — {three_detail}.",
            blocking=False, evidence=evidence,
        )
    if two and three:
        return Check(
            "C4", question, PASS,
            f"both exist: webData3 — {three_detail}; webData2 — {two_detail}. "
            f"Existence says nothing about which carries what; §5.2's choice "
            f"stands unchallenged by this result.",
            blocking=False, evidence=evidence,
        )
    return Check(
        "C4", question, FAIL,
        f"NEITHER subscription was accepted with a well-formed `user`. "
        f"webData3: {three_detail}; webData2: {two_detail}. Both are documented; "
        f"a venue refusing both means the documented shape has changed.",
        blocking=False, evidence=evidence,
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


def _verify_budget(journal: str | None):
    """This harness's §5.3 share: background class, shared pool, paced.

    C6 fixed the shadow jobs double-counting the reserve and left this tool
    out, because it is run by hand rather than scheduled. That is not a
    difference the venue observes. `InfoClient()`'s default budget is the
    interactive one -- all 1200/minute, no reserve -- so a verification run
    launched while the stack is up put a third ceiling on one egress IP:
    900 serving plus 300 shadow plus 1200 here, against a limit of 1200.

    It is background traffic by every test that matters (an operator waiting
    on a diagnostic is not a user waiting on a price), so it takes the
    background reserve and, when the calibration database is reachable,
    charges the SAME ledger the sweep and the resolver share. Then running a
    verification during a sweep is slow instead of unsound, which is the
    outcome worth having: the previous answer was "stop the shadow jobs
    first", and an operational precondition nothing enforces is one somebody
    eventually forgets.

    Paced for a reason this tool already learned the hard way. Its 300/minute
    share is less than one full run costs -- a 50-address frame sweep alone
    is 1000 weight -- and an unpaced charge turns that arithmetic into
    UNCHECKABLE results about requests that never left the process, which is
    exactly what `_is_self_inflicted` was written for. Waiting converts the
    same arithmetic into a run that takes about four minutes.

    A fallback that stayed quiet would be the C6 defect itself: believing you
    share a pool while holding a private one is invisible from inside. So the
    pool actually joined is stated on every run.
    """
    from risk_engine.market.info import (
        SHADOW_RESERVED_FRACTION,
        WEIGHT_BUDGET_PER_MINUTE,
        PacedBudget,
        WeightBudget,
    )

    target = journal or os.environ.get("SHADOW_DSN") or None
    rate = int(WEIGHT_BUDGET_PER_MINUTE * (1.0 - SHADOW_RESERVED_FRACTION))
    inner: Any
    try:
        from risk_engine.shadow.weight_ledger import open_weight_budget

        inner = open_weight_budget(
            target, reserved_fraction=SHADOW_RESERVED_FRACTION, actor="verify"
        )
    except Exception as exc:
        # Unreachable database, no psycopg, a DSN that resolves only inside
        # the compose network: all of them mean this run cannot see the other
        # jobs' spending, and none of them is a reason to refuse to verify.
        print(f"note: shared §5.3 ledger unavailable ({type(exc).__name__}: {exc}); "
              f"this run holds a PRIVATE {rate}/min window. Do not run it "
              f"alongside the shadow containers.\n")
        inner = WeightBudget(reserved_fraction=SHADOW_RESERVED_FRACTION)
    else:
        shared = type(inner).__name__ == "SharedWeightBudget"
        print(f"§5.3: {rate}/min, "
              + (f"shared with the shadow jobs via {target.split('@')[-1]}"
                 if shared else "private to this process")
              + ". A spent window is waited out, not reported as a result.\n")
    return PacedBudget(inner)


def _step(label: str) -> None:
    """Say which check is running, before it runs.

    This harness shares one 300/min pool with the shadow jobs and waits out
    a spent window rather than failing on it (C6), so a run legitimately
    spends most of its wall clock asleep — a 200-address frame sweep is 4000
    weight, about thirteen minutes of which twelve are waiting. Without a
    line per step that is indistinguishable from a hang, which is exactly the
    confusion the shadow sweep's silence caused before it was given progress
    output.

    stderr, so `--report` and piped stdout stay clean.
    """
    print(f"  … {label}", file=sys.stderr, flush=True)


def _waited_s(client) -> float:
    """Seconds this run has spent asleep on the §5.3 window, or 0.

    Tolerant of a client without a budget, because the client is injectable
    and every test drives a stub. A progress line is not worth an
    AttributeError in the middle of a verification run.
    """
    return float(getattr(getattr(client, "budget", None), "waited_s", 0.0) or 0.0)


def run_all(address: str | None, coins: list[str], days: int, samples: int,
            interval_s: float, testnet: bool, probe_ws: bool = False,
            findings: dict | None = None, frame_addresses: list[str] | None = None,
            frame_sample: int = 50, journal: str | None = None) -> list[Check]:
    from risk_engine.market.info import MAINNET_URL, TESTNET_URL

    client = InfoClient(url=TESTNET_URL if testnet else MAINNET_URL,
                        budget=_verify_budget(journal))
    checks: list[Check] = []

    _step("E5.1 meta -> margin tiers")
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

    _step(f"E5.2 candleSnapshot for {coins[0]}")
    candle_check, returns = check_candles(client, coins[0])
    checks.append(candle_check)
    hourly_vol = float(np.std(returns)) if returns is not None else None

    _step("E5.3 clearinghouseState")
    checks.append(check_clearinghouse(client, address))
    _step("B2 ledger delta types, one account")
    checks.append(check_external_flow(client, address))
    if frame_addresses is not None:
        n = min(frame_sample, len(frame_addresses))
        _step(f"B2.frame ledger types across {n} addresses "
              f"(~{n * 20 // 300 + 1} min, mostly waiting on the §5.3 window)")
        checks.append(check_frame_ledger_types(client, frame_addresses, frame_sample))
    _step(f"C1 funding clamp over {days}d x {len(coins)} "
          f"coin{'s' if len(coins) != 1 else ''}")
    checks.append(check_funding_clamp(client, coins, days))
    _step(f"C2 basis, {samples} samples every {interval_s:.0f}s "
          f"(~{samples * interval_s / 60:.0f} min by construction)")
    checks.append(check_basis(client, coins, samples, interval_s, hourly_vol))
    # Testnet has its own socket. Inferring the URL was called out as a
    # guess in `collect_addresses`, so it is not inferred here either --
    # probing testnet needs the URL passed explicitly.
    if probe_ws and not testnet:
        _step("C4 webData2 / webData3 subscribe probe")
    checks.append(check_webdata3(address=address, probe=probe_ws and not testnet))
    checks.append(check_isolated_funding())

    waited = _waited_s(client)
    if waited:
        _step(f"done; {waited:.0f}s of that was waiting for the §5.3 window")

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
    parser.add_argument("--addresses", default=None,
                        help="the shadow address list (as written by "
                             "`collect_addresses`). Runs B2 across the actual "
                             "sampling frame instead of one account — the types a "
                             "single address happens to have are not the types 500 "
                             "of them have, and an unknown one is retried forever "
                             "rather than reported (§3.3).")
    parser.add_argument("--frame-sample", dest="frame_sample", type=int, default=50,
                        help="how many addresses from that list to scan "
                             "(default 50; costs 20 weight each)")
    parser.add_argument(
        "--journal", default=None,
        help="calibration journal (Postgres DSN), so this run charges the same "
             "§5.3 pool as the shadow jobs instead of a second private one. "
             "Defaults to $SHADOW_DSN, which the compose stack already sets; a "
             "SQLite path or an unreachable DSN falls back to a private window "
             "and says so.",
    )
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
            # Said out loud, because an empty result is indistinguishable from
            # "nothing was ever recorded" and was wrong for months: the engine
            # image did not COPY the findings file, so every containerised run
            # silently reported C2 and C5 as unconfirmed while the
            # measurements that closed them sat unreadable in the repository.
            # A missing file is a legitimate state (fresh checkout) and must
            # not raise — but it must not be invisible either.
            if findings:
                print(f"recorded findings: {', '.join(sorted(findings))} "
                      f"(from {args.findings or DEFAULT_FINDINGS_PATH})\n")
            else:
                print(f"no recorded findings at "
                      f"{args.findings or DEFAULT_FINDINGS_PATH}; every "
                      f"out-of-band result will read as unconfirmed\n")
        except ValueError as exc:
            # A malformed findings file is the operator's own input, so it is
            # a usage error and not evidence about the venue -- the same
            # reasoning that makes a bad `--address` a usage error rather than
            # an E5.3 FAIL.
            parser.error(str(exc))

    frame_addresses: list[str] | None = None
    if args.addresses:
        # Loaded through the same source the cron uses, so a list this accepts
        # is a list the sweep accepts -- including its refusal to run without a
        # stated frame (B4). Loading it here also means a malformed list costs
        # no §5.3 weight.
        from risk_engine.shadow.providers import FileAddressSource

        try:
            frame_addresses = FileAddressSource(args.addresses).addresses()
        except (OSError, ValueError) as exc:
            parser.error(f"--addresses {args.addresses}: {exc}")

    print(f"live-API verification against {'testnet' if args.testnet else 'mainnet'}\n")
    checks = run_all(address, coins, args.days, args.samples,
                     args.interval_s, args.testnet, args.probe_ws, findings,
                     frame_addresses, args.frame_sample, args.journal)

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
