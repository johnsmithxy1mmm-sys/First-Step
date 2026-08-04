"""Resolver: writes the realised outcome next to each prediction (§3.3).

Runs behind the cron. For every prediction whose horizon has elapsed it
records the realised equity, the PIT value, the CRPS and whether the
predicted VaR@95 was breached.

The PIT it records is the RANDOMIZED transform, not `F(x)` (audit A-02).
The predicted distribution has an atom at total loss, so a plain CDF value
maps every realised liquidation to one identical number and the KS test in
§3.3 rejects even a perfectly calibrated model.

It also records two things §3.3 does not mention but without which the whole
calibration exercise is unsound (OPEN-QUESTIONS B2): the net external cash
flow over the horizon, and whether the set of positions changed. Equity moves
when a user deposits, withdraws, opens or closes -- none of which the model
claimed to predict. Scoring those as model error measures how actively the
sampled traders traded. Both are recorded rather than filtered here, so the
cohort choice is made at analysis time and is visible.

RETRIES ARE UNBOUNDED, AND SOME OF THEM ARE FOREVER. A failed row gets no
outcome written, so `journal.due()` hands it back on the next run, and there
is no attempt counter anywhere. For a transient failure that is the whole
point. But one class of failure is deterministic in the stored row and so
never clears: a prediction whose `address` column is not an address. Those
rows exist because `record_prediction` only started canonicalising what it
writes later, and they stay visible because the read paths deliberately do
not normalise -- `journal.py` hands back the bytes that are actually stored,
so a legacy row reads as itself instead of being papered over by the very
check it predates. `LiveSnapshotProvider.book` normalises before it fetches,
so such a row raises before any request: measured resolved=0, failed=1,
still_due=1 on three consecutive runs, with the failed list never emptying.

That is documented rather than bounded, deliberately, and the alternatives
are worth naming because two of them are worse:

  - writing an outcome for it is what the pre-normalisation code effectively
    did. A typo'd address is answered by the venue with a well-formed *empty*
    state (§5.1), which resolved as equity 0 -- liquidated=1,
    var_95_breached=1, permanently, in a table that cannot be edited (§3.4).
    A retry loop is not in the same class of harm as a fabricated
    liquidation in a published calibration score;
  - dropping the row, or marking it resolved-and-excluded after N attempts,
    silently discards an observation the schema exists to make immutable, and
    the honest version of that needs a terminal state in the schema plus the
    versioning discussion that comes with changing it (§3.3: a distribution
    change resets the window). It is not a change to make as a side effect of
    a retry bound;
  - a retry *ceiling* with no terminal state just means the row stops being
    attempted while still counting as pending, which is the same forever-row
    with the evidence removed.

So the retry stays unbounded and stops being silent instead:
`ResolveReport.permanent` lists the rows that will fail identically on the
next run and why, the CLI prints them under that heading, and the operator's
fix is a deliberate, visible correction of the journal row. The cost of the
loop itself is bounded already -- these rows raise before any request, so
they spend no §5.3 weight, only a line of output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Protocol

import numpy as np

from risk_engine.domain.types import Book, normalise_address
import logging

from risk_engine.market.info import RateLimitExceeded
from risk_engine.observability.metrics import METRICS, Metrics
from risk_engine.shadow.cron import position_fingerprint
from risk_engine.shadow.journal import CalibrationJournal

#: Returned by the pacing wrapper when the whole-run ceiling is reached, so a
#: `None` from a legitimate call can never be mistaken for the truncation
#: signal.
_CEILING = object()


class OutcomeProvider(Protocol):
    def book(self, address: str) -> Book: ...
    def spot(self) -> dict[str, float]: ...

    def external_flow(self, address: str, since: datetime, until: datetime) -> float:
        """Net deposits minus withdrawals over the window, in USD.

        Sourced from `userNonFundingLedgerUpdates`. An implementation that
        cannot supply this must say so rather than return 0.0, because a
        silent zero turns a $50k deposit into a spectacular model failure in
        the calibration score.
        """
        ...


#: How late the resolver may run before the realisation stops being a
#: measurement of the predicted horizon (audit A-04). The Info API returns
#: current state only, so a late run cannot recover the state at
#: `resolves_at`; the honest response is to flag, not to score.
DEFAULT_STALE_AFTER_S = 2 * 3600.0

#: Ceiling on one resolve run's wall clock, and how long to wait when the §5.3
#: weight window is spent. Each ADDRESS costs ~40 weight (one book + one
#: external_flow, both shared across its three variants) against 300/min, so
#: §3.3's floor of 200 addresses -- 600 rows -- needs ~27 minutes of
#: mostly-waiting; 50 minutes leaves headroom while staying inside the hourly
#: cadence so runs do not overlap. Per ROW rather than per address it was 80
#: weight and the arithmetic did not hold; see `flow_cache` in `resolve_due`.
DEFAULT_MAX_RESOLVE_SECONDS = 50 * 60.0
DEFAULT_BUDGET_WAIT_SECONDS = 5.0

#: How stale the run's price snapshot may get before it is refetched. Matches
#: the provider's own freshness cache, so refreshing costs at most one extra
#: `spot()` per minute of a paced run.
SPOT_REFRESH_SECONDS = 60.0

#: How often the resolve run says where it is.
#:
#: `resolve_due` logged NOTHING, exactly as `ShadowCron.run_once` did before
#: it was given progress output -- and the fix was applied to the sweep alone,
#: which is how this half survived a day longer than the other. Same shape:
#: a 265-address run is roughly twenty minutes of which most is spent asleep
#: on the §5.3 window by design, so working and wedged are outwardly
#: identical, and the operator assumes the worse one.
#:
#: It matters more here than on the snapshot side. A silent sweep costs a
#: day's predictions; a silent resolver costs them *after* they were paid
#: for, because an outcome collected past `stale_after_s` is flagged and
#: dropped from the gate rather than scored.
PROGRESS_LOG_SECONDS = 30.0

log = logging.getLogger("risk_engine.shadow.resolve")


def _snapshot_captured_at(pending) -> datetime:
    """When the book behind `pending.start_equity` was actually observed.

    Falls back to `predicted_at` for rows written before the snapshot carried
    a timestamp, which is the old behaviour and no worse than it was.
    """
    raw = (pending.book_snapshot or {}).get("captured_at")
    if not raw:
        return pending.predicted_at
    try:
        stamp = datetime.fromisoformat(str(raw))
    except ValueError:
        return pending.predicted_at
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    # Never widen the window past what was asked for: a snapshot stamped
    # before `predicted_at` would pull in flow the prediction already saw.
    return max(stamp, pending.predicted_at)


@dataclass(frozen=True, slots=True)
class ResolveReport:
    resolved: int
    failed: list[tuple[int, str]]
    stale: int = 0
    #: The subset of `failed` whose failure is deterministic in the stored row
    #: and will therefore recur on every future run: (prediction id, why).
    #: A subset rather than a separate bucket on purpose -- these rows did
    #: fail this run, and a caller that iterates `failed` must still see them.
    permanent: list[tuple[int, str]] = field(default_factory=list)
    #: The run hit its wall-clock ceiling with rows still due. Not an error and
    #: not a per-row failure -- the remaining rows stay pending and the next
    #: run continues them. Reported so a run that stopped short is
    #: distinguishable from one that genuinely emptied the queue.
    budget_exhausted: bool = False

    def __str__(self) -> str:
        tail = f", {self.stale} flagged stale" if self.stale else ""
        forever = (
            f" ({len(self.permanent)} of them permanently -- retried every run "
            "until the journal row is corrected)"
            if self.permanent
            else ""
        )
        truncated = (
            " -- stopped at the time ceiling with rows still due, continues next run"
            if self.budget_exhausted
            else ""
        )
        return (
            f"resolved {self.resolved} predictions, "
            f"{len(self.failed)} failed{forever}{tail}{truncated}"
        )


def _pit_uniform(prediction_id: int) -> float:
    """Deterministic U(0,1) for the randomized PIT of one prediction.

    Derived from the prediction id rather than drawn freshly so that
    re-deriving a journal row reproduces the same number, and so that the
    draw cannot be retried until it flatters the model.
    """
    return float(np.random.default_rng(0xC0FFEE ^ prediction_id).random())


def _permanent_reason(address: str) -> str | None:
    """Why a pending row can never resolve, or None if it might.

    Only one condition is claimed, and it is claimed because it is decidable
    from the row alone: an `address` column that `normalise_address` refuses
    is refused identically on every future run, before any request is made,
    for as long as the row exists. Every provider in this repository looks a
    book up by that address -- `LiveSnapshotProvider.book` normalises it
    first, the fixture provider indexes a dict with it -- so no retry can
    succeed while the stored bytes stay as they are.

    Everything else is left transient on purpose. A timeout, a 5xx, a
    provider that has not implemented `external_flow` yet: those are exactly
    the failures the unbounded retry exists for, and guessing that one of
    them is permanent is how a resolvable observation gets abandoned.
    """
    try:
        normalise_address(address)
    except ValueError as exc:
        return f"stored address is not an account address ({exc})"
    return None


def resolve_due(
    journal: CalibrationJournal,
    provider: OutcomeProvider,
    now: datetime | None = None,
    stale_after_s: float = DEFAULT_STALE_AFTER_S,
    metrics: Metrics | None = None,
    max_resolve_seconds: float = DEFAULT_MAX_RESOLVE_SECONDS,
    budget_wait_seconds: float = DEFAULT_BUDGET_WAIT_SECONDS,
) -> ResolveReport:
    now = now or datetime.now(timezone.utc)
    metrics = metrics or METRICS
    resolved = 0
    stale = 0
    failed: list[tuple[int, str]] = []
    permanent: list[tuple[int, str]] = []

    # WAIT on the §5.3 weight, do not burst-and-drop. Each resolution spends
    # ~40 weight (book + external_flow) against 300/min, and this loop used to
    # process rows as fast as it could until `RateLimitExceeded`, then file
    # the rest as failed. Combined with a fresh hourly process (empty window
    # each run) that resolved ~8 rows before refusing, against ~200 coming due
    # per day inside a 2-hour staleness window: ~15 landed, ~185 aged out and
    # were dropped from the gate (A-04). §3.3 was unreachable from the
    # resolution side exactly as it was from the snapshot side, and for the
    # same reason -- a rate limit treated as a failure instead of a pace.
    # Waiting resolves all 200 in ~27 minutes, well inside the window.
    run_started = time.monotonic()
    deadline = run_started + max_resolve_seconds
    budget_exhausted = False
    seen = 0
    waited_s = 0.0
    last_progress = run_started

    def _say_progress(force: bool = False) -> None:
        nonlocal last_progress
        at = time.monotonic()
        if not force and at - last_progress < PROGRESS_LOG_SECONDS:
            return
        last_progress = at
        log.info(
            "resolve %d rows: %d resolved, %d stale, %d failed, %.0fs elapsed "
            "(%.0fs of it waiting for the §5.3 window)",
            seen, resolved, stale, len(failed) + len(permanent),
            at - run_started, waited_s,
        )

    def _observed_now() -> datetime:
        """`now` advanced by however long this run has actually been going.

        Derived from the run's own clock rather than read from
        `datetime.now()` so an injected `now` stays authoritative — the tests
        and any deterministic replay pass one, and reading the wall clock here
        would make every such run look hours late. In a real paced run the
        elapsed term is the whole point: it is the up-to-50-minute gap between
        the run starting and this particular row's outcome being fetched.
        """
        return now + timedelta(seconds=time.monotonic() - run_started)

    def _paced(call):
        """Run an Info call, waiting through the §5.3 window rather than
        letting a rate limit surface as a per-row failure. Charges happen
        before the request, so a refused call spends nothing and retries
        cleanly. Returns a sentinel when the whole-run ceiling is reached."""
        nonlocal budget_exhausted, waited_s
        while True:
            # Before the call, not only in the handler: a slow or failing
            # venue never trips the weight limit, so a deadline consulted only
            # on `RateLimitExceeded` bounds waiting rather than the run. See
            # the same correction in `cron._paced`.
            if time.monotonic() >= deadline:
                budget_exhausted = True
                return _CEILING
            try:
                return call()
            except RateLimitExceeded:
                if time.monotonic() >= deadline:
                    budget_exhausted = True
                    return _CEILING
                _say_progress()
                time.sleep(budget_wait_seconds)
                waited_s += budget_wait_seconds

    # One address may have several pending rows (model plus both baselines);
    # they share a realised outcome, so it is fetched once per address.
    cache: dict[str, tuple[float, str, datetime]] = {}
    # External flow is a function of (address, window) only -- it is
    # byte-identical across an address's three variants, and fetching it once
    # per ROW tripled the cost of a resolution: 80 weight per address against
    # the ~40 the capacity arithmetic below budgets with. At the deployed
    # ~500-address list that is 2.7 ceiling-runs to clear a day's queue, so
    # the third hourly run starts past the 2h staleness bound and ~25% of each
    # day's observations are flagged stale and dropped from §3.3 -- the same
    # gate-unreachable failure the pacing rework fixed on the snapshot side.
    flow_cache: dict[tuple[str, datetime, datetime], float] = {}

    # Paced like everything else -- the old top-of-function fetch ran before
    # the deadline existed, so a rate limit there escaped as an error (the
    # exact G-4 latency the cron side already closed).
    spot = _paced(provider.spot)
    if spot is _CEILING:
        return ResolveReport(resolved=0, failed=[], budget_exhausted=True)
    spot_at = time.monotonic()

    due = list(journal.due(now))
    log.info(
        "resolving %d pending rows; ceiling %.0f min, anything collected more "
        "than %.0f min late is flagged stale and dropped from the gate",
        len(due), max_resolve_seconds / 60.0, stale_after_s / 60.0,
    )
    for pending in due:
        if budget_exhausted:
            break
        seen += 1
        _say_progress()
        try:
            # Prices go stale inside a paced run. The run may legitimately
            # take up to `max_resolve_seconds`, and valuing a book fetched 50
            # minutes in at the prices from minute zero puts that drift
            # straight into `actual_equity` -- on a 10x book a 0.5% move is a
            # ~5% equity error, written into an immutable row. The provider
            # caches spot for 60s of its own, so this is at most one refresh a
            # minute.
            if time.monotonic() - spot_at >= SPOT_REFRESH_SECONDS:
                fresh = _paced(provider.spot)
                if fresh is _CEILING:
                    break
                spot, spot_at = fresh, time.monotonic()

            if pending.address not in cache:
                book = _paced(lambda p=pending: provider.book(p.address))
                if book is _CEILING:
                    break
                # The moment the realisation was actually observed, not the
                # moment the run started.
                cache[pending.address] = (
                    book.equity(spot), position_fingerprint(book), _observed_now(),
                )
            actual_equity, fingerprint, observed_at = cache[pending.address]

            # The flow window has to start where `start_equity` was measured,
            # not where the sweep began. Under pacing the sweep's book fetch
            # can trail its own `predicted_at` by up to 90 minutes, and a
            # deposit landing in that gap is inside `start_equity` AND inside
            # the flow -- subtracted twice, producing a fabricated loss in a
            # write-once table (reproduced: a $50k deposit scored as a $50k
            # loss, VaR breached, PIT 0). `captured_at` is what the snapshot
            # already recorded for exactly this.
            flow_from = _snapshot_captured_at(pending)
            flow_key = (pending.address, flow_from, pending.resolves_at)
            if flow_key not in flow_cache:
                got = _paced(lambda k=flow_key: provider.external_flow(k[0], k[1], k[2]))
                if got is _CEILING:
                    break
                flow_cache[flow_key] = got
            flow = flow_cache[flow_key]
            # The model predicts the change due to market moves and funding.
            # Removing the external flow is what makes the comparison fair;
            # the raw flow is stored too, so the filtered and unfiltered
            # cohorts can both be scored.
            change = actual_equity - pending.start_equity - flow

            # Measured from when the outcome was OBSERVED, not from when the
            # run started. Pacing lets those differ by up to
            # `max_resolve_seconds`, so a realisation captured 2h40m after
            # `resolves_at` used to record a lag of 1h50m, pass the 2h
            # staleness test and be scored into the gate -- precisely the A-04
            # corruption `stale_after_s` exists to prevent, and worst for the
            # rows that sit closest to the boundary.
            lag = (observed_at - pending.resolves_at).total_seconds()
            is_stale = lag > stale_after_s
            if is_stale:
                stale += 1
                metrics.incr("shadow_stale_resolutions")
            u = _pit_uniform(pending.id)
            journal.record_outcome(
                prediction_id=pending.id,
                resolved_at=observed_at,
                actual_equity=actual_equity,
                actual_equity_change=change,
                external_flow_usd=flow,
                book_changed=fingerprint != pending.book_snapshot.get("fingerprint"),
                liquidated=actual_equity <= 0.0,
                pit=pending.distribution.pit(change, u),
                pit_u=u,
                crps=pending.distribution.crps(change),
                var_95_breached=change < -pending.var_95,
                observation_day=pending.predicted_at.astimezone(timezone.utc).date(),
                resolution_lag_s=lag,
                stale_resolution=is_stale,
            )
            resolved += 1
        except Exception as exc:  # one bad row must not stop the batch
            failed.append((pending.id, f"{type(exc).__name__}: {exc}"))
            # Classified from the row, not from the exception that surfaced
            # this time: the reason such a row is permanent is that the stored
            # address is unusable, which holds whichever call happens to raise
            # first. Recorded so the report can say "this one is forever"
            # instead of leaving a never-emptying failed list looking transient
            # (see the module docstring on why the retry is unbounded).
            reason = _permanent_reason(pending.address)
            if reason is not None:
                permanent.append((pending.id, reason))

    _say_progress(force=True)
    return ResolveReport(
        resolved=resolved, failed=failed, stale=stale, permanent=permanent,
        budget_exhausted=budget_exhausted,
    )
