"""Shadow prediction cron (§3.3).

Runs daily from the end of Phase 1. For each tracked address it writes three
predictions of the 24 h equity-change distribution -- the model and both
baselines -- and the resolver fills in the realised value a day later. While
Phases 2 and 3 are being built read-only, the validation window accumulates
on its own.

Why the distribution and not just P(liq): a liquidation is a rare binary
event, and a realistic sample will never have the statistical power to fill
calibration bins for it. The equity change, by contrast, is known for every
address after 24 h no matter what happened, so every observation carries
information. P(liq) is then checked as a consistent by-product of the same
distribution rather than as a separate rare-event test.

Rate limits (§5.3): the `WeightBudget` that enforces the reserve lives on the
`InfoClient` inside the provider, and every API call charges it before the
request. The sweep does not keep a budget of its own -- it waits on the call
that spends the real one (`book()`), so a minute's worth of weight paces the
sweep rather than truncating it. Only when the whole-sweep ceiling is reached
is the run reported as truncated.

Two failure channels, and the difference between them is load-bearing:

  - `SweepReport.skipped` is per address. A flat book, a non-positive equity,
    a request that failed for one account: the sweep continues, because the
    other 199 addresses are still a day of the §3.3 window.
  - `SweepReport.address_source_error` is the whole run. The address list is
    the sampling frame the calibration score will be published against
    (OPEN-QUESTIONS B4), so a list containing an entry that is not an address
    is not swept as if it were that frame -- but the refusal has to arrive as
    a *reported* run-level failure naming the offending index, which the CLI
    turns into a non-zero exit. It escaped as an unhandled ValueError before,
    and because an address file does not change between days, that took the
    whole 21-day window with it rather than one address.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

import numpy as np

from risk_engine.domain.types import AssetSpec, Book, MarginMode
from risk_engine.market.info import (
    SHADOW_RESERVED_FRACTION as _SHADOW_RESERVED_FRACTION,
    RateLimitExceeded,
)
from risk_engine.shadow.journal import (
    VARIANT_BASELINE_A,
    VARIANT_BASELINE_B,
    VARIANT_MODEL,
    CalibrationJournal,
)
from risk_engine.sim.engine import ModelBundle, MonteCarloEngine
from risk_engine.sim.stats import wilson_interval
from risk_engine.validation.baselines import NaiveBaseline, run_baseline_b
from risk_engine.version import DISTRIBUTION_VERSION

#: §5.3: keep three quarters of the weight budget for interactive traffic.
#: One home in `market.info`; this was a second hand-written 0.75.
SHADOW_RESERVED_FRACTION = _SHADOW_RESERVED_FRACTION


class SnapshotProvider(Protocol):
    """What the cron needs from the outside world.

    A Protocol rather than a concrete client so the harness runs against
    fixtures in tests and against the Info API in production, with no branch
    inside the job itself.
    """

    def addresses(self) -> list[str]: ...
    def book(self, address: str) -> Book: ...
    def spot(self) -> dict[str, float]: ...
    def specs(self) -> dict[str, AssetSpec]: ...


@dataclass(frozen=True, slots=True)
class SweepReport:
    started_at: datetime
    attempted: int
    written: int
    skipped: list[tuple[str, str]] = field(default_factory=list)
    budget_exhausted: bool = False
    #: Set when the address list itself could not be loaded, in which case the
    #: sweep wrote nothing and `attempted` is 0. Carries the source's own
    #: message, which names the offending index. Distinct from `skipped`
    #: because it is not one address that went wrong, it is the day.
    address_source_error: str | None = None

    @property
    def refused(self) -> bool:
        """Whether the run refused to sweep at all.

        A caller that reports success on `written == 0` cannot tell an empty
        list from a rejected one, and the second is the case that repeats
        every day until an operator edits the file.
        """
        return self.address_source_error is not None

    def __str__(self) -> str:
        if self.refused:
            return (
                f"shadow sweep {self.started_at.isoformat()}: REFUSED, nothing "
                f"written -- the address list would not load: "
                f"{self.address_source_error}"
            )
        tail = " (budget exhausted, sweep truncated)" if self.budget_exhausted else ""
        return (
            f"shadow sweep {self.started_at.isoformat()}: {self.written}/{self.attempted} "
            f"addresses written, {len(self.skipped)} skipped{tail}"
        )


#: Ceiling on one sweep's wall clock. 515 addresses at 20 weight against
#: §5.3's 300/minute is ~34 minutes of mostly waiting, so 90 minutes leaves
#: room for a list that grew without letting a pathological one pin the
#: container in a sleep loop until the next daily run collides with it.
MAX_SWEEP_SECONDS = 90.0 * 60.0

#: How long to sleep when the minute's weight is spent. The window is a
#: sliding 60 s, so a few seconds is enough to see it refill; shorter just
#: burns CPU re-asking.
BUDGET_WAIT_SECONDS = 5.0


def _bundle_understates_lower_tail(bundle) -> bool:
    """Is this bundle failing §2.3's tail gate (A10/A11)?

    Delegates to `service.state.understates_lower_tail`, whose docstring
    already promised exactly this use: "one predicate, so the serving path and
    the provenance stamp cannot drift apart in what they call a defect". There
    was no provenance stamp until now, so the promise had one caller.

    Imported inside the function: `service` builds bundles and `shadow`
    consumes them, and a module-level import here would make the shadow jobs
    depend on the serving package at import time for one boolean.

    A bundle assembled by hand (tests, benchmarks) carries no diagnostics at
    all, and that reads as "no defect" -- correct, because there is no fitted
    copula to have failed the gate.
    """
    from risk_engine.service.state import understates_lower_tail

    diagnostics = getattr(bundle, "tail_diagnostics", ())
    return bool(diagnostics) and understates_lower_tail(diagnostics)


#: How often the sweep says where it is.
#:
#: `run_once` used to log NOTHING. Between the bundle build and the final
#: report -- tens of minutes on a real list, up to `MAX_SWEEP_SECONDS` -- the
#: container emitted not one line, so an operator watching a daily job could
#: not tell a working sweep from a hung one. That is not a hypothetical: it
#: happened twice on the same deployment, and the second time the sweep was
#: healthy and the silence was the whole problem.
#:
#: The design makes it worse than an ordinary missing log line. This sweep is
#: SUPPOSED to spend most of its wall clock asleep -- pacing against §5.3's
#: window is the yield the reserve asks for -- so its normal working state and
#: a deadlock are outwardly identical, and the one the operator fears is the
#: one they will assume.
PROGRESS_LOG_SECONDS = 30.0

log = logging.getLogger("risk_engine.shadow.cron")


class ShadowCron:
    def __init__(
        self,
        provider: SnapshotProvider,
        bundle: ModelBundle,
        journal: CalibrationJournal,
        naive: NaiveBaseline,
        n_paths: int = 20_000,
        horizon_hours: int = 24,
        max_sweep_seconds: float = MAX_SWEEP_SECONDS,
        budget_wait_seconds: float = BUDGET_WAIT_SECONDS,
    ) -> None:
        self.provider = provider
        self.bundle = bundle
        self.journal = journal
        self.naive = naive
        self.n_paths = n_paths
        self.horizon_hours = horizon_hours
        self.max_sweep_seconds = max_sweep_seconds
        self.budget_wait_seconds = budget_wait_seconds
        # Whether the bundle this sweep predicts from is failing §2.3's tail
        # gate (A10/A11). Stamped on every row it writes, so `progress()` can
        # tell a diagnostic day from a gate-day weeks later.
        #
        # Read once here rather than per row: the bundle is fixed for the
        # lifetime of the sweep, and re-deriving it 800 times would invite the
        # two to disagree within one run.
        #
        # The sweep already ANNOUNCED this (`_print_defect_note`), and that was
        # the whole defence: a log line, in a container's scrollback, against a
        # gate read three weeks later from the database. Its own docstring says
        # a journal of observations collected under a known model defect,
        # indistinguishable from a clean one, "is worse than no journal -- it
        # would be read as gate progress", and nothing in the schema could tell
        # them apart.
        self.recorded_under_defect = _bundle_understates_lower_tail(bundle)
        # No budget of its own, deliberately. §5.3's weight is owned by the
        # InfoClient inside the provider, and the sweep waits on the call that
        # spends it (see run_once). A budget object here previously looked
        # like it gated the sweep and did not -- the real limit was on a
        # different object entirely -- so it is gone rather than dormant.

    def run_once(self, now: datetime | None = None, seed: int | None = None) -> SweepReport:
        now = now or datetime.now(timezone.utc)

        # The address list is read FIRST, and inside a guard. Both parts are
        # fixes for measured behaviour, and they are separate points.
        #
        # First: it goes inside a guard because `addresses()` raises on an
        # entry that is not an address (providers.py, deliberately -- the old
        # path put a typo on the wire, and §5.1's answer for an address the
        # venue does not recognise is a well-formed *empty* state, which the
        # resolver then scored as equity 0: liquidated=1, var_95_breached=1,
        # permanently, in a table that is never updated). Outside every `try`, that
        # refusal left the sweep as an unhandled ValueError out of
        # `cmd_snapshot`: with [good, '0xabc', good] the pre-refusal code
        # wrote 2 of 3 and skipped 1, and the refusal wrote 0 and printed a
        # traceback. An address file is static, so that is not one bad day, it
        # is all 21 of §3.3's -- the blast radius the refusal was supposed to
        # shrink. The list is still refused whole (sweeping the good half
        # would quietly publish a different sampling frame from the one the
        # file declares, OPEN-QUESTIONS B4), but as a run-level report that
        # names the index, which `cmd_snapshot` exits non-zero on.
        #
        # `Exception` rather than `ValueError`: a missing file (OSError), a
        # truncated one (JSONDecodeError) and a typo (ValueError) all mean the
        # same thing to the operator -- there is no list to sweep today -- and
        # each one used to produce a differently-shaped traceback.
        # KeyboardInterrupt and SystemExit are BaseException and still escape,
        # so an interrupted sweep is not reported as a bad list.
        #
        # Second: it goes first because `specs()` and `spot()` are Info
        # requests. Validating after them charged 40 weight (meta plus one
        # candleSnapshot on a 1-coin universe; more on a real universe) before
        # the list was so much as opened, which made two written claims false
        # -- providers.py's "a typo caught here costs nothing" and the
        # README's "rather than part-way through a sweep that has already
        # spent weight". Reading a local file before spending §5.3 budget
        # makes them true instead of having to soften them.
        try:
            addresses = self.provider.addresses()
        except Exception as exc:
            return SweepReport(
                started_at=now,
                attempted=0,
                written=0,
                address_source_error=f"{type(exc).__name__}: {exc}",
            )

        written = 0
        skipped: list[tuple[str, str]] = []
        exhausted = False
        attempted = 0
        waited_s = 0.0
        started_monotonic = time.monotonic()
        last_progress = started_monotonic

        def _say_progress(force: bool = False) -> None:
            """One line saying the sweep is alive and where it is.

            Reports the waiting time separately from the elapsed time, because
            those two numbers are what distinguish the failure modes: mostly
            waiting is §5.3 working as designed, while elapsed climbing with
            neither addresses nor waiting moving is a stall worth acting on.
            """
            nonlocal last_progress
            at = time.monotonic()
            if not force and at - last_progress < PROGRESS_LOG_SECONDS:
                return
            last_progress = at
            log.info(
                "sweep %d/%d addresses: %d written, %d skipped, %.0fs elapsed "
                "(%.0fs of it waiting for the §5.3 window)",
                attempted, len(addresses), written, len(skipped),
                at - started_monotonic, waited_s,
            )

        log.info(
            "sweeping %d addresses; ceiling %.0f min, %d paths per prediction",
            len(addresses), self.max_sweep_seconds / 60.0, self.n_paths,
        )

        # The deadline is set BEFORE the first weight-spending call, and
        # `specs()`/`spot()` are paced too. They charge ~80 weight (meta plus
        # a candle per coin) and used to run unguarded: on a fresh daily
        # process the window is empty so they succeed, but a `RateLimitExceeded`
        # from them would have escaped `run_once` entirely rather than pacing.
        # That is the same rate-limit-as-error footgun the loop below fixes, so
        # it is closed here rather than left latent for the day the process is
        # made long-lived.
        deadline = time.monotonic() + self.max_sweep_seconds

        def _paced(call):
            nonlocal exhausted, waited_s
            while True:
                # Checked BEFORE the call, not only in the rate-limit handler.
                # The ceiling used to bound waiting rather than the run: a slow
                # or failing venue never trips the weight limit (retries with
                # 10s timeouts run at ~40 weight/min against a 300 allowance),
                # so the deadline was never read and a sweep over 500 addresses
                # could run for hours -- past its own daily cadence, with both
                # docstrings claiming a ceiling that did not exist.
                if time.monotonic() >= deadline:
                    exhausted = True
                    return None
                try:
                    return call()
                except RateLimitExceeded:
                    if time.monotonic() >= deadline:
                        exhausted = True
                        return None
                    _say_progress()
                    waited_s += self.budget_wait_seconds
                    time.sleep(self.budget_wait_seconds)

        specs = _paced(self.provider.specs)
        spot = _paced(self.provider.spot)
        if exhausted:
            # The window was already spent before the sweep proper began. No
            # addresses were attempted, so this is truncation, not a bad list.
            return SweepReport(started_at=now, attempted=len(addresses),
                               written=0, budget_exhausted=True)
        rng = np.random.default_rng(seed if seed is not None else int(now.timestamp()))

        for address in addresses:
            # WAIT on the call that actually spends the budget, and do not
            # abandon the sweep on it. Two corrections, and the second was a
            # bug in the first version of this fix:
            #
            # (a) Break -> wait. Truncating on the budget made §3.3's gate
            #     unreachable: a sliding minute at 25% of 1200 (§5.3) buys 15
            #     addresses, the loop stopped, and a daily job delivered 15
            #     against a per-day requirement of 200. The first live run
            #     wrote 6 of 515. Sleeping until the window refills IS the
            #     yield §5.3 asks for -- it just also finishes the work.
            #
            # (b) Wait on the RIGHT budget. The first version charged a budget
            #     the cron OWNED, while the real request charges the
            #     InfoClient's budget -- two different objects (cli.py built
            #     them separately and never wired them together). So the wait
            #     was on a phantom counter: the real §5.3 limit was hit inside
            #     `book()`, surfaced as a `RateLimitExceeded` the outer handler
            #     filed as a permanent skip, and the address was dropped rather
            #     than retried. `book()` POSTs `clearinghouseState` and the
            #     client charges the budget BEFORE the request, so a refused
            #     call spends nothing and is safe to retry -- which `_paced`
            #     does, the same wrapper `specs()`/`spot()` above use.
            attempted += 1
            _say_progress()
            try:
                book = _paced(lambda a=address: self.provider.book(a))
                if exhausted:
                    # Ceiling reached mid-wait: a pathological list cannot pin
                    # the container in a sleep loop until the next daily run.
                    log.warning(
                        "sweep hit its %.0f-minute ceiling at address %d/%d; "
                        "reporting truncation (§5.3)",
                        self.max_sweep_seconds / 60.0, attempted, len(addresses),
                    )
                    break
                if not book.positions:
                    skipped.append((address, "no open positions"))
                    continue
                # Named BEFORE `equity()` touches `spot`, and naming ALL the
                # missing coins rather than whichever one a dict lookup hit
                # first (OPEN-QUESTIONS B6).
                #
                # This used to surface as a bare `KeyError: 'ATOM'` from
                # inside `equity()`, caught by the handler below. That is a
                # skip reason expressed as an implementation detail, and the
                # census stores reasons verbatim — so B6's own decision
                # procedure ("sum the KeyError counts by coin") was unsound.
                # An address holding ATOM and HYPE was filed under whichever
                # came first, so per-coin totals answer "how many addresses
                # mention this coin first", never "how many addresses would a
                # universe containing it recover". Adding the top coin by that
                # tally can recover nothing at all, if every address holding
                # it also holds a second off-universe coin.
                #
                # Listing every missing coin makes the census answer the
                # question actually being asked: an address is recovered by
                # universe U exactly when this whole list is inside U.
                off_universe = sorted({p.coin for p in book.positions} - set(spot))
                if off_universe:
                    skipped.append(
                        (address, f"off-universe: {', '.join(off_universe)}")
                    )
                    continue
                if book.equity(spot) <= 0:
                    skipped.append((address, "non-positive equity"))
                    continue
                self._predict_all(address, book, spot, specs, now, int(rng.integers(2**62)))
                written += 1
            except Exception as exc:  # one bad address must not stop the sweep
                skipped.append((address, f"{type(exc).__name__}: {exc}"))

        _say_progress(force=True)
        return SweepReport(
            started_at=now,
            attempted=len(addresses),
            written=written,
            skipped=skipped,
            budget_exhausted=exhausted,
        )

    def _predict_all(
        self,
        address: str,
        book: Book,
        spot: dict[str, float],
        specs: dict[str, AssetSpec],
        now: datetime,
        seed: int,
    ) -> None:
        snapshot = _book_snapshot(book, spot)

        model = MonteCarloEngine(self.bundle, specs).run(
            book, spot, self.horizon_hours, n_paths=self.n_paths, seed=seed, now=now
        )
        self._write(address, VARIANT_MODEL, now, snapshot, model.start_equity,
                    model.p_liq_any.point, (model.p_liq_any.ci_low, model.p_liq_any.ci_high),
                    model.var_95_usd, model.cvar_95_usd.point, model.equity_change,
                    seed, model.n_paths, model.converged)

        b = run_baseline_b(self.bundle, specs, book, spot, self.horizon_hours,
                           n_paths=self.n_paths, seed=seed, now=now)
        self._write(address, VARIANT_BASELINE_B, now, snapshot, b.start_equity,
                    b.p_liq_any.point, (b.p_liq_any.ci_low, b.p_liq_any.ci_high),
                    b.var_95_usd, b.cvar_95_usd.point, b.equity_change,
                    seed, b.n_paths, b.converged)

        a = self.naive.predict(book, spot, specs, n_draws=self.n_paths, seed=seed, now=now)
        self._write(address, VARIANT_BASELINE_A, now, snapshot, a.start_equity,
                    a.p_liq, a.p_liq_ci, a.equity_change.var(0.95), a.equity_change.cvar(0.95),
                    a.equity_change, seed, a.n_draws, True)

    def _write(self, address, variant, now, snapshot, start_equity, p_liq, ci,
               var_95, cvar_95, distribution, seed, n_paths, converged) -> None:
        self.journal.record_prediction(
            address=address,
            variant=variant,
            predicted_at=now,
            horizon_hours=self.horizon_hours,
            model_version=self.bundle.model_version,
            distribution_version=DISTRIBUTION_VERSION,
            seed=seed,
            n_paths=n_paths,
            converged=converged,
            start_equity=start_equity,
            p_liq=p_liq,
            p_liq_ci=ci,
            var_95=var_95,
            cvar_95=cvar_95,
            distribution=distribution,
            book_snapshot=snapshot,
            recorded_under_defect=self.recorded_under_defect,
        )


def _book_snapshot(book: Book, spot: dict[str, float]) -> dict:
    """What the book looked like when the prediction was made.

    Stored verbatim so a later reader can tell whether a bad score came from
    the model or from the account being restructured mid-horizon (§3.3, and
    the cohort split in OPEN-QUESTIONS B2). The position fingerprint is what
    the resolver compares against.
    """
    return {
        "cross_collateral": book.cross_collateral,
        "captured_at": book.captured_at.isoformat(),
        "spot": {c: spot[c] for c in book.coins if c in spot},
        "positions": [
            {
                "coin": p.coin,
                "size": p.size,
                "entry_price": p.entry_price,
                "mode": p.mode.value,
                "leverage": p.leverage,
                "isolated_margin": p.isolated_margin,
            }
            for p in book.positions
        ],
        "fingerprint": position_fingerprint(book),
    }


def position_fingerprint(book: Book) -> str:
    """Stable identity of a book's positions, ignoring price movement.

    Identity, not merely shape. This decides `book_changed`, and B2 says the
    gate is read off the book-UNCHANGED cohort — so anything material that
    this misses puts a row into that cohort whose realisation came from a
    different book than the one predicted.

    It used to cover coin, size and mode alone, which left two material
    changes invisible:

    - **isolated margin.** Adding or removing collateral from a pocket leaves
      total equity untouched (it moves between cross and the pocket), so the
      SCORED quantity does not move — but the pocket's liquidation distance
      does, which is most of what the prediction was about.
    - **isolated leverage.** A7 measured set leverage moving an isolated
      pocket's P(liq) from 0.02167 to 0.62915 over its own grid. A 29x change
      in the predicted quantity, recorded as "book unchanged".

    Cross leverage is deliberately NOT included, on the same measurement:
    §1.2 makes it immaterial for a cross position, and A7 pinned the
    invariance end to end through the Monte Carlo (0.161125 at 5x, 10x and
    25x, to six decimals). Including it would flag books that did not
    materially change, which costs cohort size for nothing.

    Format note: the string changed on 2026-08-04. A row written under the
    old format resolves as `book_changed=True` against the new one, which is
    the conservative direction — such rows drop OUT of the strict cohort
    rather than into it — and the blast radius today is nil because the §3.3
    counter has not started.
    """
    parts = sorted(
        f"{p.coin}:{p.size:.10g}:{p.mode.value}"
        + (f":m{p.isolated_margin:.10g}:l{p.leverage:.10g}"
           if p.mode is MarginMode.ISOLATED else "")
        for p in book.positions
    )
    return "|".join(parts)


def wilson(successes: int, n: int) -> tuple[float, float]:
    return wilson_interval(successes, n)
