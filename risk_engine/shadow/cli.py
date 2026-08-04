"""Entry points for the shadow harness (§3.3).

    python -m risk_engine.shadow snapshot --journal shadow.db --addresses addrs.json
    python -m risk_engine.shadow resolve  --journal shadow.db
    python -m risk_engine.shadow progress --journal shadow.db
    python -m risk_engine.shadow icc      --journal shadow.db

Two jobs on a daily cadence: `snapshot` writes predictions, `resolve` fills
in what actually happened a day later. Between them they accumulate the
window Phase 4 is gated on.

Both refuse to guess. `--fixture` runs against synthetic data and says so;
the live path needs an address source with a stated sampling frame
(OPEN-QUESTIONS B4) and an `external_flow` implementation, and will fail
loudly rather than quietly scoring deposits as model error.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from risk_engine.market.info import (
    SHADOW_RESERVED_FRACTION as _SHADOW_RESERVED_FRACTION,
)
from risk_engine.market.info import RateLimitExceeded
from risk_engine.shadow.cron import ShadowCron
from risk_engine.shadow.journal import CalibrationJournal
from risk_engine.shadow.metrics import COHORT_BOOK_UNCHANGED, COHORTS, calibration_report
from risk_engine.shadow.providers import FileAddressSource, LiveSnapshotProvider
from risk_engine.shadow.resolve import DEFAULT_STALE_AFTER_S, resolve_due
from risk_engine.shadow.weight_ledger import open_weight_budget
from risk_engine.validation.baselines import NaiveBaseline, historical_24h_log_returns
from risk_engine.version import DISTRIBUTION_VERSION, MODEL_VERSION

log = logging.getLogger("risk_engine.shadow")

#: §5.3: the shadow jobs keep three quarters of the weight budget for live
#: users. Re-exported from `market.info` rather than restated -- it was
#: written out here AND in `cron.py`, two copies of a number whose whole job
#: is to be the same everywhere.
SHADOW_RESERVED_FRACTION = _SHADOW_RESERVED_FRACTION


def _fixture_world():
    """Synthetic bundle plus a handful of synthetic books.

    Exists so the whole harness -- cron, resolver, journal, metrics -- is
    runnable end to end without a venue. It is not a stand-in for validation:
    predictions about a market we invented say nothing about the model's fit
    to the real one, and every row it writes is stamped with a frame that
    says so.
    """
    from datetime import timedelta

    import numpy as np

    from risk_engine.domain.types import Book, MarginMode, Position
    from risk_engine.service.state import _build_fixture_bundle

    bundle, specs, spot = _build_fixture_bundle()
    rng = np.random.default_rng(11)
    now = datetime.now(timezone.utc)
    books: dict[str, Book] = {}
    coins = [c for c in ("BTC", "ETH", "SOL", "AVAX") if c in spot]
    for i in range(12):
        equity = float(rng.uniform(25_000, 500_000))
        held = rng.choice(coins, size=int(rng.integers(2, min(5, len(coins)) + 1)), replace=False)
        leverage = float(rng.uniform(3, 18))
        positions = []
        for coin in held:
            side = 1.0 if rng.random() < 0.7 else -1.0
            notional = equity * leverage / len(held)
            positions.append(
                Position(coin, side * notional / spot[coin], spot[coin],
                         MarginMode.CROSS, 20.0)
            )
        # A well-formed address, because the journal now canonicalises what it
        # stores and would refuse a readable stub like "0xfixture000". That is
        # the right way round: a fixture run that skipped the address
        # discipline of the live path would not be exercising the live path.
        # "facade" is legal hex and says what these are.
        address = f"0xfacade{i:034x}"  # 6 + 34 = the required 40 hex digits
        books[address] = Book(
            address, equity, tuple(positions), now - timedelta(seconds=5)
        )

    class FixtureProvider:
        frame = "SYNTHETIC fixture books over a simulated market; not a sample of anything real"

        def addresses(self) -> list[str]:
            return list(books)

        def book(self, address: str) -> Book:
            return books[address]

        def spot(self) -> dict[str, float]:
            return spot

        def specs(self):
            return specs

        def external_flow(self, address, since, until) -> float:
            return 0.0

    factor = historical_24h_log_returns(
        np.diff(np.log(np.cumprod(1 + rng.normal(0, 0.008, 4000)) * 100.0))
    )
    return FixtureProvider(), bundle, NaiveBaseline(factor)


#: How long the bundle build may wait for the §5.3 window, and how long to
#: sleep between attempts. Generous because the alternative is losing the run:
#: a shared window refills within 60s, so anything past a few minutes means the
#: pool is genuinely oversubscribed rather than momentarily busy.
BUNDLE_BUDGET_WAIT_S = 5.0

#: Raised from 10 minutes on 2026-08-03, after the ceiling was reached for
#: real on the first restart under the shared pool: the resolver was building
#: its own bundle and working through pending rows, the snapshot waited out
#: its ten minutes and exited, and that day's predictions were never written.
#:
#: Ten was inconsistent with the sweep's own 90-minute ceiling -- the run was
#: allowed an hour and a half of WORK but ten minutes to START -- and the
#: costs are wildly asymmetric. Waiting longer costs wall clock on a job that
#: runs once a day and is already expected to spend most of its time asleep.
#: Giving up costs one of §3.3's 21 days, and it cannot be made up later: the
#: prediction had to be made against that day's book.
#:
#: This is defence in depth, not the fix. Scheduling was the fix -- see
#: `RESOLVE_START_DELAY_S` in deploy/docker-compose.yml, which stops the two
#: jobs starting in the same second every day. The ceiling covers whatever
#: else takes the pool.
BUNDLE_MAX_WAIT_S = 30 * 60.0


def _paced_bundle(budget, max_wait_s: float = BUNDLE_MAX_WAIT_S,
                  wait_s: float = BUNDLE_BUDGET_WAIT_S):
    """Build the live bundle, WAITING on the §5.3 window rather than dying on it.

    The bundle costs ~160 weight (meta, plus a candle snapshot and a funding
    history per coin) and it is spent before the sweep proper starts. While
    each process had a private budget that was always affordable at startup,
    because a fresh process began with a full window. Sharing the pool (C6)
    removed that guarantee: a snapshot starting while the resolver holds the
    window now meets a spent one, and an unpaced build turns that into a dead
    run -- `RateLimitExceeded` propagating out of `meta()` and killing the
    process before a single prediction is written.

    That is the same burst-and-drop failure the sweep and the resolver were
    both fixed for (a rate limit is a PACE, not an error), reintroduced one
    layer up by the change that made the pool shared. Waiting is what §5.3
    asks for; the ceiling only exists so a genuinely oversubscribed pool
    surfaces as a loud failure instead of a container asleep forever.
    """
    import time as _t

    from risk_engine.service.state import _build_live_bundle

    deadline = _t.monotonic() + max_wait_s
    waited = False
    while True:
        try:
            bundle = _build_live_bundle(serving=False, budget=budget)
            if waited:
                log.info("§5.3 window refilled; bundle built")
            return bundle
        except RateLimitExceeded:
            if _t.monotonic() >= deadline:
                raise SystemExit(
                    f"the shared §5.3 weight window stayed full for "
                    f"{max_wait_s / 60:.0f} minutes, so the bundle could not be "
                    "built. Another shadow job is holding the pool: check "
                    "whether a snapshot and a resolve are running at once, or "
                    "whether a one-off run is competing with the scheduled "
                    "container (`docker compose ps`)."
                ) from None
            if not waited:
                log.info("§5.3 window is spent; waiting for it to refill")
                waited = True
            _t.sleep(wait_s)


def _live_world(args, *, load_addresses: bool = True):
    import time as _time


    # Demanded only when it is actually READ. `cmd_resolve` passes
    # load_addresses=False and takes its addresses from the journal's own
    # pending rows -- refusing to start it for want of a sampling frame it
    # never opens destroys observations already paid for, which is the
    # opposite of what this guard is for. The module docstring prints
    # `resolve --journal shadow.db`; that command exited 1 until now.
    if load_addresses and not args.addresses:
        raise SystemExit(
            "--addresses is required for a live run: the address list is a sampling "
            "frame and it has to be chosen deliberately (OPEN-QUESTIONS B4)"
        )
    source = FileAddressSource(Path(args.addresses)) if args.addresses else None
    if load_addresses:
        # Loaded and validated HERE: before `_build_live_bundle` and before
        # the candle fetch at the bottom of this function, which are Info
        # requests charged against §5.3's budget. Built the other way round --
        # bundle first, list opened only once the sweep reached it -- a
        # one-character typo cost 40 weight on a 1-coin universe (meta plus
        # candleSnapshot) and around 80 of the 300/minute the shadow reserve
        # allows on the real path, which is what made two written claims
        # false: providers.py's "a typo caught here costs nothing" and the
        # README's "rather than part-way through a sweep that has already
        # spent weight". This ordering is what makes them true.
        #
        # The list is read twice as a result -- once here, once by the sweep.
        # That is two reads of a local file and no requests, and it is the
        # safer arrangement anyway: a file edited between the two is caught by
        # `run_once`'s own guard rather than trusted because startup liked it.
        try:
            source.addresses()
        except (OSError, ValueError) as exc:
            # ValueError covers the malformed entry (with its index) and
            # JSONDecodeError, which subclasses it; OSError covers a path that
            # is not there. SystemExit rather than a traceback because this is
            # an operator's fixable mistake, and it is the same channel the
            # missing-`--addresses` refusal above already uses.
            raise SystemExit(
                f"address list refused, and nothing was fetched: {exc}"
            ) from exc

    # `serving=False`: this path shows no number to anyone, it records
    # observations. §2.3's criterion is still run and still logged, but it does
    # not refuse the build here -- the shadow harness is the instrument that
    # measures whether an unvalidated model is any good, and refusing to
    # measure a model because it is unvalidated is circular. The refusal stays
    # where a person would see the number: `EngineState`, via the default.
    #
    # An adversarial review of the skewed-t remedy found that the SIGN of the
    # error depends on the shape of the book (`_any_liq` is a union and
    # `Position.size` is signed), so the magnitude and direction on real books
    # are unknown. This window is what establishes them. The days it records
    # do NOT count toward §3.3's gate -- the remedy bumps MODEL_VERSION and
    # resets the counter -- and `_defect_note` below says so on every run.
    # Shared across the snapshot and resolve containers when they share a
    # Postgres journal (C6). Two processes each holding a private 300/min
    # window made §5.3's reserve 50% during their overlap, and neither could
    # see the other spending. Falls back to the in-process window on sqlite,
    # which is the single-machine development case where they are the same
    # thing. Built BEFORE the bundle so the bundle's own ~160 weight of meta,
    # candle and funding fetches is charged to this pool too -- the resolver
    # rebuilds hourly, and that spend used to land on a budget nobody shared.
    budget = open_weight_budget(
        getattr(args, "journal", None),
        reserved_fraction=SHADOW_RESERVED_FRACTION,
        actor=getattr(args, "command", "shadow"),
    )
    bundle, specs, spot = _paced_bundle(budget)
    provider = LiveSnapshotProvider(source, budget=budget, universe=tuple(spot))
    # Reuse the freshly-built bundle's view of the venue rather than
    # re-fetching it per sweep. `_spot_at` has to be stamped too: it is the
    # monotonic clock reading `spot()` measures its 60-second freshness
    # window against, and leaving it at 0.0 meant the very first call in the
    # sweep saw an age of "seconds since the process booted", judged the
    # seeded prices stale, and re-fetched a candle snapshot per coin -- 20
    # weight each, for prices that were seconds old. The cache this comment
    # claims to be reusing was never once hit.
    provider._specs = specs
    provider._spot = dict(spot)
    provider._spot_at = _time.monotonic()

    import numpy as np

    # Baseline A's factor series comes off the bundle, NOT from a fresh
    # fetch. This used to re-request 90 days of BTC candles here — the same
    # 90 days `_build_live_bundle` had just fetched to fit the matrix — and
    # it did so UNPACED, on the line after `_paced_bundle` had waited out and
    # then drained the §5.3 window. Measured cost: ~56 weight (20 base plus
    # the per-item surcharge on 2160 candles) for data already in memory,
    # and a live run that died with `weight 20 exceeds remaining 0
    # (323/300 spent)` immediately after successfully waiting its turn.
    #
    # That was the fifth appearance of "a rate limit is an error, not a
    # pace" in this codebase. Pacing it would have fixed the crash and kept
    # the waste; not making the call fixes both, and leaves nothing to pace.
    hourly = bundle.factor_returns.get("BTC")
    if hourly is None:
        raise SystemExit(
            "the bundle carries no BTC return series, so Baseline A (§3.2) "
            "cannot be built. BTC is mandatory as a risk factor (§2.1) and "
            "`_build_live_bundle` refuses without it, so reaching this means "
            "the bundle came from somewhere else."
        )
    return provider, bundle, NaiveBaseline(
        historical_24h_log_returns(np.asarray(hourly))
    )


def _print_defect_note(bundle) -> None:
    """Say, on every run, what this window is recording under.

    The shadow path builds with `serving=False`, so a §2.3 violation no longer
    stops it. That is only defensible if the violation is impossible to
    overlook afterwards: a journal of observations collected under a known
    model defect, indistinguishable from a clean one, is worse than no journal
    — it would be read as gate progress.
    """
    from risk_engine.service.state import understates_lower_tail

    diagnostics = getattr(bundle, "tail_diagnostics", ())
    if not diagnostics or not understates_lower_tail(diagnostics):
        return
    worst = max(diagnostics, key=lambda d: d.empirical_lower - d.model_at_threshold)
    print(
        f"  RECORDED UNDER A KNOWN §2.3 DEFECT: the fitted copula understates "
        f"lower-tail dependence (worst pair {worst.pair[0]}/{worst.pair[1]}, "
        f"empirical {worst.empirical_lower:.3f} against model "
        f"{worst.model_at_threshold:.3f}).\n"
        f"  These observations are DIAGNOSTIC EVIDENCE, not §3.3 gate-days: the "
        f"remedy is a skewed-t, which changes the predicted distribution, bumps "
        f"MODEL_VERSION and resets the counter. They exist to measure how much "
        f"the asymmetry moves P(liq) on real books, and in which direction — an "
        f"adversarial review established the sign depends on book shape, so it "
        f"is not known. No number from this bundle is served to anyone."
    )


def cmd_snapshot(args) -> int:
    # Journal FIRST, world second. The reverse order fitted a bundle and spent
    # §5.3 API weight before discovering the journal could not be opened at
    # all -- on the first live run that was a missing psycopg, reported as a
    # bare ModuleNotFoundError traceback after two minutes of work, with the
    # §2.3 defect warning scrolled off above it.
    #
    # This is the same fix `_live_world` already carries for the address list
    # ("loaded and validated HERE: before `_build_live_bundle` and before the
    # candle fetch"), applied to the other input that can fail for free.
    with CalibrationJournal(args.journal) as journal:
        provider, bundle, naive = (
            _fixture_world() if args.fixture else _live_world(args)
        )
        cron = ShadowCron(provider, bundle, journal, naive, n_paths=args.n_paths)
        swept_at = datetime.now(timezone.utc)
        report = cron.run_once(swept_at)
        print(report)
        # Census BEFORE the refused-list early return: a run that swept
        # nothing because every address was off-universe is exactly the cohort
        # selection B6 needs on record, and a `refused` run (bad list) is the
        # one case with no census to write.
        if not report.refused:
            tally: dict[str, int] = {}
            for _, reason in report.skipped:
                tally[reason] = tally.get(reason, 0) + 1
            journal.record_sweep(
                swept_at, DISTRIBUTION_VERSION, report.attempted, report.written,
                report.budget_exhausted, tally,
            )
        if report.refused:
            # Non-zero, and loud. A daily cron that prints "0 addresses
            # written" and exits 0 is a §3.3 window that stops advancing
            # without anybody being told: the address list is static, so the
            # same entry is refused again tomorrow and every day after, and
            # the 21-day requirement is read off consecutive days. Whatever
            # watches this job has to see a failure on day one.
            print(
                "  Nothing was written, and no market data was fetched. The list "
                "is static, so this repeats every day until the entry above is "
                "corrected -- and §3.3 needs 21 days of it."
            )
            return 2
        print(f"  sampling frame: {getattr(provider, 'frame', 'UNSTATED')}")
        _print_defect_note(bundle)
        for address, reason in report.skipped:
            print(f"  skipped {address}: {reason}")
        if report.budget_exhausted:
            print(
                "  the sweep stopped on the API weight budget rather than spending "
                "into the reserve live users need (§5.3)"
            )
    return 0


def cmd_resolve(args) -> int:
    # `load_addresses=False`: the resolver takes the addresses it needs from
    # the journal's own pending rows, never from the file. Refusing to run it
    # over a typo in a list it does not read would strand yesterday's
    # predictions past `DEFAULT_STALE_AFTER_S`, and the Info API serves only
    # current state -- a resolution that arrives late cannot be recovered, it
    # is lost. Blocking the snapshot on a bad list costs a day of new
    # predictions; blocking the resolver on it destroys observations already
    # paid for.
    # Journal first here too, and it matters more: this job runs hourly, so an
    # unopenable journal would burn a bundle build and API weight every hour
    # rather than once a day.
    with CalibrationJournal(args.journal) as journal:
        provider, _, _ = (
            _fixture_world() if args.fixture else _live_world(args, load_addresses=False)
        )
        report = resolve_due(
            journal, provider, datetime.now(timezone.utc),
            stale_after_s=args.stale_after_s,
        )
        print(report)
        permanent = dict(report.permanent)
        for pid, error in report.failed:
            print(f"  prediction {pid}: {error}")
        if permanent:
            # Named as permanent, because the retry is unbounded by design
            # (see `resolve.py`) and a row that fails identically every night
            # otherwise reads as a transient blip in a list that never
            # empties -- and hides the next real failure inside itself.
            print(
                f"  {len(permanent)} of those cannot ever resolve and will be "
                "retried on every run until the journal row is corrected by "
                "hand: " + ", ".join(f"{pid} ({why})" for pid, why in permanent.items())
            )
    return 0


def cmd_progress(args) -> int:
    with CalibrationJournal(args.journal) as journal:
        progress = journal.progress(DISTRIBUTION_VERSION)
        print(f"model {MODEL_VERSION}")
        print(progress)
        if progress.resolved_observations == 0:
            print("\nNothing resolved yet; nothing to score.")
            return 0
        for cohort in COHORTS:
            try:
                report = calibration_report(journal, DISTRIBUTION_VERSION, cohort)
            except ValueError as exc:
                print(f"\ncohort {cohort}: {exc}")
                continue
            print(f"\ncohort {cohort}: n={report.n} over {report.n_days} days")
            print(f"  KS {report.ks_statistic:.4f} (p={report.ks_pvalue:.4f})")
            for name, value in report.mean_crps.items():
                print(f"  mean CRPS {name}: {value:.6g}")
            tail = report.tail
            clustered = (
                f"[{tail.clustered_ci[0]:.4f}, {tail.clustered_ci[1]:.4f}]"
                if tail.clustered_ci
                else "(too few days)"
            )
            print(
                f"  VaR@95 breach {tail.breach_rate:.4f} "
                f"naive [{tail.naive_ci[0]:.4f}, {tail.naive_ci[1]:.4f}] "
                f"day-clustered {clustered}"
            )
            print("  " + report.gate_summary.replace("\n", "\n  "))
        print(
            f"\nPhase 4 gate: {'OPEN' if progress.gate_open else 'CLOSED'}. "
            "Read it off the book_unchanged cohort and the day-clustered interval "
            "(OPEN-QUESTIONS B1)."
        )
    return 0


def cmd_icc(args) -> int:
    """Measure the intra-day correlation and size the window from it (B1)."""
    import numpy as np

    from risk_engine.shadow.clustering import (
        breach_icc_confidence_set,
        estimate_breach_icc,
        recommend_window,
    )
    from risk_engine.shadow.journal import VARIANT_MODEL
    from risk_engine.shadow.metrics import load_cohort

    version = args.version or DISTRIBUTION_VERSION
    with CalibrationJournal(args.journal) as journal:
        cohort = load_cohort(journal, version, VARIANT_MODEL, args.cohort)
        if cohort.n == 0:
            print(f"no resolved observations in cohort {args.cohort}")
            return 1
        try:
            icc = estimate_breach_icc(
                cohort.pit, cohort.breached.astype(float), cohort.days,
                np.random.default_rng(args.seed), n_boot=args.boot,
                copula=args.copula,
            )
        except ValueError as exc:
            print(f"cannot estimate: {exc}")
            return 1

        print(f"cohort {args.cohort}, distribution {version}")
        print(icc.summary())
        print(
            "\nEstimated through the PIT values rather than the breaches. A "
            "breach is a 5% event, so a day of 200 addresses carries about ten "
            "of them, and ten events cannot resolve a correlation; the PIT "
            "values carry the same co-movement across every observation. The "
            f"{args.copula}-copula map converts it back (OPEN-QUESTIONS B1)."
        )

        if args.direct_interval:
            print("\ncomputing the assumption-free interval (slow)...")
            lo, hi = breach_icc_confidence_set(
                cohort.breached.astype(float), cohort.days,
                np.random.default_rng(args.seed), n_sims=args.direct_sims,
            )
            print(
                f"  breach data alone supports ICC in [{lo:.2f}, {hi:.2f}] "
                "with no copula assumption.\n"
                "  Wide by nature, not by defect — this is what the breaches on "
                "their own establish."
            )
        print()

        try:
            rec = recommend_window(
                icc,
                addresses_per_day=args.addresses_per_day,
                target_power=args.power,
                detect_rate=args.detect,
                n_trials=args.trials,
                n_boot=args.boot_power,
                seed=args.seed,
            )
        except ValueError as exc:
            print(f"cannot size the window: {exc}")
            return 1
        print(rec.summary())
        print(
            "\nSized off the upper end of the interval, not the point estimate: "
            "sizing off the middle is wrong half the time in the direction that "
            "shortens the window, and a short window yields a gate that passes "
            "without establishing anything."
        )
        if icc.n_days < 10:
            print(
                f"\n{icc.n_days} days is little to estimate a correlation from. "
                "Treat this as a direction until the pilot has run a fortnight."
            )
    return 0


def cmd_frame(args) -> int:
    """Write an address-list template that refuses to omit its frame."""
    path = Path(args.out)
    if path.exists() and not args.force:
        raise SystemExit(f"{path} exists; pass --force to overwrite")
    path.write_text(
        json.dumps(
            {
                "frame": "",
                "_frame_help": (
                    "REQUIRED. What is this a sample OF? The Info API enumerates no "
                    "addresses, so every list is biased somehow: the leaderboard "
                    "selects on performance (the very outcome being calibrated), the "
                    "trades feed selects on activity (the cohort the book-unchanged "
                    "filter then discards). State the bias here; it is attached to "
                    "the published calibration score. See OPEN-QUESTIONS B4."
                ),
                "addresses": [],
                "_addresses_help": (
                    "Each entry is 0x followed by 40 hex digits. Case does not "
                    "matter -- paste the checksummed form a block explorer shows "
                    "you; it is folded to lowercase so one account cannot end up "
                    "in the journal twice under two spellings. Anything else is "
                    "refused when the list loads, with its index, rather than "
                    "part-way through a sweep."
                ),
            },
            indent=2,
        )
        + "\n",
        # This template exists to be hand-edited, and the field an operator
        # fills in is prose. Writing it in the platform locale means a frame
        # typed on Windows and read in the Linux container is a different
        # string, or an outright UnicodeDecodeError at cron startup.
        encoding="utf-8",
    )
    print(f"wrote {path}; fill in 'frame' and 'addresses'")
    return 0


def _journal_arg(p):
    """`--journal`, defaulting to $SHADOW_DSN.

    Every one of these commands is normally run inside the compose stack,
    where the DSN is already in the environment. Requiring it on the command
    line meant the documented one-off,

        docker compose run --rm engine -m risk_engine.shadow progress \\
          --journal "$SHADOW_DSN"

    could not work: the shell that expands `$SHADOW_DSN` is the OPERATOR's,
    where it is empty, not the container's. The image has no shell in its
    entrypoint to expand it either. So the variable is read here, where it is
    actually in scope.

    Still overridable, because a local SQLite journal is a path and not a DSN,
    and that is the normal case outside the stack.
    """
    default = os.environ.get("SHADOW_DSN") or None
    p.add_argument(
        "--journal", required=default is None, default=default,
        help="calibration journal: a SQLite path, or a Postgres DSN. Defaults "
             "to $SHADOW_DSN, which the compose stack already sets.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="risk_engine.shadow")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        _journal_arg(p)
        p.add_argument("--fixture", action="store_true",
                       help="synthetic market and books; validates nothing, runs everything")
        p.add_argument("--addresses", help="JSON address list (required for a live run)")
        return p

    snap = common(sub.add_parser("snapshot", help="write today's predictions"))
    snap.add_argument("--n-paths", dest="n_paths", type=int, default=20_000)
    snap.set_defaults(func=cmd_snapshot)

    res = common(sub.add_parser("resolve", help="fill in outcomes whose horizon has elapsed"))
    res.add_argument("--stale-after-s", dest="stale_after_s", type=float,
                     default=DEFAULT_STALE_AFTER_S)
    res.set_defaults(func=cmd_resolve)

    prog = sub.add_parser("progress", help="report the §3.3 window and calibration")
    _journal_arg(prog)
    prog.set_defaults(func=cmd_progress)

    icc = sub.add_parser(
        "icc", help="measure the intra-day correlation and size the window (B1)"
    )
    _journal_arg(icc)
    icc.add_argument("--version", help=f"distribution version (default {DISTRIBUTION_VERSION})")
    icc.add_argument("--cohort", default=COHORT_BOOK_UNCHANGED, choices=list(COHORTS))
    icc.add_argument("--addresses-per-day", dest="addresses_per_day", type=int, default=200)
    icc.add_argument("--power", type=float, default=0.8, help="target power for sizing")
    icc.add_argument("--detect", type=float, default=0.10,
                     help="true breach rate the window must be able to detect")
    icc.add_argument("--boot", type=int, default=2_000, help="bootstrap draws for the interval")
    icc.add_argument("--boot-power", dest="boot_power", type=int, default=400)
    icc.add_argument("--copula", default="t", choices=["t", "gaussian"],
                     help="latent-to-breach map; t matches the engine (§2.3)")
    icc.add_argument("--direct-interval", dest="direct_interval", action="store_true",
                     help="also invert the test on the breach data alone (slow, wide)")
    icc.add_argument("--direct-sims", dest="direct_sims", type=int, default=200)
    # `%%`, not `%`: argparse runs every help string through `%` expansion, so
    # a literal percent is a format spec. "95% lower" parses as `% lo` -- space
    # flag, `l` length modifier, `o` octal -- and `--help` died with
    # "TypeError: %o format: an integer is required, not dict". The subcommand
    # was unusable by anyone who asked it what it did.
    icc.add_argument("--trials", type=int, default=500,
                     help="Monte Carlo trials per day count; the go/no-go decision "
                          "is read off a 95%% lower confidence bound on power, and "
                          "that bound needs this many trials to be worth reading "
                          "(too few makes the recommendation itself a coin flip)")
    icc.add_argument("--seed", type=int, default=0)
    icc.set_defaults(func=cmd_icc)

    frame = sub.add_parser("init-addresses", help="write an address-list template")
    frame.add_argument("--out", required=True)
    frame.add_argument("--force", action="store_true")
    frame.set_defaults(func=cmd_frame)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
