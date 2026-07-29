"""Entry points for the shadow harness (§3.3).

    python -m risk_engine.shadow snapshot --journal shadow.db --addresses addrs.json
    python -m risk_engine.shadow resolve  --journal shadow.db
    python -m risk_engine.shadow progress --journal shadow.db

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
import sys
from datetime import datetime, timezone
from pathlib import Path

from risk_engine.shadow.cron import ShadowCron
from risk_engine.shadow.journal import CalibrationJournal
from risk_engine.shadow.metrics import COHORTS, calibration_report
from risk_engine.shadow.providers import FileAddressSource, LiveSnapshotProvider
from risk_engine.shadow.resolve import DEFAULT_STALE_AFTER_S, resolve_due
from risk_engine.validation.baselines import NaiveBaseline, historical_24h_log_returns
from risk_engine.version import DISTRIBUTION_VERSION, MODEL_VERSION

log = logging.getLogger("risk_engine.shadow")

#: §5.3: the sweep keeps three quarters of the weight budget for live users.
SHADOW_RESERVED_FRACTION = 0.75


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
        books[f"0xfixture{i:03d}"] = Book(
            f"0xfixture{i:03d}", equity, tuple(positions), now - timedelta(seconds=5)
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


def _live_world(args):
    from risk_engine.market.info import WeightBudget
    from risk_engine.service.state import _build_live_bundle

    if not args.addresses:
        raise SystemExit(
            "--addresses is required for a live run: the address list is a sampling "
            "frame and it has to be chosen deliberately (OPEN-QUESTIONS B4)"
        )
    bundle, specs, spot = _build_live_bundle()
    source = FileAddressSource(Path(args.addresses))
    budget = WeightBudget(reserved_fraction=SHADOW_RESERVED_FRACTION)
    provider = LiveSnapshotProvider(source, budget=budget, universe=tuple(spot))
    # Reuse the freshly-built bundle's view of the venue rather than
    # re-fetching it per sweep.
    provider._specs = specs
    provider._spot = dict(spot)

    import numpy as np

    from risk_engine.market.parse import parse_candles_to_log_returns

    import time as _time

    now_ms = int(_time.time() * 1000)
    candles = provider.client.candle_snapshot(
        "BTC", "1h", now_ms - 90 * 24 * 3600 * 1000, now_ms
    )
    _, hourly = parse_candles_to_log_returns(candles)
    return provider, bundle, NaiveBaseline(historical_24h_log_returns(np.asarray(hourly)))


def cmd_snapshot(args) -> int:
    provider, bundle, naive = (
        _fixture_world() if args.fixture else _live_world(args)
    )
    with CalibrationJournal(args.journal) as journal:
        cron = ShadowCron(provider, bundle, journal, naive, n_paths=args.n_paths)
        report = cron.run_once(datetime.now(timezone.utc))
        print(report)
        print(f"  sampling frame: {getattr(provider, 'frame', 'UNSTATED')}")
        for address, reason in report.skipped:
            print(f"  skipped {address}: {reason}")
        if report.budget_exhausted:
            print(
                "  the sweep stopped on the API weight budget rather than spending "
                "into the reserve live users need (§5.3)"
            )
    return 0


def cmd_resolve(args) -> int:
    provider, _, _ = _fixture_world() if args.fixture else _live_world(args)
    with CalibrationJournal(args.journal) as journal:
        report = resolve_due(
            journal, provider, datetime.now(timezone.utc),
            stale_after_s=args.stale_after_s,
        )
        print(report)
        for pid, error in report.failed:
            print(f"  prediction {pid}: {error}")
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
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {path}; fill in 'frame' and 'addresses'")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="risk_engine.shadow")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--journal", required=True, help="path to the calibration journal")
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
    prog.add_argument("--journal", required=True)
    prog.set_defaults(func=cmd_progress)

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
