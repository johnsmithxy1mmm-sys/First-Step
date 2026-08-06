"""Validation CLI — the Phase 1 gate as a command.

    python -m risk_engine.validation.cli benchmarks
    python -m risk_engine.validation.cli shadow --journal shadow.db

Exits non-zero when a gate fails. §9 requires a failing criterion to stop the
phase and be reported, not to be worked around, so this never downgrades a
failure to a warning.
"""

from __future__ import annotations

import argparse
import sys

from risk_engine.observability.metrics import METRICS
from risk_engine.validation.benchmarks import run_all
from risk_engine.version import DISTRIBUTION_VERSION, MODEL_VERSION


def cmd_benchmarks(_args: argparse.Namespace) -> int:
    print(f"model version {MODEL_VERSION}\n")
    results = run_all()
    for r in results:
        print(r)
    failed = [r for r in results if not r.passed]
    print()
    if failed:
        print(f"GATE FAILED: {len(failed)} of {len(results)} benchmarks failed.")
        print("Phase 1 is not complete. Do not weaken the tolerances (§9).")
        return 1
    print(f"GATE PASSED: {len(results)}/{len(results)} benchmarks.")
    return 0


def cmd_shadow(args: argparse.Namespace) -> int:
    from risk_engine.shadow.journal import CalibrationJournal
    from risk_engine.shadow.metrics import COHORTS, calibration_report

    with CalibrationJournal(args.journal) as journal:
        progress = journal.progress(DISTRIBUTION_VERSION)
        print(progress)
        if progress.resolved_observations == 0:
            print("\nNo resolved observations yet; nothing to score.")
            return 0
        for cohort in COHORTS:
            try:
                report = calibration_report(journal, DISTRIBUTION_VERSION, cohort)
            except ValueError as exc:
                print(f"\ncohort {cohort}: {exc}")
                continue
            print(f"\ncohort {cohort}: n={report.n} over {report.n_days} days")
            print(f"  KS statistic {report.ks_statistic:.4f} (p={report.ks_pvalue:.4f})")
            for name, value in report.mean_crps.items():
                print(f"  mean CRPS {name}: {value:.6g}")
            tail = report.tail
            print(
                f"  VaR@95 breach rate {tail.breach_rate:.4f} "
                f"naive CI [{tail.naive_ci[0]:.4f}, {tail.naive_ci[1]:.4f}]"
                + (
                    f" day-clustered CI [{tail.clustered_ci[0]:.4f}, {tail.clustered_ci[1]:.4f}]"
                    if tail.clustered_ci
                    else " (too few days to cluster)"
                )
            )
            print("  " + report.gate_summary.replace("\n", "\n  "))
        print(
            f"\nPhase 4 entry gate: {'OPEN' if progress.gate_open else 'CLOSED'}. "
            "Read it off the book_unchanged cohort and the day-clustered interval."
        )
    return 0


def cmd_metrics(_args: argparse.Namespace) -> int:
    import json

    print(json.dumps(METRICS.snapshot(), indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="risk_engine.validation.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("benchmarks", help="run the §3.1 analytical gate").set_defaults(
        func=cmd_benchmarks
    )
    shadow = sub.add_parser("shadow", help="report §3.3 shadow progress and calibration")
    shadow.add_argument("--journal", required=True, help="path to the calibration journal")
    shadow.set_defaults(func=cmd_shadow)
    sub.add_parser("metrics", help="dump the observability snapshot").set_defaults(
        func=cmd_metrics
    )

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
