"""Calibration metrics over the shadow journal (§0, §3.3).

Three cohorts, always reported together:

  all                every resolved observation
  no_external_flow   deposits and withdrawals removed
  book_unchanged     positions identical at both ends of the horizon

The gate should be read off `book_unchanged`, which is the only cohort where
the realised change is the thing the model actually predicted
(OPEN-QUESTIONS B2). It also selects against the most active traders, and
that selection effect belongs in any public calibration score.

Every interval is computed twice: naively, and clustered by calendar day.
Addresses observed on the same day share one market, so the naive interval
badly overstates precision. With ~21 days the clustered interval on a 5%
breach rate is roughly +/- 4 pp, which means §0.3's criterion as written is
not reachable inside the 21-day window §3.3 specifies -- that conflict is
reported by `tail_calibration`, not papered over (OPEN-QUESTIONS B1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from risk_engine.shadow.journal import (
    VARIANT_BASELINE_A,
    VARIANT_BASELINE_B,
    VARIANT_MODEL,
    CalibrationJournal,
)
from risk_engine.sim.stats import clustered_bootstrap_ci, ks_uniformity

COHORT_ALL = "all"
COHORT_NO_FLOW = "no_external_flow"
COHORT_BOOK_UNCHANGED = "book_unchanged"
COHORTS = (COHORT_ALL, COHORT_NO_FLOW, COHORT_BOOK_UNCHANGED)


@dataclass(frozen=True, slots=True)
class Cohort:
    name: str
    pit: np.ndarray
    crps: np.ndarray
    breached: np.ndarray
    days: np.ndarray

    @property
    def n(self) -> int:
        return self.pit.size

    @property
    def n_days(self) -> int:
        return int(np.unique(self.days).size)


@dataclass(frozen=True, slots=True)
class TailCalibration:
    breach_rate: float
    naive_ci: tuple[float, float]
    clustered_ci: tuple[float, float] | None
    target: float
    n: int
    n_days: int

    @property
    def passes_naive(self) -> bool:
        return self.naive_ci[0] <= self.target <= self.naive_ci[1]

    @property
    def passes_clustered(self) -> bool | None:
        if self.clustered_ci is None:
            return None
        return self.clustered_ci[0] <= self.target <= self.clustered_ci[1]


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    distribution_version: str
    cohort: str
    n: int
    n_days: int
    ks_statistic: float
    ks_pvalue: float
    mean_crps: dict[str, float]
    crps_beats_baseline_a: bool
    crps_beats_baseline_b: bool
    tail: TailCalibration

    @property
    def gate_summary(self) -> str:
        checks = [
            ("PIT uniform (KS p>0.05)", self.ks_pvalue > 0.05),
            ("CRPS beats baseline A", self.crps_beats_baseline_a),
            ("CRPS beats baseline B", self.crps_beats_baseline_b),
            ("VaR@95 breach in interval", self.tail.passes_clustered),
        ]
        lines = [f"{'PASS' if ok else 'FAIL'}  {name}" for name, ok in checks]
        return "\n".join(lines)


def load_cohort(
    journal: CalibrationJournal,
    version: str,
    variant: str,
    cohort: str,
    include_stale: bool = False,
) -> Cohort:
    """Rows for one cohort. Stale resolutions are excluded from all of them.

    A 24h forecast scored against a 72h realisation measures the resolver's
    punctuality, not the model (audit A-04). `include_stale` exists so the
    excluded rows can be inspected, not so they can be scored.
    """
    rows = journal.scored(version, variant)
    keep = []
    for r in rows:
        if not include_stale and r["stale_resolution"]:
            continue
        if cohort == COHORT_NO_FLOW and r["external_flow_usd"] != 0.0:
            continue
        if cohort == COHORT_BOOK_UNCHANGED and r["book_changed"]:
            continue
        keep.append(r)
    return Cohort(
        name=cohort,
        pit=np.array([r["pit"] for r in keep], dtype=np.float64),
        crps=np.array([r["crps"] for r in keep], dtype=np.float64),
        breached=np.array([bool(r["var_95_breached"]) for r in keep]),
        days=np.array([r["observation_day"] for r in keep]),
    )


def tail_calibration(cohort: Cohort, target: float = 0.05, seed: int = 0) -> TailCalibration:
    n = cohort.n
    if n == 0:
        raise ValueError(f"cohort {cohort.name} is empty")
    k = int(cohort.breached.sum())
    from risk_engine.sim.stats import wilson_interval

    naive = wilson_interval(k, n)
    clustered = None
    if cohort.n_days >= 2:
        clustered = clustered_bootstrap_ci(
            cohort.breached.astype(float),
            cohort.days,
            lambda v: float(v.mean()),
            np.random.default_rng(seed),
        )
    return TailCalibration(
        breach_rate=k / n,
        naive_ci=naive,
        clustered_ci=clustered,
        target=target,
        n=n,
        n_days=cohort.n_days,
    )


def calibration_report(
    journal: CalibrationJournal,
    distribution_version: str,
    cohort: str = COHORT_BOOK_UNCHANGED,
    seed: int = 0,
) -> CalibrationReport:
    model = load_cohort(journal, distribution_version, VARIANT_MODEL, cohort)
    if model.n == 0:
        raise ValueError(f"no resolved observations for {distribution_version}/{cohort}")
    a = load_cohort(journal, distribution_version, VARIANT_BASELINE_A, cohort)
    b = load_cohort(journal, distribution_version, VARIANT_BASELINE_B, cohort)

    ks_stat, ks_p = ks_uniformity(model.pit)
    means = {
        VARIANT_MODEL: float(model.crps.mean()),
        VARIANT_BASELINE_A: float(a.crps.mean()) if a.n else float("nan"),
        VARIANT_BASELINE_B: float(b.crps.mean()) if b.n else float("nan"),
    }
    return CalibrationReport(
        distribution_version=distribution_version,
        cohort=cohort,
        n=model.n,
        n_days=model.n_days,
        ks_statistic=ks_stat,
        ks_pvalue=ks_p,
        mean_crps=means,
        crps_beats_baseline_a=bool(a.n and means[VARIANT_MODEL] < means[VARIANT_BASELINE_A]),
        crps_beats_baseline_b=bool(b.n and means[VARIANT_MODEL] < means[VARIANT_BASELINE_B]),
        tail=tail_calibration(model, seed=seed),
    )
