"""Champion/challenger for model changes (§3.3).

§3.3's anti-overfitting rule is that a new model version does not replace the
incumbent because it looks better; it runs in shadow *alongside* the old one
and migrates only on a statistically significant CRPS improvement.

The statistics matter more than the plumbing here, and in the same way they
did for the gate itself (OPEN-QUESTIONS B1): CRPS differences across
addresses observed on the same day are not independent. One market move
drives every address at once, so a naive paired t-test over thousands of
observations will call a coin-flip difference significant. The comparison is
therefore **paired by observation and bootstrapped by day** — the calendar
day is the independent unit, exactly as it is for the breach-rate interval.

Pairing matters as much as clustering: champion and challenger score the
same realised outcomes, so the difference in their CRPS is far better
resolved than either mean. Comparing two independently-computed means throws
that away, in the same way comparing two independent Monte Carlo runs throws
away the pairing in `pre_trade_delta` (D6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from risk_engine.shadow.journal import VARIANT_MODEL, CalibrationJournal
from risk_engine.shadow.metrics import COHORT_BOOK_UNCHANGED
from risk_engine.sim.stats import clustered_mean_ci


@dataclass(frozen=True, slots=True)
class ChallengerVerdict:
    champion_version: str
    challenger_version: str
    cohort: str
    n_paired: int
    n_days: int
    champion_crps: float
    challenger_crps: float
    #: challenger - champion. Negative means the challenger is better.
    mean_difference: float
    clustered_ci: tuple[float, float] | None
    min_days: int

    @property
    def challenger_wins(self) -> bool:
        """Significant improvement, by the day-clustered interval.

        Both bounds below zero, so the challenger is better on every
        plausible reading of the data -- not merely better on average.
        """
        if self.clustered_ci is None or self.n_days < self.min_days:
            return False
        return self.clustered_ci[1] < 0.0

    @property
    def verdict(self) -> str:
        if self.n_days < self.min_days:
            return (
                f"inconclusive: {self.n_days} paired days, {self.min_days} required. "
                "Days are the independent unit, not observations (OPEN-QUESTIONS B1)."
            )
        if self.clustered_ci is None:
            return "inconclusive: too few days to cluster over"
        lo, hi = self.clustered_ci
        if hi < 0:
            return (
                f"challenger wins: CRPS {self.mean_difference:+.6g} "
                f"[{lo:+.6g}, {hi:+.6g}], significantly better"
            )
        if lo > 0:
            return (
                f"champion holds: challenger is significantly WORSE by "
                f"{self.mean_difference:+.6g} [{lo:+.6g}, {hi:+.6g}]"
            )
        return (
            f"champion holds: CRPS difference {self.mean_difference:+.6g} "
            f"[{lo:+.6g}, {hi:+.6g}] spans zero, so the change is not "
            "distinguishable and §3.3 says do not migrate"
        )

    def __str__(self) -> str:
        return (
            f"{self.challenger_version} vs {self.champion_version} "
            f"on {self.cohort} ({self.n_paired} obs over {self.n_days} days)\n"
            f"  champion CRPS   {self.champion_crps:.6g}\n"
            f"  challenger CRPS {self.challenger_crps:.6g}\n"
            f"  {self.verdict}"
        )


def compare(
    journal: CalibrationJournal,
    champion_version: str,
    challenger_version: str,
    cohort: str = COHORT_BOOK_UNCHANGED,
    seed: int = 0,
    min_days: int = 21,
) -> ChallengerVerdict:
    """Score two distribution versions against the same realised outcomes.

    Only observations both versions predicted are used. An observation the
    challenger missed -- because it started later, which it always does --
    tells us nothing about which model is better, and including it would
    compare the two on different market days.
    """
    champion = {
        (r["address"], r["observation_day"]): r
        for r in _cohort_rows(journal, champion_version, cohort)
    }
    challenger = {
        (r["address"], r["observation_day"]): r
        for r in _cohort_rows(journal, challenger_version, cohort)
    }
    shared = sorted(set(champion) & set(challenger))
    if not shared:
        return ChallengerVerdict(
            champion_version, challenger_version, cohort, 0, 0,
            float("nan"), float("nan"), float("nan"), None, min_days,
        )

    champ = np.array([champion[k]["crps"] for k in shared], dtype=np.float64)
    chall = np.array([challenger[k]["crps"] for k in shared], dtype=np.float64)
    days = np.array([k[1] for k in shared])
    diff = chall - champ

    clustered = None
    if np.unique(days).size >= 2:
        clustered = clustered_mean_ci(
            diff, days, np.random.default_rng(seed)
        )
    return ChallengerVerdict(
        champion_version=champion_version,
        challenger_version=challenger_version,
        cohort=cohort,
        n_paired=len(shared),
        n_days=int(np.unique(days).size),
        champion_crps=float(champ.mean()),
        challenger_crps=float(chall.mean()),
        mean_difference=float(diff.mean()),
        clustered_ci=clustered,
        min_days=min_days,
    )


def _cohort_rows(journal: CalibrationJournal, version: str, cohort: str) -> list[dict]:
    from risk_engine.shadow.metrics import (
        COHORT_ALL,
        COHORT_BOOK_UNCHANGED,
        COHORT_NO_FLOW,
    )

    rows = journal.scored(version, VARIANT_MODEL)
    keep = []
    for r in rows:
        if r["stale_resolution"]:
            continue
        if cohort == COHORT_NO_FLOW and r["external_flow_usd"] != 0.0:
            continue
        if cohort == COHORT_BOOK_UNCHANGED and r["book_changed"]:
            continue
        if cohort not in (COHORT_ALL, COHORT_NO_FLOW, COHORT_BOOK_UNCHANGED):
            raise ValueError(f"unknown cohort {cohort!r}")
        keep.append(r)
    return keep
