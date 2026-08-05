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

from dataclasses import dataclass, field

import numpy as np

from risk_engine.shadow.journal import (
    VARIANT_BASELINE_A,
    VARIANT_BASELINE_B,
    VARIANT_MODEL,
    CalibrationJournal,
)
from risk_engine.sim.stats import clustered_mean_ci, ks_uniformity

COHORT_ALL = "all"
COHORT_NO_FLOW = "no_external_flow"
COHORT_BOOK_UNCHANGED = "book_unchanged"
COHORTS = (COHORT_ALL, COHORT_NO_FLOW, COHORT_BOOK_UNCHANGED)


#: The prefix `cron.py` writes for an address dropped for holding a coin the
#: model has no marginal for. The rest of the string is the sorted, comma-
#: separated list of EVERY missing coin, which is what makes B6's question
#: answerable: an address is recovered by universe U exactly when its whole
#: list is inside U.
OFF_UNIVERSE_PREFIX = "off-universe: "


@dataclass(frozen=True, slots=True)
class UniverseCandidate:
    """What widening the universe to `coins` would recover (B6 step 2)."""

    added: tuple[str, ...]
    #: Addresses whose entire off-universe list falls inside the widened
    #: universe, so they would be modelled rather than skipped.
    recovered: int
    #: Addresses still dropped for an off-universe holding afterwards.
    still_dropped: int

    @property
    def coins_added(self) -> int:
        return len(self.added)


def off_universe_demand(sweeps: list[dict]) -> dict[frozenset[str], int]:
    """Off-universe coin SETS and how many address-drops each accounts for.

    Keyed on the whole set, never on individual coins, and that distinction is
    the entire point of the function. B6 records the measurement defect it is
    written against: summing `KeyError: '<COIN>'` counts per coin answers "how
    often is this coin the first one missing", which is not "how many
    addresses would a universe containing it recover". An address holding ATOM
    and HYPE is recovered by neither {ATOM} nor {HYPE} -- only by a universe
    containing both -- so a per-coin tally can promise recoveries that adding
    the coin delivers none of.

    Rows whose reason is not an off-universe drop (flat books, non-positive
    equity) are ignored here: widening the universe cannot recover them, and
    counting them would overstate what the decision buys.
    """
    demand: dict[frozenset[str], int] = {}
    for sweep in sweeps:
        for reason, count in (sweep.get("skipped_by_reason") or {}).items():
            if not reason.startswith(OFF_UNIVERSE_PREFIX):
                continue
            coins = frozenset(
                c.strip() for c in reason[len(OFF_UNIVERSE_PREFIX):].split(",")
                if c.strip()
            )
            if coins:
                demand[coins] = demand.get(coins, 0) + int(count)
    return demand


def universe_candidates(
    demand: dict[frozenset[str], int], max_added: int = 25
) -> list[UniverseCandidate]:
    """Greedy widening: at each step add the coin recovering the most drops.

    Greedy rather than exhaustive because this is a set-cover problem and the
    census carries 40-odd distinct coins; the greedy order is the standard
    approximation and, more to the point, it is the order an operator would
    actually widen in -- one coin at a time, most valuable first.

    Each row reports what the universe of that size ACTUALLY recovers, counted
    by whole sets, so the numbers are what widening delivers rather than what
    a per-coin tally would promise.
    """
    universe: set[str] = set()
    remaining = dict(demand)
    out: list[UniverseCandidate] = []
    for _ in range(max_added):
        if not remaining:
            break
        # For each candidate coin, how many drops would be recovered by adding
        # it -- i.e. sets that become fully covered, not sets that merely
        # contain it.
        gain: dict[str, int] = {}
        for coins, n in remaining.items():
            for coin in coins:
                if coins <= universe | {coin}:
                    gain[coin] = gain.get(coin, 0) + n
        if not gain:
            # No single coin completes any remaining set. Every set left needs
            # two or more additions, so the greedy step is to add the coin
            # appearing in the most remaining drops and continue.
            freq: dict[str, int] = {}
            for coins, n in remaining.items():
                for coin in coins - universe:
                    freq[coin] = freq.get(coin, 0) + n
            if not freq:
                break
            best = max(sorted(freq), key=lambda c: freq[c])
        else:
            best = max(sorted(gain), key=lambda c: gain[c])
        universe.add(best)
        recovered_now = {c for c in remaining if c <= universe}
        for c in recovered_now:
            del remaining[c]
        out.append(
            UniverseCandidate(
                added=tuple(sorted(universe)),
                recovered=sum(demand[c] for c in demand if c <= universe),
                still_dropped=sum(remaining.values()),
            )
        )
        if not remaining:
            break
    return out


def in_cohort(row: dict, cohort: str) -> bool:
    """Does one scored row belong to `cohort`? The single definition of that.

    It was written out twice -- here in `load_cohort` and again in
    `champion.py::_cohort_rows` -- and the two must agree exactly, because
    §3.1's gate is computed from the first and §0.2's migration decision from
    the second. Two populations under one cohort name would let a challenger
    be judged against a champion measured on different rows, with both numbers
    looking entirely ordinary.

    They had already drifted. An unrecognised cohort string raised in
    `champion.py` and was silently treated as `all` here, so
    `calibration_report(cohort="book_unchagned")` returned a report over every
    row, labelled with the typo, and nothing in the output said the filter had
    not been applied. That is the §3.3 gate's own population. Validation now
    happens once, here, before any row is read -- and therefore also on an
    empty journal, which is the case `champion.py` missed by validating inside
    its loop.

    Stale resolutions are NOT filtered here: `load_cohort` takes them out
    separately under `include_stale`, which exists so the excluded rows can be
    inspected rather than scored (audit A-04). Folding that in would remove
    the only way to look at them.
    """
    if cohort not in COHORTS:
        raise ValueError(f"unknown cohort {cohort!r}; expected one of {list(COHORTS)}")
    if cohort == COHORT_NO_FLOW:
        return row["external_flow_usd"] == 0.0
    if cohort == COHORT_BOOK_UNCHANGED:
        return not row["book_changed"]
    return True


@dataclass(frozen=True, slots=True)
class Cohort:
    name: str
    pit: np.ndarray
    crps: np.ndarray
    breached: np.ndarray
    days: np.ndarray
    #: `(address, observation_day)` per row, in the same order.
    #:
    #: Carried so the CRPS comparison can be PAIRED. Without it the three
    #: variants' cohorts were loaded independently and their means compared
    #: as if they described the same observations, which they need not: a row
    #: can fail on its own, and the resolver's ceiling lands between rows,
    #: not between addresses (`due()` orders by `resolves_at`, which the three
    #: variants of one address share).
    keys: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))

    def select(self, wanted: set) -> "Cohort":
        """This cohort restricted to `wanted` keys, order preserved."""
        mask = np.array([k in wanted for k in self.keys], dtype=bool)
        return Cohort(
            name=self.name, pit=self.pit[mask], crps=self.crps[mask],
            breached=self.breached[mask], days=self.days[mask],
            keys=self.keys[mask],
        )

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
    #: Observations all three variants scored — what the CRPS comparison used.
    #: Reported separately from `n` because a gap between them is the whole
    #: warning: it says some observations could not be compared at all.
    n_paired: int = 0
    #: Per variant, how many of its own rows had no counterpart. Recorded
    #: because the previous report carried the model's `n` alone, so an
    #: unpaired comparison left no trace in the output that decides §0.2.
    unpaired_dropped: dict = field(default_factory=dict)

    @property
    def gate_summary(self) -> str:
        checks = [
            ("PIT uniform (KS p>0.05)", self.ks_pvalue > 0.05),
            ("CRPS beats baseline A", self.crps_beats_baseline_a),
            ("CRPS beats baseline B", self.crps_beats_baseline_b),
            ("VaR@95 breach in interval", self.tail.passes_clustered),
        ]
        lines = [f"{'PASS' if ok else 'FAIL'}  {name}" for name, ok in checks]
        # Said whenever it happened, not only when it changed a verdict.
        # An unpaired comparison that HAPPENS to agree with the paired one is
        # still a comparison nobody could audit from the output.
        dropped = sum(self.unpaired_dropped.values())
        if dropped:
            lines.append(
                f"      note: CRPS compared over {self.n_paired} paired "
                f"observations; {dropped} row(s) had no counterpart in every "
                f"variant and were excluded from the comparison "
                f"({self.unpaired_dropped})"
            )
        return "\n".join(lines)


def _key_array(rows: list[dict]) -> np.ndarray:
    """A 1-D object array of `(address, observation_day)` TUPLES.

    Built element-wise on purpose: `np.array([(a, d), ...], dtype=object)`
    infers a 2-D array, and iterating that yields unhashable row arrays
    rather than the tuples the set intersection needs.
    """
    out = np.empty(len(rows), dtype=object)
    out[:] = [(r["address"], r["observation_day"]) for r in rows]
    return out


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
    # Before reading a row, so a mistyped cohort fails on an empty journal too.
    if cohort not in COHORTS:
        raise ValueError(f"unknown cohort {cohort!r}; expected one of {list(COHORTS)}")
    rows = journal.scored(version, variant)
    keep = [
        r for r in rows
        if (include_stale or not r["stale_resolution"]) and in_cohort(r, cohort)
    ]
    return Cohort(
        name=cohort,
        pit=np.array([r["pit"] for r in keep], dtype=np.float64),
        crps=np.array([r["crps"] for r in keep], dtype=np.float64),
        breached=np.array([bool(r["var_95_breached"]) for r in keep]),
        days=np.array([r["observation_day"] for r in keep]),
        keys=_key_array(keep),
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
        clustered = clustered_mean_ci(
            cohort.breached.astype(float),
            cohort.days,
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

    # §0.2's comparison is PAIRED, over the observations all three variants
    # actually scored. Loading the three cohorts independently and comparing
    # their means treats them as describing the same observations, and they
    # need not: a row can fail alone, and the resolver's ceiling lands
    # BETWEEN rows rather than between addresses, because `due()` orders by
    # `resolves_at` and the three variants of one address share it.
    #
    # It is not a rounding-level concern. CRPS carries the units of the
    # equity change, so its mean is dominated by the largest accounts.
    # Reproduced with one whale and twenty small accounts, the model equal or
    # better on every one: dropping a SINGLE baseline-A row made the model
    # "lose" to A (247.6 against 12.0) and "beat" B (247.6 against 249.5) on
    # the same data. Both directions are reachable, and the other one ships a
    # model that is not better — §10's forbidden side.
    #
    # Pairing also tightens the comparison, for the reason D6 gives about the
    # other estimator in this codebase: common observations cancel, so what
    # is left is the difference rather than the spread of the population.
    paired = set(model.keys) & set(a.keys) & set(b.keys)
    m_p, a_p, b_p = (c.select(paired) for c in (model, a, b))
    means = {
        VARIANT_MODEL: float(m_p.crps.mean()) if m_p.n else float("nan"),
        VARIANT_BASELINE_A: float(a_p.crps.mean()) if a_p.n else float("nan"),
        VARIANT_BASELINE_B: float(b_p.crps.mean()) if b_p.n else float("nan"),
    }
    return CalibrationReport(
        distribution_version=distribution_version,
        cohort=cohort,
        n=model.n,
        n_days=model.n_days,
        n_paired=m_p.n,
        unpaired_dropped={
            VARIANT_MODEL: model.n - m_p.n,
            VARIANT_BASELINE_A: a.n - a_p.n,
            VARIANT_BASELINE_B: b.n - b_p.n,
        },
        ks_statistic=ks_stat,
        ks_pvalue=ks_p,
        mean_crps=means,
        crps_beats_baseline_a=bool(m_p.n and means[VARIANT_MODEL] < means[VARIANT_BASELINE_A]),
        crps_beats_baseline_b=bool(m_p.n and means[VARIANT_MODEL] < means[VARIANT_BASELINE_B]),
        # PIT, KS and the tail describe the MODEL alone rather than a
        # comparison, so they keep every model observation. Restricting them
        # to the paired set would throw away data for no reason.
        tail=tail_calibration(model, seed=seed),
    )
