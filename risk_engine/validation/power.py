"""How long the shadow window has to be for §0.3's VaR criterion to mean
anything (OPEN-QUESTIONS B1).

    python -m risk_engine.validation.power

B1 records the problem: §0.3 wants the realised VaR@95 breach rate inside a
*binomial* 95% interval, and 200-500 addresses observed on the same day are
not 200-500 independent observations. One market move drives all of them, and
on a day BTC drops 8% nearly every address breaches at once. It also records
an estimate -- "roughly ±4 pp" -- which was arithmetic on the back of the day
count, not a measurement. This measures it.

Two failures, in opposite directions, and the decision needs both:

**Keeping §0.3 as written** means testing the observed rate against an
interval computed as though the observations were independent. That interval
is far too tight, so a *correctly calibrated* model fails the gate on nothing
but the luck of which days landed in the window. The false-rejection rate is
reported below; at any realistic clustering it is not close to 5%.

**Using the honest day-clustered interval** fixes the false rejections and
replaces them with the opposite problem: over 21 days the interval is so wide
that it contains 5% almost regardless of the truth. That is not a test that
passes; it is a test that cannot fail, which is worse, because it reads as
validation.

The intra-day correlation is the parameter neither the specification nor this
code can supply -- it takes real data. So it is swept rather than assumed, and
the tables below are read at whichever value the first weeks of shadow data
turn out to show. Assuming a convenient value here would be assuming the
answer.

The generating model is beta-binomial: each day draws its own breach
probability, and addresses breach independently given the day. That is the
standard one-parameter way to hold the marginal rate at 5% while varying how
much a day moves everyone together, and its intra-class correlation is
exactly the swept parameter.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass

import numpy as np

from risk_engine.sim.stats import wilson_interval

#: §0.3's target: 5% of 24 h outcomes breach the 95% VaR.
NOMINAL_BREACH_RATE = 0.05
#: §3.3's window, and the number under examination.
SPEC_DAYS = 21
SPEC_ADDRESSES = 200


def simulate_days(
    n_days: int, n_per_day: int, p: float, icc: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Breaches per day under a beta-binomial with the given intra-class
    correlation.

    `icc` is the correlation between two addresses' breach indicators on the
    same day. At 0 the days are irrelevant and the observations are genuinely
    independent; at 1 every address on a day breaches or none does, and the
    effective sample size is the day count.
    """
    counts = np.full(n_days, n_per_day, dtype=np.int64)
    if icc <= 0.0:
        return rng.binomial(n_per_day, p, size=n_days), counts
    if icc >= 1.0:
        return (rng.random(n_days) < p).astype(np.int64) * n_per_day, counts
    concentration = (1.0 - icc) / icc
    daily_p = rng.beta(p * concentration, (1.0 - p) * concentration, size=n_days)
    return rng.binomial(n_per_day, daily_p), counts


def clustered_rate_ci(
    day_sums: np.ndarray,
    day_counts: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 2_000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Day-clustered STUDENTISED interval on the pooled breach rate.

    Resamples whole days, which is what `clustered_mean_ci` does for the
    general case. Specialised here because the statistic is a ratio of sums:
    a resampled rate is `sum(picked sums) / sum(picked counts)`, so a whole
    sweep costs a few gathers instead of rebuilding the observation vector
    thousands of times. It is the same interval, and `test_power.py` checks
    that against the general implementation rather than taking it on trust.

    Studentised since 2026-08-04, with the general one, for the reason
    recorded there: a PERCENTILE bootstrap over days undercovers badly when
    the clusters are few, and 21 days is few. It covered 82.8% at ICC 0.20
    against a nominal 95%.

    That matters here specifically, and more than it does at the gate. The
    `clustered_half_width_pp` column of the table this module produces is
    what B1 sizes the shadow window from, and a half-width taken from an
    interval covering 83% is too narrow: at ICC 0.20 the honest figure is
    ~7.3pp rather than ~3.6pp. A window sized off the old number looks
    informative and is not, which is §10's direction reached by arithmetic.
    Any table printed before this date is under-stated and should be re-run.
    """
    d = day_sums.size
    if d < 2:
        raise ValueError("need at least two days to resample over them")
    total = float(day_counts.sum())
    theta = float(day_sums.sum()) / total
    resid = day_sums - day_counts * theta
    se = float(np.sqrt((d / (d - 1.0)) * float((resid**2).sum()) / total**2))
    if se <= 0.0:
        return theta, theta

    pick = rng.integers(0, d, size=(n_boot, d))
    s, c = day_sums[pick], day_counts[pick]
    tot_b = c.sum(axis=1)
    theta_b = s.sum(axis=1) / tot_b
    resid_b = s - c * theta_b[:, None]
    se_b = np.sqrt((d / (d - 1.0)) * (resid_b**2).sum(axis=1) / tot_b**2)
    ts = np.divide(theta_b - theta, se_b, out=np.zeros_like(theta_b),
                   where=se_b > 0)
    hi_t, lo_t = np.quantile(ts, [1.0 - alpha / 2, alpha / 2])
    return float(theta - hi_t * se), float(theta - lo_t * se)


@dataclass(frozen=True)
class Cell:
    days: int
    addresses_per_day: int
    icc: float
    #: How often a correctly calibrated model is rejected by §0.3 as written.
    naive_false_rejection: float
    #: Median half-width of the honest day-clustered interval, in pp.
    clustered_half_width_pp: float
    #: Probability the clustered interval excludes 5% when the true rate is
    #: 8% and 10% -- i.e. whether real miscalibration is detectable at all.
    power_at_8pct: float
    power_at_10pct: float

    def render(self) -> str:
        return (
            f"{self.days:5d} {self.addresses_per_day:7d} {self.icc:6.2f} "
            f"{self.naive_false_rejection:14.1%} "
            f"{self.clustered_half_width_pp:12.2f} "
            f"{self.power_at_8pct:10.1%} {self.power_at_10pct:11.1%}"
        )


def evaluate(
    days: int,
    addresses_per_day: int,
    icc: float,
    n_trials: int,
    n_boot: int,
    seed: int,
) -> Cell:
    rng = np.random.default_rng(seed)

    rejected = 0
    half_widths = np.empty(n_trials)
    for t in range(n_trials):
        sums, counts = simulate_days(days, addresses_per_day, NOMINAL_BREACH_RATE, icc, rng)
        total, n = int(sums.sum()), int(counts.sum())

        # §0.3 as written: is the observed rate inside the binomial interval?
        lo, hi = wilson_interval(total, n)
        if not (lo <= NOMINAL_BREACH_RATE <= hi):
            rejected += 1

        c_lo, c_hi = clustered_rate_ci(sums, counts, rng, n_boot=n_boot)
        half_widths[t] = (c_hi - c_lo) / 2.0

    def power(p_true: float) -> float:
        detected = 0
        for _ in range(n_trials):
            sums, counts = simulate_days(days, addresses_per_day, p_true, icc, rng)
            c_lo, c_hi = clustered_rate_ci(sums, counts, rng, n_boot=n_boot)
            if not (c_lo <= NOMINAL_BREACH_RATE <= c_hi):
                detected += 1
        return detected / n_trials

    return Cell(
        days=days,
        addresses_per_day=addresses_per_day,
        icc=icc,
        naive_false_rejection=rejected / n_trials,
        clustered_half_width_pp=float(np.median(half_widths)) * 100.0,
        power_at_8pct=power(0.08),
        power_at_10pct=power(0.10),
    )


def _power_trials(
    days: int,
    addresses_per_day: int,
    icc: float,
    p_true: float,
    n_trials: int,
    n_boot: int,
    seed: int,
) -> int:
    """How many of `n_trials` Monte Carlo trials the gate detects on.

    Shared by `power_at` and `power_ci_at` so the two never drift: a caller
    asking for the point estimate and a caller asking for its confidence
    bound must be describing the same simulation.
    """
    if not 0.0 < p_true < 1.0:
        raise ValueError(f"p_true must be a rate in (0, 1), got {p_true}")
    if abs(p_true - NOMINAL_BREACH_RATE) < 1e-9:
        raise ValueError(
            "p_true equals the nominal rate; power against the null is just the "
            "false-rejection rate, and asking for it this way is a sign of confusion"
        )
    rng = np.random.default_rng(seed)
    detected = 0
    for _ in range(n_trials):
        sums, counts = simulate_days(days, addresses_per_day, p_true, icc, rng)
        lo, hi = clustered_rate_ci(sums, counts, rng, n_boot=n_boot)
        if not (lo <= NOMINAL_BREACH_RATE <= hi):
            detected += 1
    return detected


def power_at(
    days: int,
    addresses_per_day: int,
    icc: float,
    p_true: float,
    n_trials: int,
    n_boot: int,
    seed: int,
) -> float:
    """Power of the day-clustered gate against an arbitrary true breach rate.

    `evaluate` reports power only at the two rates its table shows (8%, 10%).
    Window sizing must not be quantised to those: detecting a true rate of 6%
    is *harder* than detecting 8%, and the audit found that mapping every
    requested rate below 9.5% onto the 8% column recommended windows that
    were too short for the stricter target -- the §10-forbidden direction.
    This computes the requested rate itself.

    A point estimate over `n_trials` Monte Carlo trials. For deciding whether
    a window is long enough, use `power_ci_at` instead -- the point estimate
    alone lets Monte Carlo noise pick the window (audit: measured 76% against
    an 80% target on one seed at n_trials=120, which is 1.1 standard errors
    from the threshold, i.e. close to a coin flip).
    """
    return _power_trials(days, addresses_per_day, icc, p_true, n_trials, n_boot, seed) / n_trials


def power_ci_at(
    days: int,
    addresses_per_day: int,
    icc: float,
    p_true: float,
    n_trials: int,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(point, ci_low, ci_high) for power against `p_true`, Wilson over trials.

    `power_at`'s point estimate is itself a Monte Carlo draw with its own
    sampling error -- a proportion out of `n_trials` binary outcomes -- and
    a window-sizing decision that reads the point estimate against a target
    is exactly the same mistake §0.3's naive breach-rate test makes: it
    lets sampling noise decide instead of the data. `recommend_window` reads
    `ci_low` and requires it, not the point, to clear the target.
    """
    detected = _power_trials(days, addresses_per_day, icc, p_true, n_trials, n_boot, seed)
    lo, hi = wilson_interval(detected, n_trials)
    return detected / n_trials, lo, hi


def sweep(
    day_grid: list[int],
    address_grid: list[int],
    icc_grid: list[float],
    n_trials: int,
    n_boot: int,
    seed: int,
) -> list[Cell]:
    cells = []
    for i, days in enumerate(day_grid):
        for j, addresses in enumerate(address_grid):
            for k, icc in enumerate(icc_grid):
                cells.append(
                    evaluate(days, addresses, icc, n_trials, n_boot,
                             seed + 1_000 * i + 100 * j + k)
                )
    return cells


HEADER = (
    " days address    icc  naive false-rej  clustered ±pp  "
    "power@8%  power@10%"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="risk_engine.validation.power",
        description="what window §0.3's VaR criterion actually needs (B1)",
    )
    parser.add_argument("--days", default="21,30,45,60,90,120,180")
    parser.add_argument("--addresses", default="200,500")
    parser.add_argument("--icc", default="0.0,0.02,0.05,0.10,0.20,0.40")
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--boot", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--report", help="write the full grid as JSON")
    args = parser.parse_args(argv)

    day_grid = [int(x) for x in args.days.split(",")]
    address_grid = [int(x) for x in args.addresses.split(",")]
    icc_grid = [float(x) for x in args.icc.split(",")]

    cells = sweep(day_grid, address_grid, icc_grid, args.trials, args.boot, args.seed)

    print(f"§0.3 VaR criterion, {args.trials} trials per cell, "
          f"nominal breach rate {NOMINAL_BREACH_RATE:.0%}\n")
    print(HEADER)
    print("-" * len(HEADER))
    last = None
    for cell in cells:
        if last is not None and cell.days != last:
            print()
        print(cell.render())
        last = cell.days

    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump([asdict(c) for c in cells], fh, indent=2)
        print(f"\nwrote {args.report}")

    print(
        "\nnaive false-rej: how often §0.3 as written rejects a model that is "
        "correctly calibrated.\nAt 5% it would be behaving as advertised; above "
        "that it is failing good models on\nwhich days happened to land in the "
        "window.\n\nclustered ±pp: half-width of the honest day-clustered "
        "interval. §0.3's own tolerance\naround a 5% rate is about ±0.6 pp at "
        "4200 observations, so anything much above that\nmeans the window "
        "cannot answer the question it was sized for.\n\npower: probability the "
        "clustered interval excludes 5% when the truth is 8% or 10%.\nA model "
        "understating tail risk by half is what this is meant to catch; where "
        "power\nis low the gate cannot catch it, and passing says nothing."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
