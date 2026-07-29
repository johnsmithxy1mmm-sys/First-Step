"""Drift convention (§2.4) — and why the section cannot be taken literally.

§2.4 requires zero drift **in price** under the real measure: any non-zero
drift is a hidden market forecast, which this product does not make.

That is unimplementable together with §2.2's Student-t marginals. Zero price
drift means `E[exp(r)] = 1`. The moment generating function of a Student-t is
infinite for every degrees-of-freedom value, so `E[exp(r)] = +inf` no matter
what constant is added to `r`. There is no drift that makes a t-distributed
log-return series a price martingale.

Two admissible readings remain:

ZERO_LOG_RETURN
    `E[r] = 0`. The price is median-flat. This is the reading that keeps the
    up and down cases symmetric, which is what §2.4 is defending, and it is
    the default.

MEDIAN_PRESERVING_CONVEXITY
    `E[r] = -sigma^2/2` per step. This is what "zero price drift" would mean
    if the moments existed, imported as a convention. It makes longs look
    slightly riskier and shorts slightly safer than ZERO_LOG_RETURN, by
    `sigma^2/2` per step — about 8 bp over 24 h at a 4% daily vol.

Neither is a forecast: both fix the location at zero under a stated
convention, and the difference between them is far below the estimation
error on sigma. The choice is exposed rather than buried because it is a
one-sided choice about risk, and §10 forbids making those silently.
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class DriftConvention(str, Enum):
    ZERO_LOG_RETURN = "zero_log_return"
    MEDIAN_PRESERVING_CONVEXITY = "median_preserving_convexity"


def log_drift_per_step(
    convention: DriftConvention, step_vol: np.ndarray | float
) -> np.ndarray | float:
    """Per-step log-return drift implied by the convention.

    Never a function of past returns, momentum, funding, open interest or
    anything else that would constitute a forecast. The only permitted
    non-zero cash flow in the model is funding (§1.5), which is a contractual
    payment, not a prediction, and it is applied in the simulator rather than
    here.
    """
    if convention is DriftConvention.ZERO_LOG_RETURN:
        return np.zeros_like(np.asarray(step_vol, dtype=float))
    return -0.5 * np.asarray(step_vol, dtype=float) ** 2
