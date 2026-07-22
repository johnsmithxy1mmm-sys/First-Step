"""Self-calibration: the bot learns its parameters from its own data.

Nothing here is a constant-by-decree. Five learners:

  TailBiasCalibrator  — the fade's bias_discount per (category, price bucket),
                        shrunk toward a global prior on small samples.
  MarkoutFeedback     — per-market MM spread multiplier from realized markout
                        (widen where we get adversely selected, tighten where not).
  PlattCalibrator     — a 1-D logistic recalibration of p_est vs outcomes.
  CorrelationLearner  — pairwise category correlations from the tick store's
                        category-index series; replaces the expert matrix in
                        VaR/sizing once there is enough history.
  FillCalibrator      — binned mapping from PREDICTED fill probability to the
                        REALIZED fill rate of our own quotes; corrects the
                        microstructure model where it is systematically off.

All are pure functions of recorded data (testable offline). They start as the
prior/identity and only move as evidence accumulates.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict

log = logging.getLogger(__name__)

# YES-price buckets for the tail-bias learner (upper edges).
_PRICE_BUCKETS = (0.01, 0.02, 0.03, 0.05, 0.10)


def _bucket(p: float) -> float:
    for edge in _PRICE_BUCKETS:
        if p <= edge + 1e-9:
            return edge
    return _PRICE_BUCKETS[-1]


class TailBiasCalibrator:
    """Learns fade bias_discount per (category, price bucket) from resolutions.

    A faded tail 'happened' when our NO leg lost. Empirical fair P(tail) = the
    happen-rate; bias = 1 - fair/market. Shrunk toward `prior` by n/(n+k) so a
    handful of resolutions don't swing it.
    """

    def __init__(self, prior: float = 0.35, shrink_k: float = 20.0,
                 bias_cap: float = 0.60):
        self._prior = prior
        self._k = shrink_k
        self._cap = bias_cap
        # key -> [happen_count, n, sum_p_mkt_yes]
        self._buckets: dict[tuple[str, float], list[float]] = defaultdict(
            lambda: [0.0, 0.0, 0.0])

    def fit(self, resolved: list[dict]) -> "TailBiasCalibrator":
        """resolved: [{category, p_mkt (NO entry price), won (NO won?)}]."""
        self._buckets.clear()
        for r in resolved:
            p_mkt_yes = 1.0 - float(r["p_mkt"])          # fade stores the NO entry
            if not 0.0 < p_mkt_yes < 0.5:
                continue
            tail_happened = not bool(r["won"])           # NO lost -> tail happened
            slot = self._buckets[(r["category"], _bucket(p_mkt_yes))]
            slot[0] += 1.0 if tail_happened else 0.0
            slot[1] += 1.0
            slot[2] += p_mkt_yes
        return self

    def bias(self, category: str, p_mkt_yes: float) -> float:
        slot = self._buckets.get((category, _bucket(p_mkt_yes)))
        if slot is None or slot[1] <= 0 or slot[2] <= 0:
            return self._prior
        happen, n, sum_p = slot
        empirical = 1.0 - happen / sum_p                 # 1 - happen_rate/mean_p
        empirical = max(0.0, min(self._cap, empirical))
        w = n / (n + self._k)
        return w * empirical + (1.0 - w) * self._prior

    def summary(self) -> list[dict]:
        out = []
        for (cat, bkt), (happen, n, sum_p) in sorted(self._buckets.items()):
            if n <= 0:
                continue
            out.append({"category": cat, "price_bucket": bkt, "n": int(n),
                        "learned_bias": round(self.bias(cat, bkt), 3)})
        return out


class MarkoutFeedback:
    """Per-market MM spread multiplier from realized markout at a horizon.

    markout < 0 = adverse selection -> widen; >= 0 -> baseline. Multiplier in
    [1.0, max_mult]. Needs >= min_n fills before it moves off 1.0.
    """

    def __init__(self, scale: float = 0.01, max_mult: float = 2.0, min_n: int = 5):
        self._scale = scale
        self._max = max_mult
        self._min_n = min_n
        self._mult: dict[str, float] = {}

    def fit(self, markout_by_market: dict[str, tuple[float, int]]) -> "MarkoutFeedback":
        self._mult = {}
        for market_id, (avg_markout, n) in markout_by_market.items():
            if n < self._min_n or avg_markout >= 0:
                continue
            widen = min(-avg_markout / self._scale, self._max - 1.0)
            self._mult[market_id] = 1.0 + widen
        return self

    def multiplier(self, market_id: str) -> float:
        return self._mult.get(market_id, 1.0)


class CorrelationLearner:
    """Pairwise category correlations from the tick store's cat_index series.

    Input series are per-category (ts, value) rows written once per cycle
    (value = volume-weighted mean daily price change of the category). Pairs
    are aligned on shared timestamps; Pearson correlation is computed only
    when a pair has >= min_samples aligned points, and clamped away from ±1
    (a learned 1.0 would make VaR degenerate). Categories without enough
    history simply stay on the expert prior — fit() returns only what it
    actually learned.
    """

    def __init__(self, min_samples: int = 50, clamp: float = 0.95):
        self._min = min_samples
        self._clamp = clamp
        self.learned: dict[frozenset, float] = {}

    def fit(self, series: dict[str, list[tuple[float, float]]]) -> "CorrelationLearner":
        self.learned = {}
        cats = [c for c, rows in series.items() if len(rows) >= self._min]
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                a, b = cats[i], cats[j]
                rho = self._pearson_aligned(dict(series[a]), dict(series[b]))
                if rho is not None:
                    self.learned[frozenset({a, b})] = max(-self._clamp,
                                                          min(self._clamp, rho))
        return self

    def _pearson_aligned(self, a: dict[float, float],
                         b: dict[float, float]) -> float | None:
        shared = sorted(set(a) & set(b))
        if len(shared) < self._min:
            return None
        xs = [a[t] for t in shared]
        ys = [b[t] for t in shared]
        n = float(len(shared))
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        syy = sum((y - my) ** 2 for y in ys)
        if sxx <= 0 or syy <= 0:
            return None       # a flat series correlates with nothing
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        return sxy / math.sqrt(sxx * syy)


class FillCalibrator:
    """Binned predicted-vs-realized calibration of the fill-probability model.

    The microstructure fill model is a formula; this checks it against what
    actually happened to OUR quotes (filled before cancel, or not) and maps a
    predicted probability to the realized rate of its bin. Bins with fewer
    than min_per_bin outcomes fall back to the raw prediction — the map only
    speaks where it has evidence.
    """

    BIN_EDGES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)

    def __init__(self, min_per_bin: int = 20):
        self._min = min_per_bin
        self._rate: dict[int, tuple[float, int]] = {}   # bin -> (realized, n)

    def _bin(self, p: float) -> int:
        for i, edge in enumerate(self.BIN_EDGES):
            if p < edge:
                return i
        return len(self.BIN_EDGES) - 1

    def fit(self, outcomes: list[tuple[float, bool]]) -> "FillCalibrator":
        """outcomes: (predicted p_fill at placement, filled before cancel?)."""
        acc: dict[int, list[float]] = defaultdict(list)
        for p, filled in outcomes:
            if 0.0 <= p <= 1.0:
                acc[self._bin(p)].append(1.0 if filled else 0.0)
        self._rate = {b: (sum(v) / len(v), len(v)) for b, v in acc.items()}
        return self

    def calibrate(self, p: float) -> float:
        rate = self._rate.get(self._bin(p))
        if rate is None or rate[1] < self._min:
            return p
        return rate[0]

    def summary(self) -> list[dict]:
        return [{"bin": b, "realized": round(r, 3), "n": n}
                for b, (r, n) in sorted(self._rate.items())]


class PlattCalibrator:
    """1-D logistic recalibration of p_est: sigmoid(a*logit(p_est)+b).

    Fit on (p_est, outcome) pairs by gradient descent. Identity until fit with
    enough data; corrects systematic over/under-confidence walk-forward.
    """

    def __init__(self, min_samples: int = 30):
        self._a = 1.0
        self._b = 0.0
        self._fitted = False
        self._min = min_samples

    @staticmethod
    def _logit(p: float) -> float:
        p = min(max(p, 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))

    def fit(self, pairs: list[tuple[float, float]], iters: int = 500,
            lr: float = 0.1) -> "PlattCalibrator":
        pairs = [(p, y) for p, y in pairs if p is not None]
        if len(pairs) < self._min:
            self._fitted = False
            return self
        a, b = 1.0, 0.0
        xs = [(self._logit(p), y) for p, y in pairs]
        n = len(xs)
        for _ in range(iters):
            ga = gb = 0.0
            for x, y in xs:
                pred = 1.0 / (1.0 + math.exp(-(a * x + b)))
                err = pred - y
                ga += err * x
                gb += err
            a -= lr * ga / n
            b -= lr * gb / n
        self._a, self._b, self._fitted = a, b, True
        return self

    def calibrate(self, p_est: float) -> float:
        if not self._fitted:
            return p_est
        z = self._a * self._logit(p_est) + self._b
        return 1.0 / (1.0 + math.exp(-z))
