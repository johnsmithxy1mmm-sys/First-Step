"""Counters and latency histograms (§7).

Deliberately dependency-free: a dict of counters and a list of samples,
exported as plain data. Whatever scrapes this (Prometheus in
`deploy/prometheus.yml`, the health endpoint, a test) reads the snapshot.

The metrics §7 asks for that are *not* here belong to the Node service or to
the shadow harness and are recorded there:
  - refusal-to-execute count on stale data  -> services/backend (§6)
  - orders with no matching button press    -> services/backend (§5.4)
  - shadow progress, PIT/CRPS vs baselines  -> shadow/metrics.py (§3.3)
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

#: Every sample list here is bounded (audit A-11). This process is a
#: long-running service whose matrix rebuild fires every five minutes, so an
#: unbounded list of latency samples or PSD corrections is a slow memory leak
#: with no upper bound. Bounded deques keep the most recent window, which is
#: what percentiles and dashboards actually read; totals live in `counters`,
#: which is bounded by the number of distinct metric names.
MAX_LATENCY_SAMPLES = 4096
MAX_EVENT_SAMPLES = 512


def _bounded() -> deque:
    return deque(maxlen=MAX_LATENCY_SAMPLES)


@dataclass
class Metrics:
    counters: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    latencies_ms: dict[str, deque] = field(default_factory=lambda: defaultdict(_bounded))
    #: Correction magnitudes from the PSD projection (§2.1 wants each logged).
    #: Bounded window; `psd_projection_corrections` in `counters` keeps the
    #: lifetime total so nothing is silently lost from the tally.
    psd_corrections: deque = field(default_factory=lambda: deque(maxlen=MAX_EVENT_SAMPLES))
    #: Every df clamp, with the raw MLE value (§2.2 wants each logged).
    df_clamps: deque = field(default_factory=lambda: deque(maxlen=MAX_EVENT_SAMPLES))

    def incr(self, name: str, by: float = 1.0) -> None:
        self.counters[name] += by

    def observe_latency(self, stage: str, ms: float) -> None:
        self.latencies_ms[stage].append(ms)

    def percentiles(self, stage: str) -> dict[str, float]:
        vals = sorted(self.latencies_ms.get(stage, []))
        if not vals:
            return {}

        def pct(p: float) -> float:
            if len(vals) == 1:
                return vals[0]
            k = (len(vals) - 1) * p
            lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
            return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)

        return {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99), "n": float(len(vals))}

    def snapshot(self) -> dict:
        return {
            "counters": dict(self.counters),
            "latency": {stage: self.percentiles(stage) for stage in self.latencies_ms},
            "psd_corrections": list(self.psd_corrections),
            "df_clamps": list(self.df_clamps),
        }

    def reset(self) -> None:
        self.counters.clear()
        self.latencies_ms.clear()
        self.psd_corrections.clear()
        self.df_clamps.clear()


METRICS = Metrics()


class Timer:
    """`with Timer("slice"):` records one latency sample for that stage.

    §2.6 requires every stage of the pre-trade path to be measured
    separately, not just the total.
    """

    def __init__(self, stage: str, metrics: Metrics | None = None) -> None:
        self.stage = stage
        self.metrics = metrics or METRICS
        self.elapsed_ms = 0.0

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        self.metrics.observe_latency(self.stage, self.elapsed_ms)
