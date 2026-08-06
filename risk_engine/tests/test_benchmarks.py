"""§3.1 — the Phase 1 acceptance gate, as tests.

These are slow (a few minutes at full path counts) because the tolerances
§3.1 states are only meaningful when Monte Carlo noise is well below them.
They are the gate: §9 says a failing criterion stops the phase and gets
reported, so nothing here is marked xfail or given a looser bound.

Run just this file with:  pytest risk_engine/tests/test_benchmarks.py
"""

from __future__ import annotations

import pytest

from risk_engine.validation import benchmarks as bm

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("fn", bm.ALL_BENCHMARKS, ids=lambda f: f.__name__)
def test_benchmark(fn):
    result = fn()
    assert result.passed, str(result)


def test_closed_form_matches_a_hand_computed_case():
    """Guards the reference itself: with zero drift the first-passage
    probability of a driftless Brownian motion to a barrier is exactly twice
    the normal tail at that barrier."""
    from scipy import stats

    got = bm.gbm_first_passage(spot=100.0, barrier=90.0, step_vol=0.01, n_steps=24)
    import numpy as np

    want = 2 * stats.norm.cdf(np.log(0.9) / (0.01 * np.sqrt(24)))
    assert got == pytest.approx(want, rel=1e-12)


def test_run_all_reports_every_benchmark():
    results = bm.run_all()
    assert len(results) == len(bm.ALL_BENCHMARKS)
    assert all(r.detail for r in results)
