"""Тесты Монте-Карло: воспроизводимость по сиду, перцентили, вероятности."""

import numpy as np
import pytest

from finance.dcf import DCFInputs
from finance.monte_carlo import run_monte_carlo


def _base() -> DCFInputs:
    return DCFInputs(
        price_rub=18_900_000,
        schedule=[{"month": 0, "pct": 30}, {"month": 12, "pct": 35}, {"month": 24, "pct": 35}],
        rental_rate_rub_month=95_000,
        occupancy_pct=70,
        opex_pct_of_revenue=28,
        rental_growth_pct=6,
        rent_start_month=24,
        hold_years=7,
        exit_cap_rate_pct=9,
        selling_cost_pct=4,
        discount_rate_pct=18,
    )


_RANGES = {
    "occupancy_pct": {"dist": "triangular", "low": 55, "mode": 70, "high": 82},
    "rental_growth_pct": {"dist": "normal", "mean": 6, "sd": 2},
    "exit_cap_rate_pct": {"dist": "triangular", "low": 8, "mode": 9, "high": 11},
}


def test_reproducible_with_same_seed():
    r1 = run_monte_carlo(_base(), _RANGES, iterations=2000, target_irr_pct=15, seed=42)
    r2 = run_monte_carlo(_base(), _RANGES, iterations=2000, target_irr_pct=15, seed=42)
    assert np.array_equal(r1.irr_samples, r2.irr_samples)
    assert r1.p50 == r2.p50


def test_different_seed_changes_samples():
    r1 = run_monte_carlo(_base(), _RANGES, iterations=2000, target_irr_pct=15, seed=1)
    r2 = run_monte_carlo(_base(), _RANGES, iterations=2000, target_irr_pct=15, seed=2)
    assert not np.array_equal(r1.irr_samples, r2.irr_samples)


def test_percentiles_ordered():
    r = run_monte_carlo(_base(), _RANGES, iterations=5000, target_irr_pct=15, seed=42)
    assert r.p5 <= r.p50 <= r.p95


def test_probabilities_in_range():
    r = run_monte_carlo(_base(), _RANGES, iterations=5000, target_irr_pct=15, seed=42)
    assert 0 <= r.prob_below_target <= 100
    assert 0 <= r.prob_below_zero <= 100
    # Вероятность недобора target не меньше вероятности убытка (target > 0).
    assert r.prob_below_target >= r.prob_below_zero


def test_unknown_field_rejected():
    with pytest.raises(ValueError):
        run_monte_carlo(_base(), {"area_m2": {"dist": "uniform", "low": 1, "high": 2}},
                        iterations=200, target_irr_pct=15, seed=42)
