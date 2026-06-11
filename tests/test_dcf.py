"""Тесты ядра DCF: IRR, NPV, окупаемость, граничные значения."""

import numpy as np
import pytest

from finance.dcf import DCFInputs, compute_dcf, irr, npv


def test_npv_zero_at_irr():
    # При ставке, равной IRR, NPV должна обращаться в ноль.
    cf = np.array([-100.0, 110.0])
    assert npv(0.10, cf) == pytest.approx(0.0, abs=1e-9)


def test_irr_simple_one_period():
    # -100 -> 110 за один период: IRR = 10 %.
    cf = np.array([-100.0, 110.0])
    assert irr(cf) == pytest.approx(0.10, abs=1e-6)


def test_irr_known_multiperiod():
    # Классический поток: -1000, затем 5×250. IRR ≈ 7.93 %.
    cf = np.array([-1000.0, 250.0, 250.0, 250.0, 250.0, 250.0])
    rate = irr(cf)
    # Проверяем через определение: NPV(rate) == 0.
    assert npv(rate, cf) == pytest.approx(0.0, abs=1e-6)
    assert rate == pytest.approx(0.0793, abs=1e-3)


def test_irr_no_sign_change_returns_nan():
    # Только притоки — IRR не определён.
    cf = np.array([100.0, 100.0, 100.0])
    assert np.isnan(irr(cf))


def _base_inputs(**overrides) -> DCFInputs:
    base = dict(
        price_rub=10_000_000,
        schedule=[{"month": 0, "pct": 100}],
        rental_rate_rub_month=80_000,
        occupancy_pct=80,
        opex_pct_of_revenue=25,
        rental_growth_pct=5,
        rent_start_month=0,
        hold_years=5,
        exit_cap_rate_pct=9,
        selling_cost_pct=3,
        discount_rate_pct=15,
    )
    base.update(overrides)
    return DCFInputs(**base)


def test_compute_dcf_basic_metrics():
    res = compute_dcf(_base_inputs())
    # Полный единовременный платёж = цене объекта.
    assert res.total_invested == pytest.approx(10_000_000, rel=1e-9)
    # Должны быть притоки (аренда + выход) и положительный мультипликатор.
    assert res.total_distributions > 0
    assert res.moic > 1.0
    # IRR определён.
    assert not np.isnan(res.irr_annual)
    # NPV конечен.
    assert np.isfinite(res.npv)


def test_npv_sign_matches_irr_vs_discount():
    # Если IRR выше ставки дисконтирования — NPV положительна, и наоборот.
    res = compute_dcf(_base_inputs(discount_rate_pct=10))
    if res.irr_annual > 10:
        assert res.npv > 0
    else:
        assert res.npv < 0


def test_installment_timing_shifts_irr():
    # Рассрочка (поздние платежи) должна давать IRR не ниже,
    # чем единовременная оплата всей суммы на входе — деньги работают дольше.
    upfront = compute_dcf(_base_inputs(schedule=[{"month": 0, "pct": 100}]))
    staged = compute_dcf(_base_inputs(
        schedule=[{"month": 0, "pct": 40}, {"month": 12, "pct": 30}, {"month": 24, "pct": 30}]
    ))
    assert staged.irr_annual > upfront.irr_annual


def test_exit_price_growth_path():
    # Альтернативный выход через рост цены вместо cap rate.
    res = compute_dcf(_base_inputs(exit_cap_rate_pct=0, exit_price_growth_pct=8))
    # TV = price * (1+0.08)^5
    expected_tv = 10_000_000 * 1.08 ** 5
    assert res.terminal_value_gross == pytest.approx(expected_tv, rel=1e-9)


def test_payback_returns_value_or_none():
    res = compute_dcf(_base_inputs())
    # Окупаемость либо число лет в пределах горизонта, либо None.
    if res.payback_year is not None:
        assert 0 <= res.payback_year <= 5


def test_yields_year_one():
    res = compute_dcf(_base_inputs())
    # Валовая доходность = выручка года 1 / цена.
    expected_gross = (80_000 * 12 * 0.8) / 10_000_000 * 100
    assert res.gross_yield_pct == pytest.approx(expected_gross, rel=1e-9)
    # Чистая доходность ниже валовой (есть OPEX).
    assert res.net_yield_pct < res.gross_yield_pct


def test_tranche_beyond_horizon_raises():
    # Транш за горизонтом удержания — явная ошибка, не молчаливая потеря потока.
    with pytest.raises(ValueError):
        compute_dcf(_base_inputs(
            hold_years=2,
            schedule=[{"month": 0, "pct": 50}, {"month": 36, "pct": 50}],
        ))
