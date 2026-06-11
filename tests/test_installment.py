"""Тесты графика рассрочки: суммы, сходимость долей, сортировка."""

import pytest

from finance.installment import build_installment, schedule_pct_sum


def test_amounts_match_pct():
    schedule = [{"month": 0, "pct": 30}, {"month": 12, "pct": 70}]
    tranches = build_installment(20_000_000, schedule)
    assert tranches[0].amount_rub == pytest.approx(6_000_000)
    assert tranches[1].amount_rub == pytest.approx(14_000_000)


def test_total_equals_price_when_sum_100():
    schedule = [{"month": 0, "pct": 30}, {"month": 12, "pct": 35}, {"month": 24, "pct": 35}]
    tranches = build_installment(18_900_000, schedule)
    total = sum(t.amount_rub for t in tranches)
    assert total == pytest.approx(18_900_000)


def test_schedule_sorted_by_month():
    schedule = [{"month": 24, "pct": 35}, {"month": 0, "pct": 30}, {"month": 12, "pct": 35}]
    tranches = build_installment(10_000_000, schedule)
    months = [t.month for t in tranches]
    assert months == sorted(months)


def test_pct_sum_helper():
    schedule = [{"month": 0, "pct": 30}, {"month": 12, "pct": 35}, {"month": 24, "pct": 35}]
    assert schedule_pct_sum(schedule) == pytest.approx(100.0)
