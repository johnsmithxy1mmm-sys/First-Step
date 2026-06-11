"""Тесты форматирования ru-RU."""

import math

from finance.format_ru import (
    format_multiple,
    format_num,
    format_pct,
    format_rub,
)

NBSP = chr(0x202F)  # узкий неразрывный пробел (U+202F)


def test_thousands_grouping():
    assert format_num(18_900_000) == f"18{NBSP}900{NBSP}000"


def test_decimal_comma():
    assert format_num(15.24, 1) == "15,2"


def test_rub_suffix():
    assert format_rub(18_900_000) == f"18{NBSP}900{NBSP}000{NBSP}₽"


def test_pct_one_digit():
    assert format_pct(15) == f"15,0{NBSP}%"


def test_multiple_suffix():
    assert format_multiple(1.823) == "1,82×"


def test_nan_renders_dash():
    assert format_rub(math.nan) == "—"
    assert format_pct(math.nan) == "—"
    assert format_num(math.nan) == "—"


def test_negative_grouping():
    assert format_num(-1_234_567) == f"-1{NBSP}234{NBSP}567"
