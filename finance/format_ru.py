"""Форматирование чисел в локали ru-RU.

Правила:
- разделитель тысяч  — узкий неразрывный пробел (U+202F), как в типографике;
- десятичный разделитель — запятая;
- денежные суммы — с символом ₽;
- проценты — с одним знаком после запятой по умолчанию.

Все функции — чистые, без состояния, чтобы их можно было звать из шаблона.
"""

from __future__ import annotations

import math

# Узкий неразрывный пробел: не даёт разорвать число при переносе строки.
_THIN_NBSP = " "


def _group_thousands(integer_part: str) -> str:
    """Разбивает строку из цифр на группы по три справа налево."""
    sign = ""
    if integer_part.startswith("-"):
        sign, integer_part = "-", integer_part[1:]
    chunks = []
    while len(integer_part) > 3:
        chunks.insert(0, integer_part[-3:])
        integer_part = integer_part[:-3]
    chunks.insert(0, integer_part)
    return sign + _THIN_NBSP.join(chunks)


def format_num(value: float, digits: int = 0) -> str:
    """Число с разделителями тысяч и запятой в дробной части.

    >>> format_num(18900000)
    '18 900 000'  (пробелы — узкие неразрывные)
    >>> format_num(15.234, 1)
    '15,2'
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    rounded = round(float(value), digits)
    # Нормализуем -0 -> 0, чтобы не печатать «-0».
    if rounded == 0:
        rounded = abs(rounded)
    formatted = f"{rounded:.{digits}f}"
    if "." in formatted:
        int_part, frac_part = formatted.split(".")
        return f"{_group_thousands(int_part)},{frac_part}"
    return _group_thousands(formatted)


def format_rub(value: float, digits: int = 0) -> str:
    """Денежная сумма в рублях: «18 900 000 ₽»."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{format_num(value, digits)}{_THIN_NBSP}₽"


def format_pct(value: float, digits: int = 1) -> str:
    """Процент: «15,0 %». Значение передаётся в процентах (15, а не 0.15)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{format_num(value, digits)}{_THIN_NBSP}%"


def format_multiple(value: float, digits: int = 2) -> str:
    """Денежный мультипликатор: «1,82×»."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{format_num(value, digits)}×"
