"""Модель дисконтированных денежных потоков (DCF) для одного объекта.

Логика построена на ПОМЕСЯЧНОЙ сетке потоков: это позволяет корректно учесть
тайминг траншей рассрочки (которые могут приходиться на любой месяц), момент
начала арендного дохода и момент выхода. Годовые агрегаты для таблицы отчёта
получаются суммированием месяцев.

Все финансовые допущения вынесены в явные, прокомментированные формулы.
Никакой «магии по памяти»: каждое уравнение читаемо и тестируемо.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .installment import InstallmentTranche, build_installment


# --------------------------------------------------------------------------- #
# Входные параметры модели (уже провалидированы и нормализованы из конфига)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DCFInputs:
    price_rub: float
    schedule: List[dict]

    rental_rate_rub_month: float
    occupancy_pct: float
    opex_pct_of_revenue: float
    rental_growth_pct: float
    rent_start_month: int  # месяц начала арендного дохода (0 = со старта)

    hold_years: int
    exit_cap_rate_pct: float
    selling_cost_pct: float
    # Альтернатива cap-rate: годовой рост цены объекта. Если задан (> 0),
    # терминальная стоимость считается от цены, а не от дохода.
    exit_price_growth_pct: Optional[float] = None

    discount_rate_pct: float = 0.0


# --------------------------------------------------------------------------- #
# Результат расчёта
# --------------------------------------------------------------------------- #
@dataclass
class AnnualLine:
    """Строка годовой таблицы DCF."""

    year: int
    revenue: float          # эффективная выручка (аренда × загрузка)
    opex: float             # операционные расходы
    noi: float              # чистый операционный доход
    installment: float      # транши рассрочки за год (отрицательные)
    exit_proceeds: float    # чистые поступления от продажи (только год выхода)
    net_cash_flow: float    # итоговый поток года
    discount_factor: float  # коэффициент дисконтирования (на конец года)
    discounted_cash_flow: float


@dataclass
class DCFResult:
    monthly_cash_flows: np.ndarray   # помесячные потоки, индекс = месяц
    annual_lines: List[AnnualLine] = field(default_factory=list)

    npv: float = 0.0
    irr_annual: float = 0.0          # IRR в годовом выражении, в процентах
    moic: float = 0.0                # денежный мультипликатор (MOIC)
    payback_year: Optional[float] = None            # простая окупаемость, лет
    discounted_payback_year: Optional[float] = None  # дисконтированная, лет

    gross_yield_pct: float = 0.0     # валовая доходность аренды (год 1)
    net_yield_pct: float = 0.0       # чистая доходность аренды (год 1)

    total_invested: float = 0.0      # сумма всех оттоков
    total_distributions: float = 0.0  # сумма всех притоков
    terminal_value_gross: float = 0.0
    exit_proceeds_net: float = 0.0


# --------------------------------------------------------------------------- #
# Базовые финансовые примитивы
# --------------------------------------------------------------------------- #
def npv(rate_period: float, cash_flows: np.ndarray) -> float:
    """Чистая приведённая стоимость серии потоков при ставке за период.

    Поток с индексом t дисконтируется на t периодов: CF_t / (1 + rate)^t.
    """
    periods = np.arange(len(cash_flows))
    # При брекетинге IRR ставка может уходить в большие значения — глушим
    # безвредные предупреждения переполнения, результат остаётся корректным.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return float(np.sum(cash_flows / (1.0 + rate_period) ** periods))


def irr(cash_flows: np.ndarray, guess_bracket=(-0.9999, 1.0)) -> float:
    """Внутренняя норма доходности за период методом бисекции.

    Возвращает ставку за ОДИН период (для помесячных потоков — месячную).
    Метод устойчив: ищем смену знака NPV на расширяющемся интервале, затем
    уточняем бисекцией. Для типичного инвестиционного профиля (ранние оттоки,
    поздние притоки) корень единственный.

    Если суммарный поток не меняет знак (нет ни оттоков, ни притоков, либо
    проект не окупается ни при какой ставке) — возвращает NaN.
    """
    cash_flows = np.asarray(cash_flows, dtype=float)
    if not (np.any(cash_flows > 0) and np.any(cash_flows < 0)):
        return float("nan")

    lo, hi = guess_bracket
    f_lo = npv(lo, cash_flows)
    f_hi = npv(hi, cash_flows)

    # Расширяем верхнюю границу, пока не поймаем смену знака.
    expand = 0
    while f_lo * f_hi > 0 and expand < 200:
        hi *= 1.5
        f_hi = npv(hi, cash_flows)
        expand += 1
    if f_lo * f_hi > 0:
        return float("nan")  # корень не локализован

    # Бисекция до сходимости по ставке.
    for _ in range(200):
        mid = (lo + hi) / 2.0
        f_mid = npv(mid, cash_flows)
        if abs(f_mid) < 1e-9 or (hi - lo) < 1e-12:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2.0


# --------------------------------------------------------------------------- #
# Построение потоков и расчёт модели
# --------------------------------------------------------------------------- #
def _noi_for_year(inp: DCFInputs, year: int) -> float:
    """Чистый операционный доход за год year (1..hold_years).

    Выручка = базовая ставка × 12 × загрузка × (1 + рост)^(year-1).
    Рост применяется со второго года (год 1 — базовый).
    NOI = выручка × (1 − доля opex).
    """
    occupancy = inp.occupancy_pct / 100.0
    growth = inp.rental_growth_pct / 100.0
    opex_share = inp.opex_pct_of_revenue / 100.0

    gross_potential = inp.rental_rate_rub_month * 12.0
    revenue = gross_potential * occupancy * (1.0 + growth) ** (year - 1)
    return revenue * (1.0 - opex_share)


def _revenue_for_year(inp: DCFInputs, year: int) -> float:
    occupancy = inp.occupancy_pct / 100.0
    growth = inp.rental_growth_pct / 100.0
    gross_potential = inp.rental_rate_rub_month * 12.0
    return gross_potential * occupancy * (1.0 + growth) ** (year - 1)


def _terminal_value(inp: DCFInputs) -> float:
    """Терминальная (продажная) стоимость на момент выхода, ДО издержек продажи.

    Основной путь — капитализация дохода: берём NOI года, следующего за годом
    выхода (forward NOI), и делим на exit cap rate. Это стандартная оценка
    «доходным подходом»: покупатель платит за будущий поток.

    Альтернатива — если задан exit_price_growth_pct: цена объекта растёт от
    цены покупки сложным процентом на горизонте удержания.
    """
    if inp.exit_price_growth_pct is not None and inp.exit_price_growth_pct > 0:
        growth = inp.exit_price_growth_pct / 100.0
        return inp.price_rub * (1.0 + growth) ** inp.hold_years

    # Forward NOI = NOI следующего после выхода года.
    forward_noi = _noi_for_year(inp, inp.hold_years + 1)
    cap = inp.exit_cap_rate_pct / 100.0
    return forward_noi / cap


def build_monthly_cash_flows(inp: DCFInputs):
    """Строит помесячный вектор потоков и список траншей рассрочки.

    Индекс 0 — момент сделки. Длина = hold_years * 12 + 1 (включая месяц выхода).
    Возвращает (cash_flows, tranches, terminal_gross, exit_net).
    """
    months = inp.hold_years * 12
    cash_flows = np.zeros(months + 1, dtype=float)

    # 1) Транши рассрочки — отрицательные потоки в свои месяцы.
    tranches = build_installment(inp.price_rub, inp.schedule)
    for tr in tranches:
        if tr.month > months:
            # Транш позже горизонта удержания — клиент платит после выхода.
            # Это бессмысленно для модели; валидатор такого не пропустит,
            # но на всякий случай не теряем поток молча.
            raise ValueError(
                f"Транш рассрочки в месяце {tr.month} выходит за горизонт "
                f"удержания ({months} мес.). Сократите рассрочку или увеличьте hold_years."
            )
        cash_flows[tr.month] -= tr.amount_rub

    # 2) Арендный NOI — равномерно по месяцам внутри каждого года, начиная с
    #    rent_start_month. NOI года распределяется на 12 месяцев этого года.
    for year in range(1, inp.hold_years + 1):
        monthly_noi = _noi_for_year(inp, year) / 12.0
        first_month = (year - 1) * 12 + 1  # месяцы 1..12 -> год 1, и т.д.
        last_month = year * 12
        for m in range(first_month, last_month + 1):
            if m >= inp.rent_start_month and m <= months:
                cash_flows[m] += monthly_noi

    # 3) Выход — чистые поступления от продажи в последний месяц.
    terminal_gross = _terminal_value(inp)
    exit_net = terminal_gross * (1.0 - inp.selling_cost_pct / 100.0)
    cash_flows[months] += exit_net

    return cash_flows, tranches, terminal_gross, exit_net


def _annual_aggregate(inp: DCFInputs, cash_flows: np.ndarray, monthly_rate: float,
                      tranches: List[InstallmentTranche], exit_net: float) -> List[AnnualLine]:
    """Сворачивает помесячные потоки в годовые строки для таблицы DCF."""
    lines: List[AnnualLine] = []
    annual_discount = inp.discount_rate_pct / 100.0
    for year in range(1, inp.hold_years + 1):
        first_month = (year - 1) * 12 + 1
        last_month = year * 12

        revenue = _revenue_for_year(inp, year)
        noi = _noi_for_year(inp, year)
        opex = revenue - noi

        installment_year = -sum(
            tr.amount_rub for tr in tranches if first_month <= tr.month <= last_month
        )
        exit_year = exit_net if year == inp.hold_years else 0.0

        net = float(np.sum(cash_flows[first_month:last_month + 1]))
        # Дисконтируем годовой поток к концу года (год как период).
        df = 1.0 / (1.0 + annual_discount) ** year
        lines.append(AnnualLine(
            year=year,
            revenue=revenue,
            opex=opex,
            noi=noi,
            installment=installment_year,
            exit_proceeds=exit_year,
            net_cash_flow=net,
            discount_factor=df,
            discounted_cash_flow=net * df,
        ))
    return lines


def _payback(net_by_year: List[float], initial_month0: float) -> Optional[float]:
    """Год, в котором накопленный поток впервые становится неотрицательным.

    initial_month0 — поток в момент 0 (как правило, первый транш, < 0).
    Возвращает дробное число лет (линейная интерполяция внутри года) или None.
    """
    cumulative = initial_month0
    if cumulative >= 0:
        return 0.0
    for idx, net in enumerate(net_by_year, start=1):
        prev = cumulative
        cumulative += net
        if cumulative >= 0:
            # Линейная интерполяция: какая доля года понадобилась.
            fraction = (-prev) / net if net != 0 else 0.0
            return (idx - 1) + fraction
    return None


def compute_dcf(inp: DCFInputs) -> DCFResult:
    """Главная функция: считает потоки, NPV, IRR и производные метрики."""
    cash_flows, tranches, terminal_gross, exit_net = build_monthly_cash_flows(inp)

    # Месячная ставка дисконтирования из годовой (эквивалентная капитализация).
    annual_rate = inp.discount_rate_pct / 100.0
    monthly_rate = (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0

    result = DCFResult(monthly_cash_flows=cash_flows)
    result.npv = npv(monthly_rate, cash_flows)

    # IRR считаем по месячной серии, затем переводим в годовую.
    monthly_irr = irr(cash_flows)
    result.irr_annual = (
        ((1.0 + monthly_irr) ** 12 - 1.0) * 100.0
        if monthly_irr == monthly_irr  # not NaN
        else float("nan")
    )

    result.annual_lines = _annual_aggregate(inp, cash_flows, monthly_rate, tranches, exit_net)
    result.terminal_value_gross = terminal_gross
    result.exit_proceeds_net = exit_net

    # Притоки/оттоки и MOIC.
    inflows = float(np.sum(cash_flows[cash_flows > 0]))
    outflows = float(-np.sum(cash_flows[cash_flows < 0]))
    result.total_distributions = inflows
    result.total_invested = outflows
    result.moic = inflows / outflows if outflows > 0 else float("nan")

    # Доходность аренды (год 1) — от полной цены объекта.
    result.gross_yield_pct = _revenue_for_year(inp, 1) / inp.price_rub * 100.0
    result.net_yield_pct = _noi_for_year(inp, 1) / inp.price_rub * 100.0

    # Окупаемость: накапливаем годовые потоки (без месяца 0, он учтён отдельно).
    net_by_year = [ln.net_cash_flow for ln in result.annual_lines]
    # Поток месяца 0 не входит в year=1 (год 1 = месяцы 1..12), учтём его как старт.
    month0 = float(cash_flows[0])
    result.payback_year = _payback(net_by_year, month0)

    disc_by_year = [ln.discounted_cash_flow for ln in result.annual_lines]
    result.discounted_payback_year = _payback(disc_by_year, month0)

    return result
