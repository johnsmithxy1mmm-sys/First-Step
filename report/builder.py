"""Сборка PDF-меморандума из конфига объекта.

Конвейер: ObjectConfig → расчёты (finance) → графики (charts) → контекст →
Jinja2 (HTML) → Playwright (PDF). Тексты-интерпретации формируются из чисел
строго сдержанным регистром, без рекламных формулировок.
"""

from __future__ import annotations

import base64
import datetime as _dt
import re
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from config.schema import ObjectConfig
from config.loader import to_dcf_inputs
from finance.dcf import compute_dcf, DCFResult
from finance.installment import build_installment
from finance.monte_carlo import run_monte_carlo, MonteCarloResult
from finance.sensitivity import run_sensitivity
from finance.format_ru import format_rub, format_pct, format_multiple, format_num

from report import charts

_HERE = Path(__file__).resolve().parent
_TEMPLATES = _HERE / "templates"
_ASSETS = _HERE / "assets"

# Подписи штампа рекомендации (текст и CSS-класс цвета).
_STAMP = {
    "ACQUIRE": ("Приобретать", "acquire"),
    "HOLD": ("Удерживать", "hold"),
    "AVOID": ("Воздержаться", "avoid"),
}

_MONTHS_RU = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
]


def _date_str(date: _dt.date) -> str:
    return f"{date.day} {_MONTHS_RU[date.month - 1]} {date.year}"


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _sign_class(value: float) -> str:
    """CSS-класс по знаку: положительное -> pos, отрицательное -> neg."""
    return "pos" if value >= 0 else "neg"


def _payback_str(years: Optional[float]) -> str:
    if years is None:
        return "вне горизонта"
    return f"{format_num(years, 1)} г."


# --------------------------------------------------------------------------- #
# Тексты-интерпретации (сдержанный регистр, выводятся из чисел)
# --------------------------------------------------------------------------- #
def _summary_interpretation(cfg: ObjectConfig, dcf: DCFResult, mc: MonteCarloResult) -> str:
    target = cfg.assumptions.target_irr_pct
    disc = cfg.assumptions.discount_rate_pct
    parts = []
    if dcf.irr_annual >= target:
        parts.append(f"Базовый IRR {format_pct(dcf.irr_annual)} превышает целевой порог {format_pct(target)}.")
    else:
        parts.append(f"Базовый IRR {format_pct(dcf.irr_annual)} ниже целевого порога {format_pct(target)}.")
    if dcf.npv >= 0:
        parts.append(f"При ставке дисконтирования {format_pct(disc)} NPV положительна ({format_rub(dcf.npv)}).")
    else:
        parts.append(f"При требуемой доходности {format_pct(disc)} NPV отрицательна ({format_rub(dcf.npv)}): "
                     f"проект не покрывает заявленную ставку дисконтирования.")
    parts.append(f"Вероятность недобора целевого IRR — {format_pct(mc.prob_below_target)}, "
                 f"вероятность убытка — {format_pct(mc.prob_below_zero)}.")
    return " ".join(parts)


def _mc_interpretation(cfg: ObjectConfig, mc: MonteCarloResult) -> str:
    target = cfg.assumptions.target_irr_pct
    parts = [
        f"Медианный IRR — {format_pct(mc.p50)}, межквартильная оценка риска лежит между "
        f"{format_pct(mc.p5)} (P5) и {format_pct(mc.p95)} (P95)."
    ]
    if mc.prob_below_zero < 5:
        parts.append(f"Вероятность убытка незначительна ({format_pct(mc.prob_below_zero)}).")
    elif mc.prob_below_zero < 25:
        parts.append(f"Вероятность убытка умеренная ({format_pct(mc.prob_below_zero)}).")
    else:
        parts.append(f"Вероятность убытка существенна ({format_pct(mc.prob_below_zero)}).")
    parts.append(f"Целевой порог {format_pct(target)} не достигается в {format_pct(mc.prob_below_target)} исходов.")
    return " ".join(parts)


def _sensitivity_interpretation(bars) -> str:
    if not bars:
        return "Стохастические входы не заданы."
    top = bars[0]
    parts = [f"Наибольший вклад в неопределённость IRR вносит допущение «{top.label}» "
             f"(размах {format_pct(top.swing)})."]
    if len(bars) > 1:
        names = ", ".join(f"«{b.label}»" for b in bars[1:])
        parts.append(f"Далее по влиянию: {names}.")
    parts.append("Управление этими параметрами даёт основной рычаг по риску сделки.")
    return " ".join(parts)


def _exit_interpretation(cfg: ObjectConfig, dcf: DCFResult) -> str:
    price = cfg.purchase.price_rub
    multiple = dcf.terminal_value_gross / price
    if cfg.exit.exit_price_growth_pct and cfg.exit.exit_price_growth_pct > 0:
        method = (f"Оценка выхода построена на приросте стоимости объекта "
                  f"({format_pct(cfg.exit.exit_price_growth_pct)} в год).")
    else:
        method = (f"Оценка выхода построена доходным подходом "
                  f"(капитализация по ставке {format_pct(cfg.exit.exit_cap_rate_pct)}).")
    return (f"{method} Терминальная стоимость на горизонте {cfg.exit.hold_years} лет — "
            f"{format_rub(dcf.terminal_value_gross)} ({format_multiple(multiple)} к цене покупки); "
            f"за вычетом издержек продажи {format_pct(cfg.exit.selling_cost_pct)} чистые поступления — "
            f"{format_rub(dcf.exit_proceeds_net)}.")


# --------------------------------------------------------------------------- #
# Сборка контекста
# --------------------------------------------------------------------------- #
def _build_context(cfg: ObjectConfig, date: _dt.date) -> dict:
    inp = to_dcf_inputs(cfg)
    dcf = compute_dcf(inp)
    ranges = {k: v.model_dump() for k, v in cfg.monte_carlo.ranges.items()}
    mc = run_monte_carlo(inp, ranges, cfg.monte_carlo.iterations,
                         cfg.assumptions.target_irr_pct, cfg.monte_carlo.seed)
    bars = run_sensitivity(inp, ranges)
    tranches = build_installment(cfg.purchase.price_rub,
                                 [{"month": t.month, "pct": t.pct} for t in cfg.purchase.installment.schedule])

    stamp_label, stamp_class = _STAMP[cfg.advisor.recommendation]

    # Графики.
    waterfall_svg = charts.cash_flow_waterfall(dcf)
    histogram_svg = charts.irr_histogram(mc)
    tornado_svg = charts.tornado(bars, dcf.irr_annual)
    installment_svg = charts.installment_timeline(
        [t.month for t in tranches], [t.amount_rub for t in tranches], cfg.purchase.price_rub)

    # Таблица параметров.
    price = cfg.purchase.price_rub
    price_per_m2 = price / cfg.object.area_m2
    if cfg.exit.exit_price_growth_pct and cfg.exit.exit_price_growth_pct > 0:
        exit_method_row = ("Метод выхода", f"Прирост цены {format_pct(cfg.exit.exit_price_growth_pct)}/год")
    else:
        exit_method_row = ("Метод выхода", f"Cap rate {format_pct(cfg.exit.exit_cap_rate_pct)}")

    params = [
        {"title": "Объект", "rows": [
            ("Застройщик", cfg.object.developer or "—"),
            ("Микрозона", cfg.object.zone or "—"),
            ("Тип", cfg.object.type or "—"),
            ("Площадь", f"{format_num(cfg.object.area_m2, 0)} м²"),
        ]},
        {"title": "Покупка", "rows": [
            ("Цена объекта", format_rub(price)),
            ("Цена за м²", format_rub(price_per_m2)),
            ("Первый взнос", format_pct(cfg.purchase.installment.down_payment_pct)
                if cfg.purchase.installment.down_payment_pct is not None else "—"),
            ("Траншей рассрочки", str(len(cfg.purchase.installment.schedule))),
        ]},
        {"title": "Операции", "rows": [
            ("Ставка аренды", f"{format_rub(cfg.operations.rental_rate_rub_month)}/мес"),
            ("Загрузка", format_pct(cfg.operations.occupancy_pct)),
            ("Доля OPEX", format_pct(cfg.operations.opex_pct_of_revenue)),
            ("Рост ставки аренды", f"{format_pct(cfg.operations.rental_growth_pct)}/год"),
            ("Начало аренды", f"{cfg.operations.rent_start_month} мес." if cfg.operations.rent_start_month else "со старта"),
        ]},
        {"title": "Выход и допущения", "rows": [
            ("Горизонт удержания", f"{cfg.exit.hold_years} лет"),
            exit_method_row,
            ("Издержки продажи", format_pct(cfg.exit.selling_cost_pct)),
            ("Ставка дисконтирования", format_pct(cfg.assumptions.discount_rate_pct)),
            ("Целевой IRR", format_pct(cfg.assumptions.target_irr_pct)),
        ]},
    ]

    # Строки таблицы DCF.
    dcf_rows = []
    for ln in dcf.annual_lines:
        dcf_rows.append({
            "year": ln.year,
            "revenue": format_rub(ln.revenue),
            "opex": format_rub(-ln.opex),
            "noi": format_rub(ln.noi),
            "installment": format_rub(ln.installment) if ln.installment != 0 else "—",
            "installment_class": "neg-num" if ln.installment < 0 else "",
            "exit": format_rub(ln.exit_proceeds) if ln.exit_proceeds else "—",
            "net": format_rub(ln.net_cash_flow),
            "net_class": "neg-num" if ln.net_cash_flow < 0 else "",
            "disc": format_rub(ln.discounted_cash_flow),
            "disc_class": "neg-num" if ln.discounted_cash_flow < 0 else "",
        })

    # Строки таблицы рассрочки.
    installment_rows = []
    cum = 0.0
    for t in tranches:
        cum += t.pct
        installment_rows.append({
            "month": t.month,
            "pct": format_pct(t.pct),
            "amount": format_rub(t.amount_rub),
            "cum": format_pct(cum),
        })

    # Строки таблицы выхода.
    exit_rows = [
        ("Горизонт удержания", f"{cfg.exit.hold_years} лет"),
        ("Терминальная стоимость", format_rub(dcf.terminal_value_gross)),
        ("Мультипликатор к цене", format_multiple(dcf.terminal_value_gross / price)),
        ("Издержки продажи", format_pct(cfg.exit.selling_cost_pct)),
        ("Чистые поступления", format_rub(dcf.exit_proceeds_net)),
    ]

    return {
        "fonts": {
            "lora": _b64(_ASSETS / "fonts" / "Lora.ttf"),
            "lora_italic": _b64(_ASSETS / "fonts" / "Lora-Italic.ttf"),
        },
        "brand_css": (_ASSETS / "brand.css").read_text(encoding="utf-8"),
        "date_str": _date_str(date),
        "obj": {
            "name": cfg.object.name,
            "developer": cfg.object.developer,
            "zone": cfg.object.zone,
            "type": cfg.object.type,
            "area_str": f"{format_num(cfg.object.area_m2, 0)} м²",
        },
        "advisor": {
            "name": cfg.advisor.name,
            "title": cfg.advisor.title,
            "recommendation": cfg.advisor.recommendation,
            "stamp_label": stamp_label,
            "stamp_class": stamp_class,
            "thesis": cfg.advisor.thesis,
            "exit_scenario": cfg.advisor.exit_scenario,
        },
        "kpi": {
            "irr": format_pct(dcf.irr_annual),
            "irr_class": _sign_class(dcf.irr_annual),
            "npv": format_rub(dcf.npv),
            "npv_class": _sign_class(dcf.npv),
            "discount": format_pct(cfg.assumptions.discount_rate_pct),
            "moic": format_multiple(dcf.moic),
            "prob_loss": format_pct(mc.prob_below_zero),
        },
        "summary": {"interpretation": _summary_interpretation(cfg, dcf, mc)},
        "params": params,
        "dcf_rows": dcf_rows,
        "dcf": {
            "npv": format_rub(dcf.npv),
            "npv_class": "neg-num" if dcf.npv < 0 else "",
            "irr": format_pct(dcf.irr_annual),
            "payback": _payback_str(dcf.payback_year),
            "disc_payback": _payback_str(dcf.discounted_payback_year),
            "gross_yield": format_pct(dcf.gross_yield_pct),
            "net_yield": format_pct(dcf.net_yield_pct),
            "exit_net": format_rub(dcf.exit_proceeds_net),
        },
        "mc": {
            "iterations": format_num(mc.iterations),
            "target": format_pct(mc.target_irr_pct),
            "p_band": f"{format_pct(mc.p5)} · {format_pct(mc.p50)} · {format_pct(mc.p95)}",
            "prob_below_target": format_pct(mc.prob_below_target),
            "prob_below_zero": format_pct(mc.prob_below_zero),
            "below_target_class": "warn" if mc.prob_below_target >= 50 else "",
            "below_zero_class": "warn" if mc.prob_below_zero >= 10 else "",
            "interpretation": _mc_interpretation(cfg, mc),
        },
        "sensitivity": {"interpretation": _sensitivity_interpretation(bars)},
        "installment_rows": installment_rows,
        "exit_rows": exit_rows,
        "exit_block": {"interpretation": _exit_interpretation(cfg, dcf)},
        "charts": {
            "waterfall": waterfall_svg,
            "histogram": histogram_svg,
            "tornado": tornado_svg,
            "installment": installment_svg,
        },
    }


# --------------------------------------------------------------------------- #
# Рендер HTML и PDF
# --------------------------------------------------------------------------- #
def render_html(cfg: ObjectConfig, date: _dt.date) -> str:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    # SVG-графики вставляются как есть (не экранируются) — отметим safe в шаблоне?
    # Проще: отключаем автоэкранирование для полей charts через |safe в шаблоне.
    template = env.get_template("memorandum.html.j2")
    context = _build_context(cfg, date)
    return template.render(**context)


def _normalize_pdf_dates(pdf_bytes: bytes, date: _dt.date) -> bytes:
    """Заменяет временные метки PDF на фиксированную дату той же длины.

    Chromium штампует /CreationDate и /ModDate текущим временем — это ломает
    байт-в-байт воспроизводимость. Подменяем их детерминированной меткой
    (полдень даты меморандума), сохраняя длину строки, чтобы не сдвигать
    смещения объектов PDF.
    """
    fixed = f"D:{date.strftime('%Y%m%d')}120000+00'00'".encode("ascii")

    def repl(match: re.Match) -> bytes:
        key = match.group(1)
        original = match.group(2)
        # Подгоняем длину под исходную: дополняем нулями или обрезаем.
        value = fixed
        if len(value) < len(original):
            value = value + b"0" * (len(original) - len(value))
        else:
            value = value[: len(original)]
        return key + b"(" + value + b")"

    return re.sub(rb"(/(?:CreationDate|ModDate)\s?)\(([^)]*)\)", repl, pdf_bytes)


def build_pdf(cfg: ObjectConfig, output_path: str | Path, date: _dt.date) -> Path:
    """Главная функция сборки: рендерит HTML и печатает PDF."""
    from playwright.sync_api import sync_playwright

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html = render_html(cfg, date)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html, wait_until="load")
        # Дожидаемся готовности встроенных шрифтов перед печатью.
        page.evaluate("document.fonts.ready")
        pdf_bytes = page.pdf(
            format="A4",
            print_background=True,
            prefer_css_page_size=True,
        )
        browser.close()

    pdf_bytes = _normalize_pdf_dates(pdf_bytes, date)
    output_path.write_bytes(pdf_bytes)
    return output_path
