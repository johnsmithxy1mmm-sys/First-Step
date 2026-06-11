"""Графики для меморандума — строго в бренд-палитре Ink / Paper / Brass.

Каждый график рендерится в инлайновый SVG (вектор, чёткая типографика в PDF).
Тексты выводятся как контуры (svg.fonttype='path') — это делает SVG
самодостаточным и детерминированным: одинаковый вход → байт-в-байт одинаковый
график, без зависимости от системных шрифтов.

Принципы оформления (как research-записка частного банка):
- бумажный фон, без сетки по умолчанию;
- цвета только из палитры: латунь / бордо / шалфей / чернила;
- табличные, спокойные подписи; никакого «дашбордного» вида.
"""

from __future__ import annotations

import io
from typing import List

import matplotlib

matplotlib.use("Agg")  # без дисплея
import matplotlib.pyplot as plt
import numpy as np

from finance.dcf import DCFResult
from finance.monte_carlo import MonteCarloResult
from finance.sensitivity import TornadoBar

# --------------------------------------------------------------------------- #
# Бренд-палитра (зеркало CSS-токенов)
# --------------------------------------------------------------------------- #
PAPER = "#F5F1E8"
CARD = "#FBF8F1"
INK = "#211C18"
INK_SOFT = "#5C544A"
BRASS = "#9A7B4F"
BRASS_DEEP = "#7E6438"
BURGUNDY = "#6E1423"
SAGE = "#4F6151"
LINE = "#DAD2C2"


def _apply_brand_style() -> None:
    """Глобальные настройки matplotlib под айдентику и детерминизм."""
    plt.rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "font.family": "DejaVu Sans",   # чистый гротеск с поддержкой кириллицы
        "font.size": 10,
        "text.color": INK,
        "axes.edgecolor": LINE,
        "axes.labelcolor": INK_SOFT,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "axes.grid": False,             # без сетки по умолчанию
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "path",         # тексты как контуры -> детерминизм
        "svg.hashsalt": "first-step",   # фиксируем соль id-шников в SVG
    })


def _fig_to_svg(fig) -> str:
    """Сохраняет фигуру в инлайновый SVG-строку."""
    buf = io.StringIO()
    # metadata={'Date': None} убирает дату из SVG -> воспроизводимость.
    fig.savefig(buf, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    svg = buf.getvalue()
    # Отрезаем XML-преамбулу и DOCTYPE — оставляем сам <svg>, чтобы вставить инлайн.
    start = svg.find("<svg")
    return svg[start:]


def _thin_axes(ax) -> None:
    """Тонкие латунные оси без лишнего."""
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color(LINE)
    ax.tick_params(length=3, width=0.8)


# --------------------------------------------------------------------------- #
# 1. Гистограмма распределения IRR (Монте-Карло)
# --------------------------------------------------------------------------- #
def irr_histogram(mc: MonteCarloResult) -> str:
    _apply_brand_style()
    fig, ax = plt.subplots(figsize=(7.4, 3.6))

    ax.hist(mc.irr_samples, bins=48, color=BRASS, edgecolor=PAPER, linewidth=0.4, alpha=0.9)
    _thin_axes(ax)

    # Вертикальные отметки: P5, P50, P95, target.
    marks = [
        (mc.p5, BRASS_DEEP, "P5", "--"),
        (mc.p50, INK, "P50 (медиана)", "-"),
        (mc.p95, BRASS_DEEP, "P95", "--"),
        (mc.target_irr_pct, BURGUNDY, "Цель", ":"),
    ]
    ymax = ax.get_ylim()[1]
    for value, color, label, style in marks:
        ax.axvline(value, color=color, linestyle=style, linewidth=1.3)
        ax.text(value, ymax * 1.02, f"{label}\n{value:.1f}%".replace(".", ","),
                color=color, ha="center", va="bottom", fontsize=8.5)

    ax.set_xlabel("IRR, % годовых")
    ax.set_ylabel("Частота")
    ax.set_yticks([])
    ax.margins(x=0.01)
    fig.subplots_adjust(top=0.82)
    return _fig_to_svg(fig)


# --------------------------------------------------------------------------- #
# 2. Водопад денежных потоков (годовые чистые потоки)
# --------------------------------------------------------------------------- #
def cash_flow_waterfall(result: DCFResult) -> str:
    _apply_brand_style()
    fig, ax = plt.subplots(figsize=(7.4, 3.6))

    years = [ln.year for ln in result.annual_lines]
    flows = [ln.net_cash_flow for ln in result.annual_lines]
    # Притоки — шалфей, оттоки — бордо (приглушённые).
    colors = [SAGE if f >= 0 else BURGUNDY for f in flows]

    ax.bar(years, [f / 1e6 for f in flows], color=colors, edgecolor=PAPER, width=0.62)
    ax.axhline(0, color=INK, linewidth=0.8)
    _thin_axes(ax)

    ax.set_xlabel("Год удержания")
    ax.set_ylabel("Чистый поток, млн ₽")
    ax.set_xticks(years)
    return _fig_to_svg(fig)


# --------------------------------------------------------------------------- #
# 3. Торнадо-диаграмма чувствительности
# --------------------------------------------------------------------------- #
def tornado(bars: List[TornadoBar], base_irr: float) -> str:
    _apply_brand_style()
    fig, ax = plt.subplots(figsize=(7.4, max(2.4, 0.7 * len(bars) + 1.2)))

    labels = [b.label for b in bars]
    y = np.arange(len(bars))[::-1]  # крупнейший вклад сверху

    # Запас по оси X, чтобы числовые подписи у краёв баров не налезали на
    # названия категорий и не выходили за рамку.
    all_vals = [v for b in bars for v in (b.irr_low, b.irr_high)] + [base_irr]
    lo_lim, hi_lim = min(all_vals), max(all_vals)
    pad = max((hi_lim - lo_lim) * 0.14, 0.5)
    ax.set_xlim(lo_lim - pad, hi_lim + pad)

    for yi, b in zip(y, bars):
        lo = min(b.irr_low, b.irr_high)
        hi = max(b.irr_low, b.irr_high)
        ax.barh(yi, hi - lo, left=lo, color=BRASS, edgecolor=BRASS_DEEP,
                linewidth=0.6, height=0.55)
        # Подписи крайних значений IRR.
        ax.text(lo, yi, f"{lo:.1f}".replace(".", ","), va="center", ha="right",
                fontsize=8.5, color=INK_SOFT)
        ax.text(hi, yi, f"{hi:.1f}".replace(".", ","), va="center", ha="left",
                fontsize=8.5, color=INK_SOFT)

    # Базовая линия IRR.
    ax.axvline(base_irr, color=BURGUNDY, linewidth=1.2, linestyle=":")
    ax.text(base_irr, len(bars) - 0.4, f"База {base_irr:.1f}%".replace(".", ","),
            color=BURGUNDY, ha="center", va="bottom", fontsize=8.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("IRR, % годовых")
    _thin_axes(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return _fig_to_svg(fig)


# --------------------------------------------------------------------------- #
# 4. Структура рассрочки: транши и накопленный отток
# --------------------------------------------------------------------------- #
def installment_timeline(tranche_months: List[int], tranche_amounts: List[float],
                         price_rub: float) -> str:
    _apply_brand_style()
    fig, ax = plt.subplots(figsize=(7.4, 3.4))

    months = tranche_months
    amounts_m = [a / 1e6 for a in tranche_amounts]

    ax.bar(months, amounts_m, width=1.6, color=BRASS, edgecolor=BRASS_DEEP, linewidth=0.6,
           label="Транш, млн ₽")
    _thin_axes(ax)
    ax.set_xlabel("Месяц от сделки")
    ax.set_ylabel("Транш, млн ₽")

    # Накопленный оплаченный процент на второй оси.
    ax2 = ax.twinx()
    cum_pct = np.cumsum([a / price_rub * 100 for a in tranche_amounts])
    ax2.step([m for m in months] + [max(months) + 1], list(cum_pct) + [cum_pct[-1]],
             where="post", color=BURGUNDY, linewidth=1.4)
    ax2.set_ylabel("Накоплено оплаты, %", color=BURGUNDY)
    ax2.tick_params(axis="y", colors=BURGUNDY)
    ax2.set_ylim(0, 105)
    for spine in ax2.spines.values():
        spine.set_color(LINE)
        spine.set_linewidth(0.8)
    ax2.spines["top"].set_visible(False)

    ax.set_xticks(months)
    return _fig_to_svg(fig)
