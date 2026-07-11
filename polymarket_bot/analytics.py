"""Аналитический отчёт: всё, что бот накопил в леджере, — одной командой.

    python -m polymarket_bot --mode report

Секции: банк и просадка, PnL по стратегиям, лонгшот-метрики (hit rate, ROI,
Brier модели против рынка), сводка оценок, markout-анализ филлов (главный
тест качества исполнения MM), открытые позиции.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .ledger import Ledger


def compute_report(ledger: Ledger, mode: str) -> dict:
    """Собирает все метрики в один словарь (тестируется без rich)."""
    bank = ledger.bank_series()
    equity_start = bank[0]["equity"] if bank else None
    equity_now = bank[-1]["equity"] if bank else None
    hwm = max((b["hwm"] for b in bank), default=0.0)
    max_dd = 0.0
    peak = 0.0
    for b in bank:
        peak = max(peak, b["equity"])
        if peak > 0:
            max_dd = max(max_dd, (peak - b["equity"]) / peak)
    return {
        "mode": mode,
        "equity_start": equity_start,
        "equity_now": equity_now,
        "hwm": hwm,
        "max_drawdown_pct": max_dd,
        "bank_points": len(bank),
        "pnl_by_strategy": ledger.realized_pnl_by_strategy(mode),
        "longshot": ledger.metrics(mode),
        "estimates": ledger.estimates_summary(),
        "markouts": ledger.markout_stats(mode),
        "positions": ledger.open_positions(mode),
    }


def print_report(report: dict) -> None:
    c = Console()
    mode = report["mode"]

    eq_start, eq_now = report["equity_start"], report["equity_now"]
    bank_line = "снапшотов банка ещё нет — запустите бота хотя бы на один цикл"
    if eq_now is not None:
        delta = eq_now - eq_start
        bank_line = (f"Equity: ${eq_start:,.2f} -> ${eq_now:,.2f} ({delta:+,.2f}) | "
                     f"HWM ${report['hwm']:,.2f} | "
                     f"макс. просадка {report['max_drawdown_pct'] * 100:.1f}% | "
                     f"точек: {report['bank_points']}")
    c.print(Panel(bank_line, title=f"Отчёт PolyBot — режим {mode}"))

    pnl = report["pnl_by_strategy"]
    if pnl:
        t = Table(title="Реализованный PnL по стратегиям")
        t.add_column("Стратегия")
        t.add_column("PnL, $", justify="right")
        for name, value in sorted(pnl.items()):
            t.add_row(name, f"{value:+,.2f}")
        c.print(t)

    ls = report["longshot"]
    if ls["resolved_trades"]:
        t = Table(title="Лонгшоты: разрешившиеся сделки")
        for col in ("Сделок", "Hit rate", "Средний множитель", "ROI",
                    "Brier модели", "Brier рынка"):
            t.add_column(col, justify="right")
        better = ls["brier_model"] is not None and ls["brier_market"] is not None \
            and ls["brier_model"] < ls["brier_market"]
        t.add_row(
            str(ls["resolved_trades"]), f"{ls['hit_rate']:.1%}",
            f"{ls['avg_win_multiple']:.1f}x", f"{ls['roi']:+.1%}",
            f"{ls['brier_model']:.5f}" + (" (лучше рынка)" if better else ""),
            f"{ls['brier_market']:.5f}")
        c.print(t)
        attribution = ls.get("signal_pnl_attribution") or {}
        if attribution:
            t = Table(title="Атрибуция PnL по сигналам")
            t.add_column("Сигнал")
            t.add_column("PnL, $", justify="right")
            for name, value in sorted(attribution.items(), key=lambda x: -x[1]):
                t.add_row(name, f"{value:+,.2f}")
            c.print(t)
    else:
        c.print("[dim]Разрешившихся лонгшот-сделок пока нет — hit rate и Brier "
                "появятся после первых резолюций.[/dim]")

    est = report["estimates"]
    if est.get("total"):
        c.print(f"Оценок записано: {est['total']} | прошло порог edge: "
                f"{est['qualifying'] or 0} | средний edge: {est['avg_edge']:.2f}")

    markouts = report["markouts"]
    if markouts:
        t = Table(title="Markout-анализ филлов (тест adverse selection)")
        for col in ("Стратегия", "Горизонт", "Филлов", "Средний markout",
                    "в % от цены", "В нашу сторону"):
            t.add_column(col, justify="right")
        for m in markouts:
            t.add_row(
                m["strategy"], f"+{m['horizon_sec']}с", str(m["n"]),
                f"{m['avg_markout']:+.4f}", f"{m['avg_markout_pct'] * 100:+.1f}%",
                f"{m['favorable']}/{m['n']}")
        c.print(t)
        c.print("[dim]markout < 0 на покупках = после нашего филла цена падает: "
                "нас переезжают информированные — ужесточайте guard или ширьте "
                "спред. markout ~ 0 и стабильный = можно сужать спред.[/dim]")
    else:
        c.print("[dim]Markout-данных пока нет: они копятся автоматически через "
                "1 и 10 минут после каждого филла (нужен работающий бот).[/dim]")

    positions = report["positions"]
    if positions:
        t = Table(title=f"Открытые позиции ({len(positions)})")
        for col in ("Категория", "Исход / Вопрос", "Размер", "Вход", "Кост, $"):
            t.add_column(col)
        for p in sorted(positions, key=lambda p: p.cost_usd, reverse=True)[:25]:
            t.add_row(p.category, f"[{p.outcome}] {p.question[:55]}",
                      f"{p.size:,.0f}", f"{p.avg_price:.4f}", f"{p.cost_usd:,.2f}")
        c.print(t)
