"""Диагностика воронки отбора: почему рынки не проходят к сделкам.

    python -m polymarket_bot --mode diagnose

Один проход по всем активным рынкам с разбивкой, сколько отсеялось на каждом
фильтре — отдельно для лонгшот-сканера и для маркет-мейкера. Показывает,
на каком именно фильтре осыпается воронка, чтобы расширять точечно, а не
наугад.
"""

from __future__ import annotations

from collections import Counter

from rich.console import Console
from rich.table import Table

from .config import BotConfig
from .gamma import GammaClient
from .models import Market
from .scanner import Scanner
from .scorer import MarketScorer


def funnel(markets: list[Market], reject_reason) -> tuple[Counter, int]:
    """Считает причины отсева; возвращает (счётчик причин, число прошедших)."""
    reasons: Counter = Counter()
    passed = 0
    for m in markets:
        reason = reject_reason(m)
        if reason is None:
            passed += 1
        else:
            reasons[reason] += 1
    return reasons, passed


def _print_funnel(console: Console, title: str, total: int,
                  reasons: Counter, passed: int) -> None:
    t = Table(title=f"{title}: {total} рынков -> {passed} прошло")
    t.add_column("Отсеяно на фильтре")
    t.add_column("Рынков", justify="right")
    for reason, count in reasons.most_common():
        t.add_row(reason, str(count))
    t.add_row("[bold green]ПРОШЛО[/bold green]", f"[bold green]{passed}[/bold green]")
    console.print(t)


def run_diagnose(cfg: BotConfig, gamma: GammaClient | None = None) -> None:
    console = Console()
    gamma = gamma or GammaClient(cfg)
    console.print("Загружаю активные рынки Polymarket…")
    markets = gamma.fetch_active_markets()
    total = len(markets)
    console.print(f"Всего активных рынков: [bold]{total}[/bold]\n")

    scanner = Scanner(cfg)
    ls_reasons, ls_passed = funnel(markets, scanner.reject_reason)
    _print_funnel(console, "ЛОНГШОТЫ (сканер)", total, ls_reasons, ls_passed)
    console.print("[dim]«прошло» = у рынка есть дешёвый исход; дальше идёт проверка "
                  "глубины стакана и порог edge (edge >= 2.0 отсекает почти всё — "
                  "это by design).[/dim]\n")

    scorer = MarketScorer(cfg)
    mm_reasons, mm_passed = funnel(markets, scorer.reject_reason)
    _print_funnel(console, "МАРКЕТ-МЕЙКЕР (scorer)", total, mm_reasons, mm_passed)
    console.print("[dim]«прошло» = рынок годен для котирования. Если тут 0 — MM "
                  "ничего не котирует, отсюда ноль сделок. Смотрите верхнюю строку "
                  "таблицы: это фильтр, который надо ослабить в config.yaml "
                  "(market_maker:).[/dim]")
