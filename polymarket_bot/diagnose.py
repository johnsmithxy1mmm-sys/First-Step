"""Selection-funnel diagnostics: why markets don't reach trades.

    python -m polymarket_bot --mode diagnose

A single pass over all active markets, broken down by how many were dropped
at each filter — separately for the longshot scanner, the market maker and
the fade strategy. Shows exactly where the funnel collapses so you can widen
it surgically instead of guessing.
"""

from __future__ import annotations

from collections import Counter

from rich.console import Console
from rich.table import Table

from .config import BotConfig
from .estimator import Estimator
from .executor import Executor
from .fade import FadeStrategy
from .gamma import GammaClient
from .ledger import Ledger
from .models import Market
from .portfolio import Portfolio
from .scanner import Scanner
from .scorer import MarketScorer


def funnel(items: list, reject_reason) -> tuple[Counter, int]:
    """Counts reject reasons; returns (reason counter, number that passed)."""
    reasons: Counter = Counter()
    passed = 0
    for item in items:
        reason = reject_reason(item)
        if reason is None:
            passed += 1
        else:
            reasons[reason] += 1
    return reasons, passed


def _print_funnel(console: Console, title: str, total: int, unit: str,
                  reasons: Counter, passed: int) -> None:
    t = Table(title=f"{title}: {total} {unit} -> {passed} passed")
    t.add_column("Rejected at filter")
    t.add_column(unit.capitalize(), justify="right")
    for reason, count in reasons.most_common():
        t.add_row(reason, str(count))
    t.add_row("[bold green]PASSED[/bold green]", f"[bold green]{passed}[/bold green]")
    console.print(t)


def run_diagnose(cfg: BotConfig, gamma: GammaClient | None = None) -> None:
    console = Console()
    gamma = gamma or GammaClient(cfg)
    console.print("Loading active Polymarket markets…")
    markets = gamma.fetch_active_markets()
    total = len(markets)
    console.print(f"Active markets in total: [bold]{total}[/bold]\n")

    scanner = Scanner(cfg)
    ls_reasons, ls_passed = funnel(markets, scanner.reject_reason)
    _print_funnel(console, "LONGSHOTS (scanner)", total, "markets", ls_reasons, ls_passed)
    console.print("[dim]\"passed\" = the market has a cheap outcome; next comes the book "
                  "depth check and the edge threshold (edge >= 2.0 drops almost "
                  "everything — by design).[/dim]\n")

    from .resolution import ResolutionAlpha
    res = ResolutionAlpha(cfg, Ledger(":memory:"), None, None, "dry-run")
    res_reasons, res_passed = funnel(markets, res.reject_reason)
    _print_funnel(console, "RESOLUTION alpha", total, "markets", res_reasons, res_passed)
    console.print("[dim]\"passed\" = a near-resolved side (0.95-0.985, fresh volume, "
                  "imminent) with net edge after taker fee + dispute reserve.[/dim]\n")

    scorer = MarketScorer(cfg)
    mm_reasons, mm_passed = funnel(markets, scorer.reject_reason)
    _print_funnel(console, "MARKET MAKER (scorer)", total, "markets", mm_reasons, mm_passed)
    console.print("[dim]\"passed\" = market is quotable. If this is 0, MM quotes "
                  "nothing, hence zero trades. Look at the top row of the table: "
                  "that's the filter to loosen in config.yaml (market_maker:).[/dim]\n")

    # Fade funnel: runs over the first-level cheap-tail candidates, estimated.
    candidates = scanner.first_level_filter(markets)
    estimates = Estimator(cfg).estimate_all(candidates, markets)
    ledger = Ledger(":memory:")
    portfolio = Portfolio(cfg, ledger, "dry-run")
    executor = Executor(cfg, ledger, None, None, "dry-run")
    fade = FadeStrategy(cfg, ledger, portfolio, executor, "dry-run")
    fade_reasons, fade_passed = funnel(estimates, fade.reject_reason)
    _print_funnel(console, "FADE (overpriced tails)", len(estimates),
                  "candidates", fade_reasons, fade_passed)
    console.print("[dim]\"passed\" = a fadeable overpriced tail (before portfolio "
                  "caps). Input is the cheap-tail candidates, so a small number "
                  "here is expected; \"estimator sees a real longshot\" means our "
                  "own model vetoed the fade.[/dim]")
