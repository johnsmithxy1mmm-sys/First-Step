"""Selection-funnel diagnostics: why markets don't reach trades.

    python -m polymarket_bot --mode diagnose

A single pass over all active markets, broken down by how many were dropped
at each filter — separately for the longshot scanner, the market maker and
the fade strategy. Shows exactly where the funnel collapses so you can widen
it surgically instead of guessing.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

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


HOUR_BUCKETS = [
    (0, 2, "0-2h (settlement window — sprint refuses)"),
    (2, 6, "2-6h"),
    (6, 24, "6-24h"),
    (24, 48, "24-48h"),
    (48, 96, "48-96h (2-4d)"),
    (96, 168, "96-168h (4-7d)"),
    (168, 336, "168-336h (7-14d)"),
    (336, 720, "336-720h (14-30d)"),
    (720, None, "720h+ (30d+, core MM territory)"),
]


def _bucket_label(hours: float) -> str:
    for lo, hi, label in HOUR_BUCKETS:
        if hours >= lo and (hi is None or hours < hi):
            return label
    return "?"


def _print_horizon_histogram(console: Console, markets: list[Market],
                             volume_floors: list[float]) -> None:
    """Where liquid markets actually sit in time — the real fix for an empty
    SPRINT MM funnel is picking max_hours_to_resolution from THIS, not a guess."""
    now = datetime.now(timezone.utc)
    live = [m for m in markets
            if not m.closed and m.enable_order_book and len(m.clob_token_ids) >= 2
            and m.days_to_resolution(now) is not None and m.days_to_resolution(now) >= 0]

    t = Table(title="Horizon of LIQUID markets (book present, not closed) — "
                     "pick max_hours_to_resolution from where the mass is")
    t.add_column("Resolves in")
    for v in volume_floors:
        t.add_column(f"24h vol >= ${v:,.0f}", justify="right")

    counts = {v: Counter() for v in volume_floors}
    for m in live:
        hours = m.days_to_resolution(now) * 24.0
        label = _bucket_label(hours)
        for v in volume_floors:
            if m.volume_24h_usd >= v:
                counts[v][label] += 1

    for _, _, label in HOUR_BUCKETS:
        t.add_row(label, *[str(counts[v][label]) for v in volume_floors])
    t.add_row("[bold]TOTAL liquid markets[/bold]",
              *[f"[bold]{sum(counts[v].values())}[/bold]" for v in volume_floors])
    console.print(t)
    console.print("[dim]Read this top-to-bottom: find the first bucket with real "
                  "counts, set max_hours_to_resolution to its upper edge (or the "
                  "one below to stay conservative). If a column is all zeros, that "
                  "volume floor filters out everything currently on the "
                  "board — try the lower column instead.[/dim]\n")


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

    from .sprintmaker import SprintScorer
    sprint = SprintScorer(cfg)
    sp_reasons, sp_passed = funnel(markets, sprint.reject_reason)
    _print_funnel(console, "SPRINT MM (short-dated)", total, "markets", sp_reasons, sp_passed)
    console.print("[dim]\"passed\" = a LIQUID market resolving within the sprint "
                  "hours-window — fast capital turnover on spread + rebate. If 0, "
                  "loosen max_hours_to_resolution / min_volume_24h_usd in "
                  "config.yaml (sprint_mm:). Empty is expected when no liquid "
                  "market resolves that soon.[/dim]\n")

    floors = sorted({cfg.sprint_mm.min_volume_24h_usd, 20_000.0, 10_000.0}, reverse=True)
    _print_horizon_histogram(console, markets, floors)

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
