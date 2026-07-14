"""Analytics report: everything the bot accumulated in the ledger, one command.

    python -m polymarket_bot --mode report

Sections: bank and drawdown, PnL by strategy, longshot metrics (hit rate, ROI,
model Brier vs market), estimate summary, fill markout analysis (the key test
of MM execution quality), open positions.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .ledger import Ledger


def compute_report(ledger: Ledger, mode: str) -> dict:
    """Gathers all metrics into one dict (testable without rich)."""
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
    bank_line = "no bank snapshots yet — run the bot for at least one cycle"
    if eq_now is not None:
        delta = eq_now - eq_start
        bank_line = (f"Equity: ${eq_start:,.2f} -> ${eq_now:,.2f} ({delta:+,.2f}) | "
                     f"HWM ${report['hwm']:,.2f} | "
                     f"max drawdown {report['max_drawdown_pct'] * 100:.1f}% | "
                     f"points: {report['bank_points']}")
    c.print(Panel(bank_line, title=f"PolyBot report — mode {mode}"))

    pnl = report["pnl_by_strategy"]
    if pnl:
        t = Table(title="Realized PnL by strategy")
        t.add_column("Strategy")
        t.add_column("PnL, $", justify="right")
        for name, value in sorted(pnl.items()):
            t.add_row(name, f"{value:+,.2f}")
        c.print(t)

    ls = report["longshot"]
    if ls["resolved_trades"]:
        t = Table(title="Longshots: resolved trades")
        for col in ("Trades", "Hit rate", "Avg multiple", "ROI",
                    "Model Brier", "Market Brier"):
            t.add_column(col, justify="right")
        better = ls["brier_model"] is not None and ls["brier_market"] is not None \
            and ls["brier_model"] < ls["brier_market"]
        t.add_row(
            str(ls["resolved_trades"]), f"{ls['hit_rate']:.1%}",
            f"{ls['avg_win_multiple']:.1f}x", f"{ls['roi']:+.1%}",
            f"{ls['brier_model']:.5f}" + (" (better than market)" if better else ""),
            f"{ls['brier_market']:.5f}")
        c.print(t)
        attribution = ls.get("signal_pnl_attribution") or {}
        if attribution:
            t = Table(title="PnL attribution by signal")
            t.add_column("Signal")
            t.add_column("PnL, $", justify="right")
            for name, value in sorted(attribution.items(), key=lambda x: -x[1]):
                t.add_row(name, f"{value:+,.2f}")
            c.print(t)
    else:
        c.print("[dim]No resolved longshot trades yet — hit rate and Brier "
                "will appear after the first resolutions.[/dim]")

    est = report["estimates"]
    if est.get("total"):
        c.print(f"Estimates recorded: {est['total']} | passed edge threshold: "
                f"{est['qualifying'] or 0} | avg edge: {est['avg_edge']:.2f}")

    markouts = report["markouts"]
    if markouts:
        t = Table(title="Fill markout analysis (adverse-selection test)")
        for col in ("Strategy", "Horizon", "Fills", "Avg markout",
                    "% of price", "In our favor"):
            t.add_column(col, justify="right")
        for m in markouts:
            t.add_row(
                m["strategy"], f"+{m['horizon_sec']}s", str(m["n"]),
                f"{m['avg_markout']:+.4f}", f"{m['avg_markout_pct'] * 100:+.1f}%",
                f"{m['favorable']}/{m['n']}")
        c.print(t)
        c.print("[dim]markout < 0 on buys = price falls after our fill: the "
                "informed are running us over — tighten the guard or widen the "
                "spread. markout ~ 0 and stable = you can narrow the spread.[/dim]")
    else:
        c.print("[dim]No markout data yet: it accumulates automatically 1 and "
                "10 minutes after each fill (needs a running bot).[/dim]")

    positions = report["positions"]
    if positions:
        t = Table(title=f"Open positions ({len(positions)})")
        for col in ("Category", "Outcome / Question", "Size", "Entry", "Cost, $"):
            t.add_column(col)
        for p in sorted(positions, key=lambda p: p.cost_usd, reverse=True)[:25]:
            t.add_row(p.category, f"[{p.outcome}] {p.question[:55]}",
                      f"{p.size:,.0f}", f"{p.avg_price:.4f}", f"{p.cost_usd:,.2f}")
        c.print(t)
