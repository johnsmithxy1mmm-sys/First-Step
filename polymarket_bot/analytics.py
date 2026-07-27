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


def compute_report(ledger: Ledger, mode: str, fade_prior: float = 0.35) -> dict:
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
        "estimates": ledger.estimates_summary(fade_prior),
        "markouts": ledger.markout_stats(mode),
        "shadow_quotes": ledger.shadow_quote_summary(mode),
        "positions": ledger.open_positions(mode),
        "learned_bias": _learned_bias(ledger, mode, fade_prior),
        "stress": _stress(ledger.open_positions(mode)),
        "opportunities": ledger.opportunity_stats(mode),
        "top_opportunities": ledger.top_opportunities(mode),
    }


def _stress(positions) -> dict:
    from .risk2 import portfolio_stress
    return portfolio_stress(positions)


def _learned_bias(ledger: Ledger, mode: str, prior: float = 0.35) -> list[dict]:
    from .calibration import TailBiasCalibrator
    return TailBiasCalibrator(prior=prior).fit(
        ledger.resolved_for_calibration(mode, "fade")).summary()


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
        total = est["total"]
        c.print(f"Estimates recorded: {total} | passed edge threshold: "
                f"{est['qualifying'] or 0} | avg edge: {est['avg_edge']:.2f}")
        binding = est.get("binding") or 0
        informative = est.get("informative") or 0
        c.print(f"  estimator moved the fade in {binding}/{total} "
                f"({100.0 * binding / total:.1f}%) | differed from market by >5% "
                f"in {informative}/{total} ({100.0 * informative / total:.1f}%)")
        if binding == 0:
            c.print("[yellow]  the estimator changed no fade decision: every "
                    "trade rested on the bias_discount prior, so the LLM is "
                    "currently being paid for nothing[/yellow]")

    shadows = report.get("shadow_quotes") or []
    if shadows:
        t = Table(title="Shadow quotes — how close the tape came to our MM bids")
        for col, just in (("Market", "left"), ("Quotes", "right"),
                          ("Looks", "right"), ("Closest", "right"),
                          ("Avg gap", "right")):
            t.add_column(col, justify=just)
        for s in shadows[:15]:
            closest = s["closest"] if s["closest"] is not None else 0.0
            t.add_row((s["question"] or s["market_id"])[:46], str(s["quotes"]),
                      str(s["looks"] or 0), f"{closest:+.4f}",
                      f"{(s['avg_gap'] or 0.0):+.4f}")
        c.print(t)
        c.print("[dim]closest <= 0 = the tape crossed our quote at least once. A "
                "large positive closest with many looks means the flow never came "
                "near us — widen/narrow decisions should be made on this, not on "
                "an empty fill count.[/dim]")

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

        from .portfolio import event_exposure_breakdown
        rows = event_exposure_breakdown(positions)
        gross_total = sum(r[2] for r in rows)
        worst_total = sum(r[3] for r in rows)
        t = Table(title=f"Risk by event (gross ${gross_total:,.0f} -> "
                        f"true worst-case ${worst_total:,.0f})")
        for col in ("Event / market", "Legs", "Gross, $", "Worst-case, $"):
            t.add_column(col)
        for label, legs, gross, worst in rows[:15]:
            t.add_row(label, str(legs), f"{gross:,.2f}", f"{worst:,.2f}")
        c.print(t)
        c.print("[dim]Neg-risk baskets net down: exactly one outcome wins, so at "
                "most one NO leg loses — worst-case is the largest leg, not the "
                "sum. The portfolio caps use this true risk, freeing room for "
                "self-hedged clusters.[/dim]")

    st = report.get("stress") or {}
    if st.get("gross_usd"):
        c.print(f"[bold]Risk:[/bold] gross ${st['gross_usd']:,.0f} | "
                f"true worst-case ${st['worst_case_usd']:,.0f} | "
                f"largest event ${st['largest_event_usd']:,.0f} | "
                f"VaR95 ~${st['var95_usd']:,.0f}")

    opps = report.get("opportunities") or []
    if opps:
        t = Table(title="Opportunity capacity (detected windows, taken or not)")
        for col in ("Strategy", "Windows", "Sightings", "Best edge",
                    "Capacity $ (edge x depth)", "Executed"):
            t.add_column(col, justify="right")
        for o in opps:
            t.add_row(
                o["strategy"], str(o["windows"]), str(int(o["sightings"] or 0)),
                f"{(o['best_edge'] or 0) * 100:.2f}%",
                f"${o['edge_dollars'] or 0:,.2f}", str(int(o["executed"] or 0)))
        c.print(t)
        c.print("[dim]This measures REALIZABLE edge, not paper edge: distinct "
                "windows actually seen, how persistent they were, and the "
                "tradable notional at the best edge. A big 'sightings' with tiny "
                "capacity = the window is real but too thin to matter.[/dim]")
        top = report.get("top_opportunities") or []
        if top:
            t = Table(title="Top opportunities by capacity")
            for col in ("Strategy", "Label", "Seen", "Best edge", "Depth $"):
                t.add_column(col)
            for o in top[:10]:
                t.add_row(o["strategy"], o["label"][:40], str(o["sightings"]),
                          f"{(o['best_edge'] or 0) * 100:.2f}%",
                          f"${o['best_depth_usd'] or 0:,.0f}")
            c.print(t)

    learned = report.get("learned_bias") or []
    if learned:
        t = Table(title="Learned fade bias (from resolutions)")
        for col in ("Category", "Price bucket", "N", "Learned bias"):
            t.add_column(col, justify="right")
        for row in learned:
            t.add_row(row["category"], f"{row['price_bucket']:.3f}",
                      str(row["n"]), f"{row['learned_bias']:.3f}")
        c.print(t)
        c.print("[dim]bias_discount is no longer a constant — it is the empirical "
                "tail-overpricing per bucket, shrunk toward the prior on small "
                "samples.[/dim]")
