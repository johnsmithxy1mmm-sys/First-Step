"""Capacity & learning report: what the accumulated data actually says.

    python -m polymarket_bot --mode capacity

Two questions this answers from recorded data, not intuition:

  * How real is each strategy's edge? — from the opportunity ledger: how many
    distinct windows were seen, how long they lived (persistence), and the
    tradable dollars at the best edge. A window that flickers for one cycle
    with $20 of depth is not a business; one that lives minutes with $5k is.

  * When will the learned models switch on? — every self-calibrator needs a
    minimum sample count; this shows how much history has accumulated toward
    each threshold, so "self-calibration" has a visible ETA instead of being a
    promise.
"""

from __future__ import annotations

from datetime import datetime

from rich.console import Console
from rich.table import Table

from .config import BotConfig
from .ledger import Ledger

# Minimum samples each learner needs before it moves off the prior/identity
# (kept in sync with calibration.py / the estimator).
READINESS = {
    "category correlations": ("category-index points", 50),
    "fill probability": ("quote outcomes", 20),
    "p_est (Platt)": ("resolved estimates", 30),
    "fade tail bias": ("resolved fades", 20),
}


def _parse(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def opportunity_persistence(opps: list[dict]) -> list[dict]:
    """Per-strategy: windows, avg lifetime, avg sightings, best edge, capacity."""
    by: dict[str, list[dict]] = {}
    for o in opps:
        by.setdefault(o["strategy"], []).append(o)
    out = []
    for strategy, rows in sorted(by.items()):
        lifetimes = []
        for r in rows:
            a, b = _parse(r["first_seen"]), _parse(r["last_seen"])
            if a and b:
                lifetimes.append((b - a).total_seconds())
        out.append({
            "strategy": strategy,
            "windows": len(rows),
            "avg_lifetime_min": (sum(lifetimes) / len(lifetimes) / 60.0)
                                if lifetimes else 0.0,
            "avg_sightings": sum(r["sightings"] for r in rows) / len(rows),
            "best_edge": max((r["best_edge"] for r in rows), default=0.0),
            "capacity_usd": sum((r["best_edge"] or 0) * (r["best_depth_usd"] or 0)
                                for r in rows),
            "executed": sum(int(r["executed"] or 0) for r in rows),
        })
    return out


def learning_progress(cfg: BotConfig, ledger: Ledger, mode: str) -> dict:
    """How much history has accumulated toward each learner's threshold."""
    have = {
        "category correlations": 0,
        "fill probability": len(ledger.quote_outcomes(mode)),
        "p_est (Platt)": len([r for r in _resolved(ledger, mode) if r.get("p_est")]),
        "fade tail bias": len(ledger.resolved_for_calibration(mode, "fade")),
    }
    if cfg.ticks.enabled:
        from .tickstore import TickStore
        ts = TickStore(cfg.ticks.db_path)
        series = ts.category_series()
        have["category correlations"] = max((len(v) for v in series.values()), default=0)
        ts.close()
    return have


def _resolved(ledger: Ledger, mode: str) -> list[dict]:
    out = []
    for strategy in ("longshot", "fade"):
        out += ledger.resolved_for_calibration(mode, strategy)
    return out


def run_capacity(cfg: BotConfig, mode: str) -> None:
    console = Console()
    ledger = Ledger(cfg.runtime.db_path)
    try:
        persistence = opportunity_persistence(ledger.all_opportunities(mode))
        progress = learning_progress(cfg, ledger, mode)
    finally:
        ledger.close()

    console.print(f"[bold]Capacity & learning — mode {mode}[/bold]\n")

    if persistence:
        t = Table(title="Opportunity persistence (measured, not backtested)")
        for col in ("Strategy", "Windows", "Avg life (min)", "Avg sightings",
                    "Best edge", "Capacity $", "Executed"):
            t.add_column(col, justify="right")
        for p in persistence:
            t.add_row(p["strategy"], str(p["windows"]),
                      f"{p['avg_lifetime_min']:.1f}", f"{p['avg_sightings']:.1f}",
                      f"{p['best_edge'] * 100:.2f}%", f"${p['capacity_usd']:,.2f}",
                      str(p["executed"]))
        console.print(t)
        console.print("[dim]A window that lives a fraction of a minute with tiny "
                      "capacity is real but untradable; persistent windows with "
                      "real depth are the business. This is measured from live "
                      "sightings, not a backtest.[/dim]\n")
    else:
        console.print("[dim]No opportunities recorded yet — run the bot with "
                      "arbitrage/chain_arb enabled for a while.[/dim]\n")

    t = Table(title="Learning readiness (data accumulated toward each learner)")
    for col in ("Learner", "Signal", "Have", "Need", "Status"):
        t.add_column(col)
    for name, (signal, need) in READINESS.items():
        have = progress.get(name, 0)
        status = ("[green]ready[/green]" if have >= need
                  else f"[yellow]{have * 100 // need}%[/yellow]")
        t.add_row(name, signal, str(have), str(need), status)
    console.print(t)
    console.print("[dim]Until a learner is ready it falls back to the expert "
                  "prior/identity — nothing breaks, it just isn't personalized "
                  "yet. Keep the bot running (ticks record continuously) and "
                  "these fill in.[/dim]")
