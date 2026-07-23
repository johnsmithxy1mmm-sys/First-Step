"""Counterfactual replay: what would today's config have earned last week?

Every parameter change deserves a test on REALITY, not a synthetic backtest.
This replays a market-making spread over the recorded top-of-book series: at
each step it places a bid a half-spread under the mid and asks whether the
market later traded through it (a real fill event on recorded data), then marks
the fill a few steps later to see whether the spread survived adverse
selection. Sweeping the half-spread shows which value would have captured the
most on the data actually seen — a measured answer where autotune only had a
synthetic one.

    python -m polymarket_bot --mode timemachine

Honest approximation, stated plainly: a fill is "the opposite side crossed our
price within the fill horizon" (top of book only — queue position is ignored,
which is conservative), and capture is marked to the later mid (markout), so
adverse selection is charged. Pure functions of the recorded series; tested.
"""

from __future__ import annotations

from statistics import mean

from pydantic import BaseModel


class SpreadResult(BaseModel):
    half_spread: float
    placements: int
    fills: int
    fill_rate: float
    avg_capture_per_fill: float     # half_spread + markout (mid drift after fill)
    capture_per_placement: float    # fill_rate * avg_capture_per_fill


def _mid(row: tuple) -> float:
    _ts, bid, ask, *_ = row
    if bid and ask and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return bid or ask or 0.0


def simulate_spread_capture(series: list[tuple], half_spread: float,
                            fill_horizon: int = 4, markout_horizon: int = 4
                            ) -> SpreadResult:
    """Replay a passive bid a half-spread under mid over one token's recorded
    top-of-book. Capture per fill = half_spread + (mid_after - mid_now): the
    discount we bought at, plus (or minus) how the mid then drifted."""
    n = len(series)
    placements = fills = 0
    captures: list[float] = []
    last_start = n - markout_horizon
    for i in range(max(0, last_start)):
        mid = _mid(series[i])
        if mid <= 0:
            continue
        placements += 1
        our_bid = mid - half_spread
        # A fill = a later ask within the horizon crosses down to our bid.
        hit = False
        for j in range(i + 1, min(i + 1 + fill_horizon, n)):
            _ts, _bid, ask, *_ = series[j]
            if ask and ask > 0 and ask <= our_bid:
                hit = True
                break
        if not hit:
            continue
        fills += 1
        mid_after = _mid(series[min(i + markout_horizon, n - 1)])
        captures.append(half_spread + (mid_after - mid))
    avg_cap = mean(captures) if captures else 0.0
    fill_rate = fills / placements if placements else 0.0
    return SpreadResult(
        half_spread=half_spread, placements=placements, fills=fills,
        fill_rate=round(fill_rate, 4), avg_capture_per_fill=round(avg_cap, 5),
        capture_per_placement=round(fill_rate * avg_cap, 5))


def sweep(series_by_token: dict[str, list[tuple]], half_spreads: list[float],
          fill_horizon: int = 4, markout_horizon: int = 4) -> list[SpreadResult]:
    """Aggregate the replay over all tokens, one row per half-spread, best
    expected capture-per-placement first."""
    out: list[SpreadResult] = []
    for hs in half_spreads:
        placements = fills = 0
        weighted_cap = 0.0
        for series in series_by_token.values():
            r = simulate_spread_capture(series, hs, fill_horizon, markout_horizon)
            placements += r.placements
            fills += r.fills
            weighted_cap += r.avg_capture_per_fill * r.fills
        avg_cap = weighted_cap / fills if fills else 0.0
        fill_rate = fills / placements if placements else 0.0
        out.append(SpreadResult(
            half_spread=hs, placements=placements, fills=fills,
            fill_rate=round(fill_rate, 4), avg_capture_per_fill=round(avg_cap, 5),
            capture_per_placement=round(fill_rate * avg_cap, 5)))
    out.sort(key=lambda r: r.capture_per_placement, reverse=True)
    return out


def run_timemachine(cfg) -> None:  # pragma: no cover — I/O glue
    from rich.console import Console
    from rich.table import Table

    from .tickstore import TickStore

    console = Console()
    if not cfg.ticks.enabled:
        console.print("[yellow]ticks.enabled is false — nothing recorded.[/yellow]")
        return
    store = TickStore(cfg.ticks.db_path)
    # Replay over whatever tokens have been recorded (most-active first is fine).
    import sqlite3
    conn = sqlite3.connect(cfg.ticks.db_path)
    try:
        tokens = [r[0] for r in conn.execute(
            "SELECT token, COUNT(*) c FROM ticks GROUP BY token "
            "ORDER BY c DESC LIMIT 50").fetchall()]
    except sqlite3.OperationalError:
        tokens = []
    finally:
        conn.close()
    series = store.token_book_series(tokens)
    store.close()
    if not series:
        console.print("[dim]No recorded ticks yet — run the bot to accumulate "
                      "top-of-book history first.[/dim]")
        return

    base = cfg.market_maker.half_spread
    grid = sorted({round(base * f, 4) for f in (0.5, 0.75, 1.0, 1.5, 2.0)})
    rows = sweep(series, grid)
    t = Table(title=f"Counterfactual MM replay over {len(series)} recorded tokens "
                    f"(current half_spread {base})")
    for col in ("Half-spread", "Placements", "Fills", "Fill rate",
                "Capture/fill", "Capture/placement"):
        t.add_column(col, justify="right")
    for r in rows:
        mark = " ◀ best" if r is rows[0] else \
               "  (current)" if abs(r.half_spread - base) < 1e-9 else ""
        t.add_row(f"{r.half_spread:.4f}", str(r.placements), str(r.fills),
                  f"{r.fill_rate:.1%}", f"{r.avg_capture_per_fill:+.5f}",
                  f"{r.capture_per_placement:+.5f}{mark}")
    console.print(t)
    console.print("[dim]Capture/fill = the half-spread you bought under mid, plus "
                  "how the mid then drifted (adverse selection charged). "
                  "Capture/placement weights that by how often you'd fill. This "
                  "is measured on the ticks actually seen, not a synthetic "
                  "book.[/dim]")
