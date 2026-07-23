"""Lead-lag radar: which markets react LATE to their fast siblings.

Not a prediction of direction — a measurement of already-arrived information
that a slow market hasn't priced yet. From the recorded tick series of a group
of related tokens (same event / same category), for each ordered pair we shift
one series against the other and find the lag at which their moves correlate
best. A pair with a strong positive correlation at a POSITIVE lag means the
second token repeats the first's moves a few seconds later — the first LEADS,
the second LAGS. Quoting the laggard just after the leader moves is honest
micro-alpha: you are reacting to public information faster than the slow book.

All pure functions of recorded returns (testable offline). Correlation is on
first differences (returns), not levels, so a shared trend cannot manufacture a
spurious lead-lag.
"""

from __future__ import annotations

from statistics import mean, pstdev

from pydantic import BaseModel


class LeadLag(BaseModel):
    leader: str
    laggard: str
    lag_steps: int              # how many samples the laggard trails by
    correlation: float          # best cross-correlation of returns at that lag


def _resample(series: list[tuple[float, float]], step_sec: float,
              n: int) -> list[float] | None:
    """Last-observation-carried-forward onto a uniform grid of `n` points ending
    at the latest timestamp. Returns None if the series is too short/degenerate,
    or does not COVER the grid window: backfilling pre-history with the first
    observation would plant an artificial jump at the recording start, and two
    such jumps at staggered starts cross-correlate into a fabricated 'lead'."""
    if len(series) < 3:
        return None
    series = sorted(series)
    end = series[-1][0]
    grid = [end - (n - 1 - i) * step_sec for i in range(n)]
    if series[0][0] > grid[0]:
        return None                 # recording starts inside the window — skip
    out: list[float] = []
    j = 0
    for t in grid:
        while j + 1 < len(series) and series[j + 1][0] <= t:
            j += 1
        out.append(series[j][1])
    return out


def _returns(levels: list[float]) -> list[float]:
    return [levels[i + 1] - levels[i] for i in range(len(levels) - 1)]


def _corr(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 3:
        return 0.0
    ma, mb = mean(a), mean(b)
    sa, sb = pstdev(a), pstdev(b)
    if sa <= 0 or sb <= 0:
        return 0.0
    cov = mean((x - ma) * (y - mb) for x, y in zip(a, b))
    return cov / (sa * sb)


def best_lag(leader: list[float], laggard: list[float],
             max_lag: int) -> tuple[int, float]:
    """Lag (>=0) at which laggard's returns best match the leader's EARLIER
    returns, and that correlation. lag 0 = contemporaneous."""
    lr, gr = _returns(leader), _returns(laggard)
    best_l, best_c = 0, _corr(lr, gr)
    for lag in range(1, max_lag + 1):
        if len(lr) - lag < 3:
            break
        c = _corr(lr[:-lag], gr[lag:])       # leader leads laggard by `lag`
        if c > best_c:
            best_l, best_c = lag, c
    return best_l, best_c


def analyze(series_by_token: dict[str, list[tuple[float, float]]],
            *, step_sec: float = 30.0, grid: int = 120, max_lag: int = 6,
            min_corr: float = 0.5, min_moves: int = 5) -> list[LeadLag]:
    """All (leader, laggard) pairs whose lagged return-correlation clears
    min_corr, strongest first. A pair only counts at a STRICTLY positive lag —
    a contemporaneous match is co-movement, not a lead.

    min_moves: a token must have moved at least this many grid steps in the
    window. A single spike correlates near-1.0 with any other single spike at
    SOME lag — one coincidence must not read as a lead."""
    grids = {tok: _resample(s, step_sec, grid) for tok, s in series_by_token.items()}
    grids = {tok: g for tok, g in grids.items()
             if g is not None
             and sum(1 for r in _returns(g) if r != 0.0) >= min_moves}
    out: list[LeadLag] = []
    tokens = sorted(grids)
    for a in tokens:
        for b in tokens:
            if a == b:
                continue
            lag, corr = best_lag(grids[a], grids[b], max_lag)
            if lag >= 1 and corr >= min_corr:
                out.append(LeadLag(leader=a, laggard=b, lag_steps=lag,
                                   correlation=round(corr, 3)))
    out.sort(key=lambda x: x.correlation, reverse=True)
    return out


def run_leadlag(cfg, step_sec: float = 30.0) -> None:  # pragma: no cover — I/O glue
    """Report leaders/laggards within each active event, from recorded ticks."""
    from rich.console import Console
    from rich.table import Table

    from .gamma import GammaClient
    from .tickstore import TickStore

    console = Console()
    if not cfg.ticks.enabled:
        console.print("[yellow]ticks.enabled is false — nothing recorded.[/yellow]")
        return
    console.print("Loading active markets to group related tokens…")
    markets = GammaClient(cfg).fetch_active_markets()
    label: dict[str, str] = {}
    by_event: dict[str, list[str]] = {}
    for m in markets:
        if not m.event_id or len(m.clob_token_ids) < 2:
            continue
        tok = m.clob_token_ids[0]           # the Yes token represents the market
        label[tok] = m.question
        by_event.setdefault(m.event_id, []).append(tok)

    store = TickStore(cfg.ticks.db_path)
    rows: list[LeadLag] = []
    for tokens in by_event.values():
        if len(tokens) < 2:
            continue
        series = store.token_series(tokens)
        if len(series) >= 2:
            rows += analyze(series, step_sec=step_sec)
    store.close()

    if not rows:
        console.print("[dim]No lead-lag signal yet — needs several markets of the "
                      "same event watched together for a while.[/dim]")
        return
    t = Table(title="Lead-lag radar (a laggard reacts LATE to its leader)")
    for col in ("Leader", "Laggard", f"Lag ({step_sec:.0f}s steps)", "Return corr"):
        t.add_column(col)
    for r in rows[:20]:
        t.add_row(label.get(r.leader, r.leader[:20])[:40],
                  label.get(r.laggard, r.laggard[:20])[:40],
                  str(r.lag_steps), f"{r.correlation:.2f}")
    console.print(t)
    console.print("[dim]Quote the laggard just after the leader moves — you are "
                  "reacting to public info faster than the slow book, not "
                  "predicting. Corr is on returns, so a shared trend can't fake "
                  "it.[/dim]")
