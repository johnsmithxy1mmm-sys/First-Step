"""Research loop: realistic fills, Sharpe allocation, walk-forward tuning.

Pure and testable. Turns the bot from a fixed set of rules into something that
can measure itself and propose (never auto-apply) better parameters.

  simulate_maker_fill  — queue + trade-through fill model (replaces the naive
                         "our bid filled at our price").
  sharpe_allocation    — capital weights across strategies by realized Sharpe
                         (a slow bandit with floor/cap).
  walk_forward_grid    — out-of-sample parameter search; returns ranked params.
  tune_edge_ratio      — applies the search to a backtest report (edge threshold).

The book-snapshot recorder (replay.py) is the tick store these feed on.
"""

from __future__ import annotations

from itertools import product
from statistics import mean, pstdev


def simulate_maker_fill(price: float, side: str, trades: list[tuple[float, float]],
                        our_size: float, queue_ahead: float = 0.0) -> float:
    """Realistic maker fill: a resting order at `price` fills only when the market
    trades through it AND after the size queued ahead of us is consumed.

    trades: chronological (trade_price, trade_size). Returns filled size.
    """
    remaining_queue = queue_ahead
    filled = 0.0
    for tp, tsize in trades:
        crosses = (side == "BUY" and tp <= price) or (side == "SELL" and tp >= price)
        if not crosses:
            continue
        vol = tsize
        if remaining_queue > 0:                      # our queue position first
            eaten = min(remaining_queue, vol)
            remaining_queue -= eaten
            vol -= eaten
        take = min(vol, our_size - filled)
        filled += take
        if filled >= our_size:
            break
    return filled


def sharpe_allocation(pnl_series: dict[str, list[float]],
                      floor: float = 0.05, cap: float = 0.60) -> dict[str, float]:
    """Capital weights ∝ max(Sharpe, 0), clamped to [floor, cap], normalized.

    pnl_series: strategy -> cumulative PnL checkpoints. Steady earners get more.
    """
    raw: dict[str, float] = {}
    for strategy, series in pnl_series.items():
        rets = [series[i + 1] - series[i] for i in range(len(series) - 1)]
        if len(rets) < 2:
            raw[strategy] = 0.0
            continue
        sd = pstdev(rets) or 1e-9
        raw[strategy] = max(mean(rets) / sd, 0.0)
    total = sum(raw.values())
    if total <= 0:
        n = len(raw) or 1
        return {s: 1.0 / n for s in raw}
    weights = {s: v / total for s, v in raw.items()}
    weights = {s: min(max(w, floor), cap) for s, w in weights.items()}
    norm = sum(weights.values())
    return {s: w / norm for s, w in weights.items()}


def walk_forward_grid(param_grid: dict[str, list], evaluate, n_splits: int = 3
                      ) -> list[tuple[dict, float]]:
    """Rank parameter combos by mean out-of-sample score.

    evaluate(params, split_index) -> score (higher is better). Splits are the
    caller's walk-forward folds. Returns [(params, mean_score)] best-first.
    """
    keys = list(param_grid)
    results: list[tuple[dict, float]] = []
    for combo in product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        scores = [evaluate(params, i) for i in range(n_splits)]
        results.append((params, mean(scores) if scores else 0.0))
    results.sort(key=lambda r: r[1], reverse=True)
    return results


def tune_edge_ratio(report, ratios: list[float], n_splits: int = 3
                    ) -> list[tuple[dict, float]]:
    """Walk-forward search over the longshot edge threshold on a backtest report.

    Splits the resolved rows into folds; scores each ratio by out-of-sample ROI.
    Proposal only — the caller decides whether to adopt it.
    """
    rows = report.rows
    fold = max(len(rows) // n_splits, 1)

    def evaluate(params, i):
        oos = rows[i * fold:(i + 1) * fold]
        if not oos:
            return 0.0
        trades = [r for r in oos if r.p_entry > 0
                  and r.p_est / r.p_entry >= params["min_edge_ratio"]]
        if not trades:
            return 0.0
        invested = 100.0 * len(trades)
        payout = sum(100.0 / r.p_entry for r in trades if r.won)
        return (payout - invested) / invested

    return walk_forward_grid({"min_edge_ratio": ratios}, evaluate, n_splits)
