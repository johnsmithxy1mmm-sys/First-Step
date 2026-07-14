"""Backtest on closed markets: calibrate p_est before a single live dollar.

Method:
  1. Take closed Gamma markets with unambiguous resolution (price ~0/1).
  2. For each, sample the price lookback_days_before_end days before the end
     via CLOB /prices-history — the "entry price" the bot would have seen.
  3. Run the offline signals (base rates + coherence over the sampled event
     prices); LLM is off by default in backtest (cost + future-knowledge leak
     from training data).
  4. Compute: market Brier vs model Brier, a bucketed calibration table, and
     simulated PnL of a "buy everything with edge >= threshold" strategy.

Limits (honestly): momentum is unavailable in backtest (no past daily deltas),
the fill model is optimistic (entry at the sampled price).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from .clob import ClobReader
from .config import BotConfig
from .estimator.base_rates import BaseRatesSignal, load_base_rates
from .estimator.ensemble import combine
from .gamma import GammaClient
from .models import Candidate, Market

log = logging.getLogger(__name__)


@dataclass
class BacktestRow:
    market: Market
    p_entry: float          # price lookback days before resolution
    p_est: float
    won: bool
    signals: list[str] = field(default_factory=list)


@dataclass
class BacktestReport:
    rows: list[BacktestRow] = field(default_factory=list)

    def brier(self, use_model: bool) -> float | None:
        if not self.rows:
            return None
        total = sum(
            ((r.p_est if use_model else r.p_entry) - float(r.won)) ** 2
            for r in self.rows
        )
        return total / len(self.rows)

    def buckets(self) -> list[tuple[str, int, float, float]]:
        """[(range, n, avg price, actual win frequency)]."""
        edges = [0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 1.0]
        grouped: dict[int, list[BacktestRow]] = defaultdict(list)
        for r in self.rows:
            for i in range(len(edges) - 1):
                if edges[i] <= r.p_entry < edges[i + 1]:
                    grouped[i].append(r)
                    break
        out = []
        for i in sorted(grouped):
            rows = grouped[i]
            out.append((
                f"[{edges[i]:.2f}, {edges[i + 1]:.2f})",
                len(rows),
                sum(r.p_entry for r in rows) / len(rows),
                sum(r.won for r in rows) / len(rows),
            ))
        return out

    def strategy_pnl(self, min_edge_ratio: float, stake_usd: float = 100.0) -> dict:
        trades = [r for r in self.rows
                  if r.p_entry > 0 and r.p_est / r.p_entry >= min_edge_ratio]
        invested = stake_usd * len(trades)
        payout = sum(stake_usd / r.p_entry for r in trades if r.won)
        return {
            "trades": len(trades),
            "wins": sum(r.won for r in trades),
            "invested": invested,
            "payout": payout,
            "roi": (payout - invested) / invested if invested else 0.0,
        }


def run_backtest(cfg: BotConfig, gamma: GammaClient | None = None,
                 clob: ClobReader | None = None) -> BacktestReport:
    gamma = gamma or GammaClient(cfg)
    clob = clob or ClobReader(cfg)
    base_rates = BaseRatesSignal(load_base_rates(cfg.base_rates_path()))
    lookback_sec = int(cfg.backtest.lookback_days_before_end * 86400)

    markets = gamma.fetch_closed_markets(cfg.backtest.max_markets)
    report = BacktestReport()

    for m in markets:
        winner = m.resolved_winner_index()
        if winner is None or m.end_date is None or not m.clob_token_ids:
            continue
        end_ts = int(m.end_date.timestamp())
        token = m.clob_token_ids[0]  # Yes side
        history = clob.price_history(token, end_ts - lookback_sec - 86400,
                                     end_ts - lookback_sec + 86400)
        if not history:
            continue
        p_entry = history[-1][1]
        s = cfg.scanner
        if not s.price_min <= p_entry <= s.price_max:
            continue

        candidate = Candidate(market=m, outcome_index=0, token_id=token, p_mkt=p_entry)
        signals = []
        br = base_rates.evaluate(candidate, now=None)
        if br is not None:
            signals.append(br)
        est = combine(candidate, signals, cfg.estimator.market_anchor_confidence)
        report.rows.append(BacktestRow(
            market=m, p_entry=p_entry, p_est=est.p_est,
            won=(winner == 0),
            signals=[x.name for x in est.signals if x.name != "market"],
        ))

    return report


def print_report(report: BacktestReport, cfg: BotConfig) -> None:
    console = Console()
    n = len(report.rows)
    console.print(f"\n[bold]Backtest: {n} tail outcomes with price history[/bold]")
    if not n:
        console.print("Not enough data: raise backtest.max_markets "
                      "or relax the scanner filters.")
        return

    brier_mkt = report.brier(use_model=False)
    brier_model = report.brier(use_model=True)
    console.print(f"Market Brier: {brier_mkt:.5f}")
    console.print(f"Model Brier:  {brier_model:.5f} "
                  f"({'better' if brier_model < brier_mkt else 'WORSE'} than market)")

    t = Table(title="Calibration by entry-price bucket")
    for col in ("p range", "N", "Avg price", "Actual frequency", "Bias"):
        t.add_column(col)
    for rng, count, avg_p, freq in report.buckets():
        bias = "overpriced" if avg_p > freq else "underpriced"
        t.add_row(rng, str(count), f"{avg_p:.4f}", f"{freq:.4f}", bias)
    console.print(t)

    sim = report.strategy_pnl(cfg.estimator.min_edge_ratio)
    console.print(
        f"\nStrategy simulation (edge >= {cfg.estimator.min_edge_ratio:g}, $100/trade): "
        f"{sim['trades']} trades, {sim['wins']} wins, "
        f"ROI = {sim['roi'] * 100:+.1f}%"
    )
    console.print("[dim]Caveats: entry at the sampled price (optimistic), "
                  "momentum/LLM do not participate in the backtest.[/dim]")
