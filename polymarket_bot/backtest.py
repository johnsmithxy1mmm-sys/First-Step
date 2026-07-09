"""Бэктест на закрытых рынках: калибровка p_est до единого живого доллара.

Методика:
  1. Берём закрытые рынки из Gamma с однозначной резолюцией (цена ~0/1).
  2. Для каждого сэмплируем цену за lookback_days_before_end дней до конца
     через CLOB /prices-history — это «цена входа», которую видел бы бот.
  3. Прогоняем офлайн-сигналы (base rates + когерентность по сэмплированным
     ценам события); LLM в бэктесте выключен по умолчанию (стоимость + утечка
     будущего знания из обучающих данных).
  4. Считаем: Brier рынка vs Brier модели, таблицу калибровки по бакетам,
     симулированный PnL стратегии «покупать всё с edge ≥ порога».

Ограничения (честно): momentum в бэктесте недоступен (нет суточных дельт
прошлого), fill-модель оптимистична (вход по сэмплированной цене).
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
    p_entry: float          # цена за lookback дней до резолюции
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
        """[(диапазон, n, средняя цена, фактическая частота побед)]."""
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
        token = m.clob_token_ids[0]  # Yes-сторона
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
    console.print(f"\n[bold]Бэктест: {n} хвостовых исходов с историей цен[/bold]")
    if not n:
        console.print("Недостаточно данных: увеличьте backtest.max_markets "
                      "или ослабьте фильтры сканера.")
        return

    brier_mkt = report.brier(use_model=False)
    brier_model = report.brier(use_model=True)
    console.print(f"Brier рынка:  {brier_mkt:.5f}")
    console.print(f"Brier модели: {brier_model:.5f} "
                  f"({'лучше' if brier_model < brier_mkt else 'ХУЖЕ'} рынка)")

    t = Table(title="Калибровка по бакетам цены входа")
    for col in ("Диапазон p", "N", "Средняя цена", "Фактическая частота", "Bias"):
        t.add_column(col)
    for rng, count, avg_p, freq in report.buckets():
        bias = "переоценён" if avg_p > freq else "недооценён"
        t.add_row(rng, str(count), f"{avg_p:.4f}", f"{freq:.4f}", bias)
    console.print(t)

    sim = report.strategy_pnl(cfg.estimator.min_edge_ratio)
    console.print(
        f"\nСимуляция стратегии (edge >= {cfg.estimator.min_edge_ratio:g}, $100/сделка): "
        f"{sim['trades']} сделок, {sim['wins']} побед, "
        f"ROI = {sim['roi'] * 100:+.1f}%"
    )
    console.print("[dim]Оговорки: вход по сэмплированной цене (оптимистично), "
                  "momentum/LLM в бэктесте не участвуют.[/dim]")
