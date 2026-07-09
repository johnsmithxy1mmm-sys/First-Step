"""Оркестратор: режимы dry-run (дефолт) / live / backtest, циклы, graceful shutdown."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading

from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

from . import backtest as backtest_mod
from .clob import ClobReader, Trader
from .config import BotConfig
from .estimator import Estimator
from .executor import Executor
from .gamma import GammaClient
from .ledger import Ledger
from .logging_setup import setup_logging
from .models import Estimate
from .monitor import Dashboard, alert
from .portfolio import Portfolio
from .scanner import Scanner

log = logging.getLogger(__name__)


class Bot:
    def __init__(self, cfg: BotConfig, mode: str):
        self.cfg = cfg
        self.mode = mode
        self.gamma = GammaClient(cfg)
        self.clob = ClobReader(cfg)
        self.scanner = Scanner(cfg, self.clob)
        self.estimator = Estimator(cfg)
        self.ledger = Ledger(cfg.runtime.db_path)
        self.portfolio = Portfolio(cfg, self.ledger, mode)
        self.trader = Trader(cfg) if mode == "live" else None
        self.executor = Executor(cfg, self.ledger, self.clob, self.trader, mode)
        self.dashboard = Dashboard()
        self.errors: list[str] = []

    def close(self) -> None:
        self.ledger.close()

    # --- один торговый цикл ---

    def cycle(self) -> None:
        log.info("=== цикл начат (mode=%s) ===", self.mode)
        self.errors.clear()
        top: list[Estimate] = []
        marks: dict[str, float] = {}

        try:
            markets = self.gamma.fetch_active_markets()
        except Exception as exc:
            self._error(f"gamma: {exc}")
            return

        # Марки открытых позиций (нужны для equity/drawdown/выходов).
        positions = self.ledger.open_positions(self.mode)
        for p in positions:
            book = self.clob.order_book(p.token_id)
            if book is not None and book.best_bid > 0:
                marks[p.token_id] = book.best_bid
        self._settle_resolutions(markets, positions)

        observe_only = self.portfolio.observe_only(marks)
        if observe_only:
            alert("Polymarket bot: KILL-SWITCH — просадка выше лимита, режим observe-only.")

        try:
            candidates = self.scanner.scan(markets)
            estimates = self.estimator.estimate_all(candidates, markets)
            top = estimates[:20]
            qualifying = []
            for est in estimates:
                ok = self.estimator.qualifies(est)
                self.ledger.record_estimate(est, ok)
                if ok:
                    qualifying.append(est)
            log.info("оценок: %d, проходят порог edge: %d", len(estimates), len(qualifying))

            if not observe_only:
                self._enter_positions(qualifying)
                self._exit_positions(marks)
        except Exception as exc:
            self._error(f"cycle: {exc}")
            log.exception("сбой цикла")

        self.portfolio.snapshot(marks)
        self.dashboard.render(
            mode=self.mode,
            equity=self.portfolio.equity(marks),
            drawdown=self.portfolio.drawdown(marks),
            observe_only=observe_only,
            positions=self.ledger.open_positions(self.mode),
            marks=marks,
            top_estimates=top,
            errors=self.errors,
        )

    def _enter_positions(self, qualifying: list[Estimate]) -> None:
        entered = 0
        for est in qualifying:
            plan = self.portfolio.size_trade(est)
            if plan is None:
                continue
            result = self.executor.execute(plan)
            if result.status == "filled":
                entered += 1
                c = est.candidate
                msg = (f"ВХОД [{self.mode}] {result.avg_price:.4f} x {result.filled_size:,.0f} "
                       f"= ${result.avg_price * result.filled_size:,.2f} "
                       f"(edge {est.edge_ratio:.2f}) [{c.outcome}] {c.market.question[:70]}")
                log.info(msg)
                alert(msg)
        log.info("входов за цикл: %d", entered)

    def _exit_positions(self, marks: dict[str, float]) -> None:
        for position in self.ledger.open_positions(self.mode):
            mark = marks.get(position.token_id)
            if not mark:
                continue
            exit_plan = self.portfolio.exit_plan(position, mark)
            if exit_plan is None:
                continue
            size, min_price = exit_plan
            # Для продажи достаточно псевдо-оценки: executor берёт цену из книги.
            from .models import Candidate
            est = Estimate(
                candidate=Candidate(
                    market=self._market_stub(position),
                    outcome_index=0, token_id=position.token_id, p_mkt=mark),
                p_mkt=mark, p_est=mark, signals=[],
            )
            result = self.executor.execute_sell(est_to_plan(est, position.category), size, min_price)
            if result.status == "filled":
                msg = (f"ТЕЙК-ПРОФИТ [{self.mode}] продано {size:,.0f} по {result.avg_price:.4f} "
                       f"(вход {position.avg_price:.4f}) — {position.question[:60]}")
                log.info(msg)
                alert(msg)

    def _market_stub(self, position):
        from .models import Market
        return Market(
            id=position.market_id, question=position.question,
            outcomes=[position.outcome or "Yes", "No"],
            outcome_prices=[0.0, 0.0],
            clob_token_ids=[position.token_id, ""],
        )

    def _settle_resolutions(self, markets, positions) -> None:
        """Отмечает резолюции по открытым позициям, чей рынок закрылся."""
        by_id = {m.id: m for m in markets}
        for p in positions:
            m = by_id.get(p.market_id)
            if m is not None and not m.closed:
                continue
            # Рынка нет среди активных: проверяем закрытие точечно через Gamma нельзя
            # дёшево — используем данные позиции только если рынок явно закрыт.
            if m is None:
                continue
            winner = m.resolved_winner_index()
            if winner is None:
                continue
            won = p.token_id == (m.clob_token_ids[winner] if winner < len(m.clob_token_ids) else "")
            self.ledger.record_resolution(p.token_id, p.market_id, won)
            msg = (f"РЕЗОЛЮЦИЯ [{self.mode}] {'ВЫИГРЫШ' if won else 'проигрыш'}: "
                   f"{p.question[:60]} ({p.size:,.0f} шт по {p.avg_price:.4f})")
            log.info(msg)
            alert(msg)

    def _error(self, text: str) -> None:
        self.errors.append(text)
        log.error(text)


def est_to_plan(est: Estimate, category: str):
    from .models import TradePlan
    return TradePlan(estimate=est, category=category, size_usd=0.0, limit_price_cap=1.0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="polymarket_bot",
        description="Барбелл на мисспрайсинге хвостовых исходов Polymarket",
    )
    parser.add_argument("--mode", choices=("dry-run", "live", "backtest"),
                        default="dry-run")
    parser.add_argument("--config", default=None, help="путь к config.yaml")
    parser.add_argument("--once", action="store_true", help="один цикл и выход")
    parser.add_argument("--i-understand-the-risk", action="store_true",
                        dest="risk_ack", help="обязательный флаг для --mode live")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    load_dotenv()
    cfg = BotConfig.load(args.config)
    setup_logging(cfg.runtime.log_path, args.log_level)

    if args.mode == "backtest":
        report = backtest_mod.run_backtest(cfg)
        backtest_mod.print_report(report, cfg)
        return

    if args.mode == "live":
        if not args.risk_ack:
            sys.exit("live-режим требует явного флага --i-understand-the-risk")
        if not os.environ.get("POLYMARKET_PRIVATE_KEY"):
            sys.exit("POLYMARKET_PRIVATE_KEY не задан — см. .env.example")
        log.warning("LIVE-РЕЖИМ: реальные деньги.")

    bot = Bot(cfg, args.mode)
    if args.once:
        try:
            bot.cycle()
        finally:
            bot.close()
        return

    stop_event = threading.Event()

    def shutdown(signum, frame):  # noqa: ARG001
        log.info("получен сигнал %s — останавливаюсь", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    scheduler = BackgroundScheduler()
    scheduler.add_job(bot.cycle, "interval",
                      minutes=cfg.scanner.interval_minutes,
                      max_instances=1, coalesce=True)
    scheduler.start()
    bot.cycle()  # первый цикл сразу

    stop_event.wait()
    scheduler.shutdown(wait=True)
    bot.close()
    log.info("остановлен корректно.")


if __name__ == "__main__":
    main()
