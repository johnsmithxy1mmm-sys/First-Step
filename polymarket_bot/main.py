"""Оркестратор: режимы dry-run (дефолт) / live / backtest, циклы, graceful shutdown."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

from . import backtest as backtest_mod
from .arbitrage import ArbitrageScanner
from .clob import ClobReader, Trader
from .config import BotConfig
from .crossmarket import CrossMarketScanner
from .estimator import Estimator
from .executor import Executor
from .gamma import GammaClient
from .ledger import Ledger
from .logging_setup import setup_logging
from .marketmaker import MarketMaker
from .models import Estimate, Market
from .monitor import Dashboard, alert
from .niche import NicheWatcher
from .portfolio import Portfolio
from .risk import KillSwitch
from .satellite import BTC5mSatellite
from .scanner import Scanner
from .smartmoney import SmartMoneyTracker
from .ws_feed import WSFeed

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
        # WS-фид стаканов: «мгновенно» для MM и выходов; gap-detect → kill-switch.
        self.ws: WSFeed | None = None
        if cfg.ws.enabled:
            self.ws = WSFeed(cfg.ws.url,
                             on_disconnect=self._on_ws_disconnect,
                             staleness_kill_sec=cfg.risk.ws_staleness_kill_sec,
                             ping_interval_sec=cfg.ws.ping_interval_sec)
        # Стратегии: MM-ядро, арбитраж-алерты, кросс-платформа, ниши, сателлит.
        self.arb = ArbitrageScanner(cfg, self.ledger, self.clob, self.trader, mode)
        self.mm = MarketMaker(cfg, self.ledger, self.clob, self.trader, mode,
                              top_source=(self.ws.top if self.ws else None))
        self.cross = CrossMarketScanner(cfg)
        self.niche = NicheWatcher(cfg, self.ledger)
        self.smart_money = SmartMoneyTracker(cfg, self.ledger, self.niche)
        self.satellite = BTC5mSatellite(cfg, self.ledger, self.clob, self.trader, mode)
        # Kill-switch: bulk-cancel + halt. Восстановление стейта — reconcile.
        self.killswitch = KillSwitch(cfg, self.ledger, mode,
                                     cancel_all=self._cancel_everything, alert=alert)
        # Кэш активных рынков: обновляется основным циклом, быстрые стратегии
        # берут метаданные отсюда, а точные цены — из WS/живых стаканов.
        self.markets_cache: list[Market] = []

    def _cancel_everything(self) -> None:
        self.mm.shutdown()
        if self.trader is not None:
            try:
                self.trader.cancel_all()
            except Exception:
                log.exception("cancel_all")

    def _on_ws_disconnect(self, gap_sec: float) -> None:
        self.killswitch.on_ws_disconnect(gap_sec)

    def close(self) -> None:
        try:
            self._cancel_everything()   # снять все котировки
        except Exception:
            log.exception("shutdown cancel")
        if self.ws is not None:
            self.ws.stop()
        self.ledger.close()

    # --- один торговый цикл ---

    def cycle(self) -> None:
        log.info("=== цикл начат (mode=%s) ===", self.mode)
        self.errors.clear()
        top: list[Estimate] = []
        marks: dict[str, float] = {}

        try:
            markets = self.gamma.fetch_active_markets()
            self.markets_cache = markets
        except Exception as exc:
            self._error(f"gamma: {exc}")
            return

        try:
            self.niche.cycle(markets)  # №5: алерты о новых рынках в нишах
        except Exception as exc:
            self._error(f"niche: {exc}")

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

    # --- джобы дополнительных стратегий (свои интервалы, изолированные ошибки) ---

    def arb_job(self) -> None:
        """№1: neg-risk корзины. Цены проверяются по живым стаканам."""
        if not self.markets_cache:
            return
        try:
            found = self.arb.cycle(self.markets_cache)
            for a in found[:3]:
                warn = " ⚠️ подозрительно: проверьте полноту корзины" if a.suspect else ""
                will_execute = self.cfg.arbitrage.execute and not a.suspect
                note = "" if will_execute else " (execute выключен)"
                alert(f"АРБИТРАЖ {a.side}-корзина «{a.event_title[:60]}»: "
                      f"NET после комиссий +{a.net_profit_pct * 100:.2f}% "
                      f"(gross +{a.profit_pct * 100:.1f}%), "
                      f"глубина {a.max_sets_by_depth()} комплектов{warn}{note}")
        except Exception:
            log.exception("arb job")

    def mm_job(self) -> None:
        """Ядро: маркет-мейкинг. Блокируется kill-switch'ем и observe-only."""
        if not self.markets_cache:
            return
        try:
            if self.ws is not None and self.ws.healthy:
                self.killswitch.on_ws_recovered()
            if not self.killswitch.trading_allowed or self.portfolio.observe_only():
                self.mm.shutdown()
                return
            quotes = self.mm.cycle(self.markets_cache)
            # WS-подписка на токены котируемых рынков + открытых позиций.
            if self.ws is not None:
                tokens: set[str] = set()
                for q in quotes:
                    tokens.update(q.market.clob_token_ids[:2])
                for p in self.ledger.open_positions(self.mode):
                    tokens.add(p.token_id)
                if tokens:
                    self.ws.watch(tokens)
        except Exception:
            log.exception("mm job")

    def satellite_job(self) -> None:
        """Сателлит btc_5m_ta (выключен по умолчанию)."""
        try:
            if self.killswitch.trading_allowed:
                self.satellite.cycle()
        except Exception:
            log.exception("satellite job")

    def smart_money_job(self) -> None:
        """№5: алерты, когда сильные кошельки заходят в рынок."""
        try:
            self.smart_money.cycle()
        except Exception:
            log.exception("smart-money job")

    def risk_job(self) -> None:
        """Проверки kill-switch: дневной стоп, просадка, reconcile с биржей."""
        try:
            equity = self.portfolio.equity()
            self.killswitch.check_daily_loss(equity)
            self.killswitch.check_drawdown(
                equity, max(self.ledger.high_water_mark(),
                            self.cfg.portfolio.bankroll_usd))
            if self.trader is not None:
                exchange_ids = {str(o.get("id") or o.get("orderID") or "")
                                for o in self.trader.open_orders()}
                exchange_ids.discard("")
                self.killswitch.reconcile(self.mm.local_order_ids(), exchange_ids)
        except Exception:
            log.exception("risk job")

    MARKOUT_HORIZONS_SEC = (60, 600)

    def markout_job(self) -> None:
        """Замер markout: где цена через 1 и 10 минут после каждого филла."""
        try:
            for horizon in self.MARKOUT_HORIZONS_SEC:
                for fill in self.ledger.fills_needing_markout(self.mode, horizon):
                    mid = None
                    if self.ws is not None:
                        top = self.ws.top(fill["token_id"])
                        if top is not None and top.mid > 0:
                            mid = top.mid
                    if mid is None:
                        book = self.clob.order_book(fill["token_id"])
                        if book is not None and book.mid > 0:
                            mid = book.mid
                    if mid:
                        self.ledger.record_markout(
                            fill["id"], fill["token_id"], horizon,
                            fill["price"], mid)
        except Exception:
            log.exception("markout job")

    def digest_job(self) -> None:
        """Telegram-дайджест: PnL, инвентарь, атрибуция по стратегиям."""
        try:
            equity = self.portfolio.equity()
            positions = self.ledger.open_positions(self.mode)
            pnl = self.ledger.realized_pnl_by_strategy(self.mode)
            pnl_lines = "\n".join(f"  {k}: {v:+,.2f}" for k, v in pnl.items()) or "  —"
            alert(f"Дайджест [{self.mode}]\n"
                  f"Equity: ${equity:,.2f} | просадка {self.portfolio.drawdown() * 100:.1f}%\n"
                  f"Открытых позиций: {len(positions)} "
                  f"(${sum(p.cost_usd for p in positions):,.2f})\n"
                  f"Реализованный PnL по стратегиям:\n{pnl_lines}\n"
                  f"Kill-switch: {'HALT: ' + self.killswitch.reason if self.killswitch.halted else 'норма'}")
        except Exception:
            log.exception("digest job")

    def cross_job(self) -> None:
        """№2: расхождения с внешними площадками — только алерты."""
        if not self.markets_cache:
            return
        try:
            for d in self.cross.cycle(self.markets_cache):
                alert(d.describe())
        except Exception:
            log.exception("crossmarket job")


def est_to_plan(est: Estimate, category: str):
    from .models import TradePlan
    return TradePlan(estimate=est, category=category, size_usd=0.0, limit_price_cap=1.0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="polymarket_bot",
        description="Барбелл на мисспрайсинге хвостовых исходов Polymarket",
    )
    parser.add_argument("--mode",
                        choices=("dry-run", "paper", "live", "backtest",
                                 "record-books", "replay", "report"),
                        default="dry-run",
                        help="dry-run -> paper -> live (переход только вручную); "
                             "backtest/record-books/replay — офлайн-фазы")
    parser.add_argument("--config", default=None, help="путь к config.yaml")
    parser.add_argument("--once", action="store_true", help="один цикл и выход")
    parser.add_argument("--report-mode", default="paper",
                        choices=("dry-run", "paper", "live"),
                        help="чьи данные показывать в --mode report")
    parser.add_argument("--minutes", type=float, default=2880,
                        help="длительность record-books (по умолчанию 48ч)")
    parser.add_argument("--i-understand-the-risk", action="store_true",
                        dest="risk_ack", help="обязательный флаг для --mode live")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    # .env ищем и рядом с пакетом, и от текущей директории вверх.
    from .config import PACKAGE_DIR
    load_dotenv(PACKAGE_DIR / ".env")
    load_dotenv()
    cfg = BotConfig.load(args.config)
    setup_logging(cfg.runtime.log_path, args.log_level)

    if args.mode == "backtest":
        report = backtest_mod.run_backtest(cfg)
        backtest_mod.print_report(report, cfg)
        return

    if args.mode == "report":
        from .analytics import compute_report, print_report
        ledger = Ledger(cfg.runtime.db_path)
        try:
            # Отчёт по тому режиму, в котором копились данные (paper по умолчанию).
            print_report(compute_report(ledger, args.report_mode))
        finally:
            ledger.close()
        return

    snaps_db = str(Path(cfg.runtime.db_path).parent / "book_snaps.sqlite")
    if args.mode == "record-books":
        from .replay import BookRecorder
        n = BookRecorder(cfg, snaps_db).record(minutes=args.minutes)
        log.info("записано снапшотов: %d -> %s", n, snaps_db)
        return
    if args.mode == "replay":
        from . import replay as replay_mod
        ledger = Ledger(cfg.runtime.db_path)
        try:
            replay_mod.print_report(replay_mod.replay(cfg, snaps_db, ledger))
        finally:
            ledger.close()
        return

    if args.mode == "live":
        if not args.risk_ack:
            sys.exit("live-режим требует явного флага --i-understand-the-risk")
        if not os.environ.get("POLYMARKET_PRIVATE_KEY"):
            sys.exit("POLYMARKET_PRIVATE_KEY не задан — см. .env.example")
        log.warning("LIVE-РЕЖИМ: реальные деньги.")

    bot = Bot(cfg, args.mode)
    if bot.ws is not None and not args.once:
        bot.ws.start()
    if args.once:
        try:
            bot.cycle()
            bot.arb_job()
            bot.mm_job()
            bot.cross_job()
            bot.smart_money_job()
            bot.risk_job()
            bot.markout_job()
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
    if cfg.arbitrage.enabled:
        scheduler.add_job(bot.arb_job, "interval",
                          seconds=cfg.arbitrage.interval_sec,
                          max_instances=1, coalesce=True)
    if cfg.market_maker.enabled:
        scheduler.add_job(bot.mm_job, "interval",
                          seconds=cfg.market_maker.interval_sec,
                          max_instances=1, coalesce=True)
    if cfg.crossmarket.enabled:
        scheduler.add_job(bot.cross_job, "interval",
                          minutes=cfg.crossmarket.interval_min,
                          max_instances=1, coalesce=True)
    if cfg.smart_money.enabled:
        scheduler.add_job(bot.smart_money_job, "interval",
                          minutes=cfg.smart_money.interval_min,
                          max_instances=1, coalesce=True)
    if cfg.satellite.enabled:
        scheduler.add_job(bot.satellite_job, "interval", seconds=5,
                          max_instances=1, coalesce=True)
    scheduler.add_job(bot.risk_job, "interval",
                      seconds=cfg.risk.reconcile_interval_sec,
                      max_instances=1, coalesce=True)
    scheduler.add_job(bot.digest_job, "interval", hours=6,
                      max_instances=1, coalesce=True)
    scheduler.add_job(bot.markout_job, "interval", seconds=30,
                      max_instances=1, coalesce=True)
    scheduler.start()
    bot.cycle()  # первый цикл сразу

    stop_event.wait()
    scheduler.shutdown(wait=True)
    bot.close()
    log.info("остановлен корректно.")


if __name__ == "__main__":
    main()
