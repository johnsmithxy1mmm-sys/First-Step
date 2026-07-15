"""Orchestrator: dry-run (default) / live / backtest modes, cycles, graceful shutdown."""

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
from .calibration import MarkoutFeedback, TailBiasCalibrator
from .clob import ClobReader, Trader
from .config import BotConfig
from .crossmarket import CrossMarketScanner
from .estimator import Estimator
from .executor import Executor
from .fade import FadeStrategy
from .gamma import GammaClient
from .ledger import Ledger
from .logging_setup import setup_logging
from .marketmaker import MarketMaker
from .models import Estimate, Market
from .monitor import Dashboard, alert
from .niche import NicheWatcher
from .portfolio import Portfolio
from .resolution import ResolutionAlpha
from .risk import KillSwitch
from .risk2 import MarketDataGuard, StrategyCircuitBreaker
from .ruleslawyer import RulesLawyer
from .satellite import BTC5mSatellite
from .scanner import Scanner
from .smartmoney import SmartMoneySignal, SmartMoneyTracker
from .ws_feed import WSFeed

log = logging.getLogger(__name__)


class Bot:
    def __init__(self, cfg: BotConfig, mode: str):
        self.cfg = cfg
        self.mode = mode
        self.gamma = GammaClient(cfg)
        self.clob = ClobReader(cfg)
        self.scanner = Scanner(cfg, self.clob)
        # Smart-money signal (iteration 7): fed into the ensemble when enabled.
        self.smart_signal = SmartMoneySignal(cfg.smart_money.signal_max_confidence)
        self.estimator = Estimator(
            cfg, extra_signals=[self.smart_signal] if cfg.smart_money.as_signal else None)
        self.ledger = Ledger(cfg.runtime.db_path)
        self.portfolio = Portfolio(cfg, self.ledger, mode)
        self.trader = Trader(cfg) if mode == "live" else None
        self.executor = Executor(cfg, self.ledger, self.clob, self.trader, mode)
        # Self-calibration: learned from resolutions/markout, refreshed by a job.
        self.bias_calibrator = TailBiasCalibrator(prior=cfg.fade.bias_discount)
        self.markout_feedback = MarkoutFeedback()
        self.fade = FadeStrategy(cfg, self.ledger, self.portfolio, self.executor, mode,
                                 calibrator=self.bias_calibrator)
        self.dashboard = Dashboard()
        self.errors: list[str] = []
        # WS order-book feed: "instant" for MM and exits; gap-detect -> kill-switch.
        self.ws: WSFeed | None = None
        if cfg.ws.enabled:
            self.ws = WSFeed(cfg.ws.url,
                             on_disconnect=self._on_ws_disconnect,
                             staleness_kill_sec=cfg.risk.ws_staleness_kill_sec,
                             ping_interval_sec=cfg.ws.ping_interval_sec)
        # Strategies: MM core, arbitrage alerts, cross-platform, niches, satellite.
        self.arb = ArbitrageScanner(cfg, self.ledger, self.clob, self.trader, mode)
        self.mm = MarketMaker(cfg, self.ledger, self.clob, self.trader, mode,
                              top_source=(self.ws.top if self.ws else None),
                              feedback=self.markout_feedback)
        self.resolution = ResolutionAlpha(cfg, self.ledger, self.clob, self.trader, mode)
        self.cross = CrossMarketScanner(cfg)
        self.niche = NicheWatcher(cfg, self.ledger)
        self.smart_money = SmartMoneyTracker(cfg, self.ledger, self.niche)
        self.rules_lawyer = RulesLawyer(cfg, self.ledger)
        self.satellite = BTC5mSatellite(cfg, self.ledger, self.clob, self.trader, mode)
        # Risk 2.0: data-anomaly guard + per-strategy circuit breaker.
        self.data_guard = MarketDataGuard()
        self.breaker = StrategyCircuitBreaker()
        # Kill-switch: bulk-cancel + halt. State recovery is via reconcile.
        self.killswitch = KillSwitch(cfg, self.ledger, mode,
                                     cancel_all=self._cancel_everything, alert=alert)
        # Active-markets cache: refreshed by the main cycle; fast strategies take
        # metadata from here and exact prices from WS / live order books.
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
            self._cancel_everything()   # pull all quotes
        except Exception:
            log.exception("shutdown cancel")
        if self.ws is not None:
            self.ws.stop()
        self.ledger.close()

    # --- one trading cycle ---

    def cycle(self) -> None:
        log.info("=== cycle started (mode=%s) ===", self.mode)
        self.errors.clear()
        top: list[Estimate] = []
        marks: dict[str, float] = {}

        try:
            markets = self.gamma.fetch_active_markets()
            self.markets_cache = markets
        except Exception as exc:
            self._error(f"gamma: {exc}")
            return

        # Risk 2.0 data guard: corrupt feed -> pause trading, don't act on it.
        problems = self.data_guard.problems(markets)
        if problems:
            self.killswitch.trip_pause("market data anomaly: " + "; ".join(problems))

        try:
            self.niche.cycle(markets)  # #5: alerts on new markets in niches
        except Exception as exc:
            self._error(f"niche: {exc}")

        # Marks for open positions (needed for equity/drawdown/exits).
        positions = self.ledger.open_positions(self.mode)
        for p in positions:
            book = self.clob.order_book(p.token_id)
            if book is not None and book.best_bid > 0:
                marks[p.token_id] = book.best_bid
        self._settle_resolutions(markets, positions)

        observe_only = self.portfolio.observe_only(marks)
        if observe_only:
            alert("Polymarket bot: KILL-SWITCH — drawdown above limit, observe-only mode.")

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
            log.info("estimates: %d, passing edge threshold: %d", len(estimates), len(qualifying))

            if not observe_only and self.killswitch.trading_allowed:
                if self.breaker.allows("longshot"):
                    self._enter_positions(qualifying)
                if self.breaker.allows("fade"):
                    # Fade buys NO — YES-side depth (verify_depth) is irrelevant
                    # to it. Feed it candidates BEFORE the depth check; the
                    # executor checks the NO book when placing the limit order.
                    fade_candidates = self.scanner.first_level_filter(markets)
                    fade_estimates = self.estimator.estimate_all(fade_candidates, markets)
                    self.fade.cycle(fade_estimates)
                self._exit_positions(marks)
        except Exception as exc:
            self._error(f"cycle: {exc}")
            log.exception("cycle failure")

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
                msg = (f"ENTRY [{self.mode}] {result.avg_price:.4f} x {result.filled_size:,.0f} "
                       f"= ${result.avg_price * result.filled_size:,.2f} "
                       f"(edge {est.edge_ratio:.2f}) [{c.outcome}] {c.market.question[:70]}")
                log.info(msg)
                alert(msg)
        log.info("entries this cycle: %d", entered)

    def _exit_positions(self, marks: dict[str, float]) -> None:
        for position in self.ledger.open_positions(self.mode):
            mark = marks.get(position.token_id)
            if not mark:
                continue
            exit_plan = self.portfolio.exit_plan(position, mark)
            if exit_plan is None:
                continue
            size, min_price = exit_plan
            # A pseudo-estimate is enough to sell: the executor reads price from the book.
            from .models import Candidate
            est = Estimate(
                candidate=Candidate(
                    market=self._market_stub(position),
                    outcome_index=0, token_id=position.token_id, p_mkt=mark),
                p_mkt=mark, p_est=mark, signals=[],
            )
            result = self.executor.execute_sell(est_to_plan(est, position.category), size, min_price)
            if result.status == "filled":
                msg = (f"TAKE-PROFIT [{self.mode}] sold {size:,.0f} at {result.avg_price:.4f} "
                       f"(entry {position.avg_price:.4f}) — {position.question[:60]}")
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
        """Records resolutions for open positions whose market has closed."""
        by_id = {m.id: m for m in markets}
        for p in positions:
            m = by_id.get(p.market_id)
            if m is not None and not m.closed:
                continue
            # Market not among active ones: checking closure per-market via Gamma
            # isn't cheap — use position data only if the market is clearly closed.
            if m is None:
                continue
            winner = m.resolved_winner_index()
            if winner is None:
                continue
            won = p.token_id == (m.clob_token_ids[winner] if winner < len(m.clob_token_ids) else "")
            self.ledger.record_resolution(p.token_id, p.market_id, won)
            msg = (f"RESOLUTION [{self.mode}] {'WIN' if won else 'loss'}: "
                   f"{p.question[:60]} ({p.size:,.0f} sh at {p.avg_price:.4f})")
            log.info(msg)
            alert(msg)

    def _error(self, text: str) -> None:
        self.errors.append(text)
        log.error(text)

    # --- extra-strategy jobs (own intervals, isolated errors) ---

    def arb_job(self) -> None:
        """#1: neg-risk baskets. Prices are checked against live order books."""
        if not self.markets_cache:
            return
        try:
            found = self.arb.cycle(self.markets_cache)
            for a in found[:3]:
                warn = " ⚠️ suspect: verify basket completeness" if a.suspect else ""
                will_execute = self.cfg.arbitrage.execute and not a.suspect
                note = "" if will_execute else " (execute off)"
                alert(f"ARBITRAGE {a.side} basket \"{a.event_title[:60]}\": "
                      f"NET after fees +{a.net_profit_pct * 100:.2f}% "
                      f"(gross +{a.profit_pct * 100:.1f}%), "
                      f"depth {a.max_sets_by_depth()} sets{warn}{note}")
        except Exception:
            log.exception("arb job")

    def mm_job(self) -> None:
        """Core: market making. Blocked by the kill-switch and observe-only."""
        if not self.markets_cache:
            return
        try:
            if self.ws is not None and self.ws.healthy:
                self.killswitch.on_ws_recovered()
            if not self.killswitch.trading_allowed or self.portfolio.observe_only() \
                    or not self.breaker.allows("mm"):
                self.mm.shutdown()
                return
            quotes = self.mm.cycle(self.markets_cache)
            # WS subscription to tokens of quoted markets + open positions.
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
        """Satellite btc_5m_ta (disabled by default)."""
        try:
            if self.killswitch.trading_allowed:
                self.satellite.cycle()
        except Exception:
            log.exception("satellite job")

    def smart_money_job(self) -> None:
        """#5: alerts when strong wallets enter a market; refresh the signal."""
        try:
            self.smart_money.cycle()
            if self.cfg.smart_money.as_signal:
                self.smart_signal.set_hot(self.smart_money.build_hot_tokens())
        except Exception:
            log.exception("smart-money job")

    def rules_lawyer_job(self) -> None:
        """#4: LLM flags headline-vs-rules discrepancies on fresh liquid markets."""
        if not self.markets_cache:
            return
        try:
            self.rules_lawyer.cycle(self.markets_cache)
        except Exception:
            log.exception("rules-lawyer job")

    def risk_job(self) -> None:
        """Kill-switch checks: daily stop, drawdown, reconcile with the exchange."""
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
            # Per-strategy circuit breaker: a losing streak disables that strategy.
            was = set(self.breaker.disabled)
            self.breaker.update(self.ledger.realized_pnl_by_strategy(self.mode))
            for s in self.breaker.disabled - was:
                alert(f"CIRCUIT BREAKER: strategy '{s}' disabled after a losing streak.")
        except Exception:
            log.exception("risk job")

    MARKOUT_HORIZONS_SEC = (60, 600)

    def markout_job(self) -> None:
        """Markout measurement: where price is 1 and 10 minutes after each fill."""
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

    def resolution_job(self) -> None:
        """#Resolution alpha: near-riskless carry on effectively-decided markets."""
        if not self.markets_cache or not self.breaker.allows("resolution"):
            return
        try:
            self.resolution.cycle(self.markets_cache)
        except Exception:
            log.exception("resolution job")

    def calibration_job(self) -> None:
        """Refit the self-calibrators from the ledger (fade bias, MM markout)."""
        try:
            self.fade.refresh_calibration()
            self.mm.refresh_feedback()
        except Exception:
            log.exception("calibration job")

    def digest_job(self) -> None:
        """Telegram digest: PnL, inventory, per-strategy attribution."""
        try:
            equity = self.portfolio.equity()
            positions = self.ledger.open_positions(self.mode)
            pnl = self.ledger.realized_pnl_by_strategy(self.mode)
            pnl_lines = "\n".join(f"  {k}: {v:+,.2f}" for k, v in pnl.items()) or "  —"
            alert(f"Digest [{self.mode}]\n"
                  f"Equity: ${equity:,.2f} | drawdown {self.portfolio.drawdown() * 100:.1f}%\n"
                  f"Open positions: {len(positions)} "
                  f"(${sum(p.cost_usd for p in positions):,.2f})\n"
                  f"Realized PnL by strategy:\n{pnl_lines}\n"
                  f"Kill-switch: {'HALT: ' + self.killswitch.reason if self.killswitch.halted else 'normal'}")
        except Exception:
            log.exception("digest job")

    def cross_job(self) -> None:
        """#2: divergences with external venues — alerts only."""
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
        description="Barbell on the mispricing of Polymarket tail outcomes",
    )
    parser.add_argument("--mode",
                        choices=("dry-run", "paper", "live", "backtest",
                                 "record-books", "replay", "report", "diagnose",
                                 "autotune"),
                        default="dry-run",
                        help="dry-run -> paper -> live (manual promotion only); "
                             "backtest/record-books/replay — offline phases")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--once", action="store_true", help="one cycle and exit")
    parser.add_argument("--report-mode", default="paper",
                        choices=("dry-run", "paper", "live"),
                        help="whose data to show in --mode report")
    parser.add_argument("--minutes", type=float, default=2880,
                        help="record-books duration (default 48h)")
    parser.add_argument("--i-understand-the-risk", action="store_true",
                        dest="risk_ack", help="required flag for --mode live")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    # Look for .env both next to the package and up from the current directory.
    from .config import PACKAGE_DIR
    load_dotenv(PACKAGE_DIR / ".env")
    load_dotenv()
    cfg = BotConfig.load(args.config)
    setup_logging(cfg.runtime.log_path, args.log_level)

    if args.mode == "backtest":
        report = backtest_mod.run_backtest(cfg)
        backtest_mod.print_report(report, cfg)
        return

    if args.mode == "diagnose":
        from .diagnose import run_diagnose
        run_diagnose(cfg)
        return

    if args.mode == "autotune":
        from .research import tune_edge_ratio
        report = backtest_mod.run_backtest(cfg)
        ranked = tune_edge_ratio(report, [1.5, 2.0, 2.5, 3.0, 4.0])
        log.info("Walk-forward edge-ratio search (out-of-sample ROI), proposal only:")
        for params, score in ranked:
            log.info("  min_edge_ratio=%.1f -> mean OOS ROI %+.1f%%",
                     params["min_edge_ratio"], score * 100)
        if ranked:
            log.info("Best: min_edge_ratio=%.1f (current: %.1f). Apply manually if it holds.",
                     ranked[0][0]["min_edge_ratio"], cfg.estimator.min_edge_ratio)
        return

    if args.mode == "report":
        from .analytics import compute_report, print_report
        ledger = Ledger(cfg.runtime.db_path)
        try:
            # Report for the mode the data accumulated in (paper by default).
            print_report(compute_report(ledger, args.report_mode))
        finally:
            ledger.close()
        return

    snaps_db = str(Path(cfg.runtime.db_path).parent / "book_snaps.sqlite")
    if args.mode == "record-books":
        from .replay import BookRecorder
        n = BookRecorder(cfg, snaps_db).record(minutes=args.minutes)
        log.info("snapshots recorded: %d -> %s", n, snaps_db)
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
            sys.exit("live mode requires the explicit --i-understand-the-risk flag")
        if not os.environ.get("POLYMARKET_PRIVATE_KEY"):
            sys.exit("POLYMARKET_PRIVATE_KEY not set — see .env.example")
        log.warning("LIVE MODE: real money.")

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
        log.info("received signal %s — shutting down", signum)
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
    if cfg.resolution.enabled:
        scheduler.add_job(bot.resolution_job, "interval",
                          seconds=cfg.resolution.interval_sec,
                          max_instances=1, coalesce=True)
    if cfg.ruleslawyer.enabled:
        scheduler.add_job(bot.rules_lawyer_job, "interval", minutes=20,
                          max_instances=1, coalesce=True)
    scheduler.add_job(bot.risk_job, "interval",
                      seconds=cfg.risk.reconcile_interval_sec,
                      max_instances=1, coalesce=True)
    scheduler.add_job(bot.digest_job, "interval", hours=6,
                      max_instances=1, coalesce=True)
    scheduler.add_job(bot.markout_job, "interval", seconds=30,
                      max_instances=1, coalesce=True)
    scheduler.add_job(bot.calibration_job, "interval", minutes=30,
                      max_instances=1, coalesce=True)
    scheduler.start()
    bot.calibration_job()  # warm the calibrators from any prior data
    bot.cycle()  # first cycle immediately

    stop_event.wait()
    scheduler.shutdown(wait=True)
    bot.close()
    log.info("stopped cleanly.")


if __name__ == "__main__":
    main()
