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
from .arb_fastlane import ArbFastlane
from .arbitrage import ArbitrageScanner
from .chainarb import ChainArbitrage
from .alloc import StrategyAllocator
from .calibration import (CorrelationLearner, FillCalibrator, MarkoutFeedback,
                          PlattCalibrator, TailBiasCalibrator)
from .clob import ClobReader, Trader
from .config import BotConfig
from .crossmarket import CrossMarketScanner
from .estimator import Estimator
from .executor import Executor
from .fade import FadeStrategy
from .gamma import GammaClient
from .guardian import PositionGuardian
from .ledger import Ledger
from .logging_setup import setup_logging
from .marketmaker import MarketMaker
from .redeemer import Redeemer
from .sprintmaker import SprintMaker
from .models import Estimate, Market, Position
from .monitor import Dashboard, alert
from .niche import NicheWatcher
from .ops import MetricsServer, reload_config_inplace
from .portfolio import (Portfolio, classify_category, event_exposure_breakdown,
                        set_learned_correlation)
from .resolution import ResolutionAlpha
from .risk import KillSwitch
from .risk2 import MarketDataGuard, StrategyCircuitBreaker
from .ruleslawyer import RulesLawyer
from .satellite import BTC5mSatellite
from .scanner import Scanner
from .smartmoney import SmartMoneySignal, SmartMoneyTracker
from .telegram_control import TelegramControl
from .tickstore import TickStore
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
        self.platt = PlattCalibrator()
        self.estimator = Estimator(
            cfg, extra_signals=[self.smart_signal] if cfg.smart_money.as_signal else None,
            platt=self.platt if cfg.estimator.platt_enabled else None)
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
                             on_update=self.on_tick,   # event-driven fastlane
                             on_disconnect=self._on_ws_disconnect,
                             staleness_kill_sec=cfg.risk.ws_staleness_kill_sec,
                             ping_interval_sec=cfg.ws.ping_interval_sec)
        # Strategies: MM core, arbitrage alerts, cross-platform, niches, satellite.
        self.arb = ArbitrageScanner(cfg, self.ledger, self.clob, self.trader, mode)
        self.chain_arb = ChainArbitrage(cfg, self.ledger, self.clob, self.trader, mode)
        self.mm = MarketMaker(cfg, self.ledger, self.clob, self.trader, mode,
                              top_source=(self.ws.top if self.ws else None),
                              feedback=self.markout_feedback)
        # Short-dated MM (fast capital turnover): same engine, hours-horizon.
        self.sprint = SprintMaker(cfg, self.ledger, self.clob, self.trader, mode,
                                  top_source=(self.ws.top if self.ws else None),
                                  feedback=self.markout_feedback)
        self.resolution = ResolutionAlpha(cfg, self.ledger, self.clob, self.trader, mode)
        self.redeemer = Redeemer(cfg, self.trader, mode)
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
        # WS-fastlane caches: on_tick must NEVER hit the DB or the network.
        self._positions_by_token: dict[str, Position] = {}
        # Each quoting job contributes its tokens; the WS watches the UNION
        # (ws.watch replaces the desired set, so the jobs must not overwrite
        # each other's subscription).
        self._mm_watch: set[str] = set()
        self._sprint_watch: set[str] = set()
        # WS-triggered arbitrage: token -> structure key, plus the structures
        # themselves, registered by the polling jobs; on_tick flags a dirty
        # structure, the fastlane worker re-verifies it off the WS thread.
        self._basket_token_map: dict[str, tuple] = {}
        self._chain_token_map: dict[str, tuple] = {}
        self._arb_groups: dict[str, list[Market]] = {}
        self._chain_pairs: dict[tuple, tuple] = {}
        self.arb_fastlane = ArbFastlane(self._fastlane_check)
        self._observe_only = False
        # Per-strategy PnL checkpoints (one per digest) for Sharpe allocation.
        self._pnl_history: dict[str, list[float]] = {}
        # Learning loop: continuous tick/category recording feeds the learned
        # correlations; quote outcomes feed the fill calibrator; digest Sharpe
        # weights feed the acting allocator. All start as identity/prior.
        self.ticks: TickStore | None = None
        if cfg.ticks.enabled:
            self.ticks = TickStore(cfg.ticks.db_path, cfg.ticks.retention_days,
                                   cfg.ticks.flush_sec)
            self.ticks.start()
        self.corr_learner = CorrelationLearner()
        self.fill_calibrator = FillCalibrator()
        self.mm.fill_calibrator = self.fill_calibrator
        self.sprint.fill_calibrator = self.fill_calibrator
        self.allocator = StrategyAllocator(cfg)
        self._longshot_scale = 1.0
        # Two-way Telegram: control from the phone (started in the run path only).
        self.tg_control = TelegramControl(self._telegram_handlers())
        # Position guardian: early warning when a sold tail is materializing.
        self.guardian = PositionGuardian(cfg)

    def metrics_snapshot(self) -> dict:
        """Live state for /metrics and /health (never raises)."""
        try:
            positions = self.ledger.open_positions(self.mode)
            worst = sum(r[3] for r in event_exposure_breakdown(positions))
            return {
                "mode": self.mode,
                "equity": round(self.portfolio.equity(), 2),
                "drawdown": round(self.portfolio.drawdown(), 4),
                "positions": len(positions),
                "exposure": round(sum(p.cost_usd for p in positions), 2),
                "worst_case": round(worst, 2),
                "errors": len(self.errors),
                "halted": self.killswitch.halted,
                "pnl_by_strategy": self.ledger.realized_pnl_by_strategy(self.mode),
            }
        except Exception:
            return {"mode": self.mode, "halted": self.killswitch.halted}

    def _telegram_handlers(self) -> dict:
        """Slash-command handlers for two-way Telegram control."""
        def status() -> str:
            ks = self.killswitch
            state = ("HALT: " + ks.reason if ks.halted
                     else "PAUSED: " + ks.reason if ks.paused
                     else "observe-only" if self._observe_only else "active")
            positions = self.ledger.open_positions(self.mode)
            alloc = self.allocator.summary()
            return (f"mode {self.mode} | {state}\n"
                    f"equity ${self.portfolio.equity():,.2f} | "
                    f"drawdown {self.portfolio.drawdown() * 100:.1f}%\n"
                    f"open positions: {len(positions)} "
                    f"(${sum(p.cost_usd for p in positions):,.2f})"
                    + (f"\nsizing: {alloc}" if alloc else ""))

        def positions() -> str:
            ps = sorted(self.ledger.open_positions(self.mode),
                        key=lambda p: p.cost_usd, reverse=True)[:15]
            if not ps:
                return "no open positions"
            return "\n".join(f"[{p.outcome}] {p.question[:45]} — "
                             f"{p.size:,.0f} @ {p.avg_price:.3f} (${p.cost_usd:,.0f})"
                             for p in ps)

        def pnl() -> str:
            by = self.ledger.realized_pnl_by_strategy(self.mode)
            lines = "\n".join(f"  {k}: {v:+,.2f}" for k, v in sorted(by.items())) or "  —"
            caps = self.ledger.opportunity_stats(self.mode)
            cap_lines = "".join(
                f"\n  {c['strategy']}: {c['windows']} windows, "
                f"${c['edge_dollars'] or 0:,.0f} capacity" for c in caps)
            return f"Realized PnL:\n{lines}" + (
                f"\nOpportunity capacity:{cap_lines}" if cap_lines else "")

        def pause() -> str:
            self.killswitch.trip_pause("manual pause via Telegram", source="manual")
            return "⏸ paused — new orders blocked, quotes pulled. /resume to continue."

        def resume() -> str:
            self.killswitch.resume_from_pause("manual")
            allowed = self.killswitch.trading_allowed
            return ("▶ resumed — trading allowed." if allowed
                    else f"still blocked: {self.killswitch.reason}")

        def help_() -> str:
            return ("Commands:\n/status — equity, risk state, sizing\n"
                    "/positions — open book\n/pnl — realized PnL + capacity\n"
                    "/pause — stop new orders\n/resume — allow trading again")

        return {"status": status, "positions": positions, "pnl": pnl,
                "pause": pause, "resume": resume, "help": help_, "start": help_}

    def _cancel_everything(self) -> None:
        self.mm.shutdown()
        self.sprint.shutdown()
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
        self.arb_fastlane.stop()
        self.tg_control.stop()
        if self.ws is not None:
            self.ws.stop()
        if self.ticks is not None:
            self.ticks.close()
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
        # The pause source is "data" so a healthy WS cannot clear it; it lifts
        # only when the feed itself is clean again.
        problems = self.data_guard.problems(markets)
        if problems:
            self.killswitch.trip_pause("market data anomaly: " + "; ".join(problems),
                                       source="data")
        else:
            self.killswitch.resume_from_pause("data")

        try:
            self.niche.cycle(markets)  # #5: alerts on new markets in niches
        except Exception as exc:
            self._error(f"niche: {exc}")

        if self.ticks is not None:
            self._record_category_indexes(markets)

        # Heal legacy rows: trades recorded before the neg_risk column existed
        # default to 0, which blocks event netting. Live market metadata knows
        # which events ARE neg-risk — backfill so old baskets net properly.
        neg_ids = [m.id for m in markets if m.event_neg_risk or m.neg_risk]
        healed = self.ledger.backfill_neg_risk(neg_ids)
        if healed:
            log.info("ledger: backfilled neg_risk on %d legacy trade rows", healed)

        # Marks for open positions (needed for equity/drawdown/exits).
        positions = self.ledger.open_positions(self.mode)
        self._positions_by_token = {p.token_id: p for p in positions}
        for p in positions:
            book = self.clob.order_book(p.token_id)
            if book is not None and book.best_bid > 0:
                marks[p.token_id] = book.best_bid
        self._settle_resolutions(markets, positions)

        observe_only = self.portfolio.observe_only(marks)
        self._observe_only = observe_only     # cached for the WS fastlane
        if observe_only:
            alert("Polymarket bot: KILL-SWITCH — drawdown above limit, observe-only mode.")

        try:
            # ONE estimator pass per cycle (one LLM budget): both the longshot
            # path and the fade consume the same first-level estimates.
            candidates = self.scanner.first_level_filter(markets)
            estimates = self.estimator.estimate_all(candidates, markets)
            top = estimates[:20]
            qualifying = []
            for est in estimates:
                ok = self.estimator.qualifies(est)
                self.ledger.record_estimate(est, ok)
                if ok:
                    qualifying.append(est)
            log.info("estimates: %d, passing edge threshold: %d", len(estimates), len(qualifying))
            if qualifying:
                # YES-side book depth matters only to the longshot buys; check it
                # just for the few qualifying candidates (cheap).
                ok_tokens = {c.token_id for c in self.scanner.verify_depth(
                    [e.candidate for e in qualifying])}
                qualifying = [e for e in qualifying
                              if e.candidate.token_id in ok_tokens]

            if not observe_only and self.killswitch.trading_allowed:
                if self.breaker.allows("longshot"):
                    self._enter_positions(qualifying)
                if self.breaker.allows("fade"):
                    # Fade buys NO — YES-side depth is irrelevant to it; the
                    # executor checks the NO book when placing the limit order.
                    self.fade.cycle(estimates)
            if self.killswitch.trading_allowed:
                # Observe-only is a brake on NEW risk; REDUCING risk stays
                # allowed — in a drawdown, take-profits must not be frozen.
                self._exit_positions(marks)
            # Guardian WARNINGS always run — a materializing tail matters most
            # during a halt; only the auto-trim is gated (inside the pass).
            self._guardian_pass(marks)
            # Refresh the fastlane cache after this cycle's entries/exits.
            self._positions_by_token = {
                p.token_id: p for p in self.ledger.open_positions(self.mode)}
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
            plan = self.portfolio.size_trade(est, scale=self._longshot_scale)
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
            if mark:
                self._exit_one(position, mark)

    def _guardian_pass(self, marks: dict[str, float]) -> None:
        """Warn on (and optionally trim) any expensive leg whose sold tail is
        materializing. The alert dedupes with the WS-fastlane one (same key).
        Warnings run in EVERY state; the auto-trim (an order) only when the
        kill-switch allows trading."""
        c = self.cfg.guardian
        for position in self.ledger.open_positions(self.mode):
            mark = marks.get(position.token_id)
            if not mark:
                continue
            verdict = self.guardian.check(position, mark)
            if verdict is None:
                continue
            alert(verdict.describe(position.question),
                  key=f"guardian:{position.token_id}",
                  cooldown_sec=self.cfg.alerts.opportunity_cooldown_sec,
                  value=verdict.drop,
                  min_change=self.cfg.alerts.opportunity_min_edge_change)
            if c.auto_reduce and self.killswitch.trading_allowed:
                self._reduce_one(position, mark, c.reduce_fraction)

    def _reduce_one(self, position, mark: float, fraction: float) -> bool:
        """Guardian stop: sell a fraction of a materializing leg into the book."""
        import math
        size = math.floor(position.size * fraction)
        if size <= 0:
            return False
        from .models import Candidate
        est = Estimate(
            candidate=Candidate(market=self._market_stub(position),
                                outcome_index=0, token_id=position.token_id, p_mkt=mark),
            p_mkt=mark, p_est=mark, signals=[])
        min_price = round(max(mark * 0.9, 0.01), 4)   # accept slippage to actually exit
        result = self.executor.execute_sell(
            est_to_plan(est, position.category), float(size), min_price)
        if result.status == "filled":
            msg = (f"GUARDIAN reduce [{self.mode}] sold {size:,.0f} at "
                   f"{result.avg_price:.4f} (entry {position.avg_price:.4f}) — "
                   f"capping a materializing tail — {position.question[:50]}")
            log.info(msg)
            alert(msg)
            return True
        return False

    def _exit_one(self, position, mark: float) -> bool:
        """Take-profit a single position at `mark` (used by the cycle and WS fastlane)."""
        exit_plan = self.portfolio.exit_plan(position, mark)
        if exit_plan is None:
            return False
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
            return True
        return False

    def on_tick(self, token: str, top) -> None:
        """WS fastlane: instant take-profit exits and MM reprice on a tick.

        Runs in the WS thread — must never hit the DB or the network directly:
        positions and observe-only come from caches refreshed by the cycle.
        """
        try:
            # Analytics first: record the tick even when trading is frozen —
            # the learning loop wants ALL history (enqueue only, never blocks).
            if self.ticks is not None:
                self.ticks.record_tick(token, top.bid, top.ask,
                                       top.bid_size, top.ask_size,
                                       top.ts if top.ts > 0 else None)
            # Guardian WARNING runs before any trading gate: a materializing
            # tail matters MOST during a halt (disasters correlate), and the
            # alert is read-only (cached mark, non-blocking queue).
            position = self._positions_by_token.get(token)
            if position is not None and top.bid > 0:
                verdict = self.guardian.check(position, top.bid)
                if verdict is not None:
                    alert(verdict.describe(position.question),
                          key=f"guardian:{token}",
                          cooldown_sec=self.cfg.alerts.opportunity_cooldown_sec,
                          value=verdict.drop,
                          min_change=self.cfg.alerts.opportunity_min_edge_change)
            if not self.killswitch.trading_allowed:
                return
            # Exits (risk REDUCTION) run even in observe-only — only new risk
            # (MM quoting) is frozen by the drawdown brake below.
            if position is not None and top.bid > 0:
                if self._exit_one(position, top.bid):
                    self._positions_by_token.pop(token, None)
            # Arbitrage fastlane: flag the structure this token belongs to
            # (set-add only — the re-check runs in the worker thread, and its
            # execution is gated by _may_execute there).
            hit = self._basket_token_map.get(token) or self._chain_token_map.get(token)
            if hit is not None:
                self.arb_fastlane.flag(hit)
            if self._observe_only:
                return
            if self.breaker.allows("mm"):
                self.mm.react_to_tick(token, top)
            if self.cfg.sprint_mm.enabled and self.breaker.allows("sprint_mm"):
                self.sprint.react_to_tick(token, top)
        except Exception:
            log.exception("on_tick")

    def _market_stub(self, position):
        from .models import Market
        return Market(
            id=position.market_id, question=position.question,
            outcomes=[position.outcome or "Yes", "No"],
            outcome_prices=[0.0, 0.0],
            clob_token_ids=[position.token_id, ""],
        )

    def _record_category_indexes(self, markets) -> None:
        """One row per category per cycle: volume-weighted mean daily price
        change — the factor series the correlation learner fits on."""
        import time as _time
        acc: dict[str, list[float]] = {}
        for m in markets:
            if m.closed or not m.outcome_prices:
                continue
            w = max(m.volume_24h_usd, 0.0)
            if w <= 0:
                continue
            cat = classify_category(m.question, m.category)
            slot = acc.setdefault(cat, [0.0, 0.0, 0.0])
            slot[0] += w * m.one_day_price_change
            slot[1] += w
            slot[2] += 1
        ts = _time.time()
        for cat, (sw, w, n) in acc.items():
            if w > 0 and n >= 3:      # a 1-2 market "category" is not a factor
                self.ticks.record_category_index(cat, sw / w, int(n), ts)

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

    def _may_execute(self, strategy: str) -> bool:
        """Common execution gate for detect+alert strategies: the kill-switch,
        the observe-only drawdown brake and the per-strategy circuit breaker
        block ORDERS, while detection/alerts keep running for the human."""
        return (self.killswitch.trading_allowed
                and not self.portfolio.observe_only()
                and self.breaker.allows(strategy))

    def arb_job(self) -> None:
        """#1: neg-risk baskets. Prices are checked against live order books.

        Also registers the prefiltered groups with the WS fastlane, so a tick
        on any of their tokens triggers an instant re-check between polls.
        """
        if not self.markets_cache or not self.cfg.arbitrage.enabled:
            return
        try:
            may = self._may_execute("arb")
            groups = self.arb.prefilter_events(self.markets_cache)
            found = []
            for group in groups:
                a = self.arb.check_group(group, allow_execute=may)
                if a is not None:
                    found.append(a)
                    self._record_arb_opportunity(a, may)
            # Fastlane registration: token -> event group (atomic swaps).
            self._arb_groups = {g[0].event_id: g for g in groups if g[0].event_id}
            self._basket_token_map = {
                tok: ("basket", g[0].event_id)
                for g in groups if g[0].event_id
                for m in g for tok in m.clob_token_ids[:2]}
            self._sync_ws_watch(list(self._positions_by_token.values()))
            for a in found[:3]:
                warn = " ⚠️ suspect: verify basket completeness" if a.suspect else ""
                will_execute = self.cfg.arbitrage.execute and may and not a.suspect
                note = "" if will_execute else " (execute off)"
                alert(f"ARBITRAGE {a.side} basket \"{a.event_title[:60]}\": "
                      f"NET after fees +{a.net_profit_pct * 100:.2f}% "
                      f"(gross +{a.profit_pct * 100:.1f}%), "
                      f"depth {a.max_sets_by_depth()} sets{warn}{note}",
                      key=f"arb:{a.event_id}:{a.side}",
                      cooldown_sec=self.cfg.alerts.opportunity_cooldown_sec,
                      value=a.net_profit_pct,
                      min_change=self.cfg.alerts.opportunity_min_edge_change)
        except Exception:
            log.exception("arb job")

    def chain_arb_job(self) -> None:
        """Chain (ladder) arbitrage: monotonic constraints across nested markets.

        Also registers the classified pairs with the WS fastlane, so a tick on
        either leg triggers an instant re-check between polls.
        """
        if not self.markets_cache or not self.cfg.chain_arb.enabled:
            return
        try:
            may = self._may_execute("chain_arb")
            pairs = self.chain_arb.prefilter_pairs(self.markets_cache)
            found = []
            for subset, superset, kind in pairs:
                p = self.chain_arb.check_pair(subset, superset, kind,
                                              allow_execute=may)
                if p is not None:
                    found.append(p)
                    self._record_chain_opportunity(p, may)
            # Fastlane registration: the two tradable legs -> the pair.
            self._chain_pairs = {(s.id, sp.id): (s, sp, k) for s, sp, k in pairs}
            self._chain_token_map = {}
            for s, sp, _k in pairs:
                self._chain_token_map[sp.clob_token_ids[0]] = ("chain", (s.id, sp.id))
                self._chain_token_map[s.clob_token_ids[1]] = ("chain", (s.id, sp.id))
            self._sync_ws_watch(list(self._positions_by_token.values()))
            for p in found[:3]:
                will_execute = self.cfg.chain_arb.execute and may
                note = "" if will_execute else " (execute off)"
                net = p.net_profit_pct - self.cfg.chain_arb.classification_haircut
                alert(f"CHAIN ARB [{p.kind}] \"{p.event_title[:50]}\": "
                      f"NET after fee+haircut +{net * 100:.2f}% "
                      f"(gross +{p.profit_pct * 100:.1f}%), "
                      f"depth {p.max_sets_by_depth()} sets{note}",
                      key=f"chain:{p.event_id}:{p.subset.market.id}:{p.superset.market.id}",
                      cooldown_sec=self.cfg.alerts.opportunity_cooldown_sec,
                      value=net,
                      min_change=self.cfg.alerts.opportunity_min_edge_change)
        except Exception:
            log.exception("chain arb job")

    def _record_arb_opportunity(self, a, may: bool) -> None:
        """One basket window into the opportunity ledger (executed = an order
        was actually attempted for it, i.e. execute on + gates open)."""
        self.ledger.record_opportunity(
            self.mode, "arb", f"{a.event_id}:{a.side}", a.event_title[:80],
            a.net_profit_pct, a.max_sets_by_depth() * a.cost_per_set,
            executed=self.cfg.arbitrage.execute and may and not a.suspect)

    def _record_chain_opportunity(self, p, may: bool) -> None:
        net = p.net_profit_pct - self.cfg.chain_arb.classification_haircut
        self.ledger.record_opportunity(
            self.mode, "chain_arb",
            f"{p.event_id}:{p.subset.market.id}:{p.superset.market.id}",
            p.event_title[:80], net, p.max_sets_by_depth() * p.cost_per_set,
            executed=self.cfg.chain_arb.execute and may)

    def _fastlane_check(self, key: tuple) -> None:
        """Worker-thread callback: re-verify one flagged structure NOW.

        Windows found here (the short-lived ones that die between polls) are
        recorded too — otherwise measured capacity would understate exactly
        the fastest opportunities.
        """
        kind, ident = key
        if kind == "basket":
            group = self._arb_groups.get(ident)
            if group and self.cfg.arbitrage.enabled:
                may = self._may_execute("arb")
                a = self.arb.check_group(group, allow_execute=may)
                if a is not None:
                    self._record_arb_opportunity(a, may)
        elif kind == "chain":
            pair = self._chain_pairs.get(ident)
            if pair and self.cfg.chain_arb.enabled:
                subset, superset, k = pair
                may = self._may_execute("chain_arb")
                p = self.chain_arb.check_pair(subset, superset, k,
                                              allow_execute=may)
                if p is not None:
                    self._record_chain_opportunity(p, may)

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
            # MM fills may have changed positions — refresh the fastlane cache.
            positions = self.ledger.open_positions(self.mode)
            self._positions_by_token = {p.token_id: p for p in positions}
            self._mm_watch = {tok for q in quotes
                              for tok in q.market.clob_token_ids[:2]}
            self._sync_ws_watch(positions)
        except Exception:
            log.exception("mm job")

    def _sync_ws_watch(self, positions) -> None:
        """Subscribe the WS to the UNION of every consumer's tokens: both
        quoting jobs, open positions, and the arbitrage fastlane structures."""
        if self.ws is None:
            return
        tokens = set(self._mm_watch) | set(self._sprint_watch) \
            | set(self._basket_token_map) | set(self._chain_token_map)
        tokens.update(p.token_id for p in positions)
        if tokens:
            self.ws.watch(tokens)

    def sprint_job(self) -> None:
        """Short-dated MM: fast capital turnover. Same gates as the core MM."""
        if not self.markets_cache:
            return
        try:
            if self.ws is not None and self.ws.healthy:
                self.killswitch.on_ws_recovered()
            if not self.killswitch.trading_allowed or self.portfolio.observe_only() \
                    or not self.breaker.allows("sprint_mm"):
                self.sprint.shutdown()
                return
            quotes = self.sprint.cycle(self.markets_cache)
            positions = self.ledger.open_positions(self.mode)
            self._positions_by_token = {p.token_id: p for p in positions}
            self._sprint_watch = {tok for q in quotes
                                  for tok in q.market.clob_token_ids[:2]}
            self._sync_ws_watch(positions)
        except Exception:
            log.exception("sprint job")

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
            if not self.cfg.smart_money.enabled or not self.cfg.smart_money.watch_wallets:
                return
            snapshot = self.smart_money.snapshot()   # one API pass, shared below
            self.smart_money.cycle(snapshot)
            if self.cfg.smart_money.as_signal:
                self.smart_signal.set_hot(self.smart_money.build_hot_tokens(snapshot))
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
                self.killswitch.reconcile(
                    self.mm.local_order_ids() | self.sprint.local_order_ids(),
                    exchange_ids)
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
        if not self.markets_cache:
            return
        try:
            self.resolution.cycle(self.markets_cache,
                                  allow_execute=self._may_execute("resolution"))
        except Exception:
            log.exception("resolution job")

    def redemption_job(self) -> None:
        """Claim resolved winnings back into USDC (frees capital to redeploy).

        Runs even when halted for the ALERT (claiming is the human's decision
        then); on-chain execution is gated like every other order path.
        """
        try:
            self.redeemer.cycle(allow_execute=self.killswitch.trading_allowed)
        except Exception:
            log.exception("redemption job")

    def calibration_job(self) -> None:
        """Refit every learner from recorded data (bias, markout, Platt,
        correlations, fill calibration)."""
        try:
            self.fade.refresh_calibration()
            self.mm.refresh_feedback()
            if self.cfg.estimator.platt_enabled:
                pairs = [(r["p_est"], 1.0 if r["won"] else 0.0)
                         for strategy in ("longshot", "fade")
                         for r in self.ledger.resolved_for_calibration(self.mode, strategy)
                         if r.get("p_est")]
                self.platt.fit(pairs)
            if self.ticks is not None:
                self.corr_learner.fit(self.ticks.category_series())
                if self.corr_learner.learned:
                    set_learned_correlation(self.corr_learner.learned)
                    log.info("calibration: %d learned category correlations active",
                             len(self.corr_learner.learned))
            self.fill_calibrator.fit(self.ledger.quote_outcomes(self.mode))
        except Exception:
            log.exception("calibration job")

    def digest_job(self) -> None:
        """Telegram digest: PnL, inventory, attribution, Sharpe allocation hint."""
        try:
            equity = self.portfolio.equity()
            positions = self.ledger.open_positions(self.mode)
            pnl = self.ledger.realized_pnl_by_strategy(self.mode)
            pnl_lines = "\n".join(f"  {k}: {v:+,.2f}" for k, v in pnl.items()) or "  —"
            # Sharpe allocation: weights move the sizing multipliers inside the
            # allocator's clamped corridor (hard caps still apply after them).
            for k, v in pnl.items():
                self._pnl_history.setdefault(k, []).append(v)
            weights_line = ""
            series = {k: v for k, v in self._pnl_history.items() if len(v) >= 3}
            if series:
                from .research import sharpe_allocation
                w = sharpe_allocation(series)
                weights_line = ("\nCapital weights (Sharpe): "
                                + ", ".join(f"{k} {v:.0%}" for k, v in sorted(w.items())))
                changes = self.allocator.update(w)
                self.mm.size_factor = self.allocator.factor("mm")
                self.sprint.size_factor = self.allocator.factor("sprint_mm")
                self.fade.size_scale = self.allocator.factor("fade")
                self._longshot_scale = self.allocator.factor("longshot")
                if self.allocator.summary():
                    weights_line += ("\nApplied sizing multipliers "
                                     f"[{self.cfg.allocator.floor}..{self.cfg.allocator.ceil}]: "
                                     + self.allocator.summary())
                if changes:
                    log.info("digest: allocator moved %s", changes)
            alert(f"Digest [{self.mode}]\n"
                  f"Equity: ${equity:,.2f} | drawdown {self.portfolio.drawdown() * 100:.1f}%\n"
                  f"Open positions: {len(positions)} "
                  f"(${sum(p.cost_usd for p in positions):,.2f})\n"
                  f"Realized PnL by strategy:\n{pnl_lines}"
                  f"{weights_line}\n"
                  f"Kill-switch: {'HALT: ' + self.killswitch.reason if self.killswitch.halted else 'normal'}")
        except Exception:
            log.exception("digest job")

    def cross_job(self) -> None:
        """#2: divergences with external venues — alerts only."""
        if not self.markets_cache:
            return
        try:
            for d in self.cross.cycle(self.markets_cache):
                # A persistent divergence re-notifies once per cooldown, or
                # when the gap moves materially — not every 15-minute poll.
                alert(d.describe(),
                      key=f"cross:{d.poly_market.id}:{d.venue_market.url}",
                      cooldown_sec=self.cfg.alerts.opportunity_cooldown_sec,
                      value=d.gap,
                      min_change=self.cfg.alerts.opportunity_min_edge_change)
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
                                 "autotune", "sync-config", "capacity",
                                 "leadlag", "coherence"),
                        default="dry-run",
                        help="dry-run -> paper -> live (manual promotion only); "
                             "backtest/record-books/replay — offline phases; "
                             "sync-config appends example sections missing "
                             "from your config.yaml")
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

    if args.mode == "sync-config":
        from .config import DEFAULT_CONFIG_PATH, sync_config
        target = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
        added = sync_config(target)
        if added:
            print(f"Added {len(added)} section(s) to {target}: {', '.join(added)}")
            print("Your existing sections and values were not touched.")
        else:
            print(f"{target} already has every section from config.example.yaml.")
        return

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
            print_report(compute_report(ledger, args.report_mode,
                                        fade_prior=cfg.fade.bias_discount))
        finally:
            ledger.close()
        return

    if args.mode == "capacity":
        from .capacity import run_capacity
        run_capacity(cfg, args.report_mode)
        return

    if args.mode == "leadlag":
        from .leadlag import run_leadlag
        run_leadlag(cfg)
        return

    if args.mode == "coherence":
        from .coherence import run_coherence
        run_coherence(cfg)
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
        bot.arb_fastlane.start()   # WS-triggered arb re-checks (worker thread)
    if not args.once and cfg.telegram.control_enabled and bot.tg_control.enabled:
        bot.tg_control.start()     # two-way Telegram command listener
    if args.once:
        try:
            bot.cycle()
            bot.arb_job()
            bot.chain_arb_job()
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

    if hasattr(signal, "SIGHUP"):
        def reload_cfg(signum, frame):  # noqa: ARG001
            changed = reload_config_inplace(bot.cfg, args.config)
            log.info("SIGHUP: config reloaded, changed: %s", changed or "nothing")
        signal.signal(signal.SIGHUP, reload_cfg)

    metrics = None
    if cfg.ops.metrics_enabled:
        metrics = MetricsServer(bot.metrics_snapshot, cfg.ops.metrics_port,
                                bind=cfg.ops.metrics_bind)
        metrics.start()

    scheduler = BackgroundScheduler()
    scheduler.add_job(bot.cycle, "interval",
                      minutes=cfg.scanner.interval_minutes,
                      max_instances=1, coalesce=True)
    if cfg.arbitrage.enabled:
        scheduler.add_job(bot.arb_job, "interval",
                          seconds=cfg.arbitrage.interval_sec,
                          max_instances=1, coalesce=True)
    if cfg.chain_arb.enabled:
        scheduler.add_job(bot.chain_arb_job, "interval",
                          seconds=cfg.chain_arb.interval_sec,
                          max_instances=1, coalesce=True)
    if cfg.market_maker.enabled:
        scheduler.add_job(bot.mm_job, "interval",
                          seconds=cfg.market_maker.interval_sec,
                          max_instances=1, coalesce=True)
    if cfg.sprint_mm.enabled:
        scheduler.add_job(bot.sprint_job, "interval",
                          seconds=cfg.sprint_mm.interval_sec,
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
    if cfg.redemption.enabled and args.mode == "live":
        scheduler.add_job(bot.redemption_job, "interval",
                          seconds=cfg.redemption.interval_sec,
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
    if metrics is not None:
        metrics.stop()
    bot.close()
    log.info("stopped cleanly.")


if __name__ == "__main__":
    main()
