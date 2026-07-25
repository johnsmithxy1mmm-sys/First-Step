"""Execution: maker limit orders only, repricing, child-splitting, idempotency.

On tails the spread is huge — a market order into a thin book instantly
destroys the edge. So: bid just above best bid, wait for a fill with a
timeout, reprice a limited number of times, cancel the order if price moves
above the edge threshold (limit_price_cap from the portfolio).
"""

from __future__ import annotations

import logging
import math
import time

from .clob import ClobReader, Trader, round_to_tick
from .config import BotConfig
from .ledger import Ledger
from .models import ExecutionResult, TradePlan

log = logging.getLogger(__name__)

OPEN_STATUSES = {"live", "open", "delayed"}


class Executor:
    def __init__(self, cfg: BotConfig, ledger: Ledger, clob: ClobReader,
                 trader: Trader | None, mode: str):
        self._cfg = cfg.executor
        self._ledger = ledger
        self._clob = clob
        self._trader = trader          # None = dry-run: virtual orders
        self._mode = mode

    # --- idempotency ---

    def already_entered(self, token_id: str) -> bool:
        """Before any order, reconcile local state and the API's actual positions."""
        if self._ledger.has_position_or_open_buy(token_id, self._mode):
            return True
        if self._trader is not None:
            try:
                for p in self._trader.api_positions():
                    if str(p.get("asset")) == token_id and float(p.get("size") or 0) > 0:
                        return True
                for o in self._trader.open_orders():
                    if str(o.get("asset_id")) == token_id:
                        return True
            except Exception as exc:
                log.warning("reconcile: %s (assume a position exists — safer)", exc)
                return True
        return False

    # --- entry ---

    def execute(self, plan: TradePlan, strategy: str = "longshot") -> ExecutionResult:
        if self.already_entered(plan.token_id):
            return ExecutionResult(status="skipped", detail="already entered (idempotency)")

        remaining_usd = plan.size_usd
        filled_size = 0.0
        filled_usd = 0.0
        order_ids: list[str] = []

        # Split a large order into children <= max_child_order_usd.
        n_children = max(1, math.ceil(plan.size_usd / self._cfg.max_child_order_usd))
        child_usd = plan.size_usd / n_children

        for _ in range(n_children):
            usd = min(child_usd, remaining_usd)
            if usd <= 0:
                break
            result = self._place_maker_child(plan, usd)
            if result is None:
                break
            size, price, order_id = result
            filled_size += size
            filled_usd += size * price
            remaining_usd -= size * price
            if order_id:
                order_ids.append(order_id)

        if filled_size <= 0:
            return ExecutionResult(status="canceled", detail="no fill within constraints")

        avg_price = filled_usd / filled_size
        self._ledger.record_trade(
            mode=self._mode, estimate=plan.estimate, category=plan.category,
            side="BUY", price=avg_price, size=filled_size,
            order_id=order_ids[0] if order_ids else None,
            status="filled" if self._trader else "sim-filled",
            strategy=strategy,
        )
        return ExecutionResult(status="filled", filled_size=filled_size,
                               avg_price=avg_price, order_ids=order_ids)

    def _maker_bid(self, plan: TradePlan) -> float | None:
        """Maker bid price: just above best bid, never crossing ask or the cap."""
        c = plan.estimate.candidate
        book = self._clob.order_book(c.token_id)
        if book is None:
            return None
        tick = c.market.tick_size
        best_bid, best_ask = book.best_bid, book.best_ask
        if best_ask > 0 and best_ask <= plan.limit_price_cap:
            # Ask already within the allowed price: sit one tick below ask (max
            # queue priority, still a maker).
            price = best_ask - tick
        else:
            price = best_bid + tick if best_bid > 0 else tick
        price = min(price, plan.limit_price_cap)
        if best_ask > 0:
            price = min(price, best_ask - tick)  # do not cross the book
        price = round_to_tick(price, tick)
        if price < tick or price <= 0:
            return None
        return price

    def _place_maker_child(self, plan: TradePlan, usd: float) -> tuple[float, float, str | None] | None:
        """One child order: place bid, wait, reprice. -> (size, price, order_id)."""
        c = plan.estimate.candidate

        for attempt in range(self._cfg.max_reprices + 1):
            price = self._maker_bid(plan)
            if price is None:
                log.info("skip child %s: no valid maker price (edge cap %.4f)",
                         c.token_id[:16], plan.limit_price_cap)
                return None
            size = float(math.floor(usd / price))
            if size < c.market.min_order_size:
                return None

            if self._trader is None:
                # Dry-run: optimistic model — the maker bid is treated as filled
                # at our price. Real fill-rate is lower; see README.
                return size, price, None

            try:
                resp = self._trader.buy_limit(c.token_id, price, size,
                                              neg_risk=c.market.neg_risk)
            except Exception as exc:
                log.error("order failed %s: %s", c.token_id[:16], exc)
                return None
            order_id = (resp or {}).get("orderID")
            if not order_id:
                return None

            matched = self._wait_fill(order_id)
            if matched >= size * 0.99:
                return matched, price, order_id
            # No fill within the timeout: cancel and decide — reprice or give up.
            try:
                self._trader.cancel(order_id)
            except Exception:
                pass
            # Late-fill window: the order can match between our last poll and the
            # cancel landing on the exchange. Re-read once after the cancel —
            # shares owned but never recorded would silently desync the ledger
            # (reconcile catches ghost ORDERS, not ghost FILLS).
            try:
                matched = max(matched, float(
                    self._trader.order_status(order_id).get("size_matched", 0.0)))
            except Exception:
                pass
            if matched > 0:
                return matched, price, order_id  # record the (partial) fill
            log.info("reprice %d/%d for %s", attempt + 1, self._cfg.max_reprices,
                     c.token_id[:16])
        return None

    def _wait_fill(self, order_id: str) -> float:
        deadline = time.monotonic() + self._cfg.fill_timeout_sec
        matched = 0.0
        while time.monotonic() < deadline:
            try:
                status = self._trader.order_status(order_id)  # type: ignore[union-attr]
            except Exception:
                break
            matched = status.get("size_matched", 0.0)
            if status.get("status") not in OPEN_STATUSES:
                break
            time.sleep(self._cfg.poll_interval_sec)
        return matched

    # --- exit (take-profit) ---

    def execute_sell(self, plan: TradePlan, size: float, min_price: float,
                     known_bid: float | None = None) -> ExecutionResult:
        """Sell part of a position at best bid (not below min_price).

        `known_bid`: a bid the CALLER already holds (e.g. the WS tick that
        triggered this exit). When given, the REST book fetch is skipped — the
        WS fastlane runs on the recv thread, and get_with_backoff's retries
        there could stall the stream past ws_staleness_kill_sec. The tick's bid
        is also FRESHER than a round-trip re-fetch.
        """
        c = plan.estimate.candidate
        if known_bid is not None:
            best_bid = known_bid
        else:
            book = self._clob.order_book(c.token_id)
            best_bid = book.best_bid if book is not None else 0.0
        if best_bid <= 0 or best_bid < min_price:
            return ExecutionResult(status="skipped", detail="bid too thin for take-profit")
        price = round_to_tick(best_bid, c.market.tick_size)

        order_id = None
        if self._trader is not None:
            try:
                resp = self._trader.sell_limit(c.token_id, price, size,
                                               neg_risk=c.market.neg_risk)
                order_id = (resp or {}).get("orderID")
            except Exception as exc:
                return ExecutionResult(status="failed", detail=str(exc))

        self._ledger.record_trade(
            mode=self._mode, estimate=plan.estimate, category=plan.category,
            side="SELL", price=price, size=size, order_id=order_id,
            status="filled" if self._trader else "sim-filled",
        )
        return ExecutionResult(status="filled", filled_size=size, avg_price=price,
                               order_ids=[order_id] if order_id else [])
