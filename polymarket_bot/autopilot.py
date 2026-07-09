"""Автопилот: полный торговый цикл без участия человека.

Один цикл:
  1. Снять зависшие ордера (старше max_order_age_hours) — разморозить бюджет.
  2. Просканировать рынки, отскорить кандидатов, разместить ставки
     в пределах общего и дневного бюджета.
  3. Сопроводить позиции: тейк-профит («продать долю на N иксах»),
     найти выигрыши, готовые к выводу.
  4. Просканировать арбитражи neg-risk событий (сумма исходов < $1).
  5. Отправить сводку в Telegram (если настроен).

Каждый шаг изолирован: ошибка одного не останавливает остальные и не
убивает цикл — бот продолжает работать.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field, replace

import requests

from . import arb as arb_mod
from . import clob
from .config import BotConfig
from .gamma import iter_active_events, iter_active_markets, market_by_token, market_float
from .notify import notify
from .storage import BetLog
from .strategy import find_candidates, plan_bets, stake_for

# После стольких ошибок ордеров подряд цикл прерывается: что-то системно не так
# (кончился баланс, отозван allowance, API лежит) — не долбить биржу дальше.
MAX_CONSECUTIVE_ORDER_ERRORS = 5


@dataclass
class CycleReport:
    placed: int = 0
    skipped: int = 0
    spent: float = 0.0
    shares: float = 0.0
    canceled: int = 0
    take_profits: int = 0
    redeemable: list[str] = field(default_factory=list)
    arbs_found: int = 0
    arb_placed: int = 0
    errors: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    def log(self, text: str) -> None:
        print(text)
        self.lines.append(text)


def cancel_stale_orders(cfg: BotConfig, log: BetLog, trader, report: CycleReport) -> None:
    """Снимает неисполненные ордера старше лимита: maker-лимитки не должны висеть вечно."""
    for bet in log.open_orders(older_than_hours=cfg.max_order_age_hours):
        try:
            trader.cancel(bet["order_id"])
        except Exception as exc:
            # Скорее всего ордер уже исполнился или снят — фиксируем и не трогаем.
            log.update_status(bet["order_id"], "unknown-cancel-failed")
            report.errors.append(f"cancel {bet['order_id']}: {exc}")
            continue
        log.update_status(bet["order_id"], "canceled")
        report.canceled += 1
        report.log(f"  СНЯТ зависший ордер: {bet['price']:.3f} x {bet['size']:,.0f} "
                   f"[{bet['outcome']}] {bet['question'][:60]}")


def remaining_budget(cfg: BotConfig, log: BetLog) -> float:
    """Сколько ещё можно потратить с учётом общего и дневного лимитов."""
    remaining = cfg.total_budget_usd - log.spent_usd()
    if cfg.daily_budget_usd > 0:
        remaining = min(remaining, cfg.daily_budget_usd - log.spent_today_usd())
    return remaining


def place_bets(cfg: BotConfig, log: BetLog, trader, session: requests.Session,
               report: CycleReport) -> None:
    """Скан → скоринг → ставки. trader=None означает dry-run."""
    budget = remaining_budget(cfg, log)
    if budget < cfg.stake_usd * 0.5:
        report.log(f"Бюджет на ставки исчерпан (осталось ${budget:,.2f}).")
        return

    report.log("Сканирую активные рынки Polymarket...")
    markets = list(iter_active_markets(cfg, session))
    report.log(f"Рынков получено: {len(markets)}")

    candidates = find_candidates(markets, cfg, skip_token_ids=log.live_token_ids())
    planned = plan_bets(candidates, replace(cfg, total_budget_usd=budget))
    report.log(f"Кандидатов: {len(candidates)}, в плане после бюджета: {len(planned)}")

    errors_in_row = 0
    for c in planned:
        stake = stake_for(c, cfg)
        if report.spent + stake > budget:
            break

        price = c.price
        if cfg.verify_orderbook or cfg.entry_mode == "maker":
            quote = clob.get_quote(cfg, c.token_id, session)
            time.sleep(cfg.request_delay_sec)
            if quote is None:
                report.skipped += 1
                continue
            price = clob.compute_entry_price(quote, c, cfg)
            if price is None:
                report.skipped += 1
                continue
        else:
            price = clob.round_to_tick(price, c.tick_size)

        size = clob.shares_for_stake(stake, price, c.min_order_size)
        if size <= 0:
            report.skipped += 1
            continue

        order_id, status = None, "dry-run"
        if trader is not None:
            try:
                resp = trader.buy_limit(c.token_id, price, size, neg_risk=c.neg_risk)
                errors_in_row = 0
            except Exception as exc:
                errors_in_row += 1
                report.skipped += 1
                report.errors.append(f"order [{c.outcome}] {c.question[:50]}: {exc}")
                report.log(f"  ОШИБКА ордера: {c.question[:60]}: {exc}")
                if errors_in_row >= MAX_CONSECUTIVE_ORDER_ERRORS:
                    report.log("Слишком много ошибок подряд — прерываю размещение "
                               "(проверьте баланс USDC и allowances).")
                    break
                continue
            order_id = (resp or {}).get("orderID")
            status = (resp or {}).get("status", "unknown")

        log.record(
            market_id=c.market_id, question=c.question, slug=c.slug,
            outcome=c.outcome, token_id=c.token_id, price=price, size=size,
            live=trader is not None, side="BUY", order_id=order_id, status=status,
        )
        report.placed += 1
        report.spent += price * size
        report.shares += size
        mult = 1 / price if price else 0
        report.log(f"  {'ОРДЕР' if trader else 'ПЛАН '} score={c.score:.2f} {price:.3f} x "
                   f"{size:,.0f} шт = ${price * size:,.2f} (до {mult:,.0f}x) "
                   f"[{c.outcome}] {c.question[:70]}")


def fetch_positions(cfg: BotConfig, address: str, session: requests.Session) -> list[dict]:
    resp = session.get(f"{cfg.data_api_host}/positions",
                       params={"user": address, "limit": 500}, timeout=30)
    resp.raise_for_status()
    return resp.json() or []


def manage_positions(cfg: BotConfig, log: BetLog, trader, session: requests.Session,
                     report: CycleReport) -> None:
    """Тейк-профит и поиск выигрышей. Без адреса кошелька шаг пропускается."""
    address = os.environ.get("POLYMARKET_FUNDER") or os.environ.get("POLYMARKET_ADDRESS")
    if not address:
        return
    positions = fetch_positions(cfg, address, session)

    for p in positions:
        title = p.get("title") or "?"
        if p.get("redeemable"):
            value = float(p.get("currentValue") or 0)
            report.redeemable.append(f"{title[:60]} (~${value:,.2f})")
            continue

        avg = float(p.get("avgPrice") or 0)
        cur = float(p.get("curPrice") or 0)
        size_total = float(p.get("size") or 0)
        token_id = str(p.get("asset") or "")
        if avg <= 0 or size_total <= 0 or not token_id:
            continue
        if cur < avg * cfg.take_profit_multiple:
            continue
        if log.has_open_sell(token_id):
            continue

        report.log(f"  ТЕЙК-ПРОФИТ найден: {title[:60]} вход {avg:.3f} → сейчас {cur:.3f} "
                   f"({cur / avg:.0f}x)")
        market = market_by_token(cfg, token_id, session)
        tick = market_float(market or {}, "orderPriceMinTickSize") or 0.001
        neg_risk = bool((market or {}).get("negRisk", False))
        min_size = market_float(market or {}, "orderMinSize") or 5.0

        quote = clob.get_quote(cfg, token_id, session)
        if quote is None or quote.best_bid <= 0:
            continue
        # Не сливать в пустой бид: цена продажи не ниже 80% от целевого уровня.
        if quote.best_bid < avg * cfg.take_profit_multiple * 0.8:
            continue
        sell_size = float(math.floor(size_total * cfg.take_profit_fraction))
        if sell_size < min_size:
            sell_size = size_total if size_total >= min_size else 0
        if sell_size <= 0:
            continue
        price = clob.round_to_tick(quote.best_bid, tick)

        if trader is None:
            report.log(f"  (dry-run) продал бы {sell_size:,.0f} шт по {price:.3f} "
                       f"= ${price * sell_size:,.2f}")
            continue
        try:
            resp = trader.sell_limit(token_id, price, sell_size, neg_risk=neg_risk)
        except Exception as exc:
            report.errors.append(f"take-profit {title[:50]}: {exc}")
            continue
        log.record(
            market_id=str(p.get("conditionId") or ""), question=title, slug=p.get("slug") or "",
            outcome=str(p.get("outcome") or ""), token_id=token_id,
            price=price, size=sell_size, live=True, side="SELL",
            order_id=(resp or {}).get("orderID"),
            status=(resp or {}).get("status", "unknown"),
        )
        report.take_profits += 1
        report.log(f"  ПРОДАЖА {sell_size:,.0f} шт по {price:.3f} = ${price * sell_size:,.2f} "
                   f"(вход был {avg:.3f}; {100 * cfg.take_profit_fraction:.0f}% позиции, "
                   f"остаток едет бесплатно)")


def scan_arbs(cfg: BotConfig, log: BetLog, trader, session: requests.Session,
              report: CycleReport) -> list[arb_mod.ArbOpportunity]:
    """Ищет и (опционально) исполняет арбитражи neg-risk событий."""
    if not cfg.arb_scan:
        return []
    events = list(iter_active_events(cfg, session))
    arbs = arb_mod.find_arbs(events, cfg)
    report.arbs_found = len(arbs)
    for a in arbs[:10]:
        report.log(f"  АРБИТРАЖ: «{a.title[:60]}» комплект за ${a.sum_asks:.3f} "
                   f"→ гарантированные +{a.edge * 100:.1f}% ({len(a.legs)} ног)")
    if trader is None or not cfg.arb_execute:
        return arbs

    for a in arbs:
        fresh = arb_mod.verify_arb(a, cfg, session)
        if fresh is None:
            report.log(f"  арбитраж «{a.title[:50]}» не подтвердился по стаканам — пропуск")
            continue
        placed, spent = arb_mod.execute_arb(fresh, trader, log, cfg)
        if placed:
            report.arb_placed += placed
            report.spent += spent
            report.log(f"  АРБ ИСПОЛНЕН: {placed}/{len(fresh.legs)} ног, ${spent:,.2f}")
    return arbs


def summary_text(report: CycleReport, live: bool) -> str:
    mode = "LIVE" if live else "dry-run"
    lines = [f"Polymarket bot [{mode}]: ставок {report.placed}, "
             f"пропущено {report.skipped}, снято зависших {report.canceled}, "
             f"потрачено ${report.spent:,.2f}"]
    if report.placed:
        lines.append(f"Средний выигрыш одной победившей ставки: "
                     f"~${report.shares / report.placed:,.0f}")
    if report.take_profits:
        lines.append(f"Тейк-профитов размещено: {report.take_profits}")
    if report.arbs_found:
        lines.append(f"Арбитражей найдено: {report.arbs_found}"
                     + (f", исполнено ног: {report.arb_placed}" if report.arb_placed else ""))
    if report.redeemable:
        lines.append("ВЫИГРЫШИ готовы к выводу (кнопка Redeem на Polymarket):")
        lines.extend(f"  - {r}" for r in report.redeemable)
    if report.errors:
        lines.append(f"Ошибок за цикл: {len(report.errors)}")
    return "\n".join(lines)


def run_cycle(cfg: BotConfig, log: BetLog, trader) -> CycleReport:
    """Один полный цикл автопилота. Ошибки шагов не роняют цикл."""
    report = CycleReport()
    session = requests.Session()

    if trader is not None:
        try:
            cancel_stale_orders(cfg, log, trader, report)
        except Exception as exc:
            report.errors.append(f"cancel_stale: {exc}")

    try:
        place_bets(cfg, log, trader, session, report)
    except Exception as exc:
        report.errors.append(f"place_bets: {exc}")
        report.log(f"Ошибка размещения ставок: {exc}")

    try:
        manage_positions(cfg, log, trader, session, report)
    except Exception as exc:
        report.errors.append(f"manage_positions: {exc}")
        report.log(f"Ошибка сопровождения позиций: {exc}")

    try:
        scan_arbs(cfg, log, trader, session, report)
    except Exception as exc:
        report.errors.append(f"scan_arbs: {exc}")
        report.log(f"Ошибка арбитраж-сканера: {exc}")

    text = summary_text(report, live=trader is not None)
    report.log("\n" + text)
    # Уведомляем только когда есть что сказать — не спамим пустыми циклами.
    if report.placed or report.take_profits or report.redeemable or report.arbs_found:
        notify(text)
    return report
