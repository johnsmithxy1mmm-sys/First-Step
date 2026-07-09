"""CLI бота: python -m polymarket_bot <команда>.

Команды:
  scan       — найти и отскорить дешёвые исходы, показать таблицу кандидатов
  run        — один торговый цикл (по умолчанию dry-run; реальные ордера с --live)
  auto       — АВТОПИЛОТ: торговые циклы по расписанию, без участия человека
  arb        — сканер арбитражей neg-risk событий (сумма исходов < $1)
  history    — история ставок бота
  positions  — текущие позиции и PnL кошелька (data-api Polymarket)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from . import autopilot, clob
from .arb import find_arbs
from .config import BotConfig
from .gamma import iter_active_events, iter_active_markets
from .storage import BetLog
from .strategy import find_candidates

AUTO_LOG_PATH = Path(__file__).parent / "data" / "auto.log"


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="путь к JSON-конфигу (см. config.example.json)")
    parser.add_argument("--max-price", type=float, dest="max_price",
                        help="максимальная цена исхода, напр. 0.01")
    parser.add_argument("--stake", type=float, dest="stake_usd", help="долларов на одну ставку")
    parser.add_argument("--budget", type=float, dest="total_budget_usd", help="общий бюджет")
    parser.add_argument("--max-bets", type=int, dest="max_bets", help="лимит числа ставок")


def _load_config(args: argparse.Namespace) -> BotConfig:
    overrides = {k: getattr(args, k, None)
                 for k in ("max_price", "stake_usd", "total_budget_usd", "max_bets")}
    cfg = BotConfig.load(args.config, **overrides)
    cfg.validate()
    return cfg


def _make_trader(cfg: BotConfig, live: bool, yes: bool, what: str) -> clob.Trader | None:
    if not live:
        print("\n=== DRY-RUN: ордера НЕ размещаются. Для реальной торговли добавьте --live ===\n")
        return None
    if not yes:
        answer = input(f"РЕАЛЬНЫЕ деньги: {what}. Продолжить? [yes/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            sys.exit("Отменено.")
    return clob.Trader(cfg)


def cmd_scan(args: argparse.Namespace) -> None:
    cfg = _load_config(args)
    log = BetLog()
    print("Сканирую активные рынки Polymarket (это может занять минуту-две)...")
    markets = list(iter_active_markets(cfg))
    print(f"Рынков получено: {len(markets)}")
    candidates = find_candidates(markets, cfg, skip_token_ids=log.live_token_ids())
    print(f"Кандидатов после фильтров (цена ≤ {cfg.max_price:g}): {len(candidates)}\n")
    if not candidates:
        print("Ничего не найдено. Попробуйте ослабить фильтры (--max-price, min_liquidity_usd).")
        return
    print(f"{'SCORE':>5} {'ЦЕНА':>7} {'ИКСЫ':>6} {'ЛИКВ.$':>10} {'ДО КОНЦА':>9}  ИСХОД / ВОПРОС")
    for c in candidates[:50]:
        days = "-"
        if c.end_date is not None:
            days = f"{(c.end_date - datetime.now(timezone.utc)).days}д"
        print(f"{c.score:>5.2f} {c.price:>7.3f} {c.payout_multiple:>5.0f}x "
              f"{c.liquidity_usd:>10,.0f} {days:>9}  [{c.outcome}] {c.question[:75]}")
    if len(candidates) > 50:
        print(f"... и ещё {len(candidates) - 50}")


def cmd_run(args: argparse.Namespace) -> None:
    cfg = _load_config(args)
    log = BetLog()
    budget = autopilot.remaining_budget(cfg, log)
    trader = _make_trader(
        cfg, args.live, args.yes,
        f"до {cfg.max_bets} ордеров на сумму до ${min(budget, cfg.total_budget_usd):,.0f}",
    )
    autopilot.run_cycle(cfg, log, trader)


def cmd_auto(args: argparse.Namespace) -> None:
    cfg = _load_config(args)
    log = BetLog()
    trader = _make_trader(
        cfg, args.live, args.yes,
        f"автопилот каждые {cfg.auto_interval_min:g} мин, общий бюджет "
        f"${cfg.total_budget_usd:,.0f}"
        + (f", дневной ${cfg.daily_budget_usd:,.0f}" if cfg.daily_budget_usd else ""),
    )

    cycle = 0
    while True:
        cycle += 1
        started = datetime.now(timezone.utc)
        print(f"\n{'=' * 70}\nЦикл #{cycle} — {started.isoformat(timespec='seconds')}\n{'=' * 70}")
        try:
            report = autopilot.run_cycle(cfg, log, trader)
            _append_auto_log(started, report)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            # Автопилот не умирает: фиксируем и ждём следующего цикла.
            print(f"Цикл #{cycle} упал: {exc}")
        if args.once:
            break
        try:
            print(f"Сон {cfg.auto_interval_min:g} мин (Ctrl+C — остановить)...")
            time.sleep(cfg.auto_interval_min * 60)
        except KeyboardInterrupt:
            print("\nОстановлено пользователем.")
            break


def _append_auto_log(started: datetime, report: autopilot.CycleReport) -> None:
    AUTO_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUTO_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"\n--- {started.isoformat(timespec='seconds')} ---\n")
        fh.write("\n".join(report.lines) + "\n")


def cmd_arb(args: argparse.Namespace) -> None:
    cfg = _load_config(args)
    print("Сканирую neg-risk события (сумма всех исходов < $1 = безрисковая прибыль)...")
    events = list(iter_active_events(cfg))
    print(f"Событий получено: {len(events)}")
    arbs = find_arbs(events, cfg)
    if not arbs:
        print(f"Арбитражей с маржой ≥ {cfg.arb_min_edge * 100:.0f}% сейчас нет — это нормально, "
              f"они живут минуты. Держите сканер в цикле auto.")
        return
    for a in arbs:
        print(f"\n«{a.title}»: комплект из {len(a.legs)} исходов за ${a.sum_asks:.3f} "
              f"→ +{a.edge * 100:.1f}% гарантированно")
        for leg in a.legs:
            print(f"    {leg.ask:>6.3f}  {leg.question[:70]}")


def cmd_history(args: argparse.Namespace) -> None:
    log = BetLog()
    bets = log.bets
    if not bets:
        print("Ставок ещё не было.")
        return
    live = [b for b in bets if b.get("live")]
    sells = [b for b in bets if b.get("side") == "SELL"]
    print(f"Всего записей: {len(bets)} (реальных: {len(live)}, dry-run: {len(bets) - len(live)}, "
          f"продаж: {len(sells)})")
    print(f"Потрачено реально: ${log.spent_usd():,.2f} (сегодня: ${log.spent_today_usd():,.2f})")
    for b in bets[-30:]:
        tag = "LIVE" if b.get("live") else "dry "
        side = b.get("side", "BUY")
        print(f"  {b['ts']} [{tag}] {side} {b['price']:.3f} x {b['size']:,.0f} = ${b['usd']:,.2f} "
              f"[{b['outcome']}] {b['question'][:55]} ({b['status']})")


def cmd_positions(args: argparse.Namespace) -> None:
    address = args.address or os.environ.get("POLYMARKET_FUNDER")
    if not address:
        sys.exit("Укажите адрес: --address 0x... или переменную POLYMARKET_FUNDER.")
    cfg = _load_config(args)
    positions = autopilot.fetch_positions(cfg, address, requests.Session())
    if not positions:
        print("Открытых позиций нет.")
        return
    total_value = total_pnl = 0.0
    for p in positions:
        value = float(p.get("currentValue") or 0)
        pnl = float(p.get("cashPnl") or 0)
        total_value += value
        total_pnl += pnl
        flag = " <-- МОЖНО ЗАБРАТЬ" if p.get("redeemable") else ""
        print(f"  {p.get('title', '?')[:70]} [{p.get('outcome')}] "
              f"{float(p.get('size') or 0):,.0f} шт, вход {float(p.get('avgPrice') or 0):.3f}, "
              f"сейчас {float(p.get('curPrice') or 0):.3f}, PnL ${pnl:,.2f}{flag}")
    print(f"\nСтоимость позиций: ${total_value:,.2f}, суммарный PnL: ${total_pnl:,.2f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="polymarket_bot", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="показать кандидатов со скорами, без ставок")
    _add_common(p_scan)
    p_scan.set_defaults(func=cmd_scan)

    p_run = sub.add_parser("run", help="один торговый цикл (dry-run по умолчанию)")
    _add_common(p_run)
    p_run.add_argument("--live", action="store_true", help="реальные ордера вместо dry-run")
    p_run.add_argument("--yes", action="store_true", help="не спрашивать подтверждение в --live")
    p_run.set_defaults(func=cmd_run)

    p_auto = sub.add_parser("auto", help="автопилот: циклы по расписанию")
    _add_common(p_auto)
    p_auto.add_argument("--live", action="store_true", help="реальные ордера вместо dry-run")
    p_auto.add_argument("--yes", action="store_true", help="не спрашивать подтверждение в --live")
    p_auto.add_argument("--once", action="store_true", help="один цикл и выход (для cron)")
    p_auto.set_defaults(func=cmd_auto)

    p_arb = sub.add_parser("arb", help="сканер арбитражей neg-risk событий")
    _add_common(p_arb)
    p_arb.set_defaults(func=cmd_arb)

    p_hist = sub.add_parser("history", help="история ставок бота")
    p_hist.set_defaults(func=cmd_history)

    p_pos = sub.add_parser("positions", help="позиции и PnL кошелька")
    _add_common(p_pos)
    p_pos.add_argument("--address", help="адрес кошелька/прокси Polymarket")
    p_pos.set_defaults(func=cmd_positions)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except requests.RequestException as exc:
        sys.exit(f"Сетевая ошибка при обращении к API Polymarket: {exc}\n"
                 f"Проверьте доступ в интернет (Polymarket недоступен из ряда стран без VPN).")


if __name__ == "__main__":
    main()
