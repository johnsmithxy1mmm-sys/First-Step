"""CLI бота: python -m polymarket_bot <команда>.

Команды:
  scan       — найти дешёвые исходы и показать таблицу кандидатов
  run        — разместить ставки (по умолчанию dry-run; реальные ордера с --live)
  history    — история ставок бота
  positions  — текущие позиции и PnL кошелька (data-api Polymarket)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import requests

from . import clob
from .config import BotConfig
from .gamma import iter_active_markets
from .storage import BetLog
from .strategy import Candidate, find_candidates, plan_bets


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


def _collect_candidates(cfg: BotConfig, log: BetLog) -> list[Candidate]:
    print("Сканирую активные рынки Polymarket (это может занять минуту-две)...")
    markets = list(iter_active_markets(cfg))
    print(f"Рынков получено: {len(markets)}")
    candidates = find_candidates(markets, cfg, skip_token_ids=log.live_token_ids())
    print(f"Кандидатов после фильтров (цена ≤ {cfg.max_price:g}): {len(candidates)}")
    return candidates


def _print_table(rows: list[Candidate], limit: int = 50) -> None:
    if not rows:
        print("Ничего не найдено. Попробуйте ослабить фильтры (--max-price, min_liquidity_usd).")
        return
    print(f"{'ЦЕНА':>7} {'ИКСЫ':>6} {'ЛИКВ.$':>10} {'ДО КОНЦА':>9}  ИСХОД / ВОПРОС")
    for c in rows[:limit]:
        days = "-"
        if c.end_date is not None:
            from datetime import datetime, timezone
            days = f"{(c.end_date - datetime.now(timezone.utc)).days}д"
        print(f"{c.price:>7.3f} {c.payout_multiple:>5.0f}x {c.liquidity_usd:>10,.0f} {days:>9}  "
              f"[{c.outcome}] {c.question[:80]}")
    if len(rows) > limit:
        print(f"... и ещё {len(rows) - limit}")


def cmd_scan(args: argparse.Namespace) -> None:
    cfg = _load_config(args)
    log = BetLog()
    candidates = _collect_candidates(cfg, log)
    _print_table(candidates)


def cmd_run(args: argparse.Namespace) -> None:
    cfg = _load_config(args)
    log = BetLog()

    remaining = cfg.total_budget_usd - log.spent_usd()
    if remaining < cfg.stake_usd:
        sys.exit(f"Бюджет исчерпан: потрачено {log.spent_usd():.2f} из {cfg.total_budget_usd:.2f}.")
    cfg.total_budget_usd = remaining

    candidates = plan_bets(_collect_candidates(cfg, log), cfg)
    if not candidates:
        return

    trader = None
    if args.live:
        if not args.yes:
            total = min(len(candidates) * cfg.stake_usd, cfg.total_budget_usd)
            answer = input(f"РЕАЛЬНЫЕ ставки: до {len(candidates)} ордеров, "
                           f"до ${total:,.0f}. Продолжить? [yes/N] ")
            if answer.strip().lower() not in ("y", "yes"):
                sys.exit("Отменено.")
        trader = clob.Trader(cfg)
    else:
        print("\n=== DRY-RUN: ордера НЕ размещаются. Для реальной торговли добавьте --live ===\n")

    session = requests.Session()
    placed = skipped = 0
    spent = shares_total = 0.0
    for c in candidates:
        if spent + cfg.stake_usd > cfg.total_budget_usd:
            break

        price = c.price
        if cfg.verify_orderbook:
            quote = clob.get_best_ask(cfg, c.token_id, session)
            time.sleep(cfg.request_delay_sec)
            if quote is None or quote.best_ask > cfg.max_price:
                skipped += 1
                continue  # в реальном стакане дешёвой цены нет
            price = quote.best_ask

        price = clob.round_to_tick(price, c.tick_size)
        if not cfg.min_price <= price <= cfg.max_price:
            skipped += 1
            continue
        size = clob.shares_for_stake(cfg.stake_usd, price, c.min_order_size)
        if size <= 0:
            skipped += 1
            continue

        if trader is not None:
            try:
                resp = trader.buy_limit(c, price, size)
            except Exception as exc:  # не роняем всю серию из-за одного рынка
                print(f"  ОШИБКА [{c.outcome}] {c.question[:60]}: {exc}")
                skipped += 1
                continue
            order_id = (resp or {}).get("orderID")
            status = (resp or {}).get("status", "unknown")
            log.record(candidate=c, price=price, size=size, live=True,
                       order_id=order_id, status=status)
        else:
            log.record(candidate=c, price=price, size=size, live=False, status="dry-run")

        placed += 1
        spent += price * size
        shares_total += size
        mult = 1 / price if price else 0
        print(f"  {'ОРДЕР' if trader else 'ПЛАН '} {price:.3f} x {size:,.0f} шт = "
              f"${price * size:,.2f} (до {mult:,.0f}x) [{c.outcome}] {c.question[:70]}")

    mode = "размещено ордеров" if trader else "запланировано (dry-run)"
    print(f"\nИтого {mode}: {placed}, пропущено: {skipped}, задействовано: ${spent:,.2f}")
    if placed:
        avg_payout = shares_total / placed  # акция платит $1 при победе
        print(f"Средний выигрыш одной победившей ставки: ~${avg_payout:,.0f} "
              f"(окупает ~{avg_payout / cfg.stake_usd:,.0f} сгоревших ставок).")


def cmd_history(args: argparse.Namespace) -> None:
    log = BetLog()
    bets = log.bets
    if not bets:
        print("Ставок ещё не было.")
        return
    live = [b for b in bets if b.get("live")]
    print(f"Всего записей: {len(bets)} (реальных: {len(live)}, dry-run: {len(bets) - len(live)})")
    print(f"Потрачено реально: ${log.spent_usd():,.2f}")
    for b in bets[-30:]:
        tag = "LIVE" if b.get("live") else "dry "
        print(f"  {b['ts']} [{tag}] {b['price']:.3f} x {b['size']:,.0f} = ${b['usd']:,.2f} "
              f"[{b['outcome']}] {b['question'][:60]} ({b['status']})")


def cmd_positions(args: argparse.Namespace) -> None:
    address = args.address or os.environ.get("POLYMARKET_FUNDER")
    if not address:
        sys.exit("Укажите адрес: --address 0x... или переменную POLYMARKET_FUNDER.")
    cfg = _load_config(args)
    resp = requests.get(f"{cfg.data_api_host}/positions",
                        params={"user": address, "limit": 500}, timeout=30)
    resp.raise_for_status()
    positions = resp.json()
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

    p_scan = sub.add_parser("scan", help="показать кандидатов без ставок")
    _add_common(p_scan)
    p_scan.set_defaults(func=cmd_scan)

    p_run = sub.add_parser("run", help="разместить ставки (dry-run по умолчанию)")
    _add_common(p_run)
    p_run.add_argument("--live", action="store_true", help="реальные ордера вместо dry-run")
    p_run.add_argument("--yes", action="store_true", help="не спрашивать подтверждение в --live")
    p_run.set_defaults(func=cmd_run)

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
