"""Фаза 0 — разведка API. Запускать НА ВАШЕЙ МАШИНЕ (не в CI-песочнице):

    python -m polymarket_bot.phase0_smoke

Проверяет пять фактов о живой платформе и печатает вердикт по каждому:
  1. Gamma: рынки отдаются, поля rewards (rewardsMinSize/rewardsMaxSpread) на месте.
  2. CLOB REST: стакан читается.
  3. CLOB WS: подписка работает, приходят book/price_change.
  4. Версия py-clob-client и поддержка V2 (наличие OrderType.FOK и т.п.).
  5. Данные о комиссиях/категориях в Gamma (сверьте с config.yaml fees:).

Если реальные эндпоинты/схемы отличаются от конфига — правьте config.yaml
(runtime: hosts, fees:) и сообщите, что изменилось. Код бота платформенно-
зависимые вещи читает из конфига, не из констант.
"""

from __future__ import annotations

import asyncio
import json
import sys

from .config import BotConfig
from .http_client import get_with_backoff, make_client


def check_gamma(cfg: BotConfig) -> dict | None:
    client = make_client(20)
    resp = get_with_backoff(client, f"{cfg.runtime.gamma_host}/markets",
                            params={"active": "true", "closed": "false",
                                    "limit": 5}, max_retries=2)
    markets = resp.json()
    assert isinstance(markets, list) and markets, "Gamma вернул пусто"
    m = markets[0]
    rewards_fields = [k for k in m if "reward" in k.lower()]
    print(f"[OK] Gamma: {len(markets)} рынков; пример: {m.get('question', '')[:60]}")
    print(f"     rewards-поля: {rewards_fields or 'НЕ НАЙДЕНЫ — проверьте схему!'}")
    fee_fields = [k for k in m if "fee" in k.lower()]
    print(f"     fee-поля: {fee_fields or 'нет (комиссии берём из config.yaml)'}")
    return m


def check_clob_book(cfg: BotConfig, market: dict) -> str | None:
    token_ids = json.loads(market.get("clobTokenIds") or "[]")
    if not token_ids:
        print("[SKIP] CLOB book: у рынка нет clobTokenIds")
        return None
    client = make_client(20)
    resp = get_with_backoff(client, f"{cfg.runtime.clob_host}/book",
                            params={"token_id": token_ids[0]}, max_retries=2)
    book = resp.json()
    print(f"[OK] CLOB book: bids={len(book.get('bids', []))} "
          f"asks={len(book.get('asks', []))}")
    return token_ids[0]


async def check_ws(cfg: BotConfig, token_id: str) -> None:
    import websockets

    async with websockets.connect(cfg.ws.url, close_timeout=5) as ws:
        await ws.send(json.dumps({"type": "market", "assets_ids": [token_id]}))
        raw = await asyncio.wait_for(ws.recv(), timeout=15)
        payload = json.loads(raw)
        first = payload[0] if isinstance(payload, list) else payload
        print(f"[OK] CLOB WS: первое сообщение event_type={first.get('event_type')}")


def check_sdk() -> None:
    try:
        import py_clob_client
        from py_clob_client.clob_types import OrderType
        fok = hasattr(OrderType, "FOK")
        version = getattr(py_clob_client, "__version__", "?")
        print(f"[OK] py-clob-client {version}; OrderType.FOK: {fok}")
        if not fok:
            print("     ВНИМАНИЕ: FOK не найден — обновите SDK до V2-совместимого")
    except ImportError:
        print("[FAIL] py-clob-client не установлен")


def main() -> None:
    cfg = BotConfig.load()
    print("=== Фаза 0: разведка API Polymarket ===")
    failures = 0
    market = None
    token = None
    for name, fn in (("gamma", lambda: check_gamma(cfg)),):
        try:
            market = fn()
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {name}: {exc}")
    if market:
        try:
            token = check_clob_book(cfg, market)
        except Exception as exc:
            failures += 1
            print(f"[FAIL] clob book: {exc}")
    if token:
        try:
            asyncio.run(check_ws(cfg, token))
        except Exception as exc:
            failures += 1
            print(f"[FAIL] ws: {exc}")
    check_sdk()
    print("\nИтог:", "ВСЁ ОК — можно запускать Фазу 1 (dry-run)"
          if failures == 0 else f"{failures} проверок упало — правьте config.yaml")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
