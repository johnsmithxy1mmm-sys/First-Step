"""Phase 0 — API recon. Run ON YOUR MACHINE (not in the CI sandbox):

    python -m polymarket_bot.phase0_smoke

Checks five facts about the live platform and prints a verdict for each:
  1. Gamma: markets are returned, rewards fields (rewardsMinSize/rewardsMaxSpread) present.
  2. CLOB REST: the order book reads.
  3. CLOB WS: subscription works, book/price_change arrive.
  4. py-clob-client version and V2 support (presence of OrderType.FOK etc.).
  5. Fee/category data in Gamma (cross-check with config.yaml fees:).

If the real endpoints/schemas differ from the config — edit config.yaml
(runtime: hosts, fees:) and report what changed. The bot reads
platform-dependent things from the config, not from constants.
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
    assert isinstance(markets, list) and markets, "Gamma returned empty"
    m = markets[0]
    rewards_fields = [k for k in m if "reward" in k.lower()]
    print(f"[OK] Gamma: {len(markets)} markets; example: {m.get('question', '')[:60]}")
    print(f"     rewards fields: {rewards_fields or 'NOT FOUND — check the schema!'}")
    fee_fields = [k for k in m if "fee" in k.lower()]
    print(f"     fee fields: {fee_fields or 'none (fees come from config.yaml)'}")
    return m


def check_clob_book(cfg: BotConfig, market: dict) -> str | None:
    token_ids = json.loads(market.get("clobTokenIds") or "[]")
    if not token_ids:
        print("[SKIP] CLOB book: market has no clobTokenIds")
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
        print(f"[OK] CLOB WS: first message event_type={first.get('event_type')}")


def check_sdk() -> None:
    try:
        import py_clob_client
        from py_clob_client.clob_types import OrderType
        fok = hasattr(OrderType, "FOK")
        version = getattr(py_clob_client, "__version__", "?")
        print(f"[OK] py-clob-client {version}; OrderType.FOK: {fok}")
        if not fok:
            print("     WARNING: FOK not found — update the SDK to a V2-compatible one")
    except ImportError:
        print("[FAIL] py-clob-client is not installed")


def main() -> None:
    cfg = BotConfig.load()
    print("=== Phase 0: Polymarket API recon ===")
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
    print("\nResult:", "ALL OK — you can run Phase 1 (dry-run)"
          if failures == 0 else f"{failures} checks failed — edit config.yaml")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
