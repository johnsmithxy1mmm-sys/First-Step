"""Маркет-мейкер: котировки, guard от adverse selection, инвентарь, dry-run."""

from unittest import mock

import pytest

from polymarket_bot.marketmaker import MarketMaker
from polymarket_bot.models import BookLevel, OrderBook, simple_estimate

from .conftest import make_market


def mid_market(**overrides):
    defaults = dict(
        id="mm1", question="Will the Fed cut rates in September?",
        outcome_prices=[0.45, 0.55],
        clob_token_ids=["mm1-yes", "mm1-no"],
        volume_24h_usd=50_000, volume_usd=1_000_000,
    )
    defaults.update(overrides)
    return make_market(**defaults)


def book(bid=0.43, ask=0.47) -> OrderBook:
    return OrderBook(bids=[BookLevel(price=bid, size=5000)],
                     asks=[BookLevel(price=ask, size=5000)])


def make_mm(cfg, ledger, trader=None) -> MarketMaker:
    cfg.market_maker.enabled = True
    clob = mock.Mock()
    clob.order_book.return_value = book()
    return MarketMaker(cfg, ledger, clob, trader,
                       "live" if trader else "dry-run")


def test_selects_liquid_mid_markets_only(cfg, ledger):
    mm = make_mm(cfg, ledger)
    tail = mid_market(id="t", outcome_prices=[0.02, 0.98])
    illiquid = mid_market(id="i", volume_24h_usd=100)
    good = mid_market()
    selected = mm.select_markets([tail, illiquid, good])
    assert [m.id for m in selected] == ["mm1"]


def test_quote_symmetric_and_non_crossing(cfg, ledger):
    mm = make_mm(cfg, ledger)
    quote = mm.compute_quote(mid_market(), book(bid=0.43, ask=0.47))  # mid 0.45
    assert quote is not None
    assert quote.yes_bid == pytest.approx(0.44)          # mid - 0.01
    assert quote.no_bid == pytest.approx(1 - 0.46)       # 1 - (mid + 0.01)
    assert quote.captured_spread == pytest.approx(0.02)  # прибыль пары = полный спред
    assert quote.yes_bid < 0.47                          # не пересекаем ask


def test_no_quote_when_spread_too_tight(cfg, ledger):
    mm = make_mm(cfg, ledger)
    # Книжный спред 0.002 < наш полуспред 0.01: зарабатывать нечего.
    assert mm.compute_quote(mid_market(), book(bid=0.449, ask=0.451)) is None


def test_guard_on_price_jump(cfg, ledger):
    mm = make_mm(cfg, ledger)
    m = mid_market()
    assert not mm.guard_blocks(m, book(bid=0.43, ask=0.47))   # первый замер
    assert mm.guard_blocks(m, book(bid=0.48, ask=0.52))       # mid +0.05 >= 0.03
    # Cooldown держится заданное число циклов.
    for _ in range(cfg.market_maker.guard_cooldown_cycles):
        assert mm.guard_blocks(m, book(bid=0.48, ask=0.52))
    assert not mm.guard_blocks(m, book(bid=0.48, ask=0.52))


def test_guard_on_volume_spike(cfg, ledger):
    mm = make_mm(cfg, ledger)
    shocked = mid_market(volume_24h_usd=600_000, volume_usd=1_000_000)  # 60% за сутки
    assert mm.guard_blocks(shocked, book())


def test_inventory_cap_disables_heavy_side(cfg, ledger):
    cfg.market_maker.inventory_cap_usd = 100.0
    mm = make_mm(cfg, ledger)
    m = mid_market()
    assert mm.sides_allowed(m) == (True, True)
    # Накопили Yes на $150 (> кэпа): бид на Yes выключается, No остаётся.
    ledger.record_trade(mode="dry-run", estimate=simple_estimate(m, 0, 0.45),
                        category="mm", side="BUY", price=0.45, size=333.4,
                        order_id=None, status="filled", strategy="mm")
    quote_yes, quote_no = mm.sides_allowed(m)
    assert not quote_yes and quote_no


def test_dry_run_places_no_orders(cfg, ledger):
    mm = make_mm(cfg, ledger)
    quotes = mm.cycle([mid_market()])
    assert len(quotes) == 1
    assert mm._orders == {}                       # ордеров нет — только лог


def test_live_cancel_replace_and_fill_sync(cfg, ledger):
    trader = mock.Mock()
    trader.buy_limit.side_effect = [{"orderID": f"o{i}"} for i in range(10)]
    trader.order_status.return_value = {"status": "live", "size_matched": 40.0}
    mm = make_mm(cfg, ledger, trader=trader)
    m = mid_market()

    mm.cycle([m])
    assert trader.buy_limit.call_count == 2       # Yes-бид + No-бид
    assert len(mm._orders[m.id]) == 2

    mm.cycle([m])                                 # второй цикл: sync fills + cancel-replace
    assert trader.cancel.call_count == 2          # старые котировки сняты
    fills = [p for p in ledger.open_positions("live")]
    assert len(fills) == 2                        # частичные исполнения записаны
    assert all(p.size == pytest.approx(40.0) for p in fills)

    mm.shutdown()
    assert trader.cancel.call_count == 4          # свежие котировки тоже сняты
