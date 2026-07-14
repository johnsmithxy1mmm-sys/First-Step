"""Master-prompt infrastructure: fees, rate limiter, WS BookStore, kill-switch."""

import time
from unittest import mock

import pytest

from polymarket_bot.fees import FeeModel
from polymarket_bot.ratelimit import TokenBucket
from polymarket_bot.risk import KillSwitch
from polymarket_bot.ws_feed import BookStore


# --- V2 fees ---

def test_taker_fees_by_category(cfg):
    fees = FeeModel(cfg.fees)
    assert fees.taker_fee("crypto") == pytest.approx(0.07)
    assert fees.taker_fee("sports") == pytest.approx(0.03)
    assert fees.taker_fee("geopolitics") == 0.0
    assert fees.taker_fee("elections") == pytest.approx(0.04)   # politics
    assert fees.taker_fee("nature") == pytest.approx(0.05)      # weather
    # The Gamma category takes priority over our classifier.
    assert fees.taker_fee("other", gamma_category="Sports") == pytest.approx(0.03)


def test_maker_rebate_and_net_edge(cfg):
    fees = FeeModel(cfg.fees)
    assert fees.maker_rebate("crypto") == pytest.approx(0.07 * 0.35)
    # The aggressive leg on crypto loses 7 pp of edge.
    assert fees.net_taker_edge(0.10, "crypto") == pytest.approx(0.03)


def test_mm_min_half_spread(cfg):
    fees = FeeModel(cfg.fees)
    # geopolitics: rebate 0 -> half-spread = min_edge / 2.
    assert fees.mm_min_half_spread("geopolitics", min_edge_after_fees=0.02) \
        == pytest.approx(0.01)
    # sports: rebate 2*0.03*0.35=0.021 covers min_edge 0.01 -> threshold 0.
    assert fees.mm_min_half_spread("sports", min_edge_after_fees=0.01) == 0.0


# --- token bucket ---

def test_token_bucket_burst_and_refill():
    bucket = TokenBucket(rate_per_sec=100.0, burst=5.0)
    assert all(bucket.try_acquire() for _ in range(5))   # burst consumed
    assert not bucket.try_acquire()
    time.sleep(0.03)                                     # ~3 tokens refilled
    assert bucket.try_acquire()


def test_token_bucket_acquire_blocks_until_refill():
    bucket = TokenBucket(rate_per_sec=50.0, burst=1.0)
    assert bucket.acquire()
    start = time.monotonic()
    assert bucket.acquire(timeout=1.0)                   # waits ~20ms
    assert time.monotonic() - start >= 0.01


# --- WS BookStore ---

def test_bookstore_snapshot_and_top():
    store = BookStore()
    token = store.handle({
        "event_type": "book", "asset_id": "tok",
        "bids": [{"price": "0.44", "size": "100"}, {"price": "0.43", "size": "50"}],
        "asks": [{"price": "0.46", "size": "80"}],
    })
    assert token == "tok"
    top = store.top("tok")
    assert top.bid == pytest.approx(0.44)
    assert top.ask == pytest.approx(0.46)
    assert top.mid == pytest.approx(0.45)


def test_bookstore_price_change_updates_levels():
    store = BookStore()
    store.handle({"event_type": "book", "asset_id": "tok",
                  "bids": [{"price": "0.44", "size": "100"}],
                  "asks": [{"price": "0.46", "size": "80"}]})
    store.handle({"event_type": "price_change", "asset_id": "tok",
                  "changes": [{"price": "0.45", "side": "BUY", "size": "60"},
                              {"price": "0.46", "side": "SELL", "size": "0"},
                              {"price": "0.47", "side": "SELL", "size": "40"}]})
    top = store.top("tok")
    assert top.bid == pytest.approx(0.45)   # new best bid
    assert top.ask == pytest.approx(0.47)   # 0.46 removed (size 0)


def test_bookstore_ignores_garbage():
    store = BookStore()
    assert store.handle({}) is None
    assert store.handle({"event_type": "book"}) is None
    assert store.top("nope") is None


# --- kill-switch ---

def make_ks(cfg, ledger, mode="paper"):
    cancel = mock.Mock()
    alert = mock.Mock(return_value=True)
    return KillSwitch(cfg, ledger, mode, cancel_all=cancel, alert=alert), cancel, alert


def test_killswitch_daily_loss_halts(cfg, ledger):
    ks, cancel, alert = make_ks(cfg, ledger)
    ledger.snapshot_bank(cash=5000, exposure=0)          # day start: equity 5000
    ks.check_daily_loss(equity_now=4990)                 # -10 < limit 25
    assert ks.trading_allowed
    ks.check_daily_loss(equity_now=4970)                 # -30 >= 25 -> halt
    assert ks.halted and not ks.trading_allowed
    cancel.assert_called_once()
    alert.assert_called_once()


def test_killswitch_drawdown_halts(cfg, ledger):
    ks, cancel, _ = make_ks(cfg, ledger)
    ks.check_drawdown(equity_now=4000, hwm=5000)         # -20% >= 15% -> halt
    assert ks.halted


def test_killswitch_ws_pause_and_resume(cfg, ledger):
    ks, cancel, _ = make_ks(cfg, ledger)
    ks.on_ws_disconnect(12.0)
    assert ks.paused and not ks.halted
    cancel.assert_called_once()                          # quotes pulled
    ks.on_ws_recovered()
    assert ks.trading_allowed                            # auto-resume


def test_killswitch_reconcile_mismatch_halts(cfg, ledger):
    ks, _, _ = make_ks(cfg, ledger)
    ks.reconcile(local_order_ids={"a"}, exchange_order_ids={"a", "ghost-1"})
    assert ks.halted
    assert "reconcile" in ks.reason


def test_killswitch_global_exposure_gate(cfg, ledger):
    ks, _, _ = make_ks(cfg, ledger)
    assert not ks.check_global_exposure(cfg.risk.max_global_exposure_usd - 1)
    assert ks.check_global_exposure(cfg.risk.max_global_exposure_usd)
    assert not ks.halted                                 # a gate, not an emergency
