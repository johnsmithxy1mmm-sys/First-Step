"""Master-prompt infrastructure: fees, rate limiter, WS BookStore, kill-switch."""

import time
from unittest import mock

import pytest

from polymarket_bot.clob import ClobReader, Trader
from polymarket_bot.fees import FeeModel
from polymarket_bot.ratelimit import RateLimited, TokenBucket
from polymarket_bot.risk import KillSwitch
from polymarket_bot.ws_feed import BookStore


# --- V2 fees ---

def test_taker_coefficients_by_category(cfg):
    fees = FeeModel(cfg.fees)
    assert fees.taker_coef("crypto") == pytest.approx(0.07)
    assert fees.taker_coef("sports") == pytest.approx(0.03)
    assert fees.taker_coef("geopolitics") == 0.0
    assert fees.taker_coef("elections") == pytest.approx(0.04)   # politics
    assert fees.taker_coef("nature") == pytest.approx(0.05)      # weather
    # The Gamma category takes priority over our classifier.
    assert fees.taker_coef("other", gamma_category="Sports") == pytest.approx(0.03)


def test_official_worked_example_175_per_100_shares(cfg):
    """docs.polymarket.com: 100 shares at 50c in crypto = $1.75 matched fee."""
    fees = FeeModel(cfg.fees)
    assert 100 * fees.taker_fee_per_share("crypto", 0.50) == pytest.approx(1.75)


def test_fee_is_symmetric_in_price(cfg):
    """theta*p*(1-p) is unchanged by p -> 1-p: taking YES or NO costs the same."""
    fees = FeeModel(cfg.fees)
    for p in (0.05, 0.2, 0.37, 0.5):
        assert fees.taker_fee_per_share("politics", p) == \
            pytest.approx(fees.taker_fee_per_share("politics", 1.0 - p))


def test_fee_collapses_near_the_dollar(cfg):
    """The whole point of the price term: a 95c fill is ~20x cheaper to take
    than a flat-fraction model claims. Regression guard for the near-$1 desk."""
    fees = FeeModel(cfg.fees)
    flat = 0.04 * 0.95                                    # the old, wrong model
    real = fees.taker_fee_per_share("politics", 0.95)
    assert real == pytest.approx(0.04 * 0.95 * 0.05)
    assert flat / real == pytest.approx(20.0, rel=1e-6)


def test_maker_rebate_per_category(cfg):
    fees = FeeModel(cfg.fees)
    # Official shares: 25% default, 20% crypto, 15% sports.
    assert fees.rebate_share("politics") == pytest.approx(0.25)
    assert fees.rebate_share("crypto") == pytest.approx(0.20)
    assert fees.rebate_share("sports") == pytest.approx(0.15)
    assert fees.maker_rebate_per_share("crypto", 0.5) == \
        pytest.approx(0.07 * 0.25 * 0.20)


def test_net_taker_edge_uses_the_price(cfg):
    fees = FeeModel(cfg.fees)
    # 10c gross edge on a crypto fill at 50c loses only theta/4 = 1.75c.
    assert fees.net_taker_edge(0.10, "crypto", 0.50) == pytest.approx(0.10 - 0.0175)


def test_mm_min_half_spread(cfg):
    fees = FeeModel(cfg.fees)
    # geopolitics: rebate 0 -> half-spread = min_edge / 2, at any price.
    assert fees.mm_min_half_spread("geopolitics", 0.5, min_edge_after_fees=0.02) \
        == pytest.approx(0.01)
    # A near-$1 market earns almost no rebate, so the floor stays near min/2 --
    # the spread must carry the edge itself.
    near = fees.mm_min_half_spread("politics", 0.97, min_edge_after_fees=0.02)
    mid = fees.mm_min_half_spread("politics", 0.50, min_edge_after_fees=0.02)
    assert near > mid
    assert near == pytest.approx((0.02 - 2 * 0.04 * 0.97 * 0.03 * 0.25) / 2)


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


# --- rate limiter WIRED into the clients (not just the bucket in isolation) ---

def _drained(rate: float = 0.0) -> TokenBucket:
    bucket = TokenBucket(rate_per_sec=rate, burst=1.0)
    assert bucket.try_acquire()          # burst spent -> next acquire fails fast
    return bucket


def test_reader_drops_reads_when_bucket_is_dry(cfg):
    reader = ClobReader(cfg, client=mock.Mock())
    reader._reads = _drained()
    assert reader.order_book("tok") is None
    assert reader.price_history("tok", 0, 1) == []


def _bare_trader() -> Trader:
    """Trader with the network constructor bypassed — we test throttling only."""
    t = object.__new__(Trader)
    t._client = mock.Mock()
    t._orders = TokenBucket(rate_per_sec=1000.0, burst=100.0)
    t._reads = TokenBucket(rate_per_sec=1000.0, burst=100.0)
    t._funder = "0xfunder"
    t._data_api = "https://data.example"
    return t


def test_order_not_placed_when_bucket_is_dry():
    """A dropped order is safe (no orderID = no fill); a ban is not."""
    t = _bare_trader()
    t._orders = _drained()
    assert t.buy_limit("tok", 0.5, 10.0) == {}
    t._client.post_order.assert_not_called()


def test_kill_switch_cancel_all_is_never_throttled():
    """Throttling the one call that flattens the book is how safety kills you."""
    t = _bare_trader()
    t._orders = _drained()
    t.cancel_all()
    t._client.cancel_all.assert_called_once()
    # A single cancel also proceeds: risk reduction is never refused.
    t.cancel(  "order-1")
    t._client.cancel.assert_called_once_with("order-1")


def test_throttled_reads_raise_rather_than_look_empty():
    """An empty list here would read as 'nothing on the exchange' and silently
    disable the desync kill-switch / allow a double entry."""
    t = _bare_trader()
    t._reads = _drained()
    with pytest.raises(RateLimited):
        t.open_orders()
    t._reads = _drained()
    with pytest.raises(RateLimited):
        t.api_positions()
    t._reads = _drained()
    with pytest.raises(RateLimited):
        t.order_status("order-1")


def test_executor_fails_safe_on_rate_limited_reconcile(cfg, ledger):
    """already_entered must assume a position exists when it cannot verify."""
    from polymarket_bot.executor import Executor
    trader = mock.Mock()
    trader.api_positions.side_effect = RateLimited("dry")
    ex = Executor(cfg, ledger, clob=mock.Mock(), trader=trader, mode="live")
    assert ex.already_entered("tok-never-traded") is True


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
