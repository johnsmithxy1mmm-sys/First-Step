"""F-003 (CRITICAL): one un-validated ledger row turns every halt into a no-op.

`Ledger.record_trade` performs no numeric validation (INV-11). A single row with
`size = inf` (or any NaN-producing combination) makes `total_exposure` and
`realized_pnl` return NaN/inf, and NaN propagates into equity.

The kill-switch decides with `>=` comparisons. **Every comparison against NaN is
False**, so:

    loss >= max_daily_loss_usd          -> False  (daily stop never fires)
    (hwm - equity)/hwm >= max_drawdown  -> False  (drawdown halt never fires)
    exposure >= max_global_exposure_usd -> False  (entry gate never blocks)

The bot keeps trading with its entire safety layer silently disabled — no error,
no alert, no log line. This is strictly worse than crashing: a crash stops
trading, this does not.

Who can produce such a row? Any caller that computes a size from external data:
`size = floor(usd / price)` with a poisoned book price (see F-001, where NaN and
Infinity reach `top.bid`), a partial-fill quantity read from an exchange
response, or a mis-parsed Gamma field.
"""

import math

import pytest

from polymarket_bot.models import simple_estimate
from polymarket_bot.risk import KillSwitch


def _poison(ledger, market, size=float("inf"), price=0.5):
    ledger.record_trade(mode="paper", estimate=simple_estimate(market, 0, price),
                        category="mm", side="BUY", price=price, size=size,
                        order_id=None, status="filled", strategy="mm")


def test_record_trade_rejects_non_finite_size(cfg, ledger):
    from polymarket_bot.tests.conftest import make_market
    m = make_market(id="p1", clob_token_ids=["p1-y", "p1-n"])
    with pytest.raises(Exception):
        _poison(ledger, m, size=float("inf"))


def test_record_trade_rejects_negative_size(cfg, ledger):
    from polymarket_bot.tests.conftest import make_market
    m = make_market(id="p2", clob_token_ids=["p2-y", "p2-n"])
    with pytest.raises(Exception):
        _poison(ledger, m, size=-100.0)


def test_record_trade_rejects_price_outside_zero_one(cfg, ledger):
    from polymarket_bot.tests.conftest import make_market
    m = make_market(id="p3", clob_token_ids=["p3-y", "p3-n"])
    with pytest.raises(Exception):
        _poison(ledger, m, price=1.7)


def test_exposure_stays_finite_after_a_bad_row(cfg, ledger):
    from polymarket_bot.tests.conftest import make_market
    m = make_market(id="p4", clob_token_ids=["p4-y", "p4-n"])
    try:
        _poison(ledger, m, size=float("inf"))
    except Exception:
        pytest.skip("row rejected at write time — good")
    assert math.isfinite(ledger.total_exposure("paper")), "exposure is NaN/inf"
    assert math.isfinite(ledger.realized_pnl("paper")), "realized PnL is NaN/inf"


def test_daily_loss_halt_still_fires_with_a_poisoned_ledger(cfg, ledger):
    """The end-to-end consequence: the daily stop becomes unreachable."""
    from polymarket_bot.tests.conftest import make_market
    m = make_market(id="p5", clob_token_ids=["p5-y", "p5-n"])
    try:
        _poison(ledger, m, size=float("inf"))
    except Exception:
        pytest.skip("row rejected at write time — good")

    halts = []
    ks = KillSwitch(cfg, ledger, "paper", cancel_all=lambda: None,
                    alert=lambda msg: halts.append(msg) or True)
    equity = cfg.portfolio.bankroll_usd - ledger.total_exposure("paper")  # NaN
    ks.check_daily_loss(equity)
    ks.check_drawdown(equity, hwm=cfg.portfolio.bankroll_usd)
    assert ks.halted, (
        "neither the daily-loss nor the drawdown halt fired: every `>=` against "
        "NaN is False, so the whole safety layer is a no-op")
