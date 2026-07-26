"""F-007: the desync kill-switch only knows MM/sprint orders — everything else
looks like a ghost and HALTS the bot.

`main.risk_job` calls:

    killswitch.reconcile(mm.local_order_ids() | sprint.local_order_ids(),
                         exchange_ids)

and `reconcile` HALTS on `exchange_ids - local_ids`. But three other paths place
orders that can REST on the exchange as plain GTC:

    executor.py:138   buy_limit   GTC (longshot / fade entries)
    executor.py:209   sell_limit  GTC (take-profit / guardian exits)
    arbitrage.py:224  buy_limit   GTC (neg-risk basket legs)

None of their ids are in the "local" set. A single resting entry or exit order —
the normal state between placement and fill — is therefore read as an unknown
exchange order and trips a terminal HALT requiring manual restart.

This is the mirror image of the usual concern: not "desync goes unnoticed" but
"correct state misread as desync". It also erodes the real signal, because the
first false HALT teaches the operator to distrust it.
"""

from unittest import mock

from polymarket_bot.risk import KillSwitch


def test_executor_orders_are_visible_to_reconcile(cfg, ledger):
    """An order the bot itself placed must never be classed as a ghost."""
    from polymarket_bot.executor import Executor
    from polymarket_bot.main import est_to_plan
    from polymarket_bot.models import Candidate, Estimate
    from polymarket_bot.tests.conftest import make_market

    m = make_market(id="e1", clob_token_ids=["e1-y", "e1-n"])
    est = Estimate(candidate=Candidate(market=m, outcome_index=0,
                                       token_id="e1-y", p_mkt=0.5),
                   p_mkt=0.5, p_est=0.5, signals=[])
    trader = mock.Mock()
    trader.sell_limit.return_value = {"orderID": "resting-exit-1"}
    trader.order_status.return_value = {"status": "live", "size_matched": 0.0}

    ex = Executor(cfg, ledger, clob=mock.Mock(), trader=trader, mode="live")
    ex.execute_sell(est_to_plan(est, "other"), size=10.0, min_price=0.4,
                    known_bid=0.48)

    # The exchange now reports this order. Reconcile receives the same "local"
    # set main.risk_job builds — MM + sprint only.
    local_ids: set[str] = set()
    exchange_ids = {"resting-exit-1"}

    halts: list[str] = []
    ks = KillSwitch(cfg, ledger, "live", cancel_all=lambda: None,
                    alert=lambda msg: halts.append(msg) or True)
    ks.reconcile(local_ids, exchange_ids)

    assert not ks.halted, (
        "the bot HALTED on an order it placed itself: executor/arbitrage order "
        f"ids are absent from reconcile's local set ({halts})")


def test_executor_exposes_its_open_order_ids():
    """Structural requirement behind any fix: there must be something to pass."""
    from polymarket_bot.executor import Executor
    assert hasattr(Executor, "local_order_ids"), (
        "Executor has no local_order_ids(); reconcile cannot be made complete")


def test_basket_arb_exposes_its_open_order_ids():
    from polymarket_bot.arbitrage import ArbitrageScanner
    assert hasattr(ArbitrageScanner, "local_order_ids"), (
        "ArbitrageScanner has no local_order_ids(); its resting GTC legs are "
        "invisible to reconcile")
