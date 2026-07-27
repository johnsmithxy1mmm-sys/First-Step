"""Step definitions binding features/money.feature to real code.

Each scenario is an audit finding restated as a behaviour: none of them were
caught by 458 green tests, because no test drove a RESTING order, a KILLED
fill, or a POISONED number through a money path.
"""

from unittest import mock

import pytest
from pytest_bdd import given, scenarios, then, when

from polymarket_bot.models import Candidate, Estimate, simple_estimate

scenarios("features/money.feature")


class _World:
    def __init__(self, cfg, ledger):
        self.cfg, self.ledger = cfg, ledger
        self.market = None
        self.result = None
        self.write_rejected = False
        self.drift = {}


@pytest.fixture()
def world(cfg, ledger):
    cfg.executor.fill_timeout_sec = 0.05
    cfg.executor.poll_interval_sec = 0.01
    return _World(cfg, ledger)


def _market(mid="bdd1"):
    from .conftest import make_market
    return make_market(id=mid, clob_token_ids=[f"{mid}-y", f"{mid}-n"])


@given("a held position of 100 shares")
def _held(world):
    world.market = _market()
    world.ledger.record_trade(
        mode="live", estimate=simple_estimate(world.market, 0, 0.40),
        category="other", side="BUY", price=0.40, size=100.0,
        order_id="b1", status="filled", strategy="longshot")


@when("a take-profit sell is placed and never fills")
def _sell_unfilled(world):
    from polymarket_bot.executor import Executor
    from polymarket_bot.main import est_to_plan

    trader = mock.Mock()
    trader.sell_limit.return_value = {"orderID": "resting"}
    trader.order_status.return_value = {"status": "live", "size_matched": 0.0}
    trader.matched_size.return_value = 0.0
    ex = Executor(world.cfg, world.ledger, clob=mock.Mock(), trader=trader,
                  mode="live")
    est = Estimate(candidate=Candidate(market=world.market, outcome_index=0,
                                       token_id="bdd1-y", p_mkt=0.5),
                   p_mkt=0.5, p_est=0.5, signals=[])
    world.result = ex.execute_sell(est_to_plan(est, "other"), size=100.0,
                                   min_price=0.4, known_bid=0.48)


@then("the position is still open")
def _still_open(world):
    open_tokens = {p.token_id for p in world.ledger.open_positions("live")}
    assert "bdd1-y" in open_tokens


@then("no sale is recorded")
def _no_sale(world):
    assert world.result.status != "filled"


@given("a fill-or-kill buy is accepted and then killed unfilled")
def _fok_killed(world):
    world.market = _market("bdd2")
    world.trader = mock.Mock()
    world.trader.buy_limit.return_value = {"orderID": "fok"}
    world.trader.matched_size.return_value = 0.0


@when("the strategy records its result")
def _record_fok(world):
    from polymarket_bot.resolution import ResolutionAlpha, ResolutionCandidate
    from polymarket_bot.models import BookLevel, OrderBook

    clob = mock.Mock()
    clob.order_book.return_value = OrderBook(
        bids=[BookLevel(price=0.95, size=500)],
        asks=[BookLevel(price=0.96, size=500)])
    engine = ResolutionAlpha(world.cfg, world.ledger, clob, world.trader, "live")
    m = world.market
    m.outcome_prices = [0.96, 0.04]
    engine.execute(ResolutionCandidate(market=m, outcome_index=0, price=0.96,
                                       net_edge=0.02))


@then("no trade is recorded")
def _nothing_recorded(world):
    assert world.ledger.open_positions("live") == []


@when("a trade is recorded with a non-finite size")
def _poison_write(world):
    from polymarket_bot.ledger import InvalidTrade
    world.market = _market("bdd3")
    try:
        world.ledger.record_trade(
            mode="paper", estimate=simple_estimate(world.market, 0, 0.5),
            category="mm", side="BUY", price=0.5, size=float("inf"),
            order_id=None, status="filled", strategy="mm")
    except InvalidTrade:
        world.write_rejected = True


@then("the write is rejected")
def _write_rejected(world):
    assert world.write_rejected


@then("exposure remains a finite number")
def _finite_exposure(world):
    import math
    assert math.isfinite(world.ledger.total_exposure("paper"))


@given("a sale of 10 shares with no prior purchase")
def _oversold(world):
    world.market = _market("bdd4")
    world.ledger.record_trade(
        mode="paper", estimate=simple_estimate(world.market, 0, 0.5),
        category="mm", side="SELL", price=0.5, size=10.0,
        order_id=None, status="filled", strategy="mm")


@when("accounting drift is checked")
def _check_drift(world):
    world.drift = world.ledger.accounting_drift("paper")


@then("the drift is reported for that token")
def _drift_reported(world):
    assert world.drift.get("bdd4-y", 0) > 0
