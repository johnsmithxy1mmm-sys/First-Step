"""Step definitions binding features/shape.feature to real code.

The scenarios assert EFFECTS — a refusal, a sale, an unallocated dollar — rather
than the presence of a config key, so they keep holding if the thresholds are
retuned and stop holding if the rules are bypassed.
"""

from datetime import datetime, timedelta, timezone
from unittest import mock

import pytest
from pytest_bdd import given, scenarios, then, when

from polymarket_bot.fade import fade_exit_plan
from polymarket_bot.models import Position, simple_estimate
from polymarket_bot.portfolio import Portfolio

from .conftest import make_market
from .test_fade import make_estimate, make_fade

scenarios("features/shape.feature")


class _World:
    def __init__(self, cfg, ledger):
        self.cfg, self.ledger = cfg, ledger
        self.fade = None
        self.estimate = None
        self.reason = "unset"
        self.position = None
        self.exit = None
        self.size = "unset"


@pytest.fixture()
def world(cfg, ledger):
    return _World(cfg, ledger)


def _in_days(days: float):
    return datetime.now(timezone.utc) + timedelta(days=days)


# --- entry gates ---

@given("a YES tail priced at 1 cent")
def _tail_1c(world):
    world.fade, _ = make_fade(world.cfg, world.ledger)
    world.estimate = make_estimate(p_mkt=0.01, p_est=None)


@given("a YES tail priced at 5 cents resolving in 10 days")
def _tail_5c_near(world):
    world.fade, _ = make_fade(world.cfg, world.ledger)
    world.estimate = make_estimate(p_mkt=0.05, end_date=_in_days(10))


@given("a YES tail priced at 5 cents resolving in 80 days")
def _tail_5c_far(world):
    world.fade, _ = make_fade(world.cfg, world.ledger)
    world.cfg.fade.max_days_to_resolution = 90.0    # horizon gate must not fire
    world.estimate = make_estimate(p_mkt=0.05, end_date=_in_days(80))


@when("the fade evaluates it")
def _evaluate(world):
    world.reason = world.fade.reject_reason(world.estimate)
    world.plan = world.fade.plan(world.estimate)


@then("the tail is refused for its payoff shape")
def _refused_shape(world):
    assert world.reason is not None and "payoff ratio" in world.reason, world.reason


@then("the tail is refused for its return per day")
def _refused_irr(world):
    assert world.reason is not None and "edge per day" in world.reason, world.reason


@then("the tail is accepted")
def _accepted(world):
    assert world.reason is None, world.reason
    assert world.plan is not None


@then("no position is opened")
def _no_position(world):
    assert world.plan is None
    with mock.patch("polymarket_bot.fade.alert"):
        assert world.fade.cycle([world.estimate]) == 0
    assert world.ledger.open_positions("dry-run") == []


# --- exits ---

@given("a fade leg bought at 0.976")
def _fade_leg(world):
    world.position = Position(
        token_id="bdd-no", market_id="bdd", question="Will X happen?",
        outcome="No", category="other", size=128.0, avg_price=0.976,
        strategy="fade")


@when("the price rises as far as the venue allows")
def _price_to_ceiling(world):
    portfolio = Portfolio(world.cfg, world.ledger, "paper")
    world.generic = [portfolio.exit_plan(world.position, m)
                     for m in (0.98, 0.99, 0.995, 0.999)]


@then("the generic take-profit still does not fire")
def _generic_silent(world):
    assert world.generic == [None, None, None, None], world.generic


@when("the implied tail probability triples")
def _tail_triples(world):
    world.mark = 0.92                                  # tail 2.4% -> 8%
    world.exit = fade_exit_plan(world.position, world.mark, world.cfg.fade)


@when("the price reaches 0.995")
def _price_995(world):
    world.mark = 0.995
    world.exit = fade_exit_plan(world.position, world.mark, world.cfg.fade)


@then("the leg is sold")
def _leg_sold(world):
    assert world.exit is not None
    assert world.exit.size > 0
    # And the wiring actually reaches the executor, not just the rule.
    from .test_hardening import make_bot
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        import pathlib
        bot = make_bot(pathlib.Path(tmp))
        try:
            bot.executor = mock.Mock()
            bot.executor.execute_sell.return_value = mock.Mock(
                status="filled", avg_price=world.mark,
                filled_size=world.exit.size)
            assert bot._exit_one(world.position, world.mark) is True
        finally:
            bot.close()


@then("the realised loss is a fraction of the notional")
def _loss_bounded(world):
    stopped = (world.position.avg_price - world.mark) * world.position.size
    held = world.position.avg_price * world.position.size
    assert 0 < stopped < held / 10, (stopped, held)


# --- capital reservation ---

@given("the fade already holds its full reserve-adjusted budget")
def _reserve_full(world):
    c = world.cfg.portfolio
    c.max_total_exposure_pct = 0.30      # $1,500 of a $5,000 bankroll
    c.reserve_for_mm_pct = 0.15          # directional cap therefore $750
    c.max_category_pct = 1.0             # keep the category cap out of the way
    m = make_market(id="bdd-r", clob_token_ids=["r-y", "r-n"])
    world.ledger.record_trade(
        mode="paper", estimate=simple_estimate(m, 0, 0.50), category="other",
        side="BUY", price=0.50, size=1_520.0, order_id="r1",
        status="filled", strategy="fade")


@when("a further directional trade is sized")
def _size_more(world):
    portfolio = Portfolio(world.cfg, world.ledger, "paper")
    world.size = portfolio.size_usd("other", 0.965, 0.95)


@then("no capital is allocated")
def _nothing_allocated(world):
    assert world.size is None


@then("account-wide room is still available")
def _global_room_left(world):
    c = world.cfg.portfolio
    used = world.ledger.total_exposure("paper")
    assert used < c.max_total_exposure_pct * c.bankroll_usd, (
        "the global cap was the binding constraint, so this scenario would pass "
        "even without a reserve")
