"""Step definitions binding features/safety.feature to real code.

Deliberately asserts EFFECTS via the public surface (`trading_allowed`,
`check_global_exposure`) rather than private flags, so these scenarios keep
their value under refactoring — and so a mutant that removes the pause term
from `trading_allowed` cannot survive them.
"""


import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from polymarket_bot.risk import KillSwitch

scenarios("features/safety.feature")


class _World:
    def __init__(self, cfg, ledger):
        self.cancels: list[int] = []
        self.ks = KillSwitch(cfg, ledger, "paper",
                             cancel_all=lambda: self.cancels.append(1),
                             alert=lambda m: True)
        self.cfg = cfg
        self.ledger = ledger


@pytest.fixture()
def world(cfg, ledger):
    cfg.risk.max_global_exposure_usd = 300.0
    cfg.risk.max_daily_loss_usd = 25.0
    return _World(cfg, ledger)


@given(parsers.parse("a bot with a ${cap:d} global exposure cap and a "
                     "${loss:d} daily loss limit"))
def _configured(world, cap, loss):
    world.cfg.risk.max_global_exposure_usd = float(cap)
    world.cfg.risk.max_daily_loss_usd = float(loss)


@given("the websocket stream has been silent past the staleness threshold")
@when("the websocket stream has been silent past the staleness threshold")
def _ws_dead(world):
    world.ks.on_ws_disconnect(world.cfg.risk.ws_staleness_kill_sec + 1)


@given("a corrupt price feed was detected")
def _data_anomaly(world):
    world.ks.trip_pause("corrupt feed", source="data")


@when("the stream recovers")
def _ws_back(world):
    world.ks.on_ws_recovered()


@when(parsers.parse("equity has fallen by {amount:d} dollars since the start "
                    "of the day"))
def _daily_loss(world, amount):
    start = 1000.0
    world.ledger.snapshot_bank(cash=start, exposure=0.0)
    world.ks.check_daily_loss(start - amount)


@when("the ledger reports a non-finite equity")
def _nan_equity(world):
    world.ks.check_daily_loss(float("nan"))


@when("open exposure reaches the global cap")
def _at_cap(world):
    world.at_cap = world.ks.check_global_exposure(
        world.cfg.risk.max_global_exposure_usd)


@then("no new orders are allowed")
def _no_trading(world):
    assert world.ks.trading_allowed is False


@then("new orders are allowed again")
def _trading_ok(world):
    assert world.ks.trading_allowed is True


@then("all resting quotes have been cancelled")
def _cancelled(world):
    assert world.cancels, "the kill-switch did not bulk-cancel"


@then("the halt is not cleared by a pause resume")
def _halt_sticky(world):
    world.ks.resume_from_pause("ws")
    assert world.ks.halted and world.ks.trading_allowed is False


@then("new entries are refused")
def _entries_refused(world):
    assert world.at_cap is True


@then("reducing an existing position is still allowed")
def _exits_allowed(world):
    """Exits are gated by `trading_allowed`, never by the exposure cap:
    refusing to reduce risk because you are over the limit is backwards."""
    assert world.ks.trading_allowed is True
