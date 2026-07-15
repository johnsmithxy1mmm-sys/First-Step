"""Risk 2.0: stress/VaR, market-data guard, per-strategy circuit breaker."""

from polymarket_bot.models import Position
from polymarket_bot.risk2 import (MarketDataGuard, StrategyCircuitBreaker,
                                   parametric_var, portfolio_stress)

from .conftest import make_market


def pos(cost, cat="other", event="", neg=False, token="t"):
    return Position(token_id=token, market_id="m", question="Q?", outcome="No",
                    category=cat, size=cost, avg_price=1.0, event_id=event, neg_risk=neg)


def test_portfolio_stress_nets_events():
    ps = [pos(100, event="e", neg=True, token="a"),
          pos(120, event="e", neg=True, token="b"),   # same neg-risk event
          pos(80, cat="crypto", token="c")]
    st = portfolio_stress(ps)
    assert st["gross_usd"] == 300.0
    assert st["worst_case_usd"] == 120.0 + 80.0       # event nets to 120
    assert st["largest_event_usd"] == 120.0
    assert st["var95_usd"] > 0


def test_var_rises_with_correlated_exposure():
    corr = [pos(100, cat="geopolitics", token="a"), pos(100, cat="economy", token="b")]
    uncorr = [pos(100, cat="sports", token="a"), pos(100, cat="crypto", token="b")]
    # geopolitics<->economy (0.5) is more correlated than sports<->crypto (0.1).
    assert parametric_var(corr) > parametric_var(uncorr)


def test_data_guard_flags_out_of_range():
    g = MarketDataGuard()
    bad = make_market(outcome_prices=[1.5, -0.5])
    assert g.severe([bad])
    assert "out-of-range" in g.problems([bad])[0]


def test_data_guard_flags_mass_desync():
    g = MarketDataGuard()
    good = [make_market(id=f"g{i}", outcome_prices=[0.5, 0.5]) for i in range(10)]
    assert not g.severe(good)
    desynced = [make_market(id=f"d{i}", outcome_prices=[0.5, 0.9]) for i in range(10)]
    assert g.severe(desynced)


def test_circuit_breaker_trips_on_losing_streak():
    b = StrategyCircuitBreaker(losing_streak=3)
    for pnl in (0.0, -5.0, -12.0, -20.0):      # 3 consecutive drops
        b.update({"fade": pnl})
    assert not b.allows("fade")


def test_circuit_breaker_recovers():
    b = StrategyCircuitBreaker(losing_streak=2)
    for pnl in (0.0, -5.0, -12.0):
        b.update({"mm": pnl})
    assert not b.allows("mm")
    for pnl in (-8.0, -3.0):                    # two consecutive gains
        b.update({"mm": pnl})
    assert b.allows("mm")


def test_circuit_breaker_independent_strategies():
    b = StrategyCircuitBreaker(losing_streak=2)
    for i in range(3):
        b.update({"fade": -float(i) * 10, "mm": float(i) * 10})
    assert not b.allows("fade") and b.allows("mm")
