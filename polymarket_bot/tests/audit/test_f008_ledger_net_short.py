"""F-008: the ledger accepts selling what was never owned, and hides it.

Minimal counterexample found by Hypothesis (INV-24): a single SELL with no prior
BUY is written without complaint, and `open_positions` floors the result at 0
(`slot["size"] = max(slot["size"] - r["size"], 0.0)`).

Nothing is logged, nothing is flagged. The floor is what makes it dangerous: it
converts an accounting impossibility into a plausible-looking zero, so a
duplicated exit, a double-recorded partial, or an over-sized sell disappears
instead of surfacing. Combined with F-004 (exits booked as filled without
confirmation) this is precisely the state that would hide a real desync.
"""

import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polymarket_bot.models import simple_estimate

_price = st.floats(min_value=0.01, max_value=0.99,
                   allow_nan=False, allow_infinity=False)
_size = st.floats(min_value=1.0, max_value=1000.0,
                  allow_nan=False, allow_infinity=False)


def test_selling_what_was_never_bought_is_rejected_or_flagged(cfg, ledger):
    """The shrunk counterexample, pinned explicitly."""
    from polymarket_bot.tests.conftest import make_market
    m = make_market(id="ns", clob_token_ids=["ns-y", "ns-n"])
    ledger.record_trade(mode="paper", estimate=simple_estimate(m, 0, 0.5),
                        category="mm", side="SELL", price=0.5, size=1.0,
                        order_id=None, status="filled", strategy="mm")
    # open_positions still floors to 0 (a legitimate exit must never be refused
    # at write time), but the drift is now explicitly detectable instead of being
    # rendered as an ordinary zero — and risk_job HALTs on it.
    drift = ledger.accounting_drift("paper")
    assert drift and drift["ns-y"] == pytest.approx(1.0), (
        "an oversold token is still invisible: the impossible state is being "
        "rendered as a normal zero")


@given(trades=st.lists(st.tuples(st.sampled_from(["BUY", "SELL"]), _price, _size),
                       min_size=1, max_size=8))
@settings(max_examples=150, deadline=None)
def test_property_never_net_short(trades, tmp_path_factory):
    """Each example needs a FRESH ledger: a function-scoped fixture is reused
    across Hypothesis examples, so DB state would accumulate while the local
    counters reset — the test would then measure the wrong thing."""
    from polymarket_bot.config import BotConfig
    from polymarket_bot.ledger import Ledger
    from polymarket_bot.tests.conftest import make_market

    BotConfig()          # keep the import path exercised
    db = tmp_path_factory.mktemp("drift") / "l.sqlite"
    ledger = Ledger(db)
    try:
        m = make_market(id="ns2", clob_token_ids=["ns2-y", "ns2-n"])
        bought = sold = 0.0
        for side, price, size in trades:
            ledger.record_trade(mode="paper", estimate=simple_estimate(m, 0, price),
                                category="mm", side=side, price=price, size=size,
                                order_id=None, status="filled", strategy="mm")
            if side == "BUY":
                bought += size
            else:
                sold += size

        drift = ledger.accounting_drift("paper")
        # The contract is DETECTION; the exact magnitude is not asserted because
        # SQLite's SUM and Python's accumulation differ in float summation order.
        if sold > bought + 1.0:
            assert drift, f"undetected drift: bought {bought:.1f}, sold {sold:.1f}"
            assert drift["ns2-y"] > 0
        assert math.isfinite(ledger.realized_pnl("paper"))
    finally:
        ledger.close()
