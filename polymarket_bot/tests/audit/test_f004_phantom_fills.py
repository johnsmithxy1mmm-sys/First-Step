"""F-004/005/006: orders recorded as FILLED without confirming any fill.

Only two paths in the codebase verify the matched quantity: `executor.execute`
(via `_wait_fill`) and `MarketMaker._sync_live_fills`. Every other money path
records `status="filled"` for the full requested size as soon as the exchange
returns an order id — which it does for a resting order and for a killed FOK
alike.

F-004  executor.execute_sell — the EXIT path. Places a **GTC** sell and records
       the position as sold immediately. If the bid moves away the order rests
       unfilled forever while the ledger says the position is closed: phantom
       realized PnL, the position drops out of `open_positions` so take-profit
       and the guardian stop watching it, and the resting order is a ghost to
       `KillSwitch.reconcile`. Note the asymmetry — the BUY path has
       `_wait_fill` and partial handling; its mirror image has neither.

F-005  arbitrage.execute — neg-risk baskets (enabled: true by default) place
       plain **GTC** legs (no `order_type`) and record each as filled. A leg that
       does not fill leaves a partial basket: the "riskless" structure is booked
       complete while the missing leg makes it a directional bet. chainarb was
       given FOK + unwind for exactly this reason; arbitrage.py was not.

F-006  FOK sites trust the order id, not the fill. `resolution.execute` and
       `chainarb._buy_leg` treat "an order id came back" as "fully filled". A
       killed FOK still yields a response; neither checks `size_matched`.
"""

import pytest
from unittest import mock

from polymarket_bot.models import BookLevel, OrderBook


def _est(market, token, price=0.5):
    from polymarket_bot.models import Candidate, Estimate
    return Estimate(candidate=Candidate(market=market, outcome_index=0,
                                        token_id=token, p_mkt=price),
                    p_mkt=price, p_est=price, signals=[])


# --- F-004: the exit path books a sale that never happened ---

def test_execute_sell_confirms_the_fill_before_recording(cfg, ledger):
    from polymarket_bot.executor import Executor
    from polymarket_bot.main import est_to_plan
    from polymarket_bot.tests.conftest import make_market

    m = make_market(id="s1", clob_token_ids=["s1-y", "s1-n"])
    trader = mock.Mock()
    trader.buy_limit.return_value = {"orderID": "buy-1"}
    # The exchange accepts the sell and it RESTS: nothing is matched.
    trader.sell_limit.return_value = {"orderID": "sell-1"}
    trader.order_status.return_value = {"status": "live", "size_matched": 0.0}

    ex = Executor(cfg, ledger, clob=mock.Mock(), trader=trader, mode="live")
    plan = est_to_plan(_est(m, "s1-y"), "other")
    result = ex.execute_sell(plan, size=100.0, min_price=0.4, known_bid=0.48)

    assert result.status != "filled", (
        "a resting GTC sell was reported as filled: the ledger now shows the "
        "position closed while the shares are still held")


def test_unfilled_exit_does_not_erase_the_position(cfg, ledger):
    """State consequence: the position must stay visible to the guardian."""
    from polymarket_bot.executor import Executor
    from polymarket_bot.main import est_to_plan
    from polymarket_bot.models import simple_estimate
    from polymarket_bot.tests.conftest import make_market

    m = make_market(id="s2", clob_token_ids=["s2-y", "s2-n"])
    ledger.record_trade(mode="live", estimate=simple_estimate(m, 0, 0.40),
                        category="other", side="BUY", price=0.40, size=100.0,
                        order_id="b1", status="filled", strategy="longshot")
    trader = mock.Mock()
    trader.sell_limit.return_value = {"orderID": "sell-1"}
    trader.order_status.return_value = {"status": "live", "size_matched": 0.0}

    ex = Executor(cfg, ledger, clob=mock.Mock(), trader=trader, mode="live")
    ex.execute_sell(est_to_plan(_est(m, "s2-y"), "other"), size=100.0,
                    min_price=0.4, known_bid=0.48)

    still_open = [p for p in ledger.open_positions("live") if p.token_id == "s2-y"]
    assert still_open, ("the position vanished from open_positions on an UNFILLED "
                        "sell — guardian and take-profit stop watching it")


# --- F-005: a partial basket is booked as a complete one ---

def test_basket_arb_legs_are_fok_not_resting_gtc(cfg, ledger):
    """The order TYPE is the defect: a resting GTC leg cannot be atomic.

    chainarb was converted to FOK precisely so a half-built structure is
    impossible; arbitrage.py still sends the platform default (GTC).
    """
    from polymarket_bot.arbitrage import ArbitrageScanner, ArbLeg, BasketArb
    from polymarket_bot.tests.conftest import make_market

    legs = []
    for i in range(3):
        m = make_market(id=f"a{i}", event_id="ev", event_title="t",
                        clob_token_ids=[f"a{i}-y", f"a{i}-n"],
                        outcome_prices=[0.30, 0.70], neg_risk=True,
                        volume_24h_usd=50_000)
        legs.append(ArbLeg(market=m, outcome_index=0, token_id=f"a{i}-y",
                           ask=0.30, depth=9999))
    arb = BasketArb(event_id="ev", event_title="t", side="YES", legs=legs,
                    taker_coef=0.0)

    trader = mock.Mock()
    trader.buy_limit.return_value = {"orderID": "L"}
    cfg.arbitrage.execute = True
    ArbitrageScanner(cfg, ledger, mock.Mock(), trader, "live").execute(arb)

    order_types = [kw.get("order_type") for _, kw in trader.buy_limit.call_args_list]
    assert order_types and all(t == "FOK" for t in order_types), (
        f"basket legs sent as {order_types} — a resting GTC leg can stay "
        "unfilled while the ledger books the basket as complete")


def test_basket_arb_does_not_book_an_unmatched_leg_as_filled(cfg, ledger):
    from polymarket_bot.arbitrage import ArbitrageScanner, ArbLeg, BasketArb
    from polymarket_bot.tests.conftest import make_market

    legs = []
    for i in range(3):
        m = make_market(id=f"b{i}", event_id="ev2", event_title="t",
                        clob_token_ids=[f"b{i}-y", f"b{i}-n"],
                        outcome_prices=[0.30, 0.70], neg_risk=True,
                        volume_24h_usd=50_000)
        legs.append(ArbLeg(market=m, outcome_index=0, token_id=f"b{i}-y",
                           ask=0.30, depth=9999))
    arb = BasketArb(event_id="ev2", event_title="t", side="YES", legs=legs,
                    taker_coef=0.0)

    trader = mock.Mock()
    trader.buy_limit.return_value = {"orderID": "L"}
    # Nothing ever matches — every leg merely rests on the book.
    trader.order_status.return_value = {"status": "live", "size_matched": 0.0}
    cfg.arbitrage.execute = True
    ArbitrageScanner(cfg, ledger, mock.Mock(), trader, "live").execute(arb)

    assert ledger.open_positions("live") == [], (
        f"{len(ledger.open_positions('live'))} legs recorded as filled with "
        "zero matched size — phantom basket, real directional exposure")


# --- F-006: a killed FOK is recorded as a full fill ---

def test_resolution_checks_matched_size_not_just_an_order_id(cfg, ledger):
    from polymarket_bot.resolution import ResolutionAlpha
    from polymarket_bot.tests.conftest import make_market

    m = make_market(id="r1", clob_token_ids=["r1-y", "r1-n"],
                    outcome_prices=[0.96, 0.04], volume_24h_usd=50_000,
                    volume_usd=100_000)
    clob = mock.Mock()
    clob.order_book.return_value = OrderBook(
        bids=[BookLevel(price=0.95, size=500)],
        asks=[BookLevel(price=0.96, size=500)])
    trader = mock.Mock()
    # FOK killed: an id comes back, nothing matched.
    trader.buy_limit.return_value = {"orderID": "fok-1"}
    trader.order_status.return_value = {"status": "canceled", "size_matched": 0.0}

    engine = ResolutionAlpha(cfg, ledger, clob, trader, "live")
    from polymarket_bot.resolution import ResolutionCandidate
    cand = ResolutionCandidate(market=m, outcome_index=0, price=0.96, net_edge=0.02)
    engine.execute(cand)

    assert ledger.open_positions("live") == [], (
        "a killed FOK was recorded as a full fill — the ledger holds shares the "
        "account does not")
