"""Spoof guard: painted-book detection and arb execution refusal."""

from unittest import mock

from polymarket_bot.models import BookLevel, OrderBook
from polymarket_bot.spoofguard import screen_ask


def book(asks, bids):
    return OrderBook(bids=[BookLevel(price=p, size=s) for p, s in bids],
                     asks=[BookLevel(price=p, size=s) for p, s in asks])


def test_healthy_book_passes():
    b = book(asks=[(0.30, 100), (0.31, 120), (0.32, 90)],
             bids=[(0.29, 100), (0.28, 110)])
    assert screen_ask(b).suspicious is False


def test_wall_with_nothing_behind_is_flagged():
    b = book(asks=[(0.30, 5000), (0.31, 5), (0.32, 5)],   # huge touch, empty behind
             bids=[(0.29, 100)])
    v = screen_ask(b)
    assert v.suspicious and any("behind" in r for r in v.reasons)


def test_one_sided_cliff_is_flagged():
    b = book(asks=[(0.30, 6000), (0.31, 4000)],           # ask dwarfs the bid touch
             bids=[(0.29, 50)])
    v = screen_ask(b)
    assert v.suspicious and any("cliff" in r for r in v.reasons)


def test_transient_wall_vs_recent_average():
    b = book(asks=[(0.30, 4000), (0.31, 3000)], bids=[(0.29, 1000)])
    v = screen_ask(b, recent_avg_touch=200.0)             # touch is 20x its average
    assert v.suspicious and any("transient" in r for r in v.reasons)


def test_empty_or_missing_ask_is_suspicious():
    assert screen_ask(book(asks=[], bids=[(0.29, 100)])).suspicious


# --- arb execution refuses a painted leg ---

def make_scanner(cfg, ledger, books, trader):
    from polymarket_bot.arbitrage import ArbitrageScanner
    clob = mock.Mock()
    clob.order_book.side_effect = lambda t: books.get(t)
    return ArbitrageScanner(cfg, ledger, clob, trader, "live")


def test_arb_execute_skips_painted_book(cfg, ledger):
    from polymarket_bot.arbitrage import ArbLeg, BasketArb
    from .conftest import make_market
    cfg.arbitrage.execute = True
    m = make_market(min_order_size=1.0, tick_size=0.001)
    legs = [ArbLeg(market=m, outcome_index=0, token_id="a", ask=0.30, depth=100),
            ArbLeg(market=m, outcome_index=0, token_id="b", ask=0.30, depth=100)]
    arb = BasketArb(event_id="e", event_title="t", side="YES", legs=legs, taker_fee=0.0)
    painted = book(asks=[(0.30, 5000), (0.31, 5)], bids=[(0.29, 50)])
    trader = mock.Mock()
    scanner = make_scanner(cfg, ledger, {"a": painted, "b": painted}, trader)
    assert scanner.execute(arb) == 0.0
    trader.buy_limit.assert_not_called()             # nothing placed into the trap


def test_arb_execute_proceeds_on_clean_book(cfg, ledger):
    from polymarket_bot.arbitrage import ArbLeg, BasketArb
    from .conftest import make_market
    cfg.arbitrage.execute = True
    cfg.arbitrage.max_stake_usd = 100
    m = make_market(min_order_size=1.0, tick_size=0.001)
    legs = [ArbLeg(market=m, outcome_index=0, token_id="a", ask=0.30, depth=100),
            ArbLeg(market=m, outcome_index=1, token_id="b", ask=0.30, depth=100)]
    arb = BasketArb(event_id="e", event_title="t", side="YES", legs=legs, taker_fee=0.0)
    clean = book(asks=[(0.30, 100), (0.31, 120), (0.32, 90)], bids=[(0.29, 100)])
    trader = mock.Mock()
    trader.buy_limit.return_value = {"orderID": "x"}
    scanner = make_scanner(cfg, ledger, {"a": clean, "b": clean}, trader)
    assert scanner.execute(arb) > 0.0
    assert trader.buy_limit.called


def test_spoof_screen_can_be_disabled(cfg, ledger):
    from polymarket_bot.arbitrage import ArbLeg, BasketArb
    from .conftest import make_market
    cfg.arbitrage.execute = True
    cfg.arbitrage.spoof_screen = False               # opt out
    cfg.arbitrage.max_stake_usd = 100
    m = make_market(min_order_size=1.0, tick_size=0.001)
    legs = [ArbLeg(market=m, outcome_index=0, token_id="a", ask=0.30, depth=100),
            ArbLeg(market=m, outcome_index=1, token_id="b", ask=0.30, depth=100)]
    arb = BasketArb(event_id="e", event_title="t", side="YES", legs=legs, taker_fee=0.0)
    painted = book(asks=[(0.30, 5000), (0.31, 5)], bids=[(0.29, 50)])
    trader = mock.Mock()
    trader.buy_limit.return_value = {"orderID": "x"}
    scanner = make_scanner(cfg, ledger, {"a": painted, "b": painted}, trader)
    assert scanner.execute(arb) > 0.0                # screen off -> proceeds
