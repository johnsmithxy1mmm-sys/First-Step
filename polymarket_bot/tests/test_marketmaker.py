"""MM core: microprice, skew, hysteresis, rewards band, guard, paper fills."""

from unittest import mock

import pytest

from polymarket_bot.marketmaker import MarketMaker
from polymarket_bot.models import simple_estimate
from polymarket_bot.ws_feed import TopOfBook

from .conftest import make_market


def mm_market(**overrides):
    defaults = dict(
        id="mm1", question="Will the Democrats win the Senate midterms?",
        outcome_prices=[0.45, 0.55],
        clob_token_ids=["mm1-yes", "mm1-no"],
        volume_24h_usd=100_000, volume_usd=5_000_000,
        rewards_min_size=20.0, rewards_max_spread=0.03,
        end_date=None,
    )
    defaults.update(overrides)
    from datetime import datetime, timedelta, timezone
    if defaults.get("end_date") is None:
        defaults["end_date"] = datetime.now(timezone.utc) + timedelta(days=90)
    return make_market(**defaults)


def top(bid=0.43, ask=0.47, bid_size=1000.0, ask_size=1000.0) -> TopOfBook:
    return TopOfBook(bid=bid, bid_size=bid_size, ask=ask, ask_size=ask_size, ts=1.0)


def make_mm(cfg, ledger, mode="dry-run", tops=None) -> MarketMaker:
    cfg.market_maker.enabled = True
    clob = mock.Mock()
    clob.order_book.return_value = None
    source = (lambda t: tops.get(t)) if tops is not None else None
    return MarketMaker(cfg, ledger, clob, None, mode, top_source=source)


# --- fair value and quotes ---

def test_microprice_weighs_by_sizes():
    t = top(bid=0.40, ask=0.50, bid_size=3000, ask_size=1000)
    # A heavy bid pulls the microprice up: (0.40*1000 + 0.50*3000) / 4000 = 0.475
    assert t.microprice == pytest.approx(0.475)


def test_quote_symmetric_within_rewards_band(cfg, ledger):
    mm = make_mm(cfg, ledger)
    m = mm_market()
    quote = mm.compute_quote(m, top())  # microprice = mid = 0.45
    assert quote is not None
    assert quote.yes_bid == pytest.approx(0.44)
    assert quote.implied_yes_ask == pytest.approx(0.46)
    # Half-spread inside the reward band (0.03 * 0.9).
    assert quote.captured_spread / 2 <= m.rewards_max_spread * 0.9 + 1e-9
    # Size not below rewards_min_size (otherwise it will not count).
    assert quote.size >= m.rewards_min_size


def test_quote_respects_fee_breakeven(cfg, ledger):
    # Category geopolitics: taker fee 0 -> rebate 0 -> half-spread >= min_edge/2.
    cfg.risk.min_edge_after_fees = 0.02
    cfg.market_maker.half_spread = 0.001
    mm = make_mm(cfg, ledger)
    m = mm_market(question="Will NATO invoke Article 5 over the invasion?")
    quote = mm.compute_quote(m, top())
    assert quote is not None
    assert quote.captured_spread >= 0.02 - 1e-9


def test_inventory_skew_shifts_both_quotes_down(cfg, ledger):
    mm = make_mm(cfg, ledger)
    m = mm_market()
    base = mm.compute_quote(m, top())
    # Accumulated long Yes up to the limit -> fair shifts down.
    ledger.record_trade(mode="dry-run", estimate=simple_estimate(m, 0, 0.45),
                        category="mm", side="BUY",
                        price=0.45, size=cfg.risk.max_position_per_market_usd / 0.45,
                        order_id=None, status="filled", strategy="mm")
    skewed = mm.compute_quote(m, top())
    assert skewed is not None
    assert skewed.yes_bid < base.yes_bid          # bid lower
    assert skewed.implied_yes_ask < base.implied_yes_ask  # ask more aggressive


def test_requote_hysteresis(cfg, ledger):
    cfg.market_maker.requote_timer_sec = 9999
    mm = make_mm(cfg, ledger)
    m = mm_market()
    assert mm.needs_requote(m, top())             # no quote yet
    quote = mm.compute_quote(m, top())
    mm._quotes[m.id] = quote
    # Fair moved less than 2 ticks — do NOT reprice (hysteresis).
    assert not mm.needs_requote(m, top(bid=0.431, ask=0.471))
    # Move exceeds threshold — reprice.
    assert mm.needs_requote(m, top(bid=0.45, ask=0.49))


def test_sides_disabled_at_position_cap(cfg, ledger):
    mm = make_mm(cfg, ledger)
    m = mm_market()
    assert mm.sides_allowed(m) == (True, True)
    ledger.record_trade(mode="dry-run", estimate=simple_estimate(m, 0, 0.45),
                        category="mm", side="BUY",
                        price=0.45, size=(cfg.risk.max_position_per_market_usd + 10) / 0.45,
                        order_id=None, status="filled", strategy="mm")
    quote_yes, quote_no = mm.sides_allowed(m)
    assert not quote_yes and quote_no


def test_guard_on_midpoint_jump(cfg, ledger):
    import time as _time
    mm = make_mm(cfg, ledger)
    m = mm_market()
    assert not mm.guard_blocks(m, top(bid=0.43, ask=0.47))
    assert mm.guard_blocks(m, top(bid=0.48, ask=0.52))     # jump >= 0.03
    # Cooldown is wall-clock (cycles * interval_sec): call frequency can't burn it.
    for _ in range(50):
        assert mm.guard_blocks(m, top(bid=0.48, ask=0.52))
    mm._cooldown_until[m.id] = _time.time() - 1            # cooldown expired
    assert not mm.guard_blocks(m, top(bid=0.48, ask=0.52))


def test_guard_sees_cumulative_tick_moves(cfg, ledger):
    """Many small WS ticks adding up to a shock must still trip the guard."""
    mm = make_mm(cfg, ledger)
    m = mm_market()
    assert not mm.guard_blocks(m, top(bid=0.43, ask=0.47))          # anchor at 0.45
    blocked = False
    for i in range(1, 11):                                          # +0.004/tick
        t = top(bid=0.43 + 0.004 * i, ask=0.47 + 0.004 * i)
        blocked = mm.guard_blocks(m, t, anchor=False) or blocked    # per-tick calls
    assert blocked                                                  # cumulative +0.04 seen


def test_extreme_midpoint_requires_two_sided_or_exit(cfg, ledger):
    tops = {"mm1-yes": top(bid=0.94, ask=0.96), "mm1-no": top(bid=0.04, ask=0.06)}
    mm = make_mm(cfg, ledger, tops=tops)
    m = mm_market(outcome_prices=[0.95, 0.05], one_day_price_change=0.0)
    # Skewed inventory: one side allowed -> at mid>0.90 leave the market.
    ledger.record_trade(mode="dry-run", estimate=simple_estimate(m, 0, 0.95),
                        category="mm", side="BUY",
                        price=0.95, size=(cfg.risk.max_position_per_market_usd + 10) / 0.95,
                        order_id=None, status="filled", strategy="mm")
    with mock.patch.object(mm._scorer, "top_markets", return_value=[m]), \
         mock.patch.object(mm._scorer, "eligible", return_value=True):
        quotes = mm.cycle([m])
    assert quotes == []
    assert m.id not in mm._quotes


# --- paper mode ---

def test_paper_fill_when_market_trades_through(cfg, ledger):
    tops = {"mm1-yes": top(bid=0.43, ask=0.47), "mm1-no": top(bid=0.53, ask=0.57)}
    mm = make_mm(cfg, ledger, mode="paper", tops=tops)
    m = mm_market()
    quote = mm.compute_quote(m, tops["mm1-yes"])
    mm._quotes[m.id] = quote

    mm._paper_fills()                             # ask 0.47 > bid 0.44 — no fill
    assert ledger.open_positions("paper") == []

    tops["mm1-yes"] = top(bid=0.42, ask=0.44)     # market traded into our bid
    with mock.patch("polymarket_bot.marketmaker.alert") as a:
        mm._paper_fills()
    positions = ledger.open_positions("paper")
    assert len(positions) == 1
    assert positions[0].avg_price == pytest.approx(quote.yes_bid)
    a.assert_called_once()                        # fill is announced to Telegram
    assert "MM fill" in a.call_args[0][0]


def test_react_to_tick_reprices_only_quoted_markets(cfg, ledger):
    tops = {"mm1-yes": top(bid=0.43, ask=0.47), "mm1-no": top(bid=0.53, ask=0.57)}
    mm = make_mm(cfg, ledger, tops=tops)
    m = mm_market()
    with mock.patch.object(mm._scorer, "top_markets", return_value=[m]), \
         mock.patch.object(mm._scorer, "eligible", return_value=True):
        mm.cycle([m])
    assert "mm1-yes" in mm._quoted            # cycle registered the quoted market
    # A moderate move (past the requote threshold, under the guard) reprices;
    # an unknown token does nothing.
    moved = top(bid=0.44, ask=0.48)
    assert mm.react_to_tick("mm1-yes", moved) is True
    assert mm.react_to_tick("unknown-token", moved) is False


def test_dry_run_never_places_orders(cfg, ledger):
    tops = {"mm1-yes": top(), "mm1-no": top(bid=0.53, ask=0.57)}
    mm = make_mm(cfg, ledger, tops=tops)
    m = mm_market()
    with mock.patch.object(mm._scorer, "top_markets", return_value=[m]), \
         mock.patch.object(mm._scorer, "eligible", return_value=True):
        quotes = mm.cycle([m])
    assert len(quotes) == 1
    assert mm._orders == {}                       # intentions logged, no orders


# --- fill-calibration training data: quote outcomes are labeled ---

def test_paper_fill_records_filled_quote_outcome(cfg, ledger):
    tops = {"mm1-yes": top(bid=0.43, ask=0.47), "mm1-no": top(bid=0.53, ask=0.57)}
    mm = make_mm(cfg, ledger, mode="paper", tops=tops)
    m = mm_market()
    quote = mm.compute_quote(m, tops["mm1-yes"])
    assert quote.p_fill_pred >= 0                  # a prediction was attached
    mm._quotes[m.id] = quote
    tops["mm1-yes"] = top(bid=0.42, ask=0.44)      # trade through -> fill
    with mock.patch("polymarket_bot.marketmaker.alert"):
        mm._paper_fills()
    outcomes = ledger.quote_outcomes("paper")
    assert len(outcomes) == 1 and outcomes[0][1] is True    # labeled FILLED


def test_cancel_records_unfilled_quote_outcome(cfg, ledger):
    tops = {"mm1-yes": top(), "mm1-no": top(bid=0.53, ask=0.57)}
    mm = make_mm(cfg, ledger, tops=tops)
    m = mm_market()
    quote = mm.compute_quote(m, tops["mm1-yes"])
    mm._quotes[m.id] = quote
    mm._cancel_market(m.id)                         # died unfilled
    outcomes = ledger.quote_outcomes("dry-run")
    assert len(outcomes) == 1 and outcomes[0][1] is False   # labeled NOT filled


def test_size_factor_scales_quote_budget(cfg, ledger):
    tops = {"mm1-yes": top(), "mm1-no": top(bid=0.53, ask=0.57)}
    mm = make_mm(cfg, ledger, tops=tops)
    m = mm_market()
    base = mm.compute_quote(m, tops["mm1-yes"]).size
    mm.size_factor = 2.0
    scaled = mm.compute_quote(m, tops["mm1-yes"]).size
    assert scaled > base                           # allocator tilt reaches sizing


# --- fee break-even floor in a FEE-PAYING category (the case the old model missed) ---

def test_fee_floor_active_in_a_fee_paying_category(cfg, ledger):
    """The old rebate (a flat 35% of theta) drove min_half to 0.0 for politics at
    EVERY price — the break-even floor was silently disabled and a too-tight
    configured spread passed straight through. With the correct per-category
    rebate priced at the quote, the floor is real and must lift the spread.
    """
    from polymarket_bot.fees import FeeModel
    from polymarket_bot.portfolio import classify_category

    cfg.risk.min_edge_after_fees = 0.01
    cfg.market_maker.half_spread = 0.001          # deliberately below the floor
    mm = make_mm(cfg, ledger)
    m = mm_market()                               # elections, theta 0.04
    fees = FeeModel(cfg.fees)
    cat = classify_category(m.question, m.category)

    quote = mm.compute_quote(m, top())            # microprice 0.45
    assert quote is not None
    floor = fees.mm_min_half_spread(cat, 0.45, m.category, 0.01)
    assert floor > 0                              # the old model gave exactly 0
    assert quote.captured_spread >= 2 * floor - 1e-9
    # Strictly wider than the configured/tick spread -> the floor really bound.
    assert quote.captured_spread > 2 * cfg.market_maker.half_spread


def test_fee_floor_rises_toward_the_dollar(cfg, ledger):
    """Rebate ~ theta*p*(1-p) collapses near $1, so the spread must carry more
    of the edge itself. A price-independent floor could not express this.

    Compared at 0.45 vs 0.95, where the floor gap (0.25c -> 0.45c) exceeds the
    0.001 tick; nearer prices differ by less than one tick and quantize equal.
    """
    cfg.risk.min_edge_after_fees = 0.01
    cfg.market_maker.half_spread = 0.001
    mm = make_mm(cfg, ledger)

    mid = mm.compute_quote(mm_market(), top(bid=0.43, ask=0.47))
    high = mm.compute_quote(mm_market(id="mm2", outcome_prices=[0.95, 0.05]),
                            top(bid=0.94, ask=0.96))
    assert mid is not None and high is not None
    assert high.captured_spread > mid.captured_spread
