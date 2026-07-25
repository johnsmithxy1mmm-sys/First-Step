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


# --- WS recv thread must never block on the MM lock ---

def test_react_to_tick_never_blocks_on_the_cycle_lock(cfg, ledger):
    """react_to_tick runs on the websocket recv thread. cycle() holds the same
    lock across order placement, so a blocking acquire there stalls the recv
    loop, stops the last-message clock and trips ws_staleness_kill_sec — the bot
    would kill its own quoting. A skipped reprice is the correct trade.
    """
    import threading
    import time

    tops = {"mm1-yes": top(bid=0.43, ask=0.47), "mm1-no": top(bid=0.53, ask=0.57)}
    mm = make_mm(cfg, ledger, tops=tops)
    m = mm_market()
    mm._quoted = {"mm1-yes": m}

    mm._lock.acquire()                     # stand in for a long-running cycle()
    try:
        done = threading.Event()
        result = []

        def ws_thread():
            result.append(mm.react_to_tick("mm1-yes", top()))
            done.set()

        threading.Thread(target=ws_thread, daemon=True).start()
        # Must return immediately, not wait for the lock.
        assert done.wait(timeout=2.0), "react_to_tick blocked on the MM lock"
        assert result == [False]
    finally:
        mm._lock.release()

    # With the lock free it works normally again.
    assert mm.react_to_tick("mm1-yes", top()) is not None


def test_cycle_fetches_books_before_taking_the_lock(cfg, ledger):
    """Book fetches are slow network reads; held under the lock they starve the
    WS fastlane. Assert the lock is free while order_book is being called."""
    from unittest import mock

    mm = make_mm(cfg, ledger)
    m = mm_market()
    seen = []

    def slow_book(token):
        # If cycle() took the lock first, this would observe it held.
        seen.append(mm._lock.acquire(blocking=False))
        if seen[-1]:
            mm._lock.release()
        return None

    mm._clob.order_book = slow_book
    with mock.patch.object(mm._scorer, "eligible", return_value=True), \
         mock.patch.object(mm._scorer, "top_markets", return_value=[]):
        mm.cycle([m])
    assert seen and all(seen), "order_book was called while holding the MM lock"


# --- rewards-aware spread placement ---

def test_never_quotes_tighter_than_the_adverse_selection_floor(cfg, ledger):
    """The old cap at 0.9 * rewards_max_spread could force `half` BELOW
    base_half — the learned-markout / volatility floor — so a narrow reward band
    silently overrode our own toxicity protection. Now such a market is refused.
    """
    cfg.market_maker.half_spread = 0.02          # our floor, before any widening
    cfg.risk.min_edge_after_fees = 0.0           # isolate: no fee floor in play
    mm = make_mm(cfg, ledger)
    # Reward band far narrower than the floor we need.
    m = mm_market(rewards_max_spread=0.004)
    assert mm.compute_quote(m, top()) is None


def test_quotes_at_the_floor_when_the_band_allows(cfg, ledger):
    """Score is quadratic in distance from mid, so the tightest placement our
    protections permit is also the best-scoring one we may legitimately take."""
    cfg.market_maker.half_spread = 0.005
    cfg.risk.min_edge_after_fees = 0.0
    mm = make_mm(cfg, ledger)
    m = mm_market(rewards_max_spread=0.03)
    quote = mm.compute_quote(m, top())
    assert quote is not None
    # Placed at the floor, not parked out at 0.9 * band where score ~ 1%.
    assert quote.captured_spread / 2 <= 0.005 + m.tick_size + 1e-9
    from polymarket_bot.rewards import score_fraction
    assert score_fraction(quote.captured_spread / 2, m.rewards_max_spread) > 0.5


def test_fee_floor_still_wins_over_the_reward_band(cfg, ledger):
    """Break-even is never traded away for reward eligibility."""
    cfg.market_maker.half_spread = 0.001
    cfg.risk.min_edge_after_fees = 0.05          # forces a very wide floor
    mm = make_mm(cfg, ledger)
    m = mm_market(rewards_max_spread=0.01)       # band cannot hold that floor
    assert mm.compute_quote(m, top()) is None


# --- cold-start adverse selection ---

def test_toxic_tape_widens_a_market_we_never_traded(cfg, ledger):
    """Markout feedback is blind to a market we have never been filled in. The
    tape estimate must cover that gap, or the only way to learn a market is
    dangerous is to lose money in it first."""
    from polymarket_bot.toxicity import ToxicityModel

    cfg.market_maker.half_spread = 0.004
    cfg.risk.min_edge_after_fees = 0.0
    mm = make_mm(cfg, ledger)
    m = mm_market(rewards_max_spread=0.05)

    baseline = mm.compute_quote(m, top())
    assert baseline is not None

    drift = [(1000.0 + i, 0.40 + 0.002 * i - 0.005, 0.40 + 0.002 * i + 0.005,
              100.0, 100.0) for i in range(60)]
    mm.toxicity = ToxicityModel()
    mm.toxicity.fit({m.id: drift})
    assert mm.toxicity.multiplier(m.id) > 1.0

    widened = mm.compute_quote(m, top())
    assert widened is not None
    assert widened.captured_spread > baseline.captured_spread


def test_measured_markout_supersedes_the_tape_estimate(cfg, ledger):
    """Once a market has produced enough real fills, evidence wins."""
    from polymarket_bot.toxicity import ToxicityModel

    cfg.market_maker.half_spread = 0.004
    cfg.risk.min_edge_after_fees = 0.0
    mm = make_mm(cfg, ledger)
    m = mm_market(rewards_max_spread=0.05)

    drift = [(1000.0 + i, 0.40 + 0.002 * i - 0.005, 0.40 + 0.002 * i + 0.005,
              100.0, 100.0) for i in range(60)]
    mm.toxicity = ToxicityModel()
    mm.toxicity.fit({m.id: drift})
    mm._fill_counts = {m.id: 500}          # plenty of measurement
    # Realized feedback says benign -> the alarming estimate is dropped.
    assert mm._spread_mult(m.id) == pytest.approx(1.0)


# --- late fills must never vanish with a cancelled order ---

def test_cancel_market_captures_a_late_partial_fill(cfg, ledger):
    """A fill landing between the last _sync_live_fills poll and the cancel
    used to vanish with the order — shares owned, never recorded. The final
    post-cancel status read must record it and label the quote filled=True."""
    from polymarket_bot.marketmaker import TrackedOrder

    mm = make_mm(cfg, ledger)
    m = mm_market()
    trader = mock.Mock()
    trader.order_status.return_value = {"status": "canceled", "size_matched": 15.0}
    mm._trader = trader

    quote = mm.compute_quote(m, top())
    quote.p_fill_pred = 0.4
    mm._quotes[m.id] = quote
    mm._orders[m.id] = [TrackedOrder(order_id="o1", market=m, outcome_index=0,
                                     price=0.44, size=50.0, matched_recorded=0.0)]
    mm._cancel_market(m.id)

    trader.cancel.assert_called_once_with("o1")
    positions = ledger.open_positions("dry-run")
    assert len(positions) == 1 and positions[0].size == pytest.approx(15.0)
    # The calibrator label flipped to filled, not double-recorded False+True.
    outcomes = ledger.quote_outcomes("dry-run")
    assert [o for o in outcomes if o[1]] and not [o for o in outcomes if not o[1]]


def test_cancel_market_with_no_late_fill_labels_unfilled(cfg, ledger):
    from polymarket_bot.marketmaker import TrackedOrder

    mm = make_mm(cfg, ledger)
    m = mm_market()
    trader = mock.Mock()
    trader.order_status.return_value = {"status": "canceled", "size_matched": 0.0}
    mm._trader = trader
    quote = mm.compute_quote(m, top())
    quote.p_fill_pred = 0.4
    mm._quotes[m.id] = quote
    mm._orders[m.id] = [TrackedOrder(order_id="o1", market=m, outcome_index=0,
                                     price=0.44, size=50.0)]
    mm._cancel_market(m.id)
    assert ledger.open_positions("dry-run") == []
    outcomes = ledger.quote_outcomes("dry-run")
    assert [o for o in outcomes if not o[1]] and not [o for o in outcomes if o[1]]
