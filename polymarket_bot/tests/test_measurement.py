"""Measurement and capital-allocation defects the paper report exposed.

A multi-day paper run produced: 23,063 estimates with an average edge ratio of
1.00 and zero qualifying, twenty fade legs holding $1,891, zero market-maker
fills, and zero markouts. Three separate causes, each fixed and pinned here.

B1  `_paper_fills` was only ever called from `cycle()`. The MM cycle samples the
    book at its interval while the tape moves at tick rate, so a crossing that
    opened and closed between two cycles was never observed. The only strategy
    with a mechanical edge therefore generated no data at all — and "no fills"
    is indistinguishable from "no opportunity" unless the near misses are also
    recorded, which is what shadow_quotes is for.

B2  The report printed `avg edge: 1.00` and nothing else, so it could not
    distinguish "the estimator finds no mispricing" from "the estimator echoes
    the market price". Only the second means the LLM is being paid for nothing.

B3  `size_usd` handed out max_total_exposure_pct first-come-first-served. The
    fade runs every cycle over hundreds of candidates and took $1,891 while the
    market maker got $0 — not because of a decision, but because it asked first.
"""

from unittest import mock

from polymarket_bot.models import simple_estimate
from polymarket_bot.portfolio import Portfolio

from .conftest import make_market
from .test_marketmaker import make_mm, mm_market, top


def _hold(ledger, *, strategy: str, usd: float, token: str, mode: str = "paper"):
    """Record a filled BUY of `usd` notional under `strategy`."""
    m = make_market(id=f"m-{token}", clob_token_ids=[token, f"{token}-b"])
    ledger.record_trade(mode=mode, estimate=simple_estimate(m, 0, 0.50),
                        category="other", side="BUY", price=0.50,
                        size=usd / 0.50, order_id=f"o-{token}",
                        status="filled", strategy=strategy)


def _exposure_cfg(cfg, *, reserve: float) -> None:
    """Isolate the TOTAL-exposure caps: 30% global, `reserve` held for the MM.

    max_category_pct is lifted out of the way on purpose. At its 10% default a
    $760 holding trips the per-category cap first, so every assertion below would
    pass or fail for a reason that has nothing to do with the reserve — including
    the ones that are supposed to fail.
    """
    cfg.portfolio.max_total_exposure_pct = 0.30      # 0.30 * 5000 = $1,500
    cfg.portfolio.reserve_for_mm_pct = reserve       # directional cap = 1500 - this
    cfg.portfolio.max_category_pct = 1.0


# --- B1: fills must be evaluated at tick resolution ---

def test_paper_fill_is_seen_on_a_tick_not_only_on_the_cycle(cfg, ledger):
    """A crossing between two cycles must not be invisible.

    react_to_tick used to reprice only. The tape here crosses the NO bid and the
    cycle is never called again, so a fill can only be recorded if ticks are
    evaluated — which is the whole reason a multi-day run had zero of them.
    """
    m = mm_market()
    yes_tok, no_tok = m.clob_token_ids[0], m.clob_token_ids[1]
    tops = {yes_tok: top(bid=0.43, ask=0.47), no_tok: top(bid=0.53, ask=0.57)}
    mm = make_mm(cfg, ledger, mode="paper", tops=tops)

    with mock.patch("polymarket_bot.marketmaker.alert"):
        quotes = mm.cycle([m])
        assert quotes, "no quote to be filled — fixture problem, not the defect"
        quote = quotes[0]
        assert ledger.open_positions("paper") == []

        # The tape trades through our NO bid, then we get a tick for that token.
        crossed = top(bid=quote.no_bid - 0.02, ask=quote.no_bid - 0.005)
        tops[no_tok] = crossed
        mm.react_to_tick(no_tok, crossed)

    filled = ledger.open_positions("paper")
    assert filled, ("the tape crossed our quote and no fill was recorded: the MM "
                    "produces no markouts, so its economics stay unmeasured")
    assert filled[0].strategy == "mm"


def test_a_tick_that_does_not_cross_fills_nothing(cfg, ledger):
    """The tick path must not invent fills — it only observes."""
    m = mm_market()
    yes_tok, no_tok = m.clob_token_ids[0], m.clob_token_ids[1]
    tops = {yes_tok: top(bid=0.43, ask=0.47), no_tok: top(bid=0.53, ask=0.57)}
    mm = make_mm(cfg, ledger, mode="paper", tops=tops)
    with mock.patch("polymarket_bot.marketmaker.alert"):
        assert mm.cycle([m])
        mm.react_to_tick(no_tok, tops[no_tok])
    assert ledger.open_positions("paper") == []


def test_dry_run_does_not_book_paper_fills_from_ticks(cfg, ledger):
    """Only paper mode simulates fills; dry-run stays an observer."""
    m = mm_market()
    yes_tok, no_tok = m.clob_token_ids[0], m.clob_token_ids[1]
    tops = {yes_tok: top(bid=0.43, ask=0.47), no_tok: top(bid=0.53, ask=0.57)}
    mm = make_mm(cfg, ledger, mode="dry-run", tops=tops)
    with mock.patch("polymarket_bot.marketmaker.alert"):
        quotes = mm.cycle([m])
        assert quotes
        crossed = top(bid=quotes[0].no_bid - 0.02, ask=quotes[0].no_bid - 0.005)
        tops[no_tok] = crossed
        mm.react_to_tick(no_tok, crossed)
    assert ledger.open_positions("dry-run") == []


def test_the_tick_path_never_makes_a_rest_call(cfg, ledger):
    """Evaluating fills on the recv thread must not reach for HTTP.

    `_top` falls back to clob.order_book when the WS store is empty. On the
    websocket thread that blocking call stalls the recv loop, the last-message
    timestamp stops advancing, and ws_staleness_kill_sec pulls every quote — the
    bot killing its own market making to collect a paper fill.
    """
    m = mm_market()
    yes_tok, no_tok = m.clob_token_ids[0], m.clob_token_ids[1]
    tops = {yes_tok: top(bid=0.43, ask=0.47), no_tok: top(bid=0.53, ask=0.57)}
    mm = make_mm(cfg, ledger, mode="paper", tops=tops)
    with mock.patch("polymarket_bot.marketmaker.alert"):
        assert mm.cycle([m])
        # The WS store goes dark for one token: the tick path must skip it, not
        # fetch it. (cycle() is allowed the fallback — it runs on its own thread.)
        tops.pop(no_tok)
        mm._clob.order_book.reset_mock()
        mm.react_to_tick(yes_tok, tops[yes_tok])
    mm._clob.order_book.assert_not_called()


def test_a_quote_that_never_fills_still_records_how_close_the_tape_came(cfg, ledger):
    """Zero fills must still be informative.

    Without this, a run reports "0 fills" whether our spread was one tick too
    wide or the market never traded at all — and those call for opposite actions.
    """
    m = mm_market()
    yes_tok, no_tok = m.clob_token_ids[0], m.clob_token_ids[1]
    tops = {yes_tok: top(bid=0.43, ask=0.47), no_tok: top(bid=0.53, ask=0.57)}
    mm = make_mm(cfg, ledger, mode="paper", tops=tops)

    with mock.patch("polymarket_bot.marketmaker.alert"):
        quotes = mm.cycle([m])
        assert quotes
        quote = quotes[0]
        # Approach to within half a cent of the YES bid, without crossing it.
        near = quote.yes_bid + 0.005
        tops[yes_tok] = top(bid=near - 0.01, ask=near)
        mm._paper_fills()
        mm._cancel_market(m.id)             # quote retires -> one shadow row

    rows = ledger.shadow_quote_summary("paper")
    assert rows, "a retired quote left no shadow record"
    row = rows[0]
    assert row["closest"] <= 0.0051, row    # measured the near miss, not a fill
    assert row["looks"] >= 1
    assert ledger.open_positions("paper") == []


def test_shadow_gap_is_negative_when_the_tape_crossed(cfg, ledger):
    """A crossing shows up as a non-positive gap, distinguishing it from a miss."""
    m = mm_market()
    yes_tok, no_tok = m.clob_token_ids[0], m.clob_token_ids[1]
    tops = {yes_tok: top(bid=0.43, ask=0.47), no_tok: top(bid=0.53, ask=0.57)}
    mm = make_mm(cfg, ledger, mode="paper", tops=tops)
    with mock.patch("polymarket_bot.marketmaker.alert"):
        quotes = mm.cycle([m])
        assert quotes
        quote = quotes[0]
        crossed = top(bid=quote.no_bid - 0.02, ask=quote.no_bid - 0.005)
        tops[no_tok] = crossed
        mm._paper_fills()                   # fills and retires the quote
    rows = ledger.shadow_quote_summary("paper")
    assert rows and rows[0]["closest"] < 0, rows


def test_shadow_recording_never_breaks_quoting(cfg, ledger):
    """Analytics is not allowed to take the MM down with it."""
    m = mm_market()
    yes_tok, no_tok = m.clob_token_ids[0], m.clob_token_ids[1]
    tops = {yes_tok: top(bid=0.43, ask=0.47), no_tok: top(bid=0.53, ask=0.57)}
    mm = make_mm(cfg, ledger, mode="paper", tops=tops)
    with mock.patch("polymarket_bot.marketmaker.alert"), \
            mock.patch.object(ledger, "record_shadow_quote",
                              side_effect=RuntimeError("disk full")):
        quotes = mm.cycle([m])
        assert quotes
        mm._paper_fills()
        mm._cancel_market(m.id)             # must not raise


# --- B2: does the estimator change any decision? ---

def test_estimates_summary_counts_a_binding_estimate(cfg, ledger):
    """p_est below the bias-corrected price is the ONLY case that moves a fade."""
    m = make_market(id="e1", clob_token_ids=["e1-y", "e1-n"])
    # Echoes the market: p_est == p_mkt, edge_ratio 1.0 — decides nothing.
    ledger.record_estimate(simple_estimate(m, 0, 0.05), False)
    summary = ledger.estimates_summary(fade_bias=0.30)
    assert summary["total"] == 1
    assert summary["binding"] == 0
    assert summary["informative"] == 0

    # Genuinely bearish: 0.02 < 0.05 * 0.70 — here the model sets fair value.
    est = simple_estimate(m, 0, 0.05)
    est.p_est = 0.02
    ledger.record_estimate(est, False)
    summary = ledger.estimates_summary(fade_bias=0.30)
    assert summary["total"] == 2
    assert summary["binding"] == 1
    assert summary["informative"] == 1


def test_binding_threshold_follows_the_bias_prior(cfg, ledger):
    """A wider bias prior makes the estimator harder to bind — and says so.

    At bias 0.40 the model must beat p_mkt*0.60 to matter; at 0.10 it only has
    to beat p_mkt*0.90. Reporting a fixed threshold would misstate which runs
    the LLM actually influenced.
    """
    m = make_market(id="e2", clob_token_ids=["e2-y", "e2-n"])
    est = simple_estimate(m, 0, 0.05)
    est.p_est = 0.04                       # ratio 0.8
    ledger.record_estimate(est, False)
    assert ledger.estimates_summary(fade_bias=0.40)["binding"] == 0   # 0.8 > 0.6
    assert ledger.estimates_summary(fade_bias=0.10)["binding"] == 1   # 0.8 < 0.9


def test_fade_counts_how_often_the_estimator_bound(cfg, ledger):
    """The same measurement live, on the path that actually trades."""
    from .test_fade import make_estimate, make_fade
    fade, _ = make_fade(cfg, ledger)
    cfg.fade.bias_discount = 0.30
    fade.reject_reason(make_estimate(p_mkt=0.05, p_est=None))      # echo
    fade.reject_reason(make_estimate(p_mkt=0.05, p_est=0.01))      # binds
    assert fade.est_seen == 2
    assert fade.est_binding == 1


# --- B3: the directional book cannot eat the MM's room ---

def test_directional_book_stops_at_the_reserve(cfg, ledger):
    """Fade/longshot get max_total_exposure_pct MINUS the reserve, and no more."""
    _exposure_cfg(cfg, reserve=0.15)                 # directional cap = $750
    portfolio = Portfolio(cfg, ledger, "paper")

    _hold(ledger, strategy="fade", usd=760.0, token="f1")
    # Global room is NOT exhausted — $740 of the $1,500 cap is still free.
    assert ledger.total_exposure("paper") < (
        cfg.portfolio.max_total_exposure_pct * cfg.portfolio.bankroll_usd)
    assert portfolio.size_usd("other", 0.965, 0.95) is None, (
        "the directional book was allowed past its reserve while global room "
        "remained — which is how the MM ended a whole run with $0")


def test_market_maker_inventory_does_not_consume_the_reserve(cfg, ledger):
    """Same notional held by the MM leaves the directional budget untouched."""
    _exposure_cfg(cfg, reserve=0.15)
    portfolio = Portfolio(cfg, ledger, "paper")

    _hold(ledger, strategy="mm", usd=760.0, token="q1")
    size = portfolio.size_usd("other", 0.965, 0.95)
    assert size is not None and size > 0, (
        "MM inventory was charged against the directional cap — the reserve is "
        "meant to protect the MM, not to lock it out")


def test_global_cap_still_binds_above_the_reserve(cfg, ledger):
    """The reserve is an EXTRA limit; it must not raise the account-wide one."""
    _exposure_cfg(cfg, reserve=0.15)
    portfolio = Portfolio(cfg, ledger, "paper")
    _hold(ledger, strategy="mm", usd=1_600.0, token="q2")   # past 0.30 * 5000
    assert portfolio.size_usd("other", 0.965, 0.95) is None


def test_reserve_of_zero_restores_the_old_behavior(cfg, ledger):
    """With the reserve off, only the global cap applies — the previous rule."""
    _exposure_cfg(cfg, reserve=0.0)
    portfolio = Portfolio(cfg, ledger, "paper")
    _hold(ledger, strategy="fade", usd=760.0, token="f2")
    assert portfolio.size_usd("other", 0.965, 0.95) is not None


def test_positions_carry_the_strategy_that_opened_them(cfg, ledger):
    """Exit routing and the reserve both depend on this label being right."""
    _hold(ledger, strategy="fade", usd=100.0, token="s1")
    _hold(ledger, strategy="mm", usd=100.0, token="s2")
    by_token = {p.token_id: p.strategy for p in ledger.open_positions("paper")}
    assert by_token == {"s1": "fade", "s2": "mm"}


def test_a_later_sell_cannot_relabel_the_leg(cfg, ledger):
    """The OPENING strategy decides the exit rule, so it must be immutable.

    Whichever component sells writes its own strategy tag on the SELL row; if
    that overwrote the label, the next exit for a partially-sold fade leg would
    be routed through the wrong rule.
    """
    _hold(ledger, strategy="fade", usd=100.0, token="s3")
    m = make_market(id="m-s3", clob_token_ids=["s3", "s3-b"])
    ledger.record_trade(mode="paper", estimate=simple_estimate(m, 0, 0.50),
                        category="other", side="SELL", price=0.55, size=50.0,
                        order_id="sell-s3", status="filled", strategy="guardian")
    pos = [p for p in ledger.open_positions("paper") if p.token_id == "s3"]
    assert pos and pos[0].strategy == "fade"


# --- F-027: PnL attribution must follow the OPENING strategy ---

def test_realized_pnl_is_charged_to_the_strategy_that_opened_the_trade(cfg, ledger):
    """The bug a live report showed as `fade +0.00 | longshot -3.52`.

    Attribution took the LAST trade row's label, so whichever component sold owned
    the result. main._exit_one sells through the generic executor path, which fell
    through to record_trade's "longshot" default — so a fade book that was exited
    reported its entire loss against a strategy that made no trades at all.
    """
    m = make_market(id="attr", clob_token_ids=["at-y", "at-n"])
    ledger.record_trade(mode="paper", estimate=simple_estimate(m, 1, 0.976),
                        category="other", side="BUY", price=0.976, size=100.0,
                        order_id="b", status="filled", strategy="fade")
    # The exit, mislabelled the way the generic path used to label it.
    ledger.record_trade(mode="paper", estimate=simple_estimate(m, 1, 0.946),
                        category="other", side="SELL", price=0.946, size=100.0,
                        order_id="s", status="filled", strategy="longshot")

    by_strategy = ledger.realized_pnl_by_strategy("paper")
    assert by_strategy.get("fade", 0.0) < 0, (
        f"the loss was not charged to fade: {by_strategy}")
    assert "longshot" not in by_strategy or by_strategy["longshot"] == 0.0, (
        f"longshot was charged for a trade it never made: {by_strategy}")


def test_misattribution_would_have_misdirected_the_capital_allocator(cfg, ledger):
    """Why this is more than a cosmetic report defect.

    realized_pnl_by_strategy feeds research.sharpe_allocation, which sets
    fade.size_scale. Charging a losing book's losses to an idle strategy makes the
    allocator throttle the wrong one — and leave the loser at full size.
    """
    from polymarket_bot.research import sharpe_allocation

    m = make_market(id="alloc", clob_token_ids=["al-y", "al-n"])
    for i, (entry, exit_price) in enumerate([(0.976, 0.94), (0.970, 0.93)]):
        tok = f"al-{i}"
        mk = make_market(id=f"alloc{i}", clob_token_ids=[tok, f"{tok}-b"])
        ledger.record_trade(mode="paper", estimate=simple_estimate(mk, 0, entry),
                            category="other", side="BUY", price=entry, size=100.0,
                            order_id=f"b{i}", status="filled", strategy="fade")
        ledger.record_trade(mode="paper", estimate=simple_estimate(mk, 0, exit_price),
                            category="other", side="SELL", price=exit_price,
                            size=100.0, order_id=f"s{i}", status="filled",
                            strategy="longshot")
    del m

    pnl = ledger.realized_pnl_by_strategy("paper")
    assert pnl.get("fade", 0.0) < 0
    weights = sharpe_allocation({k: [v] for k, v in pnl.items()})
    # Whatever the weighting scheme, the LOSER must not outrank the idle strategy.
    assert weights.get("fade", 0.0) <= weights.get("longshot", 1.0), weights


def test_the_sell_row_itself_carries_the_opening_strategy(cfg, ledger):
    """Belt and braces: write the truth, so ad-hoc SQL is not a trap either."""
    from unittest import mock

    from polymarket_bot.executor import Executor
    from polymarket_bot.main import est_to_plan
    from polymarket_bot.models import BookLevel, OrderBook, Candidate, Estimate

    m = make_market(id="sellrow", clob_token_ids=["sr-y", "sr-n"])
    clob = mock.Mock()
    clob.order_book.return_value = OrderBook(
        bids=[BookLevel(price=0.95, size=500)],
        asks=[BookLevel(price=0.96, size=500)])
    ex = Executor(cfg, ledger, clob, None, "paper")
    est = Estimate(candidate=Candidate(market=m, outcome_index=1,
                                      token_id="sr-n", p_mkt=0.95),
                   p_mkt=0.95, p_est=0.95, signals=[])
    ex.execute_sell(est_to_plan(est, "other"), 50.0, 0.90, strategy="fade")

    row = ledger._conn.execute(
        "SELECT strategy FROM trades WHERE side='SELL'").fetchone()
    assert row["strategy"] == "fade"
