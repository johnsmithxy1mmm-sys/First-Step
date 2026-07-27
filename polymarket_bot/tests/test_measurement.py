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

import pytest

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


# --- B1b: a gap is only readable next to the tape's own movement ---

def _quote_a_market(cfg, ledger, tops, m):
    with mock.patch("polymarket_bot.marketmaker.alert"):
        mm = make_mm(cfg, ledger, mode="paper", tops=tops)
        quotes = mm.cycle([m])
        assert quotes, "fixture failed to produce a quote"
        return mm, quotes[0]


def test_a_dead_market_is_distinguishable_from_one_we_merely_track(cfg, ledger):
    """The defect in the instrument itself, caught by reading its own output.

    `yes_bid` comes from the microprice, so our bid moves WITH the book and
    `ask - our_bid` is constant whenever the book's shape is stable. The first
    live run showed closest == avg_gap to four decimals on three of four markets,
    which reads identically for "nothing trades here" and "the book moves and we
    follow it a fixed distance away" — opposite conclusions. Only the tape's own
    range separates them.
    """
    m = mm_market()
    yes_tok, no_tok = m.clob_token_ids[0], m.clob_token_ids[1]

    # Market A: the book never moves at all.
    tops = {yes_tok: top(bid=0.43, ask=0.47), no_tok: top(bid=0.53, ask=0.57)}
    mm, _ = _quote_a_market(cfg, ledger, tops, m)
    for _ in range(5):
        mm._paper_fills()
    mm._cancel_market(m.id)
    dead = ledger.shadow_quote_summary("paper")[0]
    assert dead["tape_range"] == pytest.approx(0.0, abs=1e-9), dead

    # Market B: same closing gap, but the ask travels 3 cents while we quote.
    ledger._execute("DELETE FROM shadow_quotes")
    m2 = mm_market(id="mm2", clob_token_ids=["mm2-yes", "mm2-no"])
    y2, n2 = m2.clob_token_ids[0], m2.clob_token_ids[1]
    tops2 = {y2: top(bid=0.43, ask=0.47), n2: top(bid=0.53, ask=0.57)}
    mm2, _ = _quote_a_market(cfg, ledger, tops2, m2)
    for ask in (0.47, 0.48, 0.50, 0.49, 0.47):
        tops2[y2] = top(bid=ask - 0.04, ask=ask)
        mm2._paper_fills()
    mm2._cancel_market(m2.id)
    alive = ledger.shadow_quote_summary("paper")[0]
    assert alive["tape_range"] >= 0.02, alive

    assert dead["tape_range"] < alive["tape_range"], (
        "a flat book and a 3-cent-range book produced the same measurement — "
        "the gap column on its own cannot support a widen/tighten decision")


def test_the_verdict_never_blames_our_spread_when_nothing_traded(cfg):
    """The conclusion the column exists to prevent."""
    from polymarket_bot.analytics import _shadow_verdict
    assert "no flow" in _shadow_verdict(closest=0.006, tape_range=0.0)
    assert "no flow" in _shadow_verdict(closest=0.010, tape_range=0.0005)
    # A moving tape that stays away is a different diagnosis.
    assert "no flow" not in _shadow_verdict(closest=0.010, tape_range=0.03)
    # And a genuine near miss is called out as actionable.
    assert "tighten" in _shadow_verdict(closest=0.001, tape_range=0.03)
    assert "crossed" in _shadow_verdict(closest=-0.001, tape_range=0.03)


def test_the_shadow_table_migrates_without_losing_old_rows(cfg, tmp_path):
    """An existing DB must gain the columns, not crash or start over.

    The user's paper ledger already has shadow_quotes rows from the previous
    version; CREATE TABLE IF NOT EXISTS will not add a column to it.
    """
    from polymarket_bot.ledger import Ledger
    path = str(tmp_path / "old.sqlite")
    old = Ledger(path)
    old._conn.execute("DROP TABLE shadow_quotes")
    old._conn.execute(
        "CREATE TABLE shadow_quotes (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "ts TEXT NOT NULL, mode TEXT NOT NULL, market_id TEXT NOT NULL, "
        "question TEXT, yes_bid REAL NOT NULL, no_bid REAL NOT NULL, "
        "min_gap_yes REAL NOT NULL, min_gap_no REAL NOT NULL, "
        "looks INTEGER NOT NULL, size REAL NOT NULL)")
    old._conn.execute(
        "INSERT INTO shadow_quotes (ts, mode, market_id, question, yes_bid, "
        "no_bid, min_gap_yes, min_gap_no, looks, size) "
        "VALUES ('t','paper','legacy','Q?',0.44,0.54,0.006,0.011,78,200)")
    old._conn.commit()
    old.close()

    fresh = Ledger(path)                      # _migrate runs here
    try:
        rows = fresh.shadow_quote_summary("paper")
        assert len(rows) == 1 and rows[0]["market_id"] == "legacy"
        assert rows[0]["looks"] == 78
        # NOT 0.0. A row written before the measurement existed must read as
        # unknown; back-filling a default turns silence into a finding, and the
        # verdict column then reports "no flow, unpaid" about a quote nobody
        # ever measured. This assertion previously encoded the bug.
        assert rows[0]["tape_range"] is None
        assert rows[0]["reward_frac"] is None
        assert rows[0]["rested_sec"] is None
        assert rows[0]["measured"] == 0
        fresh.record_shadow_quote(
            mode="paper", market_id="new", question="Q2?", yes_bid=0.4,
            no_bid=0.5, min_gap_yes=0.01, min_gap_no=0.02, looks=3, size=100.0,
            tape_range_yes=0.03, tape_range_no=0.0)
        assert len(fresh.shadow_quote_summary("paper")) == 2
    finally:
        fresh.close()


# --- B1c: with no fills, the reward score IS the income ---

def test_a_resting_quote_records_what_it_earned_not_only_what_it_missed(cfg, ledger):
    """MM income is spread + rebate + rewards. With zero fills the first two are
    exactly zero, so the reward score is the entire return — and nothing recorded
    it: `reward_share` was consulted only when RANKING markets, never for a quote
    that actually rested.
    """
    m = mm_market()
    yes_tok, no_tok = m.clob_token_ids[0], m.clob_token_ids[1]
    tops = {yes_tok: top(bid=0.43, ask=0.47), no_tok: top(bid=0.53, ask=0.57)}
    with mock.patch("polymarket_bot.marketmaker.alert"):
        mm = make_mm(cfg, ledger, mode="paper", tops=tops)
        assert mm.cycle([m])
        mm._paper_fills()
        mm._cancel_market(m.id)
    row = ledger.shadow_quote_summary("paper")[0]
    assert row["reward_frac"] > 0.0, (
        "a quote inside the rewards band recorded no earnings: on a no-flow "
        "market the report shows zero income where the truth is the pool share")
    assert row["rested_sec"] >= 0.0
    assert row["score_seconds"] == pytest.approx(
        row["reward_frac"] * row["rested_sec"], rel=1e-6)


def test_a_quote_outside_the_band_is_recorded_as_earning_nothing(cfg, ledger):
    """The quadratic is the point: the band edge is worth ~1%, outside it 0.

    Adverse-selection widening (toxicity, markout feedback, vol) pushes the quote
    outward. A report that counts only fills shows "3 markets quoted" either way.
    """
    from polymarket_bot.rewards import score_fraction
    v = 0.03
    assert score_fraction(0.0, v) == pytest.approx(1.0)
    assert score_fraction(0.9 * v, v) == pytest.approx(0.01, abs=1e-9)
    assert score_fraction(v, v) == 0.0
    assert score_fraction(2 * v, v) == 0.0


def test_the_verdict_separates_unpaid_idleness_from_paid_idleness(cfg):
    """A resting quote holds no inventory, so 'no flow' is not by itself waste.

    It is waste only when the quote also scores nothing. Those are different
    verdicts because they call for different actions — leave it, versus stop
    quoting that market or move the quote back inside the band.
    """
    from polymarket_bot.analytics import _shadow_verdict
    paid = _shadow_verdict(closest=0.006, tape_range=0.0, reward_frac=0.8)
    unpaid = _shadow_verdict(closest=0.006, tape_range=0.0, reward_frac=0.0)
    assert "rewards only" in paid
    assert "unpaid" in unpaid
    assert paid != unpaid


def test_a_database_with_fabricated_defaults_is_repaired(cfg, tmp_path):
    """The DB shipped by the first version of this migration must be healed.

    An earlier build added the columns with `DEFAULT 0.0`, which SQLite writes
    into every pre-existing row. The live report then showed four markets at
    "Reward score 0%, Rested -, no flow, unpaid" while each had 10-12 recorded
    quotes — a conclusion drawn entirely from a migration default.
    """
    from polymarket_bot.ledger import Ledger
    path = str(tmp_path / "fabricated.sqlite")
    led = Ledger(path)
    led._conn.execute("DELETE FROM shadow_quotes")
    # Exactly what the bad migration left behind: zeros, with real look counts.
    led._conn.execute(
        "INSERT INTO shadow_quotes (ts, mode, market_id, question, yes_bid, "
        "no_bid, min_gap_yes, min_gap_no, looks, size, tape_range_yes, "
        "tape_range_no, reward_frac, rested_sec) "
        "VALUES ('t','paper','pritzker','Will JB Pritzker win?',0.44,0.54,"
        "0.006,0.011,78,200,0.0,0.0,0.0,0.0)")
    # A genuinely measured row must survive untouched.
    led._conn.execute(
        "INSERT INTO shadow_quotes (ts, mode, market_id, question, yes_bid, "
        "no_bid, min_gap_yes, min_gap_no, looks, size, tape_range_yes, "
        "tape_range_no, reward_frac, rested_sec) "
        "VALUES ('t','paper','real','Measured market',0.44,0.54,"
        "0.002,0.011,40,200,0.0,0.0,0.81,3600.0)")
    led._conn.commit()
    led.close()

    healed = Ledger(path)                       # _migrate repairs on open
    try:
        by_id = {r["market_id"]: r for r in healed.shadow_quote_summary("paper")}
        fabricated = by_id["pritzker"]
        assert fabricated["reward_frac"] is None, fabricated
        assert fabricated["tape_range"] is None, fabricated
        assert fabricated["measured"] == 0

        measured = by_id["real"]
        assert measured["reward_frac"] == pytest.approx(0.81)
        assert measured["tape_range"] == pytest.approx(0.0)   # a REAL flat tape
        assert measured["measured"] == 1
    finally:
        healed.close()


def test_an_unmeasured_row_produces_no_verdict_about_rewards(cfg):
    """Silence must not be reported as a finding."""
    from polymarket_bot.analytics import _shadow_verdict
    assert _shadow_verdict(closest=0.006, tape_range=None) == "not measured"
    assert "not measured" in _shadow_verdict(closest=0.006, tape_range=0.0,
                                             reward_frac=None)
    # A measured zero still says what it means.
    assert _shadow_verdict(closest=0.006, tape_range=0.0,
                           reward_frac=0.0) == "no flow, unpaid"
