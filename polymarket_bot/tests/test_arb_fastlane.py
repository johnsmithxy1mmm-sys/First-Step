"""WS-triggered arbitrage fastlane: dedupe, cooldown, Bot wiring."""

from unittest import mock

from polymarket_bot.arb_fastlane import ArbFastlane
from polymarket_bot.ws_feed import TopOfBook

from .test_hardening import make_bot


# --- worker mechanics (driven synchronously via drain) ---

def test_flag_dedupes_and_drain_checks_once():
    seen = []
    lane = ArbFastlane(seen.append, min_recheck_sec=0.0)
    for _ in range(50):                       # a tick storm on one structure
        lane.flag(("basket", "ev1"))
    lane.flag(("chain", ("a", "b")))
    assert lane.drain(now=100.0) == 2
    assert sorted(seen) == [("basket", "ev1"), ("chain", ("a", "b"))]


def test_cooldown_coalesces_rechecks():
    seen = []
    lane = ArbFastlane(seen.append, min_recheck_sec=5.0)
    lane.flag(("basket", "ev1"))
    assert lane.drain(now=100.0) == 1
    lane.flag(("basket", "ev1"))
    assert lane.drain(now=102.0) == 0         # inside cooldown — skipped
    lane.flag(("basket", "ev1"))
    assert lane.drain(now=106.0) == 1         # cooldown expired — checked again
    assert len(seen) == 2


def test_check_exception_does_not_kill_the_drain():
    calls = []

    def check(key):
        calls.append(key)
        if key == ("basket", "boom"):
            raise RuntimeError("book fetch failed")

    lane = ArbFastlane(check, min_recheck_sec=0.0)
    lane.flag(("basket", "boom"))
    lane.flag(("basket", "ok"))
    assert lane.drain(now=1.0) == 2           # both attempted despite the error
    assert len(calls) == 2


# --- Bot wiring ---

def test_on_tick_flags_watched_arb_token(tmp_path):
    bot = make_bot(tmp_path)
    bot._basket_token_map = {"tokA": ("basket", "ev1")}
    bot._chain_token_map = {"tokB": ("chain", ("s", "p"))}
    with mock.patch.object(bot.arb_fastlane, "flag") as flag:
        tick = TopOfBook(bid=0.5, bid_size=1, ask=0.52, ask_size=1, ts=1.0)
        bot.on_tick("tokA", tick)
        bot.on_tick("tokB", tick)
        bot.on_tick("unrelated", tick)
    assert flag.call_args_list == [mock.call(("basket", "ev1")),
                                   mock.call(("chain", ("s", "p")))]
    bot.close()


def test_fastlane_check_routes_to_the_right_scanner(tmp_path):
    bot = make_bot(tmp_path)
    group = [mock.Mock()]
    pair = (mock.Mock(), mock.Mock(), "date")
    bot._arb_groups = {"ev1": group}
    bot._chain_pairs = {("s", "p"): pair}
    with mock.patch.object(bot.arb, "check_group") as cg, \
            mock.patch.object(bot.chain_arb, "check_pair") as cp, \
            mock.patch.object(bot, "_record_arb_opportunity") as ra, \
            mock.patch.object(bot, "_record_chain_opportunity") as rc, \
            mock.patch.object(bot, "_may_execute", return_value=False):
        bot._fastlane_check(("basket", "ev1"))
        bot._fastlane_check(("chain", ("s", "p")))
        bot._fastlane_check(("basket", "gone"))   # structure rotated out — no-op
    cg.assert_called_once_with(group, allow_execute=False)
    cp.assert_called_once_with(pair[0], pair[1], "date", allow_execute=False)
    ra.assert_called_once()                       # fastlane windows are recorded
    rc.assert_called_once()
    bot.close()


def test_fastlane_found_window_lands_in_opportunity_ledger(tmp_path):
    """The short-lived windows only the fastlane sees must reach the ledger —
    otherwise measured capacity understates the fastest opportunities."""
    from polymarket_bot.arbitrage import ArbLeg, BasketArb
    from .conftest import make_market
    bot = make_bot(tmp_path)
    leg = ArbLeg(market=make_market(), outcome_index=0, token_id="t",
                 ask=0.30, depth=100)
    arb = BasketArb(event_id="ev1", event_title="Election", side="YES",
                    legs=[leg, leg, leg], taker_coef=0.0)
    bot._arb_groups = {"ev1": [mock.Mock()]}
    with mock.patch.object(bot.arb, "check_group", return_value=arb), \
            mock.patch.object(bot, "_may_execute", return_value=False):
        bot._fastlane_check(("basket", "ev1"))
    stats = bot.ledger.opportunity_stats("paper")
    assert stats and stats[0]["strategy"] == "arb" and stats[0]["windows"] == 1
    assert stats[0]["executed"] == 0              # gates closed -> not executed
    bot.close()


def test_ws_watch_union_includes_arb_tokens(tmp_path):
    bot = make_bot(tmp_path)
    bot.ws = mock.Mock()
    bot._mm_watch = {"mm-tok"}
    bot._basket_token_map = {"arb-tok": ("basket", "ev1")}
    bot._chain_token_map = {"chain-tok": ("chain", ("s", "p"))}
    bot._sync_ws_watch([])
    watched = bot.ws.watch.call_args.args[0]
    assert {"mm-tok", "arb-tok", "chain-tok"} <= watched
    bot.ws = None
    bot.close()
