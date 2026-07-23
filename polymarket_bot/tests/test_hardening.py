"""Hardening regressions: WS hot path, guard semantics, breaker, pause sources."""

from unittest import mock

from polymarket_bot.config import BotConfig
from polymarket_bot.risk import KillSwitch
from polymarket_bot.risk2 import StrategyCircuitBreaker
from polymarket_bot.ws_feed import TopOfBook


def make_bot(tmp_path):
    from polymarket_bot.main import Bot
    cfg = BotConfig()
    cfg.runtime.db_path = str(tmp_path / "l.sqlite")
    cfg.runtime.log_path = str(tmp_path / "b.jsonl")
    cfg.runtime.llm_cache_path = str(tmp_path / "c.json")
    cfg.ticks.db_path = str(tmp_path / "ticks.sqlite")   # never the repo data dir
    cfg.ws.enabled = False
    return Bot(cfg, "paper")


# --- P0-1: on_tick must not touch the DB ---

def test_on_tick_never_hits_ledger(tmp_path):
    bot = make_bot(tmp_path)
    tick = TopOfBook(bid=0.55, bid_size=100, ask=0.57, ask_size=100, ts=1.0)
    with mock.patch.object(bot.ledger, "open_positions",
                           wraps=bot.ledger.open_positions) as scans:
        for _ in range(100):
            bot.on_tick("unknown-token", tick)
    scans.assert_not_called()
    bot.close()


def test_on_tick_exits_from_cache(tmp_path):
    bot = make_bot(tmp_path)
    from polymarket_bot.models import Position
    pos = Position(token_id="t1", market_id="m", question="Q?", outcome="Yes",
                   category="other", size=100, avg_price=0.01)
    bot._positions_by_token = {"t1": pos}
    with mock.patch.object(bot, "_exit_one", return_value=True) as exit_one:
        bot.on_tick("t1", TopOfBook(bid=0.20, bid_size=10, ask=0.22,
                                    ask_size=10, ts=1.0))
    exit_one.assert_called_once()
    assert "t1" not in bot._positions_by_token   # exited -> dropped from cache
    bot.close()


def test_on_tick_respects_observe_only(tmp_path):
    bot = make_bot(tmp_path)
    bot._observe_only = True
    with mock.patch.object(bot.mm, "react_to_tick") as react:
        bot.on_tick("x", TopOfBook(bid=0.5, bid_size=1, ask=0.52, ask_size=1, ts=1.0))
    react.assert_not_called()
    bot.close()


# --- P0-1c: alert() is non-blocking (queued) ---

def test_alert_enqueues_and_worker_sends(monkeypatch):
    import polymarket_bot.monitor as mon
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    with mock.patch.object(mon.httpx, "post") as post:
        post.return_value = mock.Mock(status_code=200)
        assert mon.alert("hello") is True     # returns instantly (queued)
        mon._queue.join()                     # worker drains it
    post.assert_called_once()
    assert post.call_args.kwargs["json"]["text"] == "hello"


def test_alert_false_without_env(monkeypatch):
    import polymarket_bot.monitor as mon
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert mon.alert("x") is False


# --- P0-3: breaker must not heal on flat checks ---

def test_breaker_stays_tripped_through_quiet_period():
    b = StrategyCircuitBreaker(losing_streak=2)
    for pnl in (0.0, -10.0, -25.0):
        b.update({"fade": pnl})
    assert not b.allows("fade")
    for _ in range(10):                       # minutes of no trades (flat PnL)
        b.update({"fade": -25.0})
    assert not b.allows("fade")               # still disabled
    b.update({"fade": -20.0})                 # real recovery (+5 over window
    b.update({"fade": -15.0})                 #  of the last 2 deltas)
    assert b.allows("fade")


# --- P0-4: pause sources are independent ---

def test_data_pause_survives_ws_recovery(cfg, ledger):
    ks = KillSwitch(cfg, ledger, "paper", cancel_all=mock.Mock(), alert=mock.Mock())
    ks.trip_pause("market data anomaly", source="data")
    assert not ks.trading_allowed
    ks.on_ws_recovered()                      # a healthy WS clears only "ws"
    assert not ks.trading_allowed             # data pause still holds
    ks.resume_from_pause("data")
    assert ks.trading_allowed


def test_both_pause_sources_must_clear(cfg, ledger):
    ks = KillSwitch(cfg, ledger, "paper", cancel_all=mock.Mock(), alert=mock.Mock())
    ks.trip_pause("ws dead", source="ws")
    ks.trip_pause("bad data", source="data")
    ks.resume_from_pause("ws")
    assert ks.paused                          # data still pausing
    ks.resume_from_pause("data")
    assert ks.trading_allowed


def test_on_tick_exits_still_run_in_observe_only(tmp_path):
    """Observe-only freezes NEW risk (MM quoting) but must not freeze
    risk REDUCTION: take-profit exits keep working in a drawdown."""
    bot = make_bot(tmp_path)
    bot._observe_only = True
    from polymarket_bot.models import Position
    pos = Position(token_id="t1", market_id="m", question="Q?", outcome="Yes",
                   category="other", size=100, avg_price=0.01)
    bot._positions_by_token = {"t1": pos}
    with mock.patch.object(bot, "_exit_one", return_value=True) as exit_one, \
            mock.patch.object(bot.mm, "react_to_tick") as react:
        bot.on_tick("t1", TopOfBook(bid=0.20, bid_size=10, ask=0.22,
                                    ask_size=10, ts=1.0))
    exit_one.assert_called_once()     # exit allowed
    react.assert_not_called()         # quoting still frozen
    bot.close()


def test_on_tick_halt_blocks_everything(tmp_path):
    """A hard HALT (unlike observe-only) freezes exits too: state is unreliable."""
    bot = make_bot(tmp_path)
    bot.killswitch.halted = True
    from polymarket_bot.models import Position
    pos = Position(token_id="t1", market_id="m", question="Q?", outcome="Yes",
                   category="other", size=100, avg_price=0.01)
    bot._positions_by_token = {"t1": pos}
    with mock.patch.object(bot, "_exit_one") as exit_one:
        bot.on_tick("t1", TopOfBook(bid=0.20, bid_size=10, ask=0.22,
                                    ask_size=10, ts=1.0))
    exit_one.assert_not_called()
    bot.close()
