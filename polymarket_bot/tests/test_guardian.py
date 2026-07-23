"""Position guardian: adverse-move detection on expensive legs."""

from unittest import mock

from polymarket_bot.config import BotConfig
from polymarket_bot.guardian import PositionGuardian
from polymarket_bot.models import Position


def pos(avg_price=0.97, size=100, token="t1"):
    return Position(token_id=token, market_id="m", question="Will X win?",
                    outcome="No", category="other", size=size, avg_price=avg_price)


def guard(**over):
    cfg = BotConfig()
    for k, v in over.items():
        setattr(cfg.guardian, k, v)
    return PositionGuardian(cfg)


def test_fires_on_adverse_drop():
    v = guard().check(pos(avg_price=0.97), mark=0.85)     # NO fell 0.12 >= 0.10
    assert v is not None
    assert abs(v.drop - 0.12) < 1e-9
    assert abs(v.tail_entry - 0.03) < 1e-9                # sold a 3% tail
    assert abs(v.tail_now - 0.15) < 1e-9                  # now a 15% tail — rising


def test_quiet_within_threshold():
    assert guard().check(pos(avg_price=0.97), mark=0.90) is None   # only −0.07


def test_ignores_cheap_legs():
    """A longshot YES bought at 0.03 dropping is EXPECTED, not a guardian event."""
    assert guard().check(pos(avg_price=0.03), mark=0.005) is None


def test_disabled_returns_none():
    assert guard(enabled=False).check(pos(), mark=0.5) is None


def test_zero_mark_ignored():
    assert guard().check(pos(), mark=0.0) is None


def test_describe_mentions_the_move():
    v = guard().check(pos(avg_price=0.97), mark=0.80)
    text = v.describe("Will Candidate Z win the election?")
    assert "GUARDIAN" in text and "0.970" in text and "0.800" in text


# --- bot wiring: cycle pass alerts and (opt-in) reduces ---

def test_cycle_pass_alerts_but_does_not_reduce_by_default(tmp_path):
    from .test_hardening import make_bot
    bot = make_bot(tmp_path)
    from polymarket_bot.models import Position as P
    p = P(token_id="t1", market_id="m", question="Q?", outcome="No",
          category="other", size=100, avg_price=0.97)
    with mock.patch.object(bot.ledger, "open_positions", return_value=[p]), \
            mock.patch("polymarket_bot.main.alert") as a, \
            mock.patch.object(bot, "_reduce_one") as reduce:
        bot._guardian_pass({"t1": 0.80})           # −0.17, well past threshold
    a.assert_called_once()
    reduce.assert_not_called()                     # auto_reduce off by default
    bot.close()


def test_cycle_pass_reduces_when_enabled(tmp_path):
    from .test_hardening import make_bot
    bot = make_bot(tmp_path)
    bot.cfg.guardian.auto_reduce = True
    from polymarket_bot.models import Position as P
    p = P(token_id="t1", market_id="m", question="Q?", outcome="No",
          category="other", size=100, avg_price=0.97)
    with mock.patch.object(bot.ledger, "open_positions", return_value=[p]), \
            mock.patch("polymarket_bot.main.alert"), \
            mock.patch.object(bot, "_reduce_one", return_value=True) as reduce:
        bot._guardian_pass({"t1": 0.80})
    reduce.assert_called_once()
    bot.close()


def test_guardian_alerts_survive_a_halt(tmp_path):
    """Disasters correlate: the warning must fire even when the kill-switch has
    halted trading — only the auto-trim is gated."""
    from .test_hardening import make_bot
    bot = make_bot(tmp_path)
    bot.cfg.guardian.auto_reduce = True
    bot.killswitch.halted = True
    from polymarket_bot.models import Position as P
    p = P(token_id="t1", market_id="m", question="Q?", outcome="No",
          category="other", size=100, avg_price=0.97)
    with mock.patch.object(bot.ledger, "open_positions", return_value=[p]), \
            mock.patch("polymarket_bot.main.alert") as a, \
            mock.patch.object(bot, "_reduce_one") as reduce:
        bot._guardian_pass({"t1": 0.80})
    a.assert_called_once()                 # the human still hears about it
    reduce.assert_not_called()             # but no order is placed while halted
    bot.close()


def test_on_tick_guardian_warns_during_halt(tmp_path):
    from .test_hardening import make_bot
    from polymarket_bot.ws_feed import TopOfBook
    bot = make_bot(tmp_path)
    bot.killswitch.halted = True
    from polymarket_bot.models import Position as P
    bot._positions_by_token = {"t1": P(
        token_id="t1", market_id="m", question="Q?", outcome="No",
        category="other", size=100, avg_price=0.97)}
    with mock.patch("polymarket_bot.main.alert") as a, \
            mock.patch.object(bot, "_exit_one") as exit_one:
        bot.on_tick("t1", TopOfBook(bid=0.80, bid_size=10, ask=0.82,
                                    ask_size=10, ts=1.0))
    a.assert_called_once()                 # warning delivered
    exit_one.assert_not_called()           # trading still fully blocked
    bot.close()
