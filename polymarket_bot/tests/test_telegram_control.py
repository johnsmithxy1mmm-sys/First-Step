"""Two-way Telegram control: command dispatch, chat gating, bot handlers."""

from unittest import mock

from polymarket_bot.telegram_control import TelegramControl


def make_control(monkeypatch, handlers=None, chat="42"):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", chat)
    return TelegramControl(handlers or {"status": lambda: "OK"})


def msg(text, chat_id=42):
    return {"update_id": 1, "message": {"chat": {"id": chat_id}, "text": text}}


def test_dispatches_known_command(monkeypatch):
    c = make_control(monkeypatch, {"status": lambda: "equity $5000"})
    assert c.handle_update(msg("/status")) == "equity $5000"


def test_ignores_other_chats(monkeypatch):
    c = make_control(monkeypatch, {"status": lambda: "OK"}, chat="42")
    assert c.handle_update(msg("/status", chat_id=999)) is None   # not the owner


def test_strips_botname_suffix_and_args(monkeypatch):
    c = make_control(monkeypatch, {"pnl": lambda: "pnl here"})
    assert c.handle_update(msg("/pnl@MyBot extra args")) == "pnl here"


def test_unknown_command_lists_options(monkeypatch):
    c = make_control(monkeypatch, {"status": lambda: "x", "pnl": lambda: "y"})
    reply = c.handle_update(msg("/nope"))
    assert "/status" in reply and "/pnl" in reply


def test_non_command_text_ignored(monkeypatch):
    c = make_control(monkeypatch, {"status": lambda: "x"})
    assert c.handle_update(msg("just chatting")) is None


def test_handler_exception_is_caught(monkeypatch):
    def boom():
        raise RuntimeError("db down")
    c = make_control(monkeypatch, {"status": boom})
    assert "failed" in c.handle_update(msg("/status"))


def test_offset_advances_past_seen_updates(monkeypatch):
    c = make_control(monkeypatch)
    c._advance_offset([{"update_id": 5}, {"update_id": 8}])
    assert c._offset == 9


def test_disabled_without_env(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert TelegramControl({}).enabled is False


# --- bot-wired handlers ---

def test_bot_handlers_status_and_pause_resume(tmp_path, monkeypatch):
    from .test_hardening import make_bot
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    bot = make_bot(tmp_path)
    h = bot._telegram_handlers()
    assert "mode paper" in h["status"]()
    # pause makes trading disallowed; resume restores it.
    assert "paused" in h["pause"]().lower()
    assert bot.killswitch.trading_allowed is False
    assert "resumed" in h["resume"]().lower()
    assert bot.killswitch.trading_allowed is True
    assert "no open positions" in h["positions"]()
    assert "Realized PnL" in h["pnl"]()
    bot.close()


def test_bot_control_gated_on_chat_id(tmp_path, monkeypatch):
    from .test_hardening import make_bot
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    bot = make_bot(tmp_path)
    # A /pause from the wrong chat must NOT touch the kill-switch.
    assert bot.tg_control.handle_update(msg("/pause", chat_id=1)) is None
    assert bot.killswitch.trading_allowed is True
    bot.close()
