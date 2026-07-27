"""F-018/019: the Telegram update stream is an external payload, never fuzzed.

This is the only path where outside input reaches bot STATE — /pause and
/resume drive the shared kill-switch. The auth gate (chat id) lives in
`handle_update`, but `_advance_offset` runs BEFORE it on every update in the
batch, so that function has to survive payloads from anyone, not just the owner.

F-018 (the serious one): a single unreadable `update_id` wedged the control
channel FOREVER. Telegram replays every update until the offset moves past it,
so `int("abc")` raising meant the offset never advanced, the run loop swallowed
the exception, and the next poll returned the same poisoned batch — for ever.
Commands later in that batch were never dispatched, which makes /pause — the
operator's remote kill — permanently unreachable.

F-019: `handle_update` called `.get()` on whatever sat in `message`/`chat` and
`.strip()` on whatever sat in `text`, so a non-object raised AttributeError and
cost the rest of the batch.
"""

import os

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from polymarket_bot.telegram_control import TelegramControl

CHAT = "12345"


@pytest.fixture()
def tg(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT)
    return TelegramControl({"status": lambda: "ok", "pause": lambda: "paused"})


def _cmd(text: str, chat: str = CHAT, update_id: int = 1) -> dict:
    return {"update_id": update_id,
            "message": {"chat": {"id": chat}, "text": text}}


# --- F-018: the offset must always make forward progress ---

def test_unreadable_update_id_does_not_wedge_the_channel(tg):
    tg._offset = 0
    tg._advance_offset([{"update_id": "abc"},
                        _cmd("/status", update_id=11)])
    assert tg._offset == 12, "the readable id in the batch must still advance"


@pytest.mark.parametrize("bad", ["abc", None, {"x": 1}, [1], 1.5e400])
def test_advance_offset_never_raises(tg, bad):
    tg._offset = 5
    tg._advance_offset([{"update_id": bad}])       # must not raise
    assert tg._offset >= 5


def test_batch_with_no_readable_id_still_advances(tg):
    """Replaying an unreadable batch for ever is strictly worse than losing it."""
    tg._offset = 7
    tg._advance_offset([{"update_id": None}, {"update_id": "x"}])
    assert tg._offset == 8


def test_control_channel_survives_a_poisoned_batch_end_to_end(tg):
    """The exact wedge, driven through the run-loop body."""
    served: list[int] = []
    tg._handlers = {"status": lambda: served.append(1) or "ok"}
    batch = [{"update_id": "abc"}, _cmd("/status", update_id=11)]
    for _ in range(3):
        updates = batch if tg._offset == 0 else []
        tg._advance_offset(updates)
        for u in updates:
            tg.handle_update(u)
    assert tg._offset == 12
    assert len(served) == 1, "the good command in the batch must be dispatched"


# --- F-019: any shape is a non-answer, never an exception ---

@pytest.mark.parametrize("update", [
    {"message": {"chat": [1], "text": "/status"}},      # chat not an object
    {"message": {"chat": "12345", "text": "/status"}},
    {"message": "hello"},                                # message not an object
    {"message": {"chat": {"id": CHAT}, "text": 123}},    # text not a string
    {"message": {"chat": {"id": CHAT}, "text": None}},
    {"message": {}},
    {},
    "not even a dict",
    None,
])
def test_malformed_updates_are_ignored_not_raised(tg, update):
    assert tg.handle_update(update) is None


# --- the auth gate still holds ---

def test_only_the_configured_chat_is_honoured(tg):
    assert tg.handle_update(_cmd("/status", chat="99999")) is None
    assert tg.handle_update(_cmd("/status")) == "ok"


def test_state_changing_command_still_requires_the_owner(tg):
    """/pause drives the kill-switch — the one command that changes state."""
    assert tg.handle_update(_cmd("/pause", chat="99999")) is None
    assert tg.handle_update(_cmd("/pause")) == "paused"


def test_unknown_command_lists_the_known_ones(tg):
    reply = tg.handle_update(_cmd("/nope"))
    assert reply is not None and "/status" in reply


def test_bare_slash_is_not_a_command(tg):
    assert tg.handle_update(_cmd("/")) is None


def test_bot_suffix_is_stripped(tg):
    assert tg.handle_update(_cmd("/status@mybot extra")) == "ok"


def test_a_failing_handler_does_not_kill_the_poller(tg):
    def boom():
        raise RuntimeError("handler exploded")
    tg._handlers = {"status": boom}
    reply = tg.handle_update(_cmd("/status"))
    assert reply is not None and "failed" in reply


# --- property: NO update shape may raise, from any sender ---

_json = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=True) | st.text(),
    lambda children: st.lists(children, max_size=3)
    | st.dictionaries(st.text(max_size=8), children, max_size=4),
    max_leaves=8,
)


@given(update=_json)
@settings(max_examples=300, deadline=None)
def test_property_no_payload_can_raise(update):
    """No fixtures on purpose: pytest fixtures are function-scoped and would be
    REUSED across Hypothesis examples, leaking state between them (and
    `monkeypatch` is itself such a fixture)."""
    bot = TelegramControl({"status": lambda: "ok"})
    bot._chat_id = CHAT               # set directly; no environment involved
    bot._token = "tok"
    bot.handle_update(update)         # returning anything is fine; raising is not
    bot._advance_offset([update] if isinstance(update, dict) else [])
