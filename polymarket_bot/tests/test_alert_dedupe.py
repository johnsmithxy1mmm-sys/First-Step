"""Alert dedupe: repeating opportunities are throttled, one-offs never are."""

from unittest import mock

import polymarket_bot.monitor as mon


def _env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")


def setup_function():
    mon._reset_alert_state()


def test_keyed_repeat_suppressed_within_cooldown(monkeypatch):
    _env(monkeypatch)
    with mock.patch.object(mon, "_should_send", wraps=mon._should_send), \
            mock.patch("polymarket_bot.monitor.queue.Queue.put") as put:
        assert mon.alert("arb x", key="arb:e1", cooldown_sec=3600) is True
        assert mon.alert("arb x", key="arb:e1", cooldown_sec=3600) is False
    assert put.call_count == 1                      # only the first got queued


def test_material_edge_change_breaks_through(monkeypatch):
    _env(monkeypatch)
    with mock.patch("polymarket_bot.monitor.queue.Queue.put") as put:
        mon.alert("arb", key="k", cooldown_sec=3600, value=0.05, min_change=0.02)
        mon.alert("arb", key="k", cooldown_sec=3600, value=0.055, min_change=0.02)  # +0.5pp
        mon.alert("arb", key="k", cooldown_sec=3600, value=0.09, min_change=0.02)   # +4pp
    assert put.call_count == 2                      # first send + the material jump


def test_unkeyed_events_never_suppressed(monkeypatch):
    _env(monkeypatch)
    with mock.patch("polymarket_bot.monitor.queue.Queue.put") as put:
        for _ in range(5):
            mon.alert("MM fill 0.44 x 50")          # a one-off, no key
    assert put.call_count == 5


def test_cooldown_expiry_realerts(monkeypatch):
    _env(monkeypatch)
    now = [1000.0]
    with mock.patch("polymarket_bot.monitor.time.time", side_effect=lambda: now[0]), \
            mock.patch("polymarket_bot.monitor.queue.Queue.put") as put:
        mon.alert("arb", key="k", cooldown_sec=100)
        now[0] = 1050.0
        mon.alert("arb", key="k", cooldown_sec=100)     # inside cooldown -> drop
        now[0] = 1200.0
        mon.alert("arb", key="k", cooldown_sec=100)     # past cooldown -> re-send
    assert put.call_count == 2


def test_different_keys_are_independent(monkeypatch):
    _env(monkeypatch)
    with mock.patch("polymarket_bot.monitor.queue.Queue.put") as put:
        mon.alert("a", key="arb:e1", cooldown_sec=3600)
        mon.alert("b", key="arb:e2", cooldown_sec=3600)
    assert put.call_count == 2


def test_should_send_pure_logic():
    mon._reset_alert_state()
    # No key -> always send.
    assert mon._should_send(None, 3600, None, None, now=1.0) is True
    # First time for a key -> send.
    assert mon._should_send("k", 3600, 0.05, 0.02, now=1.0) is True
    # Repeat, no material change -> drop.
    assert mon._should_send("k", 3600, 0.051, 0.02, now=2.0) is False
    # Repeat, material change -> send.
    assert mon._should_send("k", 3600, 0.08, 0.02, now=3.0) is True
