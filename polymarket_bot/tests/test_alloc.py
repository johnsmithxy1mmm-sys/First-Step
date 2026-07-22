"""StrategyAllocator: bounded, smoothed capital tilts from Sharpe weights."""

from polymarket_bot.alloc import StrategyAllocator
from polymarket_bot.config import BotConfig


def make_alloc(**over):
    cfg = BotConfig()
    for k, v in over.items():
        setattr(cfg.allocator, k, v)
    return StrategyAllocator(cfg)


def test_identity_before_any_update():
    assert make_alloc().factor("mm") == 1.0


def test_tilts_winner_up_loser_down_within_corridor():
    a = make_alloc(smoothing=1.0)                       # no EMA lag for the assert
    a.update({"fade": 0.9, "mm": 0.1})                  # equal weight = 0.5
    assert a.factor("fade") > 1.0 and a.factor("mm") < 1.0
    assert a.factor("fade") <= 1.3 and a.factor("mm") >= 0.7   # corridor holds


def test_corridor_clamps_extremes():
    a = make_alloc(smoothing=1.0, floor=0.5, ceil=1.5)
    a.update({"winner": 0.99, "loser": 0.01})
    assert a.factor("winner") == 1.5                    # target 1.98 -> clamped
    assert a.factor("loser") == 0.5


def test_smoothing_moves_gradually():
    a = make_alloc(smoothing=0.5, ceil=2.0)
    a.update({"only": 1.0})                             # target = 1/1 = 1.0
    a.update({"only": 1.0})
    assert a.factor("only") == 1.0
    # A jump target is approached, not snapped to.
    b = make_alloc(smoothing=0.5, ceil=2.0)
    b.update({"a": 0.8, "b": 0.2})                      # a target 1.6
    first = b.factor("a")
    b.update({"a": 0.8, "b": 0.2})
    assert first < b.factor("a") <= 1.6


def test_disabled_is_always_one():
    a = make_alloc(enabled=False)
    a.update({"fade": 0.9, "mm": 0.1})
    assert a.factor("fade") == 1.0 and a.factor("mm") == 1.0


def test_reports_only_meaningful_changes():
    a = make_alloc(smoothing=1.0)
    changed = a.update({"x": 0.5, "y": 0.5})            # both target 1.0 = no move
    assert changed == {}
