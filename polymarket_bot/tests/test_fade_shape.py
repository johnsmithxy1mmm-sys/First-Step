"""Payoff shape, IRR floor and the fade's real exit.

These cover the four defects a live paper report exposed. The book it produced
was 20 legs bought at 0.944-0.990: gross $1,891, worst case $846, and a maximum
possible upside of $37.27 — an expected loss of $36.34 at the same prices, i.e.
EV zero by construction, with no rule anywhere that could refuse or unwind it.

  * every existing filter looks at price band, horizon or edge; none looks at the
    SHAPE of the payoff, so a leg needing 99 wins to repay one loss passed;
  * `_irr_score` computed edge-per-day but only RANKED by it, and ranking is
    inert while the portfolio caps are slack, so a leg earning 0.4c over 90 days
    was entered next to one earning 2.2c over 4;
  * `portfolio.exit_plan` triggers on mark/entry >= 7.0, which for an entry at
    0.976 needs price 6.83. Prices stop at 1.0, so it returned None for the whole
    life of every fade position — those legs had exactly one exit, resolution, at
    the full notional.
"""

from datetime import datetime, timedelta, timezone

import pytest

from polymarket_bot.fade import fade_exit_plan, payoff_ratio
from polymarket_bot.models import Position
from polymarket_bot.portfolio import Portfolio

from .test_fade import make_estimate, make_fade


def fade_position(entry: float, size: float = 128.0,
                  strategy: str = "fade") -> Position:
    return Position(token_id="f-no", market_id="f1", question="Will X happen?",
                    outcome="No", category="other", size=size, avg_price=entry,
                    strategy=strategy)


# --- A2: payoff shape at entry ---

def test_payoff_ratio_is_wins_per_loss_inverted():
    assert payoff_ratio(0.99) == pytest.approx(0.010101, abs=1e-6)   # 99 wins
    assert payoff_ratio(0.976) == pytest.approx(0.024590, abs=1e-6)  # ~41 wins
    assert payoff_ratio(0.50) == pytest.approx(1.0)
    # Untradable/degenerate inputs are shape-less, not infinitely good.
    for bad in (0.0, 1.0, -0.5, 1.5):
        assert payoff_ratio(bad) == 0.0


def test_a_one_cent_tail_is_refused_on_shape(cfg, ledger):
    """Entry at 0.99: 99 wins to repay one loss. Refused before any edge math."""
    fade, _ = make_fade(cfg, ledger)
    reason = fade.reject_reason(make_estimate(p_mkt=0.01, p_est=None))
    assert reason is not None and "payoff ratio" in reason, reason
    assert fade.plan(make_estimate(p_mkt=0.01, p_est=None)) is None


def test_shape_gate_precedes_the_edge_gate(cfg, ledger):
    """The reported reason must name the real disqualifier.

    A 1c tail also fails min_edge_after_fees. If the edge check ran first,
    `--mode diagnose` would blame the edge and an operator would 'fix' it by
    lowering min_edge_after_fees — which cannot make the shape acceptable.
    """
    fade, _ = make_fade(cfg, ledger)
    cfg.fade.min_edge_after_fees = 0.0        # remove the other reason entirely
    reason = fade.reject_reason(make_estimate(p_mkt=0.01, p_est=None))
    assert reason is not None and "payoff ratio" in reason, reason


def test_a_fat_enough_tail_still_passes(cfg, ledger):
    """The gate bounds shape; it must not close the strategy."""
    fade, _ = make_fade(cfg, ledger)
    assert fade.reject_reason(make_estimate(p_mkt=0.05, p_est=None)) is None


def test_shape_gate_is_configurable_off(cfg, ledger):
    fade, _ = make_fade(cfg, ledger)
    # Clear the OTHER gates a 1c tail also trips, so this asserts the shape gate
    # alone: without min_payoff_ratio it fails on edge, then on edge-per-day.
    cfg.fade.min_payoff_ratio = 0.0
    cfg.fade.min_edge_after_fees = 0.0
    cfg.fade.min_edge_per_day = 0.0
    assert fade.reject_reason(make_estimate(p_mkt=0.01, p_est=None)) is None


# --- A3: the IRR floor rejects, not just ranks ---

def _dated(days: float, p_mkt: float = 0.05):
    return make_estimate(p_mkt=p_mkt,
                         end_date=datetime.now(timezone.utc) + timedelta(days=days))


def test_far_dated_leg_is_refused_on_edge_per_day(cfg, ledger):
    """80 days inside max_days_to_resolution, but 0.015/80 = 1.9e-4 per day."""
    fade, _ = make_fade(cfg, ledger)
    cfg.fade.max_days_to_resolution = 90.0
    reason = fade.reject_reason(_dated(80))
    assert reason is not None and "edge per day" in reason, reason


def test_the_same_edge_near_term_passes(cfg, ledger):
    """Identical edge, 5 days instead of 80 — the only difference is IRR."""
    fade, _ = make_fade(cfg, ledger)
    cfg.fade.max_days_to_resolution = 90.0
    assert fade.reject_reason(_dated(5)) is None


def test_irr_floor_can_be_disabled(cfg, ledger):
    fade, _ = make_fade(cfg, ledger)
    cfg.fade.max_days_to_resolution = 90.0
    cfg.fade.min_edge_per_day = 0.0
    assert fade.reject_reason(_dated(80)) is None


def test_ranking_alone_did_not_gate(cfg, ledger):
    """Documents WHY the floor was needed: the score orders, it never refuses.

    _irr_score ranks the far leg below the near one, and with the floor off both
    are still fadeable — so on a slack cap every far-dated leg got capital.
    """
    fade, _ = make_fade(cfg, ledger)
    cfg.fade.max_days_to_resolution = 90.0
    cfg.fade.min_edge_per_day = 0.0
    near, far = _dated(5), _dated(80)
    assert fade._irr_score(near) > fade._irr_score(far)
    assert fade.reject_reason(near) is None and fade.reject_reason(far) is None


# --- A1: the exit that could never fire ---

def test_generic_take_profit_is_unreachable_for_a_fade(cfg, ledger):
    """The root defect, pinned: not "rarely fires" — CANNOT fire, ever.

    Even at 0.999, a hair below the maximum price the venue allows, the multiple
    is 1.02 against a 7.0 threshold.
    """
    portfolio = Portfolio(cfg, ledger, "paper")
    pos = fade_position(0.976)
    for mark in (0.98, 0.99, 0.995, 0.999):
        assert portfolio.exit_plan(pos, mark) is None
    assert 0.999 / 0.976 < cfg.portfolio.take_profit_multiple


def test_tail_stop_fires_when_the_implied_tail_triples(cfg, ledger):
    """Entered at a 2.4% tail; at 8% the position is cut instead of ridden."""
    plan = fade_exit_plan(fade_position(0.976), 0.92, cfg.fade)
    assert plan is not None
    assert plan.reason == "tail-stop"
    assert plan.size == 128.0
    assert plan.min_price < 0.92          # accepts slippage to actually get out


def test_tail_stop_stays_quiet_below_the_multiple(cfg, ledger):
    """Tail 2.4% -> 4.0% is under 3x: still held, no churn."""
    assert fade_exit_plan(fade_position(0.976), 0.96, cfg.fade) is None


def test_the_stop_caps_the_loss_far_below_the_notional(cfg, ledger):
    """The point of the rule, in money.

    Held to resolution a losing leg costs the whole entry. Cut at 3x the tail it
    costs the price move — here about 2.2x the premium collected rather than 40x.
    """
    pos = fade_position(0.976)
    plan = fade_exit_plan(pos, 0.92, cfg.fade)
    assert plan is not None
    stopped_loss = (pos.avg_price - 0.92) * pos.size
    held_loss = pos.avg_price * pos.size
    assert stopped_loss < held_loss / 10


def test_payoff_exhausted_take_fires_near_one(cfg, ledger):
    """At 0.995 the leg risks 99.5c to earn 0.5c — a shape we would refuse."""
    plan = fade_exit_plan(fade_position(0.976), 0.995, cfg.fade)
    assert plan is not None
    assert plan.reason == "payoff-exhausted"
    assert plan.min_price == pytest.approx(round(0.995 * 0.99, 4))


def test_take_and_entry_gate_use_one_standard(cfg, ledger):
    """A mark we would not enter at is a mark we do not hold at.

    Same threshold on both sides, so the two rules cannot disagree.
    """
    cfg.fade.min_payoff_ratio = 0.02
    mark = 0.995
    assert payoff_ratio(mark) < cfg.fade.min_payoff_ratio
    assert fade_exit_plan(fade_position(0.976), mark, cfg.fade) is not None
    # And a mark whose remaining shape is still acceptable is held.
    assert payoff_ratio(0.97) > cfg.fade.min_payoff_ratio
    assert fade_exit_plan(fade_position(0.95), 0.97, cfg.fade) is None


def test_exit_refuses_a_cheap_leg(cfg, ledger):
    """Guards the direction of the arithmetic.

    On a longshot bought at 0.03 the 'tail' is 0.97 and tripling it is
    meaningless — routing one through tail logic would fire on noise.
    """
    assert fade_exit_plan(fade_position(0.03), 0.02, cfg.fade) is None


def test_partial_exit_fraction_is_honored(cfg, ledger):
    cfg.fade.exit_fraction = 0.5
    plan = fade_exit_plan(fade_position(0.976, size=100), 0.92, cfg.fade)
    assert plan is not None and plan.size == 50.0


def test_both_rules_can_be_disabled(cfg, ledger):
    cfg.fade.tail_stop_multiple = 0.0
    cfg.fade.early_take_enabled = False
    assert fade_exit_plan(fade_position(0.976), 0.92, cfg.fade) is None
    assert fade_exit_plan(fade_position(0.976), 0.995, cfg.fade) is None


def test_nonsense_marks_do_not_trigger_a_sale(cfg, ledger):
    for mark in (0.0, 1.0, -1.0, 2.0):
        assert fade_exit_plan(fade_position(0.976), mark, cfg.fade) is None


# --- A1 wiring: the rule reaches the position through main ---

def test_exit_one_routes_a_fade_position_to_the_fade_rule(tmp_path):
    """Integration: a real fade leg at a stopped-out mark actually sells.

    Without the routing this is a silent no-op — which is exactly what the live
    paper run was doing on all 20 positions.
    """
    from unittest import mock

    from .test_hardening import make_bot
    bot = make_bot(tmp_path)
    try:
        bot.executor = mock.Mock()
        bot.executor.execute_sell.return_value = mock.Mock(
            status="filled", avg_price=0.92, filled_size=128.0)
        assert bot._exit_one(fade_position(0.976), 0.92) is True
        args = bot.executor.execute_sell.call_args.args
        assert args[1] == 128.0                     # size sold
    finally:
        bot.close()


def test_exit_one_leaves_a_longshot_on_the_generic_rule(tmp_path):
    """The new path must not change how longshots exit."""
    from unittest import mock

    from .test_hardening import make_bot
    bot = make_bot(tmp_path)
    try:
        bot.executor = mock.Mock()
        pos = fade_position(0.03, size=100, strategy="longshot")
        # 0.03 -> 0.06 is a 2x rise, below take_profit_multiple 7.0: hold.
        assert bot._exit_one(pos, 0.06) is False
        bot.executor.execute_sell.assert_not_called()
        # 0.03 -> 0.25 is 8.3x: the generic take-profit fires as it always did.
        bot.executor.execute_sell.return_value = mock.Mock(
            status="filled", avg_price=0.25, filled_size=60.0)
        assert bot._exit_one(pos, 0.25) is True
    finally:
        bot.close()
