"""Seam tests: the class of defect this suite has repeatedly failed to catch.

Three bugs in a row (F-020, F-026, F-027) were found by a live paper run while
500+ tests, 30/30 mutants and 73% branch coverage were green. None of them was a
wrong rule. All three lived at a SEAM:

  F-020  between a rule and its RANGE — take_profit_multiple=7.0 is correct
         arithmetic that no reachable price can satisfy for an entry near 1.0.
  F-026  between an OLD BOOK and a NEW GATE — the take was written for positions
         the entry gate admits, then met twenty legs the gate would refuse, and
         liquidated three of them at or below cost.
  F-027  between the component that OPENS and the component that CLOSES — two
         ledger aggregations disagreed about who owned a token, and the losing one
         fed the capital allocator.

Unit tests missed them because a unit test asks "given these inputs, is the rule
right?" and the inputs are chosen by whoever wrote the rule — so they are the
inputs on which it works. Mutation testing missed them because a mutant asks
"would the suite notice this code changing?", not "is there a scenario nobody
wrote a test for?".

The tests here are deliberately written as INVARIANTS over generated ranges and as
whole-lifecycle flows, not as examples. They must keep holding when the thresholds
are retuned, which is the only way they can outlive the bug that motivated them.
"""

from __future__ import annotations

import ast
import pathlib
from unittest import mock

import pytest

from polymarket_bot.fade import fade_exit_plan, payoff_ratio
from polymarket_bot.models import Position, simple_estimate
from polymarket_bot.portfolio import Portfolio

from .conftest import make_market

# A coarse grid on purpose: dense enough to cross every threshold in the fade's
# operating range, sparse enough that a failure names one readable case.
MARKS = [round(0.700 + 0.005 * i, 4) for i in range(61)] + \
        [round(0.980 + 0.001 * i, 4) for i in range(20)]
ENTRIES = [0.90, 0.93, 0.944, 0.95, 0.964, 0.972, 0.976, 0.98, 0.982,
           0.985, 0.986, 0.989, 0.99, 0.995, 0.999]


def pos(entry: float, size: float = 100.0, strategy: str = "fade") -> Position:
    return Position(token_id="seam-no", market_id="seam", question="Will X?",
                    outcome="No", category="other", size=size, avg_price=entry,
                    strategy=strategy)


# =====================================================================
# Seam 1 — an old book meets a new gate  (would have caught F-026)
# =====================================================================

def test_a_take_never_realizes_a_loss_anywhere_on_the_grid(cfg):
    """INVARIANT: only a stop may exit below cost. A take is a profit or nothing.

    This is the property the F-026 rule violated. It held for every entry the
    gate admits and failed for every entry it does not — which is precisely the
    legacy book, the one case no unit test constructed.
    """
    for entry in ENTRIES:
        for mark in MARKS:
            plan = fade_exit_plan(pos(entry), mark, cfg.fade)
            if plan is None or plan.reason == "tail-stop":
                continue
            assert mark > entry, (
                f"take '{plan.reason}' fired at mark {mark} on an entry of "
                f"{entry}: that realizes {(mark - entry) * 100:+.2f}$ per 100 sh")


def test_a_stop_never_fires_in_profit(cfg):
    """The mirror invariant: a stop is loss control, never a disguised take.

    Holds for any tail_stop_multiple > 1, since (1-mark) >= k*(1-entry) with k>1
    forces mark < entry. Asserted rather than assumed, because a config that set
    the multiple below 1 would silently turn the stop into a take with a
    slippage-accepting min_price.
    """
    assert cfg.fade.tail_stop_multiple > 1.0
    for entry in ENTRIES:
        for mark in MARKS:
            plan = fade_exit_plan(pos(entry), mark, cfg.fade)
            if plan is not None and plan.reason == "tail-stop":
                assert mark < entry, (entry, mark)


def test_every_admissible_entry_keeps_a_band_where_neither_rule_fires(cfg):
    """INVARIANT: an entry the gate admits must have room to be held.

    The F-026 take used the SAME threshold as the entry gate, so a position
    opened at the boundary (1/1.02 = 0.98039) was liquidated on the next upward
    tick — buy, sell one tick later, pay the spread twice. Any future retune that
    makes the two thresholds meet again fails here.
    """
    admissible = [e for e in ENTRIES
                  if payoff_ratio(e) >= cfg.fade.min_payoff_ratio]
    assert admissible, "the gate admits nothing — grid or threshold is wrong"
    for entry in admissible:
        held = [m for m in MARKS
                if m > entry and fade_exit_plan(pos(entry), m, cfg.fade) is None]
        assert held, (
            f"an entry at {entry} passes the gate but every mark above it exits: "
            "the entry and take thresholds have collided again")


def test_positions_the_gate_would_now_refuse_are_not_force_sold_at_a_loss(cfg):
    """The legacy book, named explicitly.

    These are real entries from the run that exposed F-026. Under the new gate
    none of them is enterable, so a rule keyed to shape alone would fire on all of
    them immediately, at whatever the mark happens to be.
    """
    legacy = [(0.986, 0.982), (0.989, 0.989), (0.982, 0.982),
              (0.99, 0.99), (0.995, 0.99)]
    for entry, mark in legacy:
        assert payoff_ratio(entry) < cfg.fade.min_payoff_ratio    # not enterable
        plan = fade_exit_plan(pos(entry), mark, cfg.fade)
        if plan is not None:
            assert plan.reason == "tail-stop" and mark < entry, (
                f"{plan.reason} at {mark} on a legacy leg bought at {entry}")


# =====================================================================
# Seam 2 — a rule must be reachable at all  (would have caught F-020)
# =====================================================================

@pytest.mark.parametrize("strategy,entry", [("fade", 0.976), ("fade", 0.95),
                                           ("longshot", 0.03), ("longshot", 0.01)])
def test_every_open_position_has_some_price_at_which_it_exits(tmp_path, strategy, entry):
    """INVARIANT: for every position there EXISTS a mark in (0,1) that exits it.

    F-020 was not a wrong threshold, it was an unreachable one:
    `mark / 0.976 >= 7.0` needs price 6.83 and the venue stops at 1.0, so
    exit_plan returned None for every mark for the whole life of the position.
    A test of the rule cannot see that; only asking "is this rule satisfiable
    inside the domain?" can.
    """
    from .test_hardening import make_bot
    bot = make_bot(tmp_path)
    try:
        bot.executor = mock.Mock()
        bot.executor.execute_sell.return_value = mock.Mock(
            status="filled", avg_price=0.5, filled_size=1.0)
        p = pos(entry, strategy=strategy)
        exits = [m for m in [round(0.001 + 0.001 * i, 4) for i in range(999)]
                 if bot._exit_one(p, m)]
        assert exits, (
            f"a {strategy} position bought at {entry} cannot be exited at ANY "
            "price in (0,1) — its only exit is resolution, at full notional")
    finally:
        bot.close()


def test_the_generic_multiple_is_the_unreachable_one(cfg, ledger):
    """Pins WHY the previous test is needed, so the reason survives a refactor."""
    portfolio = Portfolio(cfg, ledger, "paper")
    entry = 0.976
    required = entry * cfg.portfolio.take_profit_multiple
    assert required > 1.0, (
        "take_profit_multiple is now reachable for a fade entry; if that was "
        "deliberate, this test should be deleted along with the reason")
    assert all(portfolio.exit_plan(pos(entry), m) is None
               for m in [0.98, 0.99, 0.999])


# =====================================================================
# Seam 3 — who OPENED vs who CLOSED  (would have caught F-027)
# =====================================================================

def _round_trip(ledger, strategy: str, *, entry=0.976, exit_price=0.94,
                closer="longshot", token="rt"):
    """Open under `strategy`, close under `closer` (the mislabelling that bit us)."""
    m = make_market(id=f"m-{token}", clob_token_ids=[token, f"{token}-b"])
    ledger.record_trade(mode="paper", estimate=simple_estimate(m, 0, entry),
                        category="other", side="BUY", price=entry, size=100.0,
                        order_id=f"b-{token}", status="filled", strategy=strategy)
    ledger.record_trade(mode="paper", estimate=simple_estimate(m, 0, exit_price),
                        category="other", side="SELL", price=exit_price, size=100.0,
                        order_id=f"s-{token}", status="filled", strategy=closer)


@pytest.mark.parametrize("owner", ["fade", "mm", "arb", "resolution", "longshot"])
def test_every_aggregation_agrees_who_owns_a_token(cfg, ledger, owner):
    """INVARIANT: ownership is a property of the OPENING trade, everywhere.

    F-027 was two aggregations disagreeing: `open_positions` took the first row
    (correct, and tested), `realized_pnl_by_strategy` took the last (wrong, and
    untested). Parametrised over every strategy so adding a sixth cannot quietly
    reintroduce the split.
    """
    _round_trip(ledger, owner, closer="longshot", token=f"tok-{owner}")
    realized = ledger.realized_pnl_by_strategy("paper")
    assert set(realized) == {owner}, (
        f"a position opened by {owner} and closed by longshot was attributed to "
        f"{set(realized)}")

    # Half-closed: the still-open leg must name the same owner.
    _round_trip(ledger, owner, exit_price=0.94, closer="guardian",
                token=f"half-{owner}")
    ledger.record_trade(
        mode="paper",
        estimate=simple_estimate(
            make_market(id="m-extra", clob_token_ids=[f"open-{owner}", "b"]), 0, 0.90),
        category="other", side="BUY", price=0.90, size=10.0,
        order_id="extra", status="filled", strategy=owner)
    still_open = {p.strategy for p in ledger.open_positions("paper")}
    assert still_open == {owner}, still_open


def test_a_mislabelled_exit_cannot_reach_the_capital_allocator(cfg, ledger):
    """Whole lifecycle: entry -> generic exit -> attribution -> allocator weight.

    The consequence, not the dict. `realized_pnl_by_strategy` feeds
    `sharpe_allocation`, whose weights become `fade.size_scale`. With the old
    attribution the loser kept full size and an idle strategy got throttled.
    """
    from polymarket_bot.research import sharpe_allocation

    for i in range(3):
        _round_trip(ledger, "fade", entry=0.976, exit_price=0.93,
                    closer="longshot", token=f"loss{i}")
    pnl = ledger.realized_pnl_by_strategy("paper")
    assert pnl.get("fade", 0.0) < 0 and "longshot" not in pnl, pnl

    weights = sharpe_allocation({k: [v] for k, v in pnl.items()})
    assert weights.get("fade", 0.0) <= weights.get("longshot", 1.0), weights


def test_the_real_exit_path_labels_its_sell_correctly(tmp_path):
    """End-to-end through main, not through a hand-written ledger row.

    The seam is main._exit_one -> Executor.execute_sell -> record_trade. A test
    that writes the SELL row itself cannot see a caller that forgets the label.
    """
    from .test_hardening import make_bot
    bot = make_bot(tmp_path)
    try:
        m = make_market(id="e2e", clob_token_ids=["e2e-y", "e2e-n"])
        bot.ledger.record_trade(
            mode="paper", estimate=simple_estimate(m, 1, 0.976), category="other",
            side="BUY", price=0.976, size=100.0, order_id="b", status="filled",
            strategy="fade")
        position = next(p for p in bot.ledger.open_positions("paper"))
        assert position.strategy == "fade"

        bot.clob = mock.Mock()
        bot.executor._clob = mock.Mock()
        bot.executor._clob.order_book.return_value = None
        assert bot._exit_one(position, 0.92, from_ws=True) is True

        row = bot.ledger._conn.execute(
            "SELECT strategy FROM trades WHERE side='SELL'").fetchone()
        assert row is not None and row["strategy"] == "fade", dict(row or {})
        assert set(bot.ledger.realized_pnl_by_strategy("paper")) == {"fade"}
    finally:
        bot.close()


# =====================================================================
# Seam 4 — no money path may default its own identity
# =====================================================================

PROD_FILES = [f for f in sorted(pathlib.Path("polymarket_bot").glob("*.py"))]


def _money_calls():
    """(file, line, attr) for every production call that writes or closes a trade."""
    out = []
    for f in PROD_FILES:
        tree = ast.parse(f.read_text())
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)):
                continue
            attr = n.func.attr
            recv = ast.unparse(n.func.value)
            is_money = attr in ("record_trade", "execute_sell") or (
                attr == "execute" and "executor" in recv)
            if is_money and f.name != "ledger.py":       # ledger.py defines it
                out.append((f, n.lineno, attr,
                            {k.arg for k in n.keywords if k.arg}))
    return out


def test_every_money_path_states_which_strategy_it_belongs_to():
    """INVARIANT: no entry or exit may inherit its identity from a default.

    F-027's root cause was a silent default: `execute_sell` omitted `strategy`,
    `record_trade` filled in "longshot", and a fade book's loss was booked against
    a strategy that had not traded. `record_trade` now requires the argument, so
    the ledger boundary cannot be bypassed — this test covers the layer above it,
    where defaults still exist for the convenience of tests.

    Zero exemptions on purpose. A new exit path is exactly the change that
    reintroduces this bug, and it will fail here rather than in a paper report.
    """
    missing = [f"{f.name}:{line} .{attr}()"
               for f, line, attr, kws in _money_calls() if "strategy" not in kws]
    assert not missing, (
        "these call sites let the strategy label fall through to a default:\n  "
        + "\n  ".join(missing))


def test_the_ledger_refuses_a_trade_with_no_strategy(ledger):
    """The boundary itself: forgetting is a TypeError, not a plausible lie."""
    m = make_market(id="req", clob_token_ids=["req-y", "req-n"])
    with pytest.raises(TypeError):
        ledger.record_trade(                       # type: ignore[call-arg]
            mode="paper", estimate=simple_estimate(m, 0, 0.5), category="other",
            side="BUY", price=0.5, size=10.0, order_id="x", status="filled")


def test_the_money_call_scan_actually_finds_the_call_sites():
    """Guards the guard: a scan that silently matches nothing always passes."""
    calls = _money_calls()
    assert len(calls) >= 8, f"only {len(calls)} money paths found — scan is broken"
    assert {attr for _, _, attr, _ in calls} >= {"record_trade", "execute_sell"}
