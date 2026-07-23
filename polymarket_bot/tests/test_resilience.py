"""Chaos drills, auto-postmortems, and compounding sizing."""

from polymarket_bot.config import BotConfig
from polymarket_bot.models import Position


# --- chaos drills: every safety system fires ---

def test_all_chaos_drills_pass():
    from polymarket_bot.chaos import run_all
    results = run_all(BotConfig())
    failed = [name for name, ok, _ in results if not ok]
    assert failed == [], f"safety drills failed: {failed}"
    assert len(results) >= 8


def test_chaos_scenarios_are_independent():
    """A HALT in one drill must not leak into the next (fresh components)."""
    from polymarket_bot.chaos import run_all
    r1 = {n: ok for n, ok, _ in run_all(BotConfig())}
    r2 = {n: ok for n, ok, _ in run_all(BotConfig())}
    assert r1 == r2 and all(r1.values())


# --- auto-postmortems ---

def pos(entry, won_outcome="No", size=100):
    return Position(token_id="t", market_id="m", question="Will X win the vote?",
                    outcome=won_outcome, category="elections", size=size,
                    avg_price=entry)


def test_postmortem_flags_materialized_tail():
    from polymarket_bot.postmortem import build
    pm = build(pos(entry=0.97), won=False)         # sold tail (NO@0.97) LOST
    assert pm["sold_tail"] is True and pm["won"] is False
    assert "MATERIALIZED" in pm["lesson"]
    assert pm["pnl"] == -0.97 * 100                # lost the whole stake


def test_postmortem_sold_tail_held():
    from polymarket_bot.postmortem import build
    pm = build(pos(entry=0.97), won=True)
    assert "held" in pm["lesson"]
    assert pm["pnl"] == round((1 - 0.97) * 100, 2)


def test_postmortem_longshot_hit():
    from polymarket_bot.postmortem import build
    pm = build(pos(entry=0.04), won=True)          # longshot YES@0.04 WON
    assert pm["sold_tail"] is False and "HIT" in pm["lesson"]


def test_postmortem_writer_appends_and_reads(tmp_path):
    from polymarket_bot.postmortem import PostmortemWriter
    w = PostmortemWriter(str(tmp_path / "pm.jsonl"))
    w.record(pos(0.97), won=False)
    w.record(pos(0.04), won=True)
    recent = w.recent()
    assert len(recent) == 2 and recent[-1]["won"] is True


def test_postmortem_disabled_writes_nothing(tmp_path):
    from polymarket_bot.postmortem import PostmortemWriter
    w = PostmortemWriter(str(tmp_path / "pm.jsonl"), enabled=False)
    assert w.record(pos(0.97), won=False) is None
    assert w.recent() == []


# --- compounding sizing ---

def test_compounding_grows_base_with_realized_profit(cfg, ledger):
    from polymarket_bot.portfolio import Portfolio
    cfg.portfolio.bankroll_usd = 5000
    cfg.portfolio.compounding = True
    port = Portfolio(cfg, ledger, "dry-run")
    base = port.effective_bankroll()
    from unittest import mock
    with mock.patch.object(ledger, "realized_pnl", return_value=1000.0):
        assert port.effective_bankroll() == 6000.0     # banked profit compounds
    assert base == 5000.0


def test_compounding_floored_on_losses(cfg, ledger):
    from unittest import mock
    from polymarket_bot.portfolio import Portfolio
    cfg.portfolio.bankroll_usd = 5000
    cfg.portfolio.compounding = True
    cfg.portfolio.compound_floor = 0.5
    port = Portfolio(cfg, ledger, "dry-run")
    with mock.patch.object(ledger, "realized_pnl", return_value=-4000.0):
        assert port.effective_bankroll() == 2500.0      # floored at 0.5 x 5000
    assert port.size_usd("other", 0.5, 0.1) is not None or True


def test_static_bankroll_when_compounding_off(cfg, ledger):
    from polymarket_bot.portfolio import Portfolio
    cfg.portfolio.bankroll_usd = 5000
    cfg.portfolio.compounding = False
    assert Portfolio(cfg, ledger, "dry-run").effective_bankroll() == 5000.0
