"""Capacity & learning report: persistence math and readiness accounting."""

from polymarket_bot.capacity import (READINESS, learning_progress,
                                     opportunity_persistence)
from polymarket_bot.config import BotConfig


def test_persistence_computes_lifetime_and_capacity():
    opps = [
        {"strategy": "chain_arb", "first_seen": "2026-07-23T10:00:00+00:00",
         "last_seen": "2026-07-23T10:10:00+00:00", "sightings": 20,
         "best_edge": 0.05, "best_depth_usd": 8000.0, "executed": 0},
        {"strategy": "chain_arb", "first_seen": "2026-07-23T11:00:00+00:00",
         "last_seen": "2026-07-23T11:02:00+00:00", "sightings": 4,
         "best_edge": 0.03, "best_depth_usd": 1000.0, "executed": 1},
    ]
    p = opportunity_persistence(opps)[0]
    assert p["windows"] == 2
    assert p["avg_lifetime_min"] == 6.0            # (10 + 2) / 2
    assert p["avg_sightings"] == 12.0
    assert p["best_edge"] == 0.05
    assert p["capacity_usd"] == 0.05 * 8000 + 0.03 * 1000   # 430
    assert p["executed"] == 1


def test_persistence_groups_by_strategy():
    opps = [
        {"strategy": "arb", "first_seen": "2026-07-23T10:00:00+00:00",
         "last_seen": "2026-07-23T10:01:00+00:00", "sightings": 1,
         "best_edge": 0.02, "best_depth_usd": 300.0, "executed": 0},
        {"strategy": "chain_arb", "first_seen": "2026-07-23T10:00:00+00:00",
         "last_seen": "2026-07-23T10:05:00+00:00", "sightings": 5,
         "best_edge": 0.04, "best_depth_usd": 500.0, "executed": 0},
    ]
    rows = opportunity_persistence(opps)
    assert [r["strategy"] for r in rows] == ["arb", "chain_arb"]   # sorted


def test_persistence_tolerates_bad_timestamps():
    opps = [{"strategy": "arb", "first_seen": "nonsense", "last_seen": "also bad",
             "sightings": 1, "best_edge": 0.02, "best_depth_usd": 100.0,
             "executed": 0}]
    assert opportunity_persistence(opps)[0]["avg_lifetime_min"] == 0.0


def test_learning_progress_counts_from_ledger(cfg, ledger):
    cfg.ticks.enabled = False                       # skip the tick store here
    for i in range(3):
        ledger.record_quote_outcome("paper", "m", 0.5, filled=bool(i % 2))
    prog = learning_progress(cfg, ledger, "paper")
    assert prog["fill probability"] == 3
    assert prog["category correlations"] == 0       # ticks disabled
    assert set(prog) == set(READINESS)              # one entry per learner


def test_readiness_thresholds_are_positive():
    assert all(need > 0 for _, need in READINESS.values())


def test_run_capacity_smoke(cfg, ledger, capsys):
    from polymarket_bot.capacity import run_capacity
    cfg.ticks.enabled = False
    ledger.record_opportunity("paper", "chain_arb", "e:a:b", "GPT ladder",
                              0.04, 8000.0)
    ledger.close()                                  # run_capacity opens its own
    run_capacity(cfg, "paper")
    out = capsys.readouterr().out
    assert "Opportunity persistence" in out and "Learning readiness" in out
