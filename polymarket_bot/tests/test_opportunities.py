"""Opportunity ledger: measured window capacity (upsert, best-of, stats)."""


def test_new_window_inserts(ledger):
    ledger.record_opportunity("paper", "arb", "e1:YES", "World Cup", 0.03, 500.0)
    stats = ledger.opportunity_stats("paper")
    assert len(stats) == 1
    row = stats[0]
    assert row["strategy"] == "arb" and row["windows"] == 1
    assert row["sightings"] == 1 and row["best_edge"] == 0.03


def test_resighting_bumps_and_keeps_best(ledger):
    ledger.record_opportunity("paper", "arb", "e1:YES", "WC", 0.03, 500.0)
    ledger.record_opportunity("paper", "arb", "e1:YES", "WC", 0.05, 300.0)  # better edge
    ledger.record_opportunity("paper", "arb", "e1:YES", "WC", 0.02, 900.0)  # deeper
    stats = ledger.opportunity_stats("paper")[0]
    assert stats["windows"] == 1          # same structure -> one window
    assert stats["sightings"] == 3
    assert stats["best_edge"] == 0.05     # best edge kept
    assert stats["depth_usd"] == 900.0    # best depth kept
    top = ledger.top_opportunities("paper")[0]
    assert top["last_seen"] >= top["first_seen"]


def test_distinct_structures_are_distinct_windows(ledger):
    ledger.record_opportunity("paper", "chain_arb", "e1:a:b", "GPT date", 0.04, 200.0)
    ledger.record_opportunity("paper", "chain_arb", "e1:c:d", "BTC value", 0.06, 100.0)
    stats = ledger.opportunity_stats("paper")[0]
    assert stats["windows"] == 2 and stats["sightings"] == 2


def test_capacity_metric_is_edge_times_depth(ledger):
    ledger.record_opportunity("paper", "arb", "k1", "A", 0.05, 1000.0)   # $50
    ledger.record_opportunity("paper", "arb", "k2", "B", 0.02, 500.0)    # $10
    stats = ledger.opportunity_stats("paper")[0]
    assert stats["edge_dollars"] == 50.0 + 10.0


def test_executed_flag_sticks_once_set(ledger):
    ledger.record_opportunity("paper", "arb", "k", "A", 0.05, 100.0, executed=True)
    ledger.record_opportunity("paper", "arb", "k", "A", 0.04, 100.0, executed=False)
    assert ledger.opportunity_stats("paper")[0]["executed"] == 1   # MAX keeps it


def test_top_ordered_by_capacity(ledger):
    ledger.record_opportunity("paper", "arb", "small", "thin", 0.10, 50.0)     # $5
    ledger.record_opportunity("paper", "chain_arb", "big", "deep", 0.03, 5000.0)  # $150
    top = ledger.top_opportunities("paper")
    assert top[0]["label"] == "deep"     # highest edge*depth first


def test_modes_isolated(ledger):
    ledger.record_opportunity("paper", "arb", "k", "A", 0.05, 100.0)
    ledger.record_opportunity("dry-run", "arb", "k", "A", 0.05, 100.0)
    assert len(ledger.opportunity_stats("paper")) == 1
    assert len(ledger.opportunity_stats("live")) == 0
