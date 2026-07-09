"""Кросс-платформенный сканер и нишевые вотчлисты."""

from unittest import mock

import pytest

from polymarket_bot.crossmarket import CrossMarketScanner, VenueMarket, similarity
from polymarket_bot.niche import NicheWatcher

from .conftest import make_market


# --- crossmarket ---

def test_title_similarity():
    a = "Will the Fed cut interest rates in September 2026?"
    b = "Fed cuts interest rates September 2026"
    assert similarity(a, b) > 0.6
    assert similarity(a, "Will Real Madrid win the Champions League?") < 0.2


def make_cross(cfg, venue_markets):
    venue = mock.Mock()
    venue.name = "kalshi"
    venue.fetch_markets.return_value = venue_markets
    return CrossMarketScanner(cfg, venues=[venue])


def test_divergence_detected_above_threshold(cfg):
    poly = make_market(
        question="Will the Fed cut interest rates in September 2026?",
        outcome_prices=[0.30, 0.70], volume_24h_usd=50_000,
    )
    kalshi = VenueMarket(venue="kalshi",
                         title="Fed cuts interest rates in September 2026",
                         yes_price=0.38)
    scanner = make_cross(cfg, [kalshi])
    found = scanner.cycle([poly])
    assert len(found) == 1
    assert found[0].gap == pytest.approx(0.08)
    assert "ПРАВИЛА РЕЗОЛЮЦИИ" in found[0].describe()


def test_small_gap_or_weak_match_ignored(cfg):
    poly = make_market(
        question="Will the Fed cut interest rates in September 2026?",
        outcome_prices=[0.30, 0.70], volume_24h_usd=50_000,
    )
    small_gap = VenueMarket(venue="kalshi",
                            title="Fed cuts interest rates in September 2026",
                            yes_price=0.32)                      # 2 п.п. < 4
    unrelated = VenueMarket(venue="kalshi",
                            title="Will it snow in Miami?", yes_price=0.90)
    assert make_cross(cfg, [small_gap]).cycle([poly]) == []
    assert make_cross(cfg, [unrelated]).cycle([poly]) == []


# --- niche ---

def test_niche_alerts_once_per_market(cfg, ledger):
    watcher = NicheWatcher(cfg, ledger)
    ukraine = make_market(id="n1", question="Will Ukraine join the EU by 2030?")
    boring = make_market(id="n2", question="Will it rain in Paris tomorrow?")

    with mock.patch("polymarket_bot.niche.alert") as alert_mock:
        hits = watcher.cycle([ukraine, boring])
    assert [(name, m.id) for name, m in hits] == [("post-soviet", "n1")]
    alert_mock.assert_called_once()
    assert "Правила:" in alert_mock.call_args[0][0]   # rules lawyering: правила в алерте

    # Повторный цикл: рынок уже виден, алерта нет.
    with mock.patch("polymarket_bot.niche.alert") as alert_mock:
        assert watcher.cycle([ukraine, boring]) == []
    alert_mock.assert_not_called()


def test_niche_crypto_watchlist(cfg, ledger):
    watcher = NicheWatcher(cfg, ledger)
    btc = make_market(id="c1", question="Will Bitcoin close above $150k this year?")
    with mock.patch("polymarket_bot.niche.alert"):
        hits = watcher.cycle([btc])
    assert hits[0][0] == "crypto"
