"""Cross-platform scanner and niche watchlists."""

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
    assert "RESOLUTION RULES" in found[0].describe()


def test_small_gap_or_weak_match_ignored(cfg):
    poly = make_market(
        question="Will the Fed cut interest rates in September 2026?",
        outcome_prices=[0.30, 0.70], volume_24h_usd=50_000,
    )
    small_gap = VenueMarket(venue="kalshi",
                            title="Fed cuts interest rates in September 2026",
                            yes_price=0.32)                      # 2 pp < 4
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
    assert "Rules:" in alert_mock.call_args[0][0]   # rules lawyering: rules in the alert

    # Second cycle: market already seen, no alert.
    with mock.patch("polymarket_bot.niche.alert") as alert_mock:
        assert watcher.cycle([ukraine, boring]) == []
    alert_mock.assert_not_called()


def test_niche_crypto_watchlist(cfg, ledger):
    watcher = NicheWatcher(cfg, ledger)
    btc = make_market(id="c1", question="Will Bitcoin close above $150k this year?")
    with mock.patch("polymarket_bot.niche.alert"):
        hits = watcher.cycle([btc])
    assert hits[0][0] == "crypto"


def test_niche_ai_and_football_watchlists(cfg, ledger):
    watcher = NicheWatcher(cfg, ledger)
    markets = [
        make_market(id="a1", question="Will OpenAI release GPT-6 by 2027?"),
        make_market(id="a2", question="Will any AI achieve AGI before 2030?"),
        make_market(id="f1", question="Will Real Madrid win the Champions League?"),
        make_market(id="f2", question="Will Arsenal finish top of the Premier League?"),
        make_market(id="u1", question="Will Ukraine join the EU by 2030?"),
    ]
    with mock.patch("polymarket_bot.niche.alert"):
        hits = {m.id: name for name, m in watcher.cycle(markets)}
    assert hits["a1"] == "ai"
    assert hits["a2"] == "ai"
    assert hits["f1"] == "football-eu"
    assert hits["f2"] == "football-eu"
    # "Ukraine" contains the letters "ai", but word boundaries + order give post-soviet.
    assert hits["u1"] == "post-soviet"


def test_niche_matches_whole_words_only(cfg, ledger):
    """Bug from a live run: 'eth' matched 'Hegseth' as a substring."""
    watcher = NicheWatcher(cfg, ledger)
    hegseth = make_market(
        id="h1", question="Will Pete Hegseth win the 2028 US Presidential Election?")
    eth = make_market(id="h2", question="Will ETH close above $10k this year?")
    with mock.patch("polymarket_bot.niche.alert"):
        hits = watcher.cycle([hegseth, eth])
    assert [(name, m.id) for name, m in hits] == [("crypto", "h2")]


def test_niche_survives_market_without_end_date(cfg, ledger):
    """Bug from a live run: a market with no date crashed the cycle, losing the "seen" mark."""
    watcher = NicheWatcher(cfg, ledger)
    dateless = make_market(id="d1", question="Will Ukraine join the EU?",
                           end_date=None)
    with mock.patch("polymarket_bot.niche.alert"):
        hits = watcher.cycle([dateless])
    assert [(name, m.id) for name, m in hits] == [("post-soviet", "d1")]
    assert "d1" in ledger.seen_market_ids()   # market marked no matter what
    # Second cycle — silence, no duplicates.
    with mock.patch("polymarket_bot.niche.alert") as alert_mock:
        assert watcher.cycle([dateless]) == []
    alert_mock.assert_not_called()
