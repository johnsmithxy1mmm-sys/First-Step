"""Funnel diagnostics: reject reasons for MM and the longshot scanner."""

from datetime import datetime, timedelta, timezone

from rich.console import Console

from polymarket_bot.diagnose import _print_chain_arb_summary, funnel
from polymarket_bot.scanner import Scanner
from polymarket_bot.scorer import MarketScorer

from .conftest import make_market


def mm_ok(**over):
    """A market that is eligible for MM."""
    d = dict(
        id="ok", question="Will the Democrats win the Senate?",
        outcome_prices=[0.45, 0.55], clob_token_ids=["y", "n"],
        volume_24h_usd=100_000, rewards_min_size=20.0, rewards_max_spread=0.03,
        one_day_price_change=0.0, best_bid=0.44, best_ask=0.46,
        end_date=datetime.now(timezone.utc) + timedelta(days=90),
        resolution_source="https://senate.gov",
    )
    d.update(over)
    return make_market(**d)


# --- MM scorer.reject_reason ---

def test_mm_reject_reasons(cfg):
    s = MarketScorer(cfg)
    assert s.reject_reason(mm_ok()) is None
    assert "volume" in s.reject_reason(mm_ok(volume_24h_usd=1000))
    assert "rewards" in s.reject_reason(mm_ok(rewards_min_size=0))
    assert "resolution" in s.reject_reason(
        mm_ok(end_date=datetime.now(timezone.utc) + timedelta(days=3)))
    assert "volatile" in s.reject_reason(mm_ok(one_day_price_change=0.2))
    assert "spread" in s.reject_reason(mm_ok(best_bid=0.449, best_ask=0.450))


def test_eligible_matches_reject_reason(cfg):
    s = MarketScorer(cfg)
    assert s.eligible(mm_ok()) is True
    assert s.eligible(mm_ok(volume_24h_usd=1000)) is False


# --- longshot scanner.reject_reason ---

def test_scanner_reject_reasons(cfg):
    sc = Scanner(cfg)
    good = make_market(outcome_prices=[0.01, 0.99])   # a cheap Yes exists
    assert sc.reject_reason(good) is None
    assert "volume" in sc.reject_reason(make_market(volume_24h_usd=100))
    assert "window" in sc.reject_reason(make_market(end_date=None))
    assert "priced" in sc.reject_reason(make_market(outcome_prices=[0.30, 0.70]))


# --- funnel aggregates ---

def test_funnel_counts(cfg):
    s = MarketScorer(cfg)
    markets = [
        mm_ok(id="a"), mm_ok(id="b"),                     # 2 pass
        mm_ok(id="c", volume_24h_usd=1000),               # volume
        mm_ok(id="d", volume_24h_usd=1000),               # volume
        mm_ok(id="e", rewards_min_size=0),                # rewards
    ]
    reasons, passed = funnel(markets, s.reject_reason)
    assert passed == 2
    assert reasons["24h volume < $50,000"] == 2
    assert reasons["not in rewards program"] == 1


# --- chain (ladder) arb summary: doesn't hit the CLOB, Gamma-price only ---

def test_chain_arb_summary_counts_classified_and_violating_pairs(cfg, capsys):
    same_date = datetime.now(timezone.utc) + timedelta(days=60)
    subset_m = make_market(
        id="btc-200k", event_id="ev1", question="Will Bitcoin reach $200,000 by Dec 2026?",
        end_date=same_date, outcome_prices=[0.10, 0.90],
        clob_token_ids=["s-y", "s-n"], volume_24h_usd=50_000)
    superset_m = make_market(
        id="btc-150k", event_id="ev1", question="Will Bitcoin reach $150,000 by Dec 2026?",
        end_date=same_date, outcome_prices=[0.05, 0.95],   # priced BELOW subset -> violation
        clob_token_ids=["p-y", "p-n"], volume_24h_usd=50_000)
    unrelated = make_market(id="other", event_id="ev1",
                            question="Will the home team win?", end_date=same_date,
                            clob_token_ids=["o-y", "o-n"], volume_24h_usd=50_000)
    console = Console()
    _print_chain_arb_summary(console, cfg, [subset_m, superset_m, unrelated])
    out = capsys.readouterr().out
    assert "CHAIN ARB" in out
    assert "3 same-event market pairs checked" in out
