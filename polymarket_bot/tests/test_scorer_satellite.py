"""Market scoring for MM and the btc_5m_ta satellite."""

from datetime import datetime, timedelta, timezone
from unittest import mock

import pytest

from polymarket_bot.satellite import (BTC5mSatellite, Candle, market_slug,
                                      seconds_to_close, ta_p_up, window_ts)
from polymarket_bot.scorer import MarketScorer

from .conftest import make_market


def mm_market(**overrides):
    defaults = dict(
        id="s1", question="Will the Chiefs win the Super Bowl?",
        outcome_prices=[0.45, 0.55],
        clob_token_ids=["s1-yes", "s1-no"],
        volume_24h_usd=100_000, volume_usd=5_000_000,
        rewards_min_size=20.0, rewards_max_spread=0.03,
        best_bid=0.44, best_ask=0.46, one_day_price_change=0.01,
        end_date=datetime.now(timezone.utc) + timedelta(days=90),
    )
    defaults.update(overrides)
    return make_market(**defaults)


# --- scoring ---

def test_scorer_accepts_good_mm_market(cfg):
    assert MarketScorer(cfg).eligible(mm_market())


def test_scorer_filters(cfg):
    s = MarketScorer(cfg)
    assert not s.eligible(mm_market(volume_24h_usd=10_000))          # volume
    near = datetime.now(timezone.utc) + timedelta(days=5)
    assert not s.eligible(mm_market(end_date=near))                  # < 30 days
    assert not s.eligible(mm_market(rewards_min_size=0))             # not in rewards
    assert not s.eligible(mm_market(one_day_price_change=0.10))      # volatile
    assert not s.eligible(mm_market(best_bid=0.449, best_ask=0.450)) # spread < 2 ticks


def test_scorer_flags_subjective_resolution(cfg):
    s = MarketScorer(cfg)
    subjective = mm_market(description="Resolves YES if, in the opinion of major "
                                       "news outlets, the event is significant.")
    assert s.uma_risk(subjective)
    assert not s.eligible(subjective)


def test_scorer_prefers_sports_over_crypto(cfg):
    s = MarketScorer(cfg)
    sports = mm_market()
    crypto = mm_market(id="c1", question="Will Bitcoin be above $120k on Friday?",
                       clob_token_ids=["c1-yes", "c1-no"])
    assert s.score(sports, None) > s.score(crypto, None)


def test_score_uses_the_real_reward_share_not_notional_depth(cfg):
    """A wall resting near the band EDGE is priced at a few percent by the
    published quadratic rule, so a market that looks crowded by notional depth
    can still be wide open to a quote placed near the midpoint. This path also
    exercises the book branch of score(), which a None book skips entirely."""
    from polymarket_bot.models import BookLevel, OrderBook

    s = MarketScorer(cfg)
    m = mm_market()                       # rewards_max_spread = 0.03, mid ~0.45
    # Same notional size, different distance from the midpoint.
    near = OrderBook(bids=[BookLevel(price=0.449, size=2000)],
                     asks=[BookLevel(price=0.451, size=2000)])
    edge = OrderBook(bids=[BookLevel(price=0.421, size=2000)],
                     asks=[BookLevel(price=0.479, size=2000)])
    assert s.score(m, edge) > s.score(m, near)
    assert s.score(m, near) > 0


def test_score_survives_a_market_with_no_rewards_band(cfg):
    """Not in the program -> not scoreable for rewards; must not raise."""
    s = MarketScorer(cfg)
    m = mm_market(rewards_max_spread=0.0, rewards_min_size=0.0)
    assert s.score(m, None) >= 0.0


# --- satellite: deterministic slugs and timing ---

def test_slug_deterministic_and_ts_multiple_of_300():
    now = 1_750_000_123.0
    slug = market_slug(now, "btc-updown-5m-{ts}")
    ts = int(slug.rsplit("-", 1)[1])
    assert ts % 300 == 0
    assert ts <= now < ts + 300
    assert window_ts(now) == ts
    assert seconds_to_close(now) == pytest.approx(ts + 300 - now)


def test_ta_p_up_direction():
    up = [Candle(open=100 + i, high=101 + i, low=99 + i, close=100.8 + i)
          for i in range(10)]
    down = [Candle(open=110 - i, high=111 - i, low=108 - i, close=109.2 - i)
            for i in range(10)]
    p_up_bull, conf_bull = ta_p_up(up)
    p_up_bear, conf_bear = ta_p_up(down)
    assert p_up_bull > 0.6 and conf_bull > 0
    assert p_up_bear < 0.4 and conf_bear > 0
    assert ta_p_up([]) == (0.5, 0.0)                     # too little data — abstain


def test_satellite_edge_accounts_for_crypto_fee(cfg, ledger):
    """Fee at a 50c fill is theta*p*(1-p) = 0.07*0.25 = 1.75c per share, not 7c.
    Threshold is 0.05, so entry needs a raw edge above 0.05 + 0.0175."""
    sat = BTC5mSatellite(cfg, ledger, clob=mock.Mock(), trader=None, mode="paper")
    # raw edge 0.05 -> 0.0325 after fee < 0.05 -> no entry.
    assert sat.decide(p_up=0.55, implied_up=0.50) is None
    # raw edge 0.10 -> 0.0825 after fee >= 0.05 -> enter UP.
    assert sat.decide(p_up=0.60, implied_up=0.50) == (0, 0.60)
    # Mirror for DOWN: the fee is symmetric in price, so the bar is the same.
    assert sat.decide(p_up=0.40, implied_up=0.50) == (1, 0.60)


def test_satellite_quarter_kelly_capped(cfg, ledger):
    sat = BTC5mSatellite(cfg, ledger, clob=mock.Mock(), trader=None, mode="paper")
    stake = sat.bet_size(p_model=0.65, price=0.52)
    assert 0 < stake <= cfg.satellite.max_bet_usd


def test_satellite_disabled_by_default(cfg, ledger):
    assert cfg.satellite.enabled is False
    sat = BTC5mSatellite(cfg, ledger, clob=mock.Mock(), trader=None, mode="paper")
    with mock.patch.object(sat, "fetch_candles") as fetch:
        sat.cycle()
    fetch.assert_not_called()


def test_satellite_trades_once_per_window(cfg, ledger):
    cfg.satellite.enabled = True
    clob = mock.Mock()
    sat = BTC5mSatellite(cfg, ledger, clob=clob, trader=None, mode="paper")
    window = window_ts(1_750_000_000.0)
    now = window + 300 - 15.0                            # T-15s: inside the entry window

    candles = [Candle(open=100 + i, high=101 + i, low=99 + i, close=100.9 + i)
               for i in range(12)]
    market = make_market(
        id="btc", question="Bitcoin up or down?",
        outcome_prices=[0.5, 0.5], clob_token_ids=["b-up", "b-down"],
        volume_24h_usd=1000, resolution_source="chainlink")
    from polymarket_bot.models import OrderBook, BookLevel
    clob.order_book.return_value = OrderBook(
        bids=[BookLevel(price=0.49, size=5000)],
        asks=[BookLevel(price=0.50, size=5000)])

    with mock.patch.object(sat, "fetch_candles", return_value=candles), \
         mock.patch.object(sat, "fetch_market", return_value=market):
        sat.cycle(now=now)
        sat.cycle(now=now + 2)                           # same window — no duplicate
    trades = ledger.open_positions("paper")
    assert len(trades) == 1
    assert trades[0].size > 0
