"""Скоринг рынков для MM и сателлит btc_5m_ta."""

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


# --- скоринг ---

def test_scorer_accepts_good_mm_market(cfg):
    assert MarketScorer(cfg).eligible(mm_market())


def test_scorer_filters(cfg):
    s = MarketScorer(cfg)
    assert not s.eligible(mm_market(volume_24h_usd=10_000))          # объём
    near = datetime.now(timezone.utc) + timedelta(days=5)
    assert not s.eligible(mm_market(end_date=near))                  # < 30 дней
    assert not s.eligible(mm_market(rewards_min_size=0))             # вне rewards
    assert not s.eligible(mm_market(one_day_price_change=0.10))      # волатилен
    assert not s.eligible(mm_market(best_bid=0.449, best_ask=0.450)) # спред < 2 тиков


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


# --- сателлит: детерминированные слаги и тайминг ---

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
    assert ta_p_up([]) == (0.5, 0.0)                     # мало данных — воздержание


def test_satellite_edge_accounts_for_crypto_fee(cfg, ledger):
    sat = BTC5mSatellite(cfg, ledger, clob=mock.Mock(), trader=None, mode="paper")
    # p_up 0.60 против implied 0.50: сырой edge 0.10, но fee crypto 0.07
    # оставляет 0.03 < порога 0.05 -> входа нет.
    assert sat.decide(p_up=0.60, implied_up=0.50) is None
    # p_up 0.65: edge после fee 0.08 >= 0.05 -> вход в UP.
    assert sat.decide(p_up=0.65, implied_up=0.50) == (0, 0.65)
    # Зеркально для DOWN.
    assert sat.decide(p_up=0.35, implied_up=0.50) == (1, 0.65)


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
    now = window + 300 - 15.0                            # T-15s: внутри окна входа

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
        sat.cycle(now=now + 2)                           # то же окно — не дублируем
    trades = ledger.open_positions("paper")
    assert len(trades) == 1
    assert trades[0].size > 0
