"""P0-5: the research/calibration helpers are actually wired into production paths."""

from unittest import mock

from polymarket_bot.calibration import PlattCalibrator
from polymarket_bot.estimator import Estimator
from polymarket_bot.replay import queue_aware_fill
from polymarket_bot.ws_feed import TopOfBook

from .conftest import make_candidate, make_market


# --- Platt inside estimate_all (not just the module) ---

def test_platt_applies_in_estimate_all(cfg):
    platt = PlattCalibrator(min_samples=4)
    # The model is systematically overconfident high, underconfident low.
    platt.fit([(0.9, 0.0), (0.9, 1.0), (0.1, 1.0), (0.1, 0.0)] * 10)
    plain = Estimator(cfg)
    calibrated = Estimator(cfg, platt=platt)
    c = make_candidate(outcome_prices=[0.03, 0.97])
    p_plain = plain.estimate_all([c], [c.market])[0].p_est
    p_cal = calibrated.estimate_all([c], [c.market])[0].p_est
    assert p_cal != p_plain                     # the calibrator touched the path


# --- queue preservation inside _requote_market ---

def test_requote_keeps_order_when_fill_likely(cfg, ledger):
    from .test_marketmaker import make_mm, mm_market, top
    cfg.market_maker.requote_min_fill_prob = 0.5
    cfg.market_maker.requote_timer_sec = 0          # force needs_requote via timer
    tops = {"mm1-yes": top(), "mm1-no": top(bid=0.53, ask=0.57)}
    mm = make_mm(cfg, ledger, tops=tops)
    m = mm_market(volume_24h_usd=5_000_000,          # heavy flow -> high fill prob
                  volume_usd=50_000_000)             # (ratio under the volume guard)
    first = mm._requote_market(m, tops["mm1-yes"])
    assert first is not None
    with mock.patch.object(mm, "_place") as place:
        again = mm._requote_market(m, tops["mm1-yes"])   # fair barely moved
    assert again is first                            # same resting order kept
    place.assert_not_called()                        # no cancel-replace churn


def test_requote_replaces_when_fill_unlikely(cfg, ledger):
    from .test_marketmaker import make_mm, mm_market, top
    cfg.market_maker.requote_min_fill_prob = 0.5
    cfg.market_maker.requote_timer_sec = 0
    tops = {"mm1-yes": top(), "mm1-no": top(bid=0.53, ask=0.57)}
    mm = make_mm(cfg, ledger, tops=tops)
    m = mm_market(volume_24h_usd=6_000)              # trickle flow -> low fill prob
    first = mm._requote_market(m, tops["mm1-yes"])
    assert first is not None
    with mock.patch.object(mm, "_place") as place:
        mm._requote_market(m, tops["mm1-yes"])
    place.assert_called_once()                       # low P(fill) -> reprice


# --- queue-aware replay fill ---

def test_queue_aware_fill_partial():
    quote = mock.Mock(yes_bid=0.44, no_bid=0.55, size=100.0)
    # The trade-through (60) must first eat the 40 queued ahead -> only 20 fill.
    t = TopOfBook(bid=0.44, bid_size=40, ask=0.43, ask_size=60, ts=1.0)
    assert queue_aware_fill(quote, 0, t) == 20.0
    # No trade-through at all -> nothing fills.
    t2 = TopOfBook(bid=0.44, bid_size=40, ask=0.45, ask_size=500, ts=1.0)
    assert queue_aware_fill(quote, 0, t2) == 0.0


def test_replay_installs_queue_aware_model(cfg, ledger, tmp_path, monkeypatch):
    import sqlite3
    from polymarket_bot import replay as replay_mod
    db = tmp_path / "snaps.sqlite"
    conn = sqlite3.connect(str(db))
    conn.executescript(replay_mod.SNAP_SCHEMA)
    m = make_market(id="r1")
    conn.execute("INSERT INTO snap_markets (market_id, payload) VALUES (?,?)",
                 (m.id, m.model_dump_json()))
    conn.commit(); conn.close()
    seen = {}
    orig_mm = replay_mod.MarketMaker
    def spy_mm(*a, **k):
        inst = orig_mm(*a, **k)
        seen["mm"] = inst
        return inst
    monkeypatch.setattr(replay_mod, "MarketMaker", spy_mm)
    replay_mod.replay(cfg, db, ledger)
    assert seen["mm"].fill_model is replay_mod.queue_aware_fill


# --- Sharpe advisory in the digest ---

def test_digest_includes_sharpe_advice(tmp_path):
    from .test_hardening import make_bot
    bot = make_bot(tmp_path)
    bot._pnl_history = {"fade": [0, 5, 10, 15], "mm": [0, 8, -6, 2]}
    with mock.patch("polymarket_bot.main.alert") as a, \
         mock.patch.object(bot.ledger, "realized_pnl_by_strategy",
                           return_value={"fade": 20.0, "mm": 3.0}):
        bot.digest_job()
    text = a.call_args[0][0]
    assert "Suggested capital weights" in text
    bot.close()
