"""Smart-money as an ensemble signal (not blind copying)."""

from unittest import mock

from polymarket_bot.smartmoney import SmartMoneySignal, SmartMoneyTracker, parse_positions
from polymarket_bot.niche import NicheWatcher

from .conftest import make_candidate


def test_signal_fires_only_on_hot_tokens():
    sig = SmartMoneySignal(max_confidence=0.4)
    sig.set_hot({"tok-hot": 0.3})
    hot = sig.evaluate(make_candidate(outcome_index=0))       # token is tok-yes-m1
    assert hot is None                                        # not in hot set
    c = make_candidate(outcome_index=0)
    c.token_id = "tok-hot"
    s = sig.evaluate(c)
    assert s is not None and s.name == "smart_money"
    assert s.confidence == 0.3 and s.p_est > c.p_mkt          # nudges estimate up


def test_signal_confidence_capped():
    sig = SmartMoneySignal(max_confidence=0.4)
    sig.set_hot({"t": 0.9})
    c = make_candidate(); c.token_id = "t"
    assert sig.evaluate(c).confidence == 0.4


def raw(**over):
    d = dict(asset="a", title="Will Russia capture Sumy?", outcome="Yes",
             size=5000, avgPrice=0.04, curPrice=0.05, cashPnl=2000.0)
    d.update(over)
    return d


def test_build_hot_tokens_from_profitable_wallet(cfg, ledger):
    cfg.smart_money.enabled = True
    cfg.smart_money.as_signal = True
    cfg.smart_money.watch_wallets = ["0xW"]
    cfg.smart_money.signal_min_pnl_usd = 1000.0
    tracker = SmartMoneyTracker(cfg, ledger, NicheWatcher(cfg, ledger))
    # Profitable wallet ($2000 PnL) holding a $200 tail (avg 0.04).
    tracker.fetch_positions = mock.Mock(
        side_effect=lambda w: parse_positions(w, [raw(asset="tail-tok")]))
    hot = tracker.build_hot_tokens()
    assert "tail-tok" in hot and 0 < hot["tail-tok"] <= cfg.smart_money.signal_max_confidence


def test_unprofitable_wallet_gives_no_signal(cfg, ledger):
    cfg.smart_money.enabled = True
    cfg.smart_money.as_signal = True
    cfg.smart_money.watch_wallets = ["0xW"]
    tracker = SmartMoneyTracker(cfg, ledger, NicheWatcher(cfg, ledger))
    tracker.fetch_positions = mock.Mock(
        side_effect=lambda w: parse_positions(w, [raw(cashPnl=-500.0)]))
    assert tracker.build_hot_tokens() == {}
