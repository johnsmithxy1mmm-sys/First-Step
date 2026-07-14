"""Smart-money tracker: position parsing, new detection, dedup, tail/niche tags."""

from unittest import mock

import pytest

from polymarket_bot.niche import NicheWatcher
from polymarket_bot.smartmoney import SmartMoneyTracker, parse_positions


def raw_pos(**over) -> dict:
    d = dict(asset="tok1", title="Will Bitcoin hit $200k?", outcome="Yes",
             size=1000, avgPrice=0.05, curPrice=0.06, cashPnl=10.0)
    d.update(over)
    return d


def make_tracker(cfg, ledger, wallets=("0xWHALE",), positions=None):
    cfg.smart_money.enabled = True
    cfg.smart_money.watch_wallets = list(wallets)
    tracker = SmartMoneyTracker(cfg, ledger, NicheWatcher(cfg, ledger))
    if positions is not None:
        tracker.fetch_positions = mock.Mock(
            side_effect=lambda w: parse_positions(w, positions.get(w, [])))
    return tracker


def test_parse_positions_tolerates_garbage():
    parsed = parse_positions("0xA", [
        raw_pos(), "not-a-dict", {}, raw_pos(asset="", title="no asset"),
        raw_pos(asset="tok2", avgPrice="bad")])
    assert [p.asset for p in parsed] == ["tok1"]     # only the valid record
    assert parsed[0].usd == pytest.approx(50.0)


def test_alerts_new_position_once(cfg, ledger):
    positions = {"0xWHALE": [raw_pos()]}
    tracker = make_tracker(cfg, ledger, positions=positions)
    with mock.patch("polymarket_bot.smartmoney.alert") as a:
        first = tracker.cycle()
    assert len(first) == 1 and a.called
    # Second cycle — the position is already seen, silence.
    with mock.patch("polymarket_bot.smartmoney.alert") as a:
        assert tracker.cycle() == []
        a.assert_not_called()


def test_ignores_dust(cfg, ledger):
    tiny = raw_pos(size=100, avgPrice=0.001)          # $0.10 < $50 threshold
    tracker = make_tracker(cfg, ledger, positions={"0xWHALE": [tiny]})
    with mock.patch("polymarket_bot.smartmoney.alert"):
        assert tracker.cycle() == []


def test_tail_and_niche_tags(cfg, ledger):
    positions = {"0xWHALE": [
        raw_pos(asset="t-tail", title="Will Russia capture Sumy?",
                avgPrice=0.04, size=5000),             # $200: tail + post-soviet niche
        raw_pos(asset="t-mid", title="Will the Chiefs win?", avgPrice=0.55),
    ]}
    tracker = make_tracker(cfg, ledger, positions=positions)
    with mock.patch("polymarket_bot.smartmoney.alert") as a:
        alerted = tracker.cycle()
    assert len(alerted) == 2
    texts = " ".join(call.args[0] for call in a.call_args_list)
    assert "TAIL" in texts and "niche:post-soviet" in texts


def test_only_niche_or_tail_filter(cfg, ledger):
    cfg.smart_money.only_niche_or_tail = True
    positions = {"0xWHALE": [
        raw_pos(asset="t-tail", title="Random market", avgPrice=0.03, size=5000),  # $150 tail
        raw_pos(asset="t-niche", title="Will OpenAI release GPT-6?", avgPrice=0.60),  # ai niche
        raw_pos(asset="t-skip", title="Boring midprice market", avgPrice=0.55),       # neither
    ]}
    tracker = make_tracker(cfg, ledger, positions=positions)
    with mock.patch("polymarket_bot.smartmoney.alert"):
        alerted = tracker.cycle()
    assert {p.asset for p in alerted} == {"t-tail", "t-niche"}   # midprice dropped
    # But all three are marked seen — no duplicates later.
    assert len(ledger.smart_money_seen_keys()) == 3


def test_disabled_or_empty_is_silent(cfg, ledger):
    cfg.smart_money.enabled = False
    cfg.smart_money.watch_wallets = ["0xWHALE"]
    assert SmartMoneyTracker(cfg, ledger, NicheWatcher(cfg, ledger)).cycle() == []
    cfg.smart_money.enabled = True
    cfg.smart_money.watch_wallets = []
    assert SmartMoneyTracker(cfg, ledger, NicheWatcher(cfg, ledger)).cycle() == []
