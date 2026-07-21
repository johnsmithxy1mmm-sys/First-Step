"""Auto-redemption: detection from data-api, alert dedupe, safe guards."""

from unittest import mock

from polymarket_bot.redeemer import Redeemable, Redeemer


def api_pos(redeemable=True, size=120.0, cond="0x" + "ab" * 32,
            title="Will X win?", neg=False):
    return {"redeemable": redeemable, "size": size, "conditionId": cond,
            "title": title, "negativeRisk": neg}


def make_redeemer(cfg, positions, mode="live"):
    cfg.redemption.enabled = True
    trader = mock.Mock()
    trader.api_positions.return_value = positions
    return Redeemer(cfg, trader, mode)


def test_scan_picks_only_redeemable(cfg):
    r = make_redeemer(cfg, [
        api_pos(redeemable=True, size=100),
        api_pos(redeemable=False, size=50),
        api_pos(redeemable=True, size=0),            # zero size — nothing to claim
        api_pos(redeemable=True, size=80, cond=""),  # no condition id — can't claim
    ])
    items = r.scan()
    assert len(items) == 1 and items[0].value_usd == 100.0


def test_scan_skips_dust(cfg):
    cfg.redemption.min_claim_usd = 5.0
    r = make_redeemer(cfg, [api_pos(size=2.0), api_pos(size=50.0)])
    assert [i.value_usd for i in r.scan()] == [50.0]


def test_no_trader_no_scan(cfg):
    cfg.redemption.enabled = True
    r = Redeemer(cfg, None, "paper")                 # paper: no on-chain funds
    assert r.scan() == []
    assert r.cycle() == []


def test_cycle_alerts_once_per_total(cfg):
    r = make_redeemer(cfg, [api_pos(size=100)])
    with mock.patch("polymarket_bot.redeemer.alert") as a:
        r.cycle()
        r.cycle()                                    # same total — no re-alert
    assert a.call_count == 1
    assert "$100.00" in a.call_args_list[0].args[0]


def test_cycle_realerts_on_new_total(cfg):
    trader = mock.Mock()
    cfg.redemption.enabled = True
    r = Redeemer(cfg, trader, "live")
    with mock.patch("polymarket_bot.redeemer.alert") as a:
        trader.api_positions.return_value = [api_pos(size=100)]
        r.cycle()
        trader.api_positions.return_value = [api_pos(size=100), api_pos(size=40)]
        r.cycle()
    assert a.call_count == 2


def test_alert_state_resets_when_claimed(cfg):
    trader = mock.Mock()
    cfg.redemption.enabled = True
    r = Redeemer(cfg, trader, "live")
    with mock.patch("polymarket_bot.redeemer.alert") as a:
        trader.api_positions.return_value = [api_pos(size=100)]
        r.cycle()
        trader.api_positions.return_value = []       # claimed externally
        r.cycle()
        trader.api_positions.return_value = [api_pos(size=100)]
        r.cycle()                                    # same $100 again — re-alert
    assert a.call_count == 2


def test_neg_risk_never_auto_redeemed(cfg):
    """Wrong adapter call loses real money — neg-risk stays manual."""
    cfg.redemption.execute = True
    r = make_redeemer(cfg, [api_pos(neg=True)])
    with mock.patch.object(r, "redeem", wraps=r.redeem) as red, \
            mock.patch("polymarket_bot.redeemer.alert"):
        r.cycle()
    red.assert_not_called()                          # filtered before redeem()
    assert r.redeem(Redeemable(condition_id="0xab", size=10, neg_risk=True)) is False


def test_redeem_refuses_without_key_or_rpc(cfg, monkeypatch):
    monkeypatch.delenv("POLYMARKET_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("POLYGON_RPC_URL", raising=False)
    r = make_redeemer(cfg, [])
    assert r.redeem(Redeemable(condition_id="0xab", size=10)) is False


def test_execute_gate_blocks_onchain_but_not_alert(cfg):
    cfg.redemption.execute = True
    r = make_redeemer(cfg, [api_pos(size=100)])
    with mock.patch.object(r, "redeem") as red, \
            mock.patch("polymarket_bot.redeemer.alert") as a:
        r.cycle(allow_execute=False)                 # kill-switch says no orders
    red.assert_not_called()
    a.assert_called_once()                           # the human still hears about it
