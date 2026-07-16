"""Resolution alpha: near-resolved detection, fee + dispute-aware net edge."""

from datetime import datetime, timedelta, timezone
from unittest import mock

from polymarket_bot.resolution import ResolutionAlpha

from .conftest import make_book, make_market


def make_res(cfg, ledger, clob=None):
    cfg.resolution.enabled = True
    return ResolutionAlpha(cfg, ledger, clob or mock.Mock(), None, "dry-run")


def near_market(**over):
    d = dict(id="r1", question="Will the incumbent win?", outcomes=["Yes", "No"],
             outcome_prices=[0.97, 0.03], clob_token_ids=["y", "n"],
             volume_24h_usd=50_000, volume_usd=100_000,   # 50% fresh
             end_date=datetime.now(timezone.utc) + timedelta(days=2),
             category="geopolitics")   # taker fee 0 -> clean edge
    d.update(over)
    return make_market(**d)


def test_detects_near_resolved(cfg, ledger):
    res = make_res(cfg, ledger)
    cand = res.evaluate(near_market())
    assert cand is not None
    assert cand.outcome_index == 0 and cand.price == 0.97
    # net edge = (1-0.97) - fee(0) - dispute(0.02) = 0.01
    assert abs(cand.net_edge - 0.01) < 1e-9


def test_rejects_thin_edge(cfg, ledger):
    res = make_res(cfg, ledger)
    # In-band 0.97 but crypto (taker fee 0.07): 0.03 - 0.07 - 0.02 < 0 -> thin.
    thin = near_market(category="crypto")
    assert res.evaluate(thin) is None
    assert "net edge" in res.reject_reason(thin)


def test_rejects_not_near(cfg, ledger):
    res = make_res(cfg, ledger)
    assert res.evaluate(near_market(outcome_prices=[0.80, 0.20])) is None


def test_rejects_stale_no_fresh_volume(cfg, ledger):
    res = make_res(cfg, ledger)
    m = near_market(volume_24h_usd=25_000, volume_usd=10_000_000)   # 0.25% fresh
    assert "stale" in res.reject_reason(m)


def test_rejects_far_horizon(cfg, ledger):
    res = make_res(cfg, ledger)
    far = near_market(end_date=datetime.now(timezone.utc) + timedelta(days=60))
    assert "not imminent" in res.reject_reason(far)


def test_rejects_market_without_end_date(cfg, ledger):
    """No end date = imminence unverifiable — must NOT count as imminent."""
    res = make_res(cfg, ledger)
    assert "not imminent" in res.reject_reason(near_market(end_date=None))


def test_alert_cap_bounds_alerts_not_just_return(cfg, ledger):
    res = make_res(cfg, ledger)
    cfg.resolution.max_alerts_per_cycle = 3
    markets = [near_market(id=f"r{i}") for i in range(10)]
    with mock.patch("polymarket_bot.resolution.alert") as a:
        found = res.cycle(markets)
    assert len(found) == 3
    assert a.call_count == 3          # alerts were capped too, not sliced after


def test_cycle_alerts_and_optionally_executes(cfg, ledger):
    clob = mock.Mock()
    clob.order_book.return_value = make_book(best_bid=0.96, best_ask=0.97, depth=100_000)
    res = make_res(cfg, ledger, clob)
    cfg.resolution.execute = True
    with mock.patch("polymarket_bot.resolution.alert") as a:
        found = res.cycle([near_market()])
    assert len(found) == 1 and a.called
    row = ledger._conn.execute(
        "SELECT strategy, side FROM trades").fetchone()
    assert row["strategy"] == "resolution" and row["side"] == "BUY"


def test_disabled_is_silent(cfg, ledger):
    res = ResolutionAlpha(cfg, ledger, mock.Mock(), None, "dry-run")  # enabled=False
    assert res.cycle([near_market()]) == []
