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


def test_allow_execute_false_alerts_but_places_nothing(cfg, ledger):
    """Kill-switch/observe-only/breaker: alerts keep flowing, orders do not."""
    clob = mock.Mock()
    clob.order_book.return_value = make_book(best_bid=0.96, best_ask=0.97,
                                             depth=100_000)
    res = make_res(cfg, ledger, clob)
    cfg.resolution.execute = True
    with mock.patch("polymarket_bot.resolution.alert") as a:
        found = res.cycle([near_market()], allow_execute=False)
    assert len(found) == 1 and a.called               # still detected + alerted
    assert ledger._conn.execute("SELECT COUNT(*) c FROM trades").fetchone()["c"] == 0


# --- UMA oracle signal (Gamma umaResolutionStatus) ---

def test_oracle_proposed_bypasses_imminence_heuristics(cfg, ledger):
    """A live on-chain proposal is a FACT: stale volume and a far end date
    (the old proxies) must not reject the market."""
    res = make_res(cfg, ledger)
    m = near_market(uma_resolution_status="proposed",
                    volume_24h_usd=100,                  # fails the volume floor
                    volume_usd=10_000_000,               # and the freshness ratio
                    end_date=datetime.now(timezone.utc) + timedelta(days=60))
    cand = res.evaluate(m)
    assert cand is not None
    assert abs(cand.net_edge - 0.01) < 1e-9              # full haircut still reserved


def test_oracle_proposed_still_needs_price_band(cfg, ledger):
    """The status does not say WHICH outcome was proposed — the price does.
    Outside the band the market disagrees or there is no meat: reject."""
    res = make_res(cfg, ledger)
    m = near_market(uma_resolution_status="proposed", outcome_prices=[0.80, 0.20])
    assert "outside" in res.reject_reason(m)


def test_oracle_resolved_drops_dispute_haircut(cfg, ledger):
    """After the dispute window the outcome is final — no dispute reserve."""
    res = make_res(cfg, ledger)
    cand = res.evaluate(near_market(uma_resolution_status="resolved"))
    assert cand is not None
    assert abs(cand.net_edge - 0.03) < 1e-9              # (1-0.97) - 0 fee - 0 haircut


def test_active_dispute_rejected_outright(cfg, ledger):
    """A challenged answer is a live bet, not a near-riskless carry."""
    res = make_res(cfg, ledger)
    for status in ("challenged", "disputed"):
        assert res.reject_reason(near_market(uma_resolution_status=status)) \
            == "UMA dispute active"


def test_no_oracle_signal_keeps_old_heuristics(cfg, ledger):
    """Without a status the original volume/imminence gates still guard."""
    res = make_res(cfg, ledger)
    assert "volume" in res.reject_reason(near_market(volume_24h_usd=100))
    assert res.evaluate(near_market()) is not None       # baseline unchanged


def test_market_model_parses_uma_fields():
    from polymarket_bot.models import Market
    from .conftest import gamma_raw_market
    raw = gamma_raw_market(umaResolutionStatus="Proposed", conditionId="0xabc")
    m = Market.from_gamma(raw)
    assert m.uma_resolution_status == "proposed"          # normalized to lower
    assert m.condition_id == "0xabc"
