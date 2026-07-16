"""Event-aware netting: neg-risk baskets count worst-case, not the sum."""

from polymarket_bot.models import Position, simple_estimate
from polymarket_bot.portfolio import (Portfolio, event_exposure_breakdown,
                                       event_netted_exposure, risk_group_key)

from .conftest import make_market


def pos(event_id="e1", neg_risk=True, token="t1", cost=100.0, category="other"):
    return Position(token_id=token, market_id="m", question="Will X win?",
                    outcome="No", category=category, size=cost, avg_price=1.0,
                    event_id=event_id, neg_risk=neg_risk)


def test_neg_risk_basket_nets_to_max():
    ps = [pos(token="a", cost=100), pos(token="b", cost=120), pos(token="c", cost=90)]
    # Exactly one outcome wins -> at most one NO leg loses -> worst-case = largest.
    assert event_netted_exposure(ps)["other"] == 120.0   # not 310


def test_non_neg_risk_sums():
    ps = [pos(token="a", cost=100, neg_risk=False, event_id=""),
          pos(token="b", cost=120, neg_risk=False, event_id="")]
    assert event_netted_exposure(ps)["other"] == 220.0


def test_different_events_do_not_net():
    ps = [pos(event_id="e1", token="a", cost=100),
          pos(event_id="e2", token="b", cost=120)]
    assert event_netted_exposure(ps)["other"] == 220.0


def test_risk_group_key():
    assert risk_group_key(pos(event_id="e1", neg_risk=True))[0] == "event"
    assert risk_group_key(pos(event_id="", neg_risk=True))[0] == "solo"
    assert risk_group_key(pos(event_id="e1", neg_risk=False))[0] == "solo"


def test_breakdown_reports_gross_and_worst():
    ps = [pos(token="a", cost=100), pos(token="b", cost=120)]
    rows = event_exposure_breakdown(ps)
    assert len(rows) == 1
    label, legs, gross, worst = rows[0]
    assert legs == 2 and gross == 220.0 and worst == 120.0


def test_ledger_round_trips_event_and_neg_risk(cfg, ledger):
    m = make_market(id="m1", clob_token_ids=["y1", "n1"], event_id="ev",
                    event_neg_risk=True, question="Will player A win the award?")
    est = simple_estimate(m, 1, 0.97)   # NO side
    ledger.record_trade(mode="dry-run", estimate=est, category="other",
                        side="BUY", price=0.97, size=100, order_id=None,
                        status="sim-filled", strategy="fade")
    p = ledger.open_positions("dry-run")[0]
    assert p.event_id == "ev" and p.neg_risk is True


def test_backfill_heals_legacy_rows(cfg, ledger):
    """Rows written before the neg_risk column (or flag) net after backfill."""
    for i in range(3):
        m = make_market(id=f"lg{i}", clob_token_ids=[f"y{i}", f"n{i}"],
                        event_id="ev-old", event_neg_risk=False,   # legacy: flag unknown
                        question="Will candidate X win the award?")
        ledger.record_trade(mode="dry-run", estimate=simple_estimate(m, 1, 0.97),
                            category="other", side="BUY", price=0.97, size=100,
                            order_id=None, status="sim-filled", strategy="fade")
    before = event_netted_exposure(ledger.open_positions("dry-run"))["other"]
    assert before == 291.0                              # 3 x 97, no netting
    # Live metadata now says these markets' event IS neg-risk.
    healed = ledger.backfill_neg_risk(["lg0", "lg1", "lg2"])
    assert healed == 3
    after = event_netted_exposure(ledger.open_positions("dry-run"))["other"]
    assert after == 97.0                                # basket nets to one leg
    assert ledger.backfill_neg_risk(["lg0", "lg1", "lg2"]) == 0   # idempotent


def test_var95_never_exceeds_max_loss():
    from polymarket_bot.risk2 import portfolio_stress
    ps = [pos(cost=500, category="geopolitics", neg_risk=False, event_id="", token="a"),
          pos(cost=500, category="economy", neg_risk=False, event_id="", token="b")]
    st = portfolio_stress(ps)
    assert st["var95_usd"] <= st["worst_case_usd"]


def test_size_usd_uses_netted_category_room(cfg, ledger):
    cfg.portfolio.bankroll_usd = 5000
    cfg.portfolio.max_category_pct = 0.10   # $500 category cap
    port = Portfolio(cfg, ledger, "dry-run")
    # 3 neg-risk NO legs of ONE event, $200 each = $600 gross but $200 worst-case.
    for i in range(3):
        m = make_market(id=f"m{i}", clob_token_ids=[f"y{i}", f"n{i}"],
                        event_id="ev", event_neg_risk=True, category="other",
                        question="Will player X win the award?")
        est = simple_estimate(m, 1, 1.0)
        ledger.record_trade(mode="dry-run", estimate=est, category="other",
                            side="BUY", price=1.0, size=200, order_id=None,
                            status="sim-filled", strategy="fade")
    assert event_netted_exposure(ledger.open_positions("dry-run"))["other"] == 200.0
    # Netted exposure $200 < $500 cap -> room remains, so a new size is allowed.
    assert port.size_usd("other", 0.98, 0.5) is not None
