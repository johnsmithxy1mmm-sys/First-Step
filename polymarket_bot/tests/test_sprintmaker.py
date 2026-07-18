"""Short-dated MM: hours-window selection, tight risk profile, tagged fills."""

from datetime import datetime, timedelta, timezone
from unittest import mock

from polymarket_bot.sprintmaker import SprintMaker, SprintScorer
from polymarket_bot.ws_feed import TopOfBook

from .conftest import make_market


def in_hours(h: float) -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=h)


def sprint_market(hours=24.0, **overrides):
    defaults = dict(
        id="sp1", question="Will the Lakers beat the Celtics tonight?",
        outcome_prices=[0.45, 0.55],
        clob_token_ids=["sp1-yes", "sp1-no"],
        volume_24h_usd=200_000, volume_usd=2_000_000,
        end_date=in_hours(hours),
        resolution_source="https://nba.com",
    )
    defaults.update(overrides)
    return make_market(**defaults)


def top(bid=0.43, ask=0.47, bid_size=1000.0, ask_size=1000.0) -> TopOfBook:
    return TopOfBook(bid=bid, bid_size=bid_size, ask=ask, ask_size=ask_size, ts=1.0)


def make_sprint(cfg, ledger, mode="dry-run", tops=None) -> SprintMaker:
    cfg.sprint_mm.enabled = True
    clob = mock.Mock()
    clob.order_book.return_value = None
    source = (lambda t: tops.get(t)) if tops is not None else None
    return SprintMaker(cfg, ledger, clob, None, mode, top_source=source)


# --- selection: the hours window ---

def test_accepts_liquid_market_inside_window(cfg):
    s = SprintScorer(cfg)
    assert s.reject_reason(sprint_market(hours=24)) is None


def test_rejects_too_far_out(cfg):
    s = SprintScorer(cfg)
    # 5 days out — the core MM would love this, sprint refuses it.
    assert s.reject_reason(sprint_market(hours=120)) == "resolution > 48h away"


def test_rejects_settlement_window(cfg):
    s = SprintScorer(cfg)
    # 30 min to resolution — direction dominates, stay out.
    r = s.reject_reason(sprint_market(hours=0.5))
    assert r is not None and "settlement window" in r


def test_rejects_illiquid(cfg):
    s = SprintScorer(cfg)
    r = s.reject_reason(sprint_market(hours=24, volume_24h_usd=1_000))
    assert r is not None and "24h volume" in r


def test_non_rewards_market_still_passes(cfg):
    """Short markets are rarely in rewards; spread+rebate is still +EV."""
    s = SprintScorer(cfg)
    m = sprint_market(hours=24, rewards_min_size=0.0, rewards_max_spread=0.0)
    assert not m.in_rewards_program
    assert s.reject_reason(m) is None


# --- the sprint maker binds the sprint profile, not the core MM's ---

def test_binds_sprint_config(cfg, ledger):
    cfg.sprint_mm.max_hours_to_resolution = 12.0
    s = make_sprint(cfg, ledger)
    assert s._cfg is cfg.sprint_mm
    assert isinstance(s._scorer, SprintScorer)
    # A 24h market is now outside the (tightened) 12h window.
    assert s._scorer.reject_reason(sprint_market(hours=24)) == "resolution > 12h away"


def test_cycle_quotes_a_short_dated_market(cfg, ledger):
    tops = {"sp1-yes": top(), "sp1-no": top(bid=0.53, ask=0.57)}
    s = make_sprint(cfg, ledger, tops=tops)
    quotes = s.cycle([sprint_market(hours=24)])
    assert len(quotes) == 1
    assert quotes[0].yes_bid < quotes[0].implied_yes_ask   # a real two-sided spread


def test_fills_tagged_sprint_mm(cfg, ledger):
    s = make_sprint(cfg, ledger)
    m = sprint_market(hours=24)
    s._record_fill(m, 0, 0.44, 50, None, "paper-filled")
    positions = ledger.open_positions("dry-run")
    assert len(positions) == 1
    assert positions[0].category == "sprint_mm"
