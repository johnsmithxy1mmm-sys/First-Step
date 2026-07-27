"""Ledger: positions, PnL, calibration metrics, signal attribution."""

import pytest

from polymarket_bot.estimator.ensemble import combine
from polymarket_bot.models import Signal

from .conftest import make_candidate


def make_estimate(p_mkt=0.01, p_est=0.03, signals=None, **overrides):
    c = make_candidate(outcome_prices=[p_mkt, 1 - p_mkt], **overrides)
    sigs = signals or [Signal(name="s", p_est=p_est, confidence=1.0)]
    return combine(c, sigs, 1e-9)


def buy(ledger, est, usd, mode="dry-run", category="nature", strategy="longshot"):
    size = usd / est.p_mkt
    ledger.record_trade(mode=mode, estimate=est, category=category, side="BUY",
                        price=est.p_mkt, size=size, order_id=None,
                        status="sim-filled", strategy=strategy)
    return size


def test_positions_and_exposure(ledger):
    est = make_estimate()
    size = buy(ledger, est, 100.0)
    positions = ledger.open_positions("dry-run")
    assert len(positions) == 1
    assert positions[0].size == pytest.approx(size)
    assert positions[0].avg_price == pytest.approx(0.01)
    assert ledger.total_exposure("dry-run") == pytest.approx(100.0)
    assert ledger.exposure_by_category("dry-run") == {"nature": pytest.approx(100.0)}


def test_sell_reduces_position_and_realizes_pnl(ledger):
    est = make_estimate()
    buy(ledger, est, 100.0)  # 10,000 sh at 0.01
    ledger.record_trade(mode="dry-run", estimate=est, category="nature", side="SELL",
                        price=0.08, size=6000, order_id=None, status="sim-filled",
                        strategy="longshot")
    positions = ledger.open_positions("dry-run")
    assert positions[0].size == pytest.approx(4000)
    # Sold 6000 at 0.08 (entry 0.01): realized (0.08-0.01)*6000 = 420.
    assert ledger.realized_pnl("dry-run") == pytest.approx(420.0)


def test_resolution_win_and_loss_pnl(ledger):
    est_win = make_estimate()
    est_loss = make_estimate(id="m2", clob_token_ids=["t2-yes", "t2-no"])
    buy(ledger, est_win, 100.0)   # 10,000 sh
    buy(ledger, est_loss, 50.0)   # 5,000 sh
    ledger.record_resolution(est_win.candidate.token_id, "m1", won=True)
    ledger.record_resolution(est_loss.candidate.token_id, "m2", won=False)

    # Win: 10,000*$1 - 100 = 9,900; loss: -50.
    assert ledger.realized_pnl("dry-run") == pytest.approx(9900.0 - 50.0)
    assert ledger.open_positions("dry-run") == []  # both positions closed


def test_metrics_hit_rate_brier_and_attribution(ledger):
    coherent = [Signal(name="coherence", p_est=0.04, confidence=0.9)]
    noisy = [Signal(name="momentum", p_est=0.02, confidence=0.3)]
    est_win = make_estimate(signals=coherent)
    est_loss = make_estimate(id="m2", clob_token_ids=["t2-yes", "t2-no"], signals=noisy)
    buy(ledger, est_win, 100.0)
    buy(ledger, est_loss, 100.0)
    ledger.record_resolution(est_win.candidate.token_id, "m1", won=True)
    ledger.record_resolution(est_loss.candidate.token_id, "m2", won=False)

    m = ledger.metrics("dry-run")
    assert m["resolved_trades"] == 2
    assert m["hit_rate"] == pytest.approx(0.5)
    assert m["avg_win_multiple"] == pytest.approx(100.0)   # win at 0.01
    assert m["invested_usd"] == pytest.approx(200.0)
    assert m["payout_usd"] == pytest.approx(10_000.0)
    assert m["roi"] == pytest.approx((10_000 - 200) / 200)
    assert m["brier_model"] is not None and m["brier_market"] is not None
    # Attribution: winning PnL assigned to coherence, losing to momentum.
    attribution = m["signal_pnl_attribution"]
    assert attribution["coherence"] == pytest.approx(9900.0)
    assert attribution["momentum"] == pytest.approx(-100.0)


def test_estimates_and_bank_snapshots(ledger):
    est = make_estimate()
    ledger.record_estimate(est, qualifies=True)
    ledger.snapshot_bank(cash=4000, exposure=1000)
    ledger.snapshot_bank(cash=3000, exposure=1500)
    assert ledger.high_water_mark() == pytest.approx(5000.0)


def test_idempotency_helper(ledger):
    est = make_estimate()
    assert not ledger.has_position_or_open_buy(est.candidate.token_id, "dry-run")
    buy(ledger, est, 10.0)
    assert ledger.has_position_or_open_buy(est.candidate.token_id, "dry-run")
    # A different mode — separate state.
    assert not ledger.has_position_or_open_buy(est.candidate.token_id, "live")
