"""Markout-аналитика и отчёт."""

from datetime import datetime, timedelta, timezone

import pytest

from polymarket_bot.analytics import compute_report
from polymarket_bot.estimator.ensemble import combine
from polymarket_bot.models import Signal, simple_estimate

from .conftest import make_candidate, make_market


def backdate_trade(ledger, trade_id: int, seconds: float) -> None:
    """Сдвигает время сделки в прошлое (симуляция прошедшего горизонта)."""
    ts = (datetime.now(timezone.utc) - timedelta(seconds=seconds)) \
        .isoformat(timespec="seconds")
    ledger._conn.execute("UPDATE trades SET ts = ? WHERE id = ?", (ts, trade_id))
    ledger._conn.commit()


def record_fill(ledger, mode="paper", price=0.44, strategy="mm") -> int:
    m = make_market(id=f"m-{strategy}", clob_token_ids=["tok-x", "tok-x-no"])
    ledger.record_trade(mode=mode, estimate=simple_estimate(m, 0, price),
                        category="mm", side="BUY", price=price, size=100,
                        order_id=None, status="paper-filled", strategy=strategy)
    row = ledger._conn.execute("SELECT MAX(id) AS id FROM trades").fetchone()
    return int(row["id"])


def test_fills_needing_markout_respects_horizon(ledger):
    trade_id = record_fill(ledger)
    # Филл только что: горизонт 60с ещё не наступил.
    assert ledger.fills_needing_markout("paper", 60) == []
    backdate_trade(ledger, trade_id, seconds=90)
    pending = ledger.fills_needing_markout("paper", 60)
    assert [f["id"] for f in pending] == [trade_id]
    # Для горизонта 600с — рано.
    assert ledger.fills_needing_markout("paper", 600) == []


def test_markout_recorded_once_and_stats(ledger):
    trade_id = record_fill(ledger, price=0.44)
    backdate_trade(ledger, trade_id, seconds=90)

    ledger.record_markout(trade_id, "tok-x", 60, fill_price=0.44, mark_price=0.42)
    # Дубль не пишется (UNIQUE), из очереди филл ушёл.
    ledger.record_markout(trade_id, "tok-x", 60, fill_price=0.44, mark_price=0.50)
    assert ledger.fills_needing_markout("paper", 60) == []

    stats = ledger.markout_stats("paper")
    assert len(stats) == 1
    s = stats[0]
    assert s["strategy"] == "mm" and s["horizon_sec"] == 60 and s["n"] == 1
    # Купили по 0.44, через минуту 0.42: markout -0.02 — adverse selection.
    assert s["avg_markout"] == pytest.approx(-0.02)
    assert s["avg_markout_pct"] == pytest.approx(-0.02 / 0.44)
    assert s["favorable"] == 0


def test_markout_stats_split_by_strategy(ledger):
    good = record_fill(ledger, strategy="mm", price=0.40)
    bad = record_fill(ledger, strategy="btc_5m", price=0.50)
    for tid in (good, bad):
        backdate_trade(ledger, tid, seconds=90)
    ledger.record_markout(good, "tok-x", 60, 0.40, 0.43)   # +0.03 в нашу сторону
    ledger.record_markout(bad, "tok-x", 60, 0.50, 0.45)    # -0.05 против
    stats = {s["strategy"]: s for s in ledger.markout_stats("paper")}
    assert stats["mm"]["favorable"] == 1
    assert stats["btc_5m"]["avg_markout"] == pytest.approx(-0.05)


def test_compute_report_smoke(ledger):
    """Отчёт собирается на леджере с данными всех видов и не падает на пустом."""
    empty = compute_report(ledger, "paper")
    assert empty["equity_now"] is None
    assert empty["positions"] == []

    # Наполняем: банк, лонгшот с резолюцией, mm-филл с markout, оценка.
    ledger.snapshot_bank(cash=5000, exposure=0)
    est = combine(make_candidate(), [Signal(name="s", p_est=0.05, confidence=1e9)], 1e-9)
    ledger.record_estimate(est, qualifies=True)
    ledger.record_trade(mode="paper", estimate=est, category="nature", side="BUY",
                        price=0.01, size=1000, order_id=None, status="paper-filled",
                        strategy="longshot")
    ledger.record_resolution(est.candidate.token_id, "m1", won=True)
    trade_id = record_fill(ledger)
    backdate_trade(ledger, trade_id, seconds=90)
    ledger.record_markout(trade_id, "tok-x", 60, 0.44, 0.46)
    ledger.snapshot_bank(cash=5990, exposure=0)

    report = compute_report(ledger, "paper")
    assert report["equity_start"] == pytest.approx(5000)
    assert report["equity_now"] == pytest.approx(5990)
    assert report["longshot"]["resolved_trades"] == 1
    assert report["longshot"]["hit_rate"] == 1.0
    assert report["estimates"]["total"] == 1
    assert len(report["markouts"]) == 1
    assert report["pnl_by_strategy"]["longshot"] == pytest.approx(1000 - 10)


def test_max_drawdown_math(ledger):
    for equity in (5000, 5500, 4400, 5000):   # пик 5500 -> дно 4400 = -20%
        ledger.snapshot_bank(cash=equity, exposure=0)
    report = compute_report(ledger, "paper")
    assert report["max_drawdown_pct"] == pytest.approx(0.2, rel=1e-6)
