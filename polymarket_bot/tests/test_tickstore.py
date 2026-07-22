"""TickStore: non-blocking recording, category series, retention, drop-oldest."""

import time

from polymarket_bot.tickstore import TickStore


def test_records_and_reads_back_category_series(tmp_path):
    ts = TickStore(str(tmp_path / "t.sqlite"))
    ts.record_category_index("crypto", 0.02, 10, ts=100.0)
    ts.record_category_index("crypto", 0.03, 10, ts=200.0)
    ts.record_category_index("economy", -0.01, 8, ts=100.0)
    ts.flush_now()
    series = ts.category_series()
    assert [v for _, v in series["crypto"]] == [0.02, 0.03]      # oldest first
    assert series["economy"] == [(100.0, -0.01)]
    ts.close()


def test_records_ticks(tmp_path):
    ts = TickStore(str(tmp_path / "t.sqlite"))
    for i in range(5):
        ts.record_tick("tok", 0.4 + i * 0.01, 0.42, 100, 100, ts=float(i))
    ts.flush_now()
    assert ts.tick_count() == 5
    ts.close()


def test_drop_oldest_on_backpressure_never_blocks(tmp_path):
    # Tiny queue: producers must not block when the writer is not draining.
    ts = TickStore(str(tmp_path / "t.sqlite"), max_queue=10)
    for i in range(100):
        ts.record_tick("tok", 0.4, 0.42, 1, 1, ts=float(i))   # 90 dropped, no hang
    assert ts.dropped >= 80
    ts.flush_now()
    assert ts.tick_count() <= 10
    ts.close()


def test_retention_prunes_old_rows(tmp_path):
    ts = TickStore(str(tmp_path / "t.sqlite"), retention_days=1.0)
    now = time.time()
    ts.record_tick("tok", 0.4, 0.42, 1, 1, ts=now - 3 * 86_400)  # 3 days old
    ts.record_tick("tok", 0.5, 0.52, 1, 1, ts=now)               # fresh
    ts.flush_now()
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "t.sqlite"))
    ts._last_prune = 0.0
    ts._prune(conn, now)
    conn.commit()
    remaining = conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
    conn.close()
    assert remaining == 1
    ts.close()


def test_reader_safe_before_any_write(tmp_path):
    ts = TickStore(str(tmp_path / "none.sqlite"))
    assert ts.category_series() == {}      # no table yet -> empty, not a crash
    assert ts.tick_count() == 0
    ts.close()
