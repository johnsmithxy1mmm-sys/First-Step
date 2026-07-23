"""Lead-lag radar: returns-correlation, lag detection, laggard identification."""

from polymarket_bot.leadlag import analyze, best_lag


def _shift(levels, lag):
    """A laggard that repeats `levels` `lag` steps later (constant fill before)."""
    return [levels[0]] * lag + levels[:-lag] if lag else list(levels)


def test_best_lag_finds_the_shift():
    leader = [0.50 + 0.01 * (i % 7) for i in range(60)]     # a moving series
    laggard = _shift(leader, 3)
    lag, corr = best_lag(leader, laggard, max_lag=6)
    assert lag == 3
    assert corr > 0.9


def test_contemporaneous_move_has_lag_zero():
    leader = [0.4 + 0.02 * (i % 5) for i in range(40)]
    lag, corr = best_lag(leader, list(leader), max_lag=6)
    assert lag == 0 and corr > 0.99


def test_flat_laggard_has_no_correlation():
    leader = [0.4 + 0.02 * (i % 5) for i in range(40)]
    lag, corr = best_lag(leader, [0.5] * 40, max_lag=6)
    assert corr == 0.0


def _series(levels, start=1000.0, step=30.0):
    return [(start + i * step, v) for i, v in enumerate(levels)]


def test_analyze_reports_leader_and_laggard():
    base = [0.5 + 0.01 * (i % 9) for i in range(150)]
    series = {
        "fast": _series(base),
        "slow": _series(_shift(base, 2)),
    }
    rows = analyze(series, step_sec=30.0, min_corr=0.5)
    assert rows
    top = rows[0]
    assert top.leader == "fast" and top.laggard == "slow"
    assert top.lag_steps >= 1


def test_analyze_ignores_pure_comovement():
    """Identical simultaneous series -> lag 0 -> not reported (co-movement)."""
    base = [0.5 + 0.01 * (i % 9) for i in range(150)]
    series = {"a": _series(base), "b": _series(base)}
    assert analyze(series, min_corr=0.5) == []


def test_analyze_survives_short_series():
    assert analyze({"a": [(1.0, 0.5), (2.0, 0.5)]}) == []   # too short -> dropped


def test_token_series_from_store(tmp_path):
    from polymarket_bot.tickstore import TickStore
    ts = TickStore(str(tmp_path / "t.sqlite"))
    for i in range(5):
        ts.record_tick("tokA", 0.40 + i * 0.01, 0.42 + i * 0.01, 10, 10, ts=float(i))
    ts.flush_now()
    series = ts.token_series(["tokA", "missing"])
    assert "missing" not in series
    assert len(series["tokA"]) == 5
    assert series["tokA"][0][1] == (0.40 + 0.42) / 2       # mid of first tick
    ts.close()
