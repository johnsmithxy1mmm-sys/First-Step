"""Counterfactual replay: passive-fill capture over recorded top-of-book."""

from polymarket_bot.timemachine import simulate_spread_capture, sweep


def bookrow(ts, bid, ask, bsz=100, asz=100):
    return (float(ts), bid, ask, bsz, asz)


def test_no_fill_when_market_never_crosses_down():
    # Ask stays well above our bid -> never fills.
    series = [bookrow(i, 0.49, 0.51) for i in range(20)]
    r = simulate_spread_capture(series, half_spread=0.02, fill_horizon=4)
    assert r.placements > 0 and r.fills == 0
    assert r.capture_per_placement == 0.0


def test_fill_when_ask_dips_to_our_bid():
    # Flat at 0.50 mid, then the ask dips to 0.48 (== mid-0.02) so our bid fills.
    series = [bookrow(0, 0.49, 0.51)]
    series += [bookrow(1, 0.47, 0.48)]           # ask 0.48 crosses our 0.48 bid
    series += [bookrow(i, 0.49, 0.51) for i in range(2, 8)]   # mid back to 0.50
    r = simulate_spread_capture(series, half_spread=0.02, fill_horizon=2,
                                markout_horizon=3)
    assert r.fills >= 1
    # Bought at 0.48, mid recovered to 0.50 -> captured ~half_spread + drift > 0.
    assert r.avg_capture_per_fill > 0


def test_adverse_selection_shows_as_negative_capture():
    # Our bid fills, then the mid keeps FALLING (we caught a falling knife).
    series = [bookrow(0, 0.49, 0.51)]            # mid 0.50, place bid 0.48
    series += [bookrow(1, 0.47, 0.48)]           # fill at 0.48
    series += [bookrow(2, 0.40, 0.42)]           # mid collapses to 0.41
    series += [bookrow(3, 0.38, 0.40)]
    series += [bookrow(4, 0.36, 0.38)]
    r = simulate_spread_capture(series, half_spread=0.02, fill_horizon=2,
                                markout_horizon=3)
    assert r.fills >= 1
    assert r.avg_capture_per_fill < 0            # spread did not survive the drop


def test_sweep_ranks_by_capture_per_placement():
    # A gently oscillating mid: a tighter spread fills more often.
    series = []
    for i in range(60):
        mid = 0.50 + (0.01 if i % 2 else -0.01)
        series.append(bookrow(i, mid - 0.015, mid + 0.015))
    rows = sweep({"tok": series}, half_spreads=[0.005, 0.01, 0.02, 0.03])
    assert rows == sorted(rows, key=lambda r: r.capture_per_placement, reverse=True)
    assert rows[0].placements > 0


def test_token_book_series_round_trip(tmp_path):
    from polymarket_bot.tickstore import TickStore
    ts = TickStore(str(tmp_path / "t.sqlite"))
    for i in range(4):
        ts.record_tick("tok", 0.49, 0.51, 100, 120, ts=float(i))
    ts.flush_now()
    series = ts.token_book_series(["tok"])
    assert len(series["tok"]) == 4
    assert series["tok"][0][1:3] == (0.49, 0.51)      # bid, ask preserved
    assert series["tok"][0][3:5] == (100, 120)        # sizes preserved
    ts.close()
