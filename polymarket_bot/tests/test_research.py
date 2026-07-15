"""Research loop: fill simulator, Sharpe allocation, walk-forward search."""

from polymarket_bot.research import (sharpe_allocation, simulate_maker_fill,
                                     walk_forward_grid)


# --- realistic maker fill ---

def test_fill_needs_trade_through():
    # A BUY at 0.50: trades above 0.50 do NOT fill us.
    assert simulate_maker_fill(0.50, "BUY", [(0.55, 100)], our_size=50) == 0.0
    # A trade at/through 0.50 fills us.
    assert simulate_maker_fill(0.50, "BUY", [(0.50, 100)], our_size=50) == 50


def test_fill_respects_queue_ahead():
    # 80 ahead of us; a 100 trade eats the queue then 20 of ours.
    assert simulate_maker_fill(0.50, "BUY", [(0.49, 100)], our_size=50,
                               queue_ahead=80) == 20


def test_fill_sell_side():
    assert simulate_maker_fill(0.60, "SELL", [(0.62, 40)], our_size=50) == 40
    assert simulate_maker_fill(0.60, "SELL", [(0.58, 40)], our_size=50) == 0.0


# --- Sharpe allocation ---

def test_sharpe_allocation_prefers_steady_earner():
    steady = [0, 10, 20, 30, 40]          # smooth up
    choppy = [0, 30, -10, 40, 0]          # same-ish mean, high vol
    w = sharpe_allocation({"steady": steady, "choppy": choppy})
    assert w["steady"] > w["choppy"]
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_sharpe_allocation_all_losing_is_equal():
    w = sharpe_allocation({"a": [0, -5, -10], "b": [0, -3, -9]})
    assert abs(w["a"] - w["b"]) < 1e-9    # no positive Sharpe -> equal weight


# --- walk-forward grid ---

def test_walk_forward_grid_finds_optimum():
    # Score peaks at x=3 regardless of split.
    def evaluate(params, i):
        return -abs(params["x"] - 3)
    ranked = walk_forward_grid({"x": [1, 2, 3, 4, 5]}, evaluate, n_splits=3)
    assert ranked[0][0]["x"] == 3
