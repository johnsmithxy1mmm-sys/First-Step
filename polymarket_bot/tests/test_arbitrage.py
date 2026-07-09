"""Арбитраж neg-risk корзин: детекция YES/NO, сайзинг по глубине, исполнение."""

from unittest import mock

import pytest

from polymarket_bot.arbitrage import ArbitrageScanner
from polymarket_bot.models import BookLevel, OrderBook

from .conftest import make_market


def negrisk_group(yes_prices, event_id="ev1"):
    return [
        make_market(
            id=f"m{i}", question=f"Will candidate {i} win?",
            outcome_prices=[p, 1 - p],
            clob_token_ids=[f"m{i}-yes", f"m{i}-no"],
            event_id=event_id, event_neg_risk=True, event_title="Election",
            volume_24h_usd=10_000,
        )
        for i, p in enumerate(yes_prices)
    ]


def book(ask: float, depth: float = 1000, bid: float | None = None) -> OrderBook:
    bid = bid if bid is not None else max(ask - 0.01, 0.001)
    return OrderBook(bids=[BookLevel(price=bid, size=depth)],
                     asks=[BookLevel(price=ask, size=depth)])


def make_scanner(cfg, ledger, books: dict, trader=None) -> ArbitrageScanner:
    clob = mock.Mock()
    clob.order_book.side_effect = lambda token: books.get(token)
    return ArbitrageScanner(cfg, ledger, clob, trader,
                            "live" if trader else "dry-run")


def test_prefilter_catches_skewed_baskets(cfg, ledger):
    scanner = make_scanner(cfg, ledger, {})
    cheap = negrisk_group([0.20, 0.30, 0.40])        # сумма 0.90 — подозрительно
    fair = negrisk_group([0.30, 0.30, 0.40], "ev2")  # сумма 1.00 — норм
    groups = scanner.prefilter_events(cheap + fair)
    assert len(groups) == 1
    assert groups[0][0].event_id == "ev1"


def test_yes_basket_detected_and_priced(cfg, ledger):
    group = negrisk_group([0.20, 0.30, 0.40])
    books = {}
    for i, ask in enumerate([0.21, 0.31, 0.41]):     # сумма ask = 0.93
        books[f"m{i}-yes"] = book(ask)
        books[f"m{i}-no"] = book(1 - ask + 0.02)     # NO дорогие — NO-корзина не выгодна
    scanner = make_scanner(cfg, ledger, books)
    arb = scanner.verify(group)
    assert arb is not None and arb.side == "YES"
    assert arb.cost_per_set == pytest.approx(0.93)
    assert arb.payout_per_set == 1.0
    assert arb.profit_pct == pytest.approx(0.07 / 0.93, rel=1e-6)


def test_no_basket_detected_when_yes_sum_above_one(cfg, ledger):
    # Yes-цены в сумме 1.10: покупка NO всех исходов платит n-1=2 за комплект.
    group = negrisk_group([0.30, 0.40, 0.40])
    books = {}
    for i, yes_ask in enumerate([0.32, 0.42, 0.42]):
        books[f"m{i}-yes"] = book(yes_ask)
        books[f"m{i}-no"] = book(1 - yes_ask + 0.03)  # NO ask: 0.71, 0.61, 0.61 = 1.93 < 2
    scanner = make_scanner(cfg, ledger, books)
    arb = scanner.verify(group)
    assert arb is not None and arb.side == "NO"
    assert arb.payout_per_set == 2.0
    assert arb.profit_per_set == pytest.approx(2.0 - 1.93, abs=1e-9)


def test_fair_books_no_arb(cfg, ledger):
    group = negrisk_group([0.30, 0.30, 0.40])
    books = {}
    for i, ask in enumerate([0.34, 0.34, 0.44]):     # сумма 1.12, NO тоже дорогие
        books[f"m{i}-yes"] = book(ask)
        books[f"m{i}-no"] = book(1 - ask + 0.05)
    assert make_scanner(cfg, ledger, books).verify(group) is None


def test_min_profit_threshold(cfg, ledger):
    cfg.arbitrage.min_profit_pct = 0.10              # требуем 10%
    group = negrisk_group([0.20, 0.30, 0.40])
    books = {}
    for i, ask in enumerate([0.21, 0.31, 0.41]):     # edge 7.5% < 10%
        books[f"m{i}-yes"] = book(ask)
        books[f"m{i}-no"] = book(0.95)
    assert make_scanner(cfg, ledger, books).verify(group) is None


def test_sizing_limited_by_depth_and_stake(cfg, ledger):
    cfg.arbitrage.execute = True
    cfg.arbitrage.max_stake_usd = 30.0
    group = negrisk_group([0.20, 0.30, 0.40])
    books = {}
    for i, ask in enumerate([0.21, 0.31, 0.41]):
        books[f"m{i}-yes"] = book(ask, depth=40)     # глубина 40 комплектов
        books[f"m{i}-no"] = book(0.95)
    scanner = make_scanner(cfg, ledger, books)
    arb = scanner.verify(group)
    spent = scanner.execute(arb)
    # stake-кэп: 30 / 0.93 = 32 комплекта < глубины 40.
    assert spent == pytest.approx(32 * 0.93, rel=1e-6)
    trades = ledger.open_positions("dry-run")
    assert len(trades) == 3                          # три ноги
    assert all(t.size == 32 for t in trades)


def test_execute_skips_tiny_windows(cfg, ledger):
    cfg.arbitrage.execute = True
    group = negrisk_group([0.20, 0.30, 0.40])
    books = {}
    for i, ask in enumerate([0.21, 0.31, 0.41]):
        books[f"m{i}-yes"] = book(ask, depth=3)      # глубина 3 < min_sets 5
        books[f"m{i}-no"] = book(0.95)
    scanner = make_scanner(cfg, ledger, books)
    arb = scanner.verify(group)
    assert arb is None or scanner.execute(arb) == 0.0
