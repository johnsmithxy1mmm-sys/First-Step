"""Арбитраж neg-risk корзин: детекция, комиссии (net edge), suspect, сайзинг."""

from unittest import mock

import pytest

from polymarket_bot.arbitrage import ArbLeg, ArbitrageScanner, BasketArb
from polymarket_bot.models import BookLevel, OrderBook

from .conftest import make_market


def negrisk_group(yes_prices, event_id="ev1", category=""):
    return [
        make_market(
            id=f"m{i}", question=f"Will candidate {i} win?",
            outcome_prices=[p, 1 - p],
            clob_token_ids=[f"m{i}-yes", f"m{i}-no"],
            event_id=event_id, event_neg_risk=True, event_title="Election",
            volume_24h_usd=10_000, category=category,
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


# --- чистая математика комиссий (без сети) ---

def test_fee_math_net_below_gross():
    leg = lambda ask: ArbLeg(market=make_market(), outcome_index=0,
                             token_id="t", ask=ask, depth=1000)
    arb = BasketArb(event_id="e", event_title="World Cup Winner", side="YES",
                    legs=[leg(0.33), leg(0.33), leg(0.32)], taker_fee=0.03)
    assert arb.cost_per_set == pytest.approx(0.98)
    assert arb.profit_pct == pytest.approx(0.02 / 0.98)     # gross +2%
    assert arb.fee_per_set == pytest.approx(0.03 * 0.98)
    # Ровно сценарий ЧМ: +2% gross, но 3% комиссии -> чистый УБЫТОК.
    assert arb.net_profit_per_set == pytest.approx(0.02 - 0.0294)
    assert arb.net_profit_pct < 0


def test_marginal_arb_rejected_after_fees(cfg, ledger):
    """+2% gross на спорт-корзине (fee 3%) — после комиссий не сделка."""
    group = negrisk_group([0.32, 0.33, 0.31], category="sports")   # ~0.98
    books = {}
    for i, ask in enumerate([0.33, 0.33, 0.32]):
        books[f"m{i}-yes"] = book(ask)
        books[f"m{i}-no"] = book(0.95)
    assert make_scanner(cfg, ledger, books).verify(group) is None


def test_prefilter_catches_skewed_baskets(cfg, ledger):
    scanner = make_scanner(cfg, ledger, {})
    cheap = negrisk_group([0.20, 0.30, 0.40])        # сумма 0.90 — подозрительно
    fair = negrisk_group([0.30, 0.30, 0.40], "ev2")  # сумма 1.00 — норм
    groups = scanner.prefilter_events(cheap + fair)
    assert len(groups) == 1
    assert groups[0][0].event_id == "ev1"


def test_yes_basket_detected_with_net_and_suspect(cfg, ledger):
    group = negrisk_group([0.20, 0.30, 0.40])        # категория other -> fee 0.04
    books = {}
    for i, ask in enumerate([0.21, 0.31, 0.41]):     # сумма ask = 0.93, gross 7.5%
        books[f"m{i}-yes"] = book(ask)
        books[f"m{i}-no"] = book(1 - ask + 0.02)
    arb = make_scanner(cfg, ledger, books).verify(group)
    assert arb is not None and arb.side == "YES"
    assert arb.cost_per_set == pytest.approx(0.93)
    assert arb.profit_pct == pytest.approx(0.07 / 0.93, rel=1e-6)      # gross
    assert arb.net_profit_pct < arb.profit_pct                        # комиссии учтены
    assert arb.suspect                                                # 7.5% > 5% -> подозрительно


def test_no_basket_detected_fee_free_category(cfg, ledger):
    # Geopolitics: taker fee 0, поэтому NO-корзина выживает после комиссий.
    group = negrisk_group([0.30, 0.40, 0.40], category="geopolitics")
    books = {}
    for i, yes_ask in enumerate([0.32, 0.42, 0.42]):
        books[f"m{i}-yes"] = book(yes_ask)
        books[f"m{i}-no"] = book(1 - yes_ask + 0.03)  # NO ask 0.71,0.61,0.61 = 1.93 < 2
    arb = make_scanner(cfg, ledger, books).verify(group)
    assert arb is not None and arb.side == "NO"
    assert arb.payout_per_set == 2.0
    assert arb.net_profit_pct == pytest.approx(arb.profit_pct)         # fee 0


def test_fair_books_no_arb(cfg, ledger):
    group = negrisk_group([0.30, 0.30, 0.40])
    books = {}
    for i, ask in enumerate([0.34, 0.34, 0.44]):
        books[f"m{i}-yes"] = book(ask)
        books[f"m{i}-no"] = book(1 - ask + 0.05)
    assert make_scanner(cfg, ledger, books).verify(group) is None


def test_min_profit_threshold(cfg, ledger):
    cfg.arbitrage.min_profit_pct = 0.10              # требуем 10% чистыми
    group = negrisk_group([0.20, 0.30, 0.40], category="geopolitics")
    books = {}
    for i, ask in enumerate([0.21, 0.31, 0.41]):     # net 7.5% < 10%
        books[f"m{i}-yes"] = book(ask)
        books[f"m{i}-no"] = book(0.95)
    assert make_scanner(cfg, ledger, books).verify(group) is None


def test_suspect_basket_not_executed(cfg, ledger):
    cfg.arbitrage.execute = True
    group = negrisk_group([0.20, 0.30, 0.40], category="geopolitics")  # fee 0
    books = {}
    for i, ask in enumerate([0.21, 0.31, 0.41]):     # gross 7.5% > 5% -> suspect
        books[f"m{i}-yes"] = book(ask, depth=40)
        books[f"m{i}-no"] = book(0.95)
    scanner = make_scanner(cfg, ledger, books)
    arb = scanner.verify(group)
    assert arb is not None and arb.suspect
    assert scanner.execute(arb) == 0.0               # подозрительную не исполняем
    assert ledger.open_positions("dry-run") == []


def test_sizing_limited_by_depth_and_stake(cfg, ledger):
    cfg.arbitrage.execute = True
    cfg.arbitrage.max_stake_usd = 30.0
    # Скромный реальный edge (3.1% gross, geopolitics fee 0) — не suspect, исполняется.
    group = negrisk_group([0.31, 0.32, 0.33], category="geopolitics")
    books = {}
    for i, ask in enumerate([0.32, 0.32, 0.33]):     # сумма 0.97
        books[f"m{i}-yes"] = book(ask, depth=40)
        books[f"m{i}-no"] = book(0.95)
    scanner = make_scanner(cfg, ledger, books)
    arb = scanner.verify(group)
    assert arb is not None and not arb.suspect
    spent = scanner.execute(arb)
    sets = int(30 // 0.97)                            # stake-кэп 30/0.97 = 30 < глубины 40
    assert spent == pytest.approx(sets * 0.97, rel=1e-6)
    trades = ledger.open_positions("dry-run")
    assert len(trades) == 3 and all(t.size == sets for t in trades)


def test_execute_skips_tiny_windows(cfg, ledger):
    cfg.arbitrage.execute = True
    group = negrisk_group([0.20, 0.30, 0.40], category="geopolitics")
    books = {}
    for i, ask in enumerate([0.21, 0.31, 0.41]):
        books[f"m{i}-yes"] = book(ask, depth=3)      # глубина 3 < min_sets 5
        books[f"m{i}-no"] = book(0.95)
    scanner = make_scanner(cfg, ledger, books)
    arb = scanner.verify(group)
    assert arb is None or scanner.execute(arb) == 0.0