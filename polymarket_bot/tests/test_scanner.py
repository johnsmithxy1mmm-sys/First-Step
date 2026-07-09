"""Сканер: фильтры первого уровня и проверка глубины книги."""

from datetime import datetime, timedelta, timezone
from unittest import mock

from polymarket_bot.models import Market
from polymarket_bot.scanner import Scanner

from .conftest import make_book, make_market


def scan1(cfg, market: Market):
    return Scanner(cfg).first_level_filter([market])


def test_accepts_tail_price_in_band(cfg):
    found = scan1(cfg, make_market())
    assert len(found) == 1
    assert found[0].p_mkt == 0.01
    assert found[0].outcome == "Yes"


def test_rejects_price_outside_band(cfg):
    assert scan1(cfg, make_market(outcome_prices=[0.10, 0.90])) == []
    assert scan1(cfg, make_market(outcome_prices=[0.001, 0.999])) == []


def test_rejects_low_volume(cfg):
    assert scan1(cfg, make_market(volume_24h_usd=100)) == []


def test_rejects_bad_resolution_window(cfg):
    soon = datetime.now(timezone.utc) + timedelta(days=1)
    far = datetime.now(timezone.utc) + timedelta(days=400)
    assert scan1(cfg, make_market(end_date=soon)) == []
    assert scan1(cfg, make_market(end_date=far)) == []
    assert scan1(cfg, make_market(end_date=None)) == []


def test_rejects_ambiguous_resolution(cfg):
    ambiguous = make_market(resolution_source="", description="short")
    assert scan1(cfg, ambiguous) == []
    # Внятное описание без источника — допустимо.
    described = make_market(resolution_source="")
    assert len(scan1(cfg, described)) == 1
    # Флаг можно выключить.
    cfg.scanner.require_resolution_clarity = False
    assert len(scan1(cfg, ambiguous)) == 1


def test_rejects_closed_or_no_orderbook(cfg):
    assert scan1(cfg, make_market(closed=True)) == []
    assert scan1(cfg, make_market(enable_order_book=False)) == []


def test_keyword_filters(cfg):
    cfg.scanner.exclude_keywords = ["earthquake"]
    assert scan1(cfg, make_market()) == []
    cfg.scanner.exclude_keywords = []
    cfg.scanner.include_keywords = ["bitcoin"]
    assert scan1(cfg, make_market()) == []


def test_depth_filter(cfg):
    clob = mock.Mock()
    scanner = Scanner(cfg, clob)
    candidates = scanner.first_level_filter([make_market()])

    clob.order_book.return_value = make_book(depth=100_000)
    assert len(scanner.verify_depth(candidates)) == 1

    thin = make_book()
    thin.bids = [lvl.model_copy(update={"size": 10}) for lvl in thin.bids]
    clob.order_book.return_value = thin  # ~$0.1 глубины — меньше $500
    assert scanner.verify_depth(candidates) == []

    clob.order_book.return_value = None
    assert scanner.verify_depth(candidates) == []
