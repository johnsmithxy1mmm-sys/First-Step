"""ClobReader.order_book: parse, empty book, and quiet 404 vs loud real errors."""

import logging
from unittest import mock

import httpx

from polymarket_bot.clob import ClobReader
from polymarket_bot.config import BotConfig


def make_reader():
    return ClobReader(BotConfig(), client=mock.Mock())


def _status_error(code: int) -> httpx.HTTPStatusError:
    req = httpx.Request("GET", "https://clob.polymarket.com/book")
    resp = httpx.Response(code, request=req)
    return httpx.HTTPStatusError(f"status {code}", request=req, response=resp)


def test_parses_book():
    r = make_reader()
    resp = mock.Mock()
    resp.json.return_value = {"bids": [{"price": "0.40", "size": "100"}],
                              "asks": [{"price": "0.42", "size": "50"}]}
    with mock.patch("polymarket_bot.clob.get_with_backoff", return_value=resp):
        book = r.order_book("tok")
    assert book.best_bid == 0.40 and book.best_ask == 0.42


def test_empty_book_is_none():
    r = make_reader()
    resp = mock.Mock()
    resp.json.return_value = {"bids": [], "asks": []}
    with mock.patch("polymarket_bot.clob.get_with_backoff", return_value=resp):
        assert r.order_book("tok") is None


def test_404_returns_none_without_warning(caplog):
    r = make_reader()
    with mock.patch("polymarket_bot.clob.get_with_backoff",
                    side_effect=_status_error(404)), \
            caplog.at_level(logging.WARNING, logger="polymarket_bot.clob"):
        assert r.order_book("375684978951539001264730") is None
    assert caplog.records == []            # a missing book is expected, not a warning


def test_real_http_error_still_warns(caplog):
    r = make_reader()
    with mock.patch("polymarket_bot.clob.get_with_backoff",
                    side_effect=_status_error(500)), \
            caplog.at_level(logging.WARNING, logger="polymarket_bot.clob"):
        assert r.order_book("tok") is None
    assert any("book" in rec.message for rec in caplog.records)   # 500 stays loud


def test_network_error_still_warns(caplog):
    r = make_reader()
    with mock.patch("polymarket_bot.clob.get_with_backoff",
                    side_effect=httpx.ConnectError("boom")), \
            caplog.at_level(logging.WARNING, logger="polymarket_bot.clob"):
        assert r.order_book("tok") is None
    assert caplog.records                  # connection failures are worth seeing
