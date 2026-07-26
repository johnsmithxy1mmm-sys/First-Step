"""F-001: WS trust boundary accepts NaN/Infinity prices (fails on HEAD).

Python's json.loads parses bare NaN/Infinity by default, and BookStore performs
no numeric validation. Consequences proven here:

  (a) NaN price keys accumulate UNREMOVABLY: on py>=3.10 hash(nan) is
      per-object, so a size=0 delete for "the same" NaN never matches a stored
      key -> unbounded growth from a repeating malformed feed.
  (b) max() over keys containing NaN is insertion-order dependent -> top.bid
      can BECOME NaN, and every consumer downstream trusts it.
  (c) Infinity flows straight through to top.bid/mid -> a poisoned tick makes
      _exit_one see mark=inf, i.e. a fake take-profit trigger.
"""

import json
import math

import pytest

from polymarket_bot.ws_feed import BookStore


def _poison(price: str, size: str = "7") -> dict:
    # Through the REAL parser on purpose: json.loads is what admits NaN/Infinity.
    return json.loads(
        '{"event_type":"price_change","asset_id":"tok","changes":'
        f'[{{"price":"{price}","size":"{size}","side":"BUY"}}]}}')


def test_nan_keys_are_unremovable_and_accumulate():
    store = BookStore()
    store.handle({"event_type": "book", "asset_id": "tok",
                  "bids": [{"price": "0.44", "size": "100"}],
                  "asks": [{"price": "0.46", "size": "80"}]})
    for _ in range(5):
        store.handle(_poison("NaN"))
    store.handle(_poison("NaN", size="0"))          # the "delete" for NaN
    nan_keys = [k for k in store._bids["tok"] if math.isnan(k)]
    assert nan_keys == [], (
        f"{len(nan_keys)} NaN keys survive their own delete — unbounded growth")


def test_nan_first_in_book_makes_top_bid_nan():
    store = BookStore()
    store.handle({"event_type": "book", "asset_id": "tok",
                  "bids": [{"price": "NaN", "size": "5"},
                           {"price": "0.44", "size": "100"}],
                  "asks": [{"price": "0.46", "size": "80"}]})
    top = store.top("tok")
    assert not math.isnan(top.bid), "top.bid is NaN — every consumer trusts this"


def test_infinity_reaches_mid_and_the_exit_trigger():
    store = BookStore()
    store.handle({"event_type": "book", "asset_id": "t2",
                  "bids": [{"price": "Infinity", "size": "5"}],
                  "asks": [{"price": "0.5", "size": "5"}]})
    top = store.top("t2")
    # on_tick gates exits with `top.bid > 0` — inf passes and becomes `mark`.
    assert math.isfinite(top.bid), "top.bid=inf passes the >0 gate into _exit_one"
