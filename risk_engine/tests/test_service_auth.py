"""The service must not be servable to a network without a token.

The interesting property is not "a good token works" -- it is that the two
ways this leaks are both closed: a bad token cannot reach an endpoint, and a
bind that would expose the engine cannot happen silently. The second is the
one that matters in a deployment, because it fails at start-up rather than
at whatever hour someone scans the port.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from risk_engine.service.app import _is_loopback, serve
from risk_engine.service.state import EngineState

TOKEN = "s3cret-token-value"

BOOK = {
    "address": "0xdemo",
    "cross_collateral": 100_000.0,
    "positions": [
        {"coin": "BTC", "size": 2.0, "entry_price": 100_000.0,
         "mode": "cross", "leverage": 20},
    ],
}


@pytest.fixture(scope="module")
def secured():
    state = EngineState.fixture()
    httpd = serve(state, "127.0.0.1", 0, token=TOKEN)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(0.2)
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    state.stop()


def _get(base: str, path: str, token: str | None = None, scheme: str = "Bearer"):
    req = urllib.request.Request(f"{base}{path}")  # noqa: S310 — loopback, started here
    if token is not None:
        req.add_header("Authorization", f"{scheme} {token}")
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
        return r.status, json.load(r)


def _post(base: str, path: str, payload: dict, token: str | None = None):
    req = urllib.request.Request(  # noqa: S310 — loopback, started here
        f"{base}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=180) as r:  # noqa: S310
        return r.status, json.load(r)


class TestBearerToken:
    def test_a_valid_token_reaches_the_engine(self, secured):
        status, body = _post(
            secured, "/portfolio_risk", {"book": BOOK, "n_paths": 2_000, "seed": 1},
            token=TOKEN,
        )
        assert status == 200
        assert body["address"] == "0xdemo"

    def test_no_token_is_refused(self, secured):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(secured, "/portfolio_risk", {"book": BOOK, "n_paths": 500}, token=None)
        assert exc.value.code == 401

    def test_a_wrong_token_is_refused(self, secured):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(secured, "/metrics", token="not-the-token")
        assert exc.value.code == 401

    def test_a_token_that_is_a_prefix_of_the_real_one_is_refused(self, secured):
        """The compare is constant-time, so this is not about timing --  it is
        that a prefix must not be accepted as the whole."""
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(secured, "/metrics", token=TOKEN[:-1])
        assert exc.value.code == 401

    def test_the_right_token_under_the_wrong_scheme_is_refused(self, secured):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(secured, "/metrics", token=TOKEN, scheme="Basic")
        assert exc.value.code == 401

    def test_the_rejection_discloses_nothing(self, secured):
        """Not why it failed, and not what was presented. "Wrong scheme"
        versus "wrong token" is free information for whoever is guessing."""
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(secured, "/metrics", token="not-the-token")
        body = exc.value.read()
        assert body == b""
        assert TOKEN not in str(exc.value.headers)

    def test_metrics_needs_the_token_too(self, secured):
        """The snapshot describes model internals and request volumes; it is
        not a public endpoint just because it holds no user data."""
        status, body = _get(secured, "/metrics", token=TOKEN)
        assert status == 200
        assert "latency" in body

    def test_health_stays_open_for_orchestrators(self, secured):
        """A container healthcheck runs before any token is in scope, and the
        payload names no user and no book."""
        status, body = _get(secured, "/health", token=None)
        assert status == 200
        assert "model_version" in body


class TestBindGuard:
    def test_an_exposed_bind_without_a_token_refuses_to_start(self):
        """The failure that matters: loud, at start-up, before a single
        request is served."""
        state = EngineState.fixture()
        try:
            with pytest.raises(RuntimeError, match="refusing to bind"):
                serve(state, "0.0.0.0", 0, token="")  # noqa: S104 — the point of the test
        finally:
            state.stop()

    def test_an_exposed_bind_with_a_token_is_allowed(self):
        state = EngineState.fixture()
        try:
            httpd = serve(state, "0.0.0.0", 0, token=TOKEN)  # noqa: S104 — with a token
            httpd.server_close()
        finally:
            state.stop()

    def test_loopback_without_a_token_is_allowed(self):
        state = EngineState.fixture()
        try:
            httpd = serve(state, "127.0.0.1", 0, token="")
            httpd.server_close()
        finally:
            state.stop()

    @pytest.mark.parametrize(
        "host,loopback",
        [
            ("127.0.0.1", True),
            ("127.0.0.53", True),
            ("::1", True),
            ("localhost", True),
            ("0.0.0.0", False),  # noqa: S104 — a table entry, not a bind
            ("::", False),
            ("10.0.0.5", False),
            ("192.168.1.4", False),
            # INADDR_ANY. Binding "" is every interface, which is exactly the
            # case this guard exists for -- it is not a synonym for localhost.
            ("", False),
            # A name cannot be shown to be loopback without resolving it, and
            # a guard that resolves is a guard DNS can move.
            ("risk.internal", False),
        ],
    )
    def test_what_counts_as_loopback(self, host, loopback):
        assert _is_loopback(host) is loopback
