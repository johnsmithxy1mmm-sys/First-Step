"""Internal REST service around the risk engine (§8).

Consumed only by the Node backend, never exposed publicly: the backend owns
rate limiting (§5.3) and the user-facing degradation contract (§6), and this
service trusts its caller to have done both.

That trust is the reason it is not enough to *intend* loopback. A bearer
token is required whenever `RISK_SERVICE_TOKEN` is set, and `serve` refuses
to bind a non-loopback address without one -- a deployment that puts this on
0.0.0.0 by accident fails to start rather than quietly serving an
unauthenticated engine to the network. Refusing is the right direction here:
the failure is loud, immediate, and happens before any request is served,
whereas the alternative is discovered by whoever finds the open port.

`/health` stays open, because orchestrators and load balancers check it
before any token is in scope, and it discloses only model version and data
age. Everything else, `/metrics` included, needs the token: the metrics
snapshot describes model internals and request volumes.

Standard library only. This endpoint speaks one dialect (POST JSON in, JSON
out) to one consumer, and the request volume is bounded by the backend's own
limits, so a framework would add a dependency and a supply-chain surface to
solve a problem that does not exist here.

Every response carries `computed_at` and `model_version`, and every risk
number carries its interval, because the wire format is where those
guarantees would otherwise be quietly lost -- the Python types enforce them
in process, and JSON does not.

`/health` reports the age of the global correlation matrix, which §6
requires as its own signal, separate from the freshness of a book snapshot:
the matrix is rebuilt every five minutes by design (§2.1), so applying the
60-second staleness rule to it would mark the whole product permanently
stale (OPEN-QUESTIONS D4).
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import os
import threading
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from risk_engine.domain.types import Book, MarginMode, Position, RiskEstimate
from risk_engine.observability.metrics import METRICS
from risk_engine.service.state import EngineState
from risk_engine.tools.portfolio_risk import portfolio_risk
from risk_engine.tools.pre_trade_delta import ProposedOrder, pre_trade_delta
from risk_engine.version import MODEL_VERSION

log = logging.getLogger("risk_engine.service")


def _estimate(e: RiskEstimate) -> dict[str, Any]:
    return {
        "point": e.point,
        "ci_low": e.ci_low,
        "ci_high": e.ci_high,
        "model_version": e.model_version,
        "computed_at": e.computed_at.isoformat(),
    }


def _input_ages(self_state, book: Book) -> dict[str, Any]:
    """The observation times of the market data a risk number was built from.

    §6 is about the age of the DATA, and the wire used to carry only
    `computed_at` -- which the engine stamps at `datetime.now()` on every
    request, so it was always ~0 and the contract's stale and hidden tiers
    were unreachable for the risk value. These two fields are what the
    backend actually thresholds:

      - `book_captured_at`: when the positions were observed. The 60s clock.
      - `prices_as_of`: when the marks were observed. They are written only by
        `EngineState.refresh()`, so they ride the five-minute matrix cadence
        and get the matrix's own longer allowance (OPEN-QUESTIONS D4). This is
        an honest report of what the engine has, not a claim that the marks
        are sub-second -- no such feed is wired.
    """
    built = self_state.built_at()
    return {
        "book_captured_at": book.captured_at.isoformat(),
        "prices_as_of": built.isoformat() if built else None,
    }


def _parse_book(payload: dict) -> Book:
    positions = []
    for p in payload.get("positions", []):
        mode = MarginMode(p["mode"])
        positions.append(
            Position(
                coin=p["coin"],
                size=float(p["size"]),
                entry_price=float(p["entry_price"]),
                mode=mode,
                leverage=float(p["leverage"]),
                isolated_margin=(
                    float(p["isolated_margin"])
                    if mode is MarginMode.ISOLATED and p.get("isolated_margin") is not None
                    else None
                ),
            )
        )
    captured = payload.get("captured_at")
    return Book(
        address=payload["address"],
        cross_collateral=float(payload["cross_collateral"]),
        positions=tuple(positions),
        captured_at=(
            datetime.fromisoformat(captured) if captured else datetime.now(timezone.utc)
        ),
    )


#: Endpoints reachable without a token. Only the one an orchestrator needs
#: before it has any credentials, and it names no user and no book.
OPEN_ROUTES = frozenset({"/health"})


def _is_loopback(host: str) -> bool:
    if host in ("", "localhost"):
        # "" is INADDR_ANY -- every interface, which is the case this guard
        # exists to catch, not a synonym for localhost.
        return host == "localhost"
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # A name that is not an IP literal cannot be shown to be loopback
        # without resolving it, and a guard that resolves is a guard that can
        # be moved by DNS. Treat it as exposed.
        return False


class RiskHandler(BaseHTTPRequestHandler):
    server_version = "hl-risk/1.0"
    state: EngineState  # injected by `serve`
    token: str | None = None  # injected by `serve`

    def log_message(self, fmt: str, *args) -> None:
        log.info("%s - %s", self.address_string(), fmt % args)

    # -- plumbing -------------------------------------------------------

    def _authorised(self) -> bool:
        """Constant-time bearer check.

        `compare_digest` rather than `==` because the comparison is against a
        secret and an early-exit compare leaks its prefix one request at a
        time. The cost of getting this right is one import.
        """
        if self.token is None:
            return True
        if self.path.rstrip("/") in OPEN_ROUTES:
            return True
        header = self.headers.get("Authorization") or ""
        scheme, _, presented = header.partition(" ")
        if scheme.lower() != "bearer":
            return False
        # Compared as BYTES. `compare_digest` raises TypeError on str operands
        # containing non-ASCII, and `_authorised` runs before the handler's
        # try/except, so `Authorization: Bearer héllo` took the connection
        # down instead of returning 401. Encoding both sides keeps the
        # constant-time property and makes every input answerable.
        return hmac.compare_digest(
            presented.strip().encode("utf-8", "surrogatepass"),
            self.token.encode("utf-8", "surrogatepass"),
        )

    def _reject_unauthorised(self) -> None:
        # No echo of what was presented, and nothing about why it failed:
        # "wrong scheme" versus "wrong token" is free information.
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Bearer realm="risk-engine"')
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length))

    def do_GET(self) -> None:
        if not self._authorised():
            self._reject_unauthorised()
            return
        if self.path.rstrip("/") == "/health":
            self._send(200, self.state.health())
        elif self.path.rstrip("/") == "/metrics":
            self._send(200, METRICS.snapshot())
        else:
            self._send(404, {"error": f"no such endpoint: {self.path}"})

    def do_POST(self) -> None:
        if not self._authorised():
            self._reject_unauthorised()
            return
        route = self.path.rstrip("/")
        try:
            payload = self._read_json()
            if route == "/portfolio_risk":
                self._send(200, self._portfolio_risk(payload))
            elif route == "/pre_trade_delta":
                self._send(200, self._pre_trade_delta(payload))
            else:
                self._send(404, {"error": f"no such endpoint: {self.path}"})
        except (KeyError, ValueError, TypeError) as exc:
            # A malformed request is the caller's error and is safe to name.
            self._send(400, {"error": f"{type(exc).__name__}: {exc}"})
        except Exception as exc:
            # Anything else is ours. The message goes to the log, not to the
            # response: §10's spirit on not leaking internals outward.
            log.error("unhandled error on %s: %s", route, traceback.format_exc())
            self._send(500, {"error": "internal error", "kind": type(exc).__name__})

    # -- endpoints ------------------------------------------------------

    def _portfolio_risk(self, payload: dict) -> dict:
        book = _parse_book(payload["book"])
        bundle, specs, spot = self.state.require_ready()
        out = portfolio_risk(
            book, spot, bundle, specs,
            n_paths=int(payload.get("n_paths", 20_000)),
            seed=payload.get("seed"),
        )
        return {
            "address": out.address,
            "model_version": bundle.model_version,
            "computed_at": out.computed_at.isoformat(),
            # §2.5: an under-resolved number is not published. The backend
            # treats publishable=false the same way it treats stale data.
            "publishable": out.publishable,
            "start_equity": out.start_equity,
            "effective_leverage": _estimate(out.effective_leverage),
            "factor_beta": _estimate(out.factor_beta),
            # Read off the interval, not the sign of the point (D3).
            "direction_detectable": out.direction_detectable,
            "factor_coin": out.factor_coin,
            "p_liq_24h": _estimate(out.p_liq_24h_any),
            "p_liq_24h_cross": _estimate(out.p_liq_24h_cross),
            "p_liq_24h_isolated": {
                k: _estimate(v) for k, v in out.p_liq_24h_isolated.items()
            },
            "p_liq_7d": _estimate(out.p_liq_7d_any),
            "p_liq_7d_cross": _estimate(out.p_liq_7d_cross),
            "p_liq_7d_isolated": {
                k: _estimate(v) for k, v in out.p_liq_7d_isolated.items()
            },
            "cvar_95_24h_usd": _estimate(out.cvar_95_24h_usd),
            "funding_cost_24h": {
                "median": out.result_24h.funding_cost.quantile(0.5),
                "p05": out.result_24h.funding_cost.quantile(0.05),
                "p95": out.result_24h.funding_cost.quantile(0.95),
            },
            "matrix_age_s": self.state.matrix_age_s(),
            **_input_ages(self.state, book),
        }

    def _pre_trade_delta(self, payload: dict) -> dict:
        book = _parse_book(payload["book"])
        o = payload["order"]
        order = ProposedOrder(
            coin=o["coin"],
            size=float(o["size"]),
            leverage=float(o["leverage"]),
            mode=MarginMode(o.get("mode", "cross")),
            isolated_margin=(
                float(o["isolated_margin"]) if o.get("isolated_margin") is not None else None
            ),
        )
        bundle, specs, spot = self.state.require_ready()
        out = pre_trade_delta(
            book, order, spot, bundle, specs,
            n_paths=int(payload.get("n_paths", 20_000)),
            seed=payload.get("seed"),
        )

        def delta(d) -> dict:
            return {
                "before": _estimate(d.before),
                "after": _estimate(d.after),
                "change": _estimate(d.change),
                # The paired test is the answer the UI acts on; the overlap
                # rule §4.2 states is reported beside it, and the flag fires
                # when following it literally would hide a real change
                # (OPEN-QUESTIONS D6).
                "distinguishable": d.distinguishable,
                "marginal_intervals_overlap": d.marginal_intervals_overlap,
                "overlap_rule_would_mislead": d.overlap_rule_would_mislead,
                "direction": d.direction,
            }

        return {
            "order": {
                "coin": order.coin, "size": order.size,
                "leverage": order.leverage, "mode": order.mode.value,
                "description": order.describe(),
            },
            "execution_price": out.execution_price,
            "model_version": bundle.model_version,
            "computed_at": out.provenance.computed_at.isoformat(),
            "publishable": out.publishable,
            "p_liq": delta(out.p_liq),
            "cvar_95_usd": delta(out.cvar_95_usd),
            "new_assets": list(out.new_assets),
            "summary": out.summary(),
            "latency_ms": out.latency_ms,
            "within_budget": out.within_budget,
            "matrix_age_s": self.state.matrix_age_s(),
            **_input_ages(self.state, book),
        }


def serve(
    state: EngineState,
    host: str = "127.0.0.1",
    port: int = 8787,
    token: str | None = None,
) -> ThreadingHTTPServer:
    """Bind the service, refusing an exposed bind without a token.

    `token` defaults to `RISK_SERVICE_TOKEN`. Passing `token=""` is an
    explicit "no auth" and is honoured only on loopback.
    """
    if token is None:
        token = os.environ.get("RISK_SERVICE_TOKEN") or ""
    if not token and not _is_loopback(host):
        raise RuntimeError(
            f"refusing to bind {host!r} without RISK_SERVICE_TOKEN: this service "
            "answers unauthenticated callers and has no rate limiting of its own "
            "(§5.3 puts that in the backend). Bind 127.0.0.1, or set a token."
        )
    handler = type(
        "BoundRiskHandler", (RiskHandler,), {"state": state, "token": token or None}
    )
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    return httpd


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="risk_engine.service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--fixture", action="store_true",
        help="run on a synthetic bundle instead of live Hyperliquid data; "
             "the live path needs api.hyperliquid.xyz, which is unreachable "
             "from some build environments (OPEN-QUESTIONS E5)",
    )
    # A spelling for the default, so a caller assembling arguments from a
    # variable always has a non-empty flag to pass. Compose substituting an
    # empty string would otherwise reach argparse as an empty positional.
    mode.add_argument(
        "--live", action="store_true",
        help="explicit opposite of --fixture; the default, named so it can be "
             "passed rather than omitted",
    )
    parser.add_argument("--refresh-seconds", type=float, default=300.0,
                        help="global matrix rebuild cadence (§2.1)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    state = EngineState.fixture() if args.fixture else EngineState.live()
    state.start_refresh_loop(args.refresh_seconds)

    httpd = serve(state, args.host, args.port)
    log.info(
        "risk service on http://%s:%d  model=%s  mode=%s  auth=%s",
        args.host, args.port, MODEL_VERSION,
        "fixture" if args.fixture else "live",
        # The token itself never reaches a log line, here or anywhere.
        "bearer" if os.environ.get("RISK_SERVICE_TOKEN") else "none (loopback)",
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        thread.join()
    except KeyboardInterrupt:
        log.info("shutting down")
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
