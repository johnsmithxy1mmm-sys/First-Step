"""Internal REST service around the risk engine (§8).

Consumed only by the Node backend, never exposed publicly: it carries no
authentication because it is expected to listen on loopback behind the
backend, and the backend owns rate limiting (§5.3) and the user-facing
degradation contract (§6).

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

import json
import logging
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


class RiskHandler(BaseHTTPRequestHandler):
    server_version = "hl-risk/1.0"
    state: EngineState  # injected by `serve`

    def log_message(self, fmt: str, *args) -> None:
        log.info("%s - %s", self.address_string(), fmt % args)

    # -- plumbing -------------------------------------------------------

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
        if self.path.rstrip("/") == "/health":
            self._send(200, self.state.health())
        elif self.path.rstrip("/") == "/metrics":
            self._send(200, METRICS.snapshot())
        else:
            self._send(404, {"error": f"no such endpoint: {self.path}"})

    def do_POST(self) -> None:
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
            "factor_beta": out.factor_beta,
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
        }


def serve(state: EngineState, host: str = "127.0.0.1", port: int = 8787) -> ThreadingHTTPServer:
    handler = type("BoundRiskHandler", (RiskHandler,), {"state": state})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    return httpd


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="risk_engine.service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--fixture", action="store_true",
        help="run on a synthetic bundle instead of live Hyperliquid data; "
             "the live path needs api.hyperliquid.xyz, which is unreachable "
             "from some build environments (OPEN-QUESTIONS E5)",
    )
    parser.add_argument("--refresh-seconds", type=float, default=300.0,
                        help="global matrix rebuild cadence (§2.1)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    state = EngineState.fixture() if args.fixture else EngineState.live()
    state.start_refresh_loop(args.refresh_seconds)

    httpd = serve(state, args.host, args.port)
    log.info(
        "risk service on http://%s:%d  model=%s  mode=%s",
        args.host, args.port, MODEL_VERSION, "fixture" if args.fixture else "live",
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
