"""Ops: Prometheus /metrics + /health server (stdlib) and config hot-reload.

No extra dependencies — a tiny http.server exposes the bot's live state for
Prometheus/Grafana and a healthcheck. SIGHUP re-reads config.yaml into the
running config objects in place (caps/thresholds apply without a restart).
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from pydantic import BaseModel

from .config import BotConfig

log = logging.getLogger(__name__)


def format_metrics(snapshot: dict) -> str:
    """Render a state snapshot as Prometheus text exposition format."""
    lines: list[str] = []

    def gauge(name: str, value, help_: str) -> None:
        lines.append(f"# HELP {name} {help_}")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {value}")

    gauge("polybot_equity_usd", snapshot.get("equity", 0.0), "Current equity")
    gauge("polybot_drawdown_pct", snapshot.get("drawdown", 0.0), "Drawdown fraction")
    gauge("polybot_open_positions", snapshot.get("positions", 0), "Open position count")
    gauge("polybot_exposure_usd", snapshot.get("exposure", 0.0), "Gross exposure")
    gauge("polybot_worst_case_usd", snapshot.get("worst_case", 0.0),
          "Event-netted worst-case exposure")
    gauge("polybot_cycle_errors", snapshot.get("errors", 0), "Errors in the last cycle")
    gauge("polybot_killswitch_halted", int(bool(snapshot.get("halted", False))),
          "1 if the kill-switch has halted trading")
    lines.append("# HELP polybot_pnl_usd Realized PnL by strategy")
    lines.append("# TYPE polybot_pnl_usd gauge")
    for strategy, pnl in (snapshot.get("pnl_by_strategy") or {}).items():
        safe = strategy.replace('"', "").replace("\\", "")
        lines.append(f'polybot_pnl_usd{{strategy="{safe}"}} {pnl}')
    return "\n".join(lines) + "\n"


class MetricsServer:
    """Serves /metrics (Prometheus) and /health (JSON) from a background thread."""

    def __init__(self, snapshot_fn, port: int = 9090, bind: str = "127.0.0.1"):
        self._fn = snapshot_fn
        self._port = port
        self._bind = bind
        self._httpd: HTTPServer | None = None

    def start(self) -> None:
        fn = self._fn

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence access logs
                pass

            def _send(self, body: bytes, content_type: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                try:
                    snap = fn()
                except Exception:
                    self.send_response(500)
                    self.end_headers()
                    return
                if self.path.startswith("/health"):
                    ok = not snap.get("halted", False)
                    self._send(json.dumps({"status": "ok" if ok else "halted",
                                           **snap}, default=str).encode(),
                               "application/json")
                elif self.path.startswith("/metrics"):
                    self._send(format_metrics(snap).encode(),
                               "text/plain; version=0.0.4")
                else:
                    self.send_response(404)
                    self.end_headers()

        self._httpd = HTTPServer((self._bind, self._port), Handler)
        threading.Thread(target=self._httpd.serve_forever, daemon=True,
                         name="metrics").start()
        log.info("metrics server on %s:%d (/metrics, /health)", self._bind, self._port)

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()


def reload_config_inplace(cfg: BotConfig, path=None) -> list[str]:
    """Re-read config.yaml into the live config objects. Returns changed fields.

    Copies changed leaf values into the existing nested config models so every
    holder of a sub-config (cfg.fade, cfg.portfolio, ...) sees the update without
    a restart. Host/runtime fields change too but only affect newly-built clients.
    """
    fresh = BotConfig.load(path)
    changed: list[str] = []
    for field in type(cfg).model_fields:
        old = getattr(cfg, field)
        new = getattr(fresh, field)
        if isinstance(old, BaseModel):
            for sub in type(old).model_fields:
                if getattr(old, sub) != getattr(new, sub):
                    setattr(old, sub, getattr(new, sub))
                    changed.append(f"{field}.{sub}")
        elif old != new:
            setattr(cfg, field, new)
            changed.append(field)
    return changed
