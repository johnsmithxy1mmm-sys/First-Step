"""Structured logging: JSONL file + readable console (rich)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from rich.logging import RichHandler


class JsonlFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(log_path: str, level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console = RichHandler(rich_tracebacks=True, show_path=False)
    console.setLevel(level)
    root.addHandler(console)

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(JsonlFormatter())
    file_handler.setLevel(logging.DEBUG)
    root.addHandler(file_handler)

    # Noisy libraries — warnings only.
    for noisy in ("httpx", "httpcore", "urllib3", "apscheduler"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
