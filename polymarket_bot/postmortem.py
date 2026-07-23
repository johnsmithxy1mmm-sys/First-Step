"""Auto-postmortems: a structured lesson from every resolved position.

When a position resolves, the bot writes a one-line, structured record of what
it entered, what happened, and the rule-based lesson — appended to a growing
JSONL so the fund documents itself. Over time this is a searchable history of
which theses paid and which of fade's tails actually materialized, without any
manual journaling. Pure `build()` (tested); the writer only appends.

The lesson is deterministic (no LLM needed): the interesting axis is entry
price — a high-priced leg (a sold tail: NO fade / carry) that LOSES is the
rare-large-loss to study; a low-priced leg (a longshot YES) that WINS is model
edge validated. An LLM narrative can be layered on later behind the estimator's
llm flag; the structured record stands on its own.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .models import Position

log = logging.getLogger(__name__)


def build(position: Position, won: bool) -> dict:
    """A structured postmortem for one resolved position (pure)."""
    entry = position.avg_price
    # PnL per share: a winning share pays $1; a loser pays $0.
    pnl = (1.0 - entry if won else -entry) * position.size
    sold_tail = entry >= 0.5          # NO fade / carry: we sold the tail
    if sold_tail and not won:
        lesson = ("rare-large-loss: a SOLD TAIL MATERIALIZED — review the "
                  "guardian threshold and whether the edge justified the size")
    elif sold_tail and won:
        lesson = "sold tail held as expected — thesis correct, small win banked"
    elif not sold_tail and won:
        lesson = "longshot HIT — the model's edge was real on this one"
    else:
        lesson = "longshot expired worthless — the expected majority outcome"
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": position.question[:120],
        "category": position.category,
        "outcome": position.outcome,
        "entry": round(entry, 4),
        "size": round(position.size, 2),
        "won": bool(won),
        "pnl": round(pnl, 2),
        "sold_tail": sold_tail,
        "lesson": lesson,
    }


class PostmortemWriter:
    def __init__(self, path: str, enabled: bool = True):
        self._path = Path(path)
        self._enabled = enabled

    def record(self, position: Position, won: bool) -> dict | None:
        if not self._enabled:
            return None
        pm = build(position, won)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(pm, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("postmortem write failed: %s", exc)
        return pm

    def recent(self, limit: int = 20) -> list[dict]:
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").splitlines()[-limit:]
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
        return out
