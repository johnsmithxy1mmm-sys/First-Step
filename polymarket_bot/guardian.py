"""Position guardian: early warning when a faded tail is materializing.

Fade (and resolution carry) sells the near-certain side: we hold an EXPENSIVE
leg (NO at ~0.97, or a carry YES at ~0.97) whose whole thesis is "the tail
won't happen". Its one real risk is that the tail DOES happen — a news event —
and by resolution the leg is worth ~0. That is fade's "rare large loss".

The guardian is the mirror of take-profit: it watches those expensive legs on
the WS fastlane and fires the moment one loses `adverse_drop` of price (the
tail probability is climbing), so a human learns about it while there is still
a book to sell into — not at settlement. `check()` is pure (no I/O), so the
tick thread only reads a cached mark; the optional auto-trim runs in the cycle.
"""

from __future__ import annotations

from pydantic import BaseModel

from .config import BotConfig
from .models import Position


class GuardianVerdict(BaseModel):
    drop: float                 # entry_price - current_mark (how much the leg lost)
    entry: float
    mark: float
    tail_entry: float           # implied tail prob when we entered (1 - entry)
    tail_now: float             # implied tail prob now (1 - mark) — this is rising

    def describe(self, question: str) -> str:
        return (f"GUARDIAN: a sold tail is materializing — {question[:60]}\n"
                f"leg {self.entry:.3f} -> {self.mark:.3f} (−{self.drop:.3f}); "
                f"tail prob {self.tail_entry:.1%} -> {self.tail_now:.1%}")


class PositionGuardian:
    def __init__(self, cfg: BotConfig):
        self._cfg = cfg.guardian

    def check(self, position: Position, mark: float) -> GuardianVerdict | None:
        """A verdict if `position` (an expensive leg) has moved adversely; else None."""
        c = self._cfg
        if not c.enabled or mark <= 0 or position.avg_price < c.min_entry_price:
            return None
        drop = position.avg_price - mark
        if drop < c.adverse_drop:
            return None
        return GuardianVerdict(
            drop=drop, entry=position.avg_price, mark=mark,
            tail_entry=max(0.0, 1.0 - position.avg_price),
            tail_now=max(0.0, 1.0 - mark))
