"""История ставок в JSON: защита от дублей и учёт потраченного."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = Path(__file__).parent / "data" / "bets.json"


class BetLog:
    def __init__(self, path: Path | str = DEFAULT_PATH):
        self.path = Path(path)
        self._bets: list[dict] = []
        if self.path.exists():
            self._bets = json.loads(self.path.read_text(encoding="utf-8"))

    @property
    def bets(self) -> list[dict]:
        return list(self._bets)

    def live_token_ids(self) -> set[str]:
        """Токены, на которые уже есть реальные ставки, — их пропускаем."""
        return {b["token_id"] for b in self._bets if b.get("live")}

    def spent_usd(self) -> float:
        return sum(b.get("usd", 0.0) for b in self._bets if b.get("live"))

    def record(self, *, candidate, price: float, size: float, live: bool,
               order_id: str | None = None, status: str = "planned") -> dict:
        bet = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "market_id": candidate.market_id,
            "question": candidate.question,
            "slug": candidate.slug,
            "outcome": candidate.outcome,
            "token_id": candidate.token_id,
            "price": price,
            "size": size,
            "usd": round(price * size, 2),
            "payout_if_win": round(size, 2),  # каждая акция платит $1 при победе
            "live": live,
            "order_id": order_id,
            "status": status,
        }
        self._bets.append(bet)
        self._save()
        return bet

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._bets, ensure_ascii=False, indent=2), encoding="utf-8"
        )
