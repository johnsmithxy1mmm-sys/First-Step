"""История ставок в JSON: защита от дублей, статусы ордеров, учёт бюджета."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_PATH = Path(__file__).parent / "data" / "bets.json"

# Статусы, при которых ордер считается ещё висящим в стакане.
OPEN_STATUSES = {"live", "open", "delayed", "unknown"}


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
        return {b["token_id"] for b in self._bets
                if b.get("live") and b.get("side", "BUY") == "BUY"
                and b.get("status") != "canceled"}

    def spent_usd(self, since: datetime | None = None) -> float:
        """Потрачено на реальные покупки (отменённые ордера не считаются)."""
        total = 0.0
        for b in self._bets:
            if not b.get("live") or b.get("side", "BUY") != "BUY":
                continue
            if b.get("status") == "canceled":
                continue
            if since is not None and datetime.fromisoformat(b["ts"]) < since:
                continue
            total += b.get("usd", 0.0)
        return total

    def spent_today_usd(self) -> float:
        day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        return self.spent_usd(since=day_start)

    def open_orders(self, older_than_hours: float = 0.0) -> list[dict]:
        """Реальные ордера, всё ещё висящие в стакане (для отмены зависших)."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)
        out = []
        for b in self._bets:
            if not b.get("live") or not b.get("order_id"):
                continue
            if b.get("status") not in OPEN_STATUSES:
                continue
            if datetime.fromisoformat(b["ts"]) <= cutoff:
                out.append(b)
        return out

    def has_open_sell(self, token_id: str) -> bool:
        """Есть ли уже висящая продажа по токену (не дублировать тейк-профит)."""
        return any(b.get("token_id") == token_id and b.get("side") == "SELL"
                   and b.get("live") and b.get("status") in OPEN_STATUSES
                   for b in self._bets)

    def update_status(self, order_id: str, status: str) -> None:
        for b in self._bets:
            if b.get("order_id") == order_id:
                b["status"] = status
        self._save()

    def record(self, *, market_id: str, question: str, slug: str, outcome: str,
               token_id: str, price: float, size: float, live: bool,
               side: str = "BUY", order_id: str | None = None,
               status: str = "planned") -> dict:
        bet = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "market_id": market_id,
            "question": question,
            "slug": slug,
            "outcome": outcome,
            "token_id": token_id,
            "side": side,
            "price": price,
            "size": size,
            "usd": round(price * size, 2),
            # Каждая акция платит $1 при победе (только для покупок).
            "payout_if_win": round(size, 2) if side == "BUY" else 0.0,
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
