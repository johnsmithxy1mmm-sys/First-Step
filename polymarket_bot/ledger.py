"""Журнал: sqlite со снапшотами сделок, оценок, резолюций и банка.

Каждая сделка пишется с полным снимком контекста (p_mkt, p_est, вклад
сигналов, книга) — после резолюции это позволяет атрибутировать PnL по
источникам edge и считать калибровку (Brier score).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .models import Estimate, Position

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    mode TEXT NOT NULL,               -- dry-run | live
    market_id TEXT NOT NULL,
    event_id TEXT,
    question TEXT,
    outcome TEXT,
    category TEXT,
    token_id TEXT NOT NULL,
    side TEXT NOT NULL,               -- BUY | SELL
    price REAL NOT NULL,
    size REAL NOT NULL,
    usd REAL NOT NULL,
    order_id TEXT,
    status TEXT,
    strategy TEXT DEFAULT 'longshot', -- longshot | arb | mm
    snapshot TEXT                     -- JSON: p_mkt, p_est, edge, signals, book
);
CREATE TABLE IF NOT EXISTS seen_markets (
    market_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS estimates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    market_id TEXT NOT NULL,
    token_id TEXT NOT NULL,
    question TEXT,
    p_mkt REAL NOT NULL,
    p_est REAL NOT NULL,
    edge_ratio REAL NOT NULL,
    qualifies INTEGER NOT NULL,
    signals TEXT
);
CREATE TABLE IF NOT EXISTS resolutions (
    token_id TEXT PRIMARY KEY,
    market_id TEXT,
    ts TEXT NOT NULL,
    won INTEGER NOT NULL,             -- 1 = исход выиграл, акция платит $1
    payout_per_share REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS bank (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    cash REAL NOT NULL,
    exposure REAL NOT NULL,
    equity REAL NOT NULL,
    hwm REAL NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Ledger:
    def __init__(self, db_path: str | Path):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + лок: в леджер пишут и основной цикл,
        # и websocket-поток fastlane (мгновенные входы/выходы).
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # --- запись ---

    def record_trade(self, *, mode: str, estimate: Estimate, category: str,
                     side: str, price: float, size: float,
                     order_id: str | None, status: str,
                     strategy: str = "longshot") -> None:
        c = estimate.candidate
        snapshot = {
            "p_mkt": estimate.p_mkt,
            "p_est": estimate.p_est,
            "edge_ratio": estimate.edge_ratio,
            "signals": [s.model_dump() for s in estimate.signals],
            "book": c.book.model_dump() if c.book else None,
        }
        self._conn.execute(
            "INSERT INTO trades (ts, mode, market_id, event_id, question, outcome, "
            "category, token_id, side, price, size, usd, order_id, status, strategy, snapshot) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (_now(), mode, c.market.id, c.market.event_id, c.market.question,
             c.outcome, category, c.token_id, side, price, size,
             round(price * size, 6), order_id, status, strategy,
             json.dumps(snapshot, ensure_ascii=False, default=str)),
        )
        self._conn.commit()

    def record_estimate(self, estimate: Estimate, qualifies: bool) -> None:
        c = estimate.candidate
        self._conn.execute(
            "INSERT INTO estimates (ts, market_id, token_id, question, p_mkt, p_est, "
            "edge_ratio, qualifies, signals) VALUES (?,?,?,?,?,?,?,?,?)",
            (_now(), c.market.id, c.token_id, c.market.question,
             estimate.p_mkt, estimate.p_est, estimate.edge_ratio,
             int(qualifies), estimate.signals_dump()),
        )
        self._conn.commit()

    def record_resolution(self, token_id: str, market_id: str, won: bool) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO resolutions (token_id, market_id, ts, won, payout_per_share) "
            "VALUES (?,?,?,?,?)",
            (token_id, market_id, _now(), int(won), 1.0 if won else 0.0),
        )
        self._conn.commit()

    def seen_market_ids(self) -> set[str]:
        return {r["market_id"] for r in self._conn.execute("SELECT market_id FROM seen_markets")}

    def mark_markets_seen(self, market_ids: list[str]) -> None:
        self._conn.executemany(
            "INSERT OR IGNORE INTO seen_markets (market_id, ts) VALUES (?, ?)",
            [(mid, _now()) for mid in market_ids],
        )
        self._conn.commit()

    def snapshot_bank(self, cash: float, exposure: float) -> None:
        equity = cash + exposure
        hwm = max(self.high_water_mark(), equity)
        self._conn.execute(
            "INSERT INTO bank (ts, cash, exposure, equity, hwm) VALUES (?,?,?,?,?)",
            (_now(), cash, exposure, equity, hwm),
        )
        self._conn.commit()

    # --- позиции и экспозиция ---

    def open_positions(self, mode: str) -> list[Position]:
        """Открытые позиции = покупки - продажи - резолюции (по токену)."""
        rows = self._conn.execute(
            "SELECT * FROM trades WHERE mode = ? AND status != 'failed' ORDER BY id", (mode,)
        ).fetchall()
        resolved = {r["token_id"] for r in self._conn.execute("SELECT token_id FROM resolutions")}

        agg: dict[str, dict] = {}
        for r in rows:
            token = r["token_id"]
            slot = agg.setdefault(token, {
                "size": 0.0, "cost": 0.0, "market_id": r["market_id"],
                "question": r["question"], "outcome": r["outcome"],
                "category": r["category"],
            })
            if r["side"] == "BUY":
                slot["size"] += r["size"]
                slot["cost"] += r["usd"]
            else:
                # Продажа уменьшает позицию по средней цене входа.
                if slot["size"] > 0:
                    avg = slot["cost"] / slot["size"]
                    slot["cost"] -= avg * min(r["size"], slot["size"])
                slot["size"] = max(slot["size"] - r["size"], 0.0)

        out = []
        for token, slot in agg.items():
            if token in resolved or slot["size"] <= 1e-9:
                continue
            out.append(Position(
                token_id=token, market_id=slot["market_id"] or "",
                question=slot["question"] or "", outcome=slot["outcome"] or "",
                category=slot["category"] or "other",
                size=slot["size"], avg_price=slot["cost"] / slot["size"],
            ))
        return out

    def exposure_by_category(self, mode: str) -> dict[str, float]:
        exposure: dict[str, float] = defaultdict(float)
        for p in self.open_positions(mode):
            exposure[p.category] += p.cost_usd
        return dict(exposure)

    def total_exposure(self, mode: str) -> float:
        return sum(p.cost_usd for p in self.open_positions(mode))

    def has_position_or_open_buy(self, token_id: str, mode: str) -> bool:
        """Идемпотентность: не дублировать вход по токену."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM trades WHERE mode = ? AND token_id = ? "
            "AND side = 'BUY' AND status != 'failed' AND status != 'canceled'",
            (mode, token_id),
        ).fetchone()
        return row["n"] > 0

    # --- PnL и метрики ---

    def realized_pnl(self, mode: str) -> float:
        """PnL по закрытым событиям: резолюции + продажи против стоимости входа."""
        rows = self._conn.execute(
            "SELECT t.token_id, t.side, t.price, t.size, t.usd, r.won "
            "FROM trades t LEFT JOIN resolutions r ON r.token_id = t.token_id "
            "WHERE t.mode = ? AND t.status != 'failed' ORDER BY t.id", (mode,)
        ).fetchall()

        per_token: dict[str, dict] = defaultdict(
            lambda: {"buy_size": 0.0, "buy_usd": 0.0, "sell_usd": 0.0,
                     "sell_size": 0.0, "won": None})
        for r in rows:
            slot = per_token[r["token_id"]]
            if r["side"] == "BUY":
                slot["buy_size"] += r["size"]
                slot["buy_usd"] += r["usd"]
            else:
                slot["sell_size"] += r["size"]
                slot["sell_usd"] += r["usd"]
            if r["won"] is not None:
                slot["won"] = bool(r["won"])

        pnl = 0.0
        for slot in per_token.values():
            avg = slot["buy_usd"] / slot["buy_size"] if slot["buy_size"] > 0 else 0.0
            pnl += slot["sell_usd"] - avg * slot["sell_size"]      # реализовано продажами
            if slot["won"] is not None:
                remaining = max(slot["buy_size"] - slot["sell_size"], 0.0)
                payout = remaining * (1.0 if slot["won"] else 0.0)
                pnl += payout - avg * remaining                    # реализовано резолюцией
        return pnl

    def first_bank_equity_since(self, ts_iso: str) -> float | None:
        """Первый снапшот equity после отметки времени (для дневного стопа)."""
        row = self._conn.execute(
            "SELECT equity FROM bank WHERE ts >= ? ORDER BY id LIMIT 1", (ts_iso,)
        ).fetchone()
        return float(row["equity"]) if row else None

    def realized_pnl_by_strategy(self, mode: str) -> dict[str, float]:
        """Реализованный PnL по стратегиям (для аллокации и атрибуции)."""
        rows = self._conn.execute(
            "SELECT t.token_id, t.side, t.size, t.usd, t.strategy, r.won "
            "FROM trades t LEFT JOIN resolutions r ON r.token_id = t.token_id "
            "WHERE t.mode = ? AND t.status != 'failed' ORDER BY t.id", (mode,)
        ).fetchall()
        per_token: dict[str, dict] = defaultdict(
            lambda: {"buy_size": 0.0, "buy_usd": 0.0, "sell_usd": 0.0,
                     "sell_size": 0.0, "won": None, "strategy": "longshot"})
        for r in rows:
            slot = per_token[r["token_id"]]
            slot["strategy"] = r["strategy"] or slot["strategy"]
            if r["side"] == "BUY":
                slot["buy_size"] += r["size"]
                slot["buy_usd"] += r["usd"]
            else:
                slot["sell_size"] += r["size"]
                slot["sell_usd"] += r["usd"]
            if r["won"] is not None:
                slot["won"] = bool(r["won"])
        out: dict[str, float] = defaultdict(float)
        for slot in per_token.values():
            avg = slot["buy_usd"] / slot["buy_size"] if slot["buy_size"] > 0 else 0.0
            pnl = slot["sell_usd"] - avg * slot["sell_size"]
            if slot["won"] is not None:
                remaining = max(slot["buy_size"] - slot["sell_size"], 0.0)
                pnl += remaining * (1.0 if slot["won"] else 0.0) - avg * remaining
            out[slot["strategy"]] += pnl
        return dict(out)

    def high_water_mark(self) -> float:
        row = self._conn.execute("SELECT MAX(hwm) AS hwm FROM bank").fetchone()
        return float(row["hwm"] or 0.0)

    def metrics(self, mode: str) -> dict:
        """Hit rate, средний множитель, Brier по p_est/p_mkt, ROI, атрибуция PnL."""
        rows = self._conn.execute(
            "SELECT t.token_id, t.price, t.size, t.usd, t.snapshot, r.won "
            "FROM trades t JOIN resolutions r ON r.token_id = t.token_id "
            "WHERE t.mode = ? AND t.side = 'BUY' AND t.status != 'failed'", (mode,)
        ).fetchall()

        n = len(rows)
        wins = sum(1 for r in rows if r["won"])
        invested = sum(r["usd"] for r in rows)
        payout = sum(r["size"] for r in rows if r["won"])
        brier_est = brier_mkt = 0.0
        multiples = []
        signal_pnl: dict[str, float] = defaultdict(float)

        for r in rows:
            outcome = float(r["won"])
            snap = json.loads(r["snapshot"] or "{}")
            p_est = float(snap.get("p_est", r["price"]))
            p_mkt = float(snap.get("p_mkt", r["price"]))
            brier_est += (p_est - outcome) ** 2
            brier_mkt += (p_mkt - outcome) ** 2
            if r["won"] and r["price"] > 0:
                multiples.append(1.0 / r["price"])
            # Атрибуция: PnL сделки распределяем по сигналам пропорционально весам.
            trade_pnl = (r["size"] if r["won"] else 0.0) - r["usd"]
            signals = [s for s in snap.get("signals", []) if s.get("name") != "market"]
            total_w = sum(s.get("confidence", 0) for s in signals)
            for s in signals:
                if total_w > 0:
                    signal_pnl[s["name"]] += trade_pnl * s.get("confidence", 0) / total_w

        return {
            "resolved_trades": n,
            "hit_rate": wins / n if n else 0.0,
            "avg_win_multiple": sum(multiples) / len(multiples) if multiples else 0.0,
            "invested_usd": invested,
            "payout_usd": payout,
            "roi": (payout - invested) / invested if invested else 0.0,
            "brier_model": brier_est / n if n else None,
            "brier_market": brier_mkt / n if n else None,
            "signal_pnl_attribution": dict(signal_pnl),
        }
