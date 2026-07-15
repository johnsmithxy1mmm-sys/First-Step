"""Ledger: sqlite with snapshots of trades, estimates, resolutions and bank.

Every trade is written with a full context snapshot (p_mkt, p_est, signal
contributions, book) — after resolution this lets us attribute PnL to edge
sources and compute calibration (Brier score).
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
    neg_risk INTEGER DEFAULT 0,       -- 1 = mutually-exclusive (neg-risk) event
    snapshot TEXT                     -- JSON: p_mkt, p_est, edge, signals, book
);
CREATE TABLE IF NOT EXISTS seen_markets (
    market_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS smart_money_seen (
    key TEXT PRIMARY KEY,             -- wallet:asset — a position already alerted on
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
    won INTEGER NOT NULL,             -- 1 = outcome won, share pays $1
    payout_per_share REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS markouts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER NOT NULL,
    token_id TEXT NOT NULL,
    horizon_sec INTEGER NOT NULL,
    fill_price REAL NOT NULL,
    mark_price REAL NOT NULL,
    markout REAL NOT NULL,            -- mark - fill; for a buy >0 = price moved our way
    ts TEXT NOT NULL,
    UNIQUE (trade_id, horizon_sec)
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
        # check_same_thread=False + lock: both the main cycle and the websocket
        # fastlane thread (instant entries/exits) write to the ledger.
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """Add columns to pre-existing tables (CREATE IF NOT EXISTS won't)."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(trades)")}
        if "neg_risk" not in cols:
            self._conn.execute("ALTER TABLE trades ADD COLUMN neg_risk INTEGER DEFAULT 0")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- locked DB access (single sqlite connection shared across threads) ---

    def _query(self, sql: str, params: tuple = ()) -> list:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _execute(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def _executemany(self, sql: str, seq) -> None:
        with self._lock:
            self._conn.executemany(sql, seq)
            self._conn.commit()

    # --- writes ---

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
        neg_risk = int(bool(c.market.event_neg_risk or c.market.neg_risk))
        self._execute(
            "INSERT INTO trades (ts, mode, market_id, event_id, question, outcome, "
            "category, token_id, side, price, size, usd, order_id, status, strategy, "
            "neg_risk, snapshot) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (_now(), mode, c.market.id, c.market.event_id, c.market.question,
             c.outcome, category, c.token_id, side, price, size,
             round(price * size, 6), order_id, status, strategy, neg_risk,
             json.dumps(snapshot, ensure_ascii=False, default=str)),
        )

    def record_estimate(self, estimate: Estimate, qualifies: bool) -> None:
        c = estimate.candidate
        self._execute(
            "INSERT INTO estimates (ts, market_id, token_id, question, p_mkt, p_est, "
            "edge_ratio, qualifies, signals) VALUES (?,?,?,?,?,?,?,?,?)",
            (_now(), c.market.id, c.token_id, c.market.question,
             estimate.p_mkt, estimate.p_est, estimate.edge_ratio,
             int(qualifies), estimate.signals_dump()),
        )

    def record_resolution(self, token_id: str, market_id: str, won: bool) -> None:
        self._execute(
            "INSERT OR REPLACE INTO resolutions (token_id, market_id, ts, won, payout_per_share) "
            "VALUES (?,?,?,?,?)",
            (token_id, market_id, _now(), int(won), 1.0 if won else 0.0),
        )

    def seen_market_ids(self) -> set[str]:
        return {r["market_id"] for r in self._query("SELECT market_id FROM seen_markets")}

    def mark_markets_seen(self, market_ids: list[str]) -> None:
        self._executemany(
            "INSERT OR IGNORE INTO seen_markets (market_id, ts) VALUES (?, ?)",
            [(mid, _now()) for mid in market_ids],
        )

    def smart_money_seen_keys(self) -> set[str]:
        return {r["key"] for r in self._query("SELECT key FROM smart_money_seen")}

    def mark_smart_money_seen(self, keys: list[str]) -> None:
        self._executemany(
            "INSERT OR IGNORE INTO smart_money_seen (key, ts) VALUES (?, ?)",
            [(k, _now()) for k in keys])

    def snapshot_bank(self, cash: float, exposure: float) -> None:
        equity = cash + exposure
        hwm = max(self.high_water_mark(), equity)
        self._execute(
            "INSERT INTO bank (ts, cash, exposure, equity, hwm) VALUES (?,?,?,?,?)",
            (_now(), cash, exposure, equity, hwm),
        )

    # --- positions and exposure ---

    def open_positions(self, mode: str) -> list[Position]:
        """Open positions = buys - sells - resolutions (per token)."""
        rows = self._query(
            "SELECT * FROM trades WHERE mode = ? AND status != 'failed' ORDER BY id", (mode,))
        resolved = {r["token_id"] for r in self._query("SELECT token_id FROM resolutions")}

        agg: dict[str, dict] = {}
        for r in rows:
            token = r["token_id"]
            slot = agg.setdefault(token, {
                "size": 0.0, "cost": 0.0, "market_id": r["market_id"],
                "question": r["question"], "outcome": r["outcome"],
                "category": r["category"], "event_id": r["event_id"] or "",
                "neg_risk": bool(r["neg_risk"] if "neg_risk" in r.keys() else 0),
            })
            if r["side"] == "BUY":
                slot["size"] += r["size"]
                slot["cost"] += r["usd"]
            else:
                # A sell reduces the position at the average entry price.
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
                event_id=slot["event_id"], neg_risk=slot["neg_risk"],
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
        """Idempotency: do not duplicate an entry for a token."""
        row = self._query(
            "SELECT COUNT(*) AS n FROM trades WHERE mode = ? AND token_id = ? "
            "AND side = 'BUY' AND status != 'failed' AND status != 'canceled'",
            (mode, token_id))[0]
        return row["n"] > 0

    # --- PnL and metrics ---

    def realized_pnl(self, mode: str) -> float:
        """PnL from closed events: resolutions + sells against entry cost."""
        rows = self._query(
            "SELECT t.token_id, t.side, t.price, t.size, t.usd, r.won "
            "FROM trades t LEFT JOIN resolutions r ON r.token_id = t.token_id "
            "WHERE t.mode = ? AND t.status != 'failed' ORDER BY t.id", (mode,))

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
            pnl += slot["sell_usd"] - avg * slot["sell_size"]      # realized by sells
            if slot["won"] is not None:
                remaining = max(slot["buy_size"] - slot["sell_size"], 0.0)
                payout = remaining * (1.0 if slot["won"] else 0.0)
                pnl += payout - avg * remaining                    # realized by resolution
        return pnl

    def first_bank_equity_since(self, ts_iso: str) -> float | None:
        """First equity snapshot after a timestamp (for the daily stop)."""
        rows = self._query(
            "SELECT equity FROM bank WHERE ts >= ? ORDER BY id LIMIT 1", (ts_iso,))
        return float(rows[0]["equity"]) if rows else None

    def realized_pnl_by_strategy(self, mode: str) -> dict[str, float]:
        """Realized PnL by strategy (for allocation and attribution)."""
        rows = self._query(
            "SELECT t.token_id, t.side, t.size, t.usd, t.strategy, r.won "
            "FROM trades t LEFT JOIN resolutions r ON r.token_id = t.token_id "
            "WHERE t.mode = ? AND t.status != 'failed' ORDER BY t.id", (mode,))
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

    # --- markout analytics: where price went after our fills ---

    def fills_needing_markout(self, mode: str, horizon_sec: int,
                              max_age_sec: float = 172_800,
                              limit: int = 20) -> list[dict]:
        """Filled buys due for a markout measurement at the horizon."""
        now = datetime.now(timezone.utc)
        rows = self._query(
            "SELECT t.id, t.token_id, t.price, t.ts FROM trades t "
            "WHERE t.mode = ? AND t.side = 'BUY' "
            "AND t.status IN ('filled', 'paper-filled') "
            "AND NOT EXISTS (SELECT 1 FROM markouts m "
            "                WHERE m.trade_id = t.id AND m.horizon_sec = ?) "
            "ORDER BY t.id", (mode, horizon_sec))
        out: list[dict] = []
        for r in rows:
            age = (now - datetime.fromisoformat(r["ts"])).total_seconds()
            if horizon_sec <= age <= max_age_sec:
                out.append(dict(r))
                if len(out) >= limit:
                    break
        return out

    def record_markout(self, trade_id: int, token_id: str, horizon_sec: int,
                       fill_price: float, mark_price: float) -> None:
        self._execute(
            "INSERT OR IGNORE INTO markouts "
            "(trade_id, token_id, horizon_sec, fill_price, mark_price, markout, ts) "
            "VALUES (?,?,?,?,?,?,?)",
            (trade_id, token_id, horizon_sec, fill_price, mark_price,
             mark_price - fill_price, _now()))

    def markout_stats(self, mode: str) -> list[dict]:
        """Average markout by (strategy, horizon): the key fill-quality test.

        For a buy, markout < 0 means adverse selection: price after our fill
        systematically drops — the informed are running us over.
        """
        rows = self._query(
            "SELECT t.strategy, m.horizon_sec, COUNT(*) AS n, "
            "AVG(m.markout) AS avg_markout, "
            "AVG(m.markout / m.fill_price) AS avg_markout_pct, "
            "SUM(CASE WHEN m.markout >= 0 THEN 1 ELSE 0 END) AS favorable "
            "FROM markouts m JOIN trades t ON t.id = m.trade_id "
            "WHERE t.mode = ? "
            "GROUP BY t.strategy, m.horizon_sec "
            "ORDER BY t.strategy, m.horizon_sec", (mode,))
        return [dict(r) for r in rows]

    def bank_series(self) -> list[dict]:
        return [dict(r) for r in self._query(
            "SELECT ts, equity, hwm FROM bank ORDER BY id")]

    def estimates_summary(self) -> dict:
        row = self._query(
            "SELECT COUNT(*) AS total, SUM(qualifies) AS qualifying, "
            "AVG(edge_ratio) AS avg_edge FROM estimates")[0]
        return dict(row)

    def high_water_mark(self) -> float:
        row = self._query("SELECT MAX(hwm) AS hwm FROM bank")[0]
        return float(row["hwm"] or 0.0)

    def metrics(self, mode: str) -> dict:
        """Hit rate, average multiple, Brier for p_est/p_mkt, ROI, PnL attribution."""
        rows = self._query(
            "SELECT t.token_id, t.price, t.size, t.usd, t.snapshot, r.won "
            "FROM trades t JOIN resolutions r ON r.token_id = t.token_id "
            "WHERE t.mode = ? AND t.side = 'BUY' AND t.status != 'failed'", (mode,))

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
            # Attribution: split the trade PnL across signals proportional to weights.
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
