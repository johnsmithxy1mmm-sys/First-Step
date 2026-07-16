"""Phase 2 — MM backtest: record order books to SQLite and replay quoting logic.

Recording: top-of-book snapshots of selected markets every N seconds (>= 48
hours for a meaningful report). Replay: run the MarketMaker quoting logic over
the recording with a paper fill model. Report: fill rate, captured spread,
inventory, an estimate of rewards time (share of snapshots with a quote in the
reward band).
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .clob import ClobReader
from .config import BotConfig
from .gamma import GammaClient
from .marketmaker import MarketMaker
from .models import Market
from .scorer import MarketScorer
from .ws_feed import TopOfBook

log = logging.getLogger(__name__)

SNAP_SCHEMA = """
CREATE TABLE IF NOT EXISTS book_snaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    market_id TEXT NOT NULL,
    token_id TEXT NOT NULL,
    bid REAL, bid_size REAL, ask REAL, ask_size REAL
);
CREATE TABLE IF NOT EXISTS snap_markets (
    market_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snaps_ts ON book_snaps (ts);
"""


class BookRecorder:
    def __init__(self, cfg: BotConfig, db_path: str | Path):
        self._cfg = cfg
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.executescript(SNAP_SCHEMA)

    def record(self, minutes: float, interval_sec: float = 15.0) -> int:
        gamma = GammaClient(self._cfg)
        clob = ClobReader(self._cfg)
        scorer = MarketScorer(self._cfg)
        markets = gamma.fetch_active_markets()
        books = {m.clob_token_ids[0]: clob.order_book(m.clob_token_ids[0])
                 for m in markets if scorer.eligible(m)}
        selected = scorer.top_markets(markets, books)
        if not selected:
            log.warning("recorder: no eligible markets")
            return 0
        for m in selected:
            self._conn.execute(
                "INSERT OR REPLACE INTO snap_markets (market_id, payload) VALUES (?,?)",
                (m.id, m.model_dump_json()))
        self._conn.commit()
        log.info("recorder: writing %d markets every %.0fs, %.0f minutes",
                 len(selected), interval_sec, minutes)

        deadline = time.time() + minutes * 60
        snaps = 0
        while time.time() < deadline:
            ts = time.time()
            for m in selected:
                for token in m.clob_token_ids[:2]:
                    book = clob.order_book(token)
                    if book is None:
                        continue
                    bid, ask = book.best_bid, book.best_ask
                    self._conn.execute(
                        "INSERT INTO book_snaps (ts, market_id, token_id, bid, bid_size, ask, ask_size) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (ts, m.id, token, bid,
                         next((l.size for l in book.bids if l.price == bid), 0.0),
                         ask,
                         next((l.size for l in book.asks if l.price == ask), 0.0)))
                    snaps += 1
            self._conn.commit()
            time.sleep(max(interval_sec - (time.time() - ts), 0.5))
        return snaps


@dataclass
class ReplayReport:
    snapshots: int = 0
    quotes_posted: int = 0
    fills: int = 0
    spread_captured_usd: float = 0.0
    reward_eligible_snaps: int = 0
    inventory_by_market: dict = field(default_factory=dict)

    @property
    def fill_rate(self) -> float:
        return self.fills / self.quotes_posted if self.quotes_posted else 0.0


def queue_aware_fill(quote, outcome_index: int, top) -> float:
    """Honest replay fill: the trade-through must consume the size queued ahead
    of us (proxied by the resting size at the touch) before any of ours fills."""
    from .research import simulate_maker_fill
    price = quote.yes_bid if outcome_index == 0 else quote.no_bid
    return simulate_maker_fill(price, "BUY", [(top.ask, top.ask_size)],
                               our_size=quote.size, queue_ahead=top.bid_size)


def replay(cfg: BotConfig, db_path: str | Path, ledger) -> ReplayReport:
    """Replay MM logic over recorded order books (queue-aware fill model)."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    markets: dict[str, Market] = {}
    for row in conn.execute("SELECT market_id, payload FROM snap_markets"):
        markets[row["market_id"]] = Market.model_validate_json(row["payload"])
    if not markets:
        log.error("replay: no markets in the recording — run --record-books first")
        return ReplayReport()

    tops: dict[str, TopOfBook] = {}

    def top_source(token: str) -> TopOfBook | None:
        return tops.get(token)

    mm = MarketMaker(cfg, ledger, clob=_NullClob(), trader=None,
                     mode="paper", top_source=top_source)
    mm.fill_model = queue_aware_fill      # honest fills: queue + trade-through
    report = ReplayReport()
    market_list = list(markets.values())

    current_ts: float | None = None
    for row in conn.execute("SELECT * FROM book_snaps ORDER BY ts, id"):
        tops[row["token_id"]] = TopOfBook(
            bid=row["bid"] or 0, bid_size=row["bid_size"] or 0,
            ask=row["ask"] or 0, ask_size=row["ask_size"] or 0, ts=row["ts"])
        if current_ts is None:
            current_ts = row["ts"]
        if row["ts"] > current_ts:            # a new point in time -> MM cycle
            report.snapshots += 1
            quotes = mm.cycle(market_list)
            report.quotes_posted += len(quotes)
            for q in quotes:
                if q.market.in_rewards_program and \
                        q.captured_spread <= 2 * q.market.rewards_max_spread:
                    report.reward_eligible_snaps += 1
            current_ts = row["ts"]

    for p in ledger.open_positions("paper"):
        report.fills += 1
        report.inventory_by_market[p.question[:40]] = round(p.cost_usd, 2)
    # Captured spread: Yes+No pairs in inventory = $1 at redemption.
    report.spread_captured_usd = ledger.realized_pnl_by_strategy("paper").get("mm", 0.0)
    return report


class _NullClob:
    def order_book(self, token_id: str):  # replay works only from snapshots
        return None


def print_report(report: ReplayReport) -> None:
    console = Console()
    t = Table(title="MM backtest (replay of recorded books)")
    t.add_column("Metric")
    t.add_column("Value")
    t.add_row("Time points", str(report.snapshots))
    t.add_row("Quotes posted", str(report.quotes_posted))
    t.add_row("Virtual fills", str(report.fills))
    t.add_row("Fill rate", f"{report.fill_rate:.1%}")
    t.add_row("Snapshots in reward band", str(report.reward_eligible_snaps))
    console.print(t)
    if report.inventory_by_market:
        console.print("Inventory:", report.inventory_by_market)
    console.print("[dim]The fill model is optimistic (trade-through); a live "
                  "decision needs Phase 3 (paper on the live stream).[/dim]")
