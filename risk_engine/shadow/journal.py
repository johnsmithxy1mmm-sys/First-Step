"""Calibration journal (§3.4).

Predictions are written before the outcome exists and are never updated;
outcomes go into a separate table keyed by prediction id. That separation is
the integrity property the whole exercise depends on -- a schema where a
prediction row could be edited after resolution would make the published
calibration score worth nothing.

SQLite here, Postgres in `schema.sql`. The two are kept structurally
identical and `test_journal.py` asserts it; SQLite is what makes the shadow
harness runnable in a test and on a laptop.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from risk_engine.sim.stats import PredictiveDistribution

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS calibration_predictions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    address              TEXT    NOT NULL,
    variant              TEXT    NOT NULL,
    predicted_at         TEXT    NOT NULL,
    horizon_hours        INTEGER NOT NULL,
    resolves_at          TEXT    NOT NULL,
    model_version        TEXT    NOT NULL,
    distribution_version TEXT    NOT NULL,
    seed                 INTEGER NOT NULL,
    n_paths              INTEGER NOT NULL,
    converged            INTEGER NOT NULL,
    start_equity         REAL    NOT NULL,
    p_liq                REAL    NOT NULL,
    p_liq_ci_low         REAL    NOT NULL,
    p_liq_ci_high        REAL    NOT NULL,
    var_95               REAL    NOT NULL,
    cvar_95              REAL    NOT NULL,
    quantile_values      TEXT    NOT NULL,
    n_quantile_levels    INTEGER NOT NULL,
    book_snapshot        TEXT    NOT NULL,
    UNIQUE (address, variant, predicted_at, distribution_version)
);
CREATE INDEX IF NOT EXISTS calibration_predictions_due_idx
    ON calibration_predictions (resolves_at);
CREATE INDEX IF NOT EXISTS calibration_predictions_version_idx
    ON calibration_predictions (distribution_version, variant);

CREATE TABLE IF NOT EXISTS calibration_outcomes (
    prediction_id        INTEGER PRIMARY KEY REFERENCES calibration_predictions(id),
    resolved_at          TEXT    NOT NULL,
    actual_equity        REAL    NOT NULL,
    actual_equity_change REAL    NOT NULL,
    external_flow_usd    REAL    NOT NULL,
    book_changed         INTEGER NOT NULL,
    liquidated           INTEGER NOT NULL,
    pit                  REAL    NOT NULL,
    pit_u                REAL    NOT NULL,
    crps                 REAL    NOT NULL,
    var_95_breached      INTEGER NOT NULL,
    observation_day      TEXT    NOT NULL,
    resolution_lag_s     REAL    NOT NULL,
    stale_resolution     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS calibration_outcomes_day_idx
    ON calibration_outcomes (observation_day);
"""

VARIANT_MODEL = "model"
VARIANT_BASELINE_A = "baseline_a"
VARIANT_BASELINE_B = "baseline_b"


@dataclass(frozen=True, slots=True)
class PendingPrediction:
    id: int
    address: str
    variant: str
    predicted_at: datetime
    resolves_at: datetime
    start_equity: float
    var_95: float
    distribution: PredictiveDistribution
    book_snapshot: dict


@dataclass(frozen=True, slots=True)
class ShadowProgress:
    """§3.3's gate, as data. Phase 4 does not start until this clears."""

    distribution_version: str
    distinct_days: int
    distinct_addresses: int
    resolved_observations: int
    required_days: int = 21
    required_addresses: int = 200

    @property
    def gate_open(self) -> bool:
        return (
            self.distinct_days >= self.required_days
            and self.distinct_addresses >= self.required_addresses
        )

    def __str__(self) -> str:
        state = "OPEN" if self.gate_open else "CLOSED"
        return (
            f"shadow gate {state} for {self.distribution_version}: "
            f"{self.distinct_days}/{self.required_days} days, "
            f"{self.distinct_addresses}/{self.required_addresses} addresses, "
            f"{self.resolved_observations} resolved observations"
        )


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("timestamps written to the journal must be timezone-aware")
    return dt.astimezone(timezone.utc).isoformat()


class CalibrationJournal:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SQLITE_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> CalibrationJournal:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- writing --------------------------------------------------------

    def record_prediction(
        self,
        address: str,
        variant: str,
        predicted_at: datetime,
        horizon_hours: int,
        model_version: str,
        distribution_version: str,
        seed: int,
        n_paths: int,
        converged: bool,
        start_equity: float,
        p_liq: float,
        p_liq_ci: tuple[float, float],
        var_95: float,
        cvar_95: float,
        distribution: PredictiveDistribution,
        book_snapshot: dict,
    ) -> int:
        resolves_at = predicted_at.timestamp() + horizon_hours * 3600
        cur = self.conn.execute(
            """
            INSERT INTO calibration_predictions (
                address, variant, predicted_at, horizon_hours, resolves_at,
                model_version, distribution_version, seed, n_paths, converged,
                start_equity, p_liq, p_liq_ci_low, p_liq_ci_high, var_95, cvar_95,
                quantile_values, n_quantile_levels, book_snapshot
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                address, variant, _iso(predicted_at), horizon_hours,
                _iso(datetime.fromtimestamp(resolves_at, timezone.utc)),
                model_version, distribution_version, seed, n_paths, int(converged),
                start_equity, p_liq, p_liq_ci[0], p_liq_ci[1], var_95, cvar_95,
                json.dumps([float(v) for v in distribution.values]),
                len(distribution.levels), json.dumps(book_snapshot),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def record_outcome(
        self,
        prediction_id: int,
        resolved_at: datetime,
        actual_equity: float,
        actual_equity_change: float,
        external_flow_usd: float,
        book_changed: bool,
        liquidated: bool,
        pit: float,
        pit_u: float,
        crps: float,
        var_95_breached: bool,
        observation_day: date,
        resolution_lag_s: float,
        stale_resolution: bool,
    ) -> None:
        """Write the realised outcome. Once, and only once.

        A plain INSERT, not INSERT OR REPLACE (audit A-09): the public
        calibration score is worth exactly as much as the guarantee that a
        recorded outcome cannot be quietly rewritten after the fact. A second
        attempt raises IntegrityError; correcting a genuinely wrong outcome
        has to be a deliberate, visible operation.
        """
        self.conn.execute(
            """
            INSERT INTO calibration_outcomes (
                prediction_id, resolved_at, actual_equity, actual_equity_change,
                external_flow_usd, book_changed, liquidated, pit, pit_u, crps,
                var_95_breached, observation_day, resolution_lag_s, stale_resolution
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                prediction_id, _iso(resolved_at), actual_equity, actual_equity_change,
                external_flow_usd, int(book_changed), int(liquidated), pit, pit_u, crps,
                int(var_95_breached), observation_day.isoformat(),
                resolution_lag_s, int(stale_resolution),
            ),
        )
        self.conn.commit()

    # -- reading --------------------------------------------------------

    def due(self, as_of: datetime) -> list[PendingPrediction]:
        rows = self.conn.execute(
            """
            SELECT p.* FROM calibration_predictions p
            LEFT JOIN calibration_outcomes o ON o.prediction_id = p.id
            WHERE o.prediction_id IS NULL AND p.resolves_at <= ?
            ORDER BY p.resolves_at
            """,
            (_iso(as_of),),
        ).fetchall()
        return [self._to_pending(r) for r in rows]

    @staticmethod
    def _to_pending(row: sqlite3.Row) -> PendingPrediction:
        values = np.array(json.loads(row["quantile_values"]), dtype=np.float64)
        levels = np.linspace(0.0, 1.0, row["n_quantile_levels"])
        return PendingPrediction(
            id=row["id"],
            address=row["address"],
            variant=row["variant"],
            predicted_at=datetime.fromisoformat(row["predicted_at"]),
            resolves_at=datetime.fromisoformat(row["resolves_at"]),
            start_equity=row["start_equity"],
            var_95=row["var_95"],
            distribution=PredictiveDistribution(levels=levels, values=values),
            book_snapshot=json.loads(row["book_snapshot"]),
        )

    def scored(self, distribution_version: str, variant: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT p.address, p.variant, o.*
            FROM calibration_outcomes o
            JOIN calibration_predictions p ON p.id = o.prediction_id
            WHERE p.distribution_version = ? AND p.variant = ?
            ORDER BY o.observation_day
            """,
            (distribution_version, variant),
        ).fetchall()

    def progress(self, distribution_version: str) -> ShadowProgress:
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS n,
                   COUNT(DISTINCT o.observation_day) AS days,
                   COUNT(DISTINCT p.address) AS addrs
            FROM calibration_outcomes o
            JOIN calibration_predictions p ON p.id = o.prediction_id
            WHERE p.distribution_version = ? AND p.variant = ?
              AND o.stale_resolution = 0
            """,
            (distribution_version, VARIANT_MODEL),
        ).fetchone()
        return ShadowProgress(
            distribution_version=distribution_version,
            distinct_days=row["days"] or 0,
            distinct_addresses=row["addrs"] or 0,
            resolved_observations=row["n"] or 0,
        )
