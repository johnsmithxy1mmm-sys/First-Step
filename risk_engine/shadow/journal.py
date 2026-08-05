"""Calibration journal (§3.4).

Predictions are written before the outcome exists and are never updated;
outcomes go into a separate table keyed by prediction id. That separation is
the integrity property the whole exercise depends on -- a schema where a
prediction row could be edited after resolution would make the published
calibration score worth nothing.

One schema, two backends. `schema.sql` is canonical (Postgres); the SQLite
form is *derived* from it rather than hand-maintained, because a second
hand-written schema is a second thing to forget to update and the failure
would surface years later as an unexplained discontinuity in a public score.
Both are exercised against a real Postgres server in the test suite, not
merely asserted to match.

The two backends disagree about what they hand back -- Postgres returns
`datetime`, `bool` and parsed JSON where SQLite returns strings and ints --
so every read goes through the normalisers below. That is deliberately the
only place the difference is allowed to exist.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from risk_engine.domain.types import normalise_address
from risk_engine.shadow.backends import Backend, canonical_ddl, open_backend, sqlite_ddl
from risk_engine.sim.stats import PredictiveDistribution

log = logging.getLogger("risk_engine.shadow.journal")

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


# -- backend normalisers ---------------------------------------------------
# Postgres and SQLite return different Python types for the same column. The
# journal's callers must not have to know which backend they are on, so every
# read is normalised here and nowhere else.


def as_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value))


def as_bool(value: Any) -> bool:
    return bool(value)


def as_json(value: Any) -> Any:
    return value if isinstance(value, (dict, list)) else json.loads(value)


def as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


class CalibrationJournal:
    """`target` is a SQLite path, or a `postgresql://` DSN."""

    def __init__(self, target: str | Path = ":memory:") -> None:
        self.backend: Backend = open_backend(target)
        self.backend.executescript(canonical_ddl() if self.is_postgres else sqlite_ddl())
        self._migrate()
        self.backend.commit()

    def _migrate(self) -> None:
        """Bring an EXISTING journal up to the current schema.

        `schema.sql` is all `CREATE TABLE IF NOT EXISTS`, which does nothing to
        a table that already exists -- so a column added to the DDL reaches a
        fresh journal and never reaches the deployed one. The shipped compose
        stack has a persistent volume, so "the deployed one" is the only
        journal that matters, and the failure is silent: every write that does
        not mention the new column keeps working.

        Additive and idempotent only. A migration that rewrote or dropped
        anything would be a migration on a write-once table (A-09), which this
        journal does not do.
        """
        # `backfill` is the value for rows that already exist -- rows whose
        # provenance was never recorded because the column did not exist when
        # they were written. It is TRUE, not FALSE, and the direction is the
        # whole point: those rows are of UNKNOWN provenance, and the §3.3 gate
        # is what Phase 4 (real money) is read off. §10 asks uncertainty to
        # resolve toward caution, and here caution means "do not count it".
        #
        # It is also the better guess on the facts. The deployment that has
        # rows predating this column logged "RECORDING UNDER A KNOWN §2.3
        # DEFECT" on every single sweep, so marking them clean would not be
        # neutral, it would be wrong in the one direction that matters.
        for table, column, ddl, backfill in (
            ("calibration_predictions", "recorded_under_defect",
             "BOOLEAN NOT NULL DEFAULT FALSE", "TRUE"),
        ):
            if self._has_column(table, column):
                continue
            try:
                self.backend.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
                # Runs exactly once, in the same transaction as the ADD, and
                # only over rows that existed before it. Not a rewrite of a
                # write-once row's content (A-09): it fills in provenance that
                # was never captured, and leaves every predicted number alone.
                # S608: `table`, `column` and `backfill` are literals from the
                # tuple three lines up, not input. A placeholder cannot be used
                # for an identifier anyway, which is why the ADD above is
                # interpolated too.
                self.backend.execute(
                    f"UPDATE {table} SET {column} = {backfill}"  # noqa: S608
                )
            except Exception as exc:
                # A concurrent journal opening at the same instant is the
                # expected case: both shadow jobs start together. Losing the
                # race is fine, the column is there either way.
                log.debug("migration %s.%s: %s", table, column, exc)

    def _has_column(self, table: str, column: str) -> bool:
        if self.is_postgres:
            rows = self._query(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = ? AND column_name = ?",
                (table, column),
            )
            return bool(rows)
        rows = self.backend.rows(self.backend.execute(f"PRAGMA table_info({table})"))
        return any(r["name"] == column for r in rows)

    @property
    def is_postgres(self) -> bool:
        return self.backend.placeholder == "%s"

    def _sql(self, sql: str) -> str:
        """Rewrite `?` placeholders for the backend's paramstyle."""
        return sql if self.backend.placeholder == "?" else sql.replace("?", "%s")

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        return self.backend.rows(self.backend.execute(self._sql(sql), params))

    def close(self) -> None:
        self.backend.close()

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
        recorded_under_defect: bool = False,
    ) -> int:
        # One account, one identity, enforced at the only place the journal is
        # written. `address` is stored as TEXT and compared byte-for-byte by
        # both backends -- there is no COLLATE anywhere in schema.sql, and
        # neither Postgres's default collation nor SQLite's BINARY folds case
        # -- so the UNIQUE constraint on (address, variant, predicted_at,
        # distribution_version) does not see two spellings of one account as a
        # duplicate at all. It would accept both and call them independent
        # predictions of different accounts.
        #
        # Canonicalising here rather than in the sweep is the point: a
        # backfill, a one-off StaticAddressSource run and a future writer all
        # come through this method, and any of them could otherwise write the
        # second identity. Nothing can correct it afterwards, because
        # predictions are never updated (§3.4, audit A-09).
        #
        # Read paths deliberately do not normalise: they hand back the bytes
        # that are actually stored, so rows written before this check existed
        # stay visible as themselves instead of being silently papered over.
        #
        # Which is the limit of what this achieves, and the limit is only
        # harmless because the journal is greenfield. A row already written
        # non-canonically is not repaired by anything here, and a journal
        # holding one spelling from before this check and the other from after
        # returns `distinct_addresses == 2` for a single real account -- §3.3's
        # gate inflated by exactly the mechanism this line prevents (measured
        # against an in-memory journal, not argued). There is no detection
        # query and no backfill because there is nothing yet to detect: the
        # shadow counter has not started and no journal database exists. That
        # is a precondition with an expiry date, not a property of the design;
        # see `normalise_address` in domain/types.py for what to write first if
        # a pre-2026-07-30 journal ever turns up.
        address = normalise_address(address)
        resolves_at = predicted_at.timestamp() + horizon_hours * 3600
        # RETURNING on both backends. SQLite has supported it since 3.35 and
        # ships far newer with every Python this targets, so the id comes
        # back the same way everywhere rather than through `lastrowid`,
        # which Postgres does not have at all.
        rows = self._query(
            """
            INSERT INTO calibration_predictions (
                address, variant, predicted_at, horizon_hours, resolves_at,
                model_version, distribution_version, seed, n_paths, converged,
                start_equity, p_liq, p_liq_ci_low, p_liq_ci_high, var_95, cvar_95,
                quantile_values, n_quantile_levels, book_snapshot,
                recorded_under_defect
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            RETURNING id
            """,
            (
                address, variant, _iso(predicted_at), horizon_hours,
                _iso(datetime.fromtimestamp(resolves_at, timezone.utc)),
                model_version, distribution_version, seed, n_paths, bool(converged),
                start_equity, p_liq, p_liq_ci[0], p_liq_ci[1], var_95, cvar_95,
                json.dumps([float(v) for v in distribution.values]),
                len(distribution.levels), json.dumps(book_snapshot),
                bool(recorded_under_defect),
            ),
        )
        self.backend.commit()
        return int(rows[0]["id"])

    def record_sweep(
        self,
        swept_at: datetime,
        distribution_version: str,
        attempted: int,
        written: int,
        budget_exhausted: bool,
        skipped_by_reason: dict[str, int],
    ) -> None:
        """Persist one sweep's census (OPEN-QUESTIONS B6).

        The per-address drop reasons were printed and nowhere durable, so a
        score computed weeks later could not state how selective its cohort
        was. This makes the disclosure travel with the data. Best-effort: a
        census write must never take down the sweep that produced real
        predictions, so a failure here is logged, not raised -- the census is
        provenance about the run, not the run's product.
        """
        try:
            # `execute`, not `_query`: `_query` calls `rows()` = `fetchall()`,
            # and psycopg raises on a fetch after an INSERT with no RETURNING
            # (the result status is COMMAND_OK, not TUPLES_OK). The broad
            # except below then swallowed it as a warning and the row never
            # committed -- so on Postgres, which is what the shipped compose
            # stack points every shadow job at, this table stayed empty every
            # day while the tests (all sqlite, where fetchall after INSERT
            # quietly returns []) passed. `record_outcome` uses `execute` for
            # this reason.
            self.backend.execute(
                self._sql(
                    """
                    INSERT INTO calibration_sweeps (
                        swept_at, observation_day, distribution_version,
                        attempted, written, budget_exhausted, skipped_by_reason
                    ) VALUES (?,?,?,?,?,?,?)
                    """
                ),
                (
                    _iso(swept_at),
                    swept_at.astimezone(timezone.utc).date().isoformat(),
                    distribution_version, int(attempted), int(written),
                    bool(budget_exhausted), json.dumps(skipped_by_reason),
                ),
            )
            self.backend.commit()
        except Exception as exc:
            # Broad on purpose: the census is provenance about the run, not
            # the run's product. A DB hiccup writing it must not discard the
            # predictions the sweep just paid §5.3 weight to produce.
            log.warning("sweep census not recorded (predictions are unaffected): %s", exc)

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
        attempt raises an integrity error; correcting a genuinely wrong
        outcome has to be a deliberate, visible operation.
        """
        self.backend.execute(
            self._sql(
                """
                INSERT INTO calibration_outcomes (
                    prediction_id, resolved_at, actual_equity, actual_equity_change,
                    external_flow_usd, book_changed, liquidated, pit, pit_u, crps,
                    var_95_breached, observation_day, resolution_lag_s, stale_resolution
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """
            ),
            (
                prediction_id, _iso(resolved_at), actual_equity, actual_equity_change,
                external_flow_usd, bool(book_changed), bool(liquidated), pit, pit_u, crps,
                bool(var_95_breached), observation_day.isoformat(),
                resolution_lag_s, bool(stale_resolution),
            ),
        )
        self.backend.commit()

    # -- reading --------------------------------------------------------

    def due(self, as_of: datetime) -> list[PendingPrediction]:
        rows = self._query(
            """
            SELECT p.* FROM calibration_predictions p
            LEFT JOIN calibration_outcomes o ON o.prediction_id = p.id
            WHERE o.prediction_id IS NULL AND p.resolves_at <= ?
            ORDER BY p.resolves_at
            """,
            (_iso(as_of),),
        )
        return [self._to_pending(r) for r in rows]

    @staticmethod
    def _to_pending(row: dict) -> PendingPrediction:
        values = np.array(as_json(row["quantile_values"]), dtype=np.float64)
        levels = np.linspace(0.0, 1.0, row["n_quantile_levels"])
        return PendingPrediction(
            id=int(row["id"]),
            address=row["address"],
            variant=row["variant"],
            predicted_at=as_dt(row["predicted_at"]),
            resolves_at=as_dt(row["resolves_at"]),
            start_equity=float(row["start_equity"]),
            var_95=float(row["var_95"]),
            distribution=PredictiveDistribution(levels=levels, values=values),
            book_snapshot=as_json(row["book_snapshot"]),
        )

    def scored(self, distribution_version: str, variant: str) -> list[dict]:
        """Resolved rows, with the backend differences already normalised."""
        rows = self._query(
            """
            SELECT p.address, p.variant, o.*
            FROM calibration_outcomes o
            JOIN calibration_predictions p ON p.id = o.prediction_id
            WHERE p.distribution_version = ? AND p.variant = ?
            ORDER BY o.observation_day
            """,
            (distribution_version, variant),
        )
        for row in rows:
            for flag in ("book_changed", "liquidated", "var_95_breached",
                         "stale_resolution"):
                row[flag] = as_bool(row[flag])
            row["observation_day"] = as_date(row["observation_day"]).isoformat()
        return rows

    def sweeps(self, distribution_version: str) -> list[dict]:
        """The per-sweep census rows (OPEN-QUESTIONS B6), normalised.

        `record_sweep` has written this table since 2026-08-01 and until now
        NOTHING read it. B6's decision procedure is stated in terms of it --
        "let the census accumulate a few days of full sweeps, then for each
        candidate universe U count the addresses whose entire off-universe
        list is inside U" -- so the data was accumulating against a question
        that could not be asked of it. That is the same shape as a metric
        computed and never read: the write is not the point, the answer is.

        `skipped_by_reason` is written with `json.dumps`, which Postgres
        stores as JSONB and hands back as a dict while SQLite hands back the
        text; `as_json` settles that here rather than in every caller.
        """
        rows = self._query(
            """
            SELECT swept_at, observation_day, attempted, written,
                   budget_exhausted, skipped_by_reason
            FROM calibration_sweeps
            WHERE distribution_version = ?
            ORDER BY swept_at
            """,
            (distribution_version,),
        )
        for row in rows:
            row["budget_exhausted"] = as_bool(row["budget_exhausted"])
            row["observation_day"] = as_date(row["observation_day"]).isoformat()
            row["skipped_by_reason"] = as_json(row["skipped_by_reason"]) or {}
        return rows

    def progress(self, distribution_version: str) -> ShadowProgress:
        row = self._query(
            """
            SELECT COUNT(*) AS n,
                   COUNT(DISTINCT o.observation_day) AS days,
                   COUNT(DISTINCT p.address) AS addrs
            FROM calibration_outcomes o
            JOIN calibration_predictions p ON p.id = o.prediction_id
            WHERE p.distribution_version = ? AND p.variant = ?
              AND o.stale_resolution = ?
              AND p.recorded_under_defect = ?
            """,
            (distribution_version, VARIANT_MODEL, False, False),
        )[0]
        return ShadowProgress(
            distribution_version=distribution_version,
            distinct_days=int(row["days"] or 0),
            distinct_addresses=int(row["addrs"] or 0),
            resolved_observations=int(row["n"] or 0),
        )
