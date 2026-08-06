"""The journal must behave identically on SQLite and Postgres (§3.4).

Run against a real Postgres server when one is reachable, skipped otherwise.
Asserting that two schemas "look the same" is not the same as running the
same code against both: the interesting differences are in what each backend
hands *back* -- Postgres returns `datetime`, `bool` and parsed JSON where
SQLite returns strings and ints -- and only an actual round trip catches a
normaliser that was forgotten.

Point a real server at it with:
    HL_TEST_POSTGRES_DSN=postgresql://user@host/db pytest -k backends
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from risk_engine.shadow.journal import VARIANT_MODEL, CalibrationJournal
from risk_engine.sim.stats import PredictiveDistribution

NOW = datetime(2026, 7, 29, 12, tzinfo=timezone.utc)
DSN = os.environ.get("HL_TEST_POSTGRES_DSN")


def addr(i: int) -> str:
    """A distinct well-formed account address per index.

    The journal canonicalises what it writes (`normalise_address`), so a
    readable stub like "0xa" is refused at the write. Real 40-hex addresses
    here also mean these tests exercise the same identity rules production
    does, rather than a laxer variant of them.
    """
    return f"0x{i:040x}"


def _fresh(target: str) -> CalibrationJournal:
    journal = CalibrationJournal(target)
    if journal.is_postgres:
        journal.backend.executescript(
            "DROP TABLE IF EXISTS calibration_outcomes, calibration_predictions"
        )
        journal = CalibrationJournal(target)
    return journal


@pytest.fixture(params=["sqlite", "postgres"])
def journal(request):
    if request.param == "postgres":
        if not DSN:
            pytest.skip("HL_TEST_POSTGRES_DSN not set")
        j = _fresh(DSN)
    else:
        j = _fresh(":memory:")
    yield j
    j.close()


def _distribution() -> PredictiveDistribution:
    return PredictiveDistribution.from_samples(np.linspace(-1000.0, 1000.0, 5_000))


def _write_prediction(journal: CalibrationJournal, address: str = addr(0)) -> int:
    return journal.record_prediction(
        address=address, variant=VARIANT_MODEL, predicted_at=NOW, horizon_hours=24,
        model_version="0.2.1", distribution_version="0.2", seed=7, n_paths=20_000,
        converged=True, start_equity=100_000.0, p_liq=0.08, p_liq_ci=(0.07, 0.09),
        var_95=5_000.0, cvar_95=8_000.0, distribution=_distribution(),
        book_snapshot={"fingerprint": "abc", "positions": []},
    )


def _write_outcome(journal: CalibrationJournal, pid: int, **overrides) -> None:
    kwargs = dict(
        prediction_id=pid, resolved_at=NOW + timedelta(hours=24), actual_equity=97_000.0,
        actual_equity_change=-3_000.0, external_flow_usd=0.0, book_changed=False,
        liquidated=False, pit=0.42, pit_u=0.5, crps=1_234.5, var_95_breached=False,
        observation_day=date(2026, 7, 29), resolution_lag_s=12.0, stale_resolution=False,
    )
    kwargs.update(overrides)
    journal.record_outcome(**kwargs)


class TestBackendParity:
    def test_a_prediction_round_trips_with_normalised_types(self, journal):
        pid = _write_prediction(journal)
        assert pid > 0
        due = journal.due(NOW + timedelta(hours=25))
        assert len(due) == 1
        pending = due[0]
        # Regardless of backend, the caller gets Python types, not whatever
        # the driver happened to produce.
        assert isinstance(pending.predicted_at, datetime)
        assert pending.predicted_at.tzinfo is not None
        assert isinstance(pending.book_snapshot, dict)
        assert pending.book_snapshot["fingerprint"] == "abc"
        assert pending.distribution.values.size == 1_001
        assert isinstance(pending.start_equity, float)

    def test_nothing_is_due_before_the_horizon_elapses(self, journal):
        _write_prediction(journal)
        assert journal.due(NOW + timedelta(hours=23)) == []

    def test_a_resolved_prediction_stops_being_due(self, journal):
        pid = _write_prediction(journal)
        _write_outcome(journal, pid)
        assert journal.due(NOW + timedelta(hours=25)) == []

    def test_scored_rows_normalise_booleans_and_dates(self, journal):
        pid = _write_prediction(journal)
        _write_outcome(journal, pid, book_changed=True, var_95_breached=True)
        row = journal.scored("0.2", VARIANT_MODEL)[0]
        # SQLite stores these as 0/1 and Postgres as booleans; callers see bool.
        assert row["book_changed"] is True
        assert row["var_95_breached"] is True
        assert row["stale_resolution"] is False
        assert row["observation_day"] == "2026-07-29"
        assert row["pit"] == pytest.approx(0.42)

    def test_an_outcome_cannot_be_rewritten(self, journal):
        """Audit A-09, on whichever backend production happens to use."""
        pid = _write_prediction(journal)
        _write_outcome(journal, pid)
        with pytest.raises(Exception) as exc:
            _write_outcome(journal, pid, pit=0.0, crps=9_999.0)
        assert "IntegrityError" in type(exc.value).__name__ or "Violation" in type(
            exc.value
        ).__name__
        assert journal.scored("0.2", VARIANT_MODEL)[0]["pit"] == pytest.approx(0.42)

    def test_a_duplicate_prediction_is_refused(self, journal):
        _write_prediction(journal)
        with pytest.raises(Exception):
            _write_prediction(journal)

    def test_progress_counts_days_and_addresses(self, journal):
        for i in range(3):
            pid = journal.record_prediction(
                address=addr(i), variant=VARIANT_MODEL, predicted_at=NOW,
                horizon_hours=24, model_version="0.2.1", distribution_version="0.2",
                seed=i, n_paths=100, converged=True, start_equity=1_000.0, p_liq=0.1,
                p_liq_ci=(0.05, 0.15), var_95=10.0, cvar_95=20.0,
                distribution=_distribution(), book_snapshot={},
            )
            _write_outcome(journal, pid)
        progress = journal.progress("0.2")
        assert progress.distinct_addresses == 3
        assert progress.distinct_days == 1
        assert progress.resolved_observations == 3
        assert not progress.gate_open

    def test_stale_resolutions_are_excluded_from_the_gate(self, journal):
        """Audit A-04 holds on both backends, including the boolean compare
        in the WHERE clause, which is exactly the sort of thing that works on
        SQLite's 0/1 and breaks on a real boolean column."""
        pid = _write_prediction(journal)
        _write_outcome(journal, pid, stale_resolution=True, resolution_lag_s=48 * 3600)
        assert journal.progress("0.2").resolved_observations == 0
        assert journal.scored("0.2", VARIANT_MODEL)[0]["stale_resolution"] is True


class TestBatchResilience:
    """The resolver catches per-row failures and carries on (§3.3). That only
    works if a failed row leaves the connection usable -- which on Postgres it
    does not, unless the backend rolls back. SQLite never showed this."""

    def test_a_failed_row_does_not_poison_the_rest_of_the_batch(self, journal):
        ids = [_write_prediction(journal, addr(i)) for i in range(3)]
        _write_outcome(journal, ids[0])
        with pytest.raises(Exception):
            _write_outcome(journal, ids[0])
        # The two remaining rows must still be writable.
        _write_outcome(journal, ids[1])
        _write_outcome(journal, ids[2])
        assert len(journal.scored("0.2", VARIANT_MODEL)) == 3
