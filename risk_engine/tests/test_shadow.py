"""§3.3, §3.4 — the shadow harness and the calibration journal."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from risk_engine.domain.types import Book, MarginMode, Position
from risk_engine.market.info import RateLimitExceeded, WeightBudget
from risk_engine.shadow.cron import ShadowCron, position_fingerprint
from risk_engine.shadow.journal import (
    VARIANT_BASELINE_A,
    VARIANT_BASELINE_B,
    VARIANT_MODEL,
    CalibrationJournal,
)
from risk_engine.shadow.metrics import (
    COHORT_ALL,
    COHORT_BOOK_UNCHANGED,
    calibration_report,
    load_cohort,
    tail_calibration,
)
from risk_engine.shadow.resolve import resolve_due
from risk_engine.sim.stats import PredictiveDistribution
from risk_engine.validation.baselines import NaiveBaseline, historical_24h_log_returns
from risk_engine.version import DISTRIBUTION_VERSION

SCHEMA_SQL = Path(__file__).resolve().parents[1] / "shadow" / "schema.sql"


class FakeProvider:
    """Stands in for the Info API; the cron and resolver see only this."""

    def __init__(self, books, spot, specs, later_books=None, flows=None):
        self._books = books
        self._later = later_books or {}
        self._spot = spot
        self._specs = specs
        self._flows = flows or {}
        self.phase = "before"

    def addresses(self):
        return list(self._books)

    def book(self, address):
        if self.phase == "after" and address in self._later:
            return self._later[address]
        return self._books[address]

    def spot(self):
        return self._spot

    def specs(self):
        return self._specs

    def external_flow(self, address, since, until):
        return self._flows.get(address, 0.0)


@pytest.fixture
def books(now):
    return {
        "0xaaa": Book("0xaaa", 100_000.0,
                      (Position("BTC", 5.0, 100_000.0, MarginMode.CROSS, 20.0),), now),
        "0xbbb": Book("0xbbb", 50_000.0,
                      (Position("ETH", 60.0, 4_000.0, MarginMode.CROSS, 10.0),
                       Position("SOL", 1_000.0, 200.0, MarginMode.ISOLATED, 10.0, 20_000.0)),
                      now),
    }


@pytest.fixture
def naive(synthetic_returns):
    return NaiveBaseline(historical_24h_log_returns(synthetic_returns["BTC"]))


class TestJournalSchema:
    def test_sqlite_and_postgres_schemas_have_the_same_columns(self):
        """The published calibration score is only as good as the guarantee
        that dev and production record the same thing."""
        sql = SCHEMA_SQL.read_text()
        journal = CalibrationJournal()
        for table in ("calibration_predictions", "calibration_outcomes"):
            block = re.search(
                rf"CREATE TABLE IF NOT EXISTS {table} \((.*?)\n\);", sql, re.S
            )
            assert block, f"{table} not found in schema.sql"
            declared = {
                m.group(1)
                for line in block.group(1).splitlines()
                if (m := re.match(r"\s{4}(\w+)\s+\w", line))
            }
            live = {
                r[1] for r in journal.conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            assert declared == live, f"{table}: {declared ^ live}"
        journal.close()

    def test_prediction_is_immutable_once_written(self, now):
        """A prediction that could be edited after resolution would make the
        whole calibration record worthless."""
        journal = CalibrationJournal()
        dist = PredictiveDistribution.from_samples(np.linspace(-100, 100, 1001))
        args = dict(
            address="0xa", variant=VARIANT_MODEL, predicted_at=now, horizon_hours=24,
            model_version="v", distribution_version="0.1", seed=1, n_paths=10,
            converged=True, start_equity=1000.0, p_liq=0.1, p_liq_ci=(0.05, 0.15),
            var_95=50.0, cvar_95=70.0, distribution=dist, book_snapshot={},
        )
        journal.record_prediction(**args)
        import sqlite3

        with pytest.raises(sqlite3.IntegrityError):
            journal.record_prediction(**args)
        journal.close()


class TestShadowSweep:
    def test_writes_model_and_both_baselines_for_every_address(
        self, bundle, specs, spot, books, naive, now
    ):
        journal = CalibrationJournal()
        provider = FakeProvider(books, spot, specs)
        report = ShadowCron(provider, bundle, journal, naive, n_paths=2_000).run_once(now)

        assert report.written == 2
        for variant in (VARIANT_MODEL, VARIANT_BASELINE_A, VARIANT_BASELINE_B):
            rows = journal.conn.execute(
                "SELECT COUNT(*) c FROM calibration_predictions WHERE variant=?", (variant,)
            ).fetchone()
            assert rows["c"] == 2, variant
        journal.close()

    def test_one_bad_address_does_not_stop_the_sweep(
        self, bundle, specs, spot, books, naive, now
    ):
        broken = dict(books)
        broken["0xdead"] = Book("0xdead", 0.0, (), now)  # no positions
        provider = FakeProvider(broken, spot, specs)
        journal = CalibrationJournal()
        report = ShadowCron(provider, bundle, journal, naive, n_paths=1_000).run_once(now)
        assert report.written == 2
        assert [a for a, _ in report.skipped] == ["0xdead"]
        journal.close()

    def test_yields_to_live_users_when_the_budget_runs_out(
        self, bundle, specs, spot, books, naive, now
    ):
        """§5.3: the background sweep must never spend into the reserve."""
        budget = WeightBudget(limit_per_minute=100, reserved_fraction=0.75)
        provider = FakeProvider(books, spot, specs)
        journal = CalibrationJournal()
        report = ShadowCron(
            provider, bundle, journal, naive, n_paths=500, budget=budget
        ).run_once(now)
        assert report.budget_exhausted
        assert report.written < len(books)
        journal.close()

    def test_budget_refuses_to_dip_into_the_reserve(self):
        budget = WeightBudget(limit_per_minute=1000, reserved_fraction=0.75)
        budget.charge(250)
        assert budget.available() == 0
        with pytest.raises(RateLimitExceeded):
            budget.charge(1)
        # An interactive caller with no reserve still has the rest.
        live = WeightBudget(limit_per_minute=1000)
        live.charge(900)
        assert live.available() == 100


class TestResolve:
    def test_fills_in_the_outcome_a_day_later(
        self, bundle, specs, spot, books, naive, now
    ):
        journal = CalibrationJournal()
        provider = FakeProvider(books, spot, specs)
        ShadowCron(provider, bundle, journal, naive, n_paths=2_000).run_once(now)

        later = now + timedelta(hours=24)
        assert journal.due(now) == []  # nothing is due before the horizon elapses
        assert len(journal.due(later)) == 6

        report = resolve_due(journal, provider, later)
        assert report.resolved == 6
        assert not report.failed
        assert journal.due(later) == []  # resolved rows are not returned again

        rows = journal.scored(DISTRIBUTION_VERSION, VARIANT_MODEL)
        assert len(rows) == 2
        for r in rows:
            assert 0.0 <= r["pit"] <= 1.0
            assert r["crps"] >= 0.0
            assert r["observation_day"] == now.date().isoformat()
        journal.close()

    def test_external_flow_is_removed_before_scoring(
        self, bundle, specs, spot, books, naive, now
    ):
        """OPEN-QUESTIONS B2: a deposit is not a model error.

        The user wires in $500k mid-horizon, so their equity jumps by that
        amount for a reason the model never claimed to predict. Scored raw,
        this address would look like a spectacular forecasting failure; the
        flow correction must leave the market-driven change behind.
        """
        journal = CalibrationJournal()
        deposit = 500_000.0
        after = {
            "0xaaa": Book("0xaaa", books["0xaaa"].cross_collateral + deposit,
                          books["0xaaa"].positions, now)
        }
        provider = FakeProvider(books, spot, specs, later_books=after,
                                flows={"0xaaa": deposit})
        ShadowCron(provider, bundle, journal, naive, n_paths=1_000).run_once(now)
        later = now + timedelta(hours=24)
        provider.phase = "after"
        resolve_due(journal, provider, later)

        rows = {r["address"]: r for r in journal.scored(DISTRIBUTION_VERSION, VARIANT_MODEL)}
        assert rows["0xaaa"]["external_flow_usd"] == deposit
        assert rows["0xaaa"]["actual_equity"] == pytest.approx(
            rows["0xaaa"]["actual_equity_change"] + deposit + 100_000.0
        )
        # Prices did not move in the fake, so all that is left is ~zero.
        assert abs(rows["0xaaa"]["actual_equity_change"]) < 1.0
        assert load_cohort(journal, DISTRIBUTION_VERSION, VARIANT_MODEL, COHORT_ALL).n == 2
        journal.close()

    def test_a_restructured_book_is_flagged_not_silently_scored(
        self, bundle, specs, spot, books, naive, now
    ):
        after = {
            "0xaaa": Book("0xaaa", 100_000.0,
                          (Position("BTC", 1.0, 100_000.0, MarginMode.CROSS, 20.0),), now)
        }
        provider = FakeProvider(books, spot, specs, later_books=after)
        journal = CalibrationJournal()
        ShadowCron(provider, bundle, journal, naive, n_paths=1_000).run_once(now)
        provider.phase = "after"
        resolve_due(journal, provider, now + timedelta(hours=24))

        rows = {r["address"]: r for r in journal.scored(DISTRIBUTION_VERSION, VARIANT_MODEL)}
        assert rows["0xaaa"]["book_changed"]
        assert not rows["0xbbb"]["book_changed"]
        unchanged = load_cohort(
            journal, DISTRIBUTION_VERSION, VARIANT_MODEL, COHORT_BOOK_UNCHANGED
        )
        assert unchanged.n == 1
        journal.close()

    def test_fingerprint_ignores_price_but_not_size(self, now):
        a = Book("0x", 1.0, (Position("BTC", 2.0, 100_000.0, MarginMode.CROSS, 5.0),), now)
        moved = Book("0x", 1.0, (Position("BTC", 2.0, 90_000.0, MarginMode.CROSS, 5.0),), now)
        resized = Book("0x", 1.0, (Position("BTC", 3.0, 100_000.0, MarginMode.CROSS, 5.0),), now)
        assert position_fingerprint(a) == position_fingerprint(moved)
        assert position_fingerprint(a) != position_fingerprint(resized)


class TestCalibrationMetrics:
    """Driven by a synthetic journal, so the metric layer is tested against
    outcomes whose truth is known rather than against the engine's own output."""

    @staticmethod
    def _fill(journal, n_days=30, per_day=15, seed=0):
        rng = np.random.default_rng(seed)
        # Separate stream for the randomized-PIT uniforms: drawing them from
        # `rng` would shift the data stream and silently change what this
        # fixture is testing.
        u_rng = np.random.default_rng(seed + 7919)
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        truth = PredictiveDistribution.from_samples(rng.normal(0, 1000, 200_000))
        wrong = PredictiveDistribution.from_samples(rng.normal(0, 4000, 200_000))
        for d in range(n_days):
            when = start + timedelta(days=d)
            shock = rng.normal(0, 1000)  # one market move shared by the day
            for i in range(per_day):
                actual = 0.7 * shock + 0.7141 * rng.normal(0, 1000)
                for variant, dist in (
                    (VARIANT_MODEL, truth),
                    (VARIANT_BASELINE_A, wrong),
                    (VARIANT_BASELINE_B, wrong),
                ):
                    pid = journal.record_prediction(
                        address=f"0x{i:03d}", variant=variant, predicted_at=when,
                        horizon_hours=24, model_version="v", distribution_version="test",
                        seed=1, n_paths=100, converged=True, start_equity=100_000.0,
                        p_liq=0.01, p_liq_ci=(0.005, 0.02),
                        var_95=-dist.quantile(0.05), cvar_95=dist.cvar(0.95),
                        distribution=dist, book_snapshot={"fingerprint": "x"},
                    )
                    u = u_rng.random()
                    journal.record_outcome(
                        prediction_id=pid, resolved_at=when + timedelta(days=1),
                        actual_equity=100_000.0 + actual, actual_equity_change=actual,
                        external_flow_usd=0.0, book_changed=False,
                        liquidated=False, pit=dist.pit(actual, u), pit_u=u,
                        crps=dist.crps(actual),
                        var_95_breached=actual < dist.quantile(0.05),
                        observation_day=(when).date(),
                        resolution_lag_s=0.0, stale_resolution=False,
                    )

    def test_model_beats_the_baselines_when_it_is_the_true_distribution(self):
        journal = CalibrationJournal()
        self._fill(journal)
        report = calibration_report(journal, "test", COHORT_ALL)
        assert report.crps_beats_baseline_a
        assert report.crps_beats_baseline_b
        assert report.ks_pvalue > 0.01
        journal.close()

    def test_day_clustering_widens_the_interval_the_spec_assumes_is_tight(self):
        """OPEN-QUESTIONS B1: observations on one day share one market, so the
        naive binomial interval overstates precision badly."""
        journal = CalibrationJournal()
        self._fill(journal)
        cohort = load_cohort(journal, "test", VARIANT_MODEL, COHORT_ALL)
        tail = tail_calibration(cohort, seed=3)
        naive_width = tail.naive_ci[1] - tail.naive_ci[0]
        clustered_width = tail.clustered_ci[1] - tail.clustered_ci[0]
        assert clustered_width > naive_width * 1.5
        journal.close()

    def test_gate_stays_closed_until_the_window_is_long_and_wide_enough(self):
        journal = CalibrationJournal()
        self._fill(journal, n_days=10, per_day=5)
        progress = journal.progress("test")
        assert progress.distinct_days == 10
        assert progress.distinct_addresses == 5
        assert not progress.gate_open
        assert "CLOSED" in str(progress)
        journal.close()

    def test_gate_opens_at_21_days_and_200_addresses(self):
        journal = CalibrationJournal()
        self._fill(journal, n_days=21, per_day=200)
        progress = journal.progress("test")
        assert progress.gate_open
        journal.close()

    def test_a_model_version_bump_resets_the_window(self, now):
        journal = CalibrationJournal()
        self._fill(journal, n_days=25, per_day=200)
        assert journal.progress("test").gate_open
        # §3.3: a distribution-affecting change starts the count again.
        assert not journal.progress("test-v2").gate_open
        journal.close()

    def test_cohorts_are_reported_separately(self):
        journal = CalibrationJournal()
        self._fill(journal, n_days=5, per_day=4)
        all_rows = load_cohort(journal, "test", VARIANT_MODEL, COHORT_ALL)
        unchanged = load_cohort(journal, "test", VARIANT_MODEL, COHORT_BOOK_UNCHANGED)
        assert all_rows.n == unchanged.n == 20
        assert all_rows.n_days == 5


class TestBaselines:
    def test_naive_baseline_has_no_correlation_structure(
        self, specs, spot, books, naive
    ):
        pred = naive.predict(books["0xbbb"], spot, specs, n_draws=5_000, seed=1)
        assert pred.start_equity > 0
        assert 0.0 <= pred.p_liq <= 1.0
        assert pred.equity_change.quantile(0.99) > pred.equity_change.quantile(0.01)

    def test_baseline_b_is_the_same_engine_with_dependence_off(
        self, bundle, specs, spot, books, now
    ):
        from risk_engine.validation.baselines import run_baseline_b

        out = run_baseline_b(bundle, specs, books["0xbbb"], spot, 24,
                             n_paths=4_000, seed=2, now=now)
        assert out.provenance.model_version == bundle.model_version
        assert out.p_liq_any.ci_low <= out.p_liq_any.point <= out.p_liq_any.ci_high

    def test_overlapping_24h_windows_are_produced_correctly(self):
        r = np.arange(100, dtype=float)
        out = historical_24h_log_returns(r)
        assert out.size == 100 - 24 + 1  # overlapping windows, not disjoint
        assert out[0] == pytest.approx(r[:24].sum())
        assert out[-1] == pytest.approx(r[-24:].sum())


class TestResolutionStaleness:
    """Audit A-04: a 24h forecast scored against a 72h realisation measures
    the resolver's punctuality, not the model."""

    def test_a_late_resolution_is_flagged_and_excluded(
        self, bundle, specs, spot, books, naive, now
    ):
        journal = CalibrationJournal()
        provider = FakeProvider(books, spot, specs)
        ShadowCron(provider, bundle, journal, naive, n_paths=1_000).run_once(now)

        report = resolve_due(journal, provider, now + timedelta(hours=72))
        assert report.resolved == 6
        assert report.stale == 6
        assert "flagged stale" in str(report)

        rows = journal.scored(DISTRIBUTION_VERSION, VARIANT_MODEL)
        assert all(r["stale_resolution"] for r in rows)
        assert all(r["resolution_lag_s"] == pytest.approx(48 * 3600.0) for r in rows)

        # Excluded from every cohort, and from the Phase-4 gate counters.
        assert load_cohort(journal, DISTRIBUTION_VERSION, VARIANT_MODEL, COHORT_ALL).n == 0
        assert load_cohort(
            journal, DISTRIBUTION_VERSION, VARIANT_MODEL, COHORT_ALL, include_stale=True
        ).n == 2
        assert journal.progress(DISTRIBUTION_VERSION).resolved_observations == 0
        journal.close()

    def test_a_punctual_resolution_is_scored(
        self, bundle, specs, spot, books, naive, now
    ):
        journal = CalibrationJournal()
        provider = FakeProvider(books, spot, specs)
        ShadowCron(provider, bundle, journal, naive, n_paths=1_000).run_once(now)
        report = resolve_due(journal, provider, now + timedelta(hours=24, minutes=20))
        assert report.stale == 0
        assert load_cohort(journal, DISTRIBUTION_VERSION, VARIANT_MODEL, COHORT_ALL).n == 2
        journal.close()

    def test_the_randomized_pit_is_reproducible_from_the_journal(
        self, bundle, specs, spot, books, naive, now
    ):
        """The uniform is derived from the prediction id, not drawn freshly:
        a row can be re-derived, and the draw cannot be retried until it
        flatters the model."""
        from risk_engine.shadow.resolve import _pit_uniform

        journal = CalibrationJournal()
        provider = FakeProvider(books, spot, specs)
        ShadowCron(provider, bundle, journal, naive, n_paths=1_000).run_once(now)
        resolve_due(journal, provider, now + timedelta(hours=24))
        for r in journal.scored(DISTRIBUTION_VERSION, VARIANT_MODEL):
            assert r["pit_u"] == pytest.approx(_pit_uniform(r["prediction_id"]))
        journal.close()


class TestOutcomeImmutability:
    """Audit A-09: the calibration record is only worth what the guarantee
    that a written outcome cannot be quietly rewritten is worth."""

    def test_a_second_outcome_for_the_same_prediction_is_refused(self, now):
        import sqlite3

        journal = CalibrationJournal()
        dist = PredictiveDistribution.from_samples(np.linspace(-100, 100, 1001))
        pid = journal.record_prediction(
            address="0xa", variant=VARIANT_MODEL, predicted_at=now, horizon_hours=24,
            model_version="v", distribution_version="0.1", seed=1, n_paths=10,
            converged=True, start_equity=1000.0, p_liq=0.1, p_liq_ci=(0.05, 0.15),
            var_95=50.0, cvar_95=70.0, distribution=dist, book_snapshot={},
        )
        kwargs = dict(
            prediction_id=pid, resolved_at=now, actual_equity=900.0,
            actual_equity_change=-100.0, external_flow_usd=0.0, book_changed=False,
            liquidated=False, pit=0.4, pit_u=0.5, crps=1.0, var_95_breached=False,
            observation_day=now.date(), resolution_lag_s=0.0, stale_resolution=False,
        )
        journal.record_outcome(**kwargs)
        with pytest.raises(sqlite3.IntegrityError):
            journal.record_outcome(**{**kwargs, "pit": 0.0, "crps": 999.0})
        assert journal.scored("0.1", VARIANT_MODEL)[0]["pit"] == pytest.approx(0.4)
        journal.close()


class TestVersionGating:
    """§3.3 anti-overfitting: a distribution-affecting change resets the
    shadow window. The audit fixes changed baseline B's distribution and the
    7d path set, so observations recorded under 0.1.x cannot be pooled with
    0.2.x -- and neither can their PIT values, which were computed by the
    non-randomized transform."""

    def test_distribution_version_tracks_major_minor_only(self):
        from risk_engine.version import DISTRIBUTION_VERSION, MODEL_VERSION

        assert MODEL_VERSION.startswith(DISTRIBUTION_VERSION + ".")
        assert DISTRIBUTION_VERSION.count(".") == 1

    def test_the_audit_fixes_moved_the_distribution_version(self):
        from risk_engine.version import DISTRIBUTION_VERSION

        assert DISTRIBUTION_VERSION != "0.1", (
            "baseline B's copula and the shared-walk horizons changed the "
            "predicted distribution; pooling 0.1.x shadow days would be "
            "exactly the overfitting §3.3 forbids"
        )
