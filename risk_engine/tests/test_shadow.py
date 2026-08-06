"""§3.3, §3.4 — the shadow harness and the calibration journal."""

from __future__ import annotations

import inspect
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

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
    Cohort,
    calibration_report,
    load_cohort,
    tail_calibration,
)
from risk_engine.shadow.providers import StaticAddressSource
from risk_engine.shadow.resolve import resolve_due
from risk_engine.sim.stats import PredictiveDistribution
from risk_engine.validation.baselines import NaiveBaseline, historical_24h_log_returns
from risk_engine.version import DISTRIBUTION_VERSION

SCHEMA_SQL = Path(__file__).resolve().parents[1] / "shadow" / "schema.sql"

# Well-formed account addresses. The journal canonicalises what it writes
# (`normalise_address`), so a readable stub like "0xaaa" is refused at the
# write -- which is the point of these being real: the fixtures exercise the
# identity rules production runs under rather than a laxer variant of them.
ADDR_A = "0x" + "a" * 40
ADDR_B = "0x" + "b" * 40
ADDR_FLAT = "0x" + "d" * 40


def addr(i: int) -> str:
    return f"0x{i:040x}"


def _insert_legacy_prediction(journal, address: str, predicted_at, hours: int = 24) -> int:
    """Write a pending prediction the way a row written before the address
    discipline existed sits in the journal today.

    Deliberately bypasses `record_prediction`, which normalises: the point of
    these rows is that they predate that check (audit A-09) and that the read
    paths hand back the stored bytes rather than papering over them. There is
    no other way to construct the state, and pretending it cannot occur is how
    the resolver's forever-retry went unnoticed.
    """
    import json

    resolves_at = predicted_at + timedelta(hours=hours)
    rows = journal._query(
        """
        INSERT INTO calibration_predictions (
            address, variant, predicted_at, horizon_hours, resolves_at,
            model_version, distribution_version, seed, n_paths, converged,
            start_equity, p_liq, p_liq_ci_low, p_liq_ci_high, var_95, cvar_95,
            quantile_values, n_quantile_levels, book_snapshot
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        RETURNING id
        """,
        (
            address, VARIANT_MODEL, predicted_at.isoformat(), hours,
            resolves_at.isoformat(), "0.2.0", DISTRIBUTION_VERSION, 1, 10, True,
            1_000.0, 0.0, 0.0, 0.0, 100.0, 100.0,
            json.dumps(list(np.linspace(-1_000.0, 1_000.0, 11))), 11, json.dumps({}),
        ),
    )
    journal.backend.commit()
    return int(rows[0]["id"])


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


class _BudgetedProvider(FakeProvider):
    """A provider whose `book()` charges a rate limit, like the real client.

    `allow` calls succeed, then `book()` raises `RateLimitExceeded` — exactly
    as `InfoClient.post` does when the §5.3 window is spent. `release_after`
    counts refusals and starts allowing again once that many have been seen,
    modelling a sliding window that refills while the sweep waits.
    """

    def __init__(self, books, spot, specs, *, allow=0, release_after=None):
        super().__init__(books, spot, specs)
        self._allowed = allow
        self._release_after = release_after
        self._refusals = 0

    def book(self, address):
        if self._allowed > 0:
            self._allowed -= 1
            return super().book(address)
        self._refusals += 1
        if self._release_after is not None and self._refusals >= self._release_after:
            self._allowed = 1_000_000  # window refilled
            self._allowed -= 1
            return super().book(address)
        raise RateLimitExceeded("weight 20 exceeds remaining 0")


@pytest.fixture
def books(now):
    return {
        ADDR_A: Book(ADDR_A, 100_000.0,
                     (Position("BTC", 5.0, 100_000.0, MarginMode.CROSS, 20.0),), now),
        ADDR_B: Book(ADDR_B, 50_000.0,
                     (Position("ETH", 60.0, 4_000.0, MarginMode.CROSS, 10.0),
                      Position("SOL", 1_000.0, 200.0, MarginMode.ISOLATED, 10.0, 20_000.0)),
                     now),
    }


@pytest.fixture
def naive(synthetic_returns):
    return NaiveBaseline(historical_24h_log_returns(synthetic_returns["BTC"]))


class TestJournalSchema:
    def test_the_sqlite_schema_is_derived_from_the_canonical_one(self):
        """One schema, not two. A hand-maintained second copy is a second
        thing to forget to update, and the failure surfaces years later as an
        unexplained discontinuity in a published calibration score."""
        from risk_engine.shadow.backends import canonical_ddl, sqlite_ddl

        canonical = canonical_ddl()
        derived = sqlite_ddl()
        for table in ("calibration_predictions", "calibration_outcomes", "calibration_sweeps"):
            assert f"CREATE TABLE IF NOT EXISTS {table}" in canonical
            assert f"CREATE TABLE IF NOT EXISTS {table}" in derived
        # Postgres-only spellings must not survive the translation.
        for postgres_only in ("BIGSERIAL", "TIMESTAMPTZ", "JSONB", "DOUBLE PRECISION"):
            assert postgres_only not in derived, postgres_only

    def test_every_declared_column_exists_in_the_live_sqlite_schema(self):
        sql = SCHEMA_SQL.read_text()
        journal = CalibrationJournal()
        for table in ("calibration_predictions", "calibration_outcomes", "calibration_sweeps"):
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
                r["name"]
                for r in journal._query(f"PRAGMA table_info({table})")
            }
            assert declared == live, f"{table}: {declared ^ live}"
        journal.close()

    def test_the_sweep_census_records_cohort_selection(self, now):
        """OPEN-QUESTIONS B6: the per-address drop reasons must be durable, so
        a score computed later can state how selective its cohort was. Printed
        and nowhere else, they scroll off a container log."""
        journal = CalibrationJournal()
        journal.record_sweep(
            now, "0.3", attempted=515, written=340, budget_exhausted=False,
            skipped_by_reason={
                "KeyError: 'ATOM'": 40, "no open positions": 90,
                "non-positive equity": 45,
            },
        )
        rows = journal._query("SELECT * FROM calibration_sweeps")
        assert len(rows) == 1
        row = rows[0]
        assert row["attempted"] == 515 and row["written"] == 340
        import json as _json
        tally = _json.loads(row["skipped_by_reason"])
        # The off-universe drop rate is recoverable per day, which is the whole
        # point — a bare "175 skipped" could not distinguish B6 selection from
        # flat books.
        off_universe = sum(v for k, v in tally.items() if k.startswith("KeyError"))
        assert off_universe == 40
        assert row["observation_day"] == now.date().isoformat()
        journal.close()

    def test_a_census_write_failure_does_not_sink_the_sweep(self, now):
        """The census is provenance, not product. A DB hiccup writing it must
        not discard predictions the sweep paid §5.3 weight to produce."""
        journal = CalibrationJournal()
        journal.backend.close()  # force every subsequent write to raise
        # Must not raise — the failure is logged and swallowed.
        journal.record_sweep(now, "0.3", 1, 1, False, {})

    def test_prediction_is_immutable_once_written(self, now):
        """A prediction that could be edited after resolution would make the
        whole calibration record worthless."""
        journal = CalibrationJournal()
        dist = PredictiveDistribution.from_samples(np.linspace(-100, 100, 1001))
        args = dict(
            address=ADDR_A, variant=VARIANT_MODEL, predicted_at=now, horizon_hours=24,
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
            rows = journal._query(
                "SELECT COUNT(*) c FROM calibration_predictions WHERE variant=?", (variant,)
            )[0]
            assert rows["c"] == 2, variant
        journal.close()

    def test_one_bad_address_does_not_stop_the_sweep(
        self, bundle, specs, spot, books, naive, now
    ):
        broken = dict(books)
        broken[ADDR_FLAT] = Book(ADDR_FLAT, 0.0, (), now)  # no positions
        provider = FakeProvider(broken, spot, specs)
        journal = CalibrationJournal()
        report = ShadowCron(provider, bundle, journal, naive, n_paths=1_000).run_once(now)
        assert report.written == 2
        assert [a for a, _ in report.skipped] == [ADDR_FLAT]
        journal.close()

    def test_the_sweep_waits_on_the_budget_the_api_actually_charges(
        self, bundle, specs, spot, books, naive, now
    ):
        """The §5.3 limit is enforced by the InfoClient's budget, charged
        INSIDE `book()`. The sweep must wait on that, not on a counter of its
        own — an earlier fix waited on a cron-owned budget while the real
        request charged a different object, so the true limit surfaced as a
        `RateLimitExceeded` skip and the address was dropped. This drives the
        real mechanism: a provider whose `book()` refuses until released.

        Ceiling 0 so a refusal ends the run immediately (instant test); with
        no ceiling it would wait and eventually succeed."""
        provider = _BudgetedProvider(books, spot, specs, allow=0)
        journal = CalibrationJournal()
        report = ShadowCron(
            provider, bundle, journal, naive, n_paths=500, max_sweep_seconds=0.0,
        ).run_once(now)
        assert report.budget_exhausted
        assert report.written == 0  # every book() refused, none dropped as a skip
        # The refusals are NOT recorded as per-address skips: a rate limit is a
        # pacing signal, not a property of the address.
        assert not any("RateLimit" in reason for _, reason in report.skipped)
        journal.close()

    def test_a_transient_rate_limit_is_waited_through_not_skipped(
        self, bundle, specs, spot, books, naive, now
    ):
        """The whole point of the fix: an address refused once is retried, not
        lost. `book()` refuses the first call and succeeds after, and the
        address must end up written."""
        provider = _BudgetedProvider(books, spot, specs, allow=0, release_after=1)
        journal = CalibrationJournal()
        report = ShadowCron(
            provider, bundle, journal, naive, n_paths=500,
            max_sweep_seconds=60.0, budget_wait_seconds=0.0,
        ).run_once(now)
        assert report.written == len(books)
        assert not report.budget_exhausted
        journal.close()

    def test_the_sweep_says_it_is_alive_while_it_works(
        self, bundle, specs, spot, books, naive, now, caplog
    ):
        """A sweep that logs nothing is indistinguishable from a hung one.

        `run_once` had no logging at all, so a real list -- tens of minutes,
        up to a 90-minute ceiling -- produced not one line between the bundle
        build and the final report. That is worse here than it would be
        elsewhere, because this sweep is DESIGNED to spend most of its wall
        clock asleep waiting on §5.3's window: its healthy state and a
        deadlock look identical from outside. It cost two rounds of
        "is it working?" on the same deployment before anyone read the code.
        """
        import logging as _logging

        provider = _BudgetedProvider(books, spot, specs, allow=len(books) + 8)
        journal = CalibrationJournal()
        with caplog.at_level(_logging.INFO, logger="risk_engine.shadow.cron"):
            ShadowCron(provider, bundle, journal, naive, n_paths=500).run_once(now)
        journal.close()

        lines = [r.getMessage() for r in caplog.records]
        assert any(f"sweeping {len(books)} addresses" in m for m in lines), lines
        # The closing line is forced rather than rate-limited: a sweep short
        # enough to finish inside one PROGRESS_LOG_SECONDS window would
        # otherwise report its start and never its result.
        assert any("addresses:" in m and "written" in m for m in lines), lines

    def test_the_progress_line_separates_waiting_from_elapsed(
        self, bundle, specs, spot, books, naive, now, caplog
    ):
        """Those two numbers are the diagnosis. Mostly-waiting is §5.3
        working as designed; elapsed climbing while neither addresses nor
        waiting move is a stall worth acting on. One combined number would
        not tell them apart, which is the question the operator actually has.
        """
        import logging as _logging

        provider = _BudgetedProvider(books, spot, specs, allow=0, release_after=1)
        journal = CalibrationJournal()
        with caplog.at_level(_logging.INFO, logger="risk_engine.shadow.cron"):
            ShadowCron(
                provider, bundle, journal, naive, n_paths=500,
                max_sweep_seconds=60.0, budget_wait_seconds=0.0,
            ).run_once(now)
        journal.close()

        final = [r.getMessage() for r in caplog.records if "elapsed" in r.getMessage()]
        assert final, [r.getMessage() for r in caplog.records]
        assert "waiting for the §5.3 window" in final[-1]

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

    def test_the_bundle_build_waits_for_the_window_instead_of_dying(self, monkeypatch):
        """A spent window at STARTUP must pace the run, not kill it.

        Observed live: a one-off snapshot started while the scheduled resolver
        held the shared pool, and `RateLimitExceeded` came straight out of
        `meta()` before a single prediction was written. Each process used to
        own its budget, so a fresh process always began with a full window and
        this could not happen; sharing the pool (C6) removed that guarantee
        and the unpaced build turned contention into a dead run.

        Same burst-and-drop failure the sweep and resolver were both fixed
        for, one layer up: a rate limit is a PACE, not an error.
        """
        from risk_engine.shadow import cli

        calls = {"n": 0}

        def flaky(*, serving, budget):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RateLimitExceeded("weight 20 exceeds remaining 0")
            return ("bundle", "specs", {"BTC": 1.0})

        monkeypatch.setattr(
            "risk_engine.service.state._build_live_bundle", flaky
        )
        out = cli._paced_bundle(budget=None, max_wait_s=5.0, wait_s=0.0)
        assert out == ("bundle", "specs", {"BTC": 1.0})
        assert calls["n"] == 3, "it should have retried rather than given up"

    def test_a_permanently_full_window_fails_loudly_rather_than_hanging(
        self, monkeypatch
    ):
        """The ceiling exists so an oversubscribed pool is visible.

        Waiting forever would leave a container asleep and the §3.3 window
        silently not advancing, which is the failure mode the whole
        pace-don't-drop design is trying to avoid in the other direction.
        """
        from risk_engine.shadow import cli

        def always_full(*, serving, budget):
            raise RateLimitExceeded("weight 20 exceeds remaining 0")

        monkeypatch.setattr(
            "risk_engine.service.state._build_live_bundle", always_full
        )
        with pytest.raises(SystemExit, match="weight window"):
            cli._paced_bundle(budget=None, max_wait_s=0.0, wait_s=0.0)


class TestTheSweepPacesRatherThanTruncating:
    """§3.3's gate was unreachable by construction.

    The budget is a sliding minute at 25% of 1200 (§5.3): 300 weight buys 15
    addresses, and the loop then `break`. §3.3 wants 200 addresses a DAY, so a
    daily job that gave up after one minute delivered 15. Measured on the
    first live run: 6 of 515 written, "budget exhausted, sweep truncated".

    OPEN-QUESTIONS B4's own table said a 500-address sweep costs ~33 minutes.
    That was the arithmetic of a sweep that WAITS. The code did not wait, so
    the documentation described behaviour the code never had.
    """

    def test_it_waits_for_the_window_instead_of_abandoning_the_list(self):
        """The refill is what makes the gate reachable. Driven through a fake
        clock so the assertion is about the logic, not about elapsed time."""
        from risk_engine.market.info import RateLimitExceeded

        class TinyBudget:
            """Three charges, then refuses until `release()` is called."""

            def __init__(self):
                self.spent = 0
                self.refusals = 0

            def charge(self, weight, now=None):
                if self.spent >= 3:
                    self.refusals += 1
                    raise RateLimitExceeded("window spent")
                self.spent += 1

            def release(self):
                self.spent = 0

        budget = TinyBudget()
        sleeps: list[float] = []

        def fake_sleep(seconds):
            sleeps.append(seconds)
            budget.release()  # the window refills while we wait

        charged = 0
        deadline_hit = False
        for _ in range(5):
            while True:
                try:
                    budget.charge(20)
                    charged += 1
                    break
                except RateLimitExceeded:
                    if len(sleeps) > 10:
                        deadline_hit = True
                        break
                    fake_sleep(5.0)
            if deadline_hit:
                break

        assert charged == 5, "the sweep must finish the list, not stop at the window"
        assert budget.refusals >= 1, "the test must actually exercise the wait"
        assert sleeps, "waiting is how it yields to live users (§5.3)"

    def test_a_ceiling_still_bounds_one_sweep(self):
        """Waiting must not become an unbounded sleep loop: a list that cannot
        finish has to end the run, not collide with tomorrow's."""
        from risk_engine.shadow.cron import MAX_SWEEP_SECONDS

        # 515 addresses at 20 weight against 300/min is ~34 minutes.
        assert MAX_SWEEP_SECONDS > 34 * 60, "no headroom over the real sweep cost"
        assert MAX_SWEEP_SECONDS < 24 * 3600, "a sweep may not outlive its own cadence"

    def test_the_gate_is_now_arithmetically_reachable(self):
        """The property that was false before: one daily sweep must be able to
        cover §3.3's address requirement inside its own ceiling."""
        from risk_engine.market.info import (
            INFO_REQUEST_WEIGHT,
            WEIGHT_BUDGET_PER_MINUTE,
        )
        from risk_engine.shadow.cron import (
            MAX_SWEEP_SECONDS,
            SHADOW_RESERVED_FRACTION,
        )

        per_minute = WEIGHT_BUDGET_PER_MINUTE * (1 - SHADOW_RESERVED_FRACTION)
        addresses_per_minute = per_minute / INFO_REQUEST_WEIGHT
        reachable = addresses_per_minute * (MAX_SWEEP_SECONDS / 60.0)
        assert reachable >= 200, (
            f"one sweep reaches {reachable:.0f} addresses, below §3.3's 200"
        )


class TestCheapInputsAreValidatedFirst:
    """An input that can fail for free must fail before one that costs.

    First live run: `psycopg` was missing from the engine image, and the
    snapshot job discovered it *after* fitting a bundle and spending §5.3 API
    weight — reported as a bare ModuleNotFoundError traceback with the §2.3
    defect warning scrolled off above it. `_live_world` already carries this
    exact fix for the address list; the journal never got it."""

    @staticmethod
    def _order_in(func) -> tuple[int, int]:
        import inspect
        src = inspect.getsource(func)
        return src.index("CalibrationJournal("), src.index("_world(")

    def test_the_snapshot_opens_the_journal_before_building_a_world(self):
        from risk_engine.shadow.cli import cmd_snapshot

        journal_at, world_at = self._order_in(cmd_snapshot)
        assert journal_at < world_at, (
            "the world is built before the journal is opened: an unopenable "
            "journal then costs a bundle fit and API weight to discover"
        )

    def test_the_resolver_does_the_same(self):
        """Hourly, so a wasted build costs 24x more than the snapshot's."""
        from risk_engine.shadow.cli import cmd_resolve

        journal_at, world_at = self._order_in(cmd_resolve)
        assert journal_at < world_at

    def test_a_missing_driver_names_the_fix_rather_than_the_import(self, monkeypatch):
        """`ModuleNotFoundError: No module named 'psycopg'` reads as a broken
        build. It was a lazily-imported driver absent from requirements while
        the shipped compose stack pointed every shadow job at Postgres."""
        import builtins

        from risk_engine.shadow.backends import PostgresBackend

        real_import = builtins.__import__

        def no_psycopg(name, *a, **k):
            if name.startswith("psycopg"):
                raise ImportError("No module named 'psycopg'")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", no_psycopg)
        with pytest.raises(ImportError, match=r"psycopg\[binary\]"):
            PostgresBackend("postgresql://x@y/z")

    def test_the_driver_is_declared_where_the_image_installs_from(self):
        """The lazy import was read as 'not a dependency of this image'. One
        image serves the engine AND the shadow jobs (Dockerfile.engine says
        two would let the journal record predictions attributed to a version
        that never produced them), and that image is what every compose
        deployment points at a real Postgres DSN."""
        reqs = Path(__file__).resolve().parents[1] / "requirements.txt"
        assert "psycopg" in reqs.read_text(encoding="utf-8")


class TestARefusedAddressList:
    """A malformed list takes the whole sweep down, and must say so.

    `AddressSource.addresses()` refuses an entry that is not an address, which
    is right -- the alternative put a typo on the wire, where §5.1's
    well-formed empty state resolved as equity 0 and wrote liquidated=1,
    var_95_breached=1 into a row that can never be edited. But the refusal
    used to leave `run_once` as an unhandled ValueError: with
    [good, '0xabc', good] the pre-refusal code wrote 2 of 3 and skipped 1, and
    the refusal wrote 0 and printed a traceback. Address files are static, so
    that is every day of §3.3's 21-day window, not one address.
    """

    def _provider(self, entries, books, spot, specs):
        source = StaticAddressSource(tuple(entries), "a deliberately broken list")

        class Provider(FakeProvider):
            def __init__(self, *a):
                super().__init__(*a)
                self.market_reads = 0

            def addresses(self):
                return source.addresses()

            def specs(self):
                self.market_reads += 1
                return super().specs()

            def spot(self):
                self.market_reads += 1
                return super().spot()

        return Provider(books, spot, specs)

    def test_a_malformed_entry_is_reported_not_raised(
        self, bundle, specs, spot, books, naive, now
    ):
        entries = [ADDR_A, "0xabc", ADDR_B]
        provider = self._provider(entries, books, spot, specs)
        journal = CalibrationJournal()

        report = ShadowCron(provider, bundle, journal, naive, n_paths=500).run_once(now)

        assert report.refused
        assert report.written == 0 and report.attempted == 0
        # The index is the actionable part: "invalid address" sends an
        # operator hunting through 200 lines by eye.
        assert "addresses[1]" in report.address_source_error
        assert "40 hex digits" in report.address_source_error
        # Not laundered into the per-address channel, which a caller reads as
        # "199 of 200 fine".
        assert report.skipped == []
        assert "REFUSED" in str(report)
        rows = journal._query("SELECT COUNT(*) c FROM calibration_predictions")[0]
        assert rows["c"] == 0
        journal.close()

    def test_the_list_is_read_before_any_market_data_is_fetched(
        self, bundle, specs, spot, books, naive, now
    ):
        """The claim `providers.py` makes: a typo caught at load costs nothing.

        `specs()` and `spot()` are Info requests. Called before the list was
        validated -- which is how `run_once` was ordered -- a one-character
        typo cost 40 weight on a 1-coin universe, around 80 of the 300/minute
        the shadow reserve allows, for a list that was never loadable.
        """
        provider = self._provider([ADDR_A, "0xabc"], books, spot, specs)
        journal = CalibrationJournal()

        report = ShadowCron(provider, bundle, journal, naive, n_paths=500).run_once(now)

        assert report.refused
        assert provider.market_reads == 0
        journal.close()

    def test_a_missing_file_is_the_same_run_level_failure(
        self, bundle, specs, spot, books, naive, now, tmp_path
    ):
        """Not only ValueError. A list that is absent, or truncated mid-write,
        leaves the operator in the same position -- nothing to sweep today --
        and each one used to produce a differently-shaped traceback."""
        from risk_engine.shadow.providers import FileAddressSource

        missing = FileAddressSource(tmp_path / "not-there.json")

        class Provider(FakeProvider):
            def addresses(self):
                return missing.addresses()

        journal = CalibrationJournal()
        report = ShadowCron(
            Provider(books, spot, specs), bundle, journal, naive, n_paths=500
        ).run_once(now)
        assert report.refused
        assert "FileNotFoundError" in report.address_source_error
        journal.close()

    def test_a_good_list_still_reports_nothing_refused(
        self, bundle, specs, spot, books, naive, now
    ):
        """The guard must not turn every sweep into a refusal."""
        provider = self._provider([ADDR_A, ADDR_B], books, spot, specs)
        journal = CalibrationJournal()
        report = ShadowCron(provider, bundle, journal, naive, n_paths=500).run_once(now)
        assert not report.refused and report.address_source_error is None
        assert report.written == 2
        journal.close()

    def test_the_cli_exits_non_zero_when_the_list_is_refused(self, monkeypatch, capsys):
        """The exit code is the whole interface for the cron that runs this.

        Turning the traceback into a report is only half the fix: a daily job
        that prints "0 addresses written" and exits 0 is a §3.3 window that
        stops advancing with nobody told, and the list is static so it stops
        advancing every day after that too. `deploy/docker-compose.yml` runs
        `snapshot ... || echo "snapshot failed"`, which only says anything at
        all because the exit code is non-zero.
        """
        from argparse import Namespace

        from risk_engine.shadow import cli as cli_mod

        class Refusing:
            frame = "a list that will not load"

            def addresses(self):
                raise ValueError("addrs.json: addresses[3]: not an address")

            def specs(self):
                pytest.fail("market data was fetched for a refused list")

            def spot(self):
                pytest.fail("market data was fetched for a refused list")

        # `_live_world` is stubbed because the real one reaches the venue;
        # `bundle` and `naive` are never touched, since `run_once` returns
        # before it needs them, and that is itself part of the claim.
        monkeypatch.setattr(
            cli_mod, "_live_world", lambda args, **kw: (Refusing(), None, None)
        )
        code = cli_mod.cmd_snapshot(
            Namespace(journal=":memory:", fixture=False, addresses="addrs.json",
                      n_paths=10)
        )
        assert code == 2
        out = capsys.readouterr().out
        assert "addresses[3]" in out
        assert "REFUSED" in out


class TestTheLiveCliSpendsNoWeightItNeedNot:
    """Two §5.3 claims `_live_world` makes, both of which were false.

    The README's, that a bad entry is refused "rather than part-way through a
    sweep that has already spent weight": the bundle was built first, so meta
    and a candle snapshot were paid for before the file was opened at all.
    And its own, that the provider reuses the bundle's prices: it seeded them
    without the timestamp `spot()` reads, so they were re-fetched anyway.
    """

    def _args(self, path):
        from argparse import Namespace

        return Namespace(addresses=str(path), fixture=False, journal=":memory:")

    def _stub_bundle(self, monkeypatch):
        """Replace the one call in `_live_world` that reaches the network.

        Everything past this point is Info requests. Raising a sentinel here
        both keeps the test offline and makes "did the refusal happen before
        any weight was spent?" a question with a yes/no answer: SystemExit
        means it did, `ReachedTheBundle` means it did not.
        """
        from risk_engine.service import state as state_mod

        class ReachedTheBundle(Exception):
            pass

        def _reached(*a, **k):
            raise ReachedTheBundle

        monkeypatch.setattr(state_mod, "_build_live_bundle", _reached)
        return ReachedTheBundle

    def _write(self, tmp_path, addresses):
        import json

        path = tmp_path / "addrs.json"
        path.write_text(json.dumps({"frame": "broken", "addresses": addresses}))
        return path

    def test_a_malformed_list_is_refused_before_the_bundle_is_built(
        self, tmp_path, monkeypatch
    ):
        from risk_engine.shadow import cli as cli_mod

        self._stub_bundle(monkeypatch)
        path = self._write(tmp_path, [ADDR_A, "0xabc"])

        with pytest.raises(SystemExit) as exc:
            cli_mod._live_world(self._args(path))
        assert "addresses[1]" in str(exc.value)
        assert "nothing was fetched" in str(exc.value)

    def test_resolve_is_not_blocked_by_a_list_it_never_reads(
        self, tmp_path, monkeypatch
    ):
        """A typo must not strand yesterday's predictions.

        The resolver takes its addresses from the journal's pending rows, and
        the Info API serves current state only -- a resolution that misses its
        window (`DEFAULT_STALE_AFTER_S`) cannot be recovered afterwards. A bad
        list has to cost a day of new predictions, never an observation
        already paid for. So `cmd_resolve` passes `load_addresses=False` and
        gets all the way to the bundle on a list `snapshot` refuses.
        """
        from risk_engine.shadow import cli as cli_mod

        reached = self._stub_bundle(monkeypatch)
        path = self._write(tmp_path, ["0xabc"])

        with pytest.raises(reached):
            cli_mod._live_world(self._args(path), load_addresses=False)

    def test_the_seeded_prices_are_actually_reused(self, tmp_path, monkeypatch):
        """`_live_world` hands the provider the bundle's prices to avoid a
        re-fetch, and used to set only `_spot`.

        `spot()` measures freshness against `_spot_at`, a `time.monotonic()`
        stamp left at its 0.0 default -- so the first call in the sweep saw an
        age of "seconds since the process started", judged prices that were
        seconds old to be stale, and re-fetched a candle snapshot per coin at
        20 weight each. The cache the comment described was never hit once.
        """
        from risk_engine.service import state as state_mod
        from risk_engine.shadow import cli as cli_mod

        calls: list[str] = []

        def _candles(self, coin, interval, start_ms, end_ms):
            calls.append(coin)
            return [{"t": 1, "c": "100000.0"}]

        # The bundle now carries the fitted return series, so `_live_world`
        # reads Baseline A's factor off it instead of fetching.
        stub_bundle = SimpleNamespace(factor_returns={"BTC": np.full(500, 0.001)})
        monkeypatch.setattr(
            state_mod, "_build_live_bundle",
            lambda *a, **k: (stub_bundle, {"BTC": object()}, {"BTC": 100_000.0}),
        )
        monkeypatch.setattr(
            "risk_engine.market.info.InfoClient.candle_snapshot", _candles
        )
        monkeypatch.setattr(
            "risk_engine.market.parse.parse_candles_to_log_returns",
            lambda c: (None, [0.001] * 500),
        )
        monkeypatch.setattr(
            cli_mod, "historical_24h_log_returns", lambda r: np.full(100, 0.01)
        )

        provider, _, _ = cli_mod._live_world(self._args(self._write(tmp_path, [ADDR_A])))
        # ZERO calls now. This used to assert exactly one -- the 90-day BTC
        # history Baseline A is fitted from -- which was the redundant fetch
        # itself: the build had already pulled that series to fit the matrix,
        # and re-requesting it cost ~56 weight, unpaced, on a window the
        # build had just drained.
        assert calls == [], "_live_world fetched candles the bundle already carries"

        assert provider.spot() == {"BTC": 100_000.0}
        assert calls == [], "spot() re-fetched prices it had just been given"


class TestResolverPacesRatherThanDroppingToStale:
    """The resolver had the snapshot's bug, made worse by staleness.

    Each resolution spends 22 weight (book + external_flow); a fresh hourly
    process resolves ~8 before the §5.3 window refuses, then filed the rest as
    failed. Against ~200 due per day inside a 2-hour staleness window, ~15
    landed and ~185 aged out and were dropped from the gate (A-04). §3.3 was
    unreachable from the resolution side, same root cause as the snapshot:
    a rate limit treated as a failure instead of a pace.
    """

    def _seed(self, journal, bundle, specs, spot, books, naive, now):
        ShadowCron(FakeProvider(books, spot, specs), bundle, journal, naive,
                   n_paths=1_000).run_once(now)
        return now + timedelta(hours=24)

    def test_a_rate_limited_resolution_is_waited_through_not_failed(
        self, bundle, specs, spot, books, naive, now
    ):
        journal = CalibrationJournal()
        later = self._seed(journal, bundle, specs, spot, books, naive, now)
        # book() refuses on the first call of the run, then the window refills.
        provider = _BudgetedProvider(books, spot, specs, allow=0, release_after=1)
        report = resolve_due(journal, provider, later,
                             max_resolve_seconds=60.0, budget_wait_seconds=0.0)
        assert report.resolved > 0
        assert not report.budget_exhausted
        # A rate limit must NOT appear as a per-row failure: those rows would
        # be counted against the batch and, on a real run, age into staleness.
        assert not any("RateLimit" in reason for _, reason in report.failed)
        journal.close()

    def test_the_run_stops_at_its_ceiling_rather_than_sleeping_forever(
        self, bundle, specs, spot, books, naive, now
    ):
        journal = CalibrationJournal()
        later = self._seed(journal, bundle, specs, spot, books, naive, now)
        provider = _BudgetedProvider(books, spot, specs, allow=0)  # never releases
        report = resolve_due(journal, provider, later, max_resolve_seconds=0.0)
        assert report.budget_exhausted
        assert report.resolved == 0
        # The un-resolved rows are still due, to be continued next run — not
        # failed, not stale, just deferred.
        assert not any("RateLimit" in reason for _, reason in report.failed)
        assert journal.due(later)  # still pending
        journal.close()

    def test_salvageable_rows_are_served_before_a_hopeless_backlog(
        self, bundle, specs, spot, books, naive, now
    ):
        """A row past the staleness bound cannot be rescued; it must not queue
        ahead of rows that still can be.

        `due()` orders oldest-first, which is right until a backlog exists and
        then inverts into a trap: rows already past `stale_after_s` are
        resolved first (each costing a book fetch, a flow fetch, and its turn),
        every one of them lands stale, and the rows still INSIDE the bound age
        out while they wait. Observed live 2026-08-05: one run resolved 376
        rows, all 376 stale, while that day's own predictions queued behind
        yesterday's corpses. 1919 predictions, zero usable observations — a
        backlog the drain order itself regenerates every day.

        Two sweeps a day apart, resolved when day 1's rows are far past the
        bound and day 2's are fresh, under a ceiling that stops the run after
        the salvageable half. Under the old order the ceiling is spent on the
        corpses and the salvageable rows get NOTHING; under the new one they
        are all scored non-stale and the corpses defer to the next run.
        """
        journal = CalibrationJournal()
        provider = FakeProvider(books, spot, specs)
        cron = ShadowCron(provider, bundle, journal, naive, n_paths=1_000)
        cron.run_once(now)                                # day 1
        cron.run_once(now + timedelta(hours=24))          # day 2

        # Day 1 resolves at now+24h: 25h late at the observation instant --
        # hopeless. Day 2 resolves at now+48h: 1h late -- salvageable.
        observe_at = now + timedelta(hours=49)
        assert len(journal.due(observe_at)) == 12, "both sweeps' rows must be due"

        # The ceiling must cut the run partway so the ORDER decides who gets
        # served. Against an instant fake a wall-clock ceiling cannot, so the
        # provider burns real time on the two fetches a day's rows genuinely
        # need. The book cache is keyed on ADDRESS alone and both days share
        # the two fixture addresses, so `book()` fires twice; the flow cache is
        # keyed on (address, window), so `external_flow` fires once per
        # address per day -- four times. At 0.2s each that is 2*0.2 + 4*0.2 =
        # 1.2s of forced work against a 0.7s ceiling: one day's rows fit
        # (0.2*2 books + 0.2*2 flows = 0.8s, whose last fetch passes the
        # deadline check at 0.6s), the other day's first uncached fetch meets
        # the deadline and defers. Which day got served is then purely the
        # order under test.
        class SlowProvider(FakeProvider):
            def book(self, address):
                time.sleep(0.2)
                return super().book(address)

            def external_flow(self, address, since, until):
                time.sleep(0.2)
                return super().external_flow(address, since, until)

        slow = SlowProvider(books, spot, specs)
        report = resolve_due(journal, slow, observe_at,
                             max_resolve_seconds=0.7)
        assert 0 < report.resolved < 12, (
            f"resolved {report.resolved}; the ceiling was meant to cut the run "
            f"partway so the order decides who gets served -- retune the pause"
        )

        model_rows = journal.scored(DISTRIBUTION_VERSION, VARIANT_MODEL)
        fresh = [r for r in model_rows if not r["stale_resolution"]]
        day2 = (now + timedelta(hours=24)).date().isoformat()
        assert fresh, (
            "the run had budget for several rows and every salvageable row was "
            "due; under the fixed order at least some of them must be scored "
            "non-stale before any ceiling is spent on the hopeless backlog"
        )
        assert all(r["observation_day"] == day2 for r in fresh)
        # And nothing is lost: whatever the ceiling deferred is still due.
        assert len(journal.due(observe_at)) == 12 - report.resolved
        journal.close()


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
            ADDR_A: Book(ADDR_A, books[ADDR_A].cross_collateral + deposit,
                         books[ADDR_A].positions, now)
        }
        provider = FakeProvider(books, spot, specs, later_books=after,
                                flows={ADDR_A: deposit})
        ShadowCron(provider, bundle, journal, naive, n_paths=1_000).run_once(now)
        later = now + timedelta(hours=24)
        provider.phase = "after"
        resolve_due(journal, provider, later)

        rows = {r["address"]: r for r in journal.scored(DISTRIBUTION_VERSION, VARIANT_MODEL)}
        assert rows[ADDR_A]["external_flow_usd"] == deposit
        assert rows[ADDR_A]["actual_equity"] == pytest.approx(
            rows[ADDR_A]["actual_equity_change"] + deposit + 100_000.0
        )
        # Prices did not move in the fake, so all that is left is ~zero.
        assert abs(rows[ADDR_A]["actual_equity_change"]) < 1.0
        assert load_cohort(journal, DISTRIBUTION_VERSION, VARIANT_MODEL, COHORT_ALL).n == 2
        journal.close()

    def test_a_restructured_book_is_flagged_not_silently_scored(
        self, bundle, specs, spot, books, naive, now
    ):
        after = {
            ADDR_A: Book(ADDR_A, 100_000.0,
                         (Position("BTC", 1.0, 100_000.0, MarginMode.CROSS, 20.0),), now)
        }
        provider = FakeProvider(books, spot, specs, later_books=after)
        journal = CalibrationJournal()
        ShadowCron(provider, bundle, journal, naive, n_paths=1_000).run_once(now)
        provider.phase = "after"
        resolve_due(journal, provider, now + timedelta(hours=24))

        rows = {r["address"]: r for r in journal.scored(DISTRIBUTION_VERSION, VARIANT_MODEL)}
        assert rows[ADDR_A]["book_changed"]
        assert not rows[ADDR_B]["book_changed"]
        unchanged = load_cohort(
            journal, DISTRIBUTION_VERSION, VARIANT_MODEL, COHORT_BOOK_UNCHANGED
        )
        assert unchanged.n == 1
        journal.close()

    def test_a_row_that_can_never_resolve_says_so_on_every_run(self, spot, now):
        """The resolver retries without bound, and some rows are forever.

        A prediction whose stored `address` is not an address -- written
        before `record_prediction` canonicalised, and left visible because the
        read paths deliberately do not normalise -- fails before any request
        is made, so it is handed back by `due()` on every future run:
        measured resolved=0, failed=1, still_due=1 on three consecutive runs,
        with the failed list never emptying. That is better than the old
        behaviour, which resolved it once as a liquidation, and it is
        documented rather than bounded (see `resolve.py`). What it must not be
        is invisible: `permanent` names it as a row that will fail identically
        next time, so a forever-failure cannot hide among transient ones.
        """
        journal = CalibrationJournal()
        _insert_legacy_prediction(journal, "0xabc", now)

        class Provider:
            """As `LiveSnapshotProvider` behaves: normalise, then fetch."""

            def book(self, address):
                from risk_engine.domain.types import normalise_address

                normalise_address(address)
                raise AssertionError("a malformed address must not reach a fetch")

            def spot(self):
                return spot

            def external_flow(self, address, since, until):
                return 0.0

        later = now + timedelta(hours=24, minutes=1)
        for _ in range(3):
            report = resolve_due(journal, Provider(), later)
            assert report.resolved == 0
            assert len(report.failed) == 1
            assert len(report.permanent) == 1
            assert "not an account address" in report.permanent[0][1]
            assert "permanently" in str(report)
            # Still due, still not silently dropped, still not resolved with a
            # fabricated outcome -- the row is intact for a deliberate fix.
            assert len(journal.due(later)) == 1
        journal.close()

    def test_a_transient_failure_is_not_called_permanent(self, spot, now):
        """The classification must stay narrow.

        A timeout or a 5xx on a perfectly good address is exactly what the
        unbounded retry exists for, and labelling it permanent would tell an
        operator to go and edit a journal row that is fine.
        """
        journal = CalibrationJournal()
        _insert_legacy_prediction(journal, ADDR_A, now)

        class Flaky:
            def book(self, address):
                raise TimeoutError("the venue took too long")

            def spot(self):
                return spot

            def external_flow(self, address, since, until):
                return 0.0

        report = resolve_due(journal, Flaky(), now + timedelta(hours=24, minutes=1))
        assert len(report.failed) == 1
        assert report.permanent == []
        assert "permanently" not in str(report)
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
    def _fill(journal, n_days=30, per_day=15, seed=0, tie_baselines=False):
        """`tie_baselines` gives all three variants the SAME distribution.

        Every observation then scores identically under all three, so the CRPS
        means coincide exactly and the "beats baseline" comparison sits on its
        boundary -- which is where `<` and `<=` differ and where nothing had
        ever put it.
        """
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
                other = truth if tie_baselines else wrong
                for variant, dist in (
                    (VARIANT_MODEL, truth),
                    (VARIANT_BASELINE_A, other),
                    (VARIANT_BASELINE_B, other),
                ):
                    pid = journal.record_prediction(
                        address=addr(i), variant=variant, predicted_at=when,
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

    def test_a_day_recorded_under_the_tail_defect_is_not_a_gate_day(self):
        """§2.3's recording mode, enforced rather than announced.

        The shadow path builds with `serving=False`, so a §2.3 violation no
        longer stops the sweep — it records instead, and prints "these
        observations are DIAGNOSTIC EVIDENCE, not §3.3 gate-days". That print
        was the entire defence. `_print_defect_note`'s own docstring states the
        standard it was failing: a journal of observations collected under a
        known model defect, indistinguishable from a clean one, "is worse than
        no journal — it would be read as gate progress". Nothing in the schema
        could tell them apart, so `progress()` counted them, and the print
        lives in a container's scrollback while the gate is read weeks later
        from the database.

        What makes it serious rather than untidy: `gate_open` is what Phase 4
        — real money — is gated on, and the remedy for the defect is a copula
        change that bumps MODEL_VERSION and resets the counter anyway. So every
        day counted here is a day that cannot survive the fix it is waiting
        for.
        """
        journal = CalibrationJournal()
        try:
            dist = PredictiveDistribution.from_samples(
                np.random.default_rng(3).normal(0, 1000, 20_000)
            )
            start = datetime(2026, 1, 1, tzinfo=timezone.utc)
            for day in range(3):
                when = start + timedelta(days=day)
                for i in range(2):
                    for under_defect in (True, False):
                        pid = journal.record_prediction(
                            address=addr(i + (100 if under_defect else 0)),
                            variant=VARIANT_MODEL, predicted_at=when,
                            horizon_hours=24, model_version="v",
                            distribution_version="test", seed=1, n_paths=100,
                            converged=True, start_equity=100_000.0,
                            p_liq=0.01, p_liq_ci=(0.005, 0.02),
                            var_95=-dist.quantile(0.05), cvar_95=dist.cvar(0.95),
                            distribution=dist, book_snapshot={"fingerprint": "x"},
                            recorded_under_defect=under_defect,
                        )
                        journal.record_outcome(
                            prediction_id=pid, resolved_at=when + timedelta(days=1),
                            actual_equity=100_000.0, actual_equity_change=0.0,
                            external_flow_usd=0.0, book_changed=False,
                            liquidated=False, pit=0.5, pit_u=0.5,
                            crps=1.0, var_95_breached=False,
                            observation_day=when.date(),
                            resolution_lag_s=0.0, stale_resolution=False,
                        )

            progress = journal.progress("test")
            assert progress.resolved_observations == 6, (
                "the clean half of the journal must still count; this test is "
                "about telling them apart, not about discarding everything"
            )
            assert progress.distinct_addresses == 2, (
                "the two defect-stamped addresses must not appear in the gate's "
                "address count"
            )
        finally:
            journal.close()

    def test_a_journal_written_before_the_stamp_existed_fails_closed(self, tmp_path):
        """`schema.sql` is all CREATE TABLE IF NOT EXISTS, so a new column
        reaches a fresh journal and never reaches the deployed one — which,
        with a persistent volume, is the only journal that matters.

        The direction of the backfill is the substance. Rows written before
        the column existed have UNKNOWN provenance, and this gate is what
        Phase 4 (real money) is read off, so §10 resolves the uncertainty
        toward not counting them. It is also the better guess on the facts:
        the deployment carrying such rows logged "RECORDING UNDER A KNOWN §2.3
        DEFECT" on every sweep, so marking them clean would not be neutral.
        """
        import re
        import sqlite3

        from risk_engine.shadow.backends import sqlite_ddl

        old_ddl = re.sub(r"\s*recorded_under_defect[^,]*,", "", sqlite_ddl())
        assert "recorded_under_defect" not in old_ddl, (
            "the stripper missed; this test would then be exercising today's "
            "schema and asserting nothing about migration"
        )
        path = tmp_path / "pre-migration.db"
        con = sqlite3.connect(path)
        con.executescript(old_ddl)
        con.execute(
            """INSERT INTO calibration_predictions
               (address, variant, predicted_at, horizon_hours, resolves_at,
                model_version, distribution_version, seed, n_paths, converged,
                start_equity, p_liq, p_liq_ci_low, p_liq_ci_high, var_95,
                cvar_95, quantile_values, n_quantile_levels, book_snapshot)
               VALUES (?, 'model', '2026-08-01T00:00:00+00:00', 24,
                       '2026-08-02T00:00:00+00:00', 'v', 'test', 1, 100, 1,
                       100000.0, 0.01, 0.0, 0.1, 1.0, 2.0, '[]', 0, '{}')""",
            (addr(1),),
        )
        con.commit()
        con.close()

        with CalibrationJournal(str(path)) as journal:
            rows = journal.backend.rows(journal.backend.execute(
                "SELECT recorded_under_defect FROM calibration_predictions"))
            assert [bool(r["recorded_under_defect"]) for r in rows] == [True], (
                "a row whose provenance was never captured must not be counted "
                "as a gate-day"
            )
            # And the column's DEFAULT stays FALSE, so a clean write after the
            # migration is a gate-day.
            dist = PredictiveDistribution.from_samples(
                np.random.default_rng(1).normal(0, 1, 500))
            journal.record_prediction(
                address=addr(2), variant=VARIANT_MODEL,
                predicted_at=datetime(2026, 8, 4, tzinfo=timezone.utc),
                horizon_hours=24, model_version="v", distribution_version="test",
                seed=1, n_paths=100, converged=True, start_equity=1.0,
                p_liq=0.0, p_liq_ci=(0.0, 0.0), var_95=0.0, cvar_95=0.0,
                distribution=dist, book_snapshot={},
            )
            rows = journal.backend.rows(journal.backend.execute(
                "SELECT recorded_under_defect FROM calibration_predictions "
                "ORDER BY id"))
            assert [bool(r["recorded_under_defect"]) for r in rows] == [True, False]

        # Idempotent: the shadow jobs open the journal together every day.
        CalibrationJournal(str(path)).close()

    def test_the_defect_stamp_is_the_predicate_the_serving_path_uses(self):
        """One predicate, which is what `understates_lower_tail` promises.

        A second definition of "the copula understates the lower tail" would
        let the served numbers and the journal's provenance disagree about
        whether a given bundle was defective — and the disagreement would be
        invisible, because each side looks self-consistent.
        """
        from risk_engine.service.state import understates_lower_tail
        from risk_engine.shadow.cron import _bundle_understates_lower_tail

        class _D:
            def __init__(self, bad):
                self._bad = bad

            def understates_lower_tail(self):
                return self._bad

        class _Bundle:
            def __init__(self, diagnostics):
                self.tail_diagnostics = diagnostics

        assert _bundle_understates_lower_tail(_Bundle((_D(True),))) is True
        assert _bundle_understates_lower_tail(_Bundle((_D(False), _D(True)))) is True
        assert _bundle_understates_lower_tail(_Bundle((_D(False),))) is False
        assert understates_lower_tail((_D(False), _D(True))) is True

        # A hand-assembled bundle carries no diagnostics, and that is not a
        # defect: there is no fitted copula to have failed the gate.
        assert _bundle_understates_lower_tail(_Bundle(())) is False

        class _NoAttr:
            pass

        assert _bundle_understates_lower_tail(_NoAttr()) is False

    def test_the_census_can_be_read_back_at_all(self):
        """`record_sweep` has written this table since 2026-08-01 and nothing
        read it, so B6's decision procedure -- stated entirely in terms of it
        -- could not be run. A write nobody reads is not provenance, it is
        storage."""
        journal = CalibrationJournal()
        try:
            when = datetime(2026, 8, 4, 9, 0, tzinfo=timezone.utc)
            journal.record_sweep(
                swept_at=when, distribution_version="test",
                attempted=515, written=216, budget_exhausted=False,
                skipped_by_reason={"off-universe: ATOM, HYPE": 10,
                                   "no open positions": 103},
            )
            rows = journal.sweeps("test")
            assert len(rows) == 1
            row = rows[0]
            assert row["attempted"] == 515 and row["written"] == 216
            assert row["budget_exhausted"] is False
            assert row["observation_day"] == "2026-08-04"
            # A dict on both backends: Postgres hands back JSONB, SQLite text.
            assert row["skipped_by_reason"]["off-universe: ATOM, HYPE"] == 10
            assert journal.sweeps("some-other-version") == []

            # The DEFAULT read pools every version, each row labelled. The
            # first reader filtered on the CURRENT version, and the day after
            # it landed the version bumped twice without the universe moving
            # -- `shadow census` then reported "no sweeps recorded" over a
            # table holding exactly the full sweeps B6 step 1 asks for. The
            # drop pattern is a fact about the address list and HL_UNIVERSE,
            # not about the copula.
            journal.record_sweep(
                swept_at=when, distribution_version="test-2",
                attempted=515, written=222, budget_exhausted=False,
                skipped_by_reason={"off-universe: HYPE": 5},
            )
            pooled = journal.sweeps()
            assert {r["distribution_version"] for r in pooled} == {"test", "test-2"}
            assert len(pooled) == 2
        finally:
            journal.close()

    def test_widening_is_counted_by_whole_sets_not_by_coin(self):
        """The measurement defect B6 names in its own text.

        A per-coin tally answers "how often is this coin the first one
        missing", never "how many addresses would a universe containing it
        recover". Ten addresses holding ATOM *and* HYPE are recovered by
        neither coin alone, so a per-coin reading promises ten recoveries that
        adding ATOM delivers none of -- and that promise is what a universe
        decision, which costs a MODEL_VERSION bump and the whole accumulated
        window, would be taken on.
        """
        from risk_engine.shadow.metrics import (
            off_universe_demand,
            universe_candidates,
        )

        sweeps = [{"skipped_by_reason": {
            "off-universe: ATOM, HYPE": 10,
            "off-universe: DOGE": 4,
            "no open positions": 99,
        }}]
        demand = off_universe_demand(sweeps)

        assert demand == {frozenset({"ATOM", "HYPE"}): 10,
                          frozenset({"DOGE"}): 4}, (
            "a flat book is not an off-universe drop and widening cannot "
            "recover it, so it must not inflate the demand"
        )

        by_size = {c.coins_added: c for c in universe_candidates(demand)}
        assert by_size[1].recovered == 4, "DOGE completes a set on its own"
        assert by_size[2].recovered == 4, (
            "adding one of ATOM/HYPE completes nothing; a per-coin tally would "
            "claim 10 here"
        )
        assert by_size[3].recovered == 14 and by_size[3].still_dropped == 0

    def test_a_coin_that_never_completes_a_set_alone_is_still_reachable(self):
        """Every remaining set needs two or more additions.

        The greedy step that picks "the coin completing the most sets" has no
        candidate here, and a loop that stopped there would report the demand
        as permanently unrecoverable when two coins clear all of it.
        """
        from risk_engine.shadow.metrics import (
            off_universe_demand,
            universe_candidates,
        )

        demand = off_universe_demand([{"skipped_by_reason": {
            "off-universe: A, B": 5,
            "off-universe: A, C": 3,
        }}])
        candidates = universe_candidates(demand)
        assert candidates, "the search gave up while the demand was reachable"
        assert candidates[-1].still_dropped == 0
        assert candidates[-1].recovered == 8

    def test_each_cohort_keeps_the_rows_its_name_claims(self):
        """The three cohort filters, none of which had a test.

        `no_external_flow` was never selected by any test in this suite, so
        inverting its predicate -- keeping exactly the rows with flow -- left
        the whole suite green. That cohort is offered on the CLI
        (`shadow icc --cohort`) and is one of the three §3.1 reports, and a
        silently inverted filter publishes a calibration score computed on the
        accounts the cohort exists to exclude, under the cohort's own name.

        Asserted as counts over a journal built with known contamination, so
        the test says which rows each cohort keeps rather than that it keeps
        some.
        """
        from risk_engine.shadow.metrics import COHORT_NO_FLOW, in_cohort

        rows = [
            {"external_flow_usd": 0.0, "book_changed": False},    # clean
            {"external_flow_usd": 250.0, "book_changed": False},  # deposited
            {"external_flow_usd": 0.0, "book_changed": True},     # traded
            {"external_flow_usd": -80.0, "book_changed": True},   # both
        ]
        kept = {c: [i for i, r in enumerate(rows) if in_cohort(r, c)]
                for c in (COHORT_ALL, COHORT_NO_FLOW, COHORT_BOOK_UNCHANGED)}

        assert kept[COHORT_ALL] == [0, 1, 2, 3]
        assert kept[COHORT_NO_FLOW] == [0, 2], (
            "no_external_flow keeps the rows with NO flow; a withdrawal is "
            "flow just as much as a deposit is"
        )
        assert kept[COHORT_BOOK_UNCHANGED] == [0, 1]

    def test_a_mistyped_cohort_is_refused_rather_than_silently_meaning_all(self):
        """It used to mean `all`, labelled with the typo.

        `load_cohort` filtered on equality against the two known names and had
        no else, so any other string fell through every filter and returned
        every row -- a §3.1 report over the whole population, named after a
        cohort that was never applied. `champion.py` raised on the same input,
        so the two halves of the §0.2 decision disagreed about what a cohort
        name even is. Both now refuse, and both refuse BEFORE reading a row, so
        an empty journal (day one, the expected state) fails the same way a
        full one does rather than returning an empty list.
        """
        from risk_engine.shadow.champion import _cohort_rows
        from risk_engine.shadow.metrics import COHORTS

        journal = CalibrationJournal()
        try:
            for cohort in ("book_unchagned", "", "ALL"):
                assert cohort not in COHORTS
                with pytest.raises(ValueError, match="unknown cohort"):
                    load_cohort(journal, "test", VARIANT_MODEL, cohort)
                with pytest.raises(ValueError, match="unknown cohort"):
                    _cohort_rows(journal, "test", cohort)
        finally:
            journal.close()

    def test_a_tie_on_crps_does_not_count_as_beating_the_baseline(self):
        """§0.2 asks for an improvement, and equal is not better.

        Every other assertion in this class sits far from the boundary -- the
        model is the true distribution and wins comfortably, or is deliberately
        wrong and loses -- so `<` could be `<=` and nothing would notice. A tie
        is not hypothetical: a challenger that changes nothing observable for
        these books scores exactly what the baseline scores, and `<=` would
        report it as having beaten a baseline it merely matched, on the check
        §0.2 reads to decide a migration.
        """
        from risk_engine.shadow.journal import VARIANT_BASELINE_A, VARIANT_BASELINE_B

        journal = CalibrationJournal()
        try:
            self._fill(journal, n_days=6, per_day=4, tie_baselines=True)
            report = calibration_report(journal, "test", COHORT_ALL)

            assert report.mean_crps[VARIANT_MODEL] == pytest.approx(
                report.mean_crps[VARIANT_BASELINE_A], rel=1e-12
            ), "the fixture did not actually tie, so this is not on the boundary"
            assert report.mean_crps[VARIANT_MODEL] == pytest.approx(
                report.mean_crps[VARIANT_BASELINE_B], rel=1e-12
            )

            assert not report.crps_beats_baseline_a
            assert not report.crps_beats_baseline_b
            assert "FAIL  CRPS beats baseline A" in report.gate_summary
            assert "FAIL  CRPS beats baseline B" in report.gate_summary
        finally:
            journal.close()

    def test_the_gates_own_pass_flags_are_read_off_the_clustered_interval(self):
        """The four §3.1 checks, at the boundaries that decide them.

        `passes_clustered` is what `gate_summary` prints for the VaR@95 row --
        `passes_naive` is computed and read by nothing -- and neither had an
        assertion anywhere: swapping the lower bound of the clustered interval
        for its upper one, so the check becomes `target == upper`, left the
        suite green. Same for the KS row's `p > 0.05`. These are the flags a
        reader of the published score uses to decide whether Phase 4 opens, so
        they are pinned here on both sides of each boundary rather than
        inferred from a well-behaved fixture that passes everything.
        """
        from risk_engine.shadow.metrics import TailCalibration

        inside = TailCalibration(breach_rate=0.05, naive_ci=(0.01, 0.09),
                                 clustered_ci=(0.02, 0.08), target=0.05, n=100, n_days=5)
        assert inside.passes_clustered is True
        assert inside.passes_naive is True

        # Target below the clustered interval, and above it. Both must fail,
        # which is what pins the interval as an interval rather than a bound.
        below = TailCalibration(breach_rate=0.20, naive_ci=(0.15, 0.25),
                                clustered_ci=(0.12, 0.30), target=0.05, n=100, n_days=5)
        above = TailCalibration(breach_rate=0.001, naive_ci=(0.0, 0.01),
                                clustered_ci=(0.0, 0.02), target=0.05, n=100, n_days=5)
        assert below.passes_clustered is False
        assert above.passes_clustered is False

        # The bounds are inclusive: an interval that just touches the target
        # contains it.
        touching = TailCalibration(breach_rate=0.05, naive_ci=(0.05, 0.09),
                                   clustered_ci=(0.05, 0.08), target=0.05, n=100, n_days=5)
        assert touching.passes_clustered is True

        # No clustered interval at all is None, not False -- "not computed" and
        # "computed and failed" are different facts about the gate.
        assert TailCalibration(breach_rate=0.05, naive_ci=(0.0, 0.1),
                               clustered_ci=None, target=0.05, n=3,
                               n_days=1).passes_clustered is None

    def test_the_KS_row_flips_on_the_five_per_cent_it_names(self):
        """`PIT uniform (KS p>0.05)` says 0.05 in its own label, so 0.05 itself
        must fail. Nothing asserted where it flips."""
        from risk_engine.shadow.journal import VARIANT_BASELINE_A, VARIANT_BASELINE_B
        from risk_engine.shadow.metrics import CalibrationReport, TailCalibration

        tail = TailCalibration(breach_rate=0.05, naive_ci=(0.0, 0.1),
                               clustered_ci=(0.0, 0.2), target=0.05, n=10, n_days=3)

        def summary_for(p):
            return CalibrationReport(
                distribution_version="test", cohort=COHORT_ALL, n=10, n_days=3,
                ks_statistic=0.1, ks_pvalue=p,
                mean_crps={VARIANT_MODEL: 1.0, VARIANT_BASELINE_A: 2.0,
                           VARIANT_BASELINE_B: 2.0},
                crps_beats_baseline_a=True, crps_beats_baseline_b=True,
                tail=tail, n_paired=10,
            ).gate_summary

        assert "PASS  PIT uniform" in summary_for(0.0500001)
        assert "FAIL  PIT uniform" in summary_for(0.05)
        assert "FAIL  PIT uniform" in summary_for(0.0499999)

    def test_one_day_of_observations_gets_no_clustered_interval(self):
        """The day-clustered bootstrap needs at least two clusters.

        `n_days >= 2` had no test at its boundary, and the two sides mean
        opposite things for the gate: below it `passes_clustered` is None and
        the VaR row cannot pass at all, at or above it the interval exists and
        the row is decided. Getting the threshold wrong by one silently
        withholds the gate's only cluster-robust check on the first day it
        could have been computed.
        """
        keys = np.array(["k0", "k1", "k2", "k3"], dtype=object)
        one_day = Cohort(
            name=COHORT_ALL,
            pit=np.full(4, 0.5), crps=np.ones(4),
            breached=np.array([True, False, False, False]),
            days=np.array(["2026-08-01"] * 4), keys=keys,
        )
        two_days = Cohort(
            name=COHORT_ALL,
            pit=np.full(4, 0.5), crps=np.ones(4),
            breached=np.array([True, False, False, False]),
            days=np.array(["2026-08-01", "2026-08-01", "2026-08-02", "2026-08-02"]),
            keys=keys,
        )

        assert one_day.n_days == 1
        assert tail_calibration(one_day).clustered_ci is None
        assert tail_calibration(one_day).passes_clustered is None

        assert two_days.n_days == 2
        assert tail_calibration(two_days).clustered_ci is not None, (
            "two distinct days is enough clusters to bootstrap over; withholding "
            "the interval there delays the gate's only cluster-robust check"
        )

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
        pred = naive.predict(books[ADDR_B], spot, specs, n_draws=5_000, seed=1)
        assert pred.start_equity > 0
        assert 0.0 <= pred.p_liq <= 1.0
        assert pred.equity_change.quantile(0.99) > pred.equity_change.quantile(0.01)

    def test_baseline_b_is_the_same_engine_with_dependence_off(
        self, bundle, specs, spot, books, now
    ):
        from risk_engine.validation.baselines import run_baseline_b

        out = run_baseline_b(bundle, specs, books[ADDR_B], spot, 24,
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
            address=ADDR_A, variant=VARIANT_MODEL, predicted_at=now, horizon_hours=24,
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


class TestBaselineAComesOffTheBundle:
    """The factor series for §3.2's Baseline A must not cost a second fetch.

    `_live_world` re-requested 90 days of BTC candles right after the paced
    build had fetched exactly that series to fit the matrix — ~56 weight
    under the published table (20 base plus the per-item surcharge on 2160
    candles) for data already in memory. Worse, the call sat outside the
    pacing, so it ran against a window the build had just drained: a live
    run died with `weight 20 exceeds remaining 0 (323/300 spent)` one line
    after successfully waiting its turn.
    """

    def test_both_builders_carry_the_series(self):
        """Live and fixture alike, so the shadow path behaves the same under
        `--fixture` as it does against the venue."""
        from risk_engine.service.state import _build_fixture_bundle

        bundle, _, _ = _build_fixture_bundle()
        assert "BTC" in bundle.factor_returns
        assert bundle.factor_returns["BTC"].size > 100

    def test_a_bundle_without_btc_is_refused_rather_than_silently_wrong(self):
        """BTC is mandatory as a risk factor (§2.1) and `_build_live_bundle`
        refuses without it, so an absent series means the bundle came from
        somewhere unexpected. Better to say so than to build Baseline A from
        whatever happens to be first."""
        from risk_engine.shadow import cli

        src = inspect.getsource(cli._live_world)
        assert 'factor_returns.get("BTC")' in src
        assert "raise SystemExit" in src

    def test_the_naive_baseline_is_built_from_it(self):
        """Guard on the guard: the series being present proves nothing if
        the baseline is fitted from something else."""
        from risk_engine.shadow import cli

        src = inspect.getsource(cli._live_world)
        assert "NaiveBaseline(" in src
        i, j = src.index("hourly = bundle.factor_returns"), src.index("NaiveBaseline(")
        assert i < j, "the series must be read before the baseline is built from it"


class TestOffUniverseSkipsNameEveryCoin:
    """B6's census has to answer the question B6 actually asks.

    An off-universe holding used to surface as a bare `KeyError: 'ATOM'`
    raised inside `equity()` and caught by the sweep's per-address handler.
    The census stores skip reasons verbatim, so B6's decision procedure —
    "sum the `KeyError: '<COIN>'` counts by coin, most-dropped first" — was
    computing something else: how often a coin is the FIRST one missing, not
    how many addresses a universe containing it would recover.

    The two differ in the direction that matters. An address holding ATOM and
    HYPE is filed under one of them; adding that one coin recovers the
    address only if the other is also added. Widening to the top of that
    tally can therefore recover nothing.
    """

    def test_all_missing_coins_are_named_not_just_the_first(self, bundle, specs,
                                                            spot, naive, now):
        book = Book("0x" + "e" * 40, 100_000.0, (
            Position("BTC", 1.0, 100_000.0, MarginMode.CROSS, 10.0),
            Position("ATOM", 500.0, 8.0, MarginMode.CROSS, 10.0),
            Position("HYPE", 200.0, 30.0, MarginMode.CROSS, 10.0),
        ), now)
        provider = FakeProvider({book.address: book}, spot, specs)
        journal = CalibrationJournal()
        report = ShadowCron(provider, bundle, journal, naive,
                            n_paths=500).run_once(now)
        journal.close()

        assert report.written == 0
        (_, reason), = report.skipped
        assert "ATOM" in reason and "HYPE" in reason, reason
        # Sorted, so the census groups identical holdings under one string
        # instead of two orderings of the same set.
        assert reason == "off-universe: ATOM, HYPE"

    def test_a_flat_account_is_still_a_different_reason(self, bundle, specs,
                                                        spot, naive, now):
        """The two must stay distinguishable. B6 says an all-off-universe book
        reads as "no open positions", byte-identical to a flat account — that
        is not what the code does, and the distinction is what makes the
        census informative: flat accounts are not recoverable by widening the
        universe and off-universe ones are."""
        flat = Book("0x" + "f" * 40, 100_000.0, (), now)
        provider = FakeProvider({flat.address: flat}, spot, specs)
        journal = CalibrationJournal()
        report = ShadowCron(provider, bundle, journal, naive,
                            n_paths=500).run_once(now)
        journal.close()
        (_, reason), = report.skipped
        assert reason == "no open positions"

    def test_an_in_universe_book_is_untouched(self, bundle, specs, spot, books,
                                              naive, now):
        """Guard on the guard: a check that skipped everything would satisfy
        the assertions above."""
        provider = FakeProvider(books, spot, specs)
        journal = CalibrationJournal()
        report = ShadowCron(provider, bundle, journal, naive,
                            n_paths=500).run_once(now)
        journal.close()
        assert report.written == len(books)
        assert not report.skipped


class TestTheResolverSaysItIsAlive:
    """The sweep's silence, one job over — and it cost more here.

    `resolve_due` logged nothing at all. A 265-address run is roughly twenty
    minutes, most of it asleep on the §5.3 window by design, so a working
    resolver and a wedged one are outwardly identical. Observed 2026-08-04:
    the container built its bundle, printed one line, and went quiet, while
    1481 predictions sat pending — some already two days past due.

    Silence costs more on this side than on the snapshot side. A silent sweep
    loses a day's predictions; a silent resolver loses them AFTER they were
    paid for, because an outcome collected past `stale_after_s` is flagged
    and dropped from the gate rather than scored.
    """

    def test_it_announces_the_queue_and_the_stale_bound(
        self, bundle, specs, spot, books, naive, now, caplog
    ):
        import logging as _logging

        journal = CalibrationJournal()
        ShadowCron(FakeProvider(books, spot, specs), bundle, journal, naive,
                   n_paths=500).run_once(now)
        later = now + timedelta(hours=24)
        with caplog.at_level(_logging.INFO, logger="risk_engine.shadow.resolve"):
            resolve_due(journal, FakeProvider(books, spot, specs), later)
        journal.close()

        lines = [r.getMessage() for r in caplog.records]
        assert any("resolving" in m and "pending rows" in m for m in lines), lines
        # The stale bound is named up front, because it is the thing that
        # turns a slow run into lost observations rather than late ones.
        assert any("stale" in m for m in lines), lines

    def test_the_closing_line_is_forced(
        self, bundle, specs, spot, books, naive, now, caplog
    ):
        """A run short enough to finish inside one progress window would
        otherwise announce its queue and never report what it did with it."""
        import logging as _logging

        journal = CalibrationJournal()
        ShadowCron(FakeProvider(books, spot, specs), bundle, journal, naive,
                   n_paths=500).run_once(now)
        later = now + timedelta(hours=24)
        with caplog.at_level(_logging.INFO, logger="risk_engine.shadow.resolve"):
            resolve_due(journal, FakeProvider(books, spot, specs), later)
        journal.close()

        lines = [r.getMessage() for r in caplog.records]
        final = [m for m in lines if "elapsed" in m]
        assert final, lines
        assert "waiting for the §5.3 window" in final[-1]


def test_the_shadow_jobs_do_not_inherit_a_healthcheck_they_cannot_pass():
    """The engine image's HEALTHCHECK polls its own :8787/health. The shadow
    jobs are cron loops that serve no HTTP, so the inherited check can never
    pass and both containers read `unhealthy` while running correctly.

    That is not cosmetic: `docker compose ps` is where an operator looks
    first, and a column that lies in the safe-looking direction on every
    healthy container teaches them to ignore it — so the one time it means
    something, it reads the same. Observed 2026-08-04, both jobs `unhealthy`
    with 1481 predictions freshly written.
    """
    import pathlib as _p

    text = (_p.Path(__file__).resolve().parents[2]
            / "deploy/docker-compose.yml").read_text(encoding="utf-8")
    # Both shadow services, and only them, disable it.
    assert text.count("disable: true") == 2, (
        "expected exactly the two shadow jobs to disable the inherited check"
    )
    for svc in ("shadow-snapshot", "shadow-resolve"):
        block = text.split(f"  {svc}:", 1)[1].split("\n  shadow-", 1)[0]
        assert "disable: true" in block, svc


class _WindowedProvider(FakeProvider):
    """A provider whose `external_flow` HONOURS the window it is asked for.

    `FakeProvider` returns the same figure for any window, which is exactly
    why the defect below could not surface in this suite: every mutation of
    the flow window's bounds survived every test. Any future test about
    WHICH flows are corrected has to use this one.
    """

    def __init__(self, books, spot, specs, later_books, flow_at, amount):
        super().__init__(books, spot, specs, later_books=later_books)
        self._flow_at, self._amount = flow_at, amount
        self.windows: list[tuple[datetime, datetime]] = []

    def external_flow(self, address, since, until):
        self.windows.append((since, until))
        return self._amount if since <= self._flow_at <= until else 0.0


class TestTheFlowWindowCoversWhatTheEquityIncludes:
    """`change = actual_equity - start_equity - flow`, and the two ends of
    that subtraction have to describe the same interval.

    `actual_equity` is the book read at `observed_at`. Under pacing that
    trails `resolves_at` by minutes to tens of minutes — legitimately, and by
    up to `stale_after_s` (2h) before it is even flagged. A flow window
    ending at `resolves_at` therefore leaves everything in that gap inside
    the equity and outside the correction, where it is scored as model error.

    Reproduced before the fix: a $50 000 deposit landing 10 minutes after
    `resolves_at`, book read 30 minutes after, recorded as a $50 000
    model-attributable change with `external_flow_usd` 0.00 and PIT 1.0000,
    NOT flagged stale. It corrupts all three scored quantities — `pit`,
    `crps` and `var_95_breached` are every one of them a function of
    `actual_equity_change`.

    The mirror of this at the START of the window was already fixed (see
    `_snapshot_captured_at`); this is the same mistake at the other end.
    """

    DEPOSIT = 50_000.0

    def _seed(self, journal, bundle, specs, spot, books, naive, now, later_books,
              flow_at):
        provider = _WindowedProvider(books, spot, specs, later_books, flow_at,
                                     self.DEPOSIT)
        ShadowCron(provider, bundle, journal, naive, n_paths=1_000).run_once(now)
        provider.phase = "after"
        return provider

    def test_a_deposit_after_the_horizon_is_corrected_not_scored(
        self, bundle, specs, spot, books, naive, now
    ):
        resolves_at = now + timedelta(hours=24)
        flow_at = resolves_at + timedelta(minutes=10)
        observed = resolves_at + timedelta(minutes=30)
        richer = {
            a: Book(b.address, b.cross_collateral + self.DEPOSIT, b.positions,
                    b.captured_at)
            for a, b in books.items()
        }
        journal = CalibrationJournal()
        provider = self._seed(journal, bundle, specs, spot, books, naive, now,
                              richer, flow_at)
        resolve_due(journal, provider, observed)

        rows = journal._query(
            "SELECT o.actual_equity_change, o.external_flow_usd, o.stale_resolution "
            "FROM calibration_outcomes o JOIN calibration_predictions p "
            "ON p.id = o.prediction_id WHERE p.variant = ?", (VARIANT_MODEL,)
        )
        journal.close()
        assert rows, "nothing resolved; the fixture is not exercising the path"
        row = rows[0]
        assert not row["stale_resolution"], (
            "a 30-minute lag is well inside the 2h bound, so staleness does not "
            "cover this — which is what makes it dangerous"
        )
        assert row["external_flow_usd"] == pytest.approx(self.DEPOSIT), (
            "the deposit landed inside the interval the measured equity covers, "
            "so it must be subtracted"
        )
        assert abs(row["actual_equity_change"]) < self.DEPOSIT / 100.0, (
            "with the flow corrected, a book whose positions and prices did not "
            "move must record ~no model-attributable change"
        )

    def test_the_window_ends_where_the_book_was_read(
        self, bundle, specs, spot, books, naive, now
    ):
        """Stated directly, because the assertion above would also pass if the
        window were widened by luck rather than by construction."""
        resolves_at = now + timedelta(hours=24)
        observed = resolves_at + timedelta(minutes=30)
        journal = CalibrationJournal()
        provider = self._seed(journal, bundle, specs, spot, books, naive, now,
                              {}, resolves_at)
        resolve_due(journal, provider, observed)
        journal.close()

        assert provider.windows, "no flow window was requested at all"
        for _since, until in provider.windows:
            assert until > resolves_at, (
                f"window ends at {until}, at or before the horizon end "
                f"{resolves_at} — the gap to the book read is unaccounted"
            )

    def test_one_flow_fetch_per_address_not_per_variant(
        self, bundle, specs, spot, books, naive, now
    ):
        """The fix must not cost what the cache saves. `observed_at` comes off
        the per-address cache, so the three variants of one address still
        share a key — if it were read per row, each address would cost three
        fetches and the resolver's capacity arithmetic (22 weight/address)
        would be wrong by 3x."""
        resolves_at = now + timedelta(hours=24)
        observed = resolves_at + timedelta(minutes=5)
        journal = CalibrationJournal()
        provider = self._seed(journal, bundle, specs, spot, books, naive, now,
                              {}, resolves_at)
        resolve_due(journal, provider, observed)
        journal.close()

        assert len(provider.windows) == len(books), (
            f"{len(provider.windows)} flow fetches for {len(books)} addresses; "
            f"the per-address cache is not being hit"
        )


class TestTheFingerprintSeesWhatMovesThePrediction:
    """`book_changed` decides the cohort B2 says the gate is read off, so a
    material change it misses admits a row whose realisation came from a
    different book than the one predicted.

    The fingerprint covered coin, size and mode. Both of the things that make
    an ISOLATED pocket riskier or safer were invisible.
    """

    NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def _iso(self, margin: float, lev: float) -> Book:
        return Book(ADDR_A, 0.0, (Position("BTC", 1.0, 100_000.0,
                                           MarginMode.ISOLATED, lev,
                                           isolated_margin=margin),), self.NOW)

    def test_adding_margin_to_a_pocket_is_a_change(self):
        """Total equity does not move — the collateral shifts from cross into
        the pocket — so the SCORED quantity hides it entirely. What moves is
        the pocket's distance to liquidation, which is what was predicted."""
        assert position_fingerprint(self._iso(10_000.0, 10.0)) != \
            position_fingerprint(self._iso(50_000.0, 10.0))

    def test_isolated_leverage_is_a_change(self):
        """A7 measured this one: set leverage moves an isolated pocket's
        P(liq) from 0.02167 to 0.62915 over its own grid."""
        assert position_fingerprint(self._iso(10_000.0, 5.0)) != \
            position_fingerprint(self._iso(10_000.0, 25.0))

    def test_cross_leverage_is_deliberately_not_a_change(self):
        """The other half, and it is not an oversight. §1.2 makes the cross
        slider immaterial and A7 pinned the invariance through the full Monte
        Carlo — 0.161125 at 5x, 10x and 25x. Flagging it would shrink the
        cohort for a book that did not materially change."""
        def cross(lev):
            return Book(ADDR_A, 100_000.0,
                        (Position("BTC", 1.0, 100_000.0, MarginMode.CROSS, lev),),
                        self.NOW)

        assert position_fingerprint(cross(5.0)) == position_fingerprint(cross(25.0))

    def test_size_mode_and_coin_still_count(self):
        """Guard on the rework: the original three must not have been lost."""
        base = self._iso(10_000.0, 10.0)
        bigger = Book(ADDR_A, 0.0, (Position("BTC", 2.0, 100_000.0,
                                             MarginMode.ISOLATED, 10.0,
                                             isolated_margin=10_000.0),), self.NOW)
        other = Book(ADDR_A, 0.0, (Position("ETH", 1.0, 100_000.0,
                                            MarginMode.ISOLATED, 10.0,
                                            isolated_margin=10_000.0),), self.NOW)
        assert position_fingerprint(base) != position_fingerprint(bigger)
        assert position_fingerprint(base) != position_fingerprint(other)


class TestABookThatLeavesTheUniverseStillResolves:
    """The fourth gate-unreachable failure, from the resolution side.

    The sweep SKIPS an address holding an off-universe coin, so a prediction
    is only ever written for an in-universe book. But B6's world is one where
    accounts drift: an account holding only BTC/ETH/SOL when its prediction
    was written may hold ZEC a day later. `spot` answers for the tracked
    universe, so `Book.equity` raises KeyError on the new position — caught
    per row, filed transient, and retried hourly FOREVER, because
    `_permanent_reason` only knows about unusable addresses.

    Observed on the live journal 2026-08-04: 30 due rows, 30 failed, 0
    resolved, every one a KeyError on ZEC, HYPE, BCH, kPEPE or TAO. Dead rows
    accumulate daily and eat the run's 50-minute ceiling ahead of rows that
    COULD resolve, pushing those past the 2h staleness bound.
    """

    class _DriftingProvider(FakeProvider):
        """Holds an off-universe position at resolution time, and can price
        it — like the live provider, which answers `mids()` from `allMids`."""

        def __init__(self, books, spot, specs, later_books, mids):
            super().__init__(books, spot, specs, later_books=later_books)
            self._mids = mids
            self.mids_calls = 0

        def mids(self):
            self.mids_calls += 1
            return dict(self._mids)

    def _drifted(self, books, now):
        """The same book plus one position in a coin the universe lacks."""
        out = {}
        for a, b in books.items():
            out[a] = Book(b.address, b.cross_collateral,
                          (*b.positions,
                           Position("ZEC", 3.0, 250.0, MarginMode.CROSS, 5.0)),
                          b.captured_at)
        return out

    def test_it_resolves_instead_of_failing_forever(
        self, bundle, specs, spot, books, naive, now
    ):
        journal = CalibrationJournal()
        provider = self._DriftingProvider(
            books, spot, specs, self._drifted(books, now), {"ZEC": 250.0})
        ShadowCron(provider, bundle, journal, naive, n_paths=1_000).run_once(now)
        provider.phase = "after"
        report = resolve_due(journal, provider, now + timedelta(hours=24))
        journal.close()

        assert report.resolved > 0, (
            f"nothing resolved; failures were {report.failed[:3]}"
        )
        assert not report.failed, report.failed[:3]

    def test_the_drifted_book_is_marked_changed(
        self, bundle, specs, spot, books, naive, now
    ):
        """Resolving it must not smuggle it into the strict cohort. The new
        position changes the fingerprint, so B2's filter excludes it from
        book-unchanged while the all-observations cohort keeps it — which is
        the point of valuing it rather than dropping it."""
        journal = CalibrationJournal()
        provider = self._DriftingProvider(
            books, spot, specs, self._drifted(books, now), {"ZEC": 250.0})
        ShadowCron(provider, bundle, journal, naive, n_paths=1_000).run_once(now)
        provider.phase = "after"
        resolve_due(journal, provider, now + timedelta(hours=24))
        rows = journal._query("SELECT book_changed FROM calibration_outcomes")
        journal.close()
        assert rows and all(r["book_changed"] for r in rows)

    def test_the_extra_prices_are_fetched_at_most_once_per_run(
        self, bundle, specs, spot, books, naive, now
    ):
        """`allMids` is weight 2 for the whole venue, but per-address it would
        still be a per-address cost the capacity arithmetic does not budget."""
        journal = CalibrationJournal()
        provider = self._DriftingProvider(
            books, spot, specs, self._drifted(books, now), {"ZEC": 250.0})
        ShadowCron(provider, bundle, journal, naive, n_paths=1_000).run_once(now)
        provider.phase = "after"
        resolve_due(journal, provider, now + timedelta(hours=24))
        journal.close()
        assert provider.mids_calls == 1, provider.mids_calls

    def test_an_in_universe_book_never_asks_for_them(
        self, bundle, specs, spot, books, naive, now
    ):
        """The common case must not pay for the rare one."""
        journal = CalibrationJournal()
        provider = self._DriftingProvider(books, spot, specs, {}, {"ZEC": 250.0})
        ShadowCron(provider, bundle, journal, naive, n_paths=1_000).run_once(now)
        provider.phase = "after"
        resolve_due(journal, provider, now + timedelta(hours=24))
        journal.close()
        assert provider.mids_calls == 0


class TestTheCrpsComparisonIsPaired:
    """§0.2 asks whether the model beats both baselines. The three variants'
    cohorts were loaded independently and their MEANS compared, which treats
    them as describing the same observations — and they need not.

    A row can fail on its own, and the resolver's ceiling lands BETWEEN rows
    rather than between addresses: `due()` orders by `resolves_at`, which the
    three variants of one address share, so a truncated run routinely leaves
    an address with one or two of its three rows resolved.

    That is not a rounding-level concern, because CRPS carries the units of
    the equity change and its mean is dominated by the largest accounts.
    """

    VERSION = DISTRIBUTION_VERSION

    def _write(self, journal, address, variant, day, crps, resolve=True):
        import json as _json

        predicted_at = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=day)
        resolves_at = predicted_at + timedelta(hours=24)
        rows = journal._query(
            """
            INSERT INTO calibration_predictions (
                address, variant, predicted_at, horizon_hours, resolves_at,
                model_version, distribution_version, seed, n_paths, converged,
                start_equity, p_liq, p_liq_ci_low, p_liq_ci_high, var_95,
                cvar_95, quantile_values, n_quantile_levels, book_snapshot
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            RETURNING id
            """,
            (address, variant, predicted_at.isoformat(), 24,
             resolves_at.isoformat(), "0.4.0", self.VERSION, 1, 10, True,
             1_000.0, 0.0, 0.0, 0.0, 100.0, 100.0,
             _json.dumps(list(np.linspace(-1e3, 1e3, 11))), 11, _json.dumps({})),
        )
        journal.backend.commit()
        if not resolve:
            return
        journal.record_outcome(
            prediction_id=int(rows[0]["id"]), resolved_at=resolves_at,
            actual_equity=1_000.0, actual_equity_change=0.0,
            external_flow_usd=0.0, book_changed=False, liquidated=False,
            pit=0.5, pit_u=0.5, crps=crps, var_95_breached=False,
            observation_day=predicted_at.date(), resolution_lag_s=0.0,
            stale_resolution=False,
        )

    def _journal_with_a_whale(self, drop_from):
        """Twenty small accounts where the model is genuinely better, plus one
        whale all three score IDENTICALLY — so no verdict may turn on it."""
        journal = CalibrationJournal()
        for i in range(20):
            a = addr(i)
            self._write(journal, a, VARIANT_MODEL, i, 10.0)
            self._write(journal, a, VARIANT_BASELINE_A, i, 12.0)
            self._write(journal, a, VARIANT_BASELINE_B, i, 12.0)
        whale = "0x" + "f" * 40
        for variant in (VARIANT_MODEL, VARIANT_BASELINE_A, VARIANT_BASELINE_B):
            self._write(journal, whale, variant, 20, 5_000.0,
                        resolve=variant != drop_from)
        return journal

    def test_one_missing_baseline_row_does_not_flip_the_verdict(self):
        """Reproduced before the fix: the model 'lost' to A (247.6 against
        12.0) and 'beat' B (247.6 against 249.5) on identical data, purely
        because one baseline-A row had not resolved."""
        journal = self._journal_with_a_whale(drop_from=VARIANT_BASELINE_A)
        report = calibration_report(journal, self.VERSION, cohort=COHORT_ALL)
        journal.close()
        assert report.crps_beats_baseline_a, report.mean_crps
        assert report.crps_beats_baseline_b, report.mean_crps

    def test_a_missing_MODEL_row_does_not_flatter_it_either(self):
        """The other direction, and the one §10 forbids: dropping the whale
        from the MODEL's own cohort would lower its mean and let a model that
        is not better appear to win."""
        journal = CalibrationJournal()
        for i in range(20):
            a = addr(i)
            self._write(journal, a, VARIANT_MODEL, i, 12.0)     # model is WORSE
            self._write(journal, a, VARIANT_BASELINE_A, i, 10.0)
            self._write(journal, a, VARIANT_BASELINE_B, i, 10.0)
        whale = "0x" + "f" * 40
        self._write(journal, whale, VARIANT_MODEL, 20, 5_000.0, resolve=False)
        self._write(journal, whale, VARIANT_BASELINE_A, 20, 5_000.0)
        self._write(journal, whale, VARIANT_BASELINE_B, 20, 5_000.0)
        report = calibration_report(journal, self.VERSION, cohort=COHORT_ALL)
        journal.close()
        assert not report.crps_beats_baseline_a, report.mean_crps
        assert not report.crps_beats_baseline_b, report.mean_crps

    def test_the_report_says_when_rows_could_not_be_paired(self):
        """Said whenever it happens, not only when it changes a verdict: an
        unpaired comparison that happens to agree is still one nobody could
        audit from the output."""
        journal = self._journal_with_a_whale(drop_from=VARIANT_BASELINE_A)
        report = calibration_report(journal, self.VERSION, cohort=COHORT_ALL)
        journal.close()
        assert report.n_paired == 20
        assert report.n == 21, "the model's own n is unchanged"
        assert sum(report.unpaired_dropped.values()) == 2
        assert "paired" in report.gate_summary

    def test_the_model_keeps_every_row_for_its_own_calibration(self):
        """PIT, KS and the tail describe the MODEL alone rather than a
        comparison, so pairing must not shrink them — that would discard data
        for no reason."""
        journal = self._journal_with_a_whale(drop_from=VARIANT_BASELINE_A)
        report = calibration_report(journal, self.VERSION, cohort=COHORT_ALL)
        journal.close()
        assert report.tail.n == 21


class TestTheBundleBuildPacesAtTheCharge:
    """The 4-coin bundle build costs ~470 weight against the shadow pool's
    300/minute usable, so NO single pass fits inside one sliding window. The
    first `_paced_bundle` retried the WHOLE build on `RateLimitExceeded`:
    every retry re-spent the head of the build, pinned the pool at its cap
    and starved the tail — both shadow jobs sat in "window is spent" for the
    full 30-minute ceiling while holding the pool themselves (observed on
    the first 4-coin start, 2026-08-06). The wait must therefore live at the
    CHARGE, where a build stretches over minutes and completes.
    """

    def test_the_build_is_handed_a_per_charge_pacing_budget(self, monkeypatch):
        import risk_engine.service.state as state
        import risk_engine.shadow.cli as shadow_cli
        from risk_engine.market.info import PacedBudget, WeightBudget

        captured = {}

        def fake_build(*, serving, budget):
            captured["serving"] = serving
            captured["budget"] = budget
            return "bundle", {}, {}

        monkeypatch.setattr(state, "_build_live_bundle", fake_build)
        inner = WeightBudget(reserved_fraction=0.75)
        out = shadow_cli._paced_bundle(inner)

        assert out == ("bundle", {}, {})
        assert isinstance(captured["budget"], PacedBudget), (
            "an unpaced budget re-creates the whole-build retry livelock: "
            "no 4-coin build can fit a 300/minute window in one pass"
        )
        assert captured["budget"].inner is inner, (
            "the pacing must wrap the SHARED ledger, not a private window"
        )
        assert captured["serving"] is False

    def test_a_build_wider_than_one_window_completes_by_waiting(self, monkeypatch):
        """The livelock scenario in miniature: a build whose total cost
        exceeds the usable window succeeds only if charges WAIT for expiry
        rather than abort the pass. Time is virtualised through the budget's
        `now` hooks in `spent`/`charge`... which `PacedBudget.charge` refuses
        to pace (a frozen clock never refills), so this drives the real
        pacing loop with a real-but-fast clock: wait_s=0 and an inner budget
        whose window drains as refusals accumulate."""
        from risk_engine.market.info import (
            PacedBudget,
            RateLimitExceeded,
        )

        class DrainingBudget:
            """Refuses twice per charge, then admits: a window that frees
            only while the caller waits, in miniature."""

            def __init__(self):
                self.refusals_left = 0
                self.charged = []

            def charge(self, weight, now=None):
                if self.refusals_left > 0:
                    self.refusals_left -= 1
                    raise RateLimitExceeded("window full")
                self.refusals_left = 2
                self.charged.append(weight)

            def charge_incurred(self, weight, now=None):
                self.charged.append(weight)

        inner = DrainingBudget()
        inner.refusals_left = 2
        paced = PacedBudget(inner, max_wait_s=5.0, wait_s=0.0)
        # Seven charges, as the 4-coin build makes (meta + candle/funding per
        # two coins here); every one must land despite the refusals.
        for weight in (20, 20, 36, 20, 36, 20, 20):
            if weight == 36:
                paced.charge_incurred(weight)
            else:
                paced.charge(weight)
        assert sum(inner.charged) == 172, "every charge must eventually land"
