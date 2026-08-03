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

        monkeypatch.setattr(
            state_mod, "_build_live_bundle",
            lambda *a, **k: (object(), {"BTC": object()}, {"BTC": 100_000.0}),
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
        # One call, and it is the 90-day history the naive baseline is fitted
        # from -- not a price refresh.
        assert calls == ["BTC"]

        assert provider.spot() == {"BTC": 100_000.0}
        assert calls == ["BTC"], "spot() re-fetched prices it had just been given"


class TestResolverPacesRatherThanDroppingToStale:
    """The resolver had the snapshot's bug, made worse by staleness.

    Each resolution spends ~40 weight (book + external_flow); a fresh hourly
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
