"""The shared §5.3 weight pool (OPEN-QUESTIONS C6).

The Postgres half runs only when `HL_TEST_POSTGRES_DSN` is set, the same
arrangement as `test_journal_backends.py` and for the same reason: CI
without a database still runs the fallback logic, and a developer with one
gets the part that actually failed in production shape -- two processes,
one pool.
"""

from __future__ import annotations

import os

import pytest

from risk_engine.market.info import (
    SERVING_RESERVED_FRACTION,
    SHADOW_RESERVED_FRACTION,
    WEIGHT_BUDGET_PER_MINUTE,
    RateLimitExceeded,
    WeightBudget,
)
from risk_engine.shadow.weight_ledger import SharedWeightBudget, open_weight_budget

PG_DSN = os.environ.get("HL_TEST_POSTGRES_DSN")


class TestTheSplitSumsToTheVenueLimit:
    def test_the_fractions_are_complements(self):
        """Serving plus shadow must be exactly the venue's window.

        The serving engine used to take all 1200 while the shadow jobs took
        300 on top: a deployment ceiling of 125% of what the venue serves.
        """
        assert SHADOW_RESERVED_FRACTION + SERVING_RESERVED_FRACTION == 1.0
        shadow = int(WEIGHT_BUDGET_PER_MINUTE * (1 - SHADOW_RESERVED_FRACTION))
        serving = int(WEIGHT_BUDGET_PER_MINUTE * (1 - SERVING_RESERVED_FRACTION))
        assert shadow + serving == WEIGHT_BUDGET_PER_MINUTE

    def test_one_home_for_the_shadow_fraction(self):
        """cli.py and cron.py re-export the same constant, not two 0.75s."""
        from risk_engine.shadow import cli, cron

        assert cli.SHADOW_RESERVED_FRACTION is cron.SHADOW_RESERVED_FRACTION
        assert cli.SHADOW_RESERVED_FRACTION == SHADOW_RESERVED_FRACTION


class TestFallback:
    def test_a_sqlite_path_gets_the_in_process_window(self):
        b = open_weight_budget("shadow.db", reserved_fraction=0.75)
        assert isinstance(b, WeightBudget)
        assert b.reserved_fraction == 0.75

    def test_none_gets_the_in_process_window(self):
        assert isinstance(open_weight_budget(None), WeightBudget)


@pytest.mark.skipif(not PG_DSN, reason="HL_TEST_POSTGRES_DSN not set")
class TestSharedPool:
    def _fresh(self, actor: str) -> SharedWeightBudget:
        b = SharedWeightBudget(PG_DSN, reserved_fraction=0.75, actor=actor)
        # Each test starts from an empty window; the table is shared state.
        with b._conn.cursor() as cur:
            cur.execute("DELETE FROM venue_weight_ledger")
        b._conn.commit()
        return b

    def test_two_processes_share_one_ceiling(self):
        """The defect this module exists for: two actors, 300 combined.

        With private budgets each actor had its own 300 and the venue saw up
        to 600/minute of background traffic -- a 50% reserve where §5.3
        promises 75%.
        """
        snap = self._fresh("snapshot")
        reso = SharedWeightBudget(PG_DSN, reserved_fraction=0.75, actor="resolve")
        charged = 0
        with pytest.raises(RateLimitExceeded):
            for i in range(31):
                (snap if i % 2 == 0 else reso).charge(20)
                charged += 20
        assert charged == 300
        snap.close()
        reso.close()

    def test_a_dsn_gets_the_shared_ledger(self):
        b = open_weight_budget(PG_DSN, reserved_fraction=0.75)
        assert isinstance(b, SharedWeightBudget)
        b.close()

    def test_a_full_window_admits_nothing_from_a_new_connection(self):
        """The race the advisory lock closes: read-decide-write is atomic."""
        first = self._fresh("first")
        for _ in range(15):
            first.charge(20)
        late = SharedWeightBudget(PG_DSN, reserved_fraction=0.75, actor="late")
        with pytest.raises(RateLimitExceeded):
            late.charge(20)
        first.close()
        late.close()
