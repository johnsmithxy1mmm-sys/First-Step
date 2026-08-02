"""One venue weight budget shared across processes (§5.3, OPEN-QUESTIONS C6).

`WeightBudget` is a per-process sliding window, and §5.3's promise -- keep
three quarters of the venue's 1200/minute for interactive users -- is a claim
about the whole deployment, not about one process. The shipped stack runs the
snapshot job and the resolve job in separate containers behind one egress IP,
each constructing its own budget with `reserved_fraction=0.75`. Two processes
at 300/minute each is 600/minute of background traffic: during their daily
overlap the reserve was 50%, not the 75% the design states, and nothing
anywhere could observe that because neither process could see the other's
spending.

The obvious repair -- halve each job's share to 150/minute -- is wrong, and
measurably so. The resolver's capacity arithmetic depends on 300/minute:
§3.3's floor of 200 addresses costs 8000 weight, which is 27 minutes at 300
and 53 at 150, past the 50-minute run ceiling. Splitting the pool statically
would re-open the gate-unreachable failure that ceiling exists to prevent,
for two jobs that are usually not even running at the same time.

So the pool is shared rather than divided: whichever job is running may use
all 300/minute, and when both run they draw from the same window. The ledger
lives in the calibration database because both jobs are already connected to
it, and because a background job that cannot reach that database cannot
record anything anyway -- there is no new failure mode, only a new user of an
existing dependency.

The SERVING engine deliberately does NOT participate. It is the interactive
traffic the reserve exists to protect, its own venue calls are the
five-minute rebuild rather than anything on the request path, and making the
product's availability depend on the calibration database would be a bad
trade for an accounting improvement. Its share is bounded the other way, by
giving it the complement (`SERVING_RESERVED_FRACTION` in `market/info.py`)
so that serving plus shadow sums to the venue's limit instead of exceeding
it.

Atomicity matters here in a way it does not in-process: two containers can
call `available()` simultaneously, both see room, and both spend it. Every
charge therefore takes a transaction-scoped advisory lock, so the
read-decide-write is serialised across processes.
"""

from __future__ import annotations

import logging

from risk_engine.market.info import (
    WEIGHT_BUDGET_PER_MINUTE,
    RateLimitExceeded,
    WeightBudget,
)

log = logging.getLogger("risk_engine.shadow.weight_ledger")

#: Any constant; it only has to be the same in every process sharing the pool.
#: Postgres advisory locks live in a global namespace keyed by this integer.
_LOCK_KEY = 0x5745494748_54 % (2**63)


class SharedWeightBudget:
    """`WeightBudget`'s interface, backed by a table every process can see.

    Deliberately duck-typed rather than a subclass: `InfoClient` only ever
    calls `charge`, and keeping this off the dataclass avoids inheriting a
    `_events` list that would look authoritative and be empty.
    """

    def __init__(
        self,
        dsn: str,
        limit_per_minute: int = WEIGHT_BUDGET_PER_MINUTE,
        reserved_fraction: float = 0.0,
        actor: str = "shadow",
    ) -> None:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - same message as backends.py
            raise ImportError(
                "a shared weight ledger needs psycopg:\n"
                "    pip install 'psycopg[binary]>=3.1'\n"
                "It is in risk_engine/requirements.txt."
            ) from exc
        self.limit_per_minute = limit_per_minute
        self.reserved_fraction = reserved_fraction
        self.actor = actor
        self._conn = psycopg.connect(dsn, autocommit=False)
        self._ensure_table()

    @property
    def usable(self) -> int:
        return int(self.limit_per_minute * (1.0 - self.reserved_fraction))

    def _ensure_table(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS venue_weight_ledger (
                    id          BIGSERIAL PRIMARY KEY,
                    charged_at  TIMESTAMPTZ NOT NULL,
                    weight      INTEGER     NOT NULL,
                    actor       TEXT        NOT NULL
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS venue_weight_ledger_at "
                "ON venue_weight_ledger (charged_at)"
            )
        self._conn.commit()

    def spent(self, now: float | None = None) -> int:
        """Weight charged by ANY process in the last minute.

        `now` is accepted and ignored: the window is measured on the database
        clock, which is the only clock all the participants share.
        """
        del now
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(SUM(weight), 0) FROM venue_weight_ledger "
                "WHERE charged_at > now() - interval '60 seconds'"
            )
            row = cur.fetchone()
        self._conn.commit()
        return int(row[0]) if row else 0

    def available(self, now: float | None = None) -> int:
        return max(0, self.usable - self.spent(now))

    def charge(self, weight: int, now: float | None = None) -> None:
        """Reserve `weight`, or raise `RateLimitExceeded`.

        Read and write happen under one transaction-scoped advisory lock, so
        two containers cannot both observe the same headroom and spend it.
        The lock is released by COMMIT or ROLLBACK either way.
        """
        del now
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_KEY,))
                # Pruning inside the same transaction keeps the table small
                # without a separate janitor; the rows are worthless once
                # outside the window.
                cur.execute(
                    "DELETE FROM venue_weight_ledger "
                    "WHERE charged_at < now() - interval '60 seconds'"
                )
                cur.execute(
                    "SELECT COALESCE(SUM(weight), 0) FROM venue_weight_ledger "
                    "WHERE charged_at > now() - interval '60 seconds'"
                )
                row = cur.fetchone()
                spent = int(row[0]) if row else 0
                remaining = max(0, self.usable - spent)
                if weight > remaining:
                    self._conn.rollback()
                    raise RateLimitExceeded(
                        f"weight {weight} exceeds remaining {remaining} in the shared "
                        f"§5.3 window ({spent}/{self.usable} spent across all shadow "
                        "processes in the last minute)"
                    )
                cur.execute(
                    "INSERT INTO venue_weight_ledger (charged_at, weight, actor) "
                    "VALUES (now(), %s, %s)",
                    (int(weight), self.actor),
                )
            self._conn.commit()
        except RateLimitExceeded:
            raise
        except Exception:
            # Postgres aborts the transaction on any failed statement and
            # refuses everything after it until rolled back; the paced callers
            # retry, and a poisoned connection would turn one hiccup into a
            # dead run.
            self._conn.rollback()
            raise

    def close(self) -> None:
        self._conn.close()


def open_weight_budget(
    target: str | None,
    limit_per_minute: int = WEIGHT_BUDGET_PER_MINUTE,
    reserved_fraction: float = 0.0,
    actor: str = "shadow",
) -> WeightBudget | SharedWeightBudget:
    """Shared ledger for a Postgres journal, in-process budget otherwise.

    A sqlite journal is a single-machine development arrangement where the
    two jobs are not running concurrently in separate containers, so the
    in-process window is already accurate there. Falling back is stated in
    the log rather than assumed, because "the reserve is per-process" is
    exactly the sort of thing that is invisible until it is measured.
    """
    text = str(target or "")
    if text.startswith(("postgres://", "postgresql://")):
        return SharedWeightBudget(
            text, limit_per_minute=limit_per_minute,
            reserved_fraction=reserved_fraction, actor=actor,
        )
    log.info(
        "weight budget is per-process (journal is not Postgres); §5.3's reserve is "
        "only shared across shadow jobs when they share a database"
    )
    return WeightBudget(
        limit_per_minute=limit_per_minute, reserved_fraction=reserved_fraction
    )
