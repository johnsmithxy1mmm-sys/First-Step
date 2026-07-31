"""SQLite and Postgres behind one journal (§3.4).

The calibration journal is the product's only durable moat, so it has to
survive on a laptop during development and on Postgres in production, and
the two must record *the same thing* — a dev/prod schema drift would show up
as an unexplained discontinuity in a public calibration score years later.

Rather than two hand-maintained schemas, there is one canonical DDL in
`schema.sql` (Postgres dialect) and a small translation to SQLite. The
translation is mechanical and the test suite asserts that both backends end
up with identical column sets.

Only two dialect differences matter here and both are contained in this
module: the parameter placeholder (`?` against `%s`) and the autoincrement
primary key. Everything else in the journal's SQL is standard.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Protocol

SCHEMA_SQL = Path(__file__).with_name("schema.sql")

#: Postgres type -> SQLite type. SQLite is dynamically typed, so these are
#: affinities rather than constraints; the point is that both backends carry
#: the same COLUMNS, which is what a cross-backend calibration record needs.
_TYPE_MAP = (
    (r"\bBIGSERIAL PRIMARY KEY\b", "INTEGER PRIMARY KEY AUTOINCREMENT"),
    (r"\bBIGINT PRIMARY KEY REFERENCES\b", "INTEGER PRIMARY KEY REFERENCES"),
    (r"\bDOUBLE PRECISION\b", "REAL"),
    (r"\bTIMESTAMPTZ\b", "TEXT"),
    (r"\bJSONB\b", "TEXT"),
    (r"\bBOOLEAN\b", "INTEGER"),
    (r"\bDATE\b", "TEXT"),
    (r"\bBIGINT\b", "INTEGER"),
)


def canonical_ddl() -> str:
    # encoding="utf-8" explicitly: schema.sql carries `§` section references in
    # its comments, and the platform default is cp1251 on a Russian-locale
    # Windows box. The DDL would still execute -- the mojibake lands in
    # comments -- but the schema is the thing a reader consults to understand
    # the journal, and shipping it garbled on one platform is a defect of the
    # documentation that matters most.
    return SCHEMA_SQL.read_text(encoding="utf-8")


def sqlite_ddl() -> str:
    """The canonical Postgres DDL, translated.

    Derived rather than duplicated: a second hand-written schema is a second
    thing to forget to update, and the failure mode is silent until someone
    compares a dev journal against a production one.
    """
    ddl = canonical_ddl()
    # Strip SQL comments so a `--` inside prose cannot be mistaken for DDL.
    ddl = re.sub(r"^\s*--.*$", "", ddl, flags=re.MULTILINE)
    for pattern, replacement in _TYPE_MAP:
        ddl = re.sub(pattern, replacement, ddl)
    return ddl


class Backend(Protocol):
    """The narrow surface the journal needs from a database."""

    @property
    def placeholder(self) -> str: ...

    def execute(self, sql: str, params: tuple = ()) -> Any: ...

    def executescript(self, sql: str) -> None: ...

    def commit(self) -> None: ...

    def close(self) -> None: ...

    def lastrowid(self, cursor: Any, table: str) -> int: ...

    def rows(self, cursor: Any) -> list[dict]: ...


class SqliteBackend:
    placeholder = "?"

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row

    def execute(self, sql: str, params: tuple = ()):
        return self.conn.execute(sql, params)

    def executescript(self, sql: str) -> None:
        self.conn.executescript(sql)

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def lastrowid(self, cursor, table: str) -> int:
        return int(cursor.lastrowid)

    def rows(self, cursor) -> list[dict]:
        return [dict(r) for r in cursor.fetchall()]


class PostgresBackend:
    """psycopg 3. Imported lazily so the engine's dependency stays numpy+scipy."""

    placeholder = "%s"

    def __init__(self, dsn: str) -> None:
        import psycopg
        from psycopg.rows import dict_row

        self.conn = psycopg.connect(dsn, row_factory=dict_row)

    def execute(self, sql: str, params: tuple = ()):
        cur = self.conn.cursor()
        try:
            cur.execute(sql, params)
        except Exception:
            # Postgres aborts the whole transaction on any failed statement
            # and refuses every subsequent command until it is rolled back --
            # SQLite does not, which is why this only shows up against a real
            # server. Without the rollback, one duplicate row would poison the
            # rest of a resolver batch, and the resolver is written to catch
            # per-row failures and carry on (§3.3). Roll back, then re-raise
            # so the caller still sees the original error.
            self.conn.rollback()
            raise
        return cur

    def executescript(self, sql: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(sql)
        self.conn.commit()

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def lastrowid(self, cursor, table: str) -> int:
        # Postgres has no lastrowid; the journal appends RETURNING id and the
        # value is already on the cursor.
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError(f"insert into {table} returned no id")
        return int(row["id"])

    def rows(self, cursor) -> list[dict]:
        return [dict(r) for r in cursor.fetchall()]


def open_backend(target: str | Path = ":memory:") -> Backend:
    """`postgresql://...` opens Postgres; anything else is a SQLite path."""
    text = str(target)
    if text.startswith(("postgres://", "postgresql://")):
        return PostgresBackend(text)
    return SqliteBackend(text)
