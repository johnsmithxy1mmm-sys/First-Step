"""Verifications that happened outside a `verify` run, recorded with provenance.

Some assumptions cannot be settled inside one command. C2 needs a basis series
spanning hours, not the minute a CLI run can spare; C5 needs a funded account
observed across an hourly funding tick, which no read-only endpoint can
substitute for. Both were answered — and `verify` went on reporting them as
unconfirmed on every run, because it had no way to know.

That is not a cosmetic problem. The summary line said "3 assumption(s) still
unconfirmed: C1, C2, C5" when one was unconfirmed and two were answered, which
invites redoing settled work and, worse, teaches an operator to discount the
line. It is the same defect the READMEs had when they claimed the live API had
never been reached: a stale statement that outlived its truth and was never
revisited because nothing forced the revisit.

So a recorded finding is a first-class object here, and the design constraints
follow from what makes such a record dangerous rather than useful:

  - **It is never mistaken for a live result.** The status renders as
    `PASS (recorded)`, and the detail leads with when, by what command, and on
    which network. An operator must be able to see they are reading history.
  - **It carries the command.** A finding whose reproduction is folklore is
    not evidence, and the point of writing it down is that the next person can
    re-run it rather than trust it.
  - **It ages visibly.** Venues change. A finding states its age in days, and
    past `STALE_AFTER_DAYS` it degrades to INCONCLUSIVE on its own rather than
    silently vouching for a year-old observation.
  - **Live data outranks it.** If a live check reaches a verdict, that verdict
    wins. A recorded PASS contradicted by a live FAIL is reported as the FAIL
    plus the contradiction, because the interesting event is precisely that
    they disagree.
  - **The network is part of the claim.** A testnet observation is evidence
    about testnet. It may be the best available and is worth recording, but
    promoting it to a mainnet claim silently is exactly the kind of laundering
    this file exists to prevent, so the network is rendered every time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Past this, a recorded finding stops vouching for anything and degrades to
#: INCONCLUSIVE. Ninety days is the same window the ledger and candle checks
#: use, chosen for consistency rather than from any evidence about how fast
#: this venue changes -- which nobody here has. It is a deliberate
#: under-claim: too short costs a re-run, too long lets a stale record stand
#: in for a fact that has moved.
STALE_AFTER_DAYS = 90

#: Where `verify` looks unless told otherwise. Version-controlled on purpose:
#: these are findings about the venue, they belong with the questions they
#: answer, and a reviewer should see one arrive in a diff.
DEFAULT_FINDINGS_PATH = Path(__file__).resolve().parents[2] / "docs/hl-risk/VERIFIED.json"


@dataclass(frozen=True, slots=True)
class RecordedFinding:
    """One out-of-band verification, and everything needed to distrust it."""

    id: str
    status: str
    observed_utc: str
    command: str
    network: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def age_days(self, now: datetime | None = None) -> float:
        observed = datetime.fromisoformat(self.observed_utc)
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        return ((now or datetime.now(timezone.utc)) - observed).total_seconds() / 86400.0

    def is_stale(self, now: datetime | None = None) -> bool:
        return self.age_days(now) > STALE_AFTER_DAYS

    def provenance(self, now: datetime | None = None) -> str:
        age = self.age_days(now)
        return (
            f"recorded {self.observed_utc} ({age:.0f}d ago) on {self.network}, "
            f"by `{self.command}`"
        )


def load_findings(path: Path | str | None = None) -> dict[str, RecordedFinding]:
    """Read the findings file, or return nothing if there is not one.

    A missing file is the normal case for a fresh checkout and must not be an
    error. A *malformed* one is an error: silently ignoring a file an operator
    wrote is how a finding gets recorded, believed, and never applied.
    """
    p = Path(path) if path is not None else DEFAULT_FINDINGS_PATH
    if not p.exists():
        return {}
    try:
        # encoding="utf-8" explicitly. Without it `read_text` uses the platform
        # locale. This file holds literal `§` characters (U+00A7, UTF-8 bytes
        # C2 A7); under cp1251 those two bytes decode as U+0412 followed by
        # U+00A7, so every section reference in a recorded finding rendered as
        # mojibake on an operator's console -- text written to be read. Under a
        # stricter locale it is worse than ugly: an ASCII default raises
        # UnicodeDecodeError and the whole command dies.
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{p}: not valid JSON ({exc}). ") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{p}: expected an object keyed by check id, got {type(payload).__name__}")

    out: dict[str, RecordedFinding] = {}
    required = ("status", "observed_utc", "command", "network", "detail")
    # Audit F-6: `Check.passed` is a prefix test ("PASS (recorded)" must count),
    # so an unvalidated status here let "PASSS" — an operator's typo — satisfy
    # a blocking check. The file is hand-edited by design; hand-edited means
    # typos are the expected input, not the surprising one.
    valid_statuses = ("PASS", "FAIL", "INCONCLUSIVE")
    for check_id, raw in payload.items():
        if check_id.startswith("_"):
            continue  # room for a "_comment" key without inventing a schema
        if not isinstance(raw, dict):
            raise ValueError(f"{p}: {check_id} must be an object, got {type(raw).__name__}")
        missing = [k for k in required if not str(raw.get(k, "")).strip()]
        if missing:
            # Refusing is the point. A finding without its command cannot be
            # re-run, and one without its date cannot age out -- both would
            # make this file a place where claims go to become permanent.
            raise ValueError(
                f"{p}: {check_id} is missing {missing}. A recorded verification "
                f"without these is a claim rather than evidence: 'command' is how "
                f"the next person reproduces it, 'observed_utc' is how it expires, "
                f"and 'network' is what it is a claim about."
            )
        if raw["status"] not in valid_statuses:
            raise ValueError(
                f"{p}: {check_id} status {raw['status']!r} is not one of "
                f"{valid_statuses}. The harness matches statuses by prefix so a "
                f"typo here would silently satisfy (or silently fail) a blocking "
                f"check rather than being noticed."
            )
        try:
            datetime.fromisoformat(raw["observed_utc"])
        except ValueError as exc:
            raise ValueError(
                f"{p}: {check_id} observed_utc {raw['observed_utc']!r} is not an "
                f"ISO-8601 timestamp, so its age cannot be computed"
            ) from exc
        out[check_id] = RecordedFinding(
            id=check_id,
            status=raw["status"],
            observed_utc=raw["observed_utc"],
            command=raw["command"],
            network=raw["network"],
            detail=raw["detail"],
            evidence=raw.get("evidence") or {},
        )
    return out
