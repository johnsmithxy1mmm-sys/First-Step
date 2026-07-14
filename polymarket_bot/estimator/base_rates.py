"""Base-rates signal: historical event frequencies from a YAML reference.

p_est for a market "will X happen by date D":
    p = 1 - (1 - annual_probability) ^ (days_to_D / 365)
— the probability of at least one Poisson-type event over the remaining term.
Applies only to the Yes side: a base rate describes the event occurring, not
its absence.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from ..models import Candidate, Signal

log = logging.getLogger(__name__)


class BaseRateEntry(BaseModel):
    name: str
    keywords_all: list[str] = Field(default_factory=list)
    keywords_any: list[str] = Field(default_factory=list)
    annual_probability: float
    confidence: float = 0.4
    note: str = ""

    def matches(self, question: str) -> bool:
        q = question.lower()
        if not all(k.lower() in q for k in self.keywords_all):
            return False
        if self.keywords_any and not any(k.lower() in q for k in self.keywords_any):
            return False
        return bool(self.keywords_all or self.keywords_any)


def load_base_rates(path: Path) -> list[BaseRateEntry]:
    if not path.exists():
        log.warning("base rates: file %s not found, signal disabled", path)
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [BaseRateEntry.model_validate(item) for item in raw]


def probability_before(annual_probability: float, days_left: float) -> float:
    """P(at least one event within days_left) given annual_probability."""
    if days_left <= 0:
        return 0.0
    p = 1.0 - (1.0 - min(annual_probability, 0.999999)) ** (days_left / 365.0)
    return min(max(p, 0.0), 0.999)


class BaseRatesSignal:
    name = "base_rates"

    def __init__(self, entries: list[BaseRateEntry]):
        self._entries = entries

    def evaluate(self, candidate: Candidate, now: datetime | None = None) -> Signal | None:
        if candidate.outcome_index != 0:
            return None  # base rates are phrased for the Yes side
        days = candidate.market.days_to_resolution(now or datetime.now(timezone.utc))
        if days is None:
            return None
        for entry in self._entries:
            if entry.matches(candidate.market.question):
                p = probability_before(entry.annual_probability, days)
                return Signal(
                    name=self.name,
                    p_est=p,
                    confidence=entry.confidence,
                    rationale=f"{entry.name}: annual={entry.annual_probability:g}, "
                              f"days={days:.0f} -> p={p:.4f} ({entry.note})",
                )
        return None
