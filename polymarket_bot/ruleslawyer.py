"""Strategy #4 support: LLM rules lawyer.

Prediction-market edge often hides in the gap between a market's headline and
the letter of its resolution rules ("Will X happen?" but the rules require an
official source that will not report in time). This module asks Claude to
compare the two and flag exploitable discrepancies via structured output.

Alerts a human (who makes the call) and, when confident, contributes a small
signal to the ensemble. Off by default; costs API calls, so it is rate-limited
and only runs on liquid markets. The Claude client is injected, so the logic is
fully testable with a mock.
"""

from __future__ import annotations

import logging
import os

from pydantic import BaseModel, Field

from .config import BotConfig
from .ledger import Ledger
from .models import Market
from .monitor import alert

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a meticulous prediction-market resolution analyst. Compare the "
    "market's headline question with the exact letter of its resolution rules. "
    "Flag a discrepancy ONLY when the literal rules would plausibly resolve the "
    "market differently from what the headline implies (e.g. a required official "
    "source, a strict date, an unusual threshold). Otherwise report no discrepancy."
)


class RulesVerdict(BaseModel):
    discrepancy: bool = Field(description="True if the letter of the rules diverges "
                                          "from the headline in an exploitable way")
    favored_side: str = Field(description="'Yes', 'No', or 'none' — the side the "
                                          "literal rules favor")
    rationale: str = Field(description="One or two sentences on the gap")


class RulesLawyer:
    def __init__(self, cfg: BotConfig, ledger: Ledger, client=None):
        self._cfg = cfg.ruleslawyer
        self._ledger = ledger
        self._client = client
        self._calls = 0

    @property
    def available(self) -> bool:
        return self._cfg.enabled and (self._client is not None
                                      or bool(os.environ.get("ANTHROPIC_API_KEY")))

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def analyze(self, market: Market) -> RulesVerdict | None:
        if not market.description.strip():
            return None
        prompt = (f"Headline: {market.question}\n"
                  f"Resolution source: {market.resolution_source or 'n/a'}\n"
                  f"Resolution rules:\n{market.description[:2000]}")
        self._calls += 1
        try:
            resp = self._get_client().messages.parse(
                model=self._cfg.model, max_tokens=self._cfg.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_format=RulesVerdict)
        except Exception as exc:            # network / API — never break the cycle
            log.warning("rules lawyer: %s", exc)
            return None
        return resp.parsed_output

    def cycle(self, markets: list[Market]) -> list[tuple[Market, RulesVerdict]]:
        """Analyze fresh liquid markets, alert on discrepancies. -> flagged."""
        if not self.available:
            return []
        self._calls = 0
        seen = self._ledger.seen_rules_ids()
        flagged: list[tuple[Market, RulesVerdict]] = []
        fresh: list[str] = []
        for m in markets:
            if self._calls >= self._cfg.max_calls_per_cycle:
                break
            if m.id in seen or m.closed or m.volume_24h_usd < self._cfg.min_volume_24h_usd:
                continue
            if not m.description.strip():
                fresh.append(m.id)     # nothing to analyze, don't revisit
                continue
            verdict = self.analyze(m)
            if verdict is None:
                continue               # API error: leave unseen so we retry later
            fresh.append(m.id)         # mark seen only after a successful analysis
            if verdict.discrepancy:
                flagged.append((m, verdict))
                alert(f"RULES DISCREPANCY [favors {verdict.favored_side}] "
                      f"{m.question[:60]}\n{verdict.rationale[:200]}\n"
                      f"https://polymarket.com/market/{m.slug}")
        if fresh:
            self._ledger.mark_rules_seen(fresh)
        return flagged
