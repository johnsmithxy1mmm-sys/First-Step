"""Scanner: first-level filters over all active markets.

The scanner's job is not to find a trade but to quickly drop what cannot be
one: illiquid markets, markets with no book, murky resolution rules,
awkward horizons. Probability estimation is the estimator's job.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .clob import ClobReader
from .config import BotConfig
from .models import Candidate, Market

log = logging.getLogger(__name__)


class Scanner:
    def __init__(self, cfg: BotConfig, clob: ClobReader | None = None):
        self._cfg = cfg
        self._clob = clob

    def first_level_filter(self, markets: list[Market],
                           now: datetime | None = None) -> list[Candidate]:
        """Filters that do not require hitting the order book."""
        s = self._cfg.scanner
        now = now or datetime.now(timezone.utc)
        out: list[Candidate] = []
        for m in markets:
            if m.closed or not m.enable_order_book:
                continue
            if m.volume_24h_usd < s.min_volume_24h_usd:
                continue

            days = m.days_to_resolution(now)
            if days is None or not s.min_days_to_resolution <= days <= s.max_days_to_resolution:
                continue

            q = m.question.lower()
            if s.include_keywords and not any(k.lower() in q for k in s.include_keywords):
                continue
            if any(k.lower() in q for k in s.exclude_keywords):
                continue

            # Ambiguous resolution rules are a source of "guessed right but the
            # market resolved otherwise". Require a source or a clear description.
            if s.require_resolution_clarity and not m.resolution_source:
                if len(m.description.strip()) < s.min_description_chars:
                    continue

            for idx, price in enumerate(m.outcome_prices):
                if not s.price_min <= price <= s.price_max:
                    continue
                if idx >= len(m.clob_token_ids) or not m.clob_token_ids[idx]:
                    continue
                out.append(Candidate(
                    market=m, outcome_index=idx,
                    token_id=m.clob_token_ids[idx], p_mkt=price,
                ))

        # Most liquid first: more realistic to actually fill on them.
        out.sort(key=lambda c: c.market.volume_24h_usd, reverse=True)
        return out[: s.max_candidates_per_cycle]

    def reject_reason(self, m, now=None) -> str | None:
        """None = market has a cheap outcome candidate; else the reject reason (diagnostics)."""
        from datetime import datetime, timezone
        s = self._cfg.scanner
        now = now or datetime.now(timezone.utc)
        if m.closed or not m.enable_order_book:
            return "no book / closed"
        if m.volume_24h_usd < s.min_volume_24h_usd:
            return f"24h volume < ${s.min_volume_24h_usd:,.0f}"
        days = m.days_to_resolution(now)
        if days is None or not s.min_days_to_resolution <= days <= s.max_days_to_resolution:
            return "outside resolution window"
        q = m.question.lower()
        if s.include_keywords and not any(k.lower() in q for k in s.include_keywords):
            return "excluded by include_keywords"
        if any(k.lower() in q for k in s.exclude_keywords):
            return "excluded by exclude_keywords"
        if s.require_resolution_clarity and not m.resolution_source \
                and len(m.description.strip()) < s.min_description_chars:
            return "unclear resolution rules"
        if not any(s.price_min <= p <= s.price_max for p in m.outcome_prices):
            return f"no outcome priced in [{s.price_min}, {s.price_max}]"
        return None

    def verify_depth(self, candidates: list[Candidate]) -> list[Candidate]:
        """Second level: real book depth on our side."""
        s = self._cfg.scanner
        if not s.verify_book_depth or self._clob is None:
            return candidates
        verified: list[Candidate] = []
        for c in candidates:
            book = self._clob.order_book(c.token_id)
            if book is None:
                continue
            depth = book.bid_depth_usd_within(s.book_depth_pct_from_mid)
            if depth < s.min_book_depth_usd:
                continue
            c.book = book
            verified.append(c)
        log.info("scanner: %d candidates after depth check (%d before)",
                 len(verified), len(candidates))
        return verified

    def scan(self, markets: list[Market]) -> list[Candidate]:
        candidates = self.first_level_filter(markets)
        log.info("scanner: %d candidates after first-level filters", len(candidates))
        return self.verify_depth(candidates)
