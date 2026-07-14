"""Strategy #5: informational edge in a niche.

Historically, private winners on prediction markets know a narrow domain
deeper than the crowd. You can't automate someone else's expertise, but you
can remove the lag between "a market appeared in my niche" and "I saw it":
this module watches for new markets on niche watchlists and alerts
immediately — from there a human who reads primary sources faster than the
Western crowd makes the call.

Strategy #4 (rules lawyering) is supported here too: the alert includes the
resolution rules so any gap between the headline and the letter of the rules
is visible at once.
"""

from __future__ import annotations

import logging
import re

from .config import BotConfig
from .ledger import Ledger
from .models import Market
from .monitor import alert

log = logging.getLogger(__name__)


class NicheWatcher:
    def __init__(self, cfg: BotConfig, ledger: Ledger):
        self._cfg = cfg.niche
        self._ledger = ledger
        # Match keywords on word boundaries: otherwise "eth" catches "Hegseth".
        self._patterns: list[tuple[str, re.Pattern]] = []
        for watchlist in self._cfg.watchlists:
            keywords = [k.lower() for k in watchlist.keywords if k.strip()]
            if not keywords:
                continue
            joined = "|".join(re.escape(k) for k in keywords)
            self._patterns.append((
                watchlist.name,
                re.compile(rf"(?<![a-z0-9])(?:{joined})(?![a-z0-9])"),
            ))

    def classify(self, text: str) -> str | None:
        """Niche name for arbitrary text (also used by the smart-money tracker)."""
        low = text.lower()
        for name, pattern in self._patterns:
            if pattern.search(low):
                return name
        return None

    def _match(self, market: Market) -> str | None:
        return self.classify(f"{market.question} {market.event_title}")

    def _alert_market(self, name: str, m: Market) -> None:
        days = m.days_to_resolution()
        days_text = f"{days:.0f}d" if days is not None else "date not set"
        text = (
            f"NEW MARKET IN NICHE [{name}]\n"
            f"{m.question}\n"
            f"Yes price: {m.outcome_prices[0] if m.outcome_prices else '?'} | "
            f"24h volume: ${m.volume_24h_usd:,.0f} | "
            f"to resolution: {days_text}\n"
            f"Resolution source: {m.resolution_source or 'NOT SET'}\n"
            f"Rules: {m.description[:500]}\n"
            f"https://polymarket.com/market/{m.slug}"
        )
        log.info(text)
        alert(text)

    def cycle(self, markets: list[Market]) -> list[tuple[str, Market]]:
        """New markets in niches: alert once per market."""
        if not self._cfg.enabled or not self._patterns:
            return []
        seen = self._ledger.seen_market_ids()
        hits: list[tuple[str, Market]] = []
        fresh_ids: list[str] = []

        for m in markets:
            if m.id in seen or m.closed:
                continue
            fresh_ids.append(m.id)
            name = self._match(m)
            if name is None:
                continue
            hits.append((name, m))
            try:
                self._alert_market(name, m)
            except Exception:
                # One malformed market must not break the cycle or the "seen" mark.
                log.exception("niche: alert for market %s", m.id)

        # Mark all new markets (not just niche ones) so we don't rescan them.
        if fresh_ids:
            self._ledger.mark_markets_seen(fresh_ids)
        return hits
