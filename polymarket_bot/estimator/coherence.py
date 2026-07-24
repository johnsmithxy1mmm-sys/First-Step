"""Cross-market coherence — priority #1 among signals.

Logical links between markets give a structural edge independent of anyone's
opinions:

1. Neg-risk baskets. In an event with mutually exclusive outcomes (elections
   etc.) the sum of Yes prices across markets must be ~1. If S = sum(p) != 1,
   every price is systematically skewed: the candidate's fair probability is
   ~ p_mkt / S. When S < 1 the basket is collectively underpriced — pure edge.

2. Calendar chains. "Event by March 31" logically implies "by June 30", so
   P(by earlier date) <= P(by later date). If the market with the later date
   is CHEAPER than the earlier one — a monotonicity violation: the later one
   is underpriced at least to the earlier one's price.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from ..models import Candidate, Market, Signal

log = logging.getLogger(__name__)

# Threshold below which a basket-sum deviation is treated as spread noise.
BASKET_TOLERANCE = 0.02

_DATE_NOISE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b"
    r"|\b\d{1,4}(st|nd|rd|th)?\b|[,.?]",
    re.IGNORECASE,
)


def normalize_question(question: str) -> str:
    """Strips dates/numbers: "X by March 31?" and "X by June 30?" give one key."""
    stripped = _DATE_NOISE.sub(" ", question.lower())
    return " ".join(stripped.split())


class CoherenceSignal:
    name = "coherence"

    def __init__(self, markets: list[Market]):
        # Baskets of neg-risk events: sum of Yes prices per event.
        self._basket_sum: dict[str, float] = {}
        self._basket_size: dict[str, int] = {}
        by_event: dict[str, list[Market]] = defaultdict(list)
        for m in markets:
            if m.event_neg_risk and m.event_id and m.outcome_prices:
                by_event[m.event_id].append(m)
        for event_id, group in by_event.items():
            if len(group) >= 2:
                self._basket_sum[event_id] = sum(m.outcome_prices[0] for m in group)
                self._basket_size[event_id] = len(group)

        # Calendar chains: markets with the same normalized question.
        self._chains: dict[str, list[Market]] = defaultdict(list)
        for m in markets:
            if m.end_date is not None and m.outcome_prices:
                self._chains[normalize_question(m.question)].append(m)

    def evaluate(self, candidate: Candidate) -> Signal | None:
        return self._basket_check(candidate) or self._calendar_check(candidate)

    def _basket_check(self, candidate: Candidate) -> Signal | None:
        if candidate.outcome_index != 0:
            return None
        s = self._basket_sum.get(candidate.market.event_id)
        if s is None or s <= 0 or abs(s - 1.0) <= BASKET_TOLERANCE:
            return None
        p_fair = min(candidate.p_mkt / s, 0.999)
        # An underpriced basket (S < 1) is structural arbitrage, max confidence.
        confidence = 0.9 if s < 1.0 else 0.7
        return Signal(
            name=self.name,
            p_est=p_fair,
            confidence=confidence,
            rationale=f"neg-risk basket {candidate.market.event_title[:50]}: "
                      f"sum(p)={s:.3f} over {self._basket_size[candidate.market.event_id]} outcomes "
                      f"-> fair={p_fair:.4f}",
        )

    def _calendar_check(self, candidate: Candidate) -> Signal | None:
        """Monotonicity boost only from a VERIFIED earlier sibling.

        Two hard lessons enforced (both fabricated real edges before):
        the deadline ORDER comes from the question TEXT and must agree with
        endDate metadata (Gamma's endDate contradicts the wording on real
        events — trusting it alone inverted chain-arb pairs); and a differing
        $ threshold breaks the implication entirely (normalize_question strips
        numbers, so "$150k by June" and "$200k by December" share a chain key
        without this check)."""
        if candidate.outcome_index != 0:
            return None
        m = candidate.market
        chain = self._chains.get(normalize_question(m.question), [])
        if len(chain) < 2 or m.end_date is None:
            return None
        from ..chainarb import _absorbing, _extract_deadline, _extract_dollar
        if not _absorbing(m.question):
            return None                 # "in June" is a window, not a deadline
        my_deadline = _extract_deadline(m)
        my_value = _extract_dollar(m.question)
        if my_deadline is None:
            return None
        earlier_max = None
        for o in chain:
            if o.id == m.id or o.end_date is None or not _absorbing(o.question):
                continue
            o_value = _extract_dollar(o.question)
            if (my_value is None) != (o_value is None) or \
                    (my_value is not None and abs(my_value - o_value) > 1e-6):
                continue                # different $ threshold = no implication
            o_deadline = _extract_deadline(o)
            if o_deadline is None or o_deadline >= my_deadline:
                continue                # need a strictly EARLIER text deadline
            if (o.end_date < m.end_date) != (o_deadline < my_deadline):
                continue                # text vs metadata disagree -> refuse
            price = o.outcome_prices[0]
            if earlier_max is None or price > earlier_max:
                earlier_max = price
        if earlier_max is None or earlier_max <= candidate.p_mkt + BASKET_TOLERANCE:
            return None
        # Monotonicity violation: ours (later) must cost >= the earlier one.
        return Signal(
            name=self.name,
            p_est=min(earlier_max, 0.999),
            confidence=0.85,
            rationale=f"calendar chain: earlier deadline trades at {earlier_max:.3f}, "
                      f"ours (later, {m.end_date.date()}) at {candidate.p_mkt:.3f} — "
                      f"violates P(early) <= P(late)",
        )
