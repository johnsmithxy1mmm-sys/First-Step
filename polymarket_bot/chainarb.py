"""Strategy: chain (ladder) arbitrage across logically nested sibling markets.

Different from arbitrage.py's neg-risk basket (same event, mutually exclusive
outcomes, Polymarket officially flags it via negRisk): this exploits a
constraint BETWEEN two markets in the same event that describe the same
underlying fact at two different "checkpoints" — a later deadline, or a lower
value threshold. By the WORDING of the two markets (not a probability model),
one YES outcome always implies the other:

  DATE ladder — "X by <date>" (an absorbing fact: once true, stays true).
    "Fed cuts by June" (subset) implies "Fed cuts by July" (superset), because
    June <= July. So YES(later deadline) can never be LESS likely than
    YES(earlier deadline): P(later) >= P(earlier), always.

  VALUE ladder — "reaches >= $V by <date>" (a continuous price path).
    "BTC reaches $200k" (subset) implies "BTC reaches $150k" (superset),
    because a continuous path above $200k necessarily crossed $150k first.
    So YES(lower threshold) can never be LESS likely than YES(higher
    threshold): P(lower $) >= P(higher $), always.

When the market violates the inequality (superset priced BELOW subset), buying
YES(superset) + NO(subset) has a worst-case payout of exactly $1/set and a
best case of $2/set — see ChainPair for the proof. If that costs < $1 net of
fees, the position cannot lose by construction, only by how much it wins.

Honest caveat: unlike neg-risk (an official Polymarket flag), this pairing is
INFERRED from question text (classify_pair below) — a wrong match (unrelated
markets, subtly different resolution rules) turns "riskless" into a real bet.
`classification_haircut` reserves margin against that, and execution is
alert-only (`execute: false`) until paper confirms the matches are sane.

The ladder DIRECTION is read from the deadline in the question TEXT, and it
must agree with Gamma's endDate. Trusting endDate alone inverted real pairs
(Gamma's endDate often lags the wording), which fabricated "+200% arbitrages" —
so on any text-vs-metadata disagreement the pair is refused, not guessed.

Fast turnover: a chain pair does not have to wait for either leg's own
resolution to realize its gain. Once the market corrects the mispricing (or
the near/subset leg resolves — informative for the far leg), both legs can
often be sold back at a profit well before either one's own end date, instead
of sitting dead until a single far-off resolution the way fade.py's tail
fades do.
"""

from __future__ import annotations

import logging
import re
from datetime import date

from pydantic import BaseModel

from .clob import ClobReader, Trader, round_to_tick
from .config import BotConfig
from .fees import FeeModel
from .ledger import Ledger
from .models import Market, simple_estimate
from .portfolio import classify_category

log = logging.getLogger(__name__)

# --- question-text classification (heuristic; same-event restriction is the
# main false-positive guard — see classify_pair) ---

_DIGIT_RE = re.compile(r"\d+")
_MONTH_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\.?\s*\d{0,2}(?:st|nd|rd|th)?,?\s*\d{0,4}\b",
    re.IGNORECASE,
)
_DOLLAR_RE = re.compile(
    r"\$\s?(\d[\d,]*(?:\.\d+)?)\s?(thousand|million|billion|k|m|b)?\b",
    re.IGNORECASE,
)
_MAGNITUDE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6,
              "b": 1e9, "billion": 1e9}
_INCREASING_WORDS = ("reach", "hit", "exceed", "surpass", "top", "cross",
                     "above", "over", "at least")
_DECREASING_WORDS = ("below", "under", "less than", "drop", "fall", "dip")
_ABSORBING_MARKERS = ("by ", "before ", "no later than")

_MONTH_NUM = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
              "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
# "<Month> [<day>,] <year>" — the resolution deadline stated in the wording.
# The 4-digit year is required (so "the 2026 Ballon d'Or" with no month never
# matches, and "2026" is never mis-read as a day); the day is optional.
_DEADLINE_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z.]*\s+"
    r"(?:(\d{1,2})(?:st|nd|rd|th)?,?\s+)?"
    r"(\d{4})\b",
    re.IGNORECASE,
)


def _extract_deadline(market: Market) -> date | None:
    """The deadline parsed from the QUESTION TEXT (the resolution truth), not
    from Gamma's endDate metadata — which for these events often disagrees with
    the text and, trusted alone, inverts the ladder (a fake "arbitrage").

    A month with no day resolves to day 28: enough to order two different
    months correctly, and never overshoots a real month-end.
    """
    m = _DEADLINE_RE.search(market.question)
    if not m:
        return None
    month = _MONTH_NUM[m.group(1).lower()[:3]]
    day = int(m.group(2)) if m.group(2) else 28
    try:
        return date(int(m.group(3)), month, day)
    except ValueError:
        return None


def _template(question: str) -> str:
    """Normalizes a question so siblings compare equal: mask dates/values,
    keep everything else. Two markets are candidate siblings only if their
    templates match exactly."""
    t = question.lower()
    t = _MONTH_RE.sub("#date#", t)
    t = _DOLLAR_RE.sub("#value#", t)
    t = _DIGIT_RE.sub("#", t)
    return " ".join(t.split())


def _extract_dollar(question: str) -> float | None:
    m = _DOLLAR_RE.search(question)
    if not m:
        return None
    return float(m.group(1).replace(",", "")) * _MAGNITUDE.get((m.group(2) or "").lower(), 1.0)


def _increasing_threshold(question: str) -> bool:
    """True only for 'reaches/exceeds $V' wording (not 'stays below $V',
    which has the opposite monotonicity) — required for VALUE ladders."""
    q = question.lower()
    if any(w in q for w in _DECREASING_WORDS):
        return False
    return any(w in q for w in _INCREASING_WORDS)


def _absorbing(question: str) -> bool:
    """True only for 'by/before <deadline>' wording (an absorbing, once-true-
    always-true fact) — required for DATE ladders. Excludes e.g. 'in March',
    which describes a specific window, not a cumulative deadline."""
    q = question.lower()
    return any(marker in q for marker in _ABSORBING_MARKERS)


def classify_pair(a: Market, b: Market) -> tuple[Market, Market, str] | None:
    """(subset, superset, kind) if (a, b) form a valid ladder pair, else None.

    subset's YES implies superset's YES by construction of the wording, so
    superset's true probability can never fall below subset's.
    """
    if a.end_date is None or b.end_date is None or a.id == b.id:
        return None
    if _template(a.question) != _template(b.question):
        return None
    va, vb = _extract_dollar(a.question), _extract_dollar(b.question)
    da, db = _extract_deadline(a), _extract_deadline(b)
    same_date = abs((a.end_date - b.end_date).total_seconds()) < 86_400
    same_value = va is not None and vb is not None and abs(va - vb) < 1e-6

    # Same deadline, different $ threshold -> continuous-path VALUE ladder.
    if same_date and va is not None and vb is not None and not same_value:
        if not _increasing_threshold(a.question):
            return None
        # If both wordings carry a deadline, they must be the SAME date — a
        # value ladder shares the deadline (only the threshold differs).
        if da is not None and db is not None and da != db:
            return None
        subset, superset = (a, b) if va > vb else (b, a)   # higher $ = subset
        return subset, superset, "value"

    # Same (or no) $ threshold, different deadline -> absorbing DATE ladder.
    value_matches = same_value or (va is None and vb is None)
    if value_matches and not same_date:
        if not (_absorbing(a.question) and _absorbing(b.question)):
            return None
        # Order by the TEXT deadline, not endDate. Refuse unless both parse and
        # differ, AND the text order agrees with endDate — a disagreement means
        # one of the two signals is wrong, and "riskless" cannot rest on a guess.
        if da is None or db is None or da == db:
            return None
        if (da < db) != (a.end_date < b.end_date):
            return None
        subset, superset = (a, b) if da < db else (b, a)   # earlier deadline = subset
        return subset, superset, "date"

    return None


class ChainLeg(BaseModel):
    market: Market
    outcome_index: int          # 0 = Yes (superset leg), 1 = No (subset leg)
    token_id: str
    ask: float
    depth: float


class ChainPair(BaseModel):
    event_id: str
    event_title: str
    kind: str                   # "date" | "value"
    subset: ChainLeg             # we buy NO here
    superset: ChainLeg           # we buy YES here
    taker_fee: float = 0.0
    note: str = ""

    @property
    def cost_per_set(self) -> float:
        return self.superset.ask + self.subset.ask

    @property
    def payout_per_set(self) -> float:
        """Worst case across the two possible states (subset can never win
        while superset loses — that state is excluded by construction)."""
        return 1.0

    @property
    def profit_per_set(self) -> float:
        return self.payout_per_set - self.cost_per_set

    @property
    def profit_pct(self) -> float:
        return self.profit_per_set / self.cost_per_set if self.cost_per_set > 0 else 0.0

    @property
    def fee_per_set(self) -> float:
        return self.taker_fee * self.cost_per_set

    @property
    def net_profit_per_set(self) -> float:
        return self.profit_per_set - self.fee_per_set

    @property
    def net_profit_pct(self) -> float:
        return self.net_profit_per_set / self.cost_per_set if self.cost_per_set > 0 else 0.0

    def max_sets_by_depth(self) -> int:
        return int(min(self.superset.depth, self.subset.depth))


class ChainArbitrage:
    def __init__(self, cfg: BotConfig, ledger: Ledger, clob: ClobReader,
                 trader: Trader | None, mode: str):
        self._cfg = cfg.chain_arb
        self._fees = FeeModel(cfg.fees)
        self._ledger = ledger
        self._clob = clob
        self._trader = trader
        self._mode = mode

    # --- selection: cheap Gamma-price prefilter before hitting real books ---

    def prefilter_pairs(self, markets: list[Market]) -> list[tuple[Market, Market, str]]:
        by_event: dict[str, list[Market]] = {}
        for m in markets:
            if (m.event_id and not m.closed and m.enable_order_book
                    and m.outcome_prices and len(m.clob_token_ids) >= 2
                    and m.volume_24h_usd >= self._cfg.min_leg_volume_24h_usd):
                by_event.setdefault(m.event_id, []).append(m)

        found = []
        for group in by_event.values():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    classified = classify_pair(group[i], group[j])
                    if classified is None:
                        continue
                    subset, superset, kind = classified
                    gross = subset.outcome_prices[0] - superset.outcome_prices[0]
                    if gross > self._cfg.prefilter_tolerance:
                        found.append((subset, superset, kind, gross))
        found.sort(key=lambda t: t[3], reverse=True)
        return [(s, sp, k) for s, sp, k, _ in found[: self._cfg.max_events_per_cycle]]

    # --- verification against real order books ---

    def verify(self, subset: Market, superset: Market, kind: str) -> ChainPair | None:
        sup_book = self._clob.order_book(superset.clob_token_ids[0])
        sub_book = self._clob.order_book(subset.clob_token_ids[1])
        if sup_book is None or sub_book is None \
                or sup_book.best_ask <= 0 or sub_book.best_ask <= 0:
            return None
        sup_depth = next((l.size for l in sorted(sup_book.asks, key=lambda x: x.price)
                          if l.size > 0), 0.0)
        sub_depth = next((l.size for l in sorted(sub_book.asks, key=lambda x: x.price)
                          if l.size > 0), 0.0)
        superset_leg = ChainLeg(market=superset, outcome_index=0,
                                token_id=superset.clob_token_ids[0],
                                ask=sup_book.best_ask, depth=sup_depth)
        subset_leg = ChainLeg(market=subset, outcome_index=1,
                              token_id=subset.clob_token_ids[1],
                              ask=sub_book.best_ask, depth=sub_depth)
        category = classify_category(superset.question, superset.category)
        taker_fee = self._fees.taker_fee(category, superset.category)
        pair = ChainPair(
            event_id=superset.event_id, event_title=superset.event_title, kind=kind,
            subset=subset_leg, superset=superset_leg, taker_fee=taker_fee,
            note=f"{subset.question[:60]!r} implies {superset.question[:60]!r}",
        )
        return pair if pair.profit_per_set > 0 else None

    # --- execution ---

    def _legs_look_painted(self, legs) -> bool:
        """Re-fetch each leg's book and refuse if the ask side looks spoofed."""
        from .spoofguard import screen_ask
        for leg in legs:
            book = self._clob.order_book(leg.token_id)
            if book is None:
                log.warning("chain arb: leg %s book vanished before execute",
                            leg.token_id[:16])
                return True
            v = screen_ask(book)
            if v.suspicious:
                log.warning("chain arb: leg %s book looks painted (%s) - skipping",
                            leg.token_id[:16], "; ".join(v.reasons))
                return True
        return False

    def execute(self, pair: ChainPair) -> float:
        """Buys the two-leg set. Returns dollars spent (0 = not executed).

        Leg risk: if one leg's limit order doesn't fill (book moved), the
        other leg is a directional position, not an arbitrage — the same
        honest caveat as arbitrage.py's neg-risk baskets.
        """
        if self._cfg.spoof_screen and self._legs_look_painted(
                [pair.superset, pair.subset]):
            return 0.0
        sets = min(
            pair.max_sets_by_depth(),
            int(self._cfg.max_stake_usd // max(pair.cost_per_set, 1e-9)),
        )
        min_size = max(int(pair.superset.market.min_order_size),
                       int(pair.subset.market.min_order_size))
        if sets < max(self._cfg.min_sets, min_size):
            return 0.0

        spent = 0.0
        for leg in (pair.superset, pair.subset):
            price = round_to_tick(leg.ask, leg.market.tick_size)
            order_id = None
            if self._trader is not None:
                try:
                    resp = self._trader.buy_limit(leg.token_id, price, float(sets),
                                                  neg_risk=leg.market.neg_risk)
                    order_id = (resp or {}).get("orderID")
                except Exception as exc:
                    log.error("chain arb leg failed %s: %s — other leg will not overpay",
                              leg.token_id[:16], exc)
                    continue
            self._ledger.record_trade(
                mode=self._mode,
                estimate=simple_estimate(leg.market, leg.outcome_index, price),
                category="chain_arb", side="BUY", price=price, size=float(sets),
                order_id=order_id,
                status="filled" if self._trader else "sim-filled",
                strategy="chain_arb",
            )
            spent += price * sets
        return spent

    # --- cycle ---

    def check_pair(self, subset: Market, superset: Market, kind: str,
                   allow_execute: bool = True) -> ChainPair | None:
        """Verify ONE classified pair against live books; execute when allowed.

        Shared by the polling cycle and the WS fastlane (which re-checks just
        the pair whose token ticked, instead of waiting for the next poll).
        """
        pair = self.verify(subset, superset, kind)
        # A dust-sized best ask can fake an "edge" nobody can trade;
        # require real depth even for the alert (same idea as arbitrage.py).
        if pair is None or pair.max_sets_by_depth() < self._cfg.min_sets:
            return None
        net_after_haircut = pair.net_profit_pct - self._cfg.classification_haircut
        if net_after_haircut < self._cfg.min_net_edge:
            return None
        log.info("CHAIN ARB [%s] %s: buy YES %s (%.3f) + NO %s (%.3f) -> "
                 "cost $%.4f/set, worst-case payout $1.00, gross +%.2f%%, "
                 "NET after fee+haircut +%.2f%%, depth %d sets",
                 pair.kind, pair.event_title[:40], pair.superset.market.question[:40],
                 pair.superset.ask, pair.subset.market.question[:40], pair.subset.ask,
                 pair.cost_per_set, pair.profit_pct * 100, net_after_haircut * 100,
                 pair.max_sets_by_depth())
        if self._cfg.execute and allow_execute:
            spent = self.execute(pair)
            if spent > 0:
                log.info("chain arb executed: $%.2f", spent)
        return pair

    def cycle(self, markets: list[Market],
              allow_execute: bool = True) -> list[ChainPair]:
        """allow_execute=False (kill-switch / observe-only / breaker): keep
        detecting and alerting — a human can still act — but place no orders."""
        if not self._cfg.enabled:
            return []
        found: list[ChainPair] = []
        for subset, superset, kind in self.prefilter_pairs(markets):
            pair = self.check_pair(subset, superset, kind, allow_execute)
            if pair is not None:
                found.append(pair)
        return found
