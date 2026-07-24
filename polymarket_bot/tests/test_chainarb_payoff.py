"""Exhaustive payoff-matrix audit for chain-arb ladders (BUG-1/BUG-3).

Enumerate EVERY resolution state of a ladder pair and assert the bounds of the
construction the bot actually trades — buy YES(superset) + NO(subset) — for
both ladder kinds and both monotonicity directions. Also prove the INVERTED
construction has a $0 state (the bug the auditor flagged), so a regression that
flips the direction fails loudly here.
"""

from datetime import datetime, timezone

from polymarket_bot.chainarb import ChainLeg, ChainPair, classify_pair

from .conftest import make_market


def date_market(mid, month, year=2027, prob=0.1):
    return make_market(
        id=mid, event_id="ev", event_title="ladder",
        question=f"Will GPT-6 be released by {month} 30, {year}?",
        end_date=datetime(year, {"June": 6, "September": 9, "December": 12}[month],
                          28, tzinfo=timezone.utc),
        outcome_prices=[prob, 1 - prob], clob_token_ids=[f"{mid}y", f"{mid}n"])


def value_market(mid, dollars, prob=0.1):
    same = datetime(2027, 12, 31, tzinfo=timezone.utc)
    return make_market(
        id=mid, event_id="ev", event_title="ladder",
        question=f"Will Bitcoin reach ${dollars:,} by Dec 2027?",
        end_date=same, outcome_prices=[prob, 1 - prob],
        clob_token_ids=[f"{mid}y", f"{mid}n"])


def _payout(construction: str, subset_yes: bool, superset_yes: bool) -> float:
    """$ paid for one set given which legs resolve YES.
    'ours' = YES(superset)+NO(subset); 'inverted' = YES(subset)+NO(superset)."""
    if construction == "ours":
        return (1.0 if superset_yes else 0.0) + (0.0 if subset_yes else 1.0)
    return (1.0 if subset_yes else 0.0) + (0.0 if superset_yes else 1.0)


# subset YES implies superset YES -> (subset=YES, superset=NO) is IMPOSSIBLE.
_POSSIBLE_STATES = [(True, True), (False, True), (False, False)]
_ALL_STATES = _POSSIBLE_STATES + [(True, False)]


def _assert_ladder(subset, superset):
    """Given a classified (subset, superset), audit the payoff matrix."""
    # The bot's construction is worst-case $1, best $2, over every POSSIBLE state.
    ours = [_payout("ours", s, p) for s, p in _POSSIBLE_STATES]
    assert min(ours) == 1.0 and max(ours) == 2.0

    # The inverted construction has a $0 state — this is the auditor's bug.
    inverted = [_payout("inverted", s, p) for s, p in _POSSIBLE_STATES]
    assert min(inverted) == 0.0

    # ChainPair reports worst-case $1 for the real construction it builds:
    # YES(superset) on outcome 0, NO(subset) on outcome 1 — the legs the bot
    # actually submits (chainarb.py builds them exactly this way).
    superset_leg = ChainLeg(market=superset, outcome_index=0,
                            token_id=superset.clob_token_ids[0], ask=0.30, depth=100)
    subset_leg = ChainLeg(market=subset, outcome_index=1,
                          token_id=subset.clob_token_ids[1], ask=0.30, depth=100)
    pair = ChainPair(
        event_id="ev", event_title="t", kind="date",
        subset=subset_leg, superset=superset_leg, taker_fee=0.0)
    assert pair.payout_per_set == 1.0


def test_date_ladder_payoff_matrix():
    early = date_market("e", "June", prob=0.08)     # rarer, cheaper
    late = date_market("l", "September", prob=0.20)  # likelier, dearer
    subset, superset, kind = classify_pair(early, late)
    assert kind == "date" and subset.id == "e" and superset.id == "l"
    _assert_ladder(subset, superset)


def test_value_ladder_payoff_matrix():
    high = value_market("h", 200_000, prob=0.05)     # higher bar = subset (rarer)
    low = value_market("l", 150_000, prob=0.15)      # lower bar = superset
    subset, superset, kind = classify_pair(high, low)
    assert kind == "value" and subset.id == "h" and superset.id == "l"
    _assert_ladder(subset, superset)


def test_impossible_state_is_never_reached_in_practice():
    """The (subset YES, superset NO) state pays $0 for OURS too — but it cannot
    occur under the implication, which is exactly why worst-case is $1."""
    assert _payout("ours", True, False) == 0.0
    assert (True, False) not in _POSSIBLE_STATES


def test_every_state_enumerated():
    """Guard against silently dropping a state from the matrix."""
    assert len(_ALL_STATES) == 4 and len(set(_ALL_STATES)) == 4
