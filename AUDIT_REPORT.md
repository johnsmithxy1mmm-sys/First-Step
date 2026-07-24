# AUDIT REPORT — Chain/Ladder Arbitrage Hardening

**Scope:** `polymarket_bot/chainarb.py` (flagship chain-arb strategy) and its
config surface. Focused batch of the four genuinely-open items from the
senior-quant / security audit. The rest of the audit's findings (date-ladder
inversion, alert dedup, value-ladder direction) were already resolved in earlier
work and are re-verified below with a proof, not re-implemented.

**Result:** `400 passed` (full suite), all chaos drills PASS, execution stays
`execute: false` by default. No risk cap or filter threshold was weakened.

---

## Re-verified (already fixed — proven, not re-touched)

### BUG-1 — Date-ladder direction (CRITICAL) — VERIFIED CORRECT
The bot buys **YES(superset) + NO(subset)**, where `subset` is the *earlier*
deadline (rarer) and `superset` the *later* deadline (likelier). Because
`subset YES ⟹ superset YES`, the state `(subset YES, superset NO)` is
impossible. Enumerating the three possible states:

| subset (early) | superset (late) | YES(sup)+NO(sub) |
|:--------------:|:---------------:|:----------------:|
| YES            | YES             | 1 + 0 = **$1**   |
| NO             | YES             | 1 + 1 = **$2**   |
| NO             | NO              | 0 + 1 = **$1**   |
| YES            | NO              | *impossible*     |

Worst-case **$1**, best **$2** → riskless iff cost < $1 net of fees. This is
now nailed down by an exhaustive payoff-matrix test that *also* proves the
inverted construction has a $0 state, so any regression that flips the direction
fails loudly. See `tests/test_chainarb_payoff.py`.

### BUG-2 — Alert dedup — VERIFIED DONE
Dedup by key + cooldown + net-edge materiality is already in place (alert
hygiene work). No change.

### BUG-3 — Value-ladder direction — VERIFIED CORRECT
Higher `$` bar = rarer = subset; requires increasing wording ("reach/exceed/…").
Direction is asserted in `test_value_ladder_payoff_matrix`. The one real gap
(negation) is closed below.

---

## Fixed in this batch

### F1 — Negation flips monotonicity (HIGH) — `chainarb.py:86,177-181`
**Root cause:** `classify_pair` inferred ladder direction from the date/value
token alone. A negated predicate ("X will **not** happen by July") is implied by
"**not** by September" — the *opposite* direction — so the inferred subset⊂superset
relation is backwards and the "riskless" set can pay $0.
**Fix:** added `_NEGATION_WORDS` + `_negated()`; `classify_pair` returns `None`
if either question is negated — refuse rather than guess.
**Test:** `test_negation_wording_refuses_pair` (test_chainarb.py).

### F2 — Implausible-edge guard (HIGH) — `chainarb.py:403-405,455-464`; config
**Root cause:** a near-riskless ladder pays a few % net, not tens. A double-digit
NET is a *symptom* that the payout matrix does not hold here (incomplete book,
mis-paired legs, a non-monotone relation the text heuristic missed) — exactly the
case where auto-executing loses money.
**Fix:** `check_pair` sets `pair.implausible = net_after_haircut > implausible_net`
(default **0.50**, in `ChainArbConfig` + `config.example.yaml`); the log tags it
`IMPLAUSIBLE — verify payout matrix`; execution is gated
`... and not pair.implausible`; `execute()` refuses an implausible pair up front.
This is a *ceiling*, not a loosened floor — it only ever blocks trades.
**Test:** `test_implausible_net_flagged_and_not_executed`.

### F3 — Leg risk handled, not just noted (HIGH) — `chainarb.py:349-395,406-431`
**Root cause:** the old `execute()` placed each leg as a plain limit and, on a
non-fill, merely logged "other leg will not overpay" — leaving a *silent
directional position* that is no longer an arbitrage.
**Fix:**
- `_buy_leg()` submits each leg **FOK** (fill-or-kill): a leg fills fully or is
  killed, never half-filled.
- If a leg is killed after an earlier leg filled, `execute()` calls
  `_unwind_leg()` on everything already filled — crossing to the bid (FOK) to
  turn the accidental directional position straight back into cash — and returns
  `0.0`. If no bid exists to unwind against, it logs an ERROR telling the
  operator to intervene rather than pretending success.
**Test:** `test_leg_unwind_when_second_leg_killed`.

### F4 — Exhaustive payoff-matrix tests (MEDIUM) — `tests/test_chainarb_payoff.py` (new)
**Root cause:** the direction correctness rested on reasoning, not a test; a
future refactor could silently re-invert a ladder.
**Fix:** new property/enumeration suite that walks **every** resolution state of
both ladder kinds and both monotonicity directions, asserts OURS is bounded
`[$1, $2]` over the possible states, asserts the INVERTED construction has a $0
state, and asserts the impossible state is excluded — so a direction flip breaks
the build. Verified against a real `ChainPair` built from real `ChainLeg` legs
(the exact legs the bot submits: YES on outcome 0, NO on outcome 1).

---

## Invariants preserved (audit standing constraints)
- **No weakened caps.** `implausible_net` is a new *upper* bound that only blocks
  trades; no floor/threshold was relaxed. A 0-signal funnel is still treated as
  information, not a bug.
- **execute: false** remains the default for chain-arb.
- **Kill-switch** untouched; all chaos drills still PASS.
- **Secrets** unchanged — nothing new logged, committed, or sent to Telegram.

## Verification
```
python -m pytest polymarket_bot/tests -q     # 400 passed
python -m polymarket_bot.main --mode chaos    # all drills PASS
```
