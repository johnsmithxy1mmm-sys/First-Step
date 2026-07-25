# AUDIT REPORT

Two passes, newest first: the **fee-model correction** (economics), then the
**chain/ladder arbitrage hardening** (correctness).

**Current state:** `411 passed`, all 8 chaos drills PASS, `execute: false`
everywhere, no risk cap or filter threshold weakened.

---

# PASS 2 — Fee model: the price term was missing (CRITICAL, economics)

## Finding

The official Polymarket taker fee is

    fee_usd = theta * shares * price * (1 - price)

where `theta` is the per-category coefficient. The bot applied
`fee = theta * notional` at **every** call site, dropping the `(1 - price)`
factor. The coefficients in config were correct; only their application was
wrong. Overstatement is exactly `1/(1 - price)`:

| price | modeled | actual | overstated |
|---|---|---|---|
| 5c | 0.20c | 0.190c | x1.1 |
| 50c | 2.00c | 1.000c | **x2** |
| 90c | 3.60c | 0.360c | **x10** |
| 95c | 3.80c | 0.190c | **x20** |
| 98.5c | 3.94c | 0.059c | **x67** |

The error is worst on expensive outcomes, i.e. precisely on the near-$1
strategies — which is why their funnels reported zero for days.

### Proof: resolution alpha was arithmetically impossible

It buys in 0.95-0.985 and requires net >= 1% after a 2% dispute haircut.
Under the old model, net was **negative in every category except geopolitics
(theta = 0)** at every price in the band — not "rare", *impossible*. Measured
net edge, before vs after (politics):

| price | net (old model) | net (correct) |
|---|---|---|
| 0.95 | −0.80c ❌ | **+2.81c** ✅ |
| 0.96 | −1.75c ❌ | **+1.85c** ✅ |
| 0.97 | −2.88c ❌ | +0.88c (correctly below gate) |

Above 0.97 it still fails the gate — honestly, because the carry is genuinely
thin against the dispute haircut, not because of a fictional fee.

Chain arb was overstated x1.9-6.0; realistic pairs now clear the 5% gate
(e.g. legs 0.25/0.68 → **+3.78%**, previously rejected).

## Fixes

| # | Fix | Files |
|---|---|---|
| F1 | `FeeModel` rewritten: `taker_fee_per_share` / `maker_rebate_per_share` / `fee_per_share_from_coef` implement `theta*p*(1-p)`. `taker_fee()` → `taker_coef()` so a coefficient can no longer be mistaken for a fee. | `fees.py` |
| F2 | Per-leg fees in both arbs: `fee_per_set = sum(theta*p_i*(1-p_i))`. Model field `taker_fee` → `taker_coef` (the misleading name *was* the root cause). | `arbitrage.py`, `chainarb.py` |
| F3 | Resolution alpha and satellite price their fee at the actual fill price. | `resolution.py`, `satellite.py` |
| F4 | **Maker rebate corrected down**: `maker_rebate_frac: 0.35` → per-category `maker_rebate_share` (default 0.25, crypto 0.20, sports 0.15). This was the one error in the *loss-making* direction: an inflated rebate drove `mm_min_half_spread` to **0.0** for politics — the MM break-even floor was effectively disabled and would have quoted arbitrarily tight. It is now a real floor that rises with price (0.250c at 50c, 0.452c at 95c). | `config.py`, `fees.py`, `marketmaker.py` |
| F5 | Removed config keys now warn instead of being silently ignored by pydantic (a stale `maker_rebate_frac: 0.35` would otherwise look active while doing nothing). | `config.py` |

## Why 400 tests did not catch it

Three tests **asserted the wrong formula** (`0.03 * 0.98`, `0.05 * 0.90`, and a
satellite case whose "no entry" depended on a 7c flat fee). They validated the
code against itself. They are now rewritten to assert the official form, plus
new guards:

- `test_official_worked_example_175_per_100_shares` — 100 shares @50c crypto = **$1.75**, the documented figure, verified end-to-end.
- `test_fee_collapses_near_the_dollar` — pins the x20 ratio at 95c.
- `test_fee_is_symmetric_in_price` — `theta*p*(1-p)` is invariant under p → 1−p.
- `test_maker_rebate_per_category`, `test_net_taker_edge_uses_the_price`, `mm_min_half_spread` rises with price.

**Lesson recorded:** no external constant in this project had a test against its
source. That is the only class of bug 400 self-consistent tests cannot find.

One honest consequence: the `+2%` three-leg basket in
`test_fee_math_is_per_leg_price_scaled` is no longer a *loss* (real fee ~1.98c
vs 2c gross → **+0.02%**). It is still an order of magnitude below the 1.5%
min-edge gate, so it is still rejected — by the gate, not by a fake fee. The
test now asserts exactly that.

---

# PASS 2b — Rate limiter wired (was dead code)

`ratelimit.py` (`TokenBucket`) and `RateLimitConfig` existed but **nothing
imported them** — the bot self-throttled nowhere. Now wired into `clob.py`:

- `ClobReader` throttles book/history reads (degrade to `None`/`[]`).
- `Trader` throttles order placement hard: a dropped order returns `{}`, which
  callers already read as "no fill" — safe, unlike a Cloudflare ban mid-session.
- **`cancel_all` is deliberately never throttled**, and a single `cancel`
  proceeds even with an empty bucket: throttling the call that flattens the book
  is how a safety system kills you. Risk reduction is never queued.
- Throttled *reads* raise `RateLimited` rather than returning an empty list.
  This matters: `open_orders`/`api_positions` feed the desync kill-switch and the
  idempotency reconcile, where `[]` reads as "nothing on the exchange" — it would
  have silently disabled the desync detector and allowed a double entry. Every
  caller already handles an exception conservatively (assume a position exists /
  skip the cycle), so raising fails safe. Covered by
  `test_executor_fails_safe_on_rate_limited_reconcile` and
  `test_kill_switch_cancel_all_is_never_throttled`.

---

# PASS 1 — Chain/Ladder Arbitrage Hardening

**Scope:** `polymarket_bot/chainarb.py` (flagship chain-arb strategy) and its
config surface. Focused batch of the four genuinely-open items from the
senior-quant / security audit. The rest of the audit's findings (date-ladder
inversion, alert dedup, value-ladder direction) were already resolved in earlier
work and are re-verified below with a proof, not re-implemented.

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
python -m pytest polymarket_bot/tests -q      # 411 passed
python -m polymarket_bot.main --mode chaos    # 8/8 drills PASS
python -m polymarket_bot.main --mode sync-config   # config layering intact
```

---

## Still open (not bugs — measurement)

The fee fix unblocks the near-$1 desk, but **no strategy has been validated
against live money yet**: `market_maker.enabled = false` and every strategy is
`execute: false`, so realized P&L is $0 by construction. The next step is
evidence, not code: run paper with MM enabled and read `markouts` +
the opportunity ledger to get the round-trip frequency, which is the one
unknown that determines whether the capital scale is worth raising.
