# Findings Register — audit/2026-07-26

**STATUS: ALL FIXED.** 18 reproducers now PASS and are the permanent
regression suite; mutation score on `risk.py` went 3/7 -> 7/7 killed. Run:

    AUDIT_REPRO=1 python -m pytest polymarket_bot/tests/audit -q

Sorted by (probability in prod) × (irreversibility).

| ID | Severity | Conf. | Class | Location | Repro | Status |
|---|---|---|---|---|---|---|
| F-003 | **Critical** | high | data-loss / safety | `ledger.py:186` `record_trade` (no validation) → `risk.py:86,98` | `test_f003_*` (5 fail) | FIXED |
| F-004 | **Critical** | high | logic / api-contract | `executor.py:196-221` `execute_sell` | `test_f004_*` (2 fail) | FIXED |
| F-005 | High | high | logic / money | `arbitrage.py:218-238` GTC legs | `test_f004_*` (2 fail) | FIXED |
| F-002 | High | high | api-contract / risk | `risk.py:104` (0 callers) vs `portfolio.py:204` | `test_f002_*` (2 fail) | FIXED |
| F-001 | High | high | numeric / trust-boundary | `ws_feed.py:56-95` `BookStore.handle` | `test_f001_*` (3 fail) | FIXED |
| F-007 | High | high | availability / safety | `main.py:781` reconcile scope | `test_f007_*` (3 fail) | FIXED |
| F-009 | High | high | test-quality | `risk.py` (whole module) | mutation: 3/7 killed | FIXED |
| F-006 | High | high | logic / money | `resolution.py:129-141`, `chainarb.py:360` | `test_f004_*` (1 fail) | FIXED |
| F-008 | Medium | high | data-integrity | `ledger.py:280-285` floor | `test_f008_*` (2 fail) | FIXED |
| F-010 | Medium | medium | idempotency | `clob.py:155` no `client_order_id` | repro + fix | FIXED |
| F-011 | Medium | low | time | `ws_feed.py:206` wall clock | repro + fix | FIXED |
| F-012 | Low | high | test-gap | `ws_feed.py:77` unknown-side fix | test added | FIXED |
| F-014 | High | high | numeric / trust-boundary | `models.py:_num/from_gamma` (never fuzzed) | `test_gamma_boundary.py` (20) | FIXED |
| F-015 | Medium | high | numeric | `clob.py:round_to_tick` tick >= 1 inverts the clamp | property test | FIXED |
| F-016 | High | high | numeric / trust-boundary | LLM cache bypasses the response schema -> unbounded `Signal.p_est` -> Kelly 520% | `test_llm_boundary.py` (21) | FIXED |
| F-017 | Medium | high | numeric | denormal tick (2.2e-313) overflows `round()` | property test | FIXED |
| F-018 | High | high | availability / safety | `telegram_control._advance_offset` wedges the control channel forever | `test_telegram_boundary.py` (24) | FIXED |
| F-019 | Medium | high | robustness | `handle_update` raises on any non-object field | same file | FIXED |
| F-020 | **Critical** | high | logic / money | `portfolio.exit_plan` take_profit_multiple=7.0 is unreachable above price 1.0 -> fade legs had NO exit | `test_fade_shape.py` (21) | FIXED |
| F-021 | High | high | risk-shape | no filter looked at (max gain / max loss); a leg needing 99 wins to repay one loss passed every gate | `test_fade_shape.py`, `shape.feature` | FIXED |
| F-022 | High | high | logic / capital | `_irr_score` ranked by edge-per-day but never rejected, so the ranking was inert while caps were slack | `test_fade_shape.py` | FIXED |
| F-023 | High | high | measurement | `_paper_fills` sampled only at the MM cycle; crossings between cycles were invisible -> 0 fills, 0 markouts in a multi-day run | `test_measurement.py` (16) | FIXED |
| F-024 | Medium | high | capital allocation | `size_usd` gave total exposure first-come-first-served: fade held $1,891, MM $0 | `test_measurement.py` | FIXED |
| F-025 | Medium | high | observability | the report showed `avg edge 1.00` and could not distinguish 'no mispricing' from 'the estimator echoes the market' | `test_measurement.py` | FIXED |

---

## F-003 — One un-validated ledger row disables every halt

```
Severity: Critical | Confidence: high | Class: data-loss / safety
Location: polymarket_bot/ledger.py:186 (record_trade) -> risk.py:86,98,105
Invariant: INV-11 (row sanity), INV-12 (halt reachability)
Trigger: any caller passing size=inf / non-finite / negative (e.g. size computed
         from a poisoned book price — see F-001 — or a mis-read fill quantity)
Repro: polymarket_bot/tests/audit/test_f003_*.py  (5 failing)
```
`record_trade` writes whatever it is handed. `size=inf` makes `total_exposure()`
and `realized_pnl()` return NaN. Every kill-switch decision is a `>=`
comparison, and **every comparison against NaN is False**:

| check | with NaN | effect |
|---|---|---|
| `loss >= max_daily_loss_usd` | False | daily stop never fires |
| `(hwm-eq)/hwm >= max_drawdown_pct` | False | drawdown halt never fires |
| `exposure >= max_global_exposure_usd` | False | entry gate never blocks |

Silent: no exception, no alert, no log. **Strictly worse than a crash** — a crash
stops trading. Blast radius: sizing, PnL, allocator, compounding, digest.
Fix direction: validate at the ledger boundary (reject non-finite/negative size,
price ∉ (0,1), usd ≉ price·size) **and** make the kill-switch fail-closed on
non-finite equity. Regression test: both halves.

## F-004 — The exit path books sales that never happened

```
Severity: Critical | Confidence: high | Class: logic / api-contract
Location: polymarket_bot/executor.py:196-221
Invariant: INV-10 (every recorded fill is a real fill)
Trigger: any take-profit / guardian exit whose GTC sell does not fill immediately
Repro: test_f004_phantom_fills.py::test_execute_sell_* (2 failing)
```
`execute_sell` places a **GTC** sell and records `status="filled"` for the full
size with no `_wait_fill`. Asymmetry: the BUY path has fill waiting, repricing
and partial handling; its mirror has none. Consequences: phantom realized PnL;
the position leaves `open_positions`, so guardian and take-profit **stop watching
a position still held**; the resting order becomes an F-007 ghost. This is the
worst-placed instance because exits are the risk-reduction path.

## F-005 — Basket arbitrage legs rest as GTC and are booked as complete

```
Severity: High | Confidence: high | Class: logic / money
Location: polymarket_bot/arbitrage.py:218-238  (arbitrage.enabled = true by default)
```
Legs go out with the platform default (GTC) — `buy_limit(...)` with no
`order_type` — and each is recorded filled on an order id alone. An unfilled leg
turns a "riskless" basket into a directional bet that the ledger reports as
complete. `chainarb` was converted to FOK + unwind for exactly this reason;
`arbitrage.py` was not. On a leg exception the loop `continue`s and still records
the others.

## F-002 — `risk.max_global_exposure_usd` is enforced nowhere

```
Severity: High | Confidence: high | Class: api-contract / risk
Location: risk.py:104 check_global_exposure (ZERO callers) vs portfolio.py:204
```
Two different global caps exist: the `$` one a user reads as the hard ceiling
(unused), and `portfolio.max_total_exposure_pct * bankroll` = $1500 (5× larger,
longshot/fade only). MM/sprint are bounded per market only: 5 × $50 × 2 sides =
**$500 of MM inventory against a configured $300 cap**, with nothing objecting.
Tightening the documented knob has no effect on the strategy holding the capital.

## F-001 — WS trust boundary accepts NaN / Infinity prices

```
Severity: High | Confidence: high | Class: numeric / trust-boundary
Location: ws_feed.py:56-95, 97-110
```
`json.loads` admits bare `NaN`/`Infinity`; `BookStore` does no numeric checks.
(a) NaN price keys are **unremovable** (per-object hash ⇒ the `size:0` delete
never matches) → unbounded growth from a repeating malformed feed.
(b) `max(bids)` with NaN present is insertion-order dependent ⇒ `top.bid` can
become NaN. (c) `Infinity` passes the `top.bid > 0` gate in `on_tick` and becomes
`mark` for `_exit_one` — a fabricated take-profit, and via F-003 a NaN ledger row.

## F-007 — Reconcile only knows MM/sprint orders → false terminal HALT

```
Severity: High | Confidence: high | Class: availability / safety
Location: main.py:781 (local set) vs executor.py:138,209 / arbitrage.py:224 (GTC)
```
`reconcile` HALTs on `exchange_ids - local_ids`, but the local set is
`mm | sprint` only. Every resting executor entry, every exit order and every
basket leg is a "ghost". The normal state between placement and fill trips a
halt that needs manual restart — and the first false HALT trains the operator to
distrust the real signal.

## F-009 — The safety module is the least-tested module (mutation evidence)

```
Severity: High | Confidence: high | Class: test-quality
Method: manual mutation (7 mutants, targeted): 3 killed, 4 SURVIVED
```
| mutant | result |
|---|---|
| `trading_allowed` drops the pause term | **SURVIVED** |
| daily-loss `>=` → `>` | **SURVIVED** |
| drawdown `>=` → `>` | **SURVIVED** |
| pause dedupe `or` → `and` | **SURVIVED** |
| reconcile direction flipped | killed |
| fee drops the price term | killed |
| rebate share ignored | killed |

The survivor that matters: nothing asserts **a pause actually stops trading**.
The chaos drill checks `paused == True`, never `trading_allowed == False`. By
contrast the fee/rebate mutants die — the modules hardened earlier this session
are genuinely covered, `risk.py` is not.

## F-006 — A killed FOK is recorded as a full fill

`resolution.py:129-141` and `chainarb.py:360` treat "an order id came back" as
"fully filled"; neither reads `size_matched`. A killed FOK still returns a
response.

## F-008 — Selling what was never owned is accepted and hidden

Hypothesis counterexample (INV-24): one `SELL` with no prior `BUY` is written,
and `open_positions` floors it to 0 (`max(size - sold, 0.0)`). The floor turns an
impossible state into a plausible zero, hiding duplicated or over-sized exits.

## SUSPECTED (no repro — do not act on these yet)

* **F-010** no `client_order_id` anywhere: a duplicated exchange response or a
  timeout-then-landed order cannot be de-duplicated. Needs an SDK-level repro.
* **F-011** staleness uses wall clock (`time.time`), not `monotonic`: a backwards
  NTP step could mask an outage window. Needs a clock-injection repro.

## NITS (style / not bugs)

* `main.py` 1150 lines mixes orchestration, jobs and helpers.
* `est_to_plan` lives in `main.py` while its collaborators live in `portfolio.py`.
* `BookStore._bids/_asks` accessed directly by tests (private reach-in).


## F-014 — The Gamma REST feed was never fuzzed (the other half of F-001)

```
Severity: High | Confidence: high | Class: numeric / trust-boundary
Location: polymarket_bot/models.py (_num, _normalize_spread, from_gamma)
Method: hand-built hostile payloads — all 9 accepted before the fix
```
F-001 hardened the websocket book against NaN/Infinity. The REST parser — the
OTHER source of every price and tick size — kept accepting whatever it was
handed: NaN and Infinity prices, prices outside [0,1], negative/NaN tick sizes,
negative order minimums, infinite volumes.

The sharpest consequence is in the order path, because `orderPriceMinTickSize`
flows straight into `clob.round_to_tick`:

| poisoned tick | effect before the fix |
|---|---|
| `NaN` | `round(price / tick)` **raises ValueError**, killing the strategy job mid-execution |
| negative | falls into the `tick <= 0` early return, which hands back the price **UNCLAMPED** — silently re-opening the 0/1 order-price hole that clamp exists to close |

Fix: non-finite numerics are treated as ABSENT at the boundary (`_num`), prices
outside a finite [0,1] reject the whole market row, and sizes/ticks fall back to
their defaults instead of being trusted (`_positive`). `round_to_tick` also
became defensive rather than relying on its callers.

## F-015 — A tick size >= 1 inverts the order-price clamp

```
Severity: Medium | Confidence: high | Class: numeric
Location: polymarket_bot/clob.py round_to_tick
Found by: the property test written for F-014, not by reading the code
```
`min(max(snapped, tick), 1 - tick)` assumes `tick < 1`. At `tick = 2.0` the
upper bound `1 - tick` is **-1.0**, so the clamp returns a NEGATIVE order price
(`round_to_tick(0.0, 2.0) == -1.0`). A tick is a price increment on a 0..1
probability, so only `(0, 1)` is meaningful; that is now enforced in both the
parser and the function.


## F-016 — The LLM cache bypassed the response schema, and p_est sizes positions

```
Severity: High | Confidence: high | Class: numeric / trust-boundary
Location: estimator/llm.py `_cached`, models.py `Signal`
```
Live responses go through `messages.parse` with a pydantic schema, so p_est and
confidence are guaranteed probabilities. The on-disk cache did not: a plain JSON
file read with `json.loads` (which accepts bare NaN/Infinity), stale,
hand-editable, truncatable by a crash mid-write.

`Signal.p_est` was an unbounded `float | None`, so a poisoned entry flowed
straight into sizing:

    cache p_est=5.0 -> Signal(p_est=5.0) -> combine -> Estimate.p_est
                    -> portfolio.size_usd -> kelly_fraction(5.0, 0.05) = 5.21
                    -> 521% of the Kelly base

Two layered defences: `Signal` now bounds p_est/confidence at the TYPE level
(protecting every signal source, not just the LLM), and `_cached` re-validates
entries, dropping a poisoned one with a warning instead of raising deep inside a
strategy.

Notable: five test files passed `confidence=1e9` to overpower the market anchor
— an out-of-contract value (confidence is documented as a 0..1 weight) that
worked only because nothing enforced the range. Those tests now use legal values
or build the Estimate directly.

## F-017 — A denormal tick size overflows int conversion

```
Severity: Medium | Confidence: high | Class: numeric
Found by: the property test from F-014 (third finding it produced)
```
`2.2250738585e-313` is finite and inside `(0, 1)`, so it passed every guard —
but `price / tick` is then ~4.5e312 and `round()` raises OverflowError on int
conversion. Real venue ticks are 0.001-0.01; `MIN_TICK = 1e-6` is now enforced
in both the parser and `round_to_tick`.


## F-018 — One unreadable update_id wedged the operator's remote kill, forever

```
Severity: High | Confidence: high | Class: availability / safety
Location: polymarket_bot/telegram_control.py `_advance_offset`
Repro: test_control_channel_survives_a_poisoned_batch_end_to_end
```
Telegram replays every update until the offset moves past it. `int("abc")`
raised, `run()`'s broad `except Exception` swallowed it, the offset never
advanced — so the next poll returned the SAME poisoned batch, for ever. Proven
end-to-end: after five polls the offset was still 0 and no command had been
dispatched.

Two things make this worse than a stuck poller:

* Commands **after** the poisoned entry in the batch are never dispatched, so
  `/pause` — the operator's remote kill-switch — becomes unreachable while the
  bot keeps trading.
* `_advance_offset` runs BEFORE the chat-id check in `handle_update`, so it
  processes payloads from **anyone** who messages the bot, not just the owner.

Fixed by making the function always make forward progress: unreadable ids are
skipped with a warning, and a batch with no readable id at all still advances
(replaying an unreadable batch for ever is strictly worse than losing it).
`int(inf)` raises OverflowError rather than ValueError — that gap in the first
version of the fix was caught by this module's own property test.

## F-019 — Any non-object field raised instead of being ignored

```
Severity: Medium | Confidence: high | Class: robustness
Location: polymarket_bot/telegram_control.py `handle_update`
```
`.get()` was called on whatever sat in `message`/`chat` and `.strip()` on
whatever sat in `text`, so a non-object raised AttributeError and cost the rest
of the batch. Every field is now shape-checked; a malformed update is a silent
non-answer. A Hypothesis property asserts that no JSON payload of any shape can
make either entry point raise.

Mutation coverage added for this module, including **"AUTH REMOVED: any chat can
drive /pause"** — the tests kill it.

---

# Second pass — findings from a live paper report

Source: a `--mode report` from a multi-day paper run, not from reading code. The
numbers that started it:

```
Equity $5,000.00 -> $4,989.08 | Realized PnL: fade +0.00
Estimates recorded: 23063 | passed edge threshold: 0 | avg edge: 1.00
No markout data yet
Open positions (20)  gross $1,891 -> true worst-case $846
```

Arithmetic on that book: maximum possible upside **$37.27**; expected loss at the
same prices **$36.34**. Those are equal because "the market price is the fair
probability" *means* EV zero — so the entire book's profitability rested on one
hardcoded constant, `fade.bias_discount = 0.40`, and a single adverse resolution
(-$124.93) would net **-$91** across all twenty positions.

None of this was a crash, an exception or a failing test. Every module behaved
exactly as written. That is what makes these findings worth recording.

## F-020 — The fade's take-profit could never fire

```
Severity: Critical | Confidence: high | Class: logic / money
Location: polymarket_bot/portfolio.py `exit_plan` (used by main.py `_exit_one`)
```
`exit_plan` triggers on `current_price / avg_price >= take_profit_multiple`,
default **7.0**. A fade leg is a NO bought high — 0.976 in the observed book — so
the trigger needs price **6.83**. Venue prices stop at 1.0, and the largest
multiple such a position can ever reach is **1.024**.

So `exit_plan` returned `None` on every call for the entire life of all twenty
positions. Not "rarely fired" — *could not* fire. The machinery was written for
longshots (buy 0.03, sell 0.21 = 7x) and the fade inherited it wholesale.

The second layer was also inert: `PositionGuardian` does match these legs
(`min_entry_price: 0.5`) but ships `auto_reduce: false`, so it emitted an alert
saying "a sold tail is materializing" and took no action. Net effect: a fade
position had exactly one exit, resolution, at the full notional.

Fixed with `fade.fade_exit_plan` — two rules in price space, where the arithmetic
is reachable:

* **tail-stop** — out when the implied tail probability has multiplied by
  `tail_stop_multiple` (2.4% -> 7.2%). Caps the per-leg loss at a few cents
  rather than ~98, which moves the required hit rate from ~97.6% to ~67%.
* **edge-captured** — out once `early_take_captured` (0.75) of *this trade's*
  maximum gain is banked. At entry 0.976 the whole prize is 2.4c, so the leg is
  sold near 0.994 with 1.8c realized and the capital recycled rather than waiting
  months for the last 0.6c.

Honest limitation stated in the code: a stop does **not** add EV. It pays the
spread and converts some spikes-that-revert into realized losses. It buys
survivability so the strategy lives long enough to be measured. Whether it is
net positive here is an empirical question the next paper run answers.

## F-021 — No filter looked at the shape of the payoff

```
Severity: High | Confidence: high | Class: risk-shape
Location: polymarket_bot/fade.py `reject_reason`
```
The gates were price band, binary-market, horizon, longshot-veto and
edge-after-fees. All correct, and jointly they admitted legs at 0.99 whose
(max gain / max loss) is **0.0101** — 99 wins to repay one loss.

Shape is a third axis, independent of time and of edge: a 4-day tail at 0.995
passes every other filter and is still unrecoverable. `min_payoff_ratio` (0.02,
i.e. at most ~50 wins per loss) is now checked *before* the edge gate on purpose
— a 1c tail fails both, and if the edge gate reported first an operator would
"fix" it by lowering `min_edge_after_fees`, which cannot make the shape sound.

## F-022 — The IRR planner ranked but never refused

```
Severity: High | Confidence: high | Class: logic / capital
Location: polymarket_bot/fade.py `_irr_score` / `cycle`
```
`_irr_score` computes edge-per-day and `cycle` sorts by it, with the comment "so
the best opportunities get capital before the portfolio caps fill". True — but
ranking only decides *order*. While the caps are slack (they usually are) every
candidate is entered regardless, so a leg earning 0.4c over 90 days was opened
next to one earning 2.2c over 4. `min_edge_per_day` (0.0005, ~20%/yr on a 0.98
leg) makes the same quantity a refusal, not just a sort key.

## F-023 — Paper fills were sampled per cycle, so the MM produced no data

```
Severity: High | Confidence: high | Class: measurement
Location: polymarket_bot/marketmaker.py `_paper_fills`, `react_to_tick`
```
`_paper_fills` was called only from `cycle()`. The cycle samples the book at its
interval; the tape moves at tick rate. A crossing that opened and closed between
two cycles was never observed — which is why a multi-day run reported **zero MM
fills and zero markouts**, leaving the only strategy with a mechanical edge
completely unmeasured while the fade accumulated twenty positions.

Fills are now evaluated on every WS tick. One hazard came with that and is fixed
in the same change: `_paper_fills` used `_top`, which falls back to a **blocking**
`clob.order_book` call. On the receive thread that stalls the recv loop, stops
the last-message timestamp advancing and trips `ws_staleness_kill_sec` — the bot
killing its own quoting to collect a paper fill. The tick path takes `ws_only`
and skips a token the WS store cannot serve. `test_the_tick_path_never_makes_a_rest_call`
pins it.

Separately, zero fills was *unreadable*: it looks identical whether our spread
was one tick too wide or the market never traded. The new `shadow_quotes` table
records, per retired quote, the closest the tape ever came to lifting it — so the
MM can be judged before it has ever been filled.

## F-024 — Exposure was handed out first-come-first-served

```
Severity: Medium | Confidence: high | Class: capital allocation
Location: polymarket_bot/portfolio.py `size_usd`
```
`max_total_exposure_pct` was a single shared pool. The fade runs every cycle over
hundreds of candidates and took $1,891 while the market maker held $0 — not by
decision, but by asking first. `reserve_for_mm_pct` (0.15) is subtracted from the
*directional* room only, as a second check beside the account-wide one, using
`total_exposure(mode, DIRECTIONAL_STRATEGIES)`. This required `Position.strategy`,
which is taken from the opening trade and never overwritten by a later sell —
otherwise a partially-sold fade leg would be routed through the wrong exit rule.

## F-025 — `avg edge 1.00` was unreadable

```
Severity: Medium | Confidence: high | Class: observability
Location: polymarket_bot/ledger.py `estimates_summary`, analytics.py
```
23,063 estimates, mean edge ratio 1.00, zero qualifying. That single number
cannot distinguish "there is no systematic mispricing to find" from "the
estimator is echoing the market price" — and only the second means the LLM is
being paid for nothing. The summary now also counts **binding** estimates
(`p_est < p_mkt * (1 - bias)`, the only case where the fade's `min()` picks the
model over the prior) and **informative** ones (`|edge_ratio - 1| > 0.05`), with
the threshold derived from the configured bias rather than hardcoded. `FadeStrategy`
counts the same thing live. If binding is 0, the report says so in plain words.

**Not fixed, deliberately:** `bias_discount = 0.40` itself. It carries the entire
expected return of the strategy and it is a guess — the config comment even reads
"AGGR: assume tails overpriced by 40% (was 30%)". `TailBiasCalibrator` exists to
replace it with a measured value and has never fired, because nothing has
resolved. Changing the number by hand would substitute one guess for another;
the honest move is to let the first resolutions set it.

## Mutation coverage

27 targeted mutants, all killed. Ten are new, covering the shape gate, the payoff
ratio arithmetic, the IRR floor, both exit rules, the exit direction guard, the MM
reserve on both sides, tick-resolution fills and the REST-fallback ban.

One mutant was **removed rather than counted**: deleting
`if directional_room <= 0: return None` is an *equivalent* mutant — a negative
room flows into `min(size, room)` and the `size < min_order_notional` floor
returns `None` anyway. No test can kill it because behaviour is identical, so
reporting it as a survivor would invent a test gap that does not exist.


## F-026 — The early take could close a position at a loss, and left a marginal entry no room

```
Severity: Medium | Confidence: high | Class: logic / money
Found by: the first live paper cycle after F-020 shipped — not by the test suite
Location: polymarket_bot/fade.py `fade_exit_plan` (the take branch)
```
The take shipped in F-020 fired when the **remaining** payoff ratio dropped below
`min_payoff_ratio` — "the entry standard applied to holding". Elegant, and wrong
in two ways that only a real book exposed.

**It ignores what was paid.** The condition is a statement about the mark alone,
so on a leg whose shape was never acceptable — which is every position opened
before the entry gate existed — it was already true at the moment of entry. The
rule therefore fired at whatever mark happened to exist. In the first cycle it
closed three positions at or below cost and logged all three as takes:

```
Ronaldo Caiado   entry 0.986  exit 0.982   -$0.50
Jair Bolsonaro   entry 0.989  exit 0.989    $0.00
Putin out by Aug entry 0.982  exit 0.982    $0.00
```

**Both thresholds were the same number.** The entry gate admits
`payoff_ratio >= 0.02`, i.e. entry <= 1/1.02 = 0.98039; the take fired below the
same ratio, i.e. mark > 0.98039. A position opened at the boundary had *zero*
holding room and would be liquidated on the next upward tick — buy 0.98039, sell
0.98040, pay the spread twice for nothing.

Replaced with a captured fraction of the trade's own maximum gain,
`(mark - entry) / (1 - entry) >= early_take_captured`. It requires `mark > entry`,
so a take can no longer realize a loss; it scales with the entry, so a marginal
position has real room; and it is the actual IRR argument — 75% of the prize
banked, capital recycled.

Applied to that same cycle it fires on 8 of the 16 exits instead of 16, and every
one of the 8 is a profit. Both stops are unaffected: they are the two exits that
mattered.

A mutant reverting the take to the remaining-ratio form is now in the suite and
is killed, so this cannot come back quietly.

**What the cycle also confirmed, in one line:**

```
fade: estimator set fair value in 0/2 scored tails (0.0%)
      — the rest ran on the bias_discount prior
```
F-025's instrumentation answered on its first run. Two tails is not a sample, but
the direction is the one the 23,063-estimate average edge of 1.00 predicted.
