# Findings Register — audit/2026-07-26

18 failing reproducers on HEAD + mutation evidence. Run:

    AUDIT_REPRO=1 python -m pytest polymarket_bot/tests/audit -q

Sorted by (probability in prod) × (irreversibility). Nothing fixed — per protocol.

| ID | Severity | Conf. | Class | Location | Repro | Status |
|---|---|---|---|---|---|---|
| F-003 | **Critical** | high | data-loss / safety | `ledger.py:186` `record_trade` (no validation) → `risk.py:86,98` | `test_f003_*` (5 fail) | open |
| F-004 | **Critical** | high | logic / api-contract | `executor.py:196-221` `execute_sell` | `test_f004_*` (2 fail) | open |
| F-005 | High | high | logic / money | `arbitrage.py:218-238` GTC legs | `test_f004_*` (2 fail) | open |
| F-002 | High | high | api-contract / risk | `risk.py:104` (0 callers) vs `portfolio.py:204` | `test_f002_*` (2 fail) | open |
| F-001 | High | high | numeric / trust-boundary | `ws_feed.py:56-95` `BookStore.handle` | `test_f001_*` (3 fail) | open |
| F-007 | High | high | availability / safety | `main.py:781` reconcile scope | `test_f007_*` (3 fail) | open |
| F-009 | High | high | test-quality | `risk.py` (whole module) | mutation: 3/7 killed | open |
| F-006 | High | high | logic / money | `resolution.py:129-141`, `chainarb.py:360` | `test_f004_*` (1 fail) | open |
| F-008 | Medium | high | data-integrity | `ledger.py:280-285` floor | `test_f008_*` (2 fail) | open |
| F-010 | Medium | medium | idempotency | `clob.py:155` no `client_order_id` | trace only | SUSPECTED |
| F-011 | Medium | low | time | `ws_feed.py:206` wall clock | trace only | SUSPECTED |
| F-012 | Low | high | test-gap | `ws_feed.py:77` unknown-side fix | no test exists | open |

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
