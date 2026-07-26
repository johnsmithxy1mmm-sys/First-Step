# Invariants — polymarket_bot (audit/2026-07-26, Phase 1)

Legend for **Enforced**: `code` = runtime check, `test` = asserted in tests,
`nowhere` = stated/implied but never verified. `nowhere` ⇒ automatic INV-finding.

## A. Money conservation & bounds

| ID | Invariant | Enforced | Notes |
|---|---|---|---|
| INV-1 | Order price ∈ [tick, 1−tick] | code (`round_to_tick` clamp) + test | fixed this session |
| INV-2 | Taker fee = θ·shares·p·(1−p), θ per category | code + test (official worked example $1.75) | |
| INV-3 | Total exposure ≤ `risk.max_global_exposure_usd` | **nowhere** — `check_global_exposure` has **zero callers**; portfolio uses a *different* cap (`max_total_exposure_pct·bankroll`), MM/sprint check only per-market caps | **INV-3 open**: MM×5 markets can exceed the global $ limit; two caps can disagree |
| INV-4 | Per-market position ≤ `max_position_per_market_usd` (×2 for two-sided MM) | code (portfolio, MM `compute_quote`, `sides_allowed`) + test | |
| INV-5 | Category exposure ≤ `max_category_pct·bankroll` (correlation-weighted) | code (portfolio) + test | longshot/fade path only — MM inventory not category-capped |
| INV-6 | Bankroll floor: `effective_bankroll ≥ floor` (compounding never below initial floor) | code | test coverage partial |
| INV-7 | Payout of chain-arb set ∈ [$1, $2] over possible states | code + exhaustive test | fixed this session |
| INV-8 | Fee/rebate: maker pays 0, rebate = share·collected fee | code + test | |

## B. State-machine / idempotency

| ID | Invariant | Enforced | Notes |
|---|---|---|---|
| INV-9 | No double entry per token (one BUY lifecycle) | code (`has_position_or_open_buy` + live `api_positions`/`open_orders` reconcile; RateLimited ⇒ assume-entered) + test | conservative: blocks re-entry forever after close |
| INV-10 | Every real fill is recorded in the ledger (incl. partial/late) | code (post-cancel final read, executor + MM) + test | fixed this session; **no client_order_id idempotency** — a duplicated exchange response can double-record (SUSPECTED, needs repro) |
| INV-11 | Ledger trade rows: size > 0, price ∈ (0,1), usd = price·size | **nowhere** (`record_trade` writes whatever it is given) | INV-11 open |
| INV-12 | Kill-switch: HALT is terminal until manual restart; PAUSE sources independent | code + chaos drill | |
| INV-13 | No NEW risk while `!trading_allowed`; risk-reduction still allowed | code (gates at each strategy + `allow_execute`) + chaos | exits deliberately bypass pause |
| INV-14 | A cancelled quote's outcome recorded exactly once (filled XOR unfilled) | code + test | fixed this session |
| INV-15 | Resolution of a token is recorded at most once (PK) and never flips | code (PK upsert?) | flip-protection untested |

## C. Monotonicity / time

| ID | Invariant | Enforced | Notes |
|---|---|---|---|
| INV-16 | WS `_last_msg_ts` moves forward; staleness gap ⇒ pause within threshold | code + chaos | wall clock, not monotonic — NTP step backwards silently suppresses/falsifies gap (SUSPECTED) |
| INV-17 | Fastlane: every flag() eventually checked (deferred, not dropped) | code + test | fixed this session |
| INV-18 | Day boundary for daily-loss = UTC | code | DST-immune by construction |
| INV-19 | `deadline` ordering in date ladders consistent with endDate cross-check | code + test | |

## D. Parsing / reversibility

| ID | Invariant | Enforced | Notes |
|---|---|---|---|
| INV-20 | Book: bid ≤ ask after every message sequence; sizes ≥ 0 | partial (crossed *input* levels skipped in tickstore mid_series; BookStore itself **accepts crossed state** from price_change sequences) | INV-20 open: consumers (microprice, paper fills) читают BookStore.top без проверки bid≤ask — mm guard проверяет только jump |
| INV-21 | Gamma market with malformed fields never crashes a job (skip, don't die) | code (per-field try/except in from_gamma) | fuzz would strengthen |
| INV-22 | config: user > profile > example > defaults; removed keys warn | code + test | |
| INV-23 | WS price_change with unknown side is ignored | code + (no test) | fixed this session, **test missing** |

## E. Uniqueness / accounting identities

| ID | Invariant | Enforced | Notes |
|---|---|---|---|
| INV-24 | `open_positions` = Σ buys − Σ sells − resolved, floored at 0 per token | code | sells > buys silently floored — masks desync (ties to INV-10/11) |
| INV-25 | `realized_pnl` = Σ(sell − avg·size) + Σ(resolution payout − avg·rest) | code | no property test vs naive reference (metamorphic candidate) |
| INV-26 | markout rows reference existing trades (FK by join) | schema join | no FK constraint enforced |

## Immediate INV-findings (no repro needed — absence is the defect)

* **INV-3**: global $ exposure cap enforced nowhere; two different global caps exist (risk.max_global_exposure_usd vs portfolio.max_total_exposure_pct) and can silently disagree. MM inventory bypasses both.
* **INV-11**: ledger accepts size ≤ 0 / price ∉ (0,1) / usd ≠ price·size without complaint — one bad caller poisons PnL, sizing, kill-switch inputs downstream.
* **INV-20**: BookStore can hold a crossed book (bid > ask) after a legal message sequence; `top.mid`/`microprice` then produce prices used for quoting/paper fills.
* **INV-16 (SUSPECTED)**: staleness clock is wall-clock; backwards NTP step masks an outage.
* **INV-23**: fixed but untested — regression can silently return.

Phase 2 attack plan will target: INV-3, INV-10 (duplicate exchange response), INV-11, INV-20, INV-24/25 (property/metamorphic), plus Appendix-A items: cancel-of-filled-order, WS reconnect snapshot gap, partial-fill ordering, restart with open orders.
