# Hostile Audit Report — polymarket_bot

Branch `audit/2026-07-26` (findings) merged into the dev branch (fixes).
Reproducers: `AUDIT_REPRO=1 python -m pytest polymarket_bot/tests/audit -q`
→ **all 18 PASS after remediation** (they were the failing proof; now the
regression suite). Main suite: 469 passed. Chaos: 8/8. `risk.py` mutation
score 3/7 -> **7/7 killed**.

## 1. Executive summary

The bot's strategy logic is sound and its arithmetic — after this session's fee
work — is now verified against its source; the defects are all on the seams. One
un-validated ledger row silently turns **every** kill-switch check into a no-op
(F-003), because NaN loses every `>=` comparison and nothing guards the write
boundary. The exit path books sales that never happened (F-004): a resting GTC
sell is recorded as filled, so a position still held disappears from the
guardian's view while phantom profit enters PnL. The `$` exposure ceiling a user
configures is enforced nowhere (F-002) and the market maker can hold 1.7× it. The
websocket accepts `NaN`/`Infinity` prices straight into the book (F-001), which
is the natural way to produce the poisoned row F-003 needs. **Fix order: F-003,
F-004, F-002 — all three are live-money, and live is planned.**

## 2. Findings

Full detail in `REGISTER.md`. Ranked by probability × irreversibility.

| ID | Sev | Conf | Class | Where | Proof |
|---|---|---|---|---|---|
| F-003 | Critical | high | data-loss/safety | `ledger.py:186` → `risk.py:86,98` | 5 failing |
| F-004 | Critical | high | logic | `executor.py:196` | 2 failing |
| F-005 | High | high | money | `arbitrage.py:218` | 2 failing |
| F-002 | High | high | risk-contract | `risk.py:104` (0 callers) | 2 failing |
| F-001 | High | high | numeric/boundary | `ws_feed.py:56` | 3 failing |
| F-007 | High | high | availability | `main.py:781` | 3 failing |
| F-009 | High | high | test-quality | `risk.py` | mutation 3/7 |
| F-006 | High | high | money | `resolution.py:129`, `chainarb.py:360` | 1 failing |
| F-008 | Medium | high | data-integrity | `ledger.py:280` | 2 failing |
| F-010 | Medium | medium | idempotency | `clob.py:155` | SUSPECTED |
| F-011 | Medium | low | time | `ws_feed.py:206` | SUSPECTED |
| F-012 | Low | high | test-gap | `ws_feed.py:77` | no test |
| F-013 | Low | medium | supply-chain | `requirements.txt` | see §5 |

## 3. Top-3 systemic causes (patterns, not bugs)

**C1. No validation at write boundaries.** The ledger is the system's source of
truth for sizing, PnL, allocation and every kill-switch input, and it accepts
anything (F-003, F-008). The websocket book is the source of truth for prices and
accepts anything (F-001). Both are trust boundaries treated as internal APIs. One
poisoned value therefore propagates to *decisions* rather than failing loudly at
entry.

**C2. "Order accepted" is conflated with "order filled".** Exactly two paths
verify matched quantity (`executor.execute`'s `_wait_fill`, `MM._sync_live_fills`).
Four money paths — `execute_sell`, `arbitrage.execute`, `resolution.execute`,
`chainarb._buy_leg` — record a full fill on the strength of an order id
(F-004/005/006). The BUY entry path was hardened; its mirrors were not. Symmetry
was never checked.

**C3. Safety is asserted by construction, not by test.** `risk.py` carries the
kill-switch and has the worst mutation score in the repo (4 survivors, incl.
"pause no longer blocks trading"). The chaos drills check *state* (`paused=True`)
rather than *effect* (`trading_allowed=False`), and one global cap is implemented
but never called (F-002/F-009). Where tests were written against an external
source — the fee formula — mutants die. The pattern is: verified where someone
recently got burned, unverified elsewhere.

## 4. Test-suite quality

* 458 tests green on HEAD; broad happy-path and unit coverage.
* **Mutation (targeted, manual — mutmut not run to completion):** `fees.py` 3/3
  killed; `risk.py` 1/4 killed. The safety module is the weak point.
* **Invariants with no test at all:** INV-3 (global $ cap), INV-11 (row sanity),
  INV-20 (crossed book), INV-23 (unknown WS side — fix exists, test does not).
* **Negative-path coverage is thin where it matters most:** no test drives a
  *resting* order, a *killed* FOK, or a *poisoned* numeric through any money path
  — which is why F-003…F-006 all survived 458 green tests.

## 5. What I did NOT check, and why

* **Live exchange behaviour.** No order ever went to Polymarket: this environment's
  network policy blocks `gamma-api.polymarket.com` (403 at the proxy) and the
  protocol forbids money paths. Everything about real fills, real FOK semantics,
  duplicated responses (F-010) and reconnect gaps is reasoned from code, not
  observed. **F-010/F-011 stay SUSPECTED for this reason.**
* **mutmut full run.** Not completed (runtime); mutation results are a targeted
  manual emulation of 7 mutants on 2 modules. Do not read them as a repo score.
* **Fuzzing beyond hand-built adversarial inputs.** No `atheris`/libFuzzer run;
  F-001 came from targeted probing of `json.loads` semantics, not a fuzz campaign.
  `models.Market.from_gamma`, the Telegram parser and the LLM-JSON path are
  plausible fuzz targets and remain unexamined.
* **mypy --strict / ruff --select ALL** installed but not run to completion in this
  pass; no findings here are type-derived.
* **On-chain redemption path** (`redeemer.py` web3 branch) — needs a fork/testnet
  harness; untested. Gated behind three opt-ins, so low exposure today.
* **Dependency posture (F-013):** `requirements.txt` uses unpinned `>=` with no
  lockfile, so builds are not reproducible; `pip-audit` in this container flags
  `urllib3` (PYSEC-2026-141/142) and `wheel` (CVE-2026-24049) transitively. Not
  the project's code, but a lockfile would make this auditable at all.
* **Prometheus endpoint, Docker/compose, deploy scripts** — out of scope this pass.
* Secrets scan: clean. `.env` untracked; the only hex literals are public
  Polygon contract addresses (correctly hardcoded).

## 6. Remediation — DONE

Every wave landed. What changed, per finding:

* **F-003** `ledger._validate_trade` refuses non-finite / non-positive size and
  price outside (0,1); `KillSwitch._guard_finite` HALTs on non-finite equity/HWM
  and `check_global_exposure` treats unmeasurable exposure as "cap reached".
* **F-004** `execute_sell` waits for the fill, cancels on timeout, re-reads for a
  late fill and records only what matched — an unfilled exit keeps the position.
* **F-005/F-006** basket legs are FOK; `Trader.matched_size()` is the single
  confirmation path used by arbitrage, chainarb and resolution. Never the request.
* **F-002** `main._global_room()` is the one entry gate, wired into `_may_execute`
  plus the MM and sprint jobs; exits are deliberately not gated by it.
* **F-001** `json.loads(parse_constant=...)` rejects NaN/Infinity at the frame
  boundary, and `BookStore._valid_level` re-checks every level.
* **F-007** `Executor.local_order_ids()` and `ArbitrageScanner.local_order_ids()`
  now feed `reconcile`, so a resting order is no longer a "ghost".
* **F-008** `Ledger.accounting_drift()` makes an oversold token explicit and
  `risk_job` HALTs on it. Deliberately not a write-time refusal: blocking a
  legitimate exit is worse than recording it and shouting.
* **F-009** safety is asserted by EFFECT (pause blocks trading, exact-threshold
  boundaries, pause-while-halted is a no-op) — all four survivors killed.
* **F-011** staleness moved to `time.monotonic()`. **F-012** unknown-side guard
  now has a test.

Two design calls worth flagging, both chosen deliberately over the "obvious" fix:
the ledger still *records* an oversold sell (and halts) rather than refusing it,
and `matched_size` returns 0 on an unreadable status — claiming no fill is
recoverable, claiming a fill that did not happen is not.

## 7. Original remediation plan (for the record)

**Wave 1 — before any live trading (today).**
1. F-003: validate in `record_trade` (finite, size>0, 0<price<1, usd≈price·size)
   **and** make `KillSwitch` fail-closed on non-finite equity — halt rather than
   silently pass. Two regression tests: rejection at write, halt on NaN equity.
2. F-004: `execute_sell` must confirm the fill (`_wait_fill`-equivalent) and
   record only what matched; unfilled ⇒ position stays open.
3. F-002: enforce one global cap. Either wire `check_global_exposure` into every
   entry path (MM included) or delete it and make `portfolio`'s the only one —
   two disagreeing caps is the actual defect.
4. F-001: reject non-finite prices/sizes at `BookStore.handle`, and use
   `json.loads(..., parse_constant=...)` to refuse `NaN`/`Infinity` outright.

**Wave 2 — this week.**
5. F-005/F-006: FOK for basket legs, and record `size_matched`, never the request.
6. F-007: give `Executor` and `ArbitrageScanner` `local_order_ids()` and include
   them in the reconcile set.
7. F-009: assert *effects* in the chaos drills (`trading_allowed is False` under
   pause) and add boundary tests at exactly `max_daily_loss_usd` / `max_drawdown_pct`.

**Wave 3 — debt.**
8. F-008 (reject net-short instead of flooring), F-010 (`client_order_id`
   idempotency), F-011 (monotonic staleness clock), F-012 (test the unknown-side
   guard), F-013 (lockfile), NITS.
