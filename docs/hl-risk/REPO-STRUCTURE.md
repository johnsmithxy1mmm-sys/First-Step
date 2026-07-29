# Repository structure — Hyperliquid Portfolio Risk Layer

This product lives in the same repository as `polymarket_bot/` but shares no
code with it. The two are independent top-level trees; nothing in
`risk_engine/` imports `polymarket_bot` or vice versa.

## Three services, three trees

```
risk_engine/                  # SERVICE 1 — Python. Phase 1 (built now).
  domain/                     #   value types: Book, Position, AssetSpec, RiskEstimate
  liquidation/                #   §1 — margin tiers, liq price, cross+N-isolated simulator
  market/                     #   §5.1 — Info client, meta/state/candle/funding parsers
  model/                      #   §2 — EWMA, Ledoit-Wolf, PSD, Student-t marginals, funding AR(1)
  sim/                        #   §2.3/2.5 — t-copula paths, Monte Carlo engine, CI statistics
  tools/                      #   §4 — portfolio_risk, pre_trade_delta, max_safe_size, funding_drag
  validation/                 #   §3.1 benchmarks, §3.2 baselines, CLI gate
  shadow/                     #   §3.3/3.4 — calibration journal, snapshot cron, resolver, metrics
  observability/              #   §7 — counters, latency histograms
  tests/                      #   pytest; the §3.1 gate lives in test_benchmarks.py
  requirements.txt

services/backend/             # SERVICE 2 — Node/TypeScript. Phase 3-4 (not built yet).
  src/ws/                     #   §5.2 sharded webData2 listener, 1000-sub/IP shard planner
  src/orchestrator/           #   WS -> Python REST -> push to frontend
  src/exchange/               #   §5.4 agent wallet, nonce, the single signing path
  src/staleness/              #   §6 degradation contract, enforced in the type system
  src/db/                     #   Postgres access; migrations shared with risk_engine/shadow
  package.json                #   @nktkas/hyperliquid, fastify, pg, zod, vitest

apps/web/                     # SERVICE 3 — Next.js 15. Phase 3 (not built yet).
  app/                        #   one screen, four widgets
  components/                 #   CI-and-age-bearing display primitives
  lib/wallet/                 #   wagmi/viem, approveBuilderFee, agent-key disclaimer + revoke
  package.json

docs/hl-risk/                 # specification deltas, open questions, methodology
```

## Why the risk engine is a separate process, not a library

The Python engine is reached over an internal REST call from Node (§8). It is
not embedded, because the matrix rebuild (§2.1, every 5 minutes plus force
triggers) is a long-running stateful job that must keep a warm Cholesky
factor in memory between requests, and because the 300 ms online budget
(§2.6) is only achievable when the offline half has already run.

## Dependencies

**`risk_engine`** — deliberately thin. `numpy`, `scipy`, `pytest` and nothing
else at Phase 1. The Info client (§5.1) is written against `urllib` from the
standard library rather than `httpx`: it issues one kind of request (POST
JSON to one host) and the weight-budget accounting in §5.3 has to be
hand-rolled regardless, so a dependency buys nothing. Postgres access uses
`psycopg[binary]` when a DSN is configured; the journal falls back to
`sqlite3` (stdlib) for local runs and tests, against the same schema.

**`services/backend`** — `@nktkas/hyperliquid` (EIP-712 signing, §5.5),
`ws`, `fastify`, `pg`, `zod`, `ioredis`, `vitest`.

**`apps/web`** — `next@15`, `react`, `tailwindcss`, `wagmi`, `viem`.

Versions are pinned per service. Python deps are pinned in
`risk_engine/requirements.txt`; the two Node services get their own
lockfiles and are never hoisted into one workspace, because the backend
holds signing keys and the frontend must not be able to reach them through a
shared `node_modules`.

## Build order

Phase 1 touches `risk_engine/` only. `services/backend/` and `apps/web/`
are created at Phases 3-4. The directories above are the plan, not stubs —
empty scaffolding that compiles but does nothing is a liability, so they are
not created until their phase starts.
