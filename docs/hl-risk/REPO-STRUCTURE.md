# Repository structure — Hyperliquid Portfolio Risk Layer

This product lives in the same repository as `polymarket_bot/` but shares no
code with it. The two are independent top-level trees; nothing in
`risk_engine/` imports `polymarket_bot` or vice versa.

> **This file described a plan, not the tree, until 2026-07-31.** It listed
> `services/backend/` and `apps/web/` as "not built yet" (both are built),
> named five backend directories of which one exists, omitted
> `risk_engine/service/`, `deploy/` and `qa/` entirely, and listed
> `max_safe_size` as a shipped tool — it does not exist (Phase 4, gate
> closed; OPEN-QUESTIONS D2). What follows is the tree as it stands. Where
> something is still a plan it is marked **PLANNED**, and keeping that
> distinction visible is the point of the file: a structure document that
> quietly describes the intended end state reads, to anyone new, as a
> description of what they can go and open.

## What is actually here

```
risk_engine/                  # SERVICE 1 — Python. Phases 1-3 and 5, built.
  domain/                     #   value types: Book, Position, AssetSpec, RiskEstimate
  liquidation/                #   §1 — margin tiers, liq price, cross+N-isolated simulator
  market/                     #   §5.1 — Info client, meta/state/candle/funding/ledger parsers,
                              #   verify harness (E5/B2/C1/C2), B4 trades-feed collector,
                              #   C5 isolated-funding probe
  model/                      #   §2 — EWMA, Ledoit-Wolf, PSD, Student-t marginals,
                              #   funding AR(1), t-copula + §2.3 tail-asymmetry diagnostic
  sim/                        #   §2.3/2.5 — t-copula paths, Monte Carlo engine, CI statistics
  service/                    #   §8 — the internal REST service and its bundle builders
  tools/                      #   §4 — portfolio_risk, pre_trade_delta, funding_drag
  validation/                 #   §3.1 benchmarks, §3.2 baselines, CLI gate, B1 power analysis
  shadow/                     #   §3.3/3.4 — journal, snapshot cron, resolver, ICC/clustering
  observability/              #   §7 — counters, latency histograms, model diagnostics
  tests/                      #   pytest, 550 tests; the §3.1 gate is test_benchmarks.py
  requirements.txt

services/backend/             # SERVICE 2 — Node/TypeScript. Phase 3, built.
  src/staleness/contract.ts   #   §6 degradation contract as a discriminated union
  src/risk/client.ts          #   calls the Python service over internal REST
  src/routes/                 #   HTTP surface
  src/ws/                     #   §5.2 listener
  src/server.ts, src/main.ts  #   wiring and entry point
  package.json                #   fastify; vitest/tsx/typescript for development

apps/web/                     # SERVICE 3 — Next.js 15. Phase 3, built.
  app/                        #   one screen (page.tsx), layout, globals
  components/                 #   Guarded, WalletBar, AgentKeyDisclaimer
  lib/contract.ts             #   the §6 contract mirrored for the client
  package.json                #   next, react, react-dom

deploy/                       # containers and the compose stack
  Dockerfile.{engine,backend,web}, docker-compose.yml, prometheus.yml,
  healthcheck.py, README.md, addresses.json

qa/                           # quality gates — QA-PROCEDURES.md, run_all.sh,
                              # mutation_check.py, requirements-dev.txt
scripts/run-stack.sh          # start the whole stack locally
docs/hl-risk/                 # specification deltas, open questions, methodology
```

**PLANNED, not present.** `src/exchange/` (§5.4 agent wallet, nonce, the
single signing path) and `src/db/` (Postgres access) are Phase 4 and do not
exist. `tools/max_safe_size.py` is Phase 4 (D2). `lib/wallet/` — wagmi/viem,
`approveBuilderFee`, agent-key revoke — is Phase 4, which is why
`AgentKeyDisclaimer.tsx` is copy rather than a flow and `apps/web` carries no
wallet dependency at all. Phase 4 does not begin until shadow validation
passes §3.3's gate (21 days × 200 addresses), so none of this is near-term.

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
`websockets` is needed only by the B4 collector and is imported lazily, so it
is not an engine dependency.

**`services/backend`** — `fastify`, plus `vitest`/`tsx`/`typescript` for
development. `@nktkas/hyperliquid` (EIP-712 signing, §5.5), `pg`, `zod`,
`ioredis` and `ws` arrive with the Phase 4 directories above; listing them
here before then misdescribes what a reader will find in `package.json`.

**`apps/web`** — `next@15`, `react`, `react-dom`. `wagmi`, `viem` and
`tailwindcss` are Phase 4, for the same reason.

Versions are pinned per service. Python runtime deps are pinned in
`risk_engine/requirements.txt`; QA tooling is separate, in
`qa/requirements-dev.txt`. The two Node services get their own lockfiles and
are never hoisted into one workspace, because the backend will hold signing
keys at Phase 4 and the frontend must not be able to reach them through a
shared `node_modules`.

## Build order

Phases 1-3 and 5 are built. Phase 4 — `max_safe_size`, the builder fee, agent
keys, the signing path — is gated on shadow validation, and its directories
are deliberately absent rather than scaffolded: empty structure that compiles
but does nothing is a liability, and it is also exactly what let this file
drift, since a listed directory reads as an existing one whether or not it is.
