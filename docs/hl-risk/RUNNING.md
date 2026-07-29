# Running the stack

Phase 3 is read-only. Nothing in this stack can move funds, and no execution
path exists yet — that is Phase 4, and Phase 4 is gated on shadow validation
(§3.3).

## One command

```bash
./scripts/run-stack.sh          # synthetic market data
./scripts/run-stack.sh --live   # live Hyperliquid data (needs API access)
```

Then open http://127.0.0.1:3000.

Three processes come up:

| Process | Port | Role |
|---|---|---|
| Python risk engine | 8787 | computes risk; loopback only, no auth |
| Node backend | 8080 | enforces the §6 degradation contract |
| Next.js frontend | 3000 | one screen, four widgets |

The browser never talks to the risk engine. The backend is the only consumer,
which is what lets the §6 contract live in one place instead of being
re-derived by the client.

## Synthetic vs live data

`--fixture` (the default) runs on a one-factor synthetic market. It exists for
two reasons: `api.hyperliquid.xyz` is unreachable from some build environments
(OPEN-QUESTIONS E5), and the §6 degradation contract has to be exercisable
without a live venue — what happens when data stops arriving is the whole
point of that contract.

Synthetic mode is labelled everywhere it could be mistaken for real: `/health`
returns `synthetic_data: true`, and the frontend shows a banner saying the
numbers describe a simulated market.

**The live path has never been exercised end to end.** Every §5.1 parser is
written against documented response shapes. Verify it before trusting it.

## Running the pieces separately

```bash
# risk engine
python3 -m risk_engine.service --fixture --port 8787

# backend
cd services/backend && npm install && npm run build
PORT=8080 RISK_SERVICE_URL=http://127.0.0.1:8787 node dist/main.js

# frontend
cd apps/web && npm install
BACKEND_URL=http://127.0.0.1:8080 npm run dev
```

## Tests

```bash
pytest risk_engine/tests -q                        # 259 tests
python -m risk_engine.validation.cli benchmarks    # the §3.1 gate, 6/6
cd services/backend && npm test                    # 23 tests, §6 contract
cd services/backend && node scripts/degradation-check.mjs   # §9 Phase 3 acceptance

# journal parity against a real server; the Postgres half skips without a DSN
HL_TEST_POSTGRES_DSN=postgresql://user@host/db pytest risk_engine/tests/test_journal_backends.py -q
```

The last one is the Phase 3 acceptance criterion, run the way the criterion
is phrased. It starts both real processes, SIGKILLs the risk engine, and
checks that the backend serves no value at all rather than a cached one:

```
4. with the risk service stopped
  [PASS] no value is served — freshness=unavailable
  [PASS] freshness is unavailable, not a cached fresh value
  [PASS] execution is blocked
  [PASS] health survives the outage rather than 500ing
```

The 60-second and 300-second boundaries themselves are covered by
clock-injected tests in `test/degradation.test.ts`; waiting five real minutes
here would buy nothing. What only a real run can prove is that a killed
process produces "unavailable" rather than a stale value, and that is what
step 4 checks.

## What the degradation contract does

| Age of the estimate | Behaviour |
|---|---|
| < 60 s | shown, age visible, execution allowed |
| 60 s – 5 min | shown, marked stale with its age, **execution blocked** |
| > 5 min | **value withheld entirely**, execution blocked |
| engine unreachable | value withheld, reason shown, execution blocked |
| estimate unpublishable (§2.5) | value withheld — the engine does not stand behind it |

Two details worth knowing:

The age is measured from the engine's own `computed_at`, not from when the
backend received the response, so a slow hop cannot launder a stale number
into a fresh one.

Beyond five minutes the payload has no `value` field at all. It is a
discriminated union, so a component cannot read a number out of a hidden
payload even by accident. §6 asks for values to be hidden rather than greyed
out, and a greyed-out number is still a number on the screen.

The correlation matrix gets its own, longer clock (11 minutes, two missed
rebuild cycles). It is rebuilt every five minutes by design (§2.1), so
thresholding it on the 60-second book clock would mark the product
permanently stale (OPEN-QUESTIONS D4).

## The shadow harness is a separate stack

Nothing above accumulates the §3.3 validation window. That runs as two daily
jobs against their own journal, independently of whether the web stack is up:

```bash
python -m risk_engine.shadow snapshot --journal shadow.db --addresses addrs.json
python -m risk_engine.shadow resolve  --journal shadow.db --addresses addrs.json
python -m risk_engine.shadow progress --journal shadow.db
```

Setup, the address-frame requirement, and what has to be settled *before*
the clock starts are in [`../../risk_engine/README.md`](../../risk_engine/README.md).
