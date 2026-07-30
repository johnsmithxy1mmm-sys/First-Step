# Deploying the read-only stack

Nothing in this stack can move funds. There is no execution path in the
codebase at all — that is Phase 4, and Phase 4 is gated on shadow validation
(§3.3).

That is worth deploying anyway, and worth deploying *before* the gate opens.
The only product metric §0 admits is day-90 retention, which needs real users
and ninety days. The shadow window needs its own 21 days. Run sequentially
those are two waits; run together they are one.

## Setup

```bash
cp deploy/.env.example deploy/.env
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'   # RISK_SERVICE_TOKEN
docker compose -f deploy/docker-compose.yml up -d --build
```

Then open http://localhost:3000.

`deploy/.env` is gitignored and excluded from the build context. The images
`COPY` only named paths, so it could not reach a layer regardless, but a
secret that never had to leave the host should not.

| Service | Port | Role |
|---|---|---|
| `engine` | 8787, internal only | computes risk; the shadow jobs run this same image |
| `backend` | 8080, internal only | enforces the §6 degradation contract |
| `web` | 3000, published | one screen, four widgets |
| `journal` | 5432, internal only | calibration journal (§3.4) |

Only `web` is published. The engine has no rate limiting of its own — §5.3
puts that in the backend — so putting it on a host interface would be
handing out an unmetered engine, which the token protects but does not
justify.

## The engine refuses to be exposed without a token

A container has to bind `0.0.0.0`, and `serve` refuses that without
`RISK_SERVICE_TOKEN`:

```
RuntimeError: refusing to bind '0.0.0.0' without RISK_SERVICE_TOKEN
```

This is intentional and it is the whole point. A missing token produces a
container that will not start, at deploy time, in front of whoever is
deploying — rather than an unauthenticated engine discovered later by
whoever scans the port. `/health` stays open so orchestrator healthchecks
work before any credential is in scope; it names no user and no book.

## Synthetic vs live

`ENGINE_MODE=--fixture` (the default) runs a synthetic one-factor market. It
exercises every moving part, validates nothing, and labels itself as
synthetic everywhere it could be mistaken for real: `/health` returns
`synthetic_data: true` and the frontend shows a banner.

Before `--live`, run the verification harness — every §5.1 parser is written
against documented response shapes and has never seen a live response:

```bash
python -m risk_engine.market.verify --report verify.json
```

It exits non-zero while anything is unverified, and names each assumption
separately. See OPEN-QUESTIONS C1, C2, C4, C5, E5.

## Starting the shadow clock

The shadow jobs sit behind a compose profile, so `up` does not start them.
Starting the §3.3 counter is a decision, not a side effect:

```bash
docker compose -f deploy/docker-compose.yml --profile shadow up -d
docker compose -f deploy/docker-compose.yml run --rm engine \
  -m risk_engine.shadow progress --journal "$SHADOW_DSN"
```

**Read `risk_engine/README.md` before you do.** A1 and A8 still move the
distribution, and §3.3 resets the window to zero when it moves, so days
accumulated before those two are settled are days that get thrown away.
C1, C2 and C5 no longer block — all three were closed against live data.

Live runs also need `deploy/addresses.json` filled in — both the list and the
`frame` field describing what it is a sample *of*. `FileAddressSource`
refuses a list without one (OPEN-QUESTIONS B4). Each address is `0x` plus 40
hex digits in any case; the checksummed form from a block explorer is fine, it
is folded to lowercase so one account cannot be journalled twice under two
spellings. A malformed entry fails the load, naming its index.

The resolver runs hourly against a daily snapshot. That is not a mistake:
a prediction resolves 24 h after it was made, and a resolution collected
late is marked stale and dropped from the gate (audit A-04). The scarce
resource is the window, not the API calls.

## What has not been verified here

The images have never been built. Registry access is blocked in the
environment this was written in (`403` pulling `python:3.11-slim`, the same
proxy policy that blocks `api.hyperliquid.xyz`), so `docker build` could not
run to completion once.

What *was* checked: `docker compose config` resolves the whole file
including both profiles and the interpolated command lines; `next build`
produces the `.next/standalone` tree the web image copies from, and
`BACKEND_URL` is confirmed baked into `routes-manifest.json` at build time,
which is why it is a build argument rather than an environment variable;
`tsc -p tsconfig.json` produces the `dist/` the backend image runs. The
layer assembly itself is unexercised — expect to fix something on the first
real build.
