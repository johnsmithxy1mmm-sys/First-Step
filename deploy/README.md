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

Before `--live`, run the verification harness. It checks the assumptions the
live path rests on and exits non-zero while any is contradicted or unproven,
naming each separately:

```bash
python -m risk_engine.market.verify --address 0x<an-account-you-can-read> --report verify.json
```

Pass `--address`: without it the B2 check (ledger delta types) cannot run at
all, and B2 is the one that has already caught a real defect — the venue
returned a `send` type this build could not classify, which would have
surfaced in production as a resolver failure retried forever behind a gate
that never advanced. Any address with ledger history works; it is public
read-only data.

This README said the parsers "have never seen a live response" until
2026-07-31. They have: E5 passed on 2026-07-29, B2 and E4 on 2026-07-31, C2
and C5 on 2026-07-30/31. What is still unproven is C1 (a protocol constant no
sample of realised rates can establish) and C4 (a WebSocket question this
harness cannot ask). See OPEN-QUESTIONS C1, C2, C4, C5, E5 and the status
table in `risk_engine/README.md`. Run the harness rather than trusting either
— a table records a past run, and the venue can change under it.

## Starting the shadow clock

The shadow jobs sit behind a compose profile, so `up` does not start them.
Starting the §3.3 counter is a decision, not a side effect:

```bash
docker compose -f deploy/docker-compose.yml --profile shadow up -d
docker compose -f deploy/docker-compose.yml run --rm engine \
  -m risk_engine.shadow progress --journal "$SHADOW_DSN"
```

**Read `risk_engine/README.md` before you do.** §3.3 resets the window to zero
whenever the distribution moves, so every question that moves it has to be
settled before days start accumulating.

The last one was **A9, settled 2026-07-31**: the copula's degrees of freedom
are now fitted rather than hardcoded, `MODEL_VERSION` is `0.3.0-phase1`, and
that bump reset the counter. It cost nothing because the counter had not
started — which is the entire reason it was done then rather than later. A1
and A8 were decided on 2026-07-30 without moving the distribution; C2, C4, C5
and B2 are closed against live data; C1 is open but non-blocking (the clamp
sits ~1760× above anything the market has done, so it never activates and
cannot move the distribution being accumulated).

### Pre-flight, in order

Each of these is cheap now and expensive once days are accumulating.

```bash
# 1. Every ledger delta type in the FRAME, not just one account. An
#    unclassifiable type is retried forever rather than reported, so it
#    withholds an address silently while the counter fails to advance.
python -m risk_engine.market.verify --addresses deploy/addresses.json \
  --frame-sample 50 --address 0x<any-account-with-history>

# 2. The distribution is frozen at what the journal will record.
python -c "from risk_engine.version import MODEL_VERSION; print(MODEL_VERSION)"

# 3. The gate still passes under that version.
python -m risk_engine.validation.cli benchmarks
```

Step 1 is the one that earns its keep. One real account produced five delta
types, three of which had to be classified from live records — and two of
those were found in consecutive runs *on the same address*. A second address
would have found both at once; fifty finds what the cohort actually holds.

### What the cron does when it fails

`SHADOW_INTERVAL_S` (default 86400) is the **period**, not the gap between
runs: the loop sleeps for the remainder after the sweep, because a 500-address
sweep takes ~33 minutes at §5.3's 25% share and the naive form would drift the
snapshot 33 minutes later every day — 21 snapshots would need 21.5 calendar
days, compounding silently against a gate measured in days.

Failures **escalate to a crash-looping container** rather than a log line:
three consecutive snapshot failures, or twelve consecutive resolve failures
(~12 h), exit non-zero. The previous form swallowed every failure into `echo`
while the container stayed healthy, so a cron that failed for three weeks
looked exactly like one that worked. Check `docker compose ps` and the exit
code, not just that the container exists.

Live runs also need `deploy/addresses.json` filled in — both the list and the
`frame` field describing what it is a sample *of*. `FileAddressSource`
refuses a list without one (OPEN-QUESTIONS B4). Each address is `0x` plus 40
hex digits in any case; the checksummed form from a block explorer is fine, it
is folded to lowercase so one account cannot be journalled twice under two
spellings. A malformed entry fails the load, naming its index.

Generate it rather than curating it by hand — the frame for the §3.3 window is
decided (the public trades feed, activity-selected; the leaderboard was
rejected because it ranks on the very outcome being calibrated):

```bash
pip install 'websockets>=12.0'     # not an engine dependency; imported lazily
python -m risk_engine.market.collect_addresses --minutes 30 \
    --out deploy/addresses.json --force
```

That writes the `frame` text as well as the list. Editing the committed
template by hand instead loses the address-format guidance next to the data,
which is how the file in this directory already came to differ from what
`shadow init-addresses` emits. Note that the file is bind-mounted read-only
into both shadow containers while `SHADOW_ARGS` defaults to `--fixture`;
flipping it to `--addresses /app/addresses.json` before the file has a frame
fails both jobs at startup.

Read the exit code rather than the fact that it was non-zero — they mean five
different things and only one of them is fixed by a longer window: `1`
collected but refused to publish (short of §3.3's 200, or `--out` exists), and
the addresses are parked in `deploy/addresses.json.refused-<window-start>`
rather than discarded; `2` the feed did not match the collector's assumed
message shape, which is a code problem, not weather; `3` no usable connection
at all — check `--ws-url` first; `4` a bad invocation, caught before anything
connects; `5` the harvest succeeded and only the *write* failed (read-only
mount, full disk, dangling symlink), in which case the complete file is on
stdout and needs redirecting, **not** another collection window. `5` was
missing from this list until 2026-07-31, and reading it as `1` is precisely
the thirty-minute mistake the two codes are separate to prevent. A run that
succeeds but
prints a `NOTE:` line about anomalies harvested a list the feed partly
disagreed with; the count and the offending records are in the file's
`_provenance.anomalies`.

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
