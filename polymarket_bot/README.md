# Polymarket Multi-Strategy Bot

A production-grade autonomous bot for Polymarket: the base strategy is a
**"barbell on the mispricing of tail outcomes"**, plus a portfolio of
additional strategies (see [STRATEGIES.md](STRATEGIES.md)):

| # | Strategy | Module | Default mode |
|---|---|---|---|
| **CORE** | **MM + liquidity rewards farming** (microprice, inventory skew, requote hysteresis, rewards band from Gamma, fee-aware spread) | `marketmaker.py` + `scorer.py` | off — enable after the dry-run → paper phases |
| satellite | T-10s TA on 5-min BTC up/down (deterministic slugs, quarter-Kelly, FOK) | `satellite.py` | **off** (`satellite.enabled: false`) |
| 1 | Structural arbitrage of neg-risk baskets | `arbitrage.py` | detect + alert (the naive sum-to-one over REST is documented as eaten — execution not recommended) |
| 2 | Cross-platform divergences (Kalshi) | `crossmarket.py` | alerts only — a human reconciles the resolution rules |
| 5 | Niche watchlists + longshot barbell | `niche.py`, `scanner.py`+`estimator/` | alerts; longshots in dry-run |

Infrastructure (master prompt 2026): `ws_feed.py` — WS order books with
reconnect/heartbeat/gap-detect; `risk.py` — kill-switch (daily stop, drawdown
from HWM, WS disconnect >10s, reconcile desync every 60s → bulk-cancel + halt);
`fees.py` — Fee Structure V2 per category (maker rebate, configurable);
`ratelimit.py` — token bucket; `replay.py` — book recording + backtest replay.

**Phase order: `--mode dry-run` → `--mode paper` (>=7 days) → `--mode live`
(manual only). Phase 0: `python -m polymarket_bot.phase0_smoke` on your
machine — reconciles the live API (endpoints, rewards fields, SDK version) with config.yaml.**

All strategies run in one process on their own intervals, write to a shared
ledger (the `strategy` column gives separate PnL attribution) and obey a shared
kill-switch. The recommended portfolio and each strategy's honest limits are in
STRATEGIES.md.

## Strategy philosophy

The naive "buy hundreds of contracts at $0.005–0.03 and wait for multi-x"
strategy has **negative EV** because of favorite-longshot bias: cheap outcomes
on prediction markets are systematically overpriced. So the bot **does not buy
tails blindly** — it is a mispricing scanner. A longshot is bought only when
the bot's own probability estimate `p_est` is materially above the market price
`p_mkt`: by default `p_est / p_mkt >= 2.0` at `p_mkt <= 0.05`.

## Architecture

```
gamma.py ──┐                     ┌── base_rates.py  (historical frequencies, YAML)
           ▼                     ├── coherence.py   (cross-market logic, priority #1)
      scanner.py ──► estimator/ ─┼── llm.py         (Claude, structured JSON, cache)
   (L1+L2 filters)   (ensemble)  └── momentum.py    (informed flow)
                         │
                         ▼
                   portfolio.py  (fractional Kelly, 1%/10%/30% caps, kill-switch 25%)
                         │
                         ▼
                   executor.py   (maker limit orders only, repricing, idempotency)
                         │
                         ▼
                    ledger.py    (sqlite: trade snapshots, Brier, PnL attribution)
                         │
                         ▼
                    monitor.py   (rich dashboard + Telegram)   ◄── main.py (orchestrator)
```

### Estimator signals

| Signal | What it does | Confidence |
|---|---|---|
| **coherence** | Logical links: neg-risk baskets (sum(p) != 1) and calendar chains (P(by earlier date) <= P(by later date)). Inconsistency = structural edge | 0.7–0.9 |
| **base_rates** | Historical frequencies from `base_rates.yaml`: p = 1−(1−annual)^(days/365). Extensible reference | from YAML (0.25–0.5) |
| **llm** | Claude (`claude-sonnet-5`, structured JSON via the official SDK) estimates p_est from the question and resolution rules. 24h cache, per-cycle call limit | <= 0.7 |
| **momentum** | A tail price rise on a volume spike = informed flow | 0.15–0.4 |

The **ensemble** is a confidence-weighted geometric mean. The market price
always participates as an anchor with weight `market_anchor_confidence` (0.85):
for the ensemble to yield edge >= 2, a signal with confidence 0.5 must diverge
from the market by ~2^((0.85+0.5)/0.5) ≈ 6.5x. This is deliberate conservatism
against longshot bias; loosen the anchor after calibrating with a backtest.

## Installation

```bash
pip install -r polymarket_bot/requirements.txt
cp polymarket_bot/.env.example polymarket_bot/.env   # fill it in; .env is in .gitignore
```

## Workflow (important!)

```bash
# 1. Backtest on closed markets: calibrate p_est before a single live dollar
python -m polymarket_bot --mode backtest

# 2. Dry-run (DEFAULT): everything for real except the money. Orders are
#    virtual, PnL is computed on real resolutions. A mandatory stage before live.
python -m polymarket_bot                     # scheduled cycles
python -m polymarket_bot --once              # one cycle (handy for cron)

# 3. Live — real money, only after weeks of successful dry-run
python -m polymarket_bot --mode live --i-understand-the-risk
```

All strategy parameters are in `config.yaml` (scanner filters, edge threshold,
Kelly λ, caps, take-profit, cycle interval). Secrets go only in `.env`.

`config.yaml` is your personal config, kept **outside git** (code updates don't
touch it). The repo ships a template `config.example.yaml`. If you have no
`config.yaml` of your own, the bot reads parameters from the template; to
customize: `cp polymarket_bot/config.example.yaml polymarket_bot/config.yaml`
and edit the copy.

## What happens in one cycle

1. **scanner** — all active markets from Gamma; filters: price in [0.002, 0.05],
   24h volume >= $5k, bid depth >= $500 within 20% of mid, resolution in
   3–120 days, unambiguous resolution rules (a source or a clear description).
2. **estimator** — the signal ensemble gives p_est; every estimate with all
   signal contributions is written to the ledger.
3. **portfolio** — fractional Kelly (λ=0.15) with hard caps: <=1% of bankroll
   per market, <=10% per category (with an expert correlation matrix:
   geopolitics↔economy 0.5 etc.), <=30% total. Drawdown >=25% →
   **kill-switch**: observe-only + alert.
4. **executor** — maker limit orders only: bid just above best bid (never
   crossing ask), a timeout, up to 3 reprices, cancel if price moved above the
   edge threshold. Large orders are split into $200 children. Idempotency:
   before an order, local state is reconciled with the API's actual positions;
   if the reconcile is unavailable, the order is **not** sent (fail-safe).
5. **exits** — a position that rose >=7x → sell 60%, the remainder is a free
   lottery ticket. Resolutions are recorded in the ledger automatically.
6. **monitor** — rich dashboard (bank, drawdown, positions, top candidates,
   errors) + Telegram alerts on entries/exits/resolutions/kill-switch.

## Ledger analytics

`ledger.metrics()`: hit rate, average realized multiple, ROI, **model Brier
score vs market Brier** (the key test: do we beat the price), PnL attribution
by signal — which edge source actually works. Every trade stores a full
snapshot (p_mkt, p_est, signal contributions, the book at entry time).

## Backtest: method and honest limits

Closed Gamma markets + CLOB price history (`/prices-history`): the price is
sampled `lookback_days_before_end` (21 days) before resolution — the bot's
"entry". Report: market vs model Brier, a bucketed calibration table, and
simulated strategy ROI. Limits: entry at the sampled price is optimistic;
momentum and LLM do not participate in the backtest (no historical deltas; the
LLM would leak future knowledge).

## ⚠️ Risks

- Live mode requires an explicit `--i-understand-the-risk`. Bet only what you
  are ready to lose entirely.
- The dry-run fill model is optimistic (the maker bid is treated as filled);
  the real fill-rate is lower — factor that into your reading of the results.
- The private key is used only by the official `py-clob-client` to sign orders
  and is read exclusively from `.env`.
- Polymarket's jurisdictional restrictions are your responsibility.

## Tests

```bash
pytest polymarket_bot/tests/ -q     # offline tests: edge filter, Kelly,
                                    # coherence, idempotency, ledger, fade, MM, ...
```
