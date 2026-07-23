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
touch it). The repo ships `config.example.yaml` — and the two stay in sync
automatically:

* **At load time** the example is the BASE layer and your `config.yaml` is an
  OVERLAY on top of it. Your file only needs the values you changed; every
  section you did not write comes live from the example. A `git pull` that
  ships a new strategy block activates it immediately with the tuned example
  values — no hand-copying. Anything you wrote always wins over the example.
* **`python -m polymarket_bot.main --mode sync-config`** materializes the
  missing sections into your `config.yaml` (comments included) so you can see
  and edit them. Append-only: sections and values you already have are never
  touched. Run it after a `git pull` whenever you want new blocks visible in
  your file.

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

## v2: event-driven, self-calibrating, portfolio-aware

Beyond the base strategies, the bot has a second layer that makes it react
faster, learn from itself, and understand its own risk:

| Area | Module | What it does |
|---|---|---|
| Event-driven core | `ws_feed.py` + `main.on_tick` | WS ticks drive instant take-profit exits and per-market MM reprices (ms, not up to 30s); scheduler keeps the slow work |
| Self-calibration | `calibration.py` | fade `bias_discount` learned per (category, price) from resolutions; per-market MM spread widened by realized markout; Platt recalibration of p_est |
| Event-aware portfolio | `portfolio.py` | a neg-risk basket's risk is its largest leg, not the sum (freeing room for self-hedged clusters); fade IRR planner (edge ÷ days) |
| MM 2.0 | `microstructure.py`, `marketmaker.py` | volatility-scaled spread/skew (Avellaneda-Stoikov), queue/fill-probability model, rewards-weighted quote sizing |
| Resolution alpha | `resolution.py` | near-riskless carry on effectively-decided markets, fee + UMA-dispute reserved |
| LLM rules lawyer | `ruleslawyer.py` | Claude compares headline vs the letter of the rules, flags exploitable gaps |
| Smart money | `smartmoney.py` | a profitable watched wallet's tail entry becomes a bounded ensemble signal |
| Risk 2.0 | `risk2.py` | portfolio stress + VaR, market-data anomaly guard, per-strategy circuit breaker |
| Research | `research.py` | realistic queue/trade-through fill sim, Sharpe allocation, walk-forward tuning (`--mode autotune`, proposal only) |
| Ops | `ops.py`, `Dockerfile` | Prometheus `/metrics` + `/health`, SIGHUP config hot-reload, docker-compose (bot + Prometheus) |

All of these default OFF or to the prior behavior; the aggressive
`config.example.yaml` profile enables the sound ones. Extra CLI modes:
`--mode diagnose` (selection funnels incl. fade & resolution), `--mode report`
(PnL, markout, learned bias, event risk, VaR), `--mode autotune`.

## Tests

```bash
pytest polymarket_bot/tests/ -q     # 177 offline tests: edge filter, Kelly,
                                    # coherence, idempotency, ledger, fade, MM,
                                    # calibration, netting, resolution, risk2, ...
```
