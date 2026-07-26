# System Model — polymarket_bot (audit/2026-07-26, Phase 0)

Recon only. No quality judgements in this document.

## 1. Entry points

| Entry | Path | Notes |
|---|---|---|
| CLI | `main.py main()` | modes: dry-run / paper / live / diagnose / report / chaos / backtest / leadlag / coherence / timemachine / capacity / sync-config / phase0 |
| Scheduler | `main.py:1113+` APScheduler `BackgroundScheduler` | ~15 interval jobs, each `max_instances=1, coalesce=True` — but **different jobs run concurrently** in the pool |
| WS callback | `WSFeed -> main.on_tick` | ms-latency path: tick recording, guardian, exits, arb flagging, MM/sprint reprice |
| Telegram | `telegram_control` long-poll | two-way commands, filtered by `TELEGRAM_CHAT_ID` |
| Webhook-like | none (no inbound HTTP server except Prometheus metrics bind 127.0.0.1) | |

## 2. Trust boundaries (data we do not control)

| Source | Enters via | Parsed in |
|---|---|---|
| Gamma REST (markets/events JSON) | httpx | `models.Market.from_gamma` |
| CLOB REST (books, order status) | httpx / py-clob-client | `clob.py` |
| CLOB WS frames | websockets | `ws_feed.BookStore.handle` |
| Data-API (positions) | httpx | `clob.api_positions` |
| Binance klines (satellite) | httpx | `satellite.py` |
| UMA/on-chain (redeemer) | web3 (opt-in) | `redeemer.py` |
| Telegram updates | httpx long-poll | `telegram_control.py` |
| LLM output (estimator) | anthropic SDK | `estimator/llm.py` (JSON from model) |
| env / .env | dotenv | keys, chat ids, RPC url |
| config.yaml | yaml.safe_load | layered under example |
| Wall clock | `time.time`, `datetime.now(utc)` | deadlines, TTLs, day boundaries |

## 3. State

| State | Where | Persistence |
|---|---|---|
| trades / resolutions / markouts / bank / quote_outcomes / opportunities | `ledger.sqlite` (single conn, `check_same_thread=False` + RLock) | durable |
| tick tape + category index | `ticks.sqlite` (writer thread + queue; readers open own conns) | durable, pruned by retention |
| book snapshots (replay) | `book_snaps.sqlite` | durable |
| L2 books | `BookStore` dicts + lock | memory |
| MM quotes/orders (`_quotes`, `_orders`, `_quoted`) | `MarketMaker` + `self._lock` | memory — **lost on restart; live reconcile via exchange `open_orders`/`api_positions`** |
| kill-switch state (halted, pause sources) | memory | lost on restart (restart = manual reset by design) |
| markets_cache, `_positions_by_token`, token maps | `main` memory, refreshed by cycle | memory |
| learned models (correlations, Platt, fill calib, toxicity, markout feedback) | memory, refit from sqlite every 30 min | derived |

## 4. Side effects (irreversibility ranked)

1. **Order placement / cancel** — `clob.Trader.buy_limit/sell_limit/cancel/cancel_all` (live only; every strategy gated by `execute:false` defaults + kill-switch + `--i-understand-the-risk`).
2. **On-chain redemption** — `redeemer` (opt-in: execute + web3 + key + RPC).
3. **Ledger writes** — trades/outcomes; feeds sizing, PnL, kill-switch, learning. Wrong write ⇒ wrong future decisions.
4. Telegram sends, Prometheus, logs — reversible noise.

## 5. Concurrency map (threads that touch shared state)

```
APScheduler pool ──> cycle / mm_job / arb_job / chain_arb_job / risk_job / markout_job / calibration_job ...
WS thread ─────────> on_tick: tickstore.enqueue, guardian, _exit_one(known_bid), fastlane.flag, mm.react_to_tick (non-blocking lock)
arb-fastlane thread> drain() -> basket/chain recheck + (gated) execute
tickstore writer ──> drains queue -> ticks.sqlite
tg-alerts thread ──> monitor queue -> Telegram
telegram_control ──> long-poll -> command handlers (pause/resume/status)
```
Shared: `Ledger` (RLock), `MarketMaker._lock`, `BookStore._lock`, monitor queues.
**Different scheduler jobs are mutually concurrent** (`max_instances=1` is per-job only): e.g. `cycle` and `mm_job` and `risk_job` can interleave on Ledger and Trader.

## 6. Hotspots (git, 12 mo; 68 commits, 7 fix/revert)

`main.py` (38 touches) ≫ `config.py` (29) > `marketmaker.py` (13) = `ledger.py` (13) > `fade.py` (9) > `chainarb.py` (8). Prior bugs this session clustered in: fees model, MM spread logic, WS-thread blocking, cancel/fill races — i.e. **money math and thread seams**, matching the hotspot files.

## 7. Known-fixed this session (do not re-report)

fee `θ·p·(1−p)`; rebate 0.35→per-category; MM 0.9-band dead zone; rate-limiter 10s on WS thread; 150s lock hold in `cycle`; `round_to_tick` 0/1 clamp; late fills on cancel (executor + MM); WS `side` book corruption; fastlane cooldown drop; chain-arb inversion/negation/implausible/FOK-unwind.
