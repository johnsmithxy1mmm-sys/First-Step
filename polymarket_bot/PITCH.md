# PolyBot — an autonomous trading system for Polymarket

**Not a "betting script" but a trading desk in one process: market making,
arbitrage, a longshot scanner and niche alerts — under a shared risk framework
that watches your money more strictly than you would yourself.**

---

## What it is

PolyBot is a production-grade bot for the world's largest prediction market.
It runs 24/7 without your involvement: it selects markets itself, places and
pulls quotes itself in milliseconds over a websocket stream, takes profit
itself and stops itself when something goes wrong. You switch it on once — and
after that you just read the Telegram digests.

## Five income sources in one system

**1. Market making + liquidity rewards (the core).** Three income streams on
the same orders: the bid-ask spread, the maker rebate and a share of
Polymarket's daily rewards pool (>$5M/mo). The only strategy on the platform
where the edge is structural — it does not require guessing outcomes. Inside:
fair value from the microprice, inventory skew, requote hysteresis, quoting
exactly inside each market's reward band.

**2. Structural arbitrage of neg-risk baskets.** When the sum of all candidate
prices for an event deviates from $1 — buying the basket yields a profit under
any outcome. The bot catches these windows against live order books from both
sides (YES and NO).

**3. A mispricing longshot scanner.** It does not scoop up cheap tails blindly
(that is mathematically loss-making) — it buys only when the signal ensemble
(historical frequencies, cross-market logic, an LLM estimate, momentum) yields
a probability multiples above the market price.

**4. Cross-platform divergences.** One outcome can be priced 4–10 pp apart on
Polymarket and Kalshi — the bot finds the pairs and sends an alert with a
reminder to reconcile the resolution rules.

**5. Your informational edge.** Niche watchlists: a new market on your topic
(geopolitics, crypto — configurable) lands in Telegram within minutes of
creation, with price and full resolution rules. Whoever reads the primary
source first is the one who earns.

## Risk control that never turns off

- **A two-level kill-switch**: a pause on data loss (WS stream dead >10 sec →
  all quotes pulled automatically) and a full halt on a daily loss, drawdown
  from the peak or a desync with the exchange — with a bulk-cancel of all
  orders and an instant alert.
- **Hard limits**: per market, per category (accounting for correlations), on
  total exposure, per day. Kelly sizing with a fractional coefficient.
- **Idempotency**: after any restart the bot reconciles with the exchange and
  never doubles positions. Reconcile every 60 seconds.
- **Limit orders only**: it physically cannot "eat" a thin book at a bad price.
  Platform fees are baked into every edge calculation.

## Transparency instead of promises

Every trade is written to the ledger with a full snapshot: the market price,
the model's estimate, each signal's contribution, the book at entry time. After
resolutions the bot computes hit rate, ROI, model Brier score vs the market and
per-strategy profit attribution itself — you see which edge source actually
works and which one is time to switch off.

The path to live money is staged and mandatory: a backtest on history →
dry-run (intentions without orders) → paper trading on the live stream → live
with small capital. Each transition is manual only; live mode requires an
explicit risk-acknowledgment flag.

## Engineering

- Python 3.11+, pydantic typing of all API objects, 123 automated tests
- Websocket order books with auto-reconnect and gap-detect; a token-bucket limiter
- A SQLite ledger, structured JSONL logs, a rich terminal dashboard
- Telegram: instant alerts + a digest every 6 hours
- One-command deployment: a systemd unit with autostart and restart
- Secrets only in .env; the private key is used exclusively by the official SDK
  to sign orders

## Who it is for

Those who want a systematic approach to prediction markets instead of manual
bets; who value risk control over yield promises; who are willing to go through
the break-in phases before live trading.

---

*PolyBot is a tool, not a guarantee of income. Trading on prediction markets
carries the risk of total loss of funds; historical or simulated returns do not
predict future ones. Jurisdictional compliance is on the user. This is not
financial advice.*
