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

## Six income sources in one system

**1. Market making + liquidity rewards (the core).** Three income streams on
the same orders: the bid-ask spread, the maker rebate and a share of
Polymarket's daily rewards pool. The only strategy on the platform where the
edge is structural — it does not require guessing outcomes. Inside: fair
value from the microprice, inventory skew, requote hysteresis, and a
reward-scoring model implemented from Polymarket's own published formula
(`S(v,s) = ((v-s)/v)^2 * b` — quadratic in distance from the midpoint) rather
than a rough proxy, so the quote sits where the score is actually earned
instead of at the edge of the band where it is nearly worthless.

**2. Structural arbitrage of neg-risk baskets.** When the sum of all candidate
prices for an event deviates from $1 — buying the basket yields a profit under
any outcome. Legs are placed fill-or-kill and confirmed by matched size, not by
order-id alone, so a partially-filled basket can never be booked as a complete,
riskless one.

**3. Chain (ladder) arbitrage.** A structural relationship BETWEEN two sibling
markets in the same event — "resolves by June" vs "resolves by July", or
"reaches $150k" vs "reaches $200k" — makes one YES outcome logically imply the
other. When the market violates that inequality, the worst-case payoff of the
correct two-leg position is proven (by an exhaustive payoff-matrix test, not
just argued) to be $1/set with no downside. This is not a strategy most bots
on the platform implement.

**4. A mispricing longshot scanner.** It does not scoop up cheap tails blindly
(that is mathematically loss-making) — it buys only when the signal ensemble
(historical frequencies, cross-market logic, an LLM estimate, momentum) yields
a probability multiples above the market price.

**5. Cross-platform divergences.** One outcome can be priced several points
apart on Polymarket and Kalshi — the bot finds the pairs and sends an alert
with a reminder to reconcile the resolution rules.

**6. Your informational edge.** Niche watchlists: a new market on your topic
(geopolitics, crypto — configurable) lands in Telegram within minutes of
creation, with price and full resolution rules. Whoever reads the primary
source first is the one who earns.

## Risk control that never turns off

- **A two-level kill-switch**: a pause on data loss (WS stream dead >10 sec →
  all quotes pulled automatically) and a full halt on a daily loss, drawdown
  from the peak, or a desync with the exchange — with a bulk-cancel of all
  orders and an instant alert. The kill-switch fails **closed**: a
  non-finite equity or exposure reading halts trading rather than being
  silently compared away, and every effect (pause actually blocks new
  orders, not just a status flag) is covered by a dedicated regression test,
  not just asserted by the code's shape.
- **One real exposure ceiling.** A single `$` cap is enforced at every entry
  path — market maker included — instead of being a number in a config file
  nothing reads.
- **Hard limits**: per market, per category (accounting for correlations), on
  total exposure, per day. Kelly sizing with a fractional coefficient.
- **Idempotency and fill confirmation**: after any restart the bot reconciles
  local order state with the exchange's actual open orders and positions
  before ever placing a duplicate. Every order path — entries, exits, and
  every arbitrage leg — confirms the exchange's matched size before writing
  a trade to the ledger; an order id alone is never treated as proof of a
  fill.
- **Limit orders only**: it physically cannot "eat" a thin book at a bad price.
  Platform fees are computed from Polymarket's own published formula
  (`theta * shares * price * (1-price)`), verified end-to-end against the
  documented worked example, not assumed to be a flat percentage.

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

**This codebase has been through a formal adversarial audit** (methodology and
every finding in `docs/audit/`), not just ordinary code review: 13 findings
across the money-recording and safety-critical paths, each proven with a
failing reproducer before being fixed, each fix backed by a permanent
regression test. The safety module's mutation-testing score — the fraction of
deliberately-injected logic bugs that a test actually catches — went from
3-of-7 to 7-of-7. That report, warts and all, ships with the code; nothing in
it was cleaned up for the pitch.

## Engineering

- Python 3.11+, pydantic typing of all API objects, **487 automated tests**
  (469 unit/integration + 18 adversarial reproducers kept as permanent
  regressions), including property-based tests (Hypothesis) on the ledger's
  accounting invariants and an exhaustive payoff-matrix proof for the
  ladder-arbitrage payout.
- Websocket order books with auto-reconnect, gap-detect on a monotonic clock,
  and numeric-boundary validation (a malformed feed cannot poison a quote with
  NaN or Infinity); a token-bucket rate limiter that sheds load without ever
  blocking the one call that cancels every order.
- A SQLite ledger with write-time validation (no trade can silently corrupt
  every downstream number), structured JSONL logs, a rich terminal dashboard.
- Telegram: instant alerts + a digest every 6 hours.
- One-command deployment: Docker/docker-compose included, or a systemd unit
  with autostart and restart.
- Secrets only in .env; the private key is used exclusively by the official SDK
  to sign orders.

## Who it is for

Those who want a systematic approach to prediction markets instead of manual
bets; who value risk control over yield promises; who are willing to go through
the break-in phases before live trading.

---

*PolyBot is a tool, not a guarantee of income. Trading on prediction markets
carries the risk of total loss of funds; historical or simulated returns do not
predict future ones. Jurisdictional compliance is on the user. This is not
financial advice. See [`LICENSE.md`](../LICENSE.md) for the full terms and
no-warranty disclaimer under which this software is provided.*
