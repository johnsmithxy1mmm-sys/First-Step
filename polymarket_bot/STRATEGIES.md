# Map of the territory: strategies and how the bot implements them

A general pattern worth accepting: **yield and "guaranteedness" are inversely
proportional everywhere**. #1 is nearly risk-free but microscopic in volume;
#5 can deliver multi-x but is no longer a system — it is a bet on your own
expertise.

**Recommended portfolio:** #3 as the cash-flow base + #1/#2 opportunistically
+ a small share of #5 in your niche. This is not financial advice, just a map
of the territory — the allocation decision is yours.

---

## #1. Structural arbitrage within Polymarket (negRisk)

In multi-outcome markets (elections, nominations) the sum of all candidate
prices sometimes deviates from $1 — buying the basket yields a mathematically
guaranteed profit regardless of the outcome. The purest edge on the platform,
but windows live seconds to minutes and are eaten by bots: you need a
websocket, speed and capital ready on the wallet. Returns of single-to-tens of
percent per year at near-zero risk, but capacity is hard-limited.

**Implementation:** `arbitrage.py` — both sides of the window: YES basket
(Σask < 1, pays $1) and NO basket (Σbid(Yes) > 1 ⟺ Σask(No) < n−1, pays $(n−1)).
Prefilter on Gamma prices → verification against live order books → sizing by
the minimum of leg depth and `max_stake_usd`. The `arbitrage:` section in
config.yaml; `execute: false` by default (detect + alert), 60-second interval.

**Honest limit:** this is REST polling, the "slow hunter" — it picks up what is
left after the websocket bots. The `min_profit_pct` threshold (1.5%) budgets
for the risk of a leg not filling.

## #2. Cross-platform arbitrage

The same outcome on Polymarket, Kalshi, Betfair is quoted with 2–10 pp gaps,
especially at news moments. The risks are not market but operational: different
resolution criteria (the main killer — an "identical" event resolves
differently), jurisdictions, capital locked across several venues.

**Implementation:** `crossmarket.py` — title matching (Jaccard over meaningful
words, threshold 0.65) Polymarket ↔ Kalshi (public REST, no auth), alert on a
gap >= 4 pp. **Auto-trading is deliberately absent**: a human reconciles the
two venues' resolution rules — every alert ends with a reminder to do so. Other
venues (Betfair etc.) plug in via the `ExternalVenue` protocol implementation.

## #3. Market making + liquidity rewards

The most "boring" and most reproducible strategy: quote two-sided liquid
markets, collect the spread and the rewards program. The main risk is adverse
selection ahead of news. The only strategy on this list where a typical month
is a plus, not a minus. **The portfolio's cash-flow base.**

**Implementation:** `marketmaker.py` — a two-sided quote via two buys (bid on
Yes at mid−s/2 + bid on No at 1−(mid+s/2); if both sides fill → Yes+No = $1 at
redemption, profit = spread). A cancel-replace cycle every 45 sec, partial
fills recorded to the ledger. Adverse-selection protection — automatic quote
pulling: a guard on mid movement (>=3 pp → cooldown), a guard on a volume spike
(daily > 50% of total turnover), an inventory-skew cap (the skewed side is
disabled). The kill-switch is shared with the rest of the bot.
`market_maker.enabled: false` by default — enable deliberately, the strategy
holds capital in quotes.

### #3b. Short-dated market making (fast capital turnover)

The core MM (#3) quotes markets 30+ days out — the same dollars stay locked in
quotes for weeks. This variant runs the **identical engine** on **liquid markets
that resolve within hours to a couple of days**, so the capital is freed at
settlement in 1-2 days and redeployed. It is the honest way to turn capital
*fast*: you still earn spread + maker rebate, you do **not** bet on direction
(the taker fee makes short-horizon direction bets negative-EV).

The trade-off is higher adverse selection near resolution, so the risk profile
is tighter: it quotes only inside a `[min, max]` **hours** window (staying OUT
of the final settlement window where direction, not spread, moves the price),
requires real two-sided liquidity, leans harder against inventory, widens the
base spread, and pulls quotes on a smaller shock.

**Implementation:** `sprintmaker.py` — `SprintMaker` subclasses `MarketMaker`
(same quoting, guard, queue-preserving requote, paper/live fills) and only
swaps selection (`SprintScorer`: hours window + high liquidity, rewards not
required) and tags fills `strategy="sprint_mm"` for separate PnL / circuit
breaker / digest attribution. `sprint_mm.enabled: false` by default. Watch the
**SPRINT MM** funnel in `--mode diagnose`; an empty funnel is normal when
nothing liquid resolves that soon.

## #4. Resolution edge (rules lawyering)

Trading the gap between the market headline and the letter of the UMA
resolution rules. Does not scale, but is not eaten by bots either.

**Implementation:** a fundamentally manual strategy — you cannot honestly
automate reading rules "better than the crowd". Support in the code: (a) the
niche alerts of #5 include the full resolution-rules text and source — a gap
between headline and letter is visible the moment the market appears; (b) the
LLM signal of the longshot strategy receives the rules description in the
prompt; (c) the scanner drops markets without a resolution source and a clear
description — what catches others must not catch us.

## #5. Informational edge in a niche

Historically every loud private win on Polymarket came from people who knew a
narrow domain deeper than the crowd (a French trader on the US elections
commissioned his own polls). A workable angle is post-Soviet geopolitics and
crypto, where primary sources read faster than the Western crowd.

**Implementation:** `niche.py` — watchlists (`niche.watchlists` in config.yaml,
default: post-soviet + crypto). The module removes the lag between "a market
appeared in my niche" and "I saw it": an instant Telegram alert on every new
market with price, volume, horizon and the full resolution rules. From there a
human decides. Deduplication via the `seen_markets` table in the ledger.

## Outside prediction markets: funding rate arbitrage (carry)

Long spot / short perp under positive funding — historically 5–15% per year in
USDC with limited risk, but with exchange risk and the risk of funding slipping
negative.

**Not implemented in this bot** — deliberately: different venues (CEX), a
different risk profile, different keys. Mixing this with the Polymarket bot in
one process is poor architecture; if you do it, do it as a separate service.

---

## How to turn on portfolio mode

```yaml
# config.yaml
market_maker:
  enabled: true        # #3 — the base (break it in on dry-run first!)
arbitrage:
  enabled: true
  execute: true        # #1 — opportunistically
crossmarket:
  enabled: true        # #2 — alerts, you trade by hand
niche:
  enabled: true        # #5 — alerts in your niche
```

All strategies live in one process on their own intervals, write to a shared
ledger with a `strategy` column (separate PnL attribution) and obey a shared
drawdown kill-switch.
