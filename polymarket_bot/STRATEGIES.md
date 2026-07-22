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

**Honest limit:** discovery is REST polling — the full board is only re-walked
every cycle. But tracked structures are faster than that: the polling job
registers every prefiltered basket (and chain pair) with the WS fastlane
(`arb_fastlane.py`), so a tick on any of their tokens triggers an immediate
re-verification against live books in a dedicated worker thread (per-structure
cooldown stops tick storms from hammering the API). New windows are still found
at polling speed; known windows are re-checked at tick speed. The
`min_profit_pct` threshold (1.5%) budgets for the risk of a leg not filling.

### #1b. Chain (ladder) arbitrage — across nested sibling markets

Same near-risk-free character as #1, but between two DIFFERENT sibling
markets in the same event that describe the same underlying fact at two
"checkpoints" — not officially flagged mutually-exclusive outcomes, but a
LOGICAL implication forced by how the two questions are worded:

  * **DATE ladder** — "Fed cuts by June" vs "Fed cuts by July": an absorbing
    fact (once true, stays true), so a later deadline can never be *less*
    likely than an earlier one.
  * **VALUE ladder** — "BTC reaches $150k by <date>" vs "$200k by <date>": a
    continuous price path, so a higher threshold can never be *more* likely
    than a lower one on the same deadline.

If the market prices violate that ordering, buying the underpriced side of
BOTH legs (YES on the underpriced superset + NO on the underpriced subset)
has a worst-case payout of $1/set and best case $2/set — it cannot lose by
construction, only by how much it wins.

**Implementation:** `chainarb.py` — `classify_pair()` infers the relationship
from question text (same event, same template with only the date/value token
differing, verb/marker checks for the correct monotonic direction), then
`verify()` prices it against live order books. Unlike negRisk, the pairing is
INFERRED, not an official Polymarket flag — `classification_haircut` reserves
margin against a bad match. `execute: false` by default (detect + alert).

**Fast turnover:** unlike fade (dead until a single far resolution), a chain
pair does not have to wait for either leg's own resolution — once the market
corrects the mispricing, or the earlier/subset leg resolves (informative for
the far leg), both legs can usually be sold back at a profit well before
either one's own end date.

**Honest limit:** the inference is text-heuristic, not a platform guarantee —
run `--mode diagnose` (the CHAIN ARB table) to see how many pairs classify
and how many currently show a Gamma-price violation before turning on
`execute: true`.

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

## The learning loop — nothing stays a constant-by-decree

The bot records its own market data and outcomes, then re-fits its parameters
from that history. Every learner starts as the prior/identity and only moves
as evidence accumulates (all pure functions of recorded data, tested offline):

- **Tick recording** (`tickstore.py`) — every WS top-of-book update and a
  per-cycle per-category price-change index are streamed to their own sqlite
  (non-blocking enqueue, drop-oldest under backpressure, retention-pruned).
  This is the raw material; without it, "self-calibration" is just a word.
- **Category correlations** (`CorrelationLearner`) — Pearson correlations of
  the recorded category-index series replace the expert VaR/sizing matrix,
  pair by pair, once a pair has enough aligned history (clamped off ±1 so VaR
  never degenerates). Unlearned pairs keep the expert prior.
- **Fill calibration** (`FillCalibrator`) — every MM quote is labeled at death
  (filled-before-cancel or not) against the fill probability predicted at
  placement; the learner maps predicted → realized and feeds the corrected
  probability back into the queue-preservation decision.
- **Fade bias, MM markout, Platt** — as before (per-bucket tail bias, per-
  market adverse-selection spread widening, p_est recalibration).
- **Acting Sharpe allocator** (`alloc.py`) — the digest's per-strategy Sharpe
  weights move sizing multipliers inside a clamped corridor (default
  0.7–1.3, EMA-smoothed). Capital tilts toward what is actually earning, but
  **every hard risk cap applies after the multiplier** — it can tilt, never
  break a limit. The circuit breaker still owns on/off.
