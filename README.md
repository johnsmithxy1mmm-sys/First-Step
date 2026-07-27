# PolyBot — Autonomous Multi-Strategy Bot for Polymarket

A production-grade, event-driven trading system for Polymarket: market
making with liquidity-reward optimization, structural and ladder arbitrage,
a calibrated mispricing scanner, and a shared risk framework — all under one
process, all tested.

**Start here, depending on what you need:**

| You want to... | Go to |
|---|---|
| Understand what this is and why it's worth something (non-technical pitch) | [`polymarket_bot/PITCH.md`](polymarket_bot/PITCH.md) |
| Read the technical documentation, architecture, and how to run it | [`polymarket_bot/README.md`](polymarket_bot/README.md) |
| See the strategy-by-strategy breakdown and honest capacity limits | [`polymarket_bot/STRATEGIES.md`](polymarket_bot/STRATEGIES.md) |
| Know the licensing terms and the no-warranty disclaimer | [`LICENSE.md`](LICENSE.md) |

## Two audit trails, not one — both kept, neither cleaned up

This codebase went through two separate rounds of scrutiny, covering
different ground. Both reports ship as-is, findings and all — nothing here
was edited down for a pitch.

- **[`AUDIT_REPORT.md`](AUDIT_REPORT.md)** — engineering/economics correctness
  passes: the Polymarket fee formula verified against its documented worked
  example, the liquidity-reward scoring model implemented from Polymarket's
  published formula, and a concurrency pass (WS-thread blocking, order-price
  bounds, late-fill capture).
- **[`docs/audit/REPORT.md`](docs/audit/REPORT.md)** — a formal adversarial
  audit run against the money-handling and safety-critical paths (13
  findings, 2 Critical, each proven with a failing reproducer before being
  fixed, each fix backed by a permanent regression test in
  `polymarket_bot/tests/audit/`).

## Quick start

```bash
pip install -r polymarket_bot/requirements.txt
cp polymarket_bot/.env.example polymarket_bot/.env   # fill in your keys
python -m polymarket_bot.main --mode sync-config      # generates config.yaml
python -m polymarket_bot.main --mode dry-run --once   # sanity check
```

Full workflow (backtest → dry-run → paper → live) is in
[`polymarket_bot/README.md`](polymarket_bot/README.md#workflow-important).

---

*Not financial advice. Trading and running trading software on prediction
markets carries the risk of total loss of funds and may be restricted in
your jurisdiction. See [`LICENSE.md`](LICENSE.md) for the full terms.*
