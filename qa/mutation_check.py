#!/usr/bin/env python3
"""Targeted mutation testing for the modules where a bug costs money.

WHY NOT mutmut/cosmic-ray over the whole repo: a full run takes hours and
reports a single repo-wide score, which is the least actionable number in
testing. Mutation is most useful as a POINTED question — "would the suite
notice if this specific guard inverted?" — asked of the code that guards
capital. Each mutant below is a real bug someone could plausibly introduce in
a refactor.

A SURVIVED mutant means the tests do not actually verify that behaviour; it is
a test gap, not a code bug. This is how F-009 was found: four survivors in
risk.py, including one that removed the pause term from `trading_allowed`,
i.e. nothing asserted that pausing actually stops trading.

Usage:
    python qa/mutation_check.py            # all mutants
    python qa/mutation_check.py --module risk.py

Exit code 1 if any mutant survives, so it can gate a release.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / "polymarket_bot"

# (file, original, mutated, description). Ordered by blast radius.
MUTANTS: list[tuple[str, str, str, str]] = [
    # --- kill-switch: the last line of defence ---
    ("risk.py", "return not self.halted and not self.paused",
     "return not self.halted", "trading_allowed ignores the pause"),
    ("risk.py", "loss >= self._cfg.max_daily_loss_usd",
     "loss > self._cfg.max_daily_loss_usd", "daily-loss boundary off by one"),
    ("risk.py", "(hwm - equity_now) / hwm >= self._cfg.max_drawdown_pct",
     "(hwm - equity_now) / hwm > self._cfg.max_drawdown_pct",
     "drawdown boundary off by one"),
    ("risk.py", "if self.halted or source in self._pause_sources:",
     "if self.halted and source in self._pause_sources:",
     "pause dedupe or->and (spurious alert while halted)"),
    ("risk.py", "ghost = exchange_order_ids - local_order_ids",
     "ghost = local_order_ids - exchange_order_ids",
     "reconcile compares the wrong direction"),
    ("risk.py", "if math.isfinite(value):", "if True:",
     "fail-closed guard on non-finite equity removed"),
    # --- fee/reward arithmetic: silently wrong money ---
    ("fees.py", "return coef * p * (1.0 - p)", "return coef * p",
     "taker fee drops the price term (the historical bug)"),
    ("fees.py", "* self.rebate_share(category, gamma_category))", "* 1.0)",
     "maker rebate share ignored"),
    ("rewards.py", "return ((max_spread - spread) / max_spread) ** 2 * size",
     "return ((max_spread - spread) / max_spread) * size",
     "reward score linear instead of quadratic"),
    # --- order pricing: an invalid price reaches the exchange ---
    ("clob.py",
     "if not (math.isfinite(price) and math.isfinite(tick)) or not MIN_TICK <= tick < 1:",
     "if tick <= 0:", "round_to_tick loses its non-finite / tick-range guard"),
    ("clob.py", "MIN_TICK = 1e-6", "MIN_TICK = 0.0",
     "denormal tick allowed (OverflowError in round())"),
    # --- write boundaries: poison entering the system of record ---
    ("ledger.py", "if not 0.0 < price < 1.0:", "if False:",
     "ledger accepts an untradable price"),
    ("ledger.py", "if size <= 0:", "if False:",
     "ledger accepts a non-positive size"),
    ("models.py", "if any(not math.isfinite(p) or not 0.0 <= p <= 1.0 for p in prices):",
     "if False:", "Gamma parser accepts NaN / out-of-range prices"),
    # --- control channel: the operator's remote kill must stay reachable ---
    ("telegram_control.py", "            self._offset += 1", "            pass",
     "unreadable batch no longer advances the offset (channel wedges)"),
    ("telegram_control.py", "if not isinstance(chat_obj, dict):", "if False:",
     "chat shape unchecked (AttributeError costs the rest of the batch)"),
    ("telegram_control.py", "if not chat or chat != str(self._chat_id):",
     "if False:", "AUTH REMOVED: any chat can drive /pause"),
    # --- payoff shape and the fade exit ---
    # Each of these mutants restores a state the code was actually in, and the
    # paper run that state produced booked $846 of worst case against $37 of
    # maximum upside, with no rule anywhere that could refuse or unwind it.
    ("fade.py", "if payoff_ratio(entry_no) < cfg.min_payoff_ratio:",
     "if False:", "shape gate removed (a 99-wins-per-loss leg is enterable)"),
    ("fade.py", "return (1.0 - entry_price) / entry_price",
     "return 1.0 - entry_price",
     "payoff ratio drops its denominator (shape gate silently loosened)"),
    ("fade.py",
     "if cfg.min_edge_per_day > 0 and edge / max(days, 0.5) < cfg.min_edge_per_day:",
     "if False:", "IRR floor removed (capital locked for months at ~1%)"),
    ("fade.py", "if (1.0 - mark) >= tail_entry * cfg.tail_stop_multiple:",
     "if (1.0 - mark) >= tail_entry:",
     "tail stop fires on any adverse tick (churns the book on noise)"),
    ("fade.py", "if (mark - entry) / tail_entry >= cfg.early_take_captured:",
     "if False:", "early take removed (capital sits out the last cent)"),
    ("fade.py", "if (mark - entry) / tail_entry >= cfg.early_take_captured:",
     "if payoff_ratio(mark) < cfg.min_payoff_ratio:",
     "take keyed to remaining shape again (closes legacy legs at a loss)"),
    ("fade.py", "if not 0.5 <= entry < 1.0 or not 0.0 < mark < 1.0:",
     "if False:",
     "exit direction guard removed (tail logic runs on cheap longshot legs)"),
    # --- capital allocation and fill measurement ---
    # NOT mutated: `if directional_room <= 0: return None`. Deleting it is an
    # EQUIVALENT mutant — a negative room flows into `min(size, room)` and the
    # `size < min_order_notional` floor below returns None anyway. No test can
    # kill it because the behaviour is identical, so counting it as a survivor
    # would be reporting a test gap that does not exist. The reserve itself is
    # mutated instead, which does change behaviour:
    ("portfolio.py",
     "reserve = max(0.0, min(cfg.reserve_for_mm_pct, cfg.max_total_exposure_pct))",
     "reserve = 0.0",
     "MM reserve set to zero (directional book eats the whole account)"),
    ("portfolio.py",
     "- self._ledger.total_exposure(self._mode, DIRECTIONAL_STRATEGIES))",
     "- self._ledger.total_exposure(self._mode))",
     "reserve charges MM inventory against itself (locks the MM out)"),
    # Mutate the GUARD, not the assignment inside it: with the guard intact the
    # assignment only ever runs on the first row, so changing it is equivalent.
    # Dropping the guard is what restores last-row-wins.
    ("ledger.py", 'if not slot["strategy"]:', 'if True:',
     "PnL attributed to whoever SOLD (a fade loss booked against longshot)"),
    ("ledger.py", 'f"ALTER TABLE shadow_quotes ADD COLUMN {col} REAL")',
     'f"ALTER TABLE shadow_quotes ADD COLUMN {col} REAL DEFAULT 0.0")',
     "migration fabricates a measurement for every pre-existing row"),
    ("ledger.py", '"WHERE rested_sec = 0.0")', '"WHERE 0")',
     "already-fabricated rows are left reading as a measured zero"),
    ("marketmaker.py",
     "        frac = score_fraction(half, quote.market.rewards_max_spread)",
     "        frac = 0.0",
     "reward accrual unrecorded (a no-flow market's ONLY income is invisible)"),
    ("marketmaker.py", 'quote.observe_ask("yes", yes_top.ask)', "pass",
     "tape range unrecorded (a dead market reads like a too-wide spread)"),
    ("marketmaker.py", "self._paper_fills(ws_only=True)", "pass",
     "fills sampled per cycle again (crossings between cycles go unrecorded)"),
    ("marketmaker.py",
     "            return self._top_source(token) if self._top_source is not None else None",
     "            return self._top(token)",
     "WS fill path may fall back to blocking REST (stalls the recv loop)"),
]

TEST_PATHS = ["polymarket_bot/tests"]


def run_tests() -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *TEST_PATHS, "-x", "-q", "--no-header",
         "-p", "no:cacheprovider", "--no-cov"],
        cwd=ROOT, capture_output=True, text=True)
    return proc.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", help="only mutants in this file")
    args = ap.parse_args()

    mutants = [m for m in MUTANTS if not args.module or m[0] == args.module]
    if not mutants:
        print(f"no mutants defined for {args.module}")
        return 0

    print("baseline (unmutated suite must be green):", end=" ", flush=True)
    if not run_tests():
        print("FAILED — fix the suite before trusting mutation results")
        return 1
    print("green\n")

    survived: list[str] = []
    for fname, old, new, label in mutants:
        path = PKG / fname
        original = path.read_text(encoding="utf-8")
        if old not in original:
            print(f"  SKIP      {fname:<12} {label} (pattern not found — "
                  f"code moved, update this mutant)")
            continue
        path.write_text(original.replace(old, new, 1), encoding="utf-8")
        try:
            still_green = run_tests()
        finally:
            path.write_text(original, encoding="utf-8")   # always restore
        if still_green:
            survived.append(f"{fname}: {label}")
            print(f"  SURVIVED  {fname:<12} {label}")
        else:
            print(f"  killed    {fname:<12} {label}")

    total = len([m for m in mutants if (PKG / m[0]).read_text(encoding="utf-8").count(m[1])])
    print(f"\nscore: {total - len(survived)}/{total} mutants killed")
    if survived:
        print("\nSURVIVORS — these behaviours are not actually verified:")
        for s in survived:
            print(f"  - {s}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
