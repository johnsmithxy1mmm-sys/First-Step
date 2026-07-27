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
