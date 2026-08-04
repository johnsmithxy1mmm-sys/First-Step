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
import atexit
import contextlib
import importlib.util
import pathlib
import signal
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / "polymarket_bot"

#: The one file currently holding a mutant, and what it said before.
#:
#: A `try/finally` around the pytest call is not enough, and this is not
#: theoretical -- it happened while this list was being written. `finally` does
#: not run when the process takes SIGTERM, so a `pkill`, a CI cancellation or a
#: timeout leaves the mutated source sitting in the working tree. What it
#: leaves is by construction the worst possible thing to leave: a plausible
#: one-token edit that the test suite is known not to catch, in a file the next
#: `git commit -a` will happily pick up. A tool whose whole job is to ask
#: whether the tests would notice a wrong line must not be the thing that
#: quietly writes one.
#: A list holding at most one entry rather than a rebindable `path | None`, so
#: the handlers below mutate it in place instead of needing `global`.
_IN_FLIGHT: list[tuple[pathlib.Path, str]] = []


def _restore_in_flight() -> None:
    while _IN_FLIGHT:
        path, original = _IN_FLIGHT.pop()
        with contextlib.suppress(OSError):
            path.write_text(original, encoding="utf-8")


def _install_restore_handlers() -> None:
    """Restore on the ways a process ends without unwinding.

    `atexit` covers a normal exit and an unhandled exception; the signal
    handlers cover SIGTERM and SIGINT, which bypass it. Each handler restores
    and then re-raises the default disposition, so the exit status still says
    the process was signalled rather than pretending it finished.
    """
    atexit.register(_restore_in_flight)

    def _on_signal(signum, _frame):
        _restore_in_flight()
        signal.signal(signum, signal.SIG_DFL)
        signal.raise_signal(signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(ValueError, OSError):
            signal.signal(sig, _on_signal)

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

#: The risk engine's mutants, on the same terms and in a separate list only
#: because they run against a different suite.
#:
#: This half exists because a systematic AST sweep over the engine found blind
#: spots the hand-written list above could not have predicted, and a one-off
#: sweep that is not re-run is a finding with a shelf life. Each entry below is
#: a mutant that SURVIVED that sweep, was diagnosed, and is now killed by a
#: named test; keeping them here is what stops the gap reopening.
#:
#: Deliberately NOT in this list, having been checked and found equivalent:
#:
#:   - `cross_alive &= cross_gap_prev > 0` and its isolated twin, changed to
#:     `>= 0`. Every observable is bit-identical, because the per-step rule
#:     (`dead = alive & (gap <= 0)`) kills the same path at s=1 whatever the
#:     t=0 guard did. Measured on `cross_liquidated`, `isolated_liquidated`,
#:     `terminal_equity` and `funding_paid`.
#:   - `mmr * side` -> `mmr / side` in `_liq_from_margin_available`: `side` is
#:     exactly +/-1, so the two agree identically.
#:   - `max_iter: int = 12` -> 13 in `_tier_consistent`: the loop converges in
#:     about two passes on every tier table in `meta`.
#:
#: Listing a survivor that cannot be killed would report a test gap that does
#: not exist, so those are named here rather than counted there.
RISK_ROOT = "risk_engine"
RISK_MUTANTS: list[tuple[str, str, str, str]] = [
    # --- §3.1's gate: the checks that decide whether the model ships ---
    ("shadow/metrics.py", 'return row["external_flow_usd"] == 0.0',
     'return row["external_flow_usd"] != 0.0',
     "no-flow cohort inverted (scores exactly the rows it excludes)"),
    ("shadow/metrics.py", 'return not row["book_changed"]',
     'return row["book_changed"]',
     "book-unchanged cohort inverted (§3.3's own gate population)"),
    ("shadow/metrics.py", "crps_beats_baseline_a=bool(m_p.n and means[VARIANT_MODEL] <",
     "crps_beats_baseline_a=bool(m_p.n and means[VARIANT_MODEL] <=",
     "a tie counts as beating baseline A (§0.2 asks for an improvement)"),
    ("shadow/metrics.py", "crps_beats_baseline_b=bool(m_p.n and means[VARIANT_MODEL] <",
     "crps_beats_baseline_b=bool(m_p.n and means[VARIANT_MODEL] <=",
     "a tie counts as beating baseline B"),
    ("shadow/metrics.py", '("PIT uniform (KS p>0.05)", self.ks_pvalue > 0.05)',
     '("PIT uniform (KS p>0.05)", self.ks_pvalue >= 0.05)',
     "KS row passes at exactly the p it names"),
    ("shadow/metrics.py", "if cohort.n_days >= 2:", "if cohort.n_days > 2:",
     "clustered interval withheld on the first day it could be computed"),
    ("shadow/metrics.py", "return self.clustered_ci[0] <= self.target",
     "return self.clustered_ci[1] <= self.target",
     "VaR@95 row read off the wrong end of the clustered interval"),
    # --- §2: the within-step correction, which had no test at all ---
    ("liquidation/simulator.py", "d = self._cross_size[None, :] - mmr * np.abs",
     "d = self._cross_size[None, :] - mmr / np.abs",
     "bridge gap-variance drift term (invisible at |size| == 1)"),
    ("liquidation/simulator.py", "d = self._iso_size[None, :] - mmr * np.abs",
     "d = self._iso_size[None, :] - mmr / np.abs",
     "same, on the isolated branch"),
    ("liquidation/simulator.py", "return np.where(var > 0", "return np.where(var >= 0",
     "0/0 leaks NaN into a hit probability (reads as 'survived')"),
    ("liquidation/simulator.py", "p = np.exp(-2.0 * g0 * g1 / var)",
     "p = np.exp(-3.0 * g0 * g1 / var)",
     "first-passage exponent (P(liq) understated, the §10 direction)"),
    # --- §1.3: the tier the answer lands in, not the tier it started in ---
    ("liquidation/margin.py", "if nxt == mmr:", "if nxt != mmr:",
     "tier-consistency convergence test inverted"),
    ("liquidation/margin.py", "return min(candidates, key=lambda p: abs(p - reference))",
     "return max(candidates, key=lambda p: abs(p - reference))",
     "two-cycle tie-break takes the LATER warning (anti-conservative)"),
    # --- §5.1 parsing: poison that evades the guards downstream ---
    ("market/parse.py", "if (closes <= 0).any():", "if (closes <= 1).any():",
     "every sub-dollar perp refused as a non-positive close"),
    # --- B4's collector: the exit code IS the operator's diagnosis ---
    ("market/collect_addresses.py", "elapsed = now - started",
     "elapsed = now + started",
     "elapsed read off the absolute clock (aborts every real run instantly)"),
    ("market/collect_addresses.py", "if n < required and not allow_short:",
     "if n <= required and not allow_short:",
     "a harvest that meets §3.3 exactly is refused as short"),
]

RISK_TEST_PATHS = ["risk_engine/tests"]


def _have_pytest_cov() -> bool:
    """Whether `--no-cov` is a flag pytest will accept.

    It is only defined by the pytest-cov plugin. Passing it unconditionally
    made this script fail with argparse's "unrecognized arguments: --no-cov"
    whenever the plugin was absent -- which `main` then reported as
    "baseline (unmutated suite must be green): FAILED", a statement about the
    suite rather than about the environment. The suite was green. A mutation
    check that misdiagnoses a missing plugin as a broken test suite sends
    whoever ran it to debug the wrong thing entirely.
    """
    return importlib.util.find_spec("pytest_cov") is not None


def run_tests(test_paths: list[str] | None = None) -> bool:
    # `--no-cov` is a speed measure, not a correctness one: coverage
    # instrumentation across dozens of mutant runs is the bulk of the wall
    # clock. Dropping it when the plugin is absent changes nothing about what
    # is measured.
    cmd = [sys.executable, "-m", "pytest", *(test_paths or TEST_PATHS), "-x", "-q",
           "--no-header", "-p", "no:cacheprovider"]
    if _have_pytest_cov():
        cmd.append("--no-cov")
    # S603: every element of `cmd` is a literal or `sys.executable`; nothing
    # here comes from a caller. check=False is the point -- a non-zero exit is
    # the signal this function exists to report, not an error to raise on.
    proc = subprocess.run(  # noqa: S603
        cmd, cwd=ROOT, capture_output=True, text=True, check=False
    )
    return proc.returncode == 0


def _run_set(name: str, root: pathlib.Path, mutants, test_paths: list[str],
             width: int) -> list[str] | None:
    """One project's mutants against its own suite. None means the baseline
    was already red, which makes every result below it meaningless."""
    print(f"=== {name} ===")
    print("baseline (unmutated suite must be green):", end=" ", flush=True)
    if not run_tests(test_paths):
        print("FAILED — fix the suite before trusting mutation results")
        return None
    print("green\n")

    survived: list[str] = []
    checked = 0
    for fname, old, new, label in mutants:
        path = root / fname
        original = path.read_text(encoding="utf-8")
        if old not in original:
            print(f"  SKIP      {fname:<{width}} {label} (pattern not found — "
                  f"code moved, update this mutant)")
            continue
        checked += 1
        _IN_FLIGHT.append((path, original))
        path.write_text(original.replace(old, new, 1), encoding="utf-8")
        try:
            still_green = run_tests(test_paths)
        finally:
            # `finally` for the ordinary paths; `_IN_FLIGHT` and the handlers
            # installed in `main` for the ones that never unwind.
            _restore_in_flight()
        if still_green:
            survived.append(f"{fname}: {label}")
            print(f"  SURVIVED  {fname:<{width}} {label}")
        else:
            print(f"  killed    {fname:<{width}} {label}")

    print(f"\n{name} score: {checked - len(survived)}/{checked} mutants killed\n")
    return survived


def main() -> int:
    _install_restore_handlers()
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", help="only mutants in this file")
    ap.add_argument("--project", choices=("bot", "risk"),
                    help="only one project's mutants (default: both)")
    args = ap.parse_args()

    sets = []
    if args.project in (None, "bot"):
        sets.append(("polymarket_bot", PKG,
                     [m for m in MUTANTS if not args.module or m[0] == args.module],
                     TEST_PATHS, 12))
    if args.project in (None, "risk"):
        sets.append(("risk_engine", ROOT / RISK_ROOT,
                     [m for m in RISK_MUTANTS if not args.module or m[0] == args.module],
                     RISK_TEST_PATHS, 28))

    if not any(mutants for _, _, mutants, _, _ in sets):
        print(f"no mutants defined for {args.module}")
        return 0

    survived: list[str] = []
    for name, root, mutants, test_paths, width in sets:
        if not mutants:
            continue
        out = _run_set(name, root, mutants, test_paths, width)
        if out is None:
            return 1
        survived.extend(f"{name}/{s}" for s in out)

    if survived:
        print("SURVIVORS — these behaviours are not actually verified:")
        for s in survived:
            print(f"  - {s}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
