"""Chaos drills: prove the safety systems actually fire under failure.

A kill-switch you never see trip is a kill-switch you don't know works. These
scenarios inject the failures that matter — a dead WS feed, corrupt market
data, an order the exchange knows about that we don't, a daily-loss breach, a
drawdown breach, a losing streak — and assert the right defense engaged. Run
before promoting to live:

    python -m polymarket_bot --mode chaos

Each scenario builds its own fresh safety components from the config (no shared
state, no network), returns (name, passed, detail), and is unit-tested. The
mode prints a pass/fail table and exits non-zero if any drill fails.
"""

from __future__ import annotations

from .config import BotConfig
from .ledger import Ledger
from .risk import KillSwitch
from .risk2 import MarketDataGuard, StrategyCircuitBreaker


def _ks(cfg: BotConfig, ledger: Ledger) -> KillSwitch:
    return KillSwitch(cfg, ledger, "paper", cancel_all=lambda: None,
                      alert=lambda _m: False)


def scenario_ws_outage(cfg, ledger) -> tuple[str, bool, str]:
    ks = _ks(cfg, ledger)
    ks.on_ws_disconnect(cfg.risk.ws_staleness_kill_sec + 5)
    ok = ks.paused and not ks.trading_allowed
    return ("WS outage -> trading paused", ok, f"paused={ks.paused}")


def scenario_ws_recovery(cfg, ledger) -> tuple[str, bool, str]:
    ks = _ks(cfg, ledger)
    ks.on_ws_disconnect(30)
    ks.on_ws_recovered()
    ok = ks.trading_allowed
    return ("WS recovers -> trading resumes", ok, f"allowed={ks.trading_allowed}")


def scenario_data_pause_needs_data_recovery(cfg, ledger) -> tuple[str, bool, str]:
    ks = _ks(cfg, ledger)
    ks.trip_pause("corrupt feed", source="data")
    ks.on_ws_recovered()            # a healthy WS must NOT clear a DATA pause
    ok = not ks.trading_allowed
    ks.resume_from_pause("data")
    return ("data pause survives WS recovery", ok and ks.trading_allowed,
            "WS recovery did not clear data pause")


def scenario_corrupt_market_data(cfg, ledger) -> tuple[str, bool, str]:
    from .models import Market
    guard = MarketDataGuard()
    bad = [Market(id=f"b{i}", question="q", clob_token_ids=["y", "n"],
                  outcomes=["Yes", "No"], outcome_prices=[1.5, -0.5])  # out of [0,1]
           for i in range(3)]
    ok = guard.severe(bad)
    return ("corrupt prices -> data guard flags", ok, f"problems={guard.problems(bad)}")


def scenario_reconcile_ghost_order(cfg, ledger) -> tuple[str, bool, str]:
    ks = _ks(cfg, ledger)
    ks.reconcile(local_order_ids=set(), exchange_order_ids={"ghost-1"})
    ok = ks.halted
    return ("unknown exchange order -> HALT", ok, f"halted={ks.halted}")


def scenario_daily_loss(cfg, ledger) -> tuple[str, bool, str]:
    ks = _ks(cfg, ledger)
    start = cfg.portfolio.bankroll_usd
    ledger.snapshot_bank(start, 0.0)        # establish today's day-start equity
    ks.check_daily_loss(start - cfg.risk.max_daily_loss_usd - 10)
    return ("daily-loss breach -> HALT", ks.halted, f"halted={ks.halted}")


def scenario_drawdown(cfg, ledger) -> tuple[str, bool, str]:
    ks = _ks(cfg, ledger)
    hwm = cfg.portfolio.bankroll_usd
    ks.check_drawdown(hwm * (1 - cfg.risk.max_drawdown_pct - 0.05), hwm)
    return ("drawdown breach -> HALT", ks.halted, f"halted={ks.halted}")


def scenario_breaker_losing_streak(cfg, ledger) -> tuple[str, bool, str]:
    br = StrategyCircuitBreaker(losing_streak=3)
    for pnl in (0.0, -5.0, -10.0, -15.0):     # four checks, three drops
        br.update({"fade": pnl})
    ok = not br.allows("fade")
    return ("losing streak -> strategy disabled", ok, f"disabled={br.disabled}")


SCENARIOS = [
    scenario_ws_outage,
    scenario_ws_recovery,
    scenario_data_pause_needs_data_recovery,
    scenario_corrupt_market_data,
    scenario_reconcile_ghost_order,
    scenario_daily_loss,
    scenario_drawdown,
    scenario_breaker_losing_streak,
]


def run_all(cfg: BotConfig) -> list[tuple[str, bool, str]]:
    ledger = Ledger(":memory:")
    try:
        return [fn(cfg, ledger) for fn in SCENARIOS]
    finally:
        ledger.close()


def run_chaos(cfg: BotConfig) -> bool:  # pragma: no cover — I/O glue
    import sys

    from rich.console import Console
    from rich.table import Table

    console = Console()
    results = run_all(cfg)
    t = Table(title="Chaos drills (safety systems under failure)")
    t.add_column("Scenario")
    t.add_column("Result")
    t.add_column("Detail")
    for name, ok, detail in results:
        t.add_row(name, "[green]PASS[/green]" if ok else "[red]FAIL[/red]", detail)
    console.print(t)
    passed = all(ok for _, ok, _ in results)
    console.print(("[green]All drills passed — safety systems armed.[/green]"
                   if passed else "[red]A drill FAILED — do not go live.[/red]"))
    if not passed:
        sys.exit(1)
    return passed
