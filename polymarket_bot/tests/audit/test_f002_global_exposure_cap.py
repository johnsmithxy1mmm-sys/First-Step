"""F-002: `risk.max_global_exposure_usd` is enforced nowhere (fails on HEAD).

Two independent global caps exist and neither covers the MM:

  * `risk.max_global_exposure_usd` ($300 default) — the number a user reads as
    "the most this bot can ever have at risk". `KillSwitch.check_global_exposure`
    implements it and has ZERO callers.
  * `portfolio.max_total_exposure_pct * bankroll` (0.30 * $5000 = $1500) — a
    *different* limit, applied only on the longshot/fade sizing path.

MM/sprint inventory is bounded per market only, so 5 markets x $50 x 2 sides =
$500 of MM inventory sits above the $300 the user configured, and nothing in
the system objects. Worse, the two caps disagree by 5x, so tightening the one
in `risk:` has no effect on the strategy that carries most of the capital.
"""

import pytest

from polymarket_bot.models import simple_estimate
from polymarket_bot.risk import KillSwitch


def test_check_global_exposure_is_dead_code(cfg, ledger):
    """It computes the right answer — nobody ever asks it."""
    ks = KillSwitch(cfg, ledger, "paper", cancel_all=lambda: None,
                    alert=lambda m: True)
    assert ks.check_global_exposure(cfg.risk.max_global_exposure_usd + 1) is True
    # Proof of deadness: no production module references it.
    import pathlib
    hits = []
    root = pathlib.Path(__file__).resolve().parents[2]
    for path in root.rglob("*.py"):
        if "tests" in path.parts or path.name == "risk.py":
            continue
        if "check_global_exposure" in path.read_text(encoding="utf-8"):
            hits.append(path.name)
    assert hits, ("check_global_exposure has no callers: risk.max_global_exposure_usd "
                  "is documentation, not a limit")


def test_mm_inventory_may_exceed_the_configured_global_cap(cfg, ledger):
    """Fill every MM market to its per-market ceiling and compare with the cap."""
    from polymarket_bot.tests.conftest import make_market

    per_market = cfg.risk.max_position_per_market_usd
    n = cfg.market_maker.max_markets
    for i in range(n):
        m = make_market(id=f"mm{i}", clob_token_ids=[f"mm{i}-y", f"mm{i}-n"],
                        outcome_prices=[0.5, 0.5])
        # Two-sided MM is allowed up to 2x the per-market cap (marketmaker.py:205).
        for outcome in (0, 1):
            ledger.record_trade(
                mode="paper", estimate=simple_estimate(m, outcome, 0.5),
                category="mm", side="BUY", price=0.5, size=per_market / 0.5,
                order_id=None, status="filled", strategy="mm")

    exposure = ledger.total_exposure("paper")
    cap = cfg.risk.max_global_exposure_usd
    assert exposure <= cap, (
        f"MM inventory ${exposure:,.0f} exceeds risk.max_global_exposure_usd "
        f"${cap:,.0f} with no guard in the system")
