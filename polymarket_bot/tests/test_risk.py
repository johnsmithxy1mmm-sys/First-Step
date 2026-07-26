

# --- F-003: a poisoned ledger row must not disable the safety layer ---

def test_ledger_rejects_rows_that_cannot_be_a_real_fill(cfg, ledger):
    """The write boundary refuses what would propagate as NaN downstream."""
    import pytest as _pytest

    from polymarket_bot.ledger import InvalidTrade
    from polymarket_bot.models import simple_estimate

    from .conftest import make_market

    m = make_market(id="v1", clob_token_ids=["v1-y", "v1-n"])
    bad = [
        dict(price=0.5, size=float("inf")),   # non-finite size -> NaN exposure
        dict(price=float("nan"), size=10.0),
        dict(price=0.5, size=-100.0),         # negative size
        dict(price=0.5, size=0.0),
        dict(price=0.0, size=10.0),           # untradable prices
        dict(price=1.0, size=10.0),
        dict(price=1.7, size=10.0),
    ]
    for kwargs in bad:
        with _pytest.raises(InvalidTrade):
            ledger.record_trade(
                mode="paper", estimate=simple_estimate(m, 0, 0.5), category="mm",
                side="BUY", order_id=None, status="filled", strategy="mm",
                **kwargs)
    # A sane row still goes in.
    ledger.record_trade(mode="paper", estimate=simple_estimate(m, 0, 0.5),
                        category="mm", side="BUY", price=0.5, size=10.0,
                        order_id=None, status="filled", strategy="mm")
    assert len(ledger.open_positions("paper")) == 1


def test_killswitch_halts_on_non_finite_equity(cfg, ledger):
    """Fail CLOSED: every `>=` against NaN is False, so comparing is not safe."""
    from polymarket_bot.risk import KillSwitch

    ks = KillSwitch(cfg, ledger, "paper", cancel_all=lambda: None,
                    alert=lambda m: True)
    ks.check_daily_loss(float("nan"))
    assert ks.halted and "not finite" in ks.reason

    ks2 = KillSwitch(cfg, ledger, "paper", cancel_all=lambda: None,
                     alert=lambda m: True)
    ks2.check_drawdown(float("nan"), hwm=1000.0)
    assert ks2.halted

    ks3 = KillSwitch(cfg, ledger, "paper", cancel_all=lambda: None,
                     alert=lambda m: True)
    # Unmeasurable exposure blocks new entries rather than waving them through.
    assert ks3.check_global_exposure(float("nan")) is True


# --- F-009: the safety layer must be asserted by EFFECT, not by state ---

def test_pause_actually_blocks_trading(cfg, ledger):
    """Mutation analysis killer: `trading_allowed` dropping its pause term
    survived every existing test, because the chaos drill checks `paused is True`
    and never that a pause STOPS anything."""
    from polymarket_bot.risk import KillSwitch

    ks = KillSwitch(cfg, ledger, "paper", cancel_all=lambda: None,
                    alert=lambda m: True)
    assert ks.trading_allowed is True
    ks.trip_pause("WS dead", source="ws")
    assert ks.paused is True
    assert ks.trading_allowed is False           # the property that matters
    ks.resume_from_pause("ws")
    assert ks.trading_allowed is True


def test_halt_blocks_trading_and_is_terminal(cfg, ledger):
    from polymarket_bot.risk import KillSwitch

    ks = KillSwitch(cfg, ledger, "paper", cancel_all=lambda: None,
                    alert=lambda m: True)
    ks.trip_halt("test")
    assert ks.trading_allowed is False
    # A pause resume must never clear a HALT.
    ks.resume_from_pause("ws")
    assert ks.halted and ks.trading_allowed is False


def test_independent_pause_sources_each_block(cfg, ledger):
    """`or` -> `and` in the dedupe guard survived; pin both sources explicitly."""
    from polymarket_bot.risk import KillSwitch

    ks = KillSwitch(cfg, ledger, "paper", cancel_all=lambda: None,
                    alert=lambda m: True)
    ks.trip_pause("stream", source="ws")
    ks.trip_pause("corrupt", source="data")
    ks.resume_from_pause("ws")
    assert ks.trading_allowed is False           # data pause still holds
    ks.resume_from_pause("data")
    assert ks.trading_allowed is True


def test_daily_loss_fires_exactly_at_the_threshold(cfg, ledger):
    """`>=` -> `>` survived: the boundary itself was never tested."""
    from polymarket_bot.risk import KillSwitch

    cfg.risk.max_daily_loss_usd = 50.0
    start = 1000.0
    ledger.snapshot_bank(cash=start, exposure=0.0)

    ks = KillSwitch(cfg, ledger, "paper", cancel_all=lambda: None,
                    alert=lambda m: True)
    ks.check_daily_loss(start - 49.99)
    assert not ks.halted                          # just inside the limit
    ks.check_daily_loss(start - 50.0)             # exactly at it
    assert ks.halted


def test_drawdown_fires_exactly_at_the_threshold(cfg, ledger):
    from polymarket_bot.risk import KillSwitch

    cfg.risk.max_drawdown_pct = 0.15
    ks = KillSwitch(cfg, ledger, "paper", cancel_all=lambda: None,
                    alert=lambda m: True)
    ks.check_drawdown(equity_now=851.0, hwm=1000.0)   # 14.9%
    assert not ks.halted
    ks.check_drawdown(equity_now=850.0, hwm=1000.0)   # exactly 15%
    assert ks.halted


def test_pause_while_halted_is_a_no_op(cfg, ledger):
    """`or` -> `and` in the dedupe guard: with `and`, a HALTED bot would still
    record a pause, re-issue a bulk-cancel and emit a misleading
    "Trading paused (auto-resume)" alert while it is in fact halted for good."""
    from polymarket_bot.risk import KillSwitch

    cancels: list[int] = []
    alerts: list[str] = []
    ks = KillSwitch(cfg, ledger, "paper",
                    cancel_all=lambda: cancels.append(1),
                    alert=lambda m: alerts.append(m) or True)
    ks.trip_halt("real reason")
    assert len(cancels) == 1 and len(alerts) == 1

    ks.trip_pause("WS dead", source="ws")
    assert not ks.paused                       # nothing recorded
    assert len(cancels) == 1                   # no second bulk-cancel
    assert not any("auto-resume" in a for a in alerts)
    assert ks.reason == "real reason"           # the halt reason is not overwritten
