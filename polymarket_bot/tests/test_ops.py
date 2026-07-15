"""Ops: Prometheus metrics formatting and in-place config hot-reload."""

from polymarket_bot.config import BotConfig
from polymarket_bot.ops import format_metrics, reload_config_inplace


def test_format_metrics_prometheus_shape():
    out = format_metrics({
        "equity": 5012.5, "drawdown": 0.03, "positions": 8, "exposure": 396.0,
        "worst_case": 120.0, "errors": 0, "halted": False,
        "pnl_by_strategy": {"fade": 12.5, "mm": -3.0},
    })
    assert "polybot_equity_usd 5012.5" in out
    assert "polybot_killswitch_halted 0" in out
    assert 'polybot_pnl_usd{strategy="fade"} 12.5' in out
    # Every metric has HELP/TYPE headers.
    assert out.count("# TYPE") >= 6


def test_health_metric_reflects_halt():
    assert "polybot_killswitch_halted 1" in format_metrics({"halted": True})


def test_reload_config_inplace_applies_changes(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("portfolio:\n  kelly_fraction: 0.15\nfade:\n  bias_discount: 0.30\n")
    cfg = BotConfig.load(path)
    assert cfg.portfolio.kelly_fraction == 0.15
    fade_ref = cfg.fade                       # a live holder of the sub-config

    path.write_text("portfolio:\n  kelly_fraction: 0.30\nfade:\n  bias_discount: 0.45\n")
    changed = reload_config_inplace(cfg, path)

    assert "portfolio.kelly_fraction" in changed
    assert cfg.portfolio.kelly_fraction == 0.30
    # The existing sub-config object was mutated in place (holders see it).
    assert fade_ref.bias_discount == 0.45


def test_reload_config_reports_nothing_when_unchanged(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("portfolio:\n  kelly_fraction: 0.20\n")
    cfg = BotConfig.load(path)
    assert reload_config_inplace(cfg, path) == []
