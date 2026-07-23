"""Risk profile switch: user > profile > example > code defaults."""

from polymarket_bot import config as config_mod
from polymarket_bot.config import PROFILES, BotConfig


def make_files(tmp_path, monkeypatch, example: str, user: str | None):
    ex = tmp_path / "config.example.yaml"
    ex.write_text(example, encoding="utf-8")
    monkeypatch.setattr(config_mod, "EXAMPLE_CONFIG_PATH", ex)
    us = tmp_path / "config.yaml"
    if user is not None:
        us.write_text(user, encoding="utf-8")
    monkeypatch.setattr(config_mod, "DEFAULT_CONFIG_PATH", us)


# The example ships the aggressive numbers inline (kelly 0.30).
EXAMPLE = """\
profile: ""
portfolio:
  kelly_fraction: 0.30
  max_market_pct: 0.025
risk:
  max_daily_loss_usd: 60
"""


def test_no_profile_keeps_example_values(tmp_path, monkeypatch):
    make_files(tmp_path, monkeypatch, EXAMPLE, None)
    cfg = BotConfig.load()
    assert cfg.portfolio.kelly_fraction == 0.30      # example stands


def test_conservative_profile_dials_down(tmp_path, monkeypatch):
    make_files(tmp_path, monkeypatch, EXAMPLE, "profile: conservative\n")
    cfg = BotConfig.load()
    assert cfg.portfolio.kelly_fraction == PROFILES["conservative"]["portfolio"]["kelly_fraction"]
    assert cfg.risk.max_daily_loss_usd == PROFILES["conservative"]["risk"]["max_daily_loss_usd"]


def test_user_value_wins_over_profile(tmp_path, monkeypatch):
    make_files(tmp_path, monkeypatch, EXAMPLE,
               "profile: conservative\nportfolio:\n  kelly_fraction: 0.22\n")
    cfg = BotConfig.load()
    assert cfg.portfolio.kelly_fraction == 0.22      # explicit override beats the preset
    # ...but knobs I did NOT set still follow the conservative preset.
    assert cfg.portfolio.max_market_pct == PROFILES["conservative"]["portfolio"]["max_market_pct"]


def test_aggressive_profile_reproduces_example(tmp_path, monkeypatch):
    make_files(tmp_path, monkeypatch, EXAMPLE, "profile: aggressive\n")
    cfg = BotConfig.load()
    assert cfg.portfolio.kelly_fraction == 0.30


def test_unknown_profile_ignored(tmp_path, monkeypatch):
    make_files(tmp_path, monkeypatch, EXAMPLE, "profile: yolo\n")
    cfg = BotConfig.load()
    assert cfg.portfolio.kelly_fraction == 0.30      # falls back to example, no crash


def test_real_example_profile_field_present():
    cfg = BotConfig.load()
    assert hasattr(cfg, "profile")
    # The shipped example keeps profile empty (its inline values are aggressive).
    assert cfg.portfolio.kelly_fraction == 0.30


def test_profiles_only_touch_known_config_keys():
    """Guard against a typo'd preset key silently doing nothing."""
    valid = set(BotConfig.model_fields)
    for bundle in PROFILES.values():
        assert set(bundle) <= valid
        for section, knobs in bundle.items():
            sub = BotConfig.model_fields[section].default_factory()
            assert set(knobs) <= set(type(sub).model_fields)
