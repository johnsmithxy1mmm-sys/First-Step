"""Config layering: example = base, config.yaml = overlay; sync-config append."""

from pathlib import Path

import yaml

from polymarket_bot import config as config_mod
from polymarket_bot.config import (BotConfig, _deep_merge, missing_sections,
                                   sync_config)


# --- pure merge semantics ---

def test_overlay_scalar_wins_base_fills_the_rest():
    base = {"fade": {"enabled": True, "bias_discount": 0.4}, "ticks": {"enabled": True}}
    over = {"fade": {"bias_discount": 0.25}}
    merged = _deep_merge(base, over)
    assert merged["fade"]["bias_discount"] == 0.25       # yours wins
    assert merged["fade"]["enabled"] is True             # base fills the gap
    assert merged["ticks"] == {"enabled": True}          # whole missing block adopted


def test_lists_replace_not_concatenate():
    base = {"niche": {"watchlists": [{"name": "a"}, {"name": "b"}]}}
    over = {"niche": {"watchlists": [{"name": "mine"}]}}
    assert _deep_merge(base, over)["niche"]["watchlists"] == [{"name": "mine"}]


def test_empty_overlay_returns_base():
    base = {"x": 1}
    assert _deep_merge(base, {}) == base


# --- load(): the example is live underneath the user's file ---

def make_files(tmp_path, monkeypatch, example: str, user: str | None):
    ex = tmp_path / "config.example.yaml"
    ex.write_text(example, encoding="utf-8")
    monkeypatch.setattr(config_mod, "EXAMPLE_CONFIG_PATH", ex)
    us = tmp_path / "config.yaml"
    if user is not None:
        us.write_text(user, encoding="utf-8")
    monkeypatch.setattr(config_mod, "DEFAULT_CONFIG_PATH", us)
    return us


EXAMPLE = """\
# The fade block.
fade:
  enabled: true
  bias_discount: 0.40

# Learning-loop recording.
ticks:
  enabled: true
  retention_days: 30
"""


def test_load_merges_example_under_user(tmp_path, monkeypatch):
    make_files(tmp_path, monkeypatch, EXAMPLE, "fade:\n  bias_discount: 0.10\n")
    cfg = BotConfig.load()
    assert cfg.fade.bias_discount == 0.10        # user's value wins
    assert cfg.fade.enabled is True              # example fills the sibling key
    assert cfg.ticks.retention_days == 30.0      # example block live with no copying


def test_load_without_user_file_is_pure_example(tmp_path, monkeypatch):
    make_files(tmp_path, monkeypatch, EXAMPLE, None)
    cfg = BotConfig.load()
    assert cfg.fade.bias_discount == 0.40


def test_real_example_is_the_live_base_layer():
    """With no personal config.yaml in the repo, load() must serve the real
    example's tuned values — not the conservative code defaults."""
    cfg = BotConfig.load()
    assert cfg.fade.max_days_to_resolution == 90.0        # example, not default 14
    assert cfg.ticks.enabled is True


# --- sync-config: materialize missing sections, append-only ---

def test_missing_sections_lists_only_absent(tmp_path, monkeypatch):
    make_files(tmp_path, monkeypatch, EXAMPLE, "fade:\n  bias_discount: 0.10\n")
    assert missing_sections() == ["ticks"]


def test_sync_appends_block_with_comments(tmp_path, monkeypatch):
    user = make_files(tmp_path, monkeypatch, EXAMPLE, "fade:\n  bias_discount: 0.10\n")
    added = sync_config()
    assert added == ["ticks"]
    text = user.read_text(encoding="utf-8")
    assert "# Learning-loop recording." in text           # comments travel along
    assert "bias_discount: 0.10" in text                  # user's value untouched
    data = yaml.safe_load(text)
    assert data["ticks"]["retention_days"] == 30
    assert data["fade"] == {"bias_discount": 0.10}        # fade NOT overwritten


def test_sync_is_idempotent(tmp_path, monkeypatch):
    user = make_files(tmp_path, monkeypatch, EXAMPLE, "fade:\n  bias_discount: 0.10\n")
    sync_config()
    before = user.read_text(encoding="utf-8")
    assert sync_config() == []                            # nothing left to add
    assert user.read_text(encoding="utf-8") == before     # and no growth


def test_sync_creates_config_from_example_when_absent(tmp_path, monkeypatch):
    user = make_files(tmp_path, monkeypatch, EXAMPLE, None)
    added = sync_config()
    assert added == ["fade", "ticks"]
    assert user.read_text(encoding="utf-8") == EXAMPLE    # full template copy


def test_synced_file_still_loads(tmp_path, monkeypatch):
    make_files(tmp_path, monkeypatch, EXAMPLE, "fade:\n  bias_discount: 0.10\n")
    sync_config()
    cfg = BotConfig.load()
    assert cfg.fade.bias_discount == 0.10
    assert cfg.ticks.enabled is True


def test_real_example_sections_all_parse():
    """Every top-level block sliced out of the real example must reassemble
    into the same keys yaml sees — guards the comment-attached slicer."""
    blocks = config_mod.example_section_blocks()
    keys = [k for k, _ in blocks]
    parsed = yaml.safe_load(Path(config_mod.EXAMPLE_CONFIG_PATH).read_text(encoding="utf-8"))
    assert keys == list(parsed.keys())
    joined = "".join(text for _, text in blocks)
    assert yaml.safe_load(joined) == parsed               # slices lose nothing
