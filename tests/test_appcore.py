"""Тесты чистых функций приложения (без запуска GUI)."""

import yaml

from appcore import config_to_flat, config_to_yaml, slug


def test_slug_from_cyrillic_name():
    assert slug("МФК «Парк»") == "мфк-парк"
    assert slug("") == "memo"


def test_config_to_flat_growth_example():
    raw = yaml.safe_load(open("objects/park.yaml", encoding="utf-8"))
    flat = config_to_flat(raw)
    assert flat["exit_method"] == "growth"
    assert flat["exit_value"] == 8
    assert flat["recommendation"].startswith("HOLD")
    assert flat["schedule"] == [(0, 30), (12, 35), (24, 35)]
    # Диапазон выхода берётся из соответствующего ключа.
    assert flat["exitr_mode"] == 8


def test_config_to_flat_cap_rate_example():
    raw = yaml.safe_load(open("objects/polyana.yaml", encoding="utf-8"))
    flat = config_to_flat(raw)
    assert flat["exit_method"] == "cap"
    assert flat["exit_value"] == 7
    assert flat["recommendation"].startswith("ACQUIRE")


def test_config_to_yaml_roundtrips():
    data = {"object": {"name": "Тест"}, "purchase": {"price_rub": 100}}
    text = config_to_yaml(data)
    assert yaml.safe_load(text) == data
    # Кириллица не должна экранироваться в \uXXXX.
    assert "Тест" in text
