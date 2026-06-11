"""Тесты загрузки и валидации конфига объекта."""

import textwrap

import pytest

from config.loader import ConfigError, load_config, to_dcf_inputs


def _write(tmp_path, text: str):
    p = tmp_path / "obj.yaml"
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


_VALID = """
    object:
      name: "Тест"
      developer: "Застройщик"
      zone: "Сочи"
      type: "апарты"
      area_m2: 40
    purchase:
      price_rub: 10000000
      installment:
        down_payment_pct: 30
        schedule:
          - {month: 0, pct: 30}
          - {month: 12, pct: 70}
    operations:
      rental_rate_rub_month: 80000
      occupancy_pct: 75
      opex_pct_of_revenue: 25
      rental_growth_pct: 5
    exit:
      hold_years: 5
      exit_cap_rate_pct: 9
      selling_cost_pct: 3
    assumptions:
      discount_rate_pct: 15
      target_irr_pct: 12
    advisor:
      recommendation: "HOLD"
      thesis: "тезис"
"""


def test_valid_config_loads(tmp_path):
    cfg = load_config(_write(tmp_path, _VALID))
    assert cfg.object.name == "Тест"
    inp = to_dcf_inputs(cfg)
    assert inp.price_rub == 10_000_000
    assert inp.hold_years == 5


def test_missing_file_message():
    with pytest.raises(ConfigError) as exc:
        load_config("/nonexistent/obj.yaml")
    assert "не найден" in str(exc.value)


def test_schedule_sum_not_100(tmp_path):
    bad = _VALID.replace("pct: 70}", "pct: 50}")
    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, bad))
    assert "100" in str(exc.value)


def test_negative_area(tmp_path):
    bad = _VALID.replace("area_m2: 40", "area_m2: -5")
    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, bad))
    assert "площадь" in str(exc.value).lower()


def test_unknown_recommendation(tmp_path):
    bad = _VALID.replace('recommendation: "HOLD"', 'recommendation: "MAYBE"')
    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, bad))
    assert "ACQUIRE" in str(exc.value)


def test_no_exit_method(tmp_path):
    bad = _VALID.replace("      exit_cap_rate_pct: 9\n", "")
    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, bad))
    assert "выход" in str(exc.value).lower()


def test_down_payment_mismatch(tmp_path):
    bad = _VALID.replace("down_payment_pct: 30", "down_payment_pct: 40")
    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, bad))
    assert "взнос" in str(exc.value).lower()
