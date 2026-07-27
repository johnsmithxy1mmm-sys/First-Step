"""F-016: the LLM cache is a trust boundary that bypassed the response schema.

Live responses go through `messages.parse` with a pydantic schema, so p_est and
confidence are guaranteed probabilities. The on-disk cache did NOT: a plain JSON
file, read with `json.loads` (which accepts bare NaN/Infinity), stale,
hand-editable, or truncatable by a crash mid-write.

The damage path is position size, not just a wrong estimate:

    cache p_est=5.0 -> Signal(p_est=5.0) -> combine -> Estimate.p_est
                    -> portfolio.size_usd -> kelly_fraction(5.0, 0.05) = 5.21
                    -> 521% of the Kelly base

Two defences, deliberately layered: `Signal` bounds p_est/confidence at the TYPE
level (protecting every signal source, not just this one), and `_cached`
re-validates entries so a poisoned file is dropped with a warning instead of
raising deep inside a strategy.
"""

import json
import math

import pydantic
import pytest

from polymarket_bot.models import Signal
from polymarket_bot.portfolio import kelly_fraction


# --- type-level bound: a probability is in [0, 1] or it is not representable ---

@pytest.mark.parametrize("bad", [5.0, -3.0, 1.5, float("inf"), float("nan")])
def test_signal_rejects_impossible_probabilities(bad):
    with pytest.raises(pydantic.ValidationError):
        Signal(name="llm", p_est=bad, confidence=0.5)


@pytest.mark.parametrize("bad", [2.0, -0.1, float("inf"), float("nan")])
def test_signal_rejects_impossible_confidence(bad):
    """`confidence` is an ensemble WEIGHT documented as 0..1. Tests used to pass
    1e9 to overpower the market anchor — an out-of-contract value that silently
    worked because nothing enforced the range."""
    with pytest.raises(pydantic.ValidationError):
        Signal(name="llm", p_est=0.3, confidence=bad)


def test_signal_still_accepts_abstention_and_sane_values():
    assert Signal(name="s", p_est=None).p_est is None      # abstained
    assert Signal(name="s", p_est=0.0, confidence=0.0).p_est == 0.0
    assert Signal(name="s", p_est=1.0, confidence=1.0).p_est == 1.0


# --- the money consequence the bound prevents ---

def test_kelly_would_oversize_on_an_unbounded_estimate():
    """Documents WHY the bound matters: Kelly is linear in p_est above the
    price, so an unbounded estimate scales the position without limit."""
    assert kelly_fraction(0.10, 0.05) < 0.10          # sane
    assert kelly_fraction(5.0, 0.05) > 5.0            # 500%+ of the base
    assert not math.isfinite(kelly_fraction(float("inf"), 0.05))


# --- cache re-validation ---

def _llm(cfg, tmp_path, cache: dict):
    from polymarket_bot.estimator.llm import LLMSignal
    path = tmp_path / "llm_cache.json"
    path.write_text(json.dumps(cache), encoding="utf-8")
    cfg.runtime.llm_cache_path = str(path)
    return LLMSignal(cfg)


@pytest.mark.parametrize("result", [
    {"p_est": 5.0, "confidence": 0.6},           # out of range
    {"p_est": -1.0, "confidence": 0.6},
    {"p_est": 0.4, "confidence": 9.0},
    {"p_est": "not a number", "confidence": 0.6},
    {"confidence": 0.6},                          # missing p_est
    {"p_est": 0.4},                               # missing confidence
    "not even a dict",
])
def test_poisoned_cache_entries_are_dropped(cfg, tmp_path, result):
    import time
    llm = _llm(cfg, tmp_path, {"m1:0": {"ts": time.time(), "result": result}})
    assert llm._cached("m1:0") is None


def test_non_finite_cache_entry_is_dropped(cfg, tmp_path):
    """json.loads accepts bare NaN/Infinity — the same hole as the WS feed."""
    import time
    path = tmp_path / "llm_cache.json"
    path.write_text('{"m1:0": {"ts": %f, "result": {"p_est": NaN, '
                    '"confidence": 0.6}}}' % time.time(), encoding="utf-8")
    cfg.runtime.llm_cache_path = str(path)
    from polymarket_bot.estimator.llm import LLMSignal
    assert LLMSignal(cfg)._cached("m1:0") is None


def test_sane_cache_entry_is_served(cfg, tmp_path):
    import time
    llm = _llm(cfg, tmp_path,
               {"m1:0": {"ts": time.time(),
                         "result": {"p_est": 0.42, "confidence": 0.6}}})
    assert llm._cached("m1:0") == {"p_est": 0.42, "confidence": 0.6}


def test_expired_cache_entry_is_not_served(cfg, tmp_path):
    llm = _llm(cfg, tmp_path,
               {"m1:0": {"ts": 0.0, "result": {"p_est": 0.42, "confidence": 0.6}}})
    assert llm._cached("m1:0") is None
