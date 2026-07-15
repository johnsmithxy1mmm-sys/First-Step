"""Self-calibration: tail-bias learner, markout feedback, Platt recalibration."""

import random

from polymarket_bot.calibration import (MarkoutFeedback, PlattCalibrator,
                                         TailBiasCalibrator)


# --- tail-bias learner ---

def test_tail_bias_prior_without_data():
    assert TailBiasCalibrator(prior=0.35).bias("other", 0.05) == 0.35


def test_tail_bias_learns_overpricing():
    cal = TailBiasCalibrator(prior=0.35)
    # 100 faded tails priced at YES=0.05, but the tail happened only twice.
    resolved = [{"category": "other", "p_mkt": 0.95, "won": i >= 2} for i in range(100)]
    cal.fit(resolved)
    assert cal.bias("other", 0.05) > 0.45   # learned overpricing above the prior


def test_tail_bias_shrinks_to_prior_on_small_sample():
    cal = TailBiasCalibrator(prior=0.35, shrink_k=20)
    resolved = [{"category": "other", "p_mkt": 0.95, "won": True} for _ in range(3)]
    cal.fit(resolved)
    assert 0.35 <= cal.bias("other", 0.05) < 0.42   # 3 samples barely move it


def test_tail_bias_summary():
    cal = TailBiasCalibrator().fit(
        [{"category": "sports", "p_mkt": 0.97, "won": True} for _ in range(10)])
    rows = cal.summary()
    assert rows and rows[0]["category"] == "sports" and rows[0]["n"] == 10


# --- markout feedback ---

def test_markout_feedback_widens_on_adverse_selection():
    fb = MarkoutFeedback(scale=0.01, max_mult=2.0, min_n=5)
    fb.fit({"m1": (-0.005, 10), "m2": (0.003, 10), "m3": (-0.01, 2)})
    assert fb.multiplier("m1") == 1.5    # -0.005/0.01 = 0.5 widen
    assert fb.multiplier("m2") == 1.0    # positive markout -> baseline
    assert fb.multiplier("m3") == 1.0    # too few fills to trust
    assert fb.multiplier("unknown") == 1.0


def test_markout_feedback_caps_widening():
    fb = MarkoutFeedback(scale=0.01, max_mult=2.0, min_n=5)
    fb.fit({"m1": (-0.5, 10)})           # huge adverse markout
    assert fb.multiplier("m1") == 2.0    # capped at max_mult


# --- Platt recalibration ---

def test_platt_identity_before_fit():
    assert PlattCalibrator().calibrate(0.3) == 0.3


def test_platt_corrects_overconfidence():
    random.seed(0)
    pairs = []
    for _ in range(300):
        pairs.append((0.9, 1.0 if random.random() < 0.6 else 0.0))
        pairs.append((0.1, 1.0 if random.random() < 0.4 else 0.0))
    cal = PlattCalibrator(min_samples=30).fit(pairs)
    assert cal.calibrate(0.9) < 0.9      # pulled toward the true 0.6
    assert cal.calibrate(0.1) > 0.1      # pulled toward the true 0.4
