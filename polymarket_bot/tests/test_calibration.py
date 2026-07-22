"""Self-calibration: tail-bias learner, markout feedback, Platt recalibration."""

import random

from polymarket_bot.calibration import (CorrelationLearner, FillCalibrator,
                                         MarkoutFeedback, PlattCalibrator,
                                         TailBiasCalibrator)


# --- CorrelationLearner ---

def _series(pairs, start=0.0):
    return [(start + i, v) for i, v in enumerate(pairs)]


def test_correlation_learns_positive_pair():
    n = 60
    a = _series([i * 0.1 for i in range(n)])
    b = _series([i * 0.1 for i in range(n)])          # identical -> rho = +1, clamped
    learned = CorrelationLearner(min_samples=50).fit({"x": a, "y": b}).learned
    assert learned[frozenset({"x", "y"})] == 0.95     # clamp keeps VaR non-degenerate


def test_correlation_learns_negative_pair():
    n = 60
    a = _series([i * 0.1 for i in range(n)])
    b = _series([-i * 0.1 for i in range(n)])
    rho = CorrelationLearner(min_samples=50).fit({"x": a, "y": b}).learned
    assert rho[frozenset({"x", "y"})] == -0.95


def test_correlation_needs_min_samples():
    a = _series([1.0, 2.0, 3.0])
    b = _series([1.0, 2.0, 3.0])
    assert CorrelationLearner(min_samples=50).fit({"x": a, "y": b}).learned == {}


def test_correlation_only_aligns_shared_timestamps():
    a = [(float(i), i * 0.1) for i in range(60)]
    b = [(float(i) + 1000, i * 0.1) for i in range(60)]   # no shared ts
    assert CorrelationLearner(min_samples=50).fit({"x": a, "y": b}).learned == {}


def test_correlation_flat_series_is_skipped():
    a = _series([1.0] * 60)                             # zero variance
    b = _series([i * 0.1 for i in range(60)])
    assert CorrelationLearner(min_samples=50).fit({"x": a, "y": b}).learned == {}


def test_learned_correlation_overrides_prior():
    from polymarket_bot import portfolio
    try:
        portfolio.set_learned_correlation({frozenset({"crypto", "economy"}): 0.8})
        assert portfolio.correlation("crypto", "economy") == 0.8
        assert portfolio.correlation("geopolitics", "economy") == 0.5   # prior kept
    finally:
        portfolio.set_learned_correlation({})           # reset global for other tests


# --- FillCalibrator ---

def test_fill_calibrator_maps_predicted_to_realized():
    # Predicted ~0.8 but only 40% actually filled -> calibrate down.
    outcomes = [(0.82, i < 4) for i in range(10)] * 3   # bin [0.8,0.9): 40% filled
    fc = FillCalibrator(min_per_bin=20).fit(outcomes)
    assert fc.calibrate(0.82) == 0.4


def test_fill_calibrator_falls_back_below_min_samples():
    fc = FillCalibrator(min_per_bin=20).fit([(0.82, True), (0.82, False)])
    assert fc.calibrate(0.82) == 0.82                   # too few -> raw prediction


def test_fill_calibrator_identity_before_fit():
    assert FillCalibrator().calibrate(0.55) == 0.55


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
