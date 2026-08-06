"""The interpolated quantile map sits on the path that produces every risk
number, so its error budget is asserted rather than assumed."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import special, stats

from risk_engine.sim.quantile_map import QuantileMap


def exact(x, source_df, target_df):
    """Reference composition, evaluated in survival space on the positive side.

    `cdf` saturates at 1.0 well before the tail this map has to cover, so a
    naive `ppf(cdf(x))` reference returns inf and would be testing scipy's
    floating point rather than the map. Both distributions are symmetric, so
    the negative side comes from oddness.
    """
    src = stats.norm if source_df is None else stats.t(df=source_df)
    tgt = stats.norm if target_df is None else stats.t(df=target_df)
    scale = 1.0 if target_df is None else np.sqrt(target_df / (target_df - 2.0))
    ax = np.abs(x)
    return np.copysign(tgt.isf(src.sf(ax)) / scale, x)


@pytest.mark.parametrize(
    "source_df,target_df",
    [
        (4.0, 2.5), (4.0, 12.0), (None, 3.0), (6.0, None), (4.0, 30.0), (2.5, 20.0),
        # Target df at the §2.2 clamp FLOOR. These are shipped configurations,
        # not corner cases: `DF_MIN` is 2.1, a Gaussian copula (source None) is
        # the §3.2 baseline, and 30.0 is the copula grid's ceiling. The grid
        # above stopped at 2.5, so the map ran 28% over its own stated error
        # budget in production without any test seeing it.
        (None, 2.1), (30.0, 2.1), (6.5, 2.1), (None, 2.5),
    ],
)
def test_body_accuracy(source_df, target_df):
    x = np.concatenate([np.linspace(-6, 6, 4001), np.array([0.0, 1e-9, -1e-9])])
    got = QuantileMap.build(source_df, target_df).apply(x)
    want = exact(x, source_df, target_df)
    scale = np.maximum(np.abs(want), 1e-3)
    assert np.max(np.abs(got - want) / scale) < 1e-5


@pytest.mark.parametrize(
    "source_df,target_df,max_x",
    [
        (4.0, 2.5, 10**2.5),
        (4.0, 12.0, 10**2.5),
        # A Gaussian source cannot be checked this far out: the reference
        # needs `t.isf` at the source's survival probability, and scipy's
        # `t.isf` is only trustworthy down to about 1e-100 (see
        # test_scipy_isf_breaks_down_in_the_far_tail). norm.sf(15) ~ 4e-51 is
        # comfortably inside that, and is still three orders of magnitude
        # past the largest |z| four million draws will ever produce.
        (None, 3.0, 15.0),
    ],
)
def test_tail_accuracy(source_df, target_df, max_x):
    """Out to survival probabilities far past anything 20k paths will draw."""
    grid = np.logspace(0.5, np.log10(max_x), 400)
    x = np.concatenate([-grid, grid])
    got = QuantileMap.build(source_df, target_df).apply(x)
    want = exact(x, source_df, target_df)
    assert np.max(np.abs(got / want - 1.0)) < 1e-4


def analytic_t_tail(survival, df):
    """Exact asymptotic quantile of a standardised t from its survival.

    S(y) -> C * y^-df, with C = Gamma((df+1)/2) df^(df/2 - 1) / (sqrt(pi)
    Gamma(df/2)). Used instead of `t.isf` where scipy's inversion breaks.
    """
    log_c = (
        special.gammaln(0.5 * (df + 1.0))
        + (0.5 * df - 1.0) * np.log(df)
        - 0.5 * np.log(np.pi)
        - special.gammaln(0.5 * df)
    )
    return np.exp((log_c - np.log(survival)) / df) / np.sqrt(df / (df - 2.0))


def test_scipy_isf_breaks_down_in_the_far_tail():
    """Documents why the reference above is bounded, and why the extrapolation
    test below does not use `t.isf`.

    At p = 1e-197 scipy returns exactly half the true quantile. The map's own
    continuation is the correct one; this pins the fact so a future reader
    does not "fix" the map to agree with scipy.
    """
    assert stats.t(df=3).isf(1e-50) == pytest.approx(analytic_t_tail(1e-50, 3.0) * np.sqrt(3),
                                                     rel=1e-4)
    assert stats.t(df=3).isf(1e-197) == pytest.approx(
        0.5 * analytic_t_tail(1e-197, 3.0) * np.sqrt(3), rel=1e-4
    )


@pytest.mark.parametrize("source_df", [4.0, None])
def test_extrapolation_beyond_the_table_follows_the_analytic_tail(source_df):
    target_df = 2.5
    m = QuantileMap.build(source_df, target_df)
    x = np.array([1e3, 1e6, 1e9]) if source_df else np.array([20.0, 25.0, 30.0])
    src = stats.norm if source_df is None else stats.t(df=source_df)
    want = analytic_t_tail(src.sf(x), target_df)
    assert np.max(np.abs(m.apply(x) / want - 1.0)) < 1e-3


def test_identity_case_is_pure_standardisation():
    m = QuantileMap.build(5.0, 5.0)
    x = np.linspace(-8, 8, 101)
    assert np.allclose(m.apply(x), x / np.sqrt(5 / 3))


def test_map_is_odd_and_strictly_increasing():
    m = QuantileMap.build(4.0, 2.5)
    x = np.linspace(-30, 30, 2001)
    y = m.apply(x)
    assert np.allclose(m.apply(-x), -y)
    assert np.all(np.diff(y) > 0)


def test_output_has_unit_variance():
    rng = np.random.default_rng(7)
    raw = stats.t(df=4.0).rvs(400_000, random_state=rng)
    y = QuantileMap.build(4.0, 2.5).apply(raw)
    # df=2.5 has finite but very heavy variance; check the interquartile
    # range against theory instead, which is what unit-variance scaling means
    # in practice for a distribution this heavy.
    q = np.quantile(y, [0.25, 0.75])
    want = stats.t(df=2.5).ppf([0.25, 0.75]) / np.sqrt(2.5 / 0.5)
    assert np.allclose(q, want, rtol=0.02)


def test_infinite_variance_target_is_refused():
    with pytest.raises(ValueError, match="infinite variance"):
        QuantileMap.build(4.0, 1.8)
