"""Phase 5 — §4.4 funding_drag, §7 observability, §3.3 champion/challenger."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from risk_engine.domain.types import Book, MarginMode, Position
from risk_engine.observability.metrics import METRICS
from risk_engine.shadow.champion import compare
from risk_engine.shadow.journal import VARIANT_MODEL, CalibrationJournal
from risk_engine.shadow.metrics import COHORT_ALL
from risk_engine.sim.stats import PredictiveDistribution
from risk_engine.tools.funding_drag import funding_drag


def addr(i: int) -> str:
    """A distinct well-formed account address per index.

    The journal canonicalises what it writes, so a stub like "0x000" is
    refused at the write. Champion/challenger pairs its two versions on
    `(address, observation_day)`, so both `_fill` runs have to derive their
    addresses the same way -- which is exactly the property that broke when
    one source spelled an account differently from another.
    """
    return f"0x{i:040x}"


def long_book(now):
    return Book(
        "0xuser", 200_000.0,
        (
            Position("BTC", 6.0, 100_000.0, MarginMode.CROSS, 20.0),
            Position("ETH", 100.0, 4_000.0, MarginMode.CROSS, 20.0),
        ),
        now,
    )


class TestFundingDrag:
    def test_returns_a_distribution_not_a_point(self, bundle, specs, spot, now):
        """§4.4 is explicit: the AR(1) with its clamp produces a genuinely
        wide spread over a week, and a median alone would let a user plan
        around a number they have even odds of beating."""
        out = funding_drag(long_book(now), spot, bundle, specs,
                           horizon_hours=168, n_paths=4_000, seed=1, now=now)
        assert out.quantile(0.95) > out.quantile(0.5) > out.quantile(0.05)
        assert out.expected.ci_low <= out.expected.point <= out.expected.ci_high
        assert "5th-95th" in out.summary()

    def test_a_week_costs_more_than_a_day(self, bundle, specs, spot, now):
        book = long_book(now)
        day = funding_drag(book, spot, bundle, specs, horizon_hours=24,
                           n_paths=4_000, seed=2, now=now)
        week = funding_drag(book, spot, bundle, specs, horizon_hours=168,
                            n_paths=4_000, seed=2, now=now)
        assert week.quantile(0.5) > day.quantile(0.5)

    def test_shorts_receive_where_longs_pay(self, bundle, specs, spot, now):
        """The sign has to survive to the user. A tool that only ever reports
        a cost would pass a one-sided test.

        Near-symmetric rather than exactly so, and the gap is not error: a
        liquidated position stops accruing funding, and a long and a short
        over the same paths die on *different* ones. Asserting exact symmetry
        would be asserting that liquidation does not truncate the cash flow,
        which is the opposite of what §1.5 models.
        """
        short = Book("0x", 200_000.0,
                     (Position("BTC", -6.0, 100_000.0, MarginMode.CROSS, 20.0),), now)
        long = Book("0x", 200_000.0,
                    (Position("BTC", 6.0, 100_000.0, MarginMode.CROSS, 20.0),), now)
        s = funding_drag(short, spot, bundle, specs, horizon_hours=168,
                         n_paths=4_000, seed=3, now=now)
        long_ = funding_drag(long, spot, bundle, specs, horizon_hours=168,
                             n_paths=4_000, seed=3, now=now)
        assert s.quantile(0.5) < 0 < long_.quantile(0.5)
        assert s.quantile(0.5) == pytest.approx(-long_.quantile(0.5), rel=0.02)

    def test_the_parts_reconcile_against_the_whole(self, bundle, specs, spot, now):
        """Per-position figures are walked on the SAME paths as the total, so
        a user can add them up and get the number they were shown."""
        out = funding_drag(long_book(now), spot, bundle, specs, horizon_hours=168,
                           n_paths=4_000, seed=4, now=now)
        assert set(out.per_position) == {"BTC", "ETH"}
        parts = sum(d.quantile(0.5) for d in out.per_position.values())
        assert parts == pytest.approx(out.quantile(0.5), rel=0.02)

    def test_share_of_equity_is_reported(self, bundle, specs, spot, now):
        out = funding_drag(long_book(now), spot, bundle, specs, horizon_hours=168,
                           n_paths=4_000, seed=5, now=now)
        expected = out.quantile(0.5) / out.start_equity
        assert float(out.share_of_equity.quantile(0.5)) == pytest.approx(expected, rel=0.05)

    def test_the_independence_caveat_travels_with_the_result(
        self, bundle, specs, spot, now
    ):
        """OPEN-QUESTIONS A8 is stated on every result, not in a footnote: a
        user planning a week-long hold is exactly who it misleads."""
        out = funding_drag(long_book(now), spot, bundle, specs, horizon_hours=168,
                           n_paths=2_000, seed=6, now=now)
        joined = " ".join(out.caveats)
        assert "independently of price" in joined
        assert "A8" in joined and "C1" in joined

    def test_a_position_without_funding_history_is_refused(
        self, bundle, specs, spot, now
    ):
        stripped = type(bundle)(
            matrix=bundle.matrix, marginals=bundle.marginals, funding={},
            funding_bounds=bundle.funding_bounds, copula_df=bundle.copula_df,
        )
        with pytest.raises(KeyError, match="no funding model"):
            funding_drag(long_book(now), spot, stripped, specs, n_paths=500, seed=7)


class TestObservability:
    def test_every_metric_ss7_names_is_exposed(self, bundle, specs, spot, now):
        """§7's list, checked against the snapshot rather than assumed."""
        METRICS.reset()
        from risk_engine.tools.pre_trade_delta import ProposedOrder, pre_trade_delta

        pre_trade_delta(long_book(now), ProposedOrder("SOL", 400.0, 20.0), spot,
                        bundle, specs, n_paths=4_000, seed=8, now=now)
        snap = METRICS.snapshot()

        # Per-stage pre-trade latency (p50/p95/p99), §7's first item.
        for stage in ("slice_submatrix", "generate_paths", "liquidation_walk"):
            assert {"p50", "p95", "p99"} <= set(snap["latency"][stage]), stage

        # The counters §7 lists, present as keys even at zero, so a dashboard
        # can render "0" rather than a gap.
        assert "psd_corrections" in snap
        assert "df_clamps" in snap
        assert isinstance(snap["counters"], dict)

    def test_the_psd_and_df_counters_record_magnitudes_not_just_counts(self):
        """§2.1 and §2.2 both ask for the *size* of each correction, because
        a count alone cannot tell a rounding error from a broken matrix."""
        from risk_engine.model.psd import project_to_correlation
        from risk_engine.observability.metrics import Metrics

        m = Metrics()
        bad = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
        project_to_correlation(bad, metrics=m)
        entry = m.snapshot()["psd_corrections"][0]
        assert {"min_eigenvalue_before", "min_eigenvalue_after",
                "frobenius_correction"} <= set(entry)


class TestChampionChallenger:
    """§3.3: a new version migrates only on a statistically significant CRPS
    improvement, never because it looks better."""

    @staticmethod
    def _fill(journal, version, crps_fn, n_days=25, per_day=8, seed=0):
        rng = np.random.default_rng(seed)
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        dist = PredictiveDistribution.from_samples(rng.normal(0, 1000, 20_000))
        for d in range(n_days):
            when = start + timedelta(days=d)
            shock = rng.normal(0, 1.0)  # one market move shared by the day
            for i in range(per_day):
                pid = journal.record_prediction(
                    address=addr(i), variant=VARIANT_MODEL, predicted_at=when,
                    horizon_hours=24, model_version=version, distribution_version=version,
                    seed=1, n_paths=100, converged=True, start_equity=100_000.0,
                    p_liq=0.01, p_liq_ci=(0.0, 0.02), var_95=100.0, cvar_95=200.0,
                    distribution=dist, book_snapshot={"fingerprint": "x"},
                )
                journal.record_outcome(
                    prediction_id=pid, resolved_at=when + timedelta(days=1),
                    actual_equity=100_000.0, actual_equity_change=0.0,
                    external_flow_usd=0.0, book_changed=False, liquidated=False,
                    pit=0.5, pit_u=0.5, crps=crps_fn(shock, rng),
                    var_95_breached=False, observation_day=when.date(),
                    resolution_lag_s=0.0, stale_resolution=False,
                )

    def test_a_clearly_better_challenger_wins(self):
        journal = CalibrationJournal()
        self._fill(journal, "1.0", lambda s, r: 100.0 + 10 * s + r.normal(0, 1), seed=1)
        self._fill(journal, "1.1", lambda s, r: 80.0 + 10 * s + r.normal(0, 1), seed=1)
        verdict = compare(journal, "1.0", "1.1", cohort=COHORT_ALL, seed=3)
        assert verdict.n_paired > 0
        assert verdict.challenger_wins, verdict.verdict
        assert verdict.clustered_ci[1] < 0
        journal.close()

    def test_an_equivalent_challenger_does_not_migrate(self):
        """The case §3.3 cares most about: no significant difference means
        keep the incumbent, because migrating on noise is how a model drifts
        toward whatever the last sample happened to favour."""
        journal = CalibrationJournal()
        self._fill(journal, "1.0", lambda s, r: 100.0 + 10 * s + r.normal(0, 5), seed=2)
        self._fill(journal, "1.1", lambda s, r: 100.0 + 10 * s + r.normal(0, 5), seed=7)
        verdict = compare(journal, "1.0", "1.1", cohort=COHORT_ALL, seed=3)
        assert not verdict.challenger_wins
        assert "not distinguishable" in verdict.verdict or "spans zero" in verdict.verdict
        journal.close()

    def test_a_worse_challenger_is_named_as_worse(self):
        journal = CalibrationJournal()
        self._fill(journal, "1.0", lambda s, r: 80.0 + 10 * s + r.normal(0, 1), seed=4)
        self._fill(journal, "1.1", lambda s, r: 100.0 + 10 * s + r.normal(0, 1), seed=4)
        verdict = compare(journal, "1.0", "1.1", cohort=COHORT_ALL, seed=3)
        assert not verdict.challenger_wins
        assert "significantly WORSE" in verdict.verdict
        journal.close()

    def test_too_few_days_is_inconclusive_not_a_win(self):
        """Days are the independent unit. A challenger with three days of a
        lucky streak must not migrate however many observations it has."""
        journal = CalibrationJournal()
        self._fill(journal, "1.0", lambda s, r: 100.0, n_days=3, per_day=200, seed=5)
        self._fill(journal, "1.1", lambda s, r: 50.0, n_days=3, per_day=200, seed=5)
        verdict = compare(journal, "1.0", "1.1", cohort=COHORT_ALL, seed=3)
        assert verdict.n_paired == 600
        assert not verdict.challenger_wins
        assert "inconclusive" in verdict.verdict
        journal.close()

    def test_only_shared_observations_are_compared(self):
        """A challenger always starts later. Scoring it on days the champion
        never saw would compare two models on different markets."""
        journal = CalibrationJournal()
        self._fill(journal, "1.0", lambda s, r: 100.0, n_days=25, per_day=4, seed=6)
        self._fill(journal, "1.1", lambda s, r: 90.0, n_days=25, per_day=2, seed=6)
        verdict = compare(journal, "1.0", "1.1", cohort=COHORT_ALL, seed=3)
        assert verdict.n_paired == 25 * 2
        journal.close()

    def test_no_overlap_is_reported_rather_than_crashing(self):
        journal = CalibrationJournal()
        self._fill(journal, "1.0", lambda s, r: 100.0, n_days=2, per_day=2, seed=8)
        verdict = compare(journal, "1.0", "9.9", cohort=COHORT_ALL)
        assert verdict.n_paired == 0
        assert not verdict.challenger_wins
        journal.close()
