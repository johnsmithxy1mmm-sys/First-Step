"""§8 — the internal REST service the Node backend consumes.

The wire format is where the guarantees the Python types enforce would
otherwise be quietly lost: JSON has no opinion about whether a number came
with an interval or a timestamp. These tests hold the boundary to the same
contract as the types.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import pytest

from risk_engine.domain.types import Book, MarginMode, Position
from risk_engine.service.app import serve
from risk_engine.service.state import EngineState, NotReady
from risk_engine.sim.engine import MonteCarloEngine

BOOK = {
    "address": "0xdemo",
    "cross_collateral": 100_000.0,
    "positions": [
        {"coin": "BTC", "size": 4.0, "entry_price": 100_000.0,
         "mode": "cross", "leverage": 20},
        {"coin": "ETH", "size": 60.0, "entry_price": 4_000.0,
         "mode": "cross", "leverage": 20},
    ],
}


def as_book(payload: dict) -> Book:
    """`BOOK` as engine types, for tests that walk it themselves.

    Derived from the same dict the request sends rather than restated, so a
    reference walk and the request it is compared against cannot describe two
    different books -- a divergence there would present as a horizon bug,
    which is the one thing the comparison exists to detect.
    """
    return Book(
        address=payload["address"],
        cross_collateral=payload["cross_collateral"],
        positions=tuple(
            Position(
                coin=p["coin"],
                size=p["size"],
                entry_price=p["entry_price"],
                mode=MarginMode(p["mode"]),
                leverage=p["leverage"],
            )
            for p in payload["positions"]
        ),
        captured_at=datetime.now(timezone.utc),
    )


@pytest.fixture(scope="module")
def service():
    state = EngineState.fixture()
    httpd = serve(state, "127.0.0.1", 0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(0.2)
    yield f"http://127.0.0.1:{port}", state
    httpd.shutdown()
    state.stop()


def call(base: str, path: str, payload: dict | None = None, timeout: float = 120.0):
    url = f"{base}{path}"
    if payload is None:
        return json.load(urllib.request.urlopen(url, timeout=timeout))  # noqa: S310
    # S310: the URL is built from a loopback base this test itself started.
    req = urllib.request.Request(  # noqa: S310
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))  # noqa: S310


class TestHealth:
    def test_reports_matrix_age_separately_from_data_freshness(self, service):
        """§6 wants the matrix age as its own signal. It is rebuilt every five
        minutes by design (§2.1), so applying the 60-second book-staleness
        rule to it would mark the product permanently stale
        (OPEN-QUESTIONS D4)."""
        base, _ = service
        h = call(base, "/health")
        assert h["ready"] is True
        assert h["matrix_age_s"] >= 0.0
        assert h["matrix_built_at"]
        assert h["model_version"]

    def test_synthetic_data_is_labelled_as_such(self, service):
        """A fixture bundle must never be mistakable for live risk."""
        base, _ = service
        assert call(base, "/health")["synthetic_data"] is True

    def test_diagnostics_expose_what_ss7_asks_for(self, service):
        base, _ = service
        d = call(base, "/health")["diagnostics"]
        assert {"shrinkage_intensity", "psd_corrected", "imputed_assets",
                "gate_correlation", "min_eigenvalue", "n_eff"} <= set(d)


class TestPortfolioRisk:
    def test_every_number_arrives_with_an_interval_and_a_timestamp(self, service):
        base, _ = service
        r = call(base, "/portfolio_risk", {"book": BOOK, "n_paths": 4_000, "seed": 1})
        for key in ("effective_leverage", "p_liq_24h", "p_liq_7d", "cvar_95_24h_usd"):
            est = r[key]
            assert est["ci_low"] <= est["point"] <= est["ci_high"], key
            assert est["computed_at"], key
            assert est["model_version"], key
        assert r["computed_at"]
        assert r["publishable"] is True

    def test_seven_day_risk_never_below_twenty_four_hour(self, service):
        """Audit A-10, held at the wire boundary too."""
        base, _ = service
        r = call(base, "/portfolio_risk", {"book": BOOK, "n_paths": 8_000, "seed": 2})
        assert r["p_liq_7d"]["point"] >= r["p_liq_24h"]["point"]

    def test_funding_cost_24h_really_is_twenty_four_hours(self, service):
        """The field is named `_24h` and the UI labels it 24h; nothing until now
        made it be 24h.

        `funding_drag` has a guard on *its* 24h default (OPEN-QUESTIONS A8,
        `test_phase5.test_the_default_horizon_is_a_day`), but `funding_drag` is
        on no shipped path -- no route constructs it, and the shadow cron calls
        the engine directly. The funding number a user actually sees comes from
        `portfolio_risk.result_24h.funding_cost`, whose horizon is
        `portfolio_risk.DAY_HOURS`: a second constant, unrelated to
        `funding_drag`'s, that no test read. Setting it to 168 published a full
        week of funding under a field named `_24h` and a panel labelled 24h --
        roughly 7x the cost of the horizon claimed -- and the suite still
        exited 0, because the only other assertions on this response are key
        presence and `p_liq_7d >= p_liq_24h`, which holds when both arms have
        collapsed onto the same horizon.

        The anchor is an independent walk of the same book on the same seed
        with both horizons written as literals, so the reference cannot follow
        the constant it is checking. Equality is exact rather than approximate
        because it is the same computation on the same seed: a mismatch means
        the horizon moved, not that Monte Carlo noise drifted.

        The week is asserted to be far from the day first. Without that the
        equality could pass for the wrong reason -- if the two arms ever became
        the same number, every horizon assertion in this file would be
        satisfied by a book whose funding does not depend on the horizon at
        all, which is exactly how the original regression stayed invisible.
        """
        base, state = service
        bundle, specs, spot = state.require_ready()
        r = call(base, "/portfolio_risk", {"book": BOOK, "n_paths": 4_000, "seed": 11})

        reference = MonteCarloEngine(bundle, specs).run_horizons(
            as_book(BOOK), spot, (24, 24 * 7), n_paths=4_000, seed=11,
        )
        day, week = reference[24].funding_cost, reference[24 * 7].funding_cost
        assert week.quantile(0.5) > 3.0 * day.quantile(0.5)

        wire = r["funding_cost_24h"]
        for key, q in (("median", 0.5), ("p05", 0.05), ("p95", 0.95)):
            assert wire[key] == pytest.approx(day.quantile(q), rel=1e-9), key
            assert wire[key] != pytest.approx(week.quantile(q), rel=1e-3), key

    def test_the_response_is_plain_json(self, service):
        """numpy scalars are not JSON serialisable, and np.float64 is a float
        subclass -- so an isinstance-based guard passes it through and the
        failure only shows up here, as a 400 on a valid request."""
        base, _ = service
        raw = json.dumps(call(base, "/portfolio_risk",
                              {"book": BOOK, "n_paths": 2_000, "seed": 3}))
        assert raw


class TestPreTradeDelta:
    def test_returns_both_significance_answers(self, service):
        base, _ = service
        d = call(base, "/pre_trade_delta", {
            "book": BOOK,
            "order": {"coin": "SOL", "size": 800, "leverage": 20, "mode": "cross"},
            "n_paths": 8_000, "seed": 4,
        })
        assert set(d["p_liq"]) >= {
            "before", "after", "change", "distinguishable",
            "marginal_intervals_overlap", "overlap_rule_would_mislead", "direction",
        }
        assert d["new_assets"] == ["SOL"]
        assert d["summary"]

    def test_latency_is_reported_whether_or_not_it_met_the_budget(self, service):
        base, _ = service
        d = call(base, "/pre_trade_delta", {
            "book": BOOK,
            "order": {"coin": "SOL", "size": 400, "leverage": 20, "mode": "cross"},
            "n_paths": 4_000, "seed": 5,
        })
        assert d["latency_ms"]["total"] > 0
        assert isinstance(d["within_budget"], bool)


class TestErrorHandling:
    def test_a_malformed_book_is_a_400_naming_the_field(self, service):
        base, _ = service
        with pytest.raises(urllib.error.HTTPError) as exc:
            call(base, "/portfolio_risk", {"book": {"address": "0x"}})
        assert exc.value.code == 400
        assert "cross_collateral" in json.loads(exc.value.read())["error"]

    def test_an_untracked_asset_is_refused_rather_than_guessed(self, service):
        base, _ = service
        with pytest.raises(urllib.error.HTTPError) as exc:
            call(base, "/pre_trade_delta", {
                "book": BOOK, "order": {"coin": "DOGE", "size": 1, "leverage": 5}})
        assert exc.value.code == 400

    def test_unknown_routes_are_404(self, service):
        base, _ = service
        with pytest.raises(urllib.error.HTTPError) as exc:
            call(base, "/nope")
        assert exc.value.code == 404


class TestStateLifecycle:
    def test_a_failed_rebuild_keeps_the_previous_bundle(self):
        """A transient venue failure must not be indistinguishable from a
        crash. The old bundle stays, the error is recorded, and the age keeps
        climbing so the backend's §6 thresholds are what eventually hide it."""
        state = EngineState.fixture()
        good_age = state.matrix_age_s()
        assert good_age is not None

        def boom():
            raise RuntimeError("venue unreachable")

        state._rebuild = boom
        assert state.refresh() is False
        assert state.require_ready()  # still serving the last good bundle
        health = state.health()
        assert health["ready"] is True
        assert "venue unreachable" in health["last_error"]
        assert health["matrix_age_s"] >= good_age
        state.stop()

    def test_a_state_that_never_built_refuses_to_serve(self):
        state = EngineState()

        def boom():
            raise RuntimeError("cold start failed")

        state._rebuild = boom
        state.refresh()
        assert state.health()["ready"] is False
        with pytest.raises(NotReady):
            state.require_ready()
        state.stop()
