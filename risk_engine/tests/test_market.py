"""§5.1 — Info response parsers.

Fixtures, not observations: `api.hyperliquid.xyz` is blocked at the proxy in
the environment this was written in, so these encode the documented shapes
and must be re-verified against the live API (OPEN-QUESTIONS E5). They are
still worth having -- they pin the parsing *decisions* (tiers from `meta`,
cross cash derived from account value minus unrealised PnL, strictness on
missing fields), which is what would otherwise rot silently.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import ClassVar

import numpy as np
import pytest

from risk_engine.domain.types import MarginMode
from risk_engine.market.info import InfoClient
from risk_engine.market.parse import (
    net_external_flow,
    parse_candles_to_log_returns,
    parse_clearinghouse_state,
    parse_funding_history,
    parse_meta,
)

META = {
    "universe": [
        {"name": "BTC", "szDecimals": 5, "maxLeverage": 40, "marginTableId": 1},
        {"name": "ETH", "szDecimals": 4, "maxLeverage": 25, "marginTableId": 2},
        {"name": "SOL", "szDecimals": 2, "maxLeverage": 20},
        {"name": "DEAD", "szDecimals": 2, "maxLeverage": 3, "isDelisted": True},
    ],
    "marginTables": [
        [1, {"description": "tiered", "marginTiers": [
            {"lowerBound": "0", "maxLeverage": 40},
            {"lowerBound": "150000000", "maxLeverage": 20},
        ]}],
        [2, {"description": "tiered", "marginTiers": [
            {"lowerBound": "0", "maxLeverage": 25},
        ]}],
    ],
}

STATE = {
    "marginSummary": {"accountValue": "132000.0", "totalNtlPos": "700000.0"},
    "crossMarginSummary": {"accountValue": "110000.0", "totalNtlPos": "500000.0"},
    "withdrawable": "40000.0",
    "assetPositions": [
        {"type": "oneWay", "position": {
            "coin": "BTC", "szi": "3.0", "entryPx": "96000.0",
            "positionValue": "300000.0", "unrealizedPnl": "12000.0",
            "marginUsed": "15000.0", "maxLeverage": 40,
            "leverage": {"type": "cross", "value": 20},
        }},
        {"type": "oneWay", "position": {
            "coin": "SOL", "szi": "-500.0", "entryPx": "210.0",
            "positionValue": "100000.0", "unrealizedPnl": "5000.0",
            # Internally consistent with the LIVE field semantics measured on
            # mainnet 2026-08-03: `marginUsed` is the pocket's EQUITY
            # (collateral + uPnL), and `rawUsd` is its net ledger cash. This
            # fixture previously carried rawUsd=22000 with the parser reading
            # it as the collateral -- a shape the venue does not produce, which
            # is how the misreading survived into a live sweep.
            # Short: cash = equity + notional = 20000 + 100000.
            "marginUsed": "20000.0", "maxLeverage": 20,
            "leverage": {"type": "isolated", "value": 5, "rawUsd": "120000.0"},
        }},
        {"type": "oneWay", "position": {
            "coin": "ZERO", "szi": "0.0", "entryPx": "1.0",
            "unrealizedPnl": "0.0", "leverage": {"type": "cross", "value": 1},
        }},
    ],
}


class TestParseMeta:
    def test_builds_tiered_specs(self):
        specs = parse_meta(META)
        assert set(specs) == {"BTC", "ETH", "SOL"}  # delisted asset dropped
        assert len(specs["BTC"].tiers) == 2
        assert specs["BTC"].maintenance_margin_rate(1e6) == pytest.approx(0.5 / 40)
        assert specs["BTC"].maintenance_margin_rate(2e8) == pytest.approx(0.5 / 20)

    def test_asset_without_a_table_gets_its_own_implied_tier(self):
        specs = parse_meta(META)
        assert len(specs["SOL"].tiers) == 1
        assert specs["SOL"].maintenance_margin_rate(1e6) == pytest.approx(0.5 / 20)

    def test_no_maintenance_rate_is_hardcoded_anywhere(self):
        """§1.3. Change the venue's table and every rate must follow."""
        halved = {
            "universe": [{"name": "BTC", "szDecimals": 5, "maxLeverage": 20,
                          "marginTableId": 1}],
            "marginTables": [[1, {"marginTiers": [{"lowerBound": "0", "maxLeverage": 20}]}]],
        }
        assert parse_meta(halved)["BTC"].maintenance_margin_rate(1.0) == pytest.approx(0.025)

    def test_rejects_an_empty_universe(self):
        with pytest.raises(ValueError, match="universe"):
            parse_meta({"universe": []})

    def test_an_unresolved_id_equal_to_max_leverage_is_the_flat_shape(self):
        """Measured against live mainnet meta, 2026-08-03.

        For an asset with no custom tiering the venue sets `marginTableId` to
        the same number as `maxLeverage` — it is not a reference into
        `marginTables` at all. Live counts over 177 assets: 34 resolve to a
        real table (ids 50-56), 143 are unresolved with id == maxLeverage
        (ATOM 5/5, GMX 3/3, SNX 3/3, ...), and ZERO are unresolved with an id
        that differs from maxLeverage. The id spaces do not overlap, so
        reading this as the flat shape cannot mask a real table.

        Without this the live snapshot refused the whole universe on ATOM and
        wrote nothing, every run.
        """
        flat = {
            "universe": [{"name": "ATOM", "szDecimals": 2, "maxLeverage": 5,
                          "marginTableId": 5}],
            "marginTables": [[56, {"marginTiers": [{"lowerBound": "0",
                                                    "maxLeverage": "40"}]}]],
        }
        spec = parse_meta(flat)["ATOM"]
        assert len(spec.tiers) == 1
        # The flat rate applies at every notional, including a huge one.
        assert spec.maintenance_margin_rate(1e9) == pytest.approx(0.5 / 5)

    def test_an_unresolved_id_that_is_not_the_flat_encoding_still_refuses(self):
        """The §10 guard this narrows must survive the narrowing.

        An id that resolves to nothing AND does not equal maxLeverage is a
        genuine venue-shape change. Absorbing it would hand a large book the
        small-size maintenance rate — understating margin, and P(liq) with it.
        """
        anomaly = {
            "universe": [{"name": "WEIRD", "szDecimals": 2, "maxLeverage": 10,
                          "marginTableId": 99}],
            "marginTables": [[56, {"marginTiers": [{"lowerBound": "0",
                                                    "maxLeverage": "40"}]}]],
        }
        with pytest.raises(ValueError, match="marginTableId"):
            parse_meta(anomaly)


class TestParseState:
    def test_splits_cross_cash_from_unrealised_pnl(self):
        book = parse_clearinghouse_state(STATE, "0xabc", datetime.now(timezone.utc))
        # crossMarginSummary.accountValue 110000 includes +12000 of cross upnl.
        assert book.cross_collateral == pytest.approx(98_000.0)
        assert book.cross_account_value({"BTC": 96_000.0}) == pytest.approx(98_000.0)

    def test_reads_modes_and_isolated_margin(self):
        book = parse_clearinghouse_state(STATE, "0xabc")
        by_coin = {p.coin: p for p in book.positions}
        assert by_coin["BTC"].mode is MarginMode.CROSS
        assert by_coin["BTC"].isolated_margin is None
        assert by_coin["SOL"].mode is MarginMode.ISOLATED
        # marginUsed (20000, the pocket's equity) minus uPnL (5000).
        # NOT rawUsd: that is the pocket's ledger cash, and reading it as
        # collateral put a long's negative cash into a field `Position`
        # refuses, dropping whole live accounts, while a short's positive cash
        # would have passed silently at ~50x the true margin (§10).
        assert by_coin["SOL"].isolated_margin == pytest.approx(15_000.0)
        assert by_coin["SOL"].size == -500.0

    def test_drops_zero_size_positions(self):
        book = parse_clearinghouse_state(STATE, "0xabc")
        assert "ZERO" not in {p.coin for p in book.positions}

    def test_refuses_an_unknown_margin_mode_rather_than_guessing(self):
        broken = {
            "crossMarginSummary": {"accountValue": "1.0"},
            "assetPositions": [{"position": {
                "coin": "BTC", "szi": "1.0", "entryPx": "1.0", "unrealizedPnl": "0.0",
                "leverage": {"type": "portfolio", "value": 1},
            }}],
        }
        with pytest.raises(ValueError, match="unknown margin mode"):
            parse_clearinghouse_state(broken, "0xabc")

    def test_refuses_a_missing_numeric_field(self):
        broken = {
            "crossMarginSummary": {"accountValue": "1.0"},
            "assetPositions": [{"position": {
                "coin": "BTC", "szi": "1.0", "unrealizedPnl": "0.0",
                "leverage": {"type": "cross", "value": 1},
            }}],
        }
        with pytest.raises(ValueError, match="entryPx"):
            parse_clearinghouse_state(broken, "0xabc")

    def test_an_empty_state_parses_to_an_empty_book_not_an_error(self):
        book = parse_clearinghouse_state(
            {"crossMarginSummary": {"accountValue": "0.0"}, "assetPositions": []}, "0x"
        )
        assert book.positions == ()


class TestLiveIsolatedMargin:
    """E6 — verbatim mainnet responses, not fixtures.

    These two are the only observations in this file. They are here because
    the documented reading of `leverage.rawUsd` was wrong and every fixture in
    this module agreed with it, so nothing failed: the shapes were consistent
    with each other and with the docs, and inconsistent with the venue.

    The venue's own `liquidationPx` is the oracle. Reconstructing it from what
    the parser produced is what makes these tests observations rather than
    another set of numbers someone typed.
    """

    # Captured from mainnet 2026-08-03.
    LONG: ClassVar[dict] = {
        "coin": "BTC", "szi": "0.00347",
        "leverage": {"type": "isolated", "value": 40, "rawUsd": "-213.037712"},
        "entryPx": "62917.0", "positionValue": "217.17342",
        "unrealizedPnl": "-1.14857", "liquidationPx": "62171.2944953124",
        "marginUsed": "4.135708", "maxLeverage": 40,
    }
    SHORT: ClassVar[dict] = {
        "coin": "BTC", "szi": "-0.00024",
        "leverage": {"type": "isolated", "value": 40, "rawUsd": "15.665304"},
        "entryPx": "63709.5", "positionValue": "15.042",
        "unrealizedPnl": "0.24828", "liquidationPx": "64466.2716049383",
        "marginUsed": "0.623304", "maxLeverage": 40,
    }

    @staticmethod
    def _parse(position: dict):
        state = {
            "marginSummary": {"accountValue": "100.0"},
            "crossMarginSummary": {"accountValue": "100.0"},
            "assetPositions": [{"type": "oneWay", "position": position}],
        }
        return parse_clearinghouse_state(state, "0x" + "cc" * 20).positions[0]

    @pytest.mark.parametrize("tag", ["LONG", "SHORT"])
    def test_the_parsed_collateral_reproduces_the_venues_liquidation_price(self, tag):
        """The check that was missing, as a test.

        If `isolated_margin` is wrong, the §1.1 condition solved at that
        collateral lands somewhere other than where the venue says the
        position liquidates. Nothing else in this file would have caught the
        `rawUsd` misreading; this does, on both sides.
        """
        raw = getattr(self, tag)
        pos = self._parse(raw)
        size, entry = float(raw["szi"]), float(raw["entryPx"])
        mmr = 0.5 / float(raw["maxLeverage"])
        # equity(P) = margin + size*(P - entry) == mmr*|size|*P
        ours = (pos.isolated_margin - size * entry) / (mmr * abs(size) - size)
        theirs = float(raw["liquidationPx"])
        assert ours == pytest.approx(theirs, rel=1e-9)

    def test_a_long_pocket_is_no_longer_dropped(self):
        """`rawUsd` is NEGATIVE for a long -- the venue buys partly on
        borrowed dollars -- so reading it as collateral made `Position` refuse
        the whole account. Five live accounts went that way in one sweep."""
        assert float(self.LONG["leverage"]["rawUsd"]) < 0
        pos = self._parse(self.LONG)
        assert pos.isolated_margin > 0
        assert pos.isolated_margin == pytest.approx(5.284278)

    def test_a_short_pocket_is_not_silently_inflated(self):
        """The dangerous half. For a short, `rawUsd` is
        `marginUsed + positionValue` -- positive, so it would have PARSED,
        at ~40x the true collateral, placing liquidation far away and
        understating P(liq). §10 forbids that direction."""
        raw_usd = float(self.SHORT["leverage"]["rawUsd"])
        assert raw_usd > 0, "positive, so nothing would have refused it"
        pos = self._parse(self.SHORT)
        assert pos.isolated_margin == pytest.approx(0.375024)
        assert raw_usd / pos.isolated_margin > 40, "the size of the near-miss"

    def test_raw_usd_is_the_pockets_cash_on_both_sides(self):
        """`rawUsd == marginUsed - sign(size)*positionValue`, exact on both.

        The sign is the whole point and is easy to get wrong -- the long form
        alone does not generalise. A long BORROWS dollars to hold the asset,
        so its cash is negative; a short HOLDS dollars against an asset it
        owes, so its cash is positive and larger than the pocket. That is what
        identifies the field as ledger cash rather than collateral, and it is
        why `probe_isolated_funding` is right to read it: cash reduces to
        `collateral - size*entry`, which carries no mark-price term.
        """
        for raw in (self.LONG, self.SHORT):
            size = float(raw["szi"])
            side = 1.0 if size > 0 else -1.0
            expected = float(raw["marginUsed"]) - side * float(raw["positionValue"])
            assert float(raw["leverage"]["rawUsd"]) == pytest.approx(expected, abs=1e-6)


class TestAgentAddressGuard:
    def test_agent_address_is_refused_before_the_request_is_made(self):
        """§5.1's named footgun: an agent address returns a well-formed empty
        state, which reads as 'no positions' -- the worst possible silent
        failure for a risk tool."""
        client = InfoClient()
        with pytest.raises(ValueError, match="real account address"):
            client.clearinghouse_state("0xagent", is_agent_address=True)
        assert client.budget.spent() == 0  # nothing was charged


class TestPublishedWeightTable:
    """§5.3's weights, as the venue publishes them (OPEN-QUESTIONS C6).

    This client charged a flat 20 for every request until 2026-08-03 and the
    entry recording that called it a safe error, because over-charging
    self-limits harder than the venue asks. Reading the published table
    showed the flat rate is wrong in BOTH directions, and the entry's whole
    justification only covered one of them.
    """

    def test_clearinghouse_state_is_the_cheap_tier(self):
        """One request per address, the dominant cost of the shadow sweep,
        billed at 2 and charged at 20 — a 10x self-limit that made §3.3's
        200-address floor look like 27 minutes of budget instead of two."""
        from risk_engine.market.info import info_request_weight

        assert info_request_weight("clearinghouseState") == 2
        assert info_request_weight("l2Book") == 2
        assert info_request_weight("userRole") == 60
        assert info_request_weight("meta") == 20, "the default still applies"

    def test_a_long_response_costs_more_than_its_base_weight(self):
        """The direction the old flat rate got DANGEROUSLY wrong.

        `candleSnapshot` and `fundingHistory` bill per item returned on top
        of their base. A 90-day hourly candle fetch is 2160 items, so the
        bundle build was spending far more than it recorded — and unrecorded
        spend is exactly what eats the reserve §5.3 promises live users.
        """
        from risk_engine.market.info import info_response_surcharge

        assert info_response_surcharge("candleSnapshot", [{}] * 2160) == 36
        assert info_response_surcharge("fundingHistory", [{}] * 720) == 36
        assert info_response_surcharge("meta", [{}] * 2160) == 0
        assert info_response_surcharge("candleSnapshot", {"not": "a list"}) == 0

    def test_the_confusingly_named_ledger_call_is_not_surcharged(self):
        """The published list contains `nonUserFundingUpdates`. The call B2's
        resolver makes per address is `userNonFundingLedgerUpdates` — a
        different endpoint whose name differs by a transposition. Reading one
        for the other would invent a surcharge on the hottest B2 path."""
        from risk_engine.market.info import info_response_surcharge

        assert info_response_surcharge("userNonFundingLedgerUpdates", [{}] * 500) == 0
        assert info_response_surcharge("nonUserFundingUpdates", [{}] * 500) == 25

    def test_the_surcharge_is_recorded_even_when_it_overshoots(self):
        """It cannot refuse: the request is already on the wire and the venue
        has already counted it. Refusing would discard a paid-for response;
        pretending it was free would understate the window. So it overshoots
        and the NEXT charge waits — one request late, which is the best
        available answer for a cost that is not knowable in advance."""
        from risk_engine.market.info import RateLimitExceeded, WeightBudget

        budget = WeightBudget(limit_per_minute=100, reserved_fraction=0.0)
        budget.charge(90)
        budget.charge_incurred(50)  # must not raise
        assert budget.spent() == 140
        assert budget.available() == 0, "floors at zero rather than going negative"
        with pytest.raises(RateLimitExceeded):
            budget.charge(1)


class TestCandlesAndFunding:
    def test_log_returns_from_candles(self):
        candles = [{"t": 3_600_000 * i, "c": str(100.0 * 1.01**i)} for i in range(5)]
        times, rets = parse_candles_to_log_returns(candles)
        assert times.size == rets.size == 4
        assert np.allclose(rets, np.log(1.01))

    def test_candles_are_sorted_before_differencing(self):
        candles = [{"t": 2, "c": "110"}, {"t": 1, "c": "100"}]
        _, rets = parse_candles_to_log_returns(candles)
        assert rets[0] == pytest.approx(np.log(1.1))

    def test_rejects_a_non_positive_close(self):
        with pytest.raises(ValueError, match="non-positive"):
            parse_candles_to_log_returns([{"t": 1, "c": "0"}, {"t": 2, "c": "1"}])

    def test_funding_history_parses_in_time_order(self):
        history = [
            {"coin": "BTC", "time": 2, "fundingRate": "0.00002"},
            {"coin": "BTC", "time": 1, "fundingRate": "0.00001"},
        ]
        times, rates = parse_funding_history(history)
        assert list(times) == [1, 2]
        assert rates[0] == pytest.approx(1e-5)


class TestParserStrictness:
    """Audit A-07 and A-08: the two places the parser was lenient, both in
    the direction of understating risk."""

    @pytest.mark.parametrize("bad", ["NaN", "nan", "Infinity", "inf", "-inf"])
    def test_non_finite_numbers_are_refused(self, bad):
        """A NaN size sails through every later `<=` check, because NaN
        compares False against everything, and lands in the journal."""
        state = {
            "crossMarginSummary": {"accountValue": "5000.0"},
            "assetPositions": [{"position": {
                "coin": "BTC", "szi": bad, "entryPx": "100000.0",
                "unrealizedPnl": "0.0", "leverage": {"type": "cross", "value": 20},
            }}],
        }
        with pytest.raises(ValueError, match="not finite"):
            parse_clearinghouse_state(state, "0xpoisoned")

    def test_non_finite_account_value_is_refused(self):
        state = {"crossMarginSummary": {"accountValue": "NaN"}, "assetPositions": []}
        with pytest.raises(ValueError, match="not finite"):
            parse_clearinghouse_state(state, "0x")

    def test_margin_summary_is_not_substituted_for_cross(self):
        """`marginSummary` covers the WHOLE account including isolated
        pockets. Using it as the cross account value double-counts every
        isolated margin -- once in cross_collateral, again in the position --
        which inflates equity and understates risk (§10)."""
        state = {
            "marginSummary": {"accountValue": "132000.0"},
            "assetPositions": [{"position": {
                "coin": "SOL", "szi": "-500.0", "entryPx": "210.0",
                "unrealizedPnl": "5000.0", "marginUsed": "20000.0",
                "leverage": {"type": "isolated", "value": 5, "rawUsd": "22000.0"},
            }}],
        }
        with pytest.raises(ValueError, match="crossMarginSummary"):
            parse_clearinghouse_state(state, "0x")

    def test_non_finite_margin_tier_is_refused(self):
        meta = {
            "universe": [{"name": "X", "szDecimals": 2, "maxLeverage": 10,
                          "marginTableId": 1}],
            "marginTables": [[1, {"marginTiers": [
                {"lowerBound": "0", "maxLeverage": "Infinity"},
            ]}]],
        }
        with pytest.raises(ValueError, match="not finite"):
            parse_meta(meta)


class TestExternalFlow:
    """B2's correction: equity moves the model does not predict.

    The dangerous direction here is silence. A deposit scored as an equity
    change is a spectacular apparent model failure, and journal rows are
    written once -- so a flow missed at resolution time is missed forever.
    Every test below is about refusing rather than guessing.
    """

    SINCE = datetime(2026, 7, 30, 0, 0, tzinfo=timezone.utc)
    UNTIL = datetime(2026, 7, 31, 0, 0, tzinfo=timezone.utc)

    def _row(self, kind, usdc, *, at=None, **extra):
        at = at or datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
        return {
            "time": int(at.timestamp() * 1000),
            "delta": {"type": kind, "usdc": str(usdc), **extra},
        }

    def test_a_deposit_is_positive_and_a_withdrawal_negative(self):
        flow = net_external_flow(
            [self._row("deposit", 50_000), self._row("withdraw", 20_000)],
            self.SINCE, self.UNTIL,
        )
        assert flow == pytest.approx(30_000.0)

    def test_the_sign_is_read_from_the_record_not_the_type_name(self):
        """A directional transfer's name says nothing about which way it went.
        Inferring it would be a coin flip on exactly the records where being
        wrong flips the sign of the correction."""
        into = net_external_flow(
            [self._row("accountClassTransfer", 1_000, toPerp=True)], self.SINCE, self.UNTIL
        )
        out = net_external_flow(
            [self._row("accountClassTransfer", 1_000, toPerp=False)], self.SINCE, self.UNTIL
        )
        assert into == pytest.approx(1_000.0)
        assert out == pytest.approx(-1_000.0)

    def test_a_directional_transfer_without_a_direction_is_refused(self):
        with pytest.raises(ValueError, match="no 'toPerp' flag"):
            net_external_flow(
                [self._row("accountClassTransfer", 1_000)], self.SINCE, self.UNTIL
            )

    def test_an_unknown_delta_type_raises_rather_than_being_ignored(self):
        """The finding this whole parser is shaped around. The venue can add a
        transfer type whenever it likes; ignoring one is invisible, permanent,
        and lands in the direction that makes the model look wrong."""
        with pytest.raises(ValueError, match="unknown ledger delta type"):
            net_external_flow(
                [self._row("someNewTransferKind", 50_000)], self.SINCE, self.UNTIL
            )

    def test_a_known_non_flow_is_ignored_without_complaint(self):
        """Liquidation moves equity and the model DOES predict it -- counting
        it as external flow would subtract the very thing being scored."""
        assert net_external_flow(
            [self._row("liquidation", 9_999)], self.SINCE, self.UNTIL
        ) == pytest.approx(0.0)

    def test_records_outside_the_window_do_not_count(self):
        before = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)
        after = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        assert net_external_flow(
            [self._row("deposit", 1_000, at=before), self._row("deposit", 2_000, at=after)],
            self.SINCE, self.UNTIL,
        ) == pytest.approx(0.0)

    def test_an_empty_ledger_is_zero_flow_not_an_error(self):
        """The overwhelmingly common case: most accounts deposit nothing on
        most days, and that genuinely is zero rather than unknown."""
        assert net_external_flow([], self.SINCE, self.UNTIL) == pytest.approx(0.0)
        assert net_external_flow(None, self.SINCE, self.UNTIL) == pytest.approx(0.0)

    def test_a_record_with_no_amount_is_refused(self):
        with pytest.raises(ValueError, match="no USD amount"):
            net_external_flow(
                [{"time": int(self.UNTIL.timestamp() * 1000) - 1,
                  "delta": {"type": "deposit"}}],
                self.SINCE, self.UNTIL,
            )


class TestDexRoutedFlows:
    """`send`, and why a type name is not enough to classify a transfer.

    The live finding (2026-07-31, mainnet): a real account's ledger carried a
    `send` this build had never seen. The record decided it -- spot to spot,
    so no perp equity moved -- but the SAME type with `sourceDex: "perp"` is
    a real outflow. Filing `send` wholesale under either table would have
    been wrong half the time, and the wrong half would have been invisible.
    """

    ME = "0xd47587702a91731dc1089b5db0932cf820151a91"
    OTHER = "0xd048870caa5a3037f507583b4762a7598251a2fc"
    SINCE = datetime(2026, 1, 1, tzinfo=timezone.utc)
    UNTIL = datetime(2027, 1, 1, tzinfo=timezone.utc)

    def _send(self, **overrides):
        """The exact shape the venue returned, verbatim, with overrides."""
        delta = {
            "type": "send", "user": self.ME, "destination": self.OTHER,
            "sourceDex": "spot", "destinationDex": "spot", "token": "HYPE",
            "amount": "5.0", "usdcValue": "206.575", "fee": "0.0",
            "nativeTokenFee": "0.0", "nonce": 1777910139749, "feeToken": "",
        }
        delta.update(overrides)
        return [{"time": 1777910224883, "hash": "0x3ef2", "delta": delta}]

    def test_the_live_record_is_not_a_perp_flow(self):
        """Spot to spot: perp equity did not move, so subtracting it would
        corrupt the very correction B2 exists to make."""
        assert net_external_flow(
            self._send(), self.SINCE, self.UNTIL, self.ME
        ) == pytest.approx(0.0)

    def test_the_same_type_out_of_perp_is_an_outflow(self):
        assert net_external_flow(
            self._send(sourceDex="perp"), self.SINCE, self.UNTIL, self.ME
        ) == pytest.approx(-206.575)

    def test_the_same_type_into_perp_is_an_inflow(self):
        """Received rather than sent, so the direction comes from
        destinationDex and the sign flips."""
        assert net_external_flow(
            self._send(user=self.OTHER, destination=self.ME, destinationDex="perp"),
            self.SINCE, self.UNTIL, self.ME,
        ) == pytest.approx(+206.575)

    def test_the_usd_value_is_used_not_the_token_amount(self):
        """`amount` is 5.0 HYPE and `usdcValue` is $206.575. Reading the
        token quantity as dollars would be a silent 40x error, and the old
        code looked only for `usdc` -- which this record does not carry --
        so it would have raised on a perfectly well-formed row."""
        flow = net_external_flow(
            self._send(sourceDex="perp"), self.SINCE, self.UNTIL, self.ME
        )
        assert flow == pytest.approx(-206.575)
        assert flow != pytest.approx(-5.0)

    def test_the_empty_dex_is_the_primary_perp_account(self):
        """The 2026-07-31 frame sweep's most expensive record. Hyperliquid
        names the primary perp dex with the empty string; `""` was filed as
        non-perp, so this $25,000 outflow scored **zero** — a $25k unexplained
        equity drop blamed on the model, in the §10-forbidden direction.

        The venue's own spelling settles it: the same field carries the
        literal "spot" in one record and "" in another, and a venue that
        writes "spot" when it means spot does not also write ""."""
        rows = self._send(
            sourceDex="", destinationDex="xyz", token="USDC",
            amount="25000.0", usdcValue="25000.0",
        )
        assert net_external_flow(
            rows, self.SINCE, self.UNTIL, self.ME
        ) == pytest.approx(-25_000.0)

    def test_a_builder_dex_is_external_to_the_account_being_modelled(self):
        """A named dex is a HIP-3 builder venue: perp in the ordinary sense,
        and a different margin space from the one `clearinghouseState`
        returns. Receiving into it does not credit the modelled book."""
        assert net_external_flow(
            self._send(user=self.OTHER, destination=self.ME,
                       sourceDex="spot", destinationDex="xyz"),
            self.SINCE, self.UNTIL, self.ME,
        ) == pytest.approx(0.0)

    def test_an_unknown_dex_name_is_recorded_rather_than_silently_accepted(self):
        """It no longer raises — new venues are a thing third parties create,
        and raising would break the §3.3 window on whichever address first
        touched one. But the reading has exactly one failure mode: a future
        ALIAS of the primary dex read as a builder venue would hide a real
        outflow. So the names are tallied for `verify` to surface."""
        from risk_engine.market.parse import UNRECOGNISED_DEX_NAMES

        UNRECOGNISED_DEX_NAMES.clear()
        net_external_flow(
            self._send(sourceDex="somethingNew"), self.SINCE, self.UNTIL, self.ME
        )
        assert UNRECOGNISED_DEX_NAMES.get("somethingnew") == 1

    def test_a_dex_routed_record_without_an_address_raises(self):
        """Direction depends on which side this account was on, which cannot
        be read off the record alone."""
        with pytest.raises(ValueError, match="no address was supplied"):
            net_external_flow(self._send(), self.SINCE, self.UNTIL, None)

    def test_a_send_involving_neither_side_raises(self):
        """A record in this account's ledger naming neither side is a shape
        this build does not understand; silently skipping it would be a guess."""
        with pytest.raises(ValueError, match="neither this account"):
            net_external_flow(
                self._send(user=self.OTHER, destination="0x" + "9" * 40),
                self.SINCE, self.UNTIL, self.ME,
            )

    def test_a_perp_to_perp_self_transfer_nets_to_zero(self):
        """Both legs are this account's perp account, so no equity entered or
        left it. Evaluated as if/elif the outflow leg would fire and the
        inflow leg would never be reached, scoring a phantom -$206 outflow --
        which the model would then have to explain as a prediction error."""
        assert net_external_flow(
            self._send(destination=self.ME, sourceDex="perp", destinationDex="perp"),
            self.SINCE, self.UNTIL, self.ME,
        ) == pytest.approx(0.0)

    def test_a_perp_to_spot_self_transfer_is_a_real_outflow(self):
        """The same account on both sides, but only one side is perp: equity
        genuinely left the account this model predicts."""
        assert net_external_flow(
            self._send(destination=self.ME, sourceDex="perp", destinationDex="spot"),
            self.SINCE, self.UNTIL, self.ME,
        ) == pytest.approx(-206.575)

    def test_a_spot_to_perp_self_transfer_is_a_real_inflow(self):
        assert net_external_flow(
            self._send(destination=self.ME, sourceDex="spot", destinationDex="perp"),
            self.SINCE, self.UNTIL, self.ME,
        ) == pytest.approx(+206.575)

    def test_a_non_perp_send_is_skipped_before_its_amount_is_read(self):
        """Spot to spot carries no perp flow, so a record missing the USD
        field must not be refused: refusing it would fail the whole resolution
        over a value this correction never uses."""
        row = self._send()
        del row[0]["delta"]["usdcValue"]
        assert net_external_flow(row, self.SINCE, self.UNTIL, self.ME) == pytest.approx(0.0)

    def test_only_the_involved_side_s_dex_has_to_parse(self):
        """This account received; the sender's dex is a field about someone
        else's account and an unrecognised value there must not fail the run."""
        assert net_external_flow(
            self._send(
                user=self.OTHER, destination=self.ME,
                sourceDex="somethingNew", destinationDex="perp",
            ),
            self.SINCE, self.UNTIL, self.ME,
        ) == pytest.approx(+206.575)

    def test_address_comparison_is_case_insensitive(self):
        """Ledger records and address lists disagree on case constantly; a
        checksummed spelling must not read as 'neither side'."""
        assert net_external_flow(
            self._send(sourceDex="perp", user=self.ME.upper().replace("0X", "0x")),
            self.SINCE, self.UNTIL, self.ME,
        ) == pytest.approx(-206.575)


class TestTheFrameSweepFindings:
    """The eight delta types the 2026-07-31 frame sweep found on 50 addresses.

    One address had five types; fifty had thirteen. Every record below is
    verbatim from that run, and each is classified by its FIELDS rather than
    its name — the rule that did the work being: a record carrying `usdc`
    moves perp collateral, while one carrying `token` and `amount` is
    denominated in a spot asset and does not.
    """

    ME = "0x706bb519b05b7dc01d048af9a5e29d1ef5d6d9e3"
    OTHER = "0x399965e15d4e61ec3529cc98b7f7ebb93b733336"
    SINCE = datetime(2026, 1, 1, tzinfo=timezone.utc)
    UNTIL = datetime(2027, 1, 1, tzinfo=timezone.utc)

    def _rows(self, delta):
        return [{"time": 1779457645318, "hash": "0x0", "delta": delta}]

    def _flow(self, delta, me=None):
        return net_external_flow(self._rows(delta), self.SINCE, self.UNTIL,
                                 self.ME if me is None else me)

    # -- perp USDC moved between two accounts, signed by which side we are on

    def test_a_sub_account_transfer_is_signed_by_side(self):
        d = {"type": "subAccountTransfer", "usdc": "2000.0",
             "user": self.ME, "destination": self.OTHER}
        assert self._flow(d) == pytest.approx(-2000.0)
        assert self._flow(d, self.OTHER) == pytest.approx(+2000.0)

    def test_an_internal_transfer_debits_the_sender_the_fee_as_well(self):
        """`fee: 1.0` beside `usdc: 40.0`: the sender loses 41, the recipient
        gains 40. Dropping the fee leaves $1 of real, explained outflow
        looking like model error on every transfer — small per record and
        systematically one-signed, which is the bias a calibration score is
        least able to absorb."""
        d = {"type": "internalTransfer", "usdc": "40.0", "fee": "1.0",
             "user": self.ME, "destination": self.OTHER}
        assert self._flow(d) == pytest.approx(-41.0)
        assert self._flow(d, self.OTHER) == pytest.approx(+40.0)

    def test_a_native_token_fee_is_not_counted_as_usd(self):
        """`nativeTokenFee` is paid in HYPE from a spot balance. Counting it
        as dollars is the same category error as reading a token `amount` as
        a USD figure."""
        d = {"type": "internalTransfer", "usdc": "40.0", "fee": "0.0",
             "nativeTokenFee": "3.5", "user": self.ME, "destination": self.OTHER}
        assert self._flow(d) == pytest.approx(-40.0)

    # -- spot-denominated: token + amount, no `usdc`, no perp movement

    @pytest.mark.parametrize("delta", [
        {"type": "gossipPriorityGasAuction", "token": "HYPE", "amount": "0.34223973"},
        {"type": "cStakingTransfer", "token": "HYPE", "amount": "25.0",
         "isDeposit": False},
        {"type": "borrowLend", "token": "USDC", "operation": "withdraw",
         "amount": "1000.0", "interestAmount": "0.93372645"},
    ], ids=["gas-auction", "staking", "borrow-lend"])
    def test_spot_denominated_records_move_no_perp_equity(self, delta):
        assert self._flow(delta) == pytest.approx(0.0)

    def test_borrow_lend_naming_usdc_is_still_not_perp_collateral(self):
        """The subtle one. `token: "USDC"` is not the same thing as `usdc:`.
        A perp-side movement of the same size would carry a USDC figure
        directly; this carries a token and a quantity, which is the spot
        convention throughout this ledger."""
        d = {"type": "borrowLend", "token": "USDC", "operation": "withdraw",
             "amount": "1000.0", "interestAmount": "0.93372645"}
        assert self._flow(d) == pytest.approx(0.0)
        # And the reading is falsifiable rather than merely plausible.
        with pytest.raises(ValueError, match="contradicts that"):
            self._flow({**d, "usdc": "1000.0"})

    # -- vault movements

    def test_a_vault_distribution_is_an_inflow(self):
        d = {"type": "vaultDistribution", "vault": "0x4342", "usdc": "88.985153"}
        assert self._flow(d) == pytest.approx(+88.985153)

    def test_a_vault_withdrawal_credits_what_arrived_not_what_was_requested(self):
        """Two USD figures and only one of them arrived. `requestedUsd` is
        310, `netWithdrawnUsd` is 294.66 after commission — reading the former
        would credit the account with $15.34 it never received, on every vault
        withdrawal in the window."""
        d = {"type": "vaultWithdraw", "vault": "0xd6e5", "user": self.ME,
             "requestedUsd": "310.0", "commission": "15.336366",
             "closingCost": "0.0", "basis": "156.636333",
             "netWithdrawnUsd": "294.663634"}
        got = self._flow(d)
        assert got == pytest.approx(+294.663634)
        assert got != pytest.approx(+310.0)


class TestAuditFindings:
    """Regressions for the 2026-07-31 adversarial audit (F-1, F-4, F-5, F-8).

    Every test here reproduces a PoC that succeeded against the shipped code:
    these are not hypothetical inputs, they are inputs that silently produced
    a wrong number (or a wrong silence) before the fix.
    """

    ME = "0x" + "a" * 40
    OTHER = "0x" + "b" * 40
    SINCE = datetime(2026, 1, 1, tzinfo=timezone.utc)
    UNTIL = datetime(2027, 1, 1, tzinfo=timezone.utc)

    def _flow(self, delta, me=None, t=1779457645318):
        return net_external_flow([{"time": t, "hash": "0x0", "delta": delta}],
                                 self.SINCE, self.UNTIL,
                                 self.ME if me is None else me)

    # -- F-1: toPerp must be a bool, not merely truthy

    def test_a_stringified_false_does_not_flip_the_sign(self):
        """PoC-1: toPerp='false' is truthy, so a perp->spot transfer of $1000
        scored +1000 — a 2x-the-amount error per record, with no exception
        anywhere. The frame sweep cannot catch this class: it proves records
        read without raising, and a sign flip does not raise."""
        with pytest.raises(ValueError, match="not a boolean"):
            self._flow({"type": "accountClassTransfer", "usdc": "1000.0",
                        "toPerp": "false"})

    def test_a_stringified_true_is_refused_the_same_way(self):
        """The value that happens to give the right answer is refused too:
        accepting 'true' while refusing 'false' would mean the check only
        fires on half the schema drift, the half that was already wrong."""
        with pytest.raises(ValueError, match="not a boolean"):
            self._flow({"type": "accountClassTransfer", "usdc": "1000.0",
                        "toPerp": "true"})

    def test_real_booleans_still_work_both_ways(self):
        assert self._flow({"type": "accountClassTransfer", "usdc": "1000.0",
                           "toPerp": True}) == pytest.approx(+1000.0)
        assert self._flow({"type": "accountClassTransfer", "usdc": "1000.0",
                           "toPerp": False}) == pytest.approx(-1000.0)

    # -- F-4: the fee's denomination comes from feeToken

    def test_a_token_denominated_fee_is_not_read_as_dollars(self):
        """PoC-2: fee='5.0', feeToken='HYPE' was subtracted as $5 — mistaking
        ~$200 of HYPE for five dollars, the same category error as reading a
        token amount as USD, one field over. A token fee is paid from a token
        balance, which is not in Book.equity: its perp contribution is zero,
        exactly like nativeTokenFee."""
        got = self._flow({"type": "send", "user": self.ME, "destination": self.OTHER,
                          "sourceDex": "", "destinationDex": "spot", "token": "USDC",
                          "amount": "100.0", "usdcValue": "100.0",
                          "fee": "5.0", "feeToken": "HYPE"})
        assert got == pytest.approx(-100.0)

    def test_a_usdc_fee_still_debits_the_sender(self):
        for fee_token in ("", "USDC", "usdc"):
            got = self._flow({"type": "send", "user": self.ME,
                              "destination": self.OTHER, "sourceDex": "",
                              "destinationDex": "spot", "token": "USDC",
                              "amount": "100.0", "usdcValue": "100.0",
                              "fee": "5.0", "feeToken": fee_token})
            assert got == pytest.approx(-105.0), fee_token

    # -- F-5: a record without `time` cannot bypass classification

    def test_a_timeless_record_is_refused_not_skipped(self):
        """PoC-3: the window filter sat before classification, so an
        unclassifiable $9,999 delta with no `time` scored a silent zero and
        the frame sweep marked its type 'readable' without ever parsing it."""
        with pytest.raises(ValueError, match="no 'time'"):
            net_external_flow(
                [{"hash": "0x0", "delta": {"type": "absolutelyUnknownKind",
                                           "usdc": "9999"}}],
                self.SINCE, self.UNTIL, self.ME,
            )

    def test_an_out_of_window_record_is_still_skipped(self):
        """The refusal is for missing time, not for out-of-range time — a
        record from before the window genuinely does not belong to it."""
        assert net_external_flow(
            [{"time": 1, "delta": {"type": "absolutelyUnknownKind"}}],
            self.SINCE, self.UNTIL, self.ME,
        ) == pytest.approx(0.0)

    # -- F-8: rewardsClaim is now falsifiable

    def test_a_rewards_claim_carrying_perp_collateral_is_refused(self):
        """PoC-4: rewardsClaim sat outside the guarded set with zero live
        records behind it, so a rewards programme paying into perp USDC would
        have been a hidden inflow scored as model error. The model does not
        predict rewards, so liquidation's exemption does not apply; the only
        honest basis for non-flow is spot denomination, which the guard now
        checks per record."""
        with pytest.raises(ValueError, match="contradicts that"):
            self._flow({"type": "rewardsClaim", "usdc": "500.0"})

    def test_a_spot_denominated_rewards_claim_is_still_a_non_flow(self):
        assert self._flow({"type": "rewardsClaim", "token": "PURR",
                           "amount": "12.5"}) == pytest.approx(0.0)


class TestSpotOnlyTransfers:
    """`spotTransfer`, and why a spot balance is not the predicted quantity.

    The second live finding (2026-07-31, mainnet), surfaced by `verify` on the
    same account one round after `send`:

        {"type": "spotTransfer", "token": "UFART", "amount": "20.0",
         "usdcValue": "4.9884", "user": "0x2000...010d",
         "destination": "0xd475...", "fee": "0.0", ...}

    An airdrop landing in a spot wallet. It was filed as
    directional-needs-`toPerp`, the record carries no such flag, and B2
    refused it. The classification is settled by what the model predicts, not
    by what the transfer is called: `Book.equity` is cross collateral plus the
    isolated pockets, so a spot balance is not in it and $4.99 of a memecoin
    arriving there changes nothing being forecast.
    """

    ME = "0xd47587702a91731dc1089b5db0932cf820151a91"
    SENDER = "0x200000000000000000000000000000000000010d"
    SINCE = datetime(2026, 1, 1, tzinfo=timezone.utc)
    UNTIL = datetime(2027, 1, 1, tzinfo=timezone.utc)

    def _spot_transfer(self, **overrides):
        """The exact shape the venue returned, verbatim, with overrides."""
        delta = {
            "type": "spotTransfer", "token": "UFART", "amount": "20.0",
            "usdcValue": "4.9884", "user": self.SENDER, "destination": self.ME,
            "fee": "0.0", "nativeTokenFee": "0.0", "nonce": 1219326, "feeToken": "",
        }
        delta.update(overrides)
        return [{"time": 1778067666017, "hash": "0x025c", "delta": delta}]

    def test_the_live_record_is_not_a_perp_flow(self):
        assert net_external_flow(
            self._spot_transfer(), self.SINCE, self.UNTIL, self.ME
        ) == pytest.approx(0.0)

    def test_it_needs_no_address_to_classify(self):
        """Unlike `send`, nothing about the direction matters: neither side is
        the perp account, so there is no sign to get wrong."""
        assert net_external_flow(
            self._spot_transfer(), self.SINCE, self.UNTIL, None
        ) == pytest.approx(0.0)

    def test_a_spot_transfer_naming_the_perp_account_is_refused(self):
        """The guard on the inference. Filing this as a non-flow rests on
        'it only touches spot', which is read off a type name and a handful of
        records — not guaranteed by the venue. If that ever stops being true,
        silently skipping the record hides a real flow."""
        with pytest.raises(ValueError, match="contradicts that"):
            net_external_flow(
                self._spot_transfer(toPerp=True), self.SINCE, self.UNTIL, self.ME
            )

    def test_the_guard_covers_dex_fields_too(self):
        with pytest.raises(ValueError, match="sourceDex"):
            net_external_flow(
                self._spot_transfer(sourceDex="perp"), self.SINCE, self.UNTIL, self.ME
            )

    def test_a_liquidation_is_not_held_to_the_spot_only_guard(self):
        """`liquidation` is also a non-flow, for an entirely different reason:
        it is a perp event the model PREDICTS. Asserting it never names the
        perp account would refuse correct records — which is why the guard is
        scoped to SPOT_ONLY_NON_FLOW_TYPES rather than every non-flow."""
        from risk_engine.market.parse import (
            NON_FLOW_DELTA_TYPES,
            SPOT_ONLY_NON_FLOW_TYPES,
        )

        assert "liquidation" in NON_FLOW_DELTA_TYPES
        assert "liquidation" not in SPOT_ONLY_NON_FLOW_TYPES
        rows = [{"time": int(self.UNTIL.timestamp() * 1000) - 1,
                 "delta": {"type": "liquidation", "sourceDex": "perp"}}]
        assert net_external_flow(
            rows, self.SINCE, self.UNTIL, self.ME
        ) == pytest.approx(0.0)

    def test_a_spot_transfer_and_a_real_deposit_net_correctly(self):
        """The whole point: the deposit counts, the airdrop does not."""
        rows = [
            *self._spot_transfer(),
            {"time": 1778067666018,
             "delta": {"type": "deposit", "usdc": "5000.0"}},
        ]
        assert net_external_flow(
            rows, self.SINCE, self.UNTIL, self.ME
        ) == pytest.approx(+5000.0)
