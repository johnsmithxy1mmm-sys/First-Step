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

import numpy as np
import pytest

from risk_engine.domain.types import MarginMode
from risk_engine.market.info import InfoClient
from risk_engine.market.parse import (
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
            "marginUsed": "20000.0", "maxLeverage": 20,
            "leverage": {"type": "isolated", "value": 5, "rawUsd": "22000.0"},
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
        assert by_coin["SOL"].isolated_margin == pytest.approx(22_000.0)
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


class TestAgentAddressGuard:
    def test_agent_address_is_refused_before_the_request_is_made(self):
        """§5.1's named footgun: an agent address returns a well-formed empty
        state, which reads as 'no positions' -- the worst possible silent
        failure for a risk tool."""
        client = InfoClient()
        with pytest.raises(ValueError, match="real account address"):
            client.clearinghouse_state("0xagent", is_agent_address=True)
        assert client.budget.spent() == 0  # nothing was charged


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
