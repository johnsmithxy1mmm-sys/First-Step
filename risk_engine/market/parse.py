"""Parsers turning Info responses into domain objects (§5.1).

Shapes come from the documentation, not from observation -- see the warning
in `info.py`. Every parser is strict: an unexpected shape raises rather than
defaulting, because the failure mode of a lenient parser here is a book that
looks smaller or safer than it is.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from risk_engine.domain.types import AssetSpec, Book, MarginMode, MarginTier, Position


def _f(value, field: str) -> float:
    """Hyperliquid returns numbers as strings; refuse to guess about missing ones.

    Non-finite values are rejected here rather than downstream (audit A-07):
    `float("NaN")` parses happily, and NaN then evaluates False in every
    subsequent range check, so a single poisoned field would travel intact
    through position validation, the simulator and into the calibration
    journal. `float("inf")` is equally inadmissible as a price or a size.
    """
    if value is None:
        raise ValueError(f"missing required numeric field {field!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"field {field!r} is not finite: {value!r}")
    return out


def parse_meta(meta: dict) -> dict[str, AssetSpec]:
    """`meta` -> per-asset spec including the margin tier table (§1.3).

    Handles both shapes the documentation describes: a `marginTables` list
    keyed by `marginTableId`, and assets that carry only `maxLeverage`. In
    the latter case the table is the single implied tier -- not a default, a
    derivation from the asset's own published leverage.
    """
    universe = meta.get("universe")
    if not isinstance(universe, list) or not universe:
        raise ValueError("meta.universe missing or empty")

    tables: dict[int, list[MarginTier]] = {}
    for entry in meta.get("marginTables", []) or []:
        if not (isinstance(entry, list) and len(entry) == 2):
            raise ValueError(f"unexpected marginTables entry: {entry!r}")
        table_id, body = entry
        tiers = [
            MarginTier(_f(t.get("lowerBound"), "lowerBound"), _f(t.get("maxLeverage"), "maxLeverage"))
            for t in body.get("marginTiers", [])
        ]
        if tiers:
            tables[int(table_id)] = tiers

    specs: dict[str, AssetSpec] = {}
    for asset in universe:
        name = asset.get("name")
        if not name:
            raise ValueError(f"universe entry without a name: {asset!r}")
        if asset.get("isDelisted"):
            continue
        max_lev = _f(asset.get("maxLeverage"), f"{name}.maxLeverage")
        table_id = asset.get("marginTableId")
        tiers = tables.get(int(table_id)) if table_id is not None else None
        if not tiers:
            tiers = [MarginTier(0.0, max_lev)]
        specs[name] = AssetSpec(
            name=name,
            sz_decimals=int(asset.get("szDecimals", 0)),
            max_leverage=max_lev,
            tiers=tuple(tiers),
        )
    return specs


def parse_clearinghouse_state(
    state: dict, address: str, captured_at: datetime | None = None
) -> Book:
    """`clearinghouseState` -> Book.

    `cross_collateral` is derived as cross account value minus the unrealised
    PnL of the cross positions, because the simulator needs the cash and the
    PnL to move independently.
    """
    captured_at = captured_at or datetime.now(timezone.utc)
    positions: list[Position] = []
    cross_upnl = 0.0

    for entry in state.get("assetPositions", []) or []:
        p = entry.get("position") or {}
        size = _f(p.get("szi"), "szi")
        if size == 0:
            continue
        coin = p.get("coin")
        if not coin:
            raise ValueError(f"position without a coin: {p!r}")
        lev = p.get("leverage") or {}
        raw_mode = lev.get("type")
        if raw_mode not in ("cross", "isolated"):
            raise ValueError(f"{coin}: unknown margin mode {raw_mode!r}")
        mode = MarginMode(raw_mode)
        upnl = _f(p.get("unrealizedPnl"), f"{coin}.unrealizedPnl")
        iso_margin = None
        if mode is MarginMode.ISOLATED:
            # `rawUsd` is the collateral moved into the pocket; `marginUsed`
            # is the fallback the docs also expose.
            raw = lev.get("rawUsd", p.get("marginUsed"))
            iso_margin = _f(raw, f"{coin}.isolated margin")
        else:
            cross_upnl += upnl
        positions.append(
            Position(
                coin=coin,
                size=size,
                entry_price=_f(p.get("entryPx"), f"{coin}.entryPx"),
                mode=mode,
                leverage=_f(lev.get("value"), f"{coin}.leverage.value"),
                isolated_margin=iso_margin,
            )
        )

    # No fallback to `marginSummary` (audit A-08). That field summarises the
    # WHOLE account, isolated pockets included, so using it as the cross
    # account value double-counts every isolated margin: once inside
    # `cross_collateral` and again in each position's `isolated_margin`. The
    # resulting equity is too high, which understates risk -- the one
    # direction §10 prohibits. Every other field in this parser is strict;
    # this one has no business being lenient either.
    cross_summary = state.get("crossMarginSummary")
    if cross_summary is None:
        raise ValueError(
            "clearinghouseState has no crossMarginSummary; refusing to substitute "
            "marginSummary, which includes isolated margin and would double-count it"
        )
    cross_account_value = _f(cross_summary.get("accountValue"), "accountValue")
    return Book(
        address=address,
        cross_collateral=cross_account_value - cross_upnl,
        positions=tuple(positions),
        captured_at=captured_at,
    )


def parse_candles_to_log_returns(candles: list) -> tuple[np.ndarray, np.ndarray]:
    """`candleSnapshot` -> (close timestamps in ms, hourly log returns).

    Gaps are not interpolated. A missing hour means the return across it is
    a multi-hour return, which would understate the per-hour volatility if
    silently folded in, so gaps are reported by the timestamps and the caller
    decides.
    """
    if not candles:
        raise ValueError("empty candle snapshot")
    rows = sorted(candles, key=lambda c: int(c["t"]))
    closes = np.array([float(c["c"]) for c in rows], dtype=np.float64)
    times = np.array([int(c["t"]) for c in rows], dtype=np.int64)
    if (closes <= 0).any():
        raise ValueError("non-positive close price in candle snapshot")
    return times[1:], np.diff(np.log(closes))


def parse_funding_history(history: list) -> tuple[np.ndarray, np.ndarray]:
    """`fundingHistory` -> (timestamps in ms, hourly funding rates)."""
    if not history:
        raise ValueError("empty funding history")
    rows = sorted(history, key=lambda h: int(h["time"]))
    return (
        np.array([int(h["time"]) for h in rows], dtype=np.int64),
        np.array([float(h["fundingRate"]) for h in rows], dtype=np.float64),
    )


#: `userNonFundingLedgerUpdates` delta types that move USDC in or out of the
#: perp account, with the sign the account experiences: +1 means equity
#: arrived, -1 means it left. Anything not listed here is not a flow and is
#: ignored -- but `net_external_flow` refuses an UNKNOWN type rather than
#: ignoring it, because a type this table has never seen is exactly how a new
#: kind of transfer becomes a silent scoring error (OPEN-QUESTIONS B2).
EXTERNAL_FLOW_SIGNS: dict[str, float] = {
    "deposit": +1.0,
    "withdraw": -1.0,
    # Spot<->perp movements are external to the perp book the model predicts,
    # even though they never leave the venue. `usdClassTransfer` carries a
    # `toPerp` flag that decides the direction.
    "accountClassTransfer": 0.0,
    "internalTransfer": 0.0,
    "subAccountTransfer": 0.0,
    "vaultDeposit": -1.0,
    "vaultWithdraw": +1.0,
    "spotTransfer": 0.0,
}

#: Types that are NOT external flow: the model either predicts them or they do
#: not touch perp equity. Listed explicitly so an unknown type is genuinely
#: unknown rather than silently falling through this set.
NON_FLOW_DELTA_TYPES = frozenset({
    "liquidation",
    "rewardsClaim",
    "spotGenesis",
})


def net_external_flow(updates: list, since: datetime, until: datetime) -> float:
    """`userNonFundingLedgerUpdates` -> net USD into the perp account.

    Positive means equity arrived from outside; negative means it left. This
    is the quantity B2 needs subtracted before an equity change can be scored
    as model error, and getting it wrong in the quiet direction -- returning
    0.0 for something unrecognised -- turns a deposit into a spectacular
    apparent miss in the calibration record.

    So an unrecognised `type` RAISES. That is deliberate and it is the whole
    design: the venue can add a transfer type at any time, and the failure
    mode of ignoring one is invisible, permanent (journal rows are written
    once) and lands in the direction that makes the model look wrong. A
    resolver failure naming the record is recoverable; a silently dropped
    $50k is not.

    Signed from the ACCOUNT's perspective, and the sign convention is read
    from the record rather than assumed: a directional transfer carries a
    flag saying which way it went, and guessing it from the type name alone
    would be a coin flip on half the cases.
    """
    start_ms = int(since.timestamp() * 1000)
    end_ms = int(until.timestamp() * 1000)
    total = 0.0
    for row in updates or []:
        when = row.get("time")
        if when is None or not (start_ms <= int(when) <= end_ms):
            continue
        delta = row.get("delta") or {}
        kind = delta.get("type")
        if kind is None:
            raise ValueError(
                f"ledger update carries no delta.type, so it cannot be classified as "
                f"flow or non-flow: {row!r}"
            )
        if kind in NON_FLOW_DELTA_TYPES:
            continue
        if kind not in EXTERNAL_FLOW_SIGNS:
            raise ValueError(
                f"unknown ledger delta type {kind!r}. It is neither a known external "
                f"flow nor a known non-flow, and guessing would either score a real "
                f"transfer as model error or hide one. Add it to EXTERNAL_FLOW_SIGNS "
                f"or NON_FLOW_DELTA_TYPES in market/parse.py once its meaning is "
                f"confirmed (OPEN-QUESTIONS B2). Record: {row!r}"
            )
        amount = delta.get("usdc")
        if amount is None:
            raise ValueError(f"{kind} record carries no usdc amount: {row!r}")
        sign = EXTERNAL_FLOW_SIGNS[kind]
        if sign == 0.0:
            # A directional transfer. The venue states the direction; inferring
            # it from the type name would be a guess on exactly the records
            # where being wrong flips the sign of the correction.
            to_perp = delta.get("toPerp")
            if to_perp is None:
                raise ValueError(
                    f"{kind} is directional but carries no 'toPerp' flag, so which way "
                    f"the money went cannot be read: {row!r}"
                )
            sign = +1.0 if to_perp else -1.0
        total += sign * abs(float(amount))
    return total
