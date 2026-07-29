"""Parsers turning Info responses into domain objects (§5.1).

Shapes come from the documentation, not from observation -- see the warning
in `info.py`. Every parser is strict: an unexpected shape raises rather than
defaulting, because the failure mode of a lenient parser here is a book that
looks smaller or safer than it is.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from risk_engine.domain.types import AssetSpec, Book, MarginMode, MarginTier, Position


def _f(value, field: str) -> float:
    """Hyperliquid returns numbers as strings; refuse to guess about missing ones."""
    if value is None:
        raise ValueError(f"missing required numeric field {field!r}")
    return float(value)


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

    cross_summary = state.get("crossMarginSummary") or state.get("marginSummary")
    if cross_summary is None:
        raise ValueError("clearinghouseState has neither crossMarginSummary nor marginSummary")
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
