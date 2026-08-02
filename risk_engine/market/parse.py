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
    # A list or dict where a number belongs raises TypeError from `float`, and
    # TypeError is outside the contract every caller here documents and every
    # harness probe catches -- `net_external_flow` promises ValueError, and the
    # B2 checks catch only that. A wrong-typed field is a malformed record, not
    # a different class of failure, so it is reported as one.
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        # ValueError is re-raised too, purely to name the field: these messages
        # are what the B2 sweep prints when it reports an unreadable record,
        # and "could not convert string to float: 'garbage'" does not say which
        # field of which record went wrong.
        raise ValueError(
            f"field {field!r} is not a number: {value!r} ({type(value).__name__})"
        ) from exc
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
        if table_id is not None:
            # An asset that DECLARES a table and whose table we failed to read
            # is not the same thing as an asset that carries only maxLeverage,
            # and collapsing the two is a §10 violation rather than a
            # robustness nicety: the fallback single tier is the most
            # PERMISSIVE one (mmr = 0.5/maxLeverage at every size), so a large
            # book silently gets the small-size maintenance rate. On the real
            # BTC table that is 0.0125 instead of 0.05 at a $200M notional --
            # maintenance margin understated 4x, liquidation modelled further
            # away than it is, P(liq) understated. Silent, and in the one
            # direction §10 forbids. Refuse instead; a venue shape change here
            # must be seen, not absorbed.
            tiers = tables.get(int(table_id))
            if not tiers:
                raise ValueError(
                    f"{name} declares marginTableId {table_id!r} but no usable tier "
                    f"table was parsed for it (known ids: {sorted(tables)}). Falling "
                    "back to a single maxLeverage tier would understate maintenance "
                    "margin for large positions (§1.3, §10), so this refuses instead."
                )
        else:
            # The documented second shape: no table id at all, so the single
            # implied tier IS the asset's published leverage -- a derivation,
            # not a default.
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
    # `_f`, not a bare `float`: a `"NaN"` close parses happily and then evades
    # the `closes <= 0` guard below (every comparison against NaN is False),
    # so the poison reaches the EWMA, the copula fit and the §2.3 tail gate --
    # where `understates_lower_tail` is `empirical - model > margin` and
    # therefore ALSO False. A single bad candle would quietly disarm the one
    # check that refuses to serve an understated tail. This is the module's
    # stated contract (line 4) and `_f`'s own A-07 rationale; both parsers
    # below simply were not using it.
    closes = np.array(
        [_f(c.get("c"), f"candle[{i}].c") for i, c in enumerate(rows)], dtype=np.float64
    )
    times = np.array([int(c["t"]) for c in rows], dtype=np.int64)
    if (closes <= 0).any():
        raise ValueError("non-positive close price in candle snapshot")
    return times[1:], np.diff(np.log(closes))


def parse_funding_history(history: list) -> tuple[np.ndarray, np.ndarray]:
    """`fundingHistory` -> (timestamps in ms, hourly funding rates)."""
    if not history:
        raise ValueError("empty funding history")
    rows = sorted(history, key=lambda h: int(h["time"]))
    # `_f` for the same reason as the candle closes: a NaN rate defeats
    # `FundingBounds.validate_against_history` (`np.abs(r).max()` is NaN, and
    # `NaN > cap` is False), so a genuine cap breach in the same series stops
    # being reported -- the C1 check that exists to catch a stale funding bound
    # can no longer fail. `fit_ar1` would then silently drop the NaN rows too.
    return (
        np.array([int(h["time"]) for h in rows], dtype=np.int64),
        np.array(
            [_f(h.get("fundingRate"), f"fundingHistory[{i}].fundingRate")
             for i, h in enumerate(rows)],
            dtype=np.float64,
        ),
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
    "vaultDeposit": -1.0,
    "vaultWithdraw": +1.0,
    # The vault paying out to a depositor: USDC arriving in the perp account.
    # Carries `usdc`, like every other perp-denominated record. Observed
    # 2026-07-31: {"type": "vaultDistribution", "vault": "0x4342...",
    # "usdc": "88.985153"}.
    "vaultDistribution": +1.0,
}

#: Types that always move USDC between two PERP accounts, so the sign depends
#: only on which side this account was on. Distinct from `DEX_ROUTED_TYPES`:
#: there is no dex to read because both legs are perp by construction.
#:
#: Both were filed as directional-needs-`toPerp` and neither carries one --
#: they carry `user`/`destination`, exactly like `send`. Observed 2026-07-31:
#:
#:     {"type": "internalTransfer", "usdc": "40.0", "user": "0xe2b2...",
#:      "destination": "0xbab0...", "fee": "1.0"}
#:     {"type": "subAccountTransfer", "usdc": "2000.0", "user": "0x706b...",
#:      "destination": "0x3999..."}
#:
#: `usdc` rather than `token`/`amount` is the discriminator that puts these on
#: the perp side: every spot-denominated record in this venue's ledger names a
#: token and a quantity, while perp-denominated ones carry a USDC figure
#: directly.
PERP_ADDRESS_ROUTED_TYPES = frozenset({"internalTransfer", "subAccountTransfer"})

#: Types that are NOT external flow: the model either predicts them or they do
#: not touch perp equity. Listed explicitly so an unknown type is genuinely
#: unknown rather than silently falling through this set.
NON_FLOW_DELTA_TYPES = frozenset({
    "liquidation",
    "rewardsClaim",
    "spotGenesis",
    # A token moving between two SPOT balances. Observed 2026-07-31 on
    # mainnet, an airdrop landing in a real account:
    #
    #     {"type": "spotTransfer", "token": "UFART", "amount": "20.0",
    #      "usdcValue": "4.9884", "user": "0x2000...010d",
    #      "destination": "0xd475...", "fee": "0.0", ...}
    #
    # It was filed under EXTERNAL_FLOW_SIGNS as directional-needs-`toPerp`,
    # and the record carries no `toPerp` -- nor any `sourceDex`, nor any
    # other field naming the perp account -- so B2 refused it.
    #
    # It is a non-flow, and the reason is what the model predicts rather than
    # what the transfer is called. `Book.equity` is cross collateral plus the
    # isolated pockets: the PERP account. A spot balance is not in it. Twenty
    # UFART arriving in a spot wallet changes nothing the model forecasts, so
    # scoring $4.99 as an external flow would corrupt the correction exactly
    # as counting a spot-to-spot `send` would.
    #
    # `_assert_no_perp_leg` below is what keeps this from being the guess it
    # would otherwise be: if the venue ever emits a `spotTransfer` carrying a
    # `toPerp` or a perp dex, this refuses rather than skipping it.
    "spotTransfer",
    # The three below were all found by the 2026-07-31 frame sweep, and all
    # three are classified by the SAME structural rule rather than by their
    # names: a record naming a `token` and an `amount` is denominated in a
    # spot asset, while anything moving perp collateral carries `usdc`. None
    # of these three carries `usdc`; none of them can therefore be scored in
    # the currency `Book.equity` is measured in, and none of them moves it.
    #
    #   {"type": "gossipPriorityGasAuction", "token": "HYPE", "amount": "0.342"}
    #   {"type": "cStakingTransfer", "token": "HYPE", "amount": "25.0",
    #    "isDeposit": false}
    #   {"type": "borrowLend", "token": "USDC", "operation": "withdraw",
    #    "amount": "1000.0", "interestAmount": "0.934"}
    #
    # `borrowLend` is the one worth pausing on, because "USDC" in a record is
    # not the same thing as perp collateral. It names `token: "USDC"` with an
    # `amount`, which is the spot convention; a perp-side movement of the same
    # size would have said `usdc: "1000.0"`. So this is the spot borrow/lend
    # market, and the guard below is what makes that reading falsifiable
    # instead of merely plausible.
    "gossipPriorityGasAuction",
    "cStakingTransfer",
    "borrowLend",
})

#: The subset of `NON_FLOW_DELTA_TYPES` that is non-flow *because it never
#: touches the perp account*, as opposed to non-flow for a different reason.
#: The distinction decides which records may be checked for a perp leg.
#:
#: `liquidation` is the reason this is a subset and not the whole set: it is a
#: perp event, and it is excluded from external flow because the model
#: PREDICTS it -- it is the outcome being forecast, not money arriving from
#: outside. Asserting that a liquidation never names the perp account would
#: refuse correct records.
SPOT_ONLY_NON_FLOW_TYPES = frozenset({
    "spotTransfer", "spotGenesis",
    # Added with the types themselves 2026-07-31. Membership here is what
    # makes their classification falsifiable: each is filed as a non-flow
    # because it is denominated in a spot asset, and the guard refuses any of
    # them that turns up carrying perp collateral instead.
    "gossipPriorityGasAuction", "cStakingTransfer", "borrowLend",
    # Audit F-8. `rewardsClaim` sat in NON_FLOW_DELTA_TYPES from the start,
    # OUTSIDE this guarded subset, with not one live record behind it — so
    # its classification was unfalsifiable: a rewards programme paying into
    # perp USDC would have been skipped silently, a hidden inflow scored as
    # model error. The model does not predict rewards, so "the model
    # predicts it" (liquidation's exemption) does not apply; the only honest
    # basis for non-flow is spot denomination, and that basis is exactly
    # what this guard checks per record. In here, a perp-denominated
    # rewardsClaim refuses loudly instead.
    "rewardsClaim",
})

#: Fields whose presence on a supposedly spot-only record would mean this
#: build's reading of that type is wrong. Checked rather than assumed, because
#: "spot" is being inferred from a type name and a handful of live records.
PERP_LEG_FIELDS = ("toPerp", "sourceDex", "destinationDex")

#: Types whose flow-ness cannot be decided from the type name at all, because
#: the SAME type covers movements that touch the perp account and movements
#: that never go near it. They are routed by their `sourceDex`/`destinationDex`
#: fields instead.
#:
#: `send` is the live example, and it is why this category exists. Observed
#: 2026-07-31 on mainnet:
#:
#:     {"type": "send", "user": "0xd475...", "destination": "0xd048...",
#:      "sourceDex": "spot", "destinationDex": "spot", "token": "HYPE",
#:      "amount": "5.0", "usdcValue": "206.575", ...}
#:
#: That one is a HYPE transfer between two spot accounts — it does not touch
#: perp equity and must not be subtracted. But the same type with
#: `sourceDex: "perp"` is $206 leaving the perp account, which must be. Filing
#: `send` wholesale under either table would have been wrong half the time,
#: and the half it got wrong would be invisible.
DEX_ROUTED_TYPES = frozenset({"send"})

#: The `dex` values naming **the primary perpetuals account this model
#: predicts** -- the one `clearinghouseState` returns and `Book.equity`
#: measures.
#:
#: `""` is in this set, and it was in the OPPOSITE set until 2026-07-31. That
#: was a bug with teeth. Hyperliquid identifies the primary perp dex by the
#: empty string (`{"type": "meta", "dex": ""}` is how its universe is
#: requested); builder-deployed perp dexes carry a name. Observed live:
#:
#:     {"type": "send", "sourceDex": "", "destinationDex": "xyz",
#:      "token": "USDC", "amount": "25000.0", "usdcValue": "25000.0", ...}
#:
#: That is $25,000 leaving the primary perp account for a builder dex. Filed
#: as `"" == spot` it scored **zero**, and a $25k outflow scored as zero is a
#: $25k unexplained equity drop blamed on the model -- the §10-forbidden
#: direction, in the single largest record the sweep found.
#:
#: The record itself is what settles it: the same field carries the literal
#: `"spot"` in one row and `""` in another. A venue that spells spot as
#: `"spot"` when it means spot does not also spell it `""`.
PERP_DEX_VALUES = frozenset({"", "perp", "perps"})

#: Values known NOT to be the primary perp account. `spot` is the spot
#: balance, which `Book.equity` does not include.
NON_PERP_DEX_VALUES = frozenset({"spot"})

#: Dex names seen that are neither the primary perp nor a known non-perp
#: venue -- builder-deployed dexes (HIP-3), as far as this build can tell.
#: Module-level and mutable on purpose: it is a tally for `verify` to report,
#: not state anything depends on. Treating an unknown name as non-primary is
#: the safe reading for every case except a future ALIAS of the primary dex,
#: which would hide a real flow, so the names are surfaced rather than merely
#: handled.
UNRECOGNISED_DEX_NAMES: dict[str, int] = {}


def _dex_touches_perp(value: object, kind: str, row: object) -> bool:
    """Whether this leg names the PRIMARY perp account, the one modelled.

    "Primary" carries the weight. `Book.equity` is what `clearinghouseState`
    returns for the default dex; a builder-deployed dex is a separate margin
    space, so for this account's purposes it is as external as spot.
    """
    if value is None:
        raise ValueError(
            f"{kind} is routed by dex but the record does not say which: {row!r}"
        )
    text = str(value).strip().lower()
    if text in PERP_DEX_VALUES:
        return True
    if text in NON_PERP_DEX_VALUES:
        return False
    # A NAMED dex that is not the primary one. Hyperliquid lets builders deploy
    # their own perp dexes (HIP-3), each a separate margin space with its own
    # `clearinghouseState`. It is "perp" in the ordinary sense and it is NOT
    # the account this model predicts, so for the purpose of that account it is
    # as external as spot: money moving primary -> "xyz" has left the book
    # `Book.equity` measures, and must be scored as an outflow.
    #
    # This used to raise, which was the right call when the only unknown value
    # imaginable was a spelling of the primary dex. It is the wrong call now
    # that new dex names are a thing third parties create: raising means the
    # §3.3 window breaks on whichever address first touches a new venue, and
    # the failure is retried forever rather than reported.
    #
    # So it is accepted as non-primary and RECORDED, because the one way this
    # goes wrong is a future alias for the primary dex being read as a builder
    # one, which would hide a real flow. `verify` surfaces the tally; a name
    # that turns out to be the primary dex belongs in PERP_DEX_VALUES.
    UNRECOGNISED_DEX_NAMES[text] = UNRECOGNISED_DEX_NAMES.get(text, 0) + 1
    return False


def _assert_no_perp_leg(delta: dict, kind: str, row: object) -> None:
    """Refuse a supposedly spot-only record that names the perp account.

    `spotTransfer` is filed as a non-flow because a spot balance is not in
    `Book.equity`. That is an inference from a type name plus a handful of
    live records, not a guarantee the venue has made, and the cost of it
    being wrong is asymmetric: a perp-touching record silently skipped is a
    real flow scored as model error, with nothing anywhere saying so.

    So the inference is checked on every record rather than trusted once. If
    a `spotTransfer` ever arrives with a `toPerp` or a `sourceDex`, this
    stops the run and quotes it, which is how the reading gets corrected
    instead of quietly rotting.
    """
    present = [f for f in PERP_LEG_FIELDS if delta.get(f) is not None]
    # `usdc` is the structural discriminator these types are classified BY:
    # every spot-denominated record in this ledger names a `token` and an
    # `amount`, while anything moving perp collateral carries a USDC figure
    # directly. A spot-only type that grows a `usdc` field has stopped being
    # what this table says it is. Zero is not evidence of movement, so only a
    # non-zero value counts.
    try:
        if float(delta.get("usdc") or 0.0):
            present.append("usdc")
    except (TypeError, ValueError):
        present.append("usdc")
    if present:
        raise ValueError(
            f"{kind} is classified as a non-flow because it moves value outside "
            f"the perp account this model predicts (spot balances and staking are "
            f"not part of `Book.equity`). This record contradicts that: it carries "
            f"{present!r}. Either the venue changed the type or this build's "
            f"reading of it was always wrong -- move it to EXTERNAL_FLOW_SIGNS, "
            f"PERP_ADDRESS_ROUTED_TYPES or DEX_ROUTED_TYPES in market/parse.py "
            f"(OPEN-QUESTIONS B2). Record: {row!r}"
        )


def _delta_amount_usd(delta: dict, kind: str, row: object) -> float:
    """The USD value of a delta, whichever field this type carries it in.

    `deposit`/`withdraw` carry `usdc`; a `send` of a non-USDC token carries
    `amount` in that token plus `usdcValue` in dollars, and reading `usdc`
    there would have raised "no usdc amount" on a perfectly well-formed
    record. Only USD-denominated fields are accepted -- `amount` alone is a
    token quantity and treating 5.0 HYPE as $5 would be a silent 40x error.
    """
    # Order matters, and `netWithdrawnUsd` is last for a reason. A
    # `vaultWithdraw` carries neither `usdc` nor `usdcValue`:
    #
    #     {"type": "vaultWithdraw", "requestedUsd": "310.0",
    #      "commission": "15.336366", "closingCost": "0.0",
    #      "basis": "156.636333", "netWithdrawnUsd": "294.663634"}
    #
    # Two USD figures, and only one of them arrived. `requestedUsd` is what
    # was asked for; `netWithdrawnUsd` is what the account actually received
    # after commission. Reading the former would overstate this inflow by
    # $15.34 -- and would do it on every vault withdrawal in the window, in
    # the direction that credits the account with money it never got.
    # `requestedUsd` is deliberately absent from this list.
    for field in ("usdc", "usdcValue", "netWithdrawnUsd"):
        value = delta.get(field)
        if value is not None:
            # `_f`, not `float`: a NaN amount propagates to `net_external_flow`'s
            # running total, and NaN survives every downstream guard --
            # `verify`'s per-type reconstruction check (`> 1e-9` is False for
            # NaN) reports PASS, and the resolver writes `actual_equity_change
            # = NaN` into the write-once journal, where `pit()` returns a
            # fabricated 1.0 (searchsorted puts NaN past the end) and CRPS is
            # NaN forever. This function's own docstring says returning
            # something quiet is the one failure it exists to prevent.
            return abs(_f(value, f"{kind}.{field}"))
    raise ValueError(
        f"{kind} record carries no USD amount (looked for 'usdc', 'usdcValue' and "
        f"'netWithdrawnUsd'): {row!r}"
    )


def _transfer_fee_usd(delta: dict) -> float:
    """The fee the SENDER pays on top of the amount, in USD.

    `internalTransfer` carries `fee: "1.0"` beside `usdc: "40.0"`: the sender
    is debited 41 and the recipient credited 40. Dropping the fee leaves $1 of
    real, explained outflow looking like model error on every transfer in the
    window — small per record, and systematically one-signed, which is the
    kind of bias a calibration score is least able to absorb.

    Only USD-denominated fees are counted, and `feeToken` is what decides
    (audit F-4): `send` names the fee's denomination explicitly, and reading
    `fee: "5.0", feeToken: "HYPE"` as five dollars mistakes ~$200 of HYPE for
    $5 — the same category error as reading a token `amount` as dollars, one
    field over. A token-denominated fee is paid from a token balance, which
    is not in `Book.equity`, so its correct contribution to the perp flow is
    zero — the same treatment `nativeTokenFee` has always had.
    """
    fee_token = str(delta.get("feeToken") or "").strip().upper()
    if fee_token not in ("", "USDC"):
        return 0.0
    raw = delta.get("fee")
    if raw is None:
        return 0.0
    # A malformed fee used to become $0 here, silently. That is the exact bias
    # this docstring calls the kind "a calibration score is least able to
    # absorb" -- one-signed and invisible -- applied to precisely the inputs
    # every neighbouring parser refuses loudly. `_f` refuses instead, so a
    # venue shape change surfaces as a B2 finding rather than as drift.
    return abs(_f(raw, "transfer fee"))


def net_external_flow(
    updates: list, since: datetime, until: datetime, address: str | None = None
) -> float:
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

    Signed from the ACCOUNT's perspective, and every sign is read from the
    record rather than inferred from the type name. Three shapes, and the
    third is why `address` is a parameter:

      - fixed-sign types (`deposit`, `withdraw`, `vaultDeposit`, ...);
      - directional types carrying a `toPerp` flag;
      - dex-routed types (`send`), where the SAME type covers a transfer that
        touches the perp account and one that never goes near it. Direction
        needs to know whether this account was the sender or the recipient,
        which cannot be read off the record alone.

    `address` may be omitted only when no dex-routed record is present; a
    `send` without it raises rather than being guessed at.
    """
    start_ms = int(since.timestamp() * 1000)
    end_ms = int(until.timestamp() * 1000)
    me = address.strip().lower() if address else None
    total = 0.0
    for row in updates or []:
        when = row.get("time")
        if when is None:
            # Audit F-5: this used to `continue`, and the window filter sits
            # BEFORE classification — so a timeless record bypassed the
            # unknown-type refusal entirely. An unclassifiable $9,999 delta
            # with no `time` scored a silent zero, and the frame sweep marked
            # its type "readable" without ever parsing it. A record that
            # cannot be placed in any window cannot be excluded from this one.
            raise ValueError(
                f"ledger update carries no 'time', so it cannot be placed inside "
                f"or outside the window and cannot be skipped as out-of-range: {row!r}"
            )
        # `int(when)` on a list/dict raises TypeError, which escapes this
        # function's documented ValueError contract -- and the two B2 harness
        # probes catch only ValueError, so one malformed record crashed the
        # very pass whose job is to report every malformed record. Convert it
        # to the contract's error so the check reports FAIL instead of dying.
        try:
            when_ms = int(when)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"ledger update has an unreadable 'time' ({when!r}), so it cannot be "
                f"placed inside or outside the window: {row!r}"
            ) from exc
        if not (start_ms <= when_ms <= end_ms):
            continue
        delta = row.get("delta") or {}
        kind = delta.get("type")
        if kind is None:
            raise ValueError(
                f"ledger update carries no delta.type, so it cannot be classified as "
                f"flow or non-flow: {row!r}"
            )
        if kind in NON_FLOW_DELTA_TYPES:
            # Verified per record rather than taken on trust, but only for the
            # types whose non-flow status rests on never touching perp. A
            # liquidation is a perp event and is excluded for a different
            # reason (the model predicts it), so it is not checked here.
            if kind in SPOT_ONLY_NON_FLOW_TYPES:
                _assert_no_perp_leg(delta, kind, row)
            continue

        if kind in DEX_ROUTED_TYPES or kind in PERP_ADDRESS_ROUTED_TYPES:
            # Two shapes sharing one code path, because the only thing that
            # differs is whether a dex has to be read. `send` moves value
            # between arbitrary venues and states them; `internalTransfer` and
            # `subAccountTransfer` move perp USDC by construction, so both
            # legs are the primary perp account and there is nothing to route
            # by except which side this account was on.
            always_perp = kind in PERP_ADDRESS_ROUTED_TYPES
            if me is None:
                raise ValueError(
                    f"{kind} is routed by dex and its direction depends on whether this "
                    f"account sent or received, but no address was supplied to compare "
                    f"against: {row!r}"
                )
            sender = str(delta.get("user") or "").strip().lower()
            recipient = str(delta.get("destination") or "").strip().lower()
            if me not in (sender, recipient):
                raise ValueError(
                    f"{kind} names neither this account as sender nor as recipient, so "
                    f"it cannot be signed: queried {me!r}, record {row!r}"
                )
            # The two legs are evaluated INDEPENDENTLY rather than as
            # if/elif, because an account can be on both sides of the same
            # record: a self-transfer names this address as sender and as
            # recipient. Chained, the send leg would fire and the receive leg
            # would never be reached, so a perp->perp self-send -- which moves
            # no perp equity at all -- would score as a full outflow. Summing
            # both legs makes it cancel, which is what actually happened.
            #
            # Each leg reads only the dex on its own side, so a record where
            # this account is on one side only never forces the other side's
            # field to parse.
            left = me == sender and (
                always_perp or _dex_touches_perp(delta.get("sourceDex"), kind, row))
            arrived = me == recipient and (
                always_perp or _dex_touches_perp(delta.get("destinationDex"), kind, row))
            # Amount is read only if a leg fired: a spot-to-spot transfer moves
            # no perp equity, and refusing it for a missing USD field would
            # reject records this correction does not even use.
            if left:
                # The fee rides with the OUTBOUND leg only. The sender is
                # debited amount + fee; the recipient is credited the amount.
                total -= _delta_amount_usd(delta, kind, row) + _transfer_fee_usd(delta)
            if arrived:
                total += _delta_amount_usd(delta, kind, row)
            continue

        if kind not in EXTERNAL_FLOW_SIGNS:
            raise ValueError(
                f"unknown ledger delta type {kind!r}. It is neither a known external "
                f"flow nor a known non-flow, and guessing would either score a real "
                f"transfer as model error or hide one. Add it to EXTERNAL_FLOW_SIGNS "
                f"NON_FLOW_DELTA_TYPES, PERP_ADDRESS_ROUTED_TYPES or "
                f"DEX_ROUTED_TYPES in market/parse.py once its meaning is "
                f"confirmed. Read the fields, not the name: a record carrying "
                f"`usdc` moves perp collateral, while one carrying `token` and "
                f"`amount` is denominated in a spot asset and does not "
                f"(OPEN-QUESTIONS B2). Record: {row!r}"
            )
        amount = _delta_amount_usd(delta, kind, row)
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
            # Audit F-1: a strict bool, not truthiness. The direction is the
            # only thing computed here, and truthiness gets it silently wrong
            # on exactly the inputs a schema drift would produce: the STRING
            # "false" is truthy, so a venue that started emitting stringified
            # booleans would flip every perp->spot transfer into an inflow —
            # a 2x-the-amount error per record, with no exception anywhere.
            # The frame sweep cannot catch that class: it proves records read
            # without raising, and a sign flip does not raise.
            if not isinstance(to_perp, bool):
                raise ValueError(
                    f"{kind} carries toPerp={to_perp!r} ({type(to_perp).__name__}), "
                    f"not a boolean. Guessing direction from truthiness flips the "
                    f"sign on stringified booleans ('false' is truthy). If the venue "
                    f"changed its schema, confirm the new encoding before mapping "
                    f"it (OPEN-QUESTIONS B2). Record: {row!r}"
                )
            sign = +1.0 if to_perp else -1.0
        total += sign * amount
    return total
