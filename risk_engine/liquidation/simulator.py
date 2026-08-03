"""Vectorised liquidation simulation (§1).

The book splits into ONE cross pool and N independent isolated positions,
and the simulation tracks N+1 separate liquidation conditions on every step
of every path (§1.1). A single aggregated condition over the whole book is a
different, wrong model: an isolated position blowing up must not touch the
cross pool, and cross equity draining must not close an isolated position.

Liquidation condition, per step, per path:

    cross pool:   cross_cash + sum(upnl_cross) < sum(mmr_j * |size_j| * P_j)
    isolated i:   iso_margin_i + upnl_i       < mmr_i * |size_i| * P_i

with `mmr_j` looked up from the asset's margin table at the position's
notional *on that step* (§1.3), and `cross_cash` / `iso_margin_i` reduced by
hourly funding on that step (§1.5).

Within-step monitoring
----------------------
Checking the condition only at step closes would miss every excursion that
breaches and recovers inside the hour, which understates P(liq) -- the
direction §10 forbids. Each step therefore also applies a Brownian-bridge
correction to the margin gap `g = equity - maintenance_margin`: given `g` at
both ends of the step and the local variance of `dg`, the probability that
the gap went negative somewhere inside the step is

    P(hit) = exp(-2 * g0 * g1 / var(dg))

which is the standard first-passage result for a Brownian bridge to an
absorbing barrier at zero. `var(dg)` comes from a delta approximation:
`dg ~ sum_j (dg/dP_j) dP_j` with `dg/dP_j = size_j - mmr_j * |size_j|`, and
`cov(dP_j, dP_k) = P_j P_k sigma_j sigma_k rho_jk`.

This is an approximation (the gap is not exactly Brownian inside the step,
and tier boundaries are not differentiable) but it is a far smaller error
than ignoring intra-step excursions entirely, and it removes the need for a
Broadie-Glasserman-Kou barrier shift when checking against the continuous-
monitoring closed form in §3.1.1.

Known simplification, documented per §1.6: a breached cross pool is closed
in full and its equity set to zero, rather than partially liquidated. That
overstates severity (conservative) and is roughly neutral for "did a
liquidation happen in 24 h". Its effect on the terminal-equity distribution
used by §3.3 is noted in docs/hl-risk/OPEN-QUESTIONS.md D5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from risk_engine.domain.types import AssetSpec, Book, MarginMode, Position


@dataclass(frozen=True, slots=True)
class BridgeContext:
    """Inputs for the within-step correction.

    `step_vol` is the per-step (hourly) log-return volatility per asset
    column; `corr` is the correlation matrix over the same columns. Both are
    already computed for path generation, so this costs nothing extra.
    """

    step_vol: np.ndarray  # (A,)
    corr: np.ndarray  # (A, A)


@dataclass(frozen=True, slots=True)
class SimulationOutcome:
    cross_liquidated: np.ndarray  # (P,) bool
    isolated_liquidated: np.ndarray  # (P, I) bool
    start_equity: float
    terminal_equity: np.ndarray  # (P,)
    funding_paid: np.ndarray  # (P,) total USD funding over the horizon
    isolated_coins: tuple[str, ...]
    bridge_applied: bool

    @property
    def any_liquidated(self) -> np.ndarray:
        if self.isolated_liquidated.size:
            return self.cross_liquidated | self.isolated_liquidated.any(axis=1)
        return self.cross_liquidated

    @property
    def equity_change(self) -> np.ndarray:
        return self.terminal_equity - self.start_equity

    @property
    def equity_return(self) -> np.ndarray:
        if self.start_equity <= 0:
            raise ValueError("start equity is non-positive; the account is already gone")
        return self.equity_change / self.start_equity


def _constant_mmr(position_specs: list[AssetSpec]) -> np.ndarray | None:
    """A (1, N) row of maintenance rates when every table has ONE tier, else None.

    `AssetSpec.__post_init__` validates that the first tier starts at notional
    0, so a one-row table answers `maintenance_margin_rate` with the same
    number at every notional: the per-path `searchsorted` in `_mmr` is a
    lookup with exactly one possible answer, repeated for 20 000 paths on
    every one of 24 steps.

    This skips a search whose result is known in advance. It does NOT freeze a
    tier: a multi-row table returns None here and still resolves per path per
    step (§1.3), which is the case that matters for a book large enough to
    cross a boundary.

    Read-only on purpose. The row is cached on the simulator and handed back
    from every `_mmr` call, where the general path returns a fresh array each
    time; an in-place write by some future caller would otherwise corrupt the
    rate for every remaining step AND every later run on the same simulator.
    Downward is the direction §10 forbids, so it fails loudly instead.
    """
    if not position_specs or any(len(s.tiers) != 1 for s in position_specs):
        return None
    row = np.array(
        [float(s.maintenance_margin_rate(0.0)) for s in position_specs],
        dtype=np.float64,
    )[None, :]
    row.flags.writeable = False
    return row


class LiquidationSimulator:
    """Walks price and funding paths through the §1.1 conditions."""

    def __init__(
        self,
        book: Book,
        specs: dict[str, AssetSpec],
        asset_columns: dict[str, int],
    ) -> None:
        missing = [p.coin for p in book.positions if p.coin not in asset_columns]
        if missing:
            raise KeyError(f"no price column for {missing}")
        missing_specs = [p.coin for p in book.positions if p.coin not in specs]
        if missing_specs:
            raise KeyError(f"no margin table for {missing_specs} -- fetch `meta` first")

        self.book = book
        self.specs = specs
        self.asset_columns = asset_columns

        cross = book.cross_positions
        iso = book.isolated_positions
        self._cross = cross
        self._iso = iso
        self._cross_cols = np.array([asset_columns[p.coin] for p in cross], dtype=np.intp)
        self._iso_cols = np.array([asset_columns[p.coin] for p in iso], dtype=np.intp)
        self._cross_size = np.array([p.size for p in cross], dtype=np.float64)
        self._iso_size = np.array([p.size for p in iso], dtype=np.float64)
        self._cross_entry = np.array([p.entry_price for p in cross], dtype=np.float64)
        self._iso_entry = np.array([p.entry_price for p in iso], dtype=np.float64)
        self._iso_margin0 = np.array(
            [p.isolated_margin for p in iso], dtype=np.float64
        )
        self._cross_specs = [specs[p.coin] for p in cross]
        self._iso_specs = [specs[p.coin] for p in iso]
        self._cross_const_mmr = _constant_mmr(self._cross_specs)
        self._iso_const_mmr = _constant_mmr(self._iso_specs)

    # ---- helpers -------------------------------------------------------

    def _mmr(self, positions_specs: list[AssetSpec], notional_abs: np.ndarray,
             constant: np.ndarray | None) -> np.ndarray:
        """(P, N) maintenance margin rates, tier looked up per path per position.

        `constant` short-circuits the lookup when every table has one tier --
        see `_constant_mmr`. It is broadcast rather than materialised: the
        callers only read it, and `_constant_mmr` marks it read-only so a
        future in-place edit fails loudly instead of corrupting the rate for
        every subsequent step and run (a downward corruption would understate
        risk, which §10 forbids).
        """
        if constant is not None:
            return constant
        out = np.empty_like(notional_abs)
        for j, spec in enumerate(positions_specs):
            out[:, j] = spec.maintenance_margin_rate(notional_abs[:, j])
        return out

    @staticmethod
    def _bridge_hit_prob(g0: np.ndarray, g1: np.ndarray, var: np.ndarray) -> np.ndarray:
        """P(gap touched zero inside the step | endpoints g0, g1 both > 0).

        The formula is only defined for positive endpoints. It is evaluated
        for every path and masked afterwards, so gaps are floored at zero
        first: a path whose endpoint is already negative has been caught by
        the direct check, and letting its negative product through would
        overflow the exponential.
        """
        g0 = np.maximum(g0, 0.0)
        g1 = np.maximum(g1, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.exp(-2.0 * g0 * g1 / var)
        return np.where(var > 0, np.clip(p, 0.0, 1.0), 0.0)

    def _cross_gap_var(
        self, prices: np.ndarray, mmr: np.ndarray, bridge: BridgeContext
    ) -> np.ndarray:
        """var(dg) for the cross pool, (P,)."""
        # d_j = size_j - mmr_j * |size_j|; v_j = d_j * P_j * sigma_j
        d = self._cross_size[None, :] - mmr * np.abs(self._cross_size)[None, :]
        v = d * prices * bridge.step_vol[self._cross_cols][None, :]
        sub = bridge.corr[np.ix_(self._cross_cols, self._cross_cols)]
        return np.einsum("pa,ab,pb->p", v, sub, v, optimize=True)

    def _iso_gap_var(
        self, prices: np.ndarray, mmr: np.ndarray, bridge: BridgeContext
    ) -> np.ndarray:
        """var(dg) for each isolated position, (P, I). Single-asset, so no cross terms."""
        d = self._iso_size[None, :] - mmr * np.abs(self._iso_size)[None, :]
        v = d * prices * bridge.step_vol[self._iso_cols][None, :]
        return v * v

    # ---- main loop -----------------------------------------------------

    def run(
        self,
        price_paths: np.ndarray,
        funding_paths: np.ndarray | None = None,
        bridge: BridgeContext | None = None,
        bridge_uniforms: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> SimulationOutcome:
        """Run every path to the final step. See `run_checkpointed`."""
        n_steps = price_paths.shape[1] - 1
        return self.run_checkpointed(
            price_paths, (n_steps,), funding_paths, bridge, bridge_uniforms
        )[n_steps]

    def run_checkpointed(
        self,
        price_paths: np.ndarray,
        checkpoint_steps: tuple[int, ...],
        funding_paths: np.ndarray | None = None,
        bridge: BridgeContext | None = None,
        bridge_uniforms: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict[int, SimulationOutcome]:
        """Run every path once, snapshotting the outcome at each checkpoint.

        price_paths     (P, S+1, A) -- column 0 is the current price
        checkpoint_steps steps (1..S) at which to record an outcome; a walk
                        with checkpoints (24, 168) returns the 24h and 7d
                        answers from the SAME paths. Because a path's alive
                        set only ever shrinks, the liquidation flags at a
                        later checkpoint are a superset of an earlier one --
                        which is what makes P(liq, 7d) >= P(liq, 24h) hold
                        pathwise instead of merely in expectation
                        (audit A-10).
        funding_paths   (P, S, A)   -- hourly funding rate applied on each step
        bridge          within-step correction inputs; None disables it, which
                        is only correct for benchmarks that deliberately want
                        discrete monitoring (it understates P(liq)).
        bridge_uniforms ((P, S), (P, S, I)) uniforms driving the correction.
                        Passed in rather than drawn here so that two runs
                        sharing `BaseRandomness` are bit-identical, which is
                        what §3.1.5's strict monotonicity requires.
        """
        if price_paths.ndim != 3:
            raise ValueError(f"price_paths must be (paths, steps+1, assets), got {price_paths.shape}")
        n_paths, n_nodes, _ = price_paths.shape
        n_steps = n_nodes - 1
        if n_steps < 1:
            raise ValueError("need at least one step")
        if funding_paths is not None and funding_paths.shape[1] != n_steps:
            raise ValueError(
                f"funding_paths has {funding_paths.shape[1]} steps, prices have {n_steps}"
            )
        if bridge is not None and bridge_uniforms is None:
            raise ValueError("the bridge correction consumes uniforms; pass bridge_uniforms")
        if bridge_uniforms is not None:
            u_cross, u_iso = bridge_uniforms
            want_cross = (n_paths, n_steps)
            if u_cross.shape != want_cross:
                raise ValueError(f"cross bridge uniforms {u_cross.shape} != {want_cross}")
            want_iso = (n_paths, n_steps, len(self._iso))
            if u_iso.shape != want_iso:
                raise ValueError(f"isolated bridge uniforms {u_iso.shape} != {want_iso}")
        if not checkpoint_steps:
            raise ValueError("need at least one checkpoint step")
        cp_set = set(int(s) for s in checkpoint_steps)
        if any(s < 1 or s > n_steps for s in cp_set):
            raise ValueError(f"checkpoints {sorted(cp_set)} outside 1..{n_steps}")

        nc, ni = len(self._cross), len(self._iso)

        cross_cash = np.full(n_paths, self.book.cross_collateral, dtype=np.float64)
        iso_margin = np.tile(self._iso_margin0, (n_paths, 1)) if ni else np.zeros((n_paths, 0))
        cross_alive = np.ones(n_paths, dtype=bool)
        iso_alive = np.ones((n_paths, ni), dtype=bool)
        funding_paid = np.zeros(n_paths, dtype=np.float64)

        start_prices = {coin: float(price_paths[0, 0, col]) for coin, col in self.asset_columns.items()}
        start_equity = self.book.equity(start_prices)

        cross_gap_prev, iso_gap_prev = self._initial_gaps(price_paths[:, 0, :], nc, ni)
        # A book that is already below maintenance margin is liquidated at
        # t=0; it is not a prediction, it is a fact about the snapshot.
        cross_alive &= cross_gap_prev > 0
        iso_alive &= iso_gap_prev > 0

        outcomes: dict[int, SimulationOutcome] = {}
        for s in range(1, n_steps + 1):
            px = price_paths[:, s, :]

            if nc:
                cross_px = px[:, self._cross_cols]
                if funding_paths is not None:
                    rate = funding_paths[:, s - 1, :][:, self._cross_cols]
                    # A long pays when the rate is positive; signed notional
                    # gets the sign right for both sides in one expression.
                    pay = (rate * (self._cross_size[None, :] * cross_px)).sum(axis=1)
                    pay = np.where(cross_alive, pay, 0.0)
                    cross_cash -= pay
                    funding_paid += pay
                upnl = self._cross_size[None, :] * (cross_px - self._cross_entry[None, :])
                equity = cross_cash + upnl.sum(axis=1)
                notional = np.abs(self._cross_size)[None, :] * cross_px
                mmr = self._mmr(self._cross_specs, notional, self._cross_const_mmr)
                gap = equity - (mmr * notional).sum(axis=1)

                dead = cross_alive & (gap <= 0)
                if bridge is not None:
                    survivors = cross_alive & ~dead
                    if survivors.any():
                        var = self._cross_gap_var(cross_px, mmr, bridge)
                        phit = self._bridge_hit_prob(cross_gap_prev, gap, var)
                        dead |= survivors & (bridge_uniforms[0][:, s - 1] < phit)
                cross_alive &= ~dead
                cross_gap_prev = gap

            if ni:
                iso_px = px[:, self._iso_cols]
                if funding_paths is not None:
                    rate = funding_paths[:, s - 1, :][:, self._iso_cols]
                    pay = rate * (self._iso_size[None, :] * iso_px)
                    pay = np.where(iso_alive, pay, 0.0)
                    iso_margin -= pay
                    funding_paid += pay.sum(axis=1)
                upnl = self._iso_size[None, :] * (iso_px - self._iso_entry[None, :])
                equity = iso_margin + upnl
                notional = np.abs(self._iso_size)[None, :] * iso_px
                mmr = self._mmr(self._iso_specs, notional, self._iso_const_mmr)
                gap = equity - mmr * notional

                dead = iso_alive & (gap <= 0)
                if bridge is not None:
                    survivors = iso_alive & ~dead
                    if survivors.any():
                        var = self._iso_gap_var(iso_px, mmr, bridge)
                        phit = self._bridge_hit_prob(iso_gap_prev, gap, var)
                        dead |= survivors & (bridge_uniforms[1][:, s - 1, :] < phit)
                iso_alive &= ~dead
                iso_gap_prev = gap

            if s in cp_set:
                # `~alive` and `_terminal_equity` allocate fresh arrays;
                # `funding_paid` keeps accumulating, so it is copied.
                outcomes[s] = SimulationOutcome(
                    cross_liquidated=~cross_alive,
                    isolated_liquidated=~iso_alive,
                    start_equity=start_equity,
                    terminal_equity=self._terminal_equity(
                        price_paths[:, s, :], cross_cash, cross_alive, iso_margin, iso_alive
                    ),
                    funding_paid=funding_paid.copy(),
                    isolated_coins=tuple(p.coin for p in self._iso),
                    bridge_applied=bridge is not None,
                )

        return outcomes

    # ---- pieces of the loop, kept separate so they can be tested ---------

    def _initial_gaps(
        self, px0: np.ndarray, nc: int, ni: int
    ) -> tuple[np.ndarray, np.ndarray]:
        n_paths = px0.shape[0]
        if nc:
            cross_px = px0[:, self._cross_cols]
            upnl = self._cross_size[None, :] * (cross_px - self._cross_entry[None, :])
            notional = np.abs(self._cross_size)[None, :] * cross_px
            mmr = self._mmr(self._cross_specs, notional, self._cross_const_mmr)
            cross_gap = (
                self.book.cross_collateral + upnl.sum(axis=1) - (mmr * notional).sum(axis=1)
            )
        else:
            # No cross positions: the pool holds cash and cannot breach,
            # because its maintenance requirement is identically zero.
            cross_gap = np.full(n_paths, np.inf)
        if ni:
            iso_px = px0[:, self._iso_cols]
            upnl = self._iso_size[None, :] * (iso_px - self._iso_entry[None, :])
            notional = np.abs(self._iso_size)[None, :] * iso_px
            mmr = self._mmr(self._iso_specs, notional, self._iso_const_mmr)
            iso_gap = self._iso_margin0[None, :] + upnl - mmr * notional
        else:
            iso_gap = np.zeros((n_paths, 0))
        return cross_gap, iso_gap

    def _terminal_equity(
        self,
        px: np.ndarray,
        cross_cash: np.ndarray,
        cross_alive: np.ndarray,
        iso_margin: np.ndarray,
        iso_alive: np.ndarray,
    ) -> np.ndarray:
        if self._cross:
            cross_px = px[:, self._cross_cols]
            upnl = self._cross_size[None, :] * (cross_px - self._cross_entry[None, :])
            cross_equity = cross_cash + upnl.sum(axis=1)
        else:
            cross_equity = cross_cash
        # §1.6: a liquidated pool is written to zero. Real liquidations leave
        # a small residual; zero is the conservative side of that.
        total = np.where(cross_alive, cross_equity, 0.0)
        if self._iso:
            iso_px = px[:, self._iso_cols]
            upnl = self._iso_size[None, :] * (iso_px - self._iso_entry[None, :])
            iso_equity = np.where(iso_alive, iso_margin + upnl, 0.0)
            total = total + iso_equity.sum(axis=1)
        return total


def constant_price_paths(
    prices: dict[str, float], asset_columns: dict[str, int], n_paths: int, n_steps: int
) -> np.ndarray:
    """Flat paths -- used by tests that want the funding leg in isolation."""
    a = len(asset_columns)
    out = np.empty((n_paths, n_steps + 1, a), dtype=np.float64)
    for coin, col in asset_columns.items():
        out[:, :, col] = prices[coin]
    return out


def single_asset_book(
    coin: str,
    size: float,
    entry: float,
    collateral: float,
    mode: MarginMode = MarginMode.CROSS,
    leverage: float = 1.0,
    address: str = "0xtest",
    captured_at=None,
) -> Book:
    """Convenience constructor used throughout the benchmark suite."""
    from datetime import datetime, timezone

    iso = collateral if mode is MarginMode.ISOLATED else None
    cross_cash = 0.0 if mode is MarginMode.ISOLATED else collateral
    pos = Position(coin, size, entry, mode, leverage, iso)
    return Book(
        address=address,
        cross_collateral=cross_cash,
        positions=(pos,),
        captured_at=captured_at or datetime.now(timezone.utc),
    )
