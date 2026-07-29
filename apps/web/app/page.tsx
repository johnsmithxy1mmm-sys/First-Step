'use client';

/**
 * One screen, four widgets (§8).
 *
 * Phase 3 is read-only. The execution button exists and is wired to the
 * server-side gate, but it is inert: `max_safe_size` and the builder fee are
 * Phase 4, and Phase 4 does not start until shadow validation clears (§3.3).
 * The button is present now because §6's rule — that it is blocked on stale
 * data — is part of *this* phase's acceptance criterion and has to be
 * demonstrable.
 */

import { useCallback, useEffect, useState } from 'react';

import { GuardedPanel, EstimateValue } from '@/components/Guarded';
import { AgentKeyDisclaimer } from '@/components/AgentKeyDisclaimer';
import { WalletBar } from '@/components/WalletBar';
import type { Guarded, PortfolioRiskValue, PreTradeDeltaValue } from '@/lib/contract';
import { hasValue, pct, pp, usd } from '@/lib/contract';

const DEMO_BOOK = {
  address: '0xdemo',
  cross_collateral: 100_000,
  positions: [
    { coin: 'BTC', size: 4, entry_price: 100_000, mode: 'cross', leverage: 20 },
    { coin: 'ETH', size: 60, entry_price: 4_000, mode: 'cross', leverage: 20 },
  ],
};

async function post<T>(path: string, body: unknown): Promise<Guarded<T>> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    return {
      freshness: 'unavailable',
      reason: `backend returned ${res.status}`,
      computedAt: null,
      ageMs: null,
      degraded: true,
      execution: { allowed: false, reasons: [`backend returned ${res.status}`] },
    };
  }
  return (await res.json()) as Guarded<T>;
}

export default function Page() {
  const [risk, setRisk] = useState<Guarded<PortfolioRiskValue> | null>(null);
  const [delta, setDelta] = useState<Guarded<PreTradeDeltaValue> | null>(null);
  const [health, setHealth] = useState<{ ok: boolean; syntheticData?: boolean } | null>(null);
  const [size, setSize] = useState('800');

  const refresh = useCallback(async () => {
    setRisk(
      await post<PortfolioRiskValue>('/api/portfolio_risk', { book: DEMO_BOOK, n_paths: 20000 }),
    );
    try {
      setHealth(await (await fetch('/api/health')).json());
    } catch {
      setHealth({ ok: false });
    }
  }, []);

  useEffect(() => {
    void refresh();
    // Re-poll well inside the 60-second staleness window, so a healthy
    // system never *looks* stale purely because of the polling cadence.
    const id = setInterval(() => void refresh(), 20_000);
    return () => clearInterval(id);
  }, [refresh]);

  const runDelta = useCallback(async () => {
    setDelta(null);
    setDelta(
      await post<PreTradeDeltaValue>('/api/pre_trade_delta', {
        book: DEMO_BOOK,
        order: { coin: 'SOL', size: Number(size), leverage: 20, mode: 'cross' },
        n_paths: 20000,
      }),
    );
  }, [size]);

  const gate = risk?.execution ?? { allowed: false, reasons: ['no data yet'] };

  return (
    <main className="mx-auto max-w-5xl px-6 py-10">
      <WalletBar health={health} />

      <div className="mt-6 grid gap-5 md:grid-cols-2">
        {/* 1 — portfolio risk (§4.1) */}
        <GuardedPanel
          title="Portfolio risk"
          subtitle="Probability of liquidation, 24h and 7d"
          payload={risk}
        >
          {(v) => (
            <div className="space-y-4">
              <div>
                <p className="text-xs text-neutral-500">P(liquidation) within 24h</p>
                <EstimateValue estimate={v.p_liq_24h} format={pct} emphasis />
              </div>
              <div>
                <p className="text-xs text-neutral-500">Within 7 days</p>
                <EstimateValue estimate={v.p_liq_7d} format={pct} />
              </div>
              {Object.entries(v.p_liq_24h_isolated).length > 0 ? (
                <div>
                  <p className="text-xs text-neutral-500">
                    Isolated positions are tracked separately — one pocket failing does not
                    touch the cross pool
                  </p>
                  {Object.entries(v.p_liq_24h_isolated).map(([coin, est]) => (
                    <div key={coin} className="mt-1 flex justify-between text-sm">
                      <span>{coin}</span>
                      <span className="tabular-nums">{pct(est.point)}</span>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
          )}
        </GuardedPanel>

        {/* 2 — effective leverage (§4.1) */}
        <GuardedPanel
          title="Effective leverage"
          subtitle="Book volatility relative to BTC"
          payload={risk}
        >
          {(v) => (
            <div className="space-y-4">
              <EstimateValue estimate={v.effective_leverage} format={(x) => `${x.toFixed(2)}×`} emphasis />
              <p className="text-xs leading-relaxed text-neutral-500">
                Your book&apos;s 24h volatility is {v.effective_leverage.point.toFixed(2)}× BTC&apos;s.
                This ratio has no direction: a market-neutral book with large idiosyncratic
                variance scores the same as an outright long.
              </p>
              <div className="rounded bg-neutral-50 px-3 py-2">
                <p className="text-xs text-neutral-500">
                  Beta to {v.factor_coin} — the number that carries direction
                </p>
                {v.direction_detectable ? (
                  <>
                    <EstimateValue estimate={v.factor_beta} format={(x) => x.toFixed(2)} />
                    <p className="mt-1 text-xs text-neutral-400">
                      Your book leans {v.factor_beta.point > 0 ? 'long' : 'short'} {v.factor_coin}.
                    </p>
                  </>
                ) : (
                  <>
                    <EstimateValue estimate={v.factor_beta} format={(x) => x.toFixed(2)} />
                    <p className="mt-1 text-xs text-neutral-400">
                      This interval spans zero, so the model cannot resolve which way your
                      book leans. Reading a direction off the point estimate would be
                      reading it off noise.
                    </p>
                  </>
                )}
              </div>
              <div className="border-t border-neutral-100 pt-3">
                <p className="text-xs text-neutral-500">CVaR 95% over 24h</p>
                <EstimateValue estimate={v.cvar_95_24h_usd} format={usd} />
                <p className="mt-1 text-xs text-neutral-400">
                  Average loss in the worst 5% of outcomes.
                </p>
              </div>
            </div>
          )}
        </GuardedPanel>

        {/* 3 — pre-trade delta (§4.2) */}
        <GuardedPanel
          title="Pre-trade check"
          subtitle="What a proposed order does to your risk"
          payload={delta}
        >
          {(v) => (
            <div className="space-y-3">
              <p className="text-sm font-medium">{v.summary}</p>
              {v.p_liq.distinguishable ? (
                <>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <p className="text-xs text-neutral-500">P(liq) before</p>
                      <p className="tabular-nums">{pct(v.p_liq.before.point)}</p>
                    </div>
                    <div>
                      <p className="text-xs text-neutral-500">after</p>
                      <p className="tabular-nums">{pct(v.p_liq.after.point)}</p>
                    </div>
                  </div>
                  <p className="text-xs text-neutral-500 tabular-nums">
                    Change {pp(v.p_liq.change.point)} (95%: {pp(v.p_liq.change.ci_low)} to{' '}
                    {pp(v.p_liq.change.ci_high)})
                  </p>
                  {v.p_liq.overlap_rule_would_mislead ? (
                    <p className="rounded border border-amber-200 bg-amber-50 px-3 py-2 text-xs leading-relaxed text-amber-900">
                      The before and after intervals overlap, so comparing them side by
                      side would have said &ldquo;no detectable change&rdquo; about this
                      order. That comparison is not the right one here: both books are
                      walked over identical price paths, so the interval on the{' '}
                      <em>difference</em> is several times tighter than either side.
                      Non-overlap implies a real change; overlap does not imply its
                      absence.
                    </p>
                  ) : null}
                </>
              ) : (
                <p className="rounded bg-neutral-50 px-3 py-2 text-xs leading-relaxed text-neutral-600">
                  No change this run can resolve: {pp(v.p_liq.change.point)} with an
                  interval of {pp(v.p_liq.change.ci_low)} to {pp(v.p_liq.change.ci_high)},
                  which spans zero. That is not the same as no change — it means this
                  order&apos;s effect is smaller than what the simulation can separate
                  from its own noise. Showing an arrow would be false precision.
                </p>
              )}
              {v.new_assets.length > 0 ? (
                <p className="text-xs text-neutral-400">
                  New to your book: {v.new_assets.join(', ')}
                </p>
              ) : null}
            </div>
          )}
        </GuardedPanel>

        {/* 4 — funding drag (§4.4) */}
        <GuardedPanel
          title="Funding cost"
          subtitle="Distribution over the next 24h, not a point"
          payload={risk}
        >
          {(v) => (
            <div className="space-y-2">
              <div className="text-3xl font-semibold tabular-nums">
                {usd(v.funding_cost_24h.median)}
              </div>
              <p className="text-xs text-neutral-500 tabular-nums">
                5th–95th percentile {usd(v.funding_cost_24h.p05)} – {usd(v.funding_cost_24h.p95)}
              </p>
              <p className="text-xs leading-relaxed text-neutral-400">
                Charged hourly and simulated on every step of every path, because over a
                week it materially erodes collateral.
              </p>
            </div>
          )}
        </GuardedPanel>
      </div>

      {/* Order form + the execution gate (§6) */}
      <section className="mt-5 rounded-lg border border-neutral-200 bg-white p-5 shadow-sm">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-neutral-500">
          Proposed order
        </h2>
        <div className="mt-3 flex flex-wrap items-end gap-3">
          <label className="text-sm">
            <span className="block text-xs text-neutral-500">Buy SOL, size</span>
            <input
              value={size}
              onChange={(e) => setSize(e.target.value)}
              className="mt-1 w-32 rounded border border-neutral-300 px-2 py-1 tabular-nums"
              inputMode="decimal"
            />
          </label>
          <button
            onClick={() => void runDelta()}
            className="rounded bg-neutral-900 px-4 py-2 text-sm font-medium text-white hover:bg-neutral-700"
          >
            Check risk impact
          </button>

          <div className="ml-auto text-right">
            <button
              disabled
              title={
                gate.allowed
                  ? 'Execution arrives in Phase 4, after shadow validation clears'
                  : gate.reasons.join('; ')
              }
              className="cursor-not-allowed rounded bg-neutral-200 px-4 py-2 text-sm font-medium text-neutral-500"
            >
              Execute at recommended size
            </button>
            <p className="mt-1 max-w-md text-xs text-neutral-500">
              {gate.allowed
                ? 'Data is fresh enough to execute on. The button is inert until Phase 4 — execution and the builder fee are gated on shadow validation.'
                : `Blocked: ${gate.reasons.join('; ')}`}
            </p>
          </div>
        </div>
      </section>

      <AgentKeyDisclaimer />

      <footer className="mt-8 border-t border-neutral-200 pt-4 text-xs leading-relaxed text-neutral-500">
        <p>
          These are probability estimates under stated assumptions, not predictions and not
          advice. The model does not forecast prices: it assumes zero drift and estimates the
          distribution of outcomes around that.
        </p>
        {risk && hasValue(risk) ? (
          <p className="mt-2 tabular-nums">
            Model {risk.value.model_version} · computed {risk.computedAt}
          </p>
        ) : null}
      </footer>
    </main>
  );
}
