/**
 * §9 Phase 3 acceptance, driven through the real HTTP surface.
 *
 * "When the risk service is forcibly stopped, values disappear within 5
 * minutes; between 60 and 300 seconds they are marked stale; verified by a
 * manual test with the service stopped."
 *
 * A fake risk client stands in for the engine so the clock can be moved and
 * the service can be "stopped" deterministically. The same scenario is then
 * run against real processes in `scripts/degradation-check.mjs`, because a
 * criterion that says "verified by stopping the service" deserves to be
 * verified by actually stopping a service.
 */

import { afterEach, describe, expect, it } from 'vitest';

import type { FastifyInstance } from 'fastify';
import { buildServer } from '../src/server.js';
import type { Health, PortfolioRisk, RiskClient } from '../src/risk/client.js';
import { RiskServiceError } from '../src/risk/client.js';

const T0 = new Date('2026-07-29T12:00:00.000Z');

function estimate(point: number, computedAt: Date) {
  return {
    point,
    ci_low: point * 0.9,
    ci_high: point * 1.1,
    model_version: '0.2.0-phase1',
    computed_at: computedAt.toISOString(),
  };
}

function payload(computedAt: Date, publishable = true): PortfolioRisk {
  return {
    address: '0xdemo',
    model_version: '0.2.0-phase1',
    computed_at: computedAt.toISOString(),
    publishable,
    start_equity: 100_000,
    effective_leverage: estimate(6.9, computedAt),
    factor_beta: 6.7,
    factor_coin: 'BTC',
    p_liq_24h: estimate(0.08, computedAt),
    p_liq_24h_cross: estimate(0.08, computedAt),
    p_liq_24h_isolated: {},
    p_liq_7d: estimate(0.19, computedAt),
    p_liq_7d_cross: estimate(0.19, computedAt),
    p_liq_7d_isolated: {},
    cvar_95_24h_usd: estimate(55_000, computedAt),
    funding_cost_24h: { median: 120, p05: -40, p95: 400 },
    matrix_age_s: 60,
  };
}

/** A risk engine that can be stopped, exactly like the real process. */
class FakeEngine {
  running = true;
  computedAt = T0;
  publishable = true;
  matrixAgeS: number | null = 60;

  stop() {
    this.running = false;
  }

  asClient(): RiskClient {
    const engine = this;
    return {
      async health(): Promise<Health> {
        if (!engine.running) throw new RiskServiceError('connect ECONNREFUSED');
        return {
          ready: true,
          model_version: '0.2.0-phase1',
          synthetic_data: true,
          matrix_age_s: engine.matrixAgeS,
          matrix_built_at: T0.toISOString(),
          last_error: null,
          assets: ['BTC', 'ETH', 'SOL'],
        };
      },
      async portfolioRisk(): Promise<PortfolioRisk> {
        if (!engine.running) throw new RiskServiceError('connect ECONNREFUSED');
        return { ...payload(engine.computedAt, engine.publishable), matrix_age_s: engine.matrixAgeS };
      },
      async preTradeDelta(): Promise<never> {
        throw new RiskServiceError('not used in this test');
      },
    } as unknown as RiskClient;
  }
}

const BOOK = { book: { address: '0xdemo', cross_collateral: 1, positions: [] } };

let app: FastifyInstance | undefined;
afterEach(async () => {
  await app?.close();
  app = undefined;
});

async function ask(engine: FakeEngine, nowMs: number) {
  app = buildServer({ risk: engine.asClient(), now: () => nowMs });
  const res = await app.inject({ method: 'POST', url: '/api/portfolio_risk', payload: BOOK });
  return res.json();
}

describe('§9 Phase 3: stopping the risk service', () => {
  it('serves fresh values while the engine is up and current', async () => {
    const engine = new FakeEngine();
    const body = await ask(engine, T0.getTime() + 5_000);
    expect(body.freshness).toBe('fresh');
    expect(body.value.p_liq_24h.point).toBe(0.08);
    expect(body.execution.allowed).toBe(true);
  });

  it('marks values stale between 60 and 300 seconds, with the age visible', async () => {
    const engine = new FakeEngine();
    for (const seconds of [60, 120, 299]) {
      const body = await ask(engine, T0.getTime() + seconds * 1000);
      expect(body.freshness, `at ${seconds}s`).toBe('stale');
      expect(body.ageMs).toBe(seconds * 1000);
      expect(body.value).toBeDefined();
      expect(body.execution.allowed, `at ${seconds}s`).toBe(false);
      await app?.close();
      app = undefined;
    }
  });

  it('hides values entirely from 300 seconds', async () => {
    const engine = new FakeEngine();
    const body = await ask(engine, T0.getTime() + 300_000);
    expect(body.freshness).toBe('hidden');
    expect(body.value).toBeUndefined();
    expect(body.execution.allowed).toBe(false);
  });

  it('reports unavailable — never a stale value — once the engine is stopped', async () => {
    const engine = new FakeEngine();
    engine.stop();
    const body = await ask(engine, T0.getTime() + 5_000);
    expect(body.freshness).toBe('unavailable');
    expect(body.value).toBeUndefined();
    expect(body.reason).toContain('ECONNREFUSED');
    expect(body.execution.allowed).toBe(false);
  });

  it('health reports the engine as unreachable rather than failing the request', async () => {
    const engine = new FakeEngine();
    engine.stop();
    app = buildServer({ risk: engine.asClient() });
    const res = await app.inject({ method: 'GET', url: '/api/health' });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.ok).toBe(false);
    expect(body.engineReachable).toBe(false);
  });

  it('withholds an under-resolved estimate exactly like an unavailable one (§2.5)', async () => {
    const engine = new FakeEngine();
    engine.publishable = false;
    const body = await ask(engine, T0.getTime() + 1_000);
    expect(body.freshness).toBe('unavailable');
    expect(body.value).toBeUndefined();
  });

  it('blocks execution on a cold matrix even when the risk number is fresh', async () => {
    const engine = new FakeEngine();
    engine.matrixAgeS = 1_200;
    const body = await ask(engine, T0.getTime() + 1_000);
    expect(body.freshness).toBe('fresh');
    expect(body.execution.allowed).toBe(false);
    expect(body.execution.reasons.join(' ')).toContain('correlation matrix');
  });

  it('never returns a value field together with a non-displayable freshness', async () => {
    // The invariant the whole contract exists to guarantee, checked across
    // every state rather than trusted from the type signature.
    const engine = new FakeEngine();
    for (const [seconds, stopped] of [
      [5, false],
      [120, false],
      [400, false],
      [5, true],
    ] as const) {
      engine.running = !stopped;
      const body = await ask(engine, T0.getTime() + seconds * 1000);
      const displayable = body.freshness === 'fresh' || body.freshness === 'stale';
      expect(Object.hasOwn(body, 'value'), JSON.stringify(body.freshness)).toBe(displayable);
      await app?.close();
      app = undefined;
    }
  });
});
