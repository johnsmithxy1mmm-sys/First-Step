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
    factor_beta: estimate(6.7, computedAt),
    direction_detectable: true,
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
  /**
   * Observation times of the data, separate from `computedAt`. Left undefined
   * by default so the pre-existing cases still exercise the fallback path an
   * engine build without these fields takes.
   */
  bookCapturedAt: Date | undefined = undefined;
  pricesAsOf: Date | undefined = undefined;

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
        const body: PortfolioRisk = {
          ...payload(engine.computedAt, engine.publishable),
          matrix_age_s: engine.matrixAgeS,
        };
        if (engine.bookCapturedAt) body.book_captured_at = engine.bookCapturedAt.toISOString();
        if (engine.pricesAsOf) body.prices_as_of = engine.pricesAsOf.toISOString();
        return body;
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

  it('judges freshness on the age of the DATA, not on when the maths ran', async () => {
    // The defect this pins: the engine stamps `computed_at` at
    // `datetime.now()` on every request, so measuring against it always gave
    // ~0ms and `fresh`. The stale and hidden tiers were unreachable for any
    // real risk number, and a crash-time answer computed from ten-minute-old
    // positions was served as confidently current with the execution gate
    // open. Every case below has a computed_at of *now*.
    const engine = new FakeEngine();
    const at = T0.getTime() + 3_600_000;

    for (const [bookAgeS, expected] of [
      [5, 'fresh'],
      [90, 'stale'],
      [400, 'hidden'],
    ] as const) {
      engine.computedAt = new Date(at); // the maths just ran
      engine.bookCapturedAt = new Date(at - bookAgeS * 1000);
      engine.pricesAsOf = new Date(at); // prices current, isolate the book
      const body = await ask(engine, at);
      expect(body.freshness, `book ${bookAgeS}s old`).toBe(expected);
      expect(Object.hasOwn(body, 'value')).toBe(expected !== 'hidden');
      if (expected !== 'fresh') expect(body.execution.allowed).toBe(false);
      await app?.close();
      app = undefined;
    }
  });

  it('withholds the value on a cold matrix, instead of only blocking execution', async () => {
    // Matrix age used to feed ONLY the execution gate, so an engine that was
    // up with a matrix stuck for twenty minutes rendered its numbers labelled
    // `fresh` while /api/health said ok:false and the banner claimed no
    // numbers were being shown. Both halves wrong, and the shown numbers were
    // the stale ones.
    const engine = new FakeEngine();
    const at = T0.getTime() + 3_600_000;
    engine.computedAt = new Date(at);
    engine.bookCapturedAt = new Date(at);
    engine.pricesAsOf = new Date(at - 1_200_000); // 20 minutes cold

    const body = await ask(engine, at);
    expect(body.freshness).toBe('hidden');
    expect(body.value).toBeUndefined();
    expect(body.reason).toContain('mark prices');
    expect(body.execution.allowed).toBe(false);
  });

  it('does not mark a routinely-rebuilt matrix stale (D4)', async () => {
    // The counterweight: §2.1 rebuilds every five minutes by design, so the
    // marks must NOT be judged on the 60s book clock or the product would be
    // permanently stale.
    const engine = new FakeEngine();
    const at = T0.getTime() + 3_600_000;
    engine.computedAt = new Date(at);
    engine.bookCapturedAt = new Date(at);
    engine.pricesAsOf = new Date(at - 290_000); // just under one rebuild cycle

    const body = await ask(engine, at);
    expect(body.freshness).toBe('fresh');
    expect(body.execution.allowed).toBe(true);
  });

  it('reports the age of whichever input is actually out of date', async () => {
    const engine = new FakeEngine();
    const at = T0.getTime() + 3_600_000;
    engine.computedAt = new Date(at);
    engine.bookCapturedAt = new Date(at - 120_000); // stale on its 60s clock
    engine.pricesAsOf = new Date(at - 300_000); // older, but fine on its own
    const body = await ask(engine, at);
    expect(body.freshness).toBe('stale');
    expect(body.ageMs).toBe(120_000);
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
