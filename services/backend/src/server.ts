/**
 * HTTP surface for the frontend.
 *
 * Nothing here computes risk; it orchestrates the Python engine and applies
 * the §6 contract. The one rule that governs every route: a response either
 * carries a value with its age, or carries no value at all.
 */

import Fastify from 'fastify';
import type { FastifyInstance } from 'fastify';

import type { Guarded } from './staleness/contract.js';
import {
  MATRIX_STALE_AFTER_MS,
  executionGate,
  hasValue,
  unavailable,
} from './staleness/contract.js';
import type { PortfolioRisk, PreTradeDelta, RiskClient } from './risk/client.js';
import { guarded } from './risk/client.js';

export interface ServerOptions {
  readonly risk: RiskClient;
  readonly logger?: boolean;
  readonly now?: () => number;
}

interface RiskQuery {
  /** An inline book (the demo path), or… */
  book?: unknown;
  /** …an account address for the engine to fetch the live book of (the
   * wallet path). The engine prefers `book` when both are present. */
  address?: string;
  n_paths?: number;
  seed?: number;
}

interface DeltaQuery extends RiskQuery {
  order: unknown;
}

function ageMsFromSeconds(seconds: number | null | undefined): number | null {
  return seconds === null || seconds === undefined ? null : seconds * 1000;
}

export function buildServer(options: ServerOptions): FastifyInstance {
  const app = Fastify({ logger: options.logger ?? false });
  const now = options.now ?? Date.now;
  const risk = options.risk;

  /**
   * §6: the health endpoint reports the age of the last successful matrix
   * rebuild. Reported on its own clock, not the 60-second book clock — the
   * matrix is rebuilt every five minutes by design (OPEN-QUESTIONS D4).
   */
  app.get('/api/health', async (_request, reply) => {
    try {
      const health = await risk.health();
      const matrixAgeMs = ageMsFromSeconds(health.matrix_age_s);
      const matrixStale =
        matrixAgeMs === null || matrixAgeMs >= MATRIX_STALE_AFTER_MS;
      return reply.send({
        ok: health.ready && !matrixStale,
        engineReachable: true,
        modelVersion: health.model_version,
        syntheticData: health.synthetic_data,
        matrixAgeMs,
        matrixStale,
        matrixStaleAfterMs: MATRIX_STALE_AFTER_MS,
        lastError: health.last_error,
        assets: health.assets,
      });
    } catch (error) {
      // The engine being unreachable is a health answer, not a 500: the
      // frontend needs to render "unavailable", and it can only do that if
      // this call succeeds.
      return reply.send({
        ok: false,
        engineReachable: false,
        reason: error instanceof Error ? error.message : String(error),
        matrixAgeMs: null,
        matrixStale: true,
      });
    }
  });

  app.post<{ Body: RiskQuery }>('/api/portfolio_risk', async (request, reply) => {
    const body = request.body;
    if (!body?.book && !body?.address) {
      return reply.status(400).send({ error: 'body.book or body.address is required' });
    }
    const result = await guarded<PortfolioRisk>(
      () => risk.portfolioRisk(body),
      now,
    );
    return reply.send(withGate(result, now));
  });

  app.post<{ Body: DeltaQuery }>('/api/pre_trade_delta', async (request, reply) => {
    const body = request.body;
    if ((!body?.book && !body?.address) || !body?.order) {
      return reply
        .status(400)
        .send({ error: 'body.order plus body.book or body.address is required' });
    }
    const result = await guarded<PreTradeDelta>(
      () => risk.preTradeDelta(body),
      now,
    );
    return reply.send(withGate(result, now));
  });

  return app;
}

/**
 * Attach the execution gate to a guarded payload.
 *
 * The gate is computed here, server-side, and shipped as a decision rather
 * than as inputs for the client to re-derive. §6 says the execution button is
 * blocked on stale data and that this must not be softened for conversion;
 * a rule the client re-implements is a rule that can drift.
 */
function withGate<T extends { matrix_age_s: number | null; publishable: boolean }>(
  payload: Guarded<T>,
  now: () => number,
): unknown {
  const matrixAgeMs = hasValue(payload)
    ? ageMsFromSeconds(payload.value.matrix_age_s)
    : null;
  const gate = executionGate({
    risk: payload,
    publishable: hasValue(payload) ? payload.value.publishable : false,
    matrixAgeMs,
  });
  void now;
  return { ...payload, execution: gate };
}

export { unavailable };
