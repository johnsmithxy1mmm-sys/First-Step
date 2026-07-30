/**
 * Client for the Python risk service (§8).
 *
 * Every failure mode here ends in the same place: no value reaches the user
 * rather than a stale one. The service being down, timing out, returning a
 * malformed body, or returning a number the engine itself marked
 * unpublishable are all treated identically by the layer above, because to
 * a trader they mean the same thing — nobody can currently tell you what
 * your risk is.
 */

import type { Guarded } from '../staleness/contract.js';
import { guard, unavailable } from '../staleness/contract.js';

export interface Estimate {
  point: number;
  ci_low: number;
  ci_high: number;
  model_version: string;
  computed_at: string;
}

export interface PortfolioRisk {
  address: string;
  model_version: string;
  computed_at: string;
  publishable: boolean;
  start_equity: number;
  effective_leverage: Estimate;
  /** Signed, and interval-bearing like every other number (OPEN-QUESTIONS D3). */
  factor_beta: Estimate;
  /** Read off beta's interval, not the sign of its point estimate. */
  direction_detectable: boolean;
  factor_coin: string;
  p_liq_24h: Estimate;
  p_liq_24h_cross: Estimate;
  p_liq_24h_isolated: Record<string, Estimate>;
  p_liq_7d: Estimate;
  p_liq_7d_cross: Estimate;
  p_liq_7d_isolated: Record<string, Estimate>;
  cvar_95_24h_usd: Estimate;
  /**
   * Funding over 24 HOURS, not over the 7 days `p_liq_7d` covers. It comes
   * from `portfolio_risk`'s `result_24h`, whose horizon lives in
   * `portfolio_risk.DAY_HOURS` — a constant separate from `funding_drag`'s,
   * guarded by its own test in `risk_engine/tests/test_service.py`.
   *
   * The engine draws funding independently of price and that bias compounds
   * hour by hour, so 24h is the horizon over which it stays below the
   * estimation error on the rate itself (OPEN-QUESTIONS A8) — the reason no
   * funding figure is published at a week. Funding accrues hourly, so a week
   * is roughly 7× this; anyone presenting it beside a 7-day number owes the
   * user that scaling and the caveat. `FundingDrag.caveats` states both, but
   * that type never crosses this wire, so the disclosure travels with this
   * field and is rendered in `apps/web/app/page.tsx`.
   */
  funding_cost_24h: { median: number; p05: number; p95: number };
  matrix_age_s: number | null;
}

export interface DeltaBlock {
  before: Estimate;
  after: Estimate;
  change: Estimate;
  distinguishable: boolean;
  marginal_intervals_overlap: boolean;
  overlap_rule_would_mislead: boolean;
  direction: number;
}

export interface PreTradeDelta {
  order: { coin: string; size: number; leverage: number; mode: string; description: string };
  execution_price: number;
  model_version: string;
  computed_at: string;
  publishable: boolean;
  p_liq: DeltaBlock;
  cvar_95_usd: DeltaBlock;
  new_assets: string[];
  summary: string;
  latency_ms: Record<string, number>;
  within_budget: boolean;
  matrix_age_s: number | null;
}

export interface Health {
  ready: boolean;
  model_version: string;
  synthetic_data: boolean;
  matrix_age_s: number | null;
  matrix_built_at: string | null;
  last_error: string | null;
  assets: string[];
}

export class RiskServiceError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = 'RiskServiceError';
  }
}

export interface RiskClientOptions {
  baseUrl?: string;
  timeoutMs?: number;
  /** Bearer token for the engine. Defaults to `RISK_SERVICE_TOKEN`. */
  token?: string;
}

export class RiskClient {
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly token: string;

  constructor(options: RiskClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? 'http://127.0.0.1:8787').replace(/\/$/, '');
    this.timeoutMs = options.timeoutMs ?? 10_000;
    // Empty means the engine is on loopback with no auth, which is the
    // development default and which the engine itself refuses to allow on any
    // other interface.
    this.token = options.token ?? process.env.RISK_SERVICE_TOKEN ?? '';
  }

  private async request<T>(path: string, body?: unknown): Promise<T> {
    // An explicit timeout, not the platform default: a hung risk service must
    // degrade to "unavailable" on a known clock rather than holding the
    // request open until something else gives up.
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const headers: Record<string, string> = {};
      if (this.token) headers.authorization = `Bearer ${this.token}`;
      const init: RequestInit =
        body === undefined
          ? { method: 'GET', headers, signal: controller.signal }
          : {
              method: 'POST',
              headers: { ...headers, 'content-type': 'application/json' },
              body: JSON.stringify(body),
              signal: controller.signal,
            };
      const response = await fetch(`${this.baseUrl}${path}`, init);
      if (!response.ok) {
        const text = await response.text().catch(() => '');
        throw new RiskServiceError(
          `risk service ${response.status} on ${path}: ${text.slice(0, 300)}`,
          response.status,
        );
      }
      return (await response.json()) as T;
    } catch (error) {
      if (error instanceof RiskServiceError) throw error;
      const reason = error instanceof Error ? error.message : String(error);
      throw new RiskServiceError(`risk service unreachable on ${path}: ${reason}`);
    } finally {
      clearTimeout(timer);
    }
  }

  health(): Promise<Health> {
    return this.request<Health>('/health');
  }

  portfolioRisk(body: unknown): Promise<PortfolioRisk> {
    return this.request<PortfolioRisk>('/portfolio_risk', body);
  }

  preTradeDelta(body: unknown): Promise<PreTradeDelta> {
    return this.request<PreTradeDelta>('/pre_trade_delta', body);
  }
}

/**
 * Run a risk call and wrap whatever comes back in the §6 contract.
 *
 * The engine's own `computed_at` is what the age is measured from, so a slow
 * hop cannot launder a stale number into a fresh one. An unpublishable
 * result (§2.5) is withheld exactly like an unavailable one — the engine has
 * said it does not stand behind the number, and passing it on with a caveat
 * would be worse than saying nothing.
 */
export async function guarded<T extends { computed_at: string; publishable: boolean }>(
  call: () => Promise<T>,
  now: () => number = Date.now,
): Promise<Guarded<T>> {
  let result: T;
  try {
    result = await call();
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    return unavailable(reason);
  }
  if (!result.publishable) {
    return unavailable(
      'the engine could not resolve this estimate to its required confidence interval (§2.5)',
    );
  }
  const computedAt = new Date(result.computed_at);
  if (Number.isNaN(computedAt.getTime())) {
    return unavailable('the risk service returned an unreadable timestamp');
  }
  return guard(result, computedAt, { now: now() });
}
