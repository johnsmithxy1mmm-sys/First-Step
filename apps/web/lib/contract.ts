/**
 * The client half of the §6 degradation contract.
 *
 * The backend already decided freshness and whether execution is allowed;
 * this file only mirrors the shape so the compiler can enforce the same rule
 * on the render side: `value` exists only on the displayable variants, so a
 * component physically cannot read a number out of a hidden payload.
 *
 * None of these thresholds are re-implemented here. A rule the client
 * re-derives is a rule that drifts, and §6 says specifically not to soften
 * this one for conversion.
 */

export interface Estimate {
  point: number;
  ci_low: number;
  ci_high: number;
  model_version: string;
  computed_at: string;
}

export interface PortfolioRiskValue {
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
  cvar_95_24h_usd: Estimate;
  /**
   * 24 HOURS of funding, not the 7 days `p_liq_7d` above covers — and the two
   * sit in the same grid, so a reader will compare them. Funding accrues
   * hourly: a week is roughly 7× this. It is published at 24h only because
   * funding is simulated independently of price and that bias grows with the
   * horizon (OPEN-QUESTIONS A8), so a week-long figure is indicative rather
   * than calibrated. Any component rendering this owes the user the horizon
   * and that scaling in visible copy; the funding panel in `app/page.tsx` is
   * where they are said.
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

export interface PreTradeDeltaValue {
  order: { coin: string; size: number; leverage: number; mode: string; description: string };
  execution_price: number;
  computed_at: string;
  publishable: boolean;
  p_liq: DeltaBlock;
  cvar_95_usd: DeltaBlock;
  new_assets: string[];
  summary: string;
  within_budget: boolean;
  matrix_age_s: number | null;
}

export interface ExecutionGate {
  allowed: boolean;
  reasons: string[];
}

export type Guarded<T> =
  | {
      freshness: 'fresh' | 'stale';
      value: T;
      computedAt: string;
      ageMs: number;
      degraded: boolean;
      execution: ExecutionGate;
    }
  | {
      freshness: 'hidden' | 'unavailable';
      reason: string;
      computedAt: string | null;
      ageMs: number | null;
      degraded: true;
      execution: ExecutionGate;
    };

export function hasValue<T>(
  p: Guarded<T>,
): p is Extract<Guarded<T>, { freshness: 'fresh' | 'stale' }> {
  return p.freshness === 'fresh' || p.freshness === 'stale';
}

export function formatAge(ms: number | null): string {
  if (ms === null) return 'unknown';
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  return `${m}m ${s % 60}s ago`;
}

export const pct = (x: number) => `${(x * 100).toFixed(2)}%`;
/** A probability *difference*, in percentage points, signed. A change of
 *  +0.17 pp is a different quantity from a level of 0.17% and reads wrong
 *  when formatted as one (OPEN-QUESTIONS D6). */
export const pp = (x: number) => `${x >= 0 ? '+' : '−'}${Math.abs(x * 100).toFixed(2)} pp`;
export const usd = (x: number) =>
  x.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });


/**
 * How long a payload the client could not refresh may stay on screen.
 *
 * Matches the backend's `HIDE_AFTER_MS`, and it exists because the backend's
 * copy cannot help here. The backend judges freshness when it is ASKED; if
 * the browser cannot reach it — offline, DNS gone, a connection black-holed
 * so `fetch` neither resolves nor rejects — nothing re-evaluates anything and
 * the last payload stays on screen with the age badge it arrived with.
 *
 * §9's Phase 3 criterion is that values DISAPPEAR within five minutes of the
 * risk service stopping. That was tested by stopping the risk service, which
 * the backend survives and reports; it was never tested by cutting the
 * browser off from the backend, where the same five minutes was unbounded.
 */
export const CLIENT_HIDE_AFTER_MS = 300_000;

/**
 * A payload downgraded by how long the CLIENT has been unable to refresh it.
 *
 * Belt and braces with the fetch error path, not a duplicate of it: a fetch
 * that REJECTS can be caught and turned into `unavailable`, and a fetch that
 * HANGS cannot — it never settles, so no catch runs and no state updates.
 * Only a clock the client owns can expire that.
 */
export function expireLocally<T>(
  payload: Guarded<T> | null,
  receivedAt: number | null,
  now: number = Date.now(),
  hideAfterMs: number = CLIENT_HIDE_AFTER_MS,
): Guarded<T> | null {
  if (payload === null || receivedAt === null) return payload;
  const sinceRefresh = now - receivedAt;
  if (sinceRefresh < hideAfterMs) return payload;
  return {
    freshness: 'unavailable',
    reason: `this value could not be refreshed for ${Math.round(
      sinceRefresh / 1000,
    )}s, so it is withheld rather than shown at whatever age it last reported`,
    computedAt: payload.computedAt ?? null,
    ageMs: null,
    degraded: true,
    execution: {
      allowed: false,
      reasons: ['the browser has not been able to reach the backend'],
    },
  } as Guarded<T>;
}
