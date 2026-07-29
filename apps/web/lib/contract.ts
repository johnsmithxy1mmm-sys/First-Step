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
  factor_beta: number;
  factor_coin: string;
  p_liq_24h: Estimate;
  p_liq_24h_cross: Estimate;
  p_liq_24h_isolated: Record<string, Estimate>;
  p_liq_7d: Estimate;
  cvar_95_24h_usd: Estimate;
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
export const usd = (x: number) =>
  x.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });
