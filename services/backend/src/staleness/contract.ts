/**
 * The §6 degradation contract, in the type system.
 *
 * §6's requirement is not "show an age somewhere" — it is that displaying a
 * value without its age must be *impossible*, and that beyond five minutes
 * the values are HIDDEN rather than greyed out. A greyed-out number is still
 * a number on the screen, and a trader under stress reads it.
 *
 * So the wire type is a discriminated union. There is no shape that carries a
 * value without also carrying its age and freshness, and the `hidden` and
 * `unavailable` variants carry no value field at all — a frontend cannot
 * render a stale number even by accident, because there is nothing to render.
 * That is the difference between a convention and a contract.
 *
 * The worst failure this product has is showing a confident, stale risk
 * number during a crash. Everything here exists for that one case.
 */

/** §6: data older than this is displayed, but marked stale with its age. */
export const STALE_AFTER_MS = 60_000;
/** §6: beyond this the values are hidden outright. */
export const HIDE_AFTER_MS = 300_000;

/**
 * §2.1 rebuilds the global correlation matrix every five minutes, so it is
 * routinely older than STALE_AFTER_MS and thresholding it on that clock
 * would mark the product permanently stale (OPEN-QUESTIONS D4). It gets its
 * own, longer allowance: two missed rebuild cycles.
 */
export const MATRIX_STALE_AFTER_MS = 660_000;

export type Freshness = 'fresh' | 'stale' | 'hidden' | 'unavailable';

/** A value the UI may display, together with how old it is. */
export interface FreshPayload<T> {
  readonly freshness: 'fresh' | 'stale';
  readonly value: T;
  readonly computedAt: string;
  readonly ageMs: number;
  /** True only for 'stale'; the UI must show the age prominently. */
  readonly degraded: boolean;
}

/**
 * No value. Deliberately has no `value` field, so a consumer cannot read one
 * — the compiler stops it, and at runtime there is nothing there.
 */
export interface WithheldPayload {
  readonly freshness: 'hidden' | 'unavailable';
  readonly reason: string;
  readonly computedAt: string | null;
  readonly ageMs: number | null;
  readonly degraded: true;
}

export type Guarded<T> = FreshPayload<T> | WithheldPayload;

export function hasValue<T>(p: Guarded<T>): p is FreshPayload<T> {
  return p.freshness === 'fresh' || p.freshness === 'stale';
}

export interface GuardOptions {
  readonly staleAfterMs?: number;
  readonly hideAfterMs?: number;
  readonly now?: number;
}

/**
 * Wrap a computed value in the contract.
 *
 * `computedAt` is the moment the engine produced the number, not the moment
 * the backend received it: a slow hop must not be able to launder a stale
 * number into a fresh one.
 */
export function guard<T>(
  value: T,
  computedAt: Date,
  options: GuardOptions = {},
): Guarded<T> {
  const now = options.now ?? Date.now();
  const staleAfter = options.staleAfterMs ?? STALE_AFTER_MS;
  const hideAfter = options.hideAfterMs ?? HIDE_AFTER_MS;
  const ageMs = now - computedAt.getTime();
  const iso = computedAt.toISOString();

  if (ageMs >= hideAfter) {
    return {
      freshness: 'hidden',
      reason: `risk data is ${Math.round(ageMs / 1000)}s old; beyond the ${Math.round(
        hideAfter / 1000,
      )}s limit it is withheld rather than shown`,
      computedAt: iso,
      ageMs,
      degraded: true,
    };
  }
  if (ageMs >= staleAfter) {
    return { freshness: 'stale', value, computedAt: iso, ageMs, degraded: true };
  }
  return { freshness: 'fresh', value, computedAt: iso, ageMs, degraded: false };
}

export function unavailable(reason: string): WithheldPayload {
  return {
    freshness: 'unavailable',
    reason,
    computedAt: null,
    ageMs: null,
    degraded: true,
  };
}

/**
 * §6: the execution button is blocked on anything but fresh data.
 *
 * "The product has no right to earn a fee on an execution it cannot price
 * the risk of. Do not soften this for conversion." Stale counts as blocked,
 * not merely warned — a 90-second-old liquidation probability during a crash
 * is precisely the number that should not be traded on.
 *
 * `publishable` folds in §2.5: an under-resolved estimate is treated exactly
 * like stale data, because it is a number the engine itself does not stand
 * behind.
 */
export interface ExecutionGateInput {
  readonly risk: Guarded<unknown>;
  readonly publishable: boolean;
  readonly matrixAgeMs: number | null;
  readonly matrixStaleAfterMs?: number;
}

export interface ExecutionGate {
  readonly allowed: boolean;
  readonly reasons: readonly string[];
}

export function executionGate(input: ExecutionGateInput): ExecutionGate {
  const reasons: string[] = [];
  const matrixLimit = input.matrixStaleAfterMs ?? MATRIX_STALE_AFTER_MS;

  if (!hasValue(input.risk)) {
    reasons.push(input.risk.reason);
  } else if (input.risk.freshness !== 'fresh') {
    reasons.push(
      `risk data is ${Math.round(input.risk.ageMs / 1000)}s old; execution requires fresh data`,
    );
  }
  if (!input.publishable) {
    reasons.push(
      'the estimate did not reach its required confidence interval and is not published (§2.5)',
    );
  }
  if (input.matrixAgeMs === null) {
    reasons.push('the correlation matrix has never been built');
  } else if (input.matrixAgeMs >= matrixLimit) {
    reasons.push(
      `the correlation matrix is ${Math.round(input.matrixAgeMs / 1000)}s old, past its ${Math.round(
        matrixLimit / 1000,
      )}s limit`,
    );
  }
  return { allowed: reasons.length === 0, reasons };
}
