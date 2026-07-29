'use client';

/**
 * Display primitives that cannot render a number without its uncertainty
 * and its age.
 *
 * §4: an estimate is never shown as a bare point. §6: the age is always
 * visible, and beyond five minutes the value is *withheld* rather than
 * greyed out — a greyed number is still a number on the screen, and a
 * stressed trader reads it.
 *
 * The enforcement is structural, not a convention: `GuardedPanel` takes the
 * discriminated union, and the render callback is only invoked on the
 * variants that actually carry a value.
 */

import type { Estimate, Guarded } from '@/lib/contract';
import { formatAge, hasValue } from '@/lib/contract';

export function AgeBadge({ ageMs, degraded }: { ageMs: number | null; degraded: boolean }) {
  return (
    <span
      className={
        'rounded px-2 py-0.5 text-xs font-medium ' +
        (degraded ? 'bg-amber-100 text-amber-900' : 'bg-neutral-100 text-neutral-600')
      }
      title="Age of the risk estimate at the moment the engine produced it"
    >
      {formatAge(ageMs)}
    </span>
  );
}

/**
 * A number with its 95% interval. The interval is not an optional decoration
 * — it is the honest width of what is known, and §4 makes returning a point
 * without one impossible upstream.
 */
export function EstimateValue({
  estimate,
  format,
  emphasis = false,
}: {
  estimate: Estimate;
  format: (x: number) => string;
  emphasis?: boolean;
}) {
  return (
    <div>
      <div className={emphasis ? 'text-3xl font-semibold tabular-nums' : 'text-xl tabular-nums'}>
        {format(estimate.point)}
      </div>
      <div className="mt-1 text-xs text-neutral-500 tabular-nums">
        95% interval {format(estimate.ci_low)} – {format(estimate.ci_high)}
      </div>
    </div>
  );
}

export function GuardedPanel<T>({
  title,
  subtitle,
  payload,
  children,
}: {
  title: string;
  subtitle?: string;
  payload: Guarded<T> | null;
  children: (value: T, degraded: boolean) => React.ReactNode;
}) {
  return (
    <section className="rounded-lg border border-neutral-200 bg-white p-5 shadow-sm">
      <header className="mb-4 flex items-start justify-between gap-4">
        <div>
          <h2 className="text-sm font-semibold uppercase tracking-wide text-neutral-500">
            {title}
          </h2>
          {subtitle ? <p className="mt-0.5 text-xs text-neutral-400">{subtitle}</p> : null}
        </div>
        {payload ? <AgeBadge ageMs={payload.ageMs} degraded={payload.degraded} /> : null}
      </header>

      {payload === null ? (
        <p className="text-sm text-neutral-400">Loading…</p>
      ) : hasValue(payload) ? (
        <>
          {payload.freshness === 'stale' ? (
            <p className="mb-3 rounded border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-900">
              This estimate is {formatAge(payload.ageMs)} and may not reflect the current
              market. Execution is blocked until it refreshes.
            </p>
          ) : null}
          {children(payload.value, payload.degraded)}
        </>
      ) : (
        /*
         * §6: values are hidden, not greyed. There is no number to show here
         * because the payload does not contain one — the type has no `value`
         * field on this variant.
         */
        <div className="rounded border border-red-200 bg-red-50 px-3 py-4">
          <p className="text-sm font-medium text-red-900">Risk estimate unavailable</p>
          <p className="mt-1 text-xs text-red-800">{payload.reason}</p>
          <p className="mt-2 text-xs text-red-700">
            Nothing is shown rather than something out of date. A stale liquidation
            probability during a fast market is worse than no number at all.
          </p>
        </div>
      )}
    </section>
  );
}
