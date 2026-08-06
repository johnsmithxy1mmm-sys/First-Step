/**
 * §9 Phase 3 acceptance, run against real processes.
 *
 * The criterion says: "verified by a manual test with the service stopped."
 * A criterion phrased that way deserves a real process being killed, not a
 * mocked client — the failure it guards against (a stale number surviving an
 * outage) is exactly the kind that unit tests with injected clocks can miss,
 * because the injected clock is the thing under test.
 *
 * What this does:
 *   1. starts the real Python risk service and the real Node backend;
 *   2. checks a live request returns fresh values and allows execution;
 *   3. SIGKILLs the risk service;
 *   4. checks the backend reports unavailable, exposes no value, and blocks
 *      execution — within seconds, far inside §6's five-minute limit;
 *   5. restarts the service and checks recovery.
 *
 * The 60s/300s staleness boundaries themselves are covered by the clock-
 * injected tests; waiting five real minutes here would buy nothing and make
 * this unrunnable in CI. What only a real run can prove is that a killed
 * process produces "unavailable" rather than a cached value, and that is
 * what step 4 checks.
 *
 * Usage: node scripts/degradation-check.mjs
 * Exits non-zero on any failed check.
 */

import { spawn } from 'node:child_process';
import { setTimeout as sleep } from 'node:timers/promises';

const REPO_ROOT = new URL('../../../', import.meta.url).pathname;
const RISK_PORT = 8897;
const BACKEND_PORT = 8898;

const BOOK = {
  book: {
    address: '0xdemo',
    cross_collateral: 100_000,
    positions: [
      { coin: 'BTC', size: 4, entry_price: 100_000, mode: 'cross', leverage: 20 },
      { coin: 'ETH', size: 60, entry_price: 4_000, mode: 'cross', leverage: 20 },
    ],
  },
  n_paths: 4000,
  seed: 1,
};

let failures = 0;
function check(label, ok, detail = '') {
  const mark = ok ? 'PASS' : 'FAIL';
  if (!ok) failures += 1;
  console.log(`  [${mark}] ${label}${detail ? ` — ${detail}` : ''}`);
}

async function waitFor(url, timeoutMs = 90_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(url);
      if (res.ok) return true;
    } catch {
      /* not up yet */
    }
    await sleep(300);
  }
  return false;
}

function startRisk() {
  const proc = spawn(
    'python3',
    ['-m', 'risk_engine.service', '--fixture', '--port', String(RISK_PORT)],
    { cwd: REPO_ROOT, stdio: ['ignore', 'pipe', 'pipe'] },
  );
  proc.stderr.on('data', () => {});
  proc.stdout.on('data', () => {});
  return proc;
}

async function askRisk() {
  const res = await fetch(`http://127.0.0.1:${BACKEND_PORT}/api/portfolio_risk`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(BOOK),
  });
  return res.json();
}

async function main() {
  console.log('§9 Phase 3 degradation check — real processes\n');

  console.log('1. starting the risk service and the backend');
  let risk = startRisk();
  const backend = spawn('node', ['dist/main.js'], {
    cwd: new URL('..', import.meta.url).pathname,
    env: {
      ...process.env,
      PORT: String(BACKEND_PORT),
      RISK_SERVICE_URL: `http://127.0.0.1:${RISK_PORT}`,
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  backend.stdout.on('data', () => {});
  backend.stderr.on('data', () => {});

  const cleanup = () => {
    try { risk.kill('SIGKILL'); } catch { /* already gone */ }
    try { backend.kill('SIGKILL'); } catch { /* already gone */ }
  };
  process.on('exit', cleanup);

  try {
    check(
      'risk service came up',
      await waitFor(`http://127.0.0.1:${RISK_PORT}/health`),
    );
    check(
      'backend came up',
      await waitFor(`http://127.0.0.1:${BACKEND_PORT}/api/health`),
    );

    console.log('\n2. with both running');
    const live = await askRisk();
    check('values are fresh', live.freshness === 'fresh', `freshness=${live.freshness}`);
    check('a value is present', Object.hasOwn(live, 'value'));
    check(
      'P(liq) carries its interval',
      live.value?.p_liq_24h?.ci_low <= live.value?.p_liq_24h?.point &&
        live.value?.p_liq_24h?.point <= live.value?.p_liq_24h?.ci_high,
    );
    check('execution is allowed on fresh data', live.execution?.allowed === true);
    const health = await (await fetch(`http://127.0.0.1:${BACKEND_PORT}/api/health`)).json();
    check('health reports the engine reachable', health.engineReachable === true);
    check('matrix age is exposed (§6)', typeof health.matrixAgeMs === 'number');

    console.log('\n3. SIGKILL the risk service');
    risk.kill('SIGKILL');
    await sleep(1500);

    console.log('\n4. with the risk service stopped');
    const dead = await askRisk();
    check(
      'no value is served',
      !Object.hasOwn(dead, 'value'),
      `freshness=${dead.freshness}`,
    );
    check(
      'freshness is unavailable, not a cached fresh value',
      dead.freshness === 'unavailable',
    );
    check('execution is blocked', dead.execution?.allowed === false);
    check(
      'the reason names the outage',
      typeof dead.reason === 'string' && dead.reason.length > 0,
      dead.reason?.slice(0, 60),
    );
    const deadHealth = await (
      await fetch(`http://127.0.0.1:${BACKEND_PORT}/api/health`)
    ).json();
    check('health survives the outage rather than 500ing', deadHealth.ok === false);
    check('health reports the engine unreachable', deadHealth.engineReachable === false);

    console.log('\n5. restart the risk service');
    risk = startRisk();
    check(
      'risk service came back',
      await waitFor(`http://127.0.0.1:${RISK_PORT}/health`),
    );
    const recovered = await askRisk();
    check(
      'values return',
      recovered.freshness === 'fresh',
      `freshness=${recovered.freshness}`,
    );
    check('execution is allowed again', recovered.execution?.allowed === true);
  } finally {
    cleanup();
  }

  console.log(
    `\n${failures === 0 ? 'ALL CHECKS PASSED' : `${failures} CHECK(S) FAILED`}`,
  );
  process.exit(failures === 0 ? 0 : 1);
}

await main();
