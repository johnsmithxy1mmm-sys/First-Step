/**
 * Backend entry point.
 *
 * Deliberately does not fail to start when the risk engine is down. §6 is
 * about what the product does when data stops arriving, and it cannot do
 * anything at all if the process that renders "unavailable" refuses to boot.
 */

import { RiskClient } from './risk/client.js';
import { buildServer } from './server.js';

const port = Number(process.env['PORT'] ?? 8080);
const host = process.env['HOST'] ?? '127.0.0.1';
const riskUrl = process.env['RISK_SERVICE_URL'] ?? 'http://127.0.0.1:8787';

const app = buildServer({
  risk: new RiskClient({ baseUrl: riskUrl }),
  logger: true,
});

try {
  await app.listen({ port, host });
  app.log.info(`backend listening on http://${host}:${port}, risk engine at ${riskUrl}`);
} catch (error) {
  app.log.error(error);
  process.exit(1);
}
