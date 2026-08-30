// Downloads the demo models from the repository's GitHub release (tag `weights`) into
// demo/web/models/ and verifies every file against the SHA-256 pinned in models.lock.json
// (supply-chain policy: nothing unverified is staged; a mismatch deletes the download and
// fails). Files that already exist with the right hash are kept. Zero dependencies.
//
//   pnpm run fetch:models            # all entries of models.lock.json
//   pnpm run fetch:models -- vitt    # only entries whose name contains "vitt"
import { createHash } from 'node:crypto';
import { createWriteStream, existsSync, mkdirSync, readFileSync, statSync, unlinkSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const lock = JSON.parse(readFileSync(path.join(root, 'models.lock.json'), 'utf8'));
const outDir = path.join(root, 'models');
const filters = process.argv.slice(2);
mkdirSync(outDir, { recursive: true });

function sha256(file) {
  const hash = createHash('sha256');
  hash.update(readFileSync(file));
  return hash.digest('hex');
}

let failed = 0;
for (const entry of lock.files) {
  if (filters.length > 0 && !filters.some((f) => entry.name.includes(f))) {
    continue;
  }
  const dst = path.join(outDir, entry.name);
  if (existsSync(dst) && statSync(dst).size === entry.bytes && sha256(dst) === entry.sha256) {
    console.log(`[fetch-models] ok       ${entry.name}`);
    continue;
  }
  const url = `${lock.baseUrl}/${entry.name}`;
  console.log(`[fetch-models] download ${entry.name} (${(entry.bytes / 1e6).toFixed(1)} MB)`);
  const response = await fetch(url, { redirect: 'follow' });
  if (!response.ok || response.body === null) {
    console.error(`[fetch-models] FAILED   ${entry.name}: HTTP ${response.status}`);
    failed += 1;
    continue;
  }
  const tmp = `${dst}.part`;
  await pipeline(Readable.fromWeb(response.body), createWriteStream(tmp));
  const digest = sha256(tmp);
  if (statSync(tmp).size !== entry.bytes || digest !== entry.sha256) {
    unlinkSync(tmp);
    console.error(`[fetch-models] REJECTED ${entry.name}: sha256 ${digest} does not match the pinned ${entry.sha256}`);
    failed += 1;
    continue;
  }
  if (existsSync(dst)) {
    unlinkSync(dst);
  }
  const { renameSync } = await import('node:fs');
  renameSync(tmp, dst);
  console.log(`[fetch-models] verified ${entry.name}`);
}
if (failed > 0) {
  console.error(`[fetch-models] ${failed} file(s) failed`);
  process.exit(1);
}
console.log('[fetch-models] done — run `pnpm run prepare:assets` (or dev / build) to stage the models');
