// Ensures the electron binary exists. pnpm's store-side build cache can mark
// electron as "built" while the hardlinked package carries no dist/ — in that
// case rerun electron's own install.js (which downloads the official release
// and verifies it against checksums.json / SHASUMS256).
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const electronDir = path.join(root, 'node_modules', 'electron');

const binary =
  process.platform === 'darwin'
    ? path.join(electronDir, 'dist', 'Electron.app', 'Contents', 'MacOS', 'Electron')
    : path.join(electronDir, 'dist', process.platform === 'win32' ? 'electron.exe' : 'electron');

if (!existsSync(electronDir)) {
  console.warn('[ensure-electron] electron package not installed; skipping');
  process.exit(0);
}

if (existsSync(binary)) {
  process.exit(0);
}

console.log('[ensure-electron] electron binary missing — running electron/install.js');
const result = spawnSync(process.execPath, ['install.js'], {
  cwd: electronDir,
  stdio: 'inherit',
});
if (result.status !== 0 || !existsSync(binary)) {
  console.error('[ensure-electron] failed to provision the electron binary');
  process.exit(result.status ?? 1);
}
console.log('[ensure-electron] done');
