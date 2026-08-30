// Zero-dependency dev orchestrator (replaces npm-run-all + wait-on to keep
// the supply-chain surface minimal): starts the vite dev server, waits for
// its port, compiles the electron main process, then launches electron.
import { spawn } from 'node:child_process';
import net from 'node:net';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const DEV_PORT = 5274; // keep in sync with vite.config.ts / electron/main.ts
const nodeBin = process.execPath;
const viteMain = path.join(root, 'node_modules', 'vite', 'bin', 'vite.js');
const tscMain = path.join(root, 'node_modules', 'typescript', 'bin', 'tsc');

const children = [];
let shuttingDown = false;

function shutdown(code) {
  if (shuttingDown) {
    return;
  }
  shuttingDown = true;
  for (const child of children) {
    if (!child.killed) {
      child.kill('SIGTERM');
    }
  }
  process.exit(code);
}

process.on('SIGINT', () => shutdown(130));
process.on('SIGTERM', () => shutdown(143));

function run(cmd, args, opts = {}) {
  const child = spawn(cmd, args, { stdio: 'inherit', cwd: root, ...opts });
  children.push(child);
  return child;
}

function runToCompletion(cmd, args) {
  return new Promise((resolve, reject) => {
    const child = run(cmd, args);
    child.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`${path.basename(cmd)} exited with ${code}`));
      }
    });
  });
}

function waitForTcp(port, host, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const attempt = () => {
      const socket = net.connect({ port, host });
      socket.once('connect', () => {
        socket.destroy();
        resolve();
      });
      socket.once('error', () => {
        socket.destroy();
        if (Date.now() > deadline) {
          reject(new Error(`timed out waiting for tcp:${port}`));
        } else {
          setTimeout(attempt, 250);
        }
      });
    };
    attempt();
  });
}

const vite = run(nodeBin, [viteMain]);
vite.on('close', (code) => shutdown(code ?? 0));

try {
  await waitForTcp(DEV_PORT, '127.0.0.1', 60000);
  await runToCompletion(nodeBin, [tscMain, '-p', 'tsconfig.electron.json']);
} catch (error) {
  console.error(`[dev] ${error.message}`);
  shutdown(1);
}

const { default: electron } = await import('electron');
const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE;
// forward CLI options (e.g. --web-inference-worker main) to the electron main process
const passthroughArgs = process.argv.slice(2);
const app = spawn(electron, ['dist-electron/main.js', '--dev', ...passthroughArgs], {
  stdio: 'inherit',
  cwd: root,
  env,
});
children.push(app);
app.on('close', () => shutdown(0));
