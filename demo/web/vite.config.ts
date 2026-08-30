import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Dev / preview port: 5274 (5173 is commonly taken by other vite projects).
// Keep in sync with scripts/dev.mjs and electron/main.ts.
export const DEV_PORT = 5274;

// Cross-origin isolation (SharedArrayBuffer) for the multi-threaded wasm
// backend; every asset of the page is same-origin, so require-corp is safe.
// (The Electron shell injects the same headers itself for http(s) and enables
// SharedArrayBuffer by feature flag for file://.)
const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  base: './',
  plugins: [react()],
  worker: {
    // iife workers keep working when the packaged app is loaded from file://
    // (module workers are blocked there) — same setup as PINTO0309/soma
    format: 'iife',
  },
  build: {
    target: 'es2022',
  },
  server: {
    port: DEV_PORT,
    strictPort: true,
    headers: isolationHeaders,
  },
  preview: {
    port: DEV_PORT,
    strictPort: true,
    headers: isolationHeaders,
  },
});
