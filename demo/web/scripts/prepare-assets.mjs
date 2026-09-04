// Copies the onnxruntime-web wasm assets into public/wasm/ort and stages the
// ONNX models into public/models with a manifest. Two sources:
//   demo/web/models/      deimv2_*_boxes_only_webgpu.onnx (DEIMv2-Wholebody49 detectors rewritten
//                         for onnxruntime-web by scripts/onnx_web_compat.py; the demo only uses boxes,
//                         so the variants with a masks output are not staged), yolov9_*.onnx
//                         (YOLOv9-Wholebody34 raw-head exports, decoded + NMS in the app) and hrffa_*.onnx
//   ../../data/models/    hrffa_*.onnx plus yawnet_distill_{064,096,128}_unified_v6u_kappa*.onnx (head orientation,
//                         optional) only (the HRFFA graphs run unmodified; the original
//                         DEIMv2 exports do NOT load in onnxruntime-web and are not staged)
// Files above MAX_MODEL_BYTES (the 1.2 GB ViT-L teacher) are skipped unless
// HRFFA_WEB_INCLUDE_LARGE=1 is set.
//
// Everything the page loads is served from its own origin: no CDN, no
// third-party script.
import { copyFileSync, existsSync, mkdirSync, readdirSync, statSync, unlinkSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const MODEL_SOURCES = [
  { dir: path.join(root, 'models'), pattern: /^(deimv2_.*_boxes_only.*|yolov9_.*|hrffa_.*|yawnet_distill_(?:064|096|128)_unified_v6u_kappa[^/]*)\.onnx$/i },
  { dir: path.join(root, '..', '..', 'data', 'models'), pattern: /^(hrffa_.*|yawnet_distill_(?:064|096|128)_unified_v6u_kappa[^/]*)\.onnx$/i },
];
const MAX_MODEL_BYTES = 300 * 1024 * 1024;
const includeLarge = process.env.HRFFA_WEB_INCLUDE_LARGE === '1';

function ensureDir(dirPath) {
  mkdirSync(dirPath, { recursive: true });
}

function listModelFiles(dir, pattern) {
  if (!existsSync(dir)) {
    return [];
  }
  return readdirSync(dir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && pattern.test(entry.name))
    .map((entry) => path.join(dir, entry.name));
}

function copyModels() {
  const modelOutDir = path.join(root, 'public', 'models');
  ensureDir(modelOutDir);
  for (const entry of readdirSync(modelOutDir, { withFileTypes: true })) {
    if (entry.isFile() && (entry.name.endsWith('.onnx') || entry.name === 'manifest.json')) {
      unlinkSync(path.join(modelOutDir, entry.name));
    }
  }

  const seen = new Set();
  const manifest = [];
  let skipped = 0;
  for (const { dir, pattern } of MODEL_SOURCES) {
    for (const src of listModelFiles(dir, pattern)) {
      const name = path.basename(src);
      if (seen.has(name)) {
        continue;
      }
      const bytes = statSync(src).size;
      if (bytes > MAX_MODEL_BYTES && !includeLarge) {
        console.warn(`[prepare-assets] skipped ${name} (${(bytes / 1e6).toFixed(0)} MB > limit; set HRFFA_WEB_INCLUDE_LARGE=1 to stage it)`);
        skipped += 1;
        continue;
      }
      seen.add(name);
      copyFileSync(src, path.join(modelOutDir, name));
      manifest.push({ name, bytes });
    }
  }
  manifest.sort((a, b) => a.name.localeCompare(b.name));
  writeFileSync(path.join(modelOutDir, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(`[prepare-assets] staged ${manifest.length} model(s)${skipped > 0 ? `, skipped ${skipped}` : ''}`);
  if (manifest.length === 0) {
    console.warn('[prepare-assets] no models found — run `uv run python scripts/onnx_web_compat.py data/models/deimv2_*.onnx --out-dir demo/web/models` and keep hrffa_*.onnx in data/models/');
  }
}

function copyWasm() {
  // onnxruntime-web runtime assets. The webgpu EP of ort 1.27 loads the
  // ASYNCIFY wasm variant at runtime — omitting it fails session creation
  // with a permanently cached "previous call to 'initWasm()' failed".
  const ortDistSrc = path.join(root, 'node_modules', 'onnxruntime-web', 'dist');
  const ortWasmDst = path.join(root, 'public', 'wasm', 'ort');
  const ortFiles = [
    'ort-wasm-simd-threaded.wasm',
    'ort-wasm-simd-threaded.mjs',
    'ort-wasm-simd-threaded.jsep.wasm',
    'ort-wasm-simd-threaded.jsep.mjs',
    'ort-wasm-simd-threaded.asyncify.wasm',
    'ort-wasm-simd-threaded.asyncify.mjs',
    'ort-wasm-simd-threaded.jspi.wasm',
    'ort-wasm-simd-threaded.jspi.mjs',
  ];
  if (!existsSync(ortDistSrc)) {
    console.warn(`[prepare-assets] onnxruntime-web dist directory missing: ${ortDistSrc}`);
    return;
  }
  ensureDir(ortWasmDst);
  for (const file of ortFiles) {
    const src = path.join(ortDistSrc, file);
    if (existsSync(src)) {
      copyFileSync(src, path.join(ortWasmDst, file));
    } else {
      console.warn(`[prepare-assets] onnxruntime-web asset missing: ${src}`);
    }
  }
}

copyModels();
copyWasm();
console.log('[prepare-assets] done');
