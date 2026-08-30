// 専用推論 Worker: モデルのロードと 2 段パイプライン(検出 → 整列)をここで動かし、
// UI スレッドを塞がない。メインスレッドは 1 メッセージにつき 1 フレーム(RGBA)を送り、
// 頭部ごとの結果(bbox・点・可視性)を受け取る。

import { setAssetBaseUrl, type Accelerator, type OrtModel } from '../runtime/engine';
import { loadOrtModel } from '../runtime/ort';
import { FaceAligner } from '../hrffa/aligner';
import { HeadDetector } from '../hrffa/detector';
import { HrffaPipeline } from '../hrffa/pipeline';
import type { InputNorm } from '../hrffa/constants';
import type { FrameOutput, FrameSource } from '../hrffa/types';

export interface WorkerInitMessage {
  type: 'init';
  accelerator: Accelerator;
  numThreads: number;
  assetBaseUrl: string;
  detectorUrl: string;
  alignerUrl: string;
  headScoreThreshold: number;
  cropPad: number;
  inputNorm: InputNorm;
}

export interface WorkerFrameMessage {
  type: 'frame';
  rgba: ArrayBuffer;
  width: number;
  height: number;
}

export interface WorkerParamsMessage {
  type: 'params';
  headScoreThreshold?: number;
  cropPad?: number;
}

export type MainToWorkerMessage = WorkerInitMessage | WorkerFrameMessage | WorkerParamsMessage | { type: 'stop' };

export interface WorkerReadyInfo {
  accelerator: Accelerator;
  note: string | null;
  detectorInput: [number, number];
  detectorFormat: string;
  alignerInput: number;
  alignerBatch: 'N' | '1';
}

export type WorkerToMainMessage =
  | ({ type: 'ready' } & WorkerReadyInfo)
  | { type: 'initError'; message: string }
  | { type: 'result'; output: FrameOutput }
  | { type: 'frameError'; message: string }
  | { type: 'paramsOk' }
  | { type: 'stopped' };

let pipeline: HrffaPipeline | null = null;
let engines: OrtModel[] = [];
let frameCtx: OffscreenCanvasRenderingContext2D | null = null;

const post = (message: WorkerToMainMessage): void => {
  self.postMessage(message);
};

async function fetchBytes(url: string): Promise<Uint8Array> {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Failed to fetch model: ${response.status} ${response.statusText} (${url})`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

async function initialize(msg: WorkerInitMessage): Promise<void> {
  setAssetBaseUrl(msg.assetBaseUrl);
  let accel: Accelerator = msg.accelerator;
  let note: string | null = null;

  const loadWithFallback = async (bytes: Uint8Array): Promise<OrtModel> => {
    try {
      return await loadOrtModel(bytes, accel, msg.numThreads);
    } catch (error) {
      if (accel === 'webgpu') {
        note = `WebGPU init failed — fell back to WASM: ${error instanceof Error ? error.message : String(error)}`;
        accel = 'wasm';
        return loadOrtModel(bytes, accel, msg.numThreads);
      }
      throw error;
    }
  };

  const detectorEngine = await loadWithFallback(await fetchBytes(msg.detectorUrl));
  engines.push(detectorEngine);
  const alignerEngine = await loadWithFallback(await fetchBytes(msg.alignerUrl));
  engines.push(alignerEngine);

  const detector = new HeadDetector(detectorEngine, msg.headScoreThreshold);
  const aligner = new FaceAligner(alignerEngine, msg.inputNorm, msg.cropPad);
  pipeline = new HrffaPipeline(detector, aligner);
  // ウォームアップ: 初回実行のシェーダコンパイル・メモリ確保を計測から外す(Python デモと同じ)
  const warm = new OffscreenCanvas(detector.inWidth, detector.inHeight);
  const warmSource: FrameSource = {
    width: warm.width,
    height: warm.height,
    draw: (ctx, w, h) => ctx.drawImage(warm, 0, 0, w, h),
    drawCrop: (ctx, sx, sy, sw, sh, dw, dh) => ctx.drawImage(warm, sx, sy, sw, sh, 0, 0, dw, dh),
  };
  await detector.detect(warmSource);
  await aligner.align(warmSource, [{ x1: 0, y1: 0, x2: warm.width - 1, y2: warm.height - 1, score: 1 }]);
  post({
    type: 'ready',
    accelerator: accel,
    note,
    detectorInput: [detector.inWidth, detector.inHeight],
    detectorFormat: detector.format,
    alignerInput: aligner.size,
    alignerBatch: aligner.dynamicBatch ? 'N' : '1',
  });
}

async function processFrame(msg: WorkerFrameMessage): Promise<void> {
  if (pipeline === null) {
    post({ type: 'frameError', message: 'worker not initialized' });
    return;
  }
  if (frameCtx === null || frameCtx.canvas.width !== msg.width || frameCtx.canvas.height !== msg.height) {
    const canvas = new OffscreenCanvas(msg.width, msg.height);
    frameCtx = canvas.getContext('2d', { willReadFrequently: false });
    if (frameCtx === null) {
      post({ type: 'frameError', message: '2D canvas context unavailable in worker' });
      return;
    }
  }
  frameCtx.putImageData(new ImageData(new Uint8ClampedArray(msg.rgba), msg.width, msg.height), 0, 0);
  const frameCanvas = frameCtx.canvas;
  const source: FrameSource = {
    width: msg.width,
    height: msg.height,
    draw: (ctx, w, h) => ctx.drawImage(frameCanvas, 0, 0, w, h),
    drawCrop: (ctx, sx, sy, sw, sh, dw, dh) => ctx.drawImage(frameCanvas, sx, sy, sw, sh, 0, 0, dw, dh),
  };
  const output = await pipeline.process(source);
  post({ type: 'result', output });
}

function applyParams(msg: WorkerParamsMessage): void {
  if (pipeline !== null) {
    if (msg.headScoreThreshold !== undefined) {
      pipeline.detector.scoreThreshold = msg.headScoreThreshold;
    }
    if (msg.cropPad !== undefined) {
      pipeline.aligner.cropPad = msg.cropPad;
    }
  }
  post({ type: 'paramsOk' });
}

// ort は wasm 初期化失敗を内部にキャッシュして後から一般的な "previous call to 'initWasm()' failed"
// を投げるため、初期化中に出た console.error / 未処理の reject を捕まえて最初の原因を報告する。
const initDiagnostics: string[] = [];
for (const level of ['error', 'warn'] as const) {
  const original = console[level].bind(console);
  console[level] = (...args: unknown[]) => {
    initDiagnostics.push(
      args
        .map((a) => (a instanceof Error ? `${a.message}\n${a.stack ?? ''}` : String(a)))
        .join(' ')
        .slice(0, 500),
    );
    original(...args);
  };
}
self.addEventListener('unhandledrejection', (event) => {
  const reason = (event as PromiseRejectionEvent).reason;
  initDiagnostics.push(
    `unhandledrejection: ${reason instanceof Error ? `${reason.message}\n${reason.stack ?? ''}` : String(reason)}`.slice(0, 500),
  );
});

async function handle(msg: MainToWorkerMessage): Promise<void> {
  if (msg.type === 'init') {
    try {
      await initialize(msg);
    } catch (error) {
      const detail = initDiagnostics.length > 0 ? ` | diagnostics: ${initDiagnostics.slice(0, 4).join(' || ')}` : '';
      post({
        type: 'initError',
        message: `${error instanceof Error ? `${error.message}\n${error.stack ?? ''}` : String(error)}${detail}`,
      });
    }
  } else if (msg.type === 'frame') {
    try {
      await processFrame(msg);
    } catch (error) {
      post({ type: 'frameError', message: error instanceof Error ? error.message : String(error) });
    }
  } else if (msg.type === 'params') {
    applyParams(msg);
  } else if (msg.type === 'stop') {
    for (const engine of engines) {
      try {
        engine.dispose();
      } catch {
        // already disposed
      }
    }
    engines = [];
    pipeline = null;
    post({ type: 'stopped' });
  }
}

// メッセージは到着順に直列処理する(1 送信につき 1 応答、応答順 = 送信順)。
let queue: Promise<void> = Promise.resolve();
self.onmessage = (event: MessageEvent<MainToWorkerMessage>) => {
  const msg = event.data;
  queue = queue.then(() => handle(msg));
};
