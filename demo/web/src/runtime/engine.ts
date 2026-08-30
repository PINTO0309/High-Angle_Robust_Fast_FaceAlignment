// 推論エンジンの抽象と、models/ と wasm/ を解決する基準 URL。
// Worker には document が無く、自身のスクリプト URL は dist/assets/ 配下になるため、
// メインスレッドが init メッセージで document.baseURI を渡す。

export type Accelerator = 'webgpu' | 'wasm';
// 推論の実行場所: 専用 Worker(既定)か UI スレッド(?worker=main / Electron の --web-inference-worker main)
export type WorkerMode = 'dedicated' | 'main';

export interface TensorIn {
  data: Float32Array;
  dims: number[];
}

export interface TensorOut {
  data: Float32Array;
  dims: number[];
}

export interface OrtModel {
  accelerator: Accelerator;
  inputNames: string[];
  // 先頭入力の宣言形状(記号次元は -1)
  inputDims: number[];
  outputNames: string[];
  run(feeds: Record<string, TensorIn>): Promise<Record<string, TensorOut>>;
  dispose(): void;
}

let assetBaseUrl: string = typeof document !== 'undefined' ? document.baseURI : self.location.href;

export function setAssetBaseUrl(url: string): void {
  assetBaseUrl = url;
}

export function assetUrl(relative: string): string {
  return new URL(relative, assetBaseUrl).href;
}

export function activeWorkerMode(): WorkerMode {
  return new URLSearchParams(window.location.search).get('worker') === 'main' ? 'main' : 'dedicated';
}
