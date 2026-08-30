// 専用推論 Worker のメインスレッド側クライアント: RGBA フレームを転送し結果を待つ。
// Worker はメッセージを到着順に直列処理して 1 送信につき 1 応答を返すので、待ち手を FIFO で対応付ける。

// inline (blob URL) worker: from file:// pages (packaged Electron app) the CSP
// 'self' source cannot match worker scripts (opaque origin), while blob:
// workers stay allowed — same setup as PINTO0309/soma
import InferenceWorker from '../workers/inference.worker.ts?worker&inline';
import type { Accelerator } from './engine';
import type { InputNorm } from '../hrffa/constants';
import type { FrameOutput } from '../hrffa/types';
import type { MainToWorkerMessage, WorkerReadyInfo, WorkerToMainMessage } from '../workers/inference.worker';

export interface WorkerInitOptions {
  accelerator: Accelerator;
  numThreads: number;
  detectorUrl: string;
  alignerUrl: string;
  headScoreThreshold: number;
  cropPad: number;
  inputNorm: InputNorm;
}

export class WorkerPipeline {
  private worker: Worker;
  private waiters: Array<(msg: WorkerToMainMessage) => void> = [];
  private terminated = false;

  constructor() {
    this.worker = new InferenceWorker();
    this.worker.onmessage = (event: MessageEvent<WorkerToMainMessage>) => {
      const resolve = this.waiters.shift();
      resolve?.(event.data);
    };
    this.worker.onerror = (event: ErrorEvent) => {
      const pending = this.waiters;
      this.waiters = [];
      for (const resolve of pending) {
        resolve({ type: 'frameError', message: `inference worker error: ${event.message}` });
      }
    };
  }

  private post(message: MainToWorkerMessage, transfer?: Transferable[]): Promise<WorkerToMainMessage> {
    return new Promise((resolve) => {
      this.waiters.push(resolve);
      this.worker.postMessage(message, transfer ?? []);
    });
  }

  async init(opts: WorkerInitOptions): Promise<WorkerReadyInfo> {
    const reply = await this.post({
      type: 'init',
      accelerator: opts.accelerator,
      numThreads: opts.numThreads,
      assetBaseUrl: document.baseURI,
      detectorUrl: new URL(opts.detectorUrl, document.baseURI).href,
      alignerUrl: new URL(opts.alignerUrl, document.baseURI).href,
      headScoreThreshold: opts.headScoreThreshold,
      cropPad: opts.cropPad,
      inputNorm: opts.inputNorm,
    });
    if (reply.type === 'ready') {
      return reply;
    }
    throw new Error(reply.type === 'initError' ? reply.message : `unexpected worker reply: ${reply.type}`);
  }

  async process(frame: ImageData): Promise<FrameOutput> {
    const buffer = frame.data.buffer as ArrayBuffer;
    const reply = await this.post({ type: 'frame', rgba: buffer, width: frame.width, height: frame.height }, [buffer]);
    if (reply.type === 'result') {
      return reply.output;
    }
    throw new Error(reply.type === 'frameError' ? reply.message : `unexpected worker reply: ${reply.type}`);
  }

  async setParams(params: { headScoreThreshold?: number; cropPad?: number }): Promise<void> {
    await this.post({ type: 'params', ...params });
  }

  // 停止: Worker に 'stop' を送り、セッション解放の完了応答(直列処理なので進行中のフレームの後に来る)を
  // 待ってから terminate する。応答が来ない場合は 3 秒で打ち切る
  async dispose(): Promise<void> {
    if (this.terminated) {
      return;
    }
    this.terminated = true;
    const stopped = this.post({ type: 'stop' });
    await Promise.race([stopped, new Promise<void>((resolve) => setTimeout(resolve, 3000))]);
    this.worker.terminate();
    this.waiters = [];
  }
}
