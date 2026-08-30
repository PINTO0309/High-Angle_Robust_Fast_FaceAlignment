// フレーム → 頭部検出 → 頭部ごとのランドマーク推定(2 段パイプライン)。

import type { FaceAligner } from './aligner';
import type { HeadDetector } from './detector';
import type { FrameOutput, FrameSource } from './types';

// どの段で失敗したかをエラー文言に付ける(ort の実行エラーはノード名しか含まない)
async function stage<T>(name: string, fn: () => Promise<T>): Promise<T> {
  try {
    return await fn();
  } catch (error) {
    throw new Error(`${name}: ${error instanceof Error ? error.message : String(error)}`);
  }
}

export class HrffaPipeline {
  readonly detector: HeadDetector;
  readonly aligner: FaceAligner;

  constructor(detector: HeadDetector, aligner: FaceAligner) {
    this.detector = detector;
    this.aligner = aligner;
  }

  async process(src: FrameSource): Promise<FrameOutput> {
    const t0 = performance.now();
    const boxes = await stage('detector', () => this.detector.detect(src));
    const t1 = performance.now();
    const heads = await stage('aligner', () => this.aligner.align(src, boxes));
    const t2 = performance.now();
    return {
      heads,
      stats: { detectMs: t1 - t0, alignMs: t2 - t1, nHeads: heads.length },
    };
  }
}
