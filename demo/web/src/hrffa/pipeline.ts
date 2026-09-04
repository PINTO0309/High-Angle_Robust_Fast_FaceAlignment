// フレーム → 頭部検出 → 頭部ごとのランドマーク推定(2 段パイプライン)。

import type { FaceAligner } from './aligner';
import type { HeadDetector } from './detector';
import type { HeadOrientation } from './orientation';
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
  readonly orientation: HeadOrientation | null;

  constructor(detector: HeadDetector, aligner: FaceAligner, orientation: HeadOrientation | null = null) {
    this.detector = detector;
    this.aligner = aligner;
    this.orientation = orientation;
  }

  async process(src: FrameSource): Promise<FrameOutput> {
    const t0 = performance.now();
    const boxes = await stage('detector', () => this.detector.detect(src));
    const t1 = performance.now();
    const heads = await stage('aligner', () => this.aligner.align(src, boxes));
    const t2 = performance.now();
    if (this.orientation !== null && heads.length > 0) {
      const degrees = await stage('orientation', () => this.orientation!.estimate(src, boxes));
      heads.forEach((h, i) => {
        h.orientationDeg = degrees[i];
      });
    }
    const t3 = performance.now();
    return {
      heads,
      stats: { detectMs: t1 - t0, alignMs: t2 - t1, orientMs: t3 - t2, nHeads: heads.length },
    };
  }
}
