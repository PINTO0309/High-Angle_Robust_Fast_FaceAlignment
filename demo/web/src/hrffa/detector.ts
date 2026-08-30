// 頭部検出。2 種類の export に対応する:
//   - DEIMv2-Wholebody49: 後処理込みの `label_xyxy_score` [1, Q, 6](class, x1, y1, x2, y2, score。座標は [0,1] 正規化、
//     `orig_target_sizes` 入力を持つ export では絶対座標)
//   - YOLOv9-Wholebody34(PINTO_model_zoo の生 head 出力): `[1, 4 + C, A]`(または `[1, A, 4 + C]`)、cx / cy / w / h は
//     入力画素空間、続く C 個がクラス確率。閾値と NMS(IoU 0.5)はここで行う
// 前処理はどちらも入力サイズへの直リサイズ・RGB・/255・正規化なし(Python デモと同じ)。どちらの語彙も class 7 = head。

import { HEAD_CLASS_ID, NMS_IOU_THRESHOLD } from './constants';
import { create2dContext, type Any2DContext } from '../runtime/canvas';
import type { OrtModel, TensorIn, TensorOut } from '../runtime/engine';
import type { FrameSource, HeadBox } from './types';

export type DetectorFormat = 'deimv2' | 'yolo-raw';

function iou(a: HeadBox, b: HeadBox): number {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return union > 0 ? inter / union : 0;
}

// 貪欲 NMS(スコア降順、IoU が閾値以上の箱を抑制)
function nms(boxes: HeadBox[], threshold: number): HeadBox[] {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const kept: HeadBox[] = [];
  for (const box of sorted) {
    if (kept.every((k) => iou(k, box) < threshold)) {
      kept.push(box);
    }
  }
  return kept;
}

export class HeadDetector {
  readonly inWidth: number;
  readonly inHeight: number;
  readonly format: DetectorFormat;
  scoreThreshold: number;
  private engine: OrtModel;
  private ctx: Any2DContext;
  private input: Float32Array;

  constructor(engine: OrtModel, scoreThreshold: number) {
    const dims = engine.inputDims;
    if (dims.length !== 4 || dims[1] !== 3 || dims[2] <= 0 || dims[3] <= 0) {
      throw new Error(`Detector input must be [N, 3, H, W] with fixed H/W, got [${dims.join(', ')}]`);
    }
    this.engine = engine;
    this.inHeight = dims[2];
    this.inWidth = dims[3];
    this.scoreThreshold = scoreThreshold;
    this.format = engine.outputNames.includes('label_xyxy_score') ? 'deimv2' : 'yolo-raw';
    this.ctx = create2dContext(this.inWidth, this.inHeight);
    this.input = new Float32Array(3 * this.inWidth * this.inHeight);
  }

  private preprocess(src: FrameSource): void {
    const w = this.inWidth;
    const h = this.inHeight;
    src.draw(this.ctx, w, h);
    const rgba = this.ctx.getImageData(0, 0, w, h).data;
    const n = w * h;
    const out = this.input;
    for (let i = 0; i < n; i += 1) {
      out[i] = rgba[i * 4] / 255;
      out[n + i] = rgba[i * 4 + 1] / 255;
      out[2 * n + i] = rgba[i * 4 + 2] / 255;
    }
  }

  private clip(box: HeadBox, src: FrameSource): HeadBox | null {
    const x1 = Math.min(Math.max(box.x1, 0), src.width - 1);
    const x2 = Math.min(Math.max(box.x2, 0), src.width - 1);
    const y1 = Math.min(Math.max(box.y1, 0), src.height - 1);
    const y2 = Math.min(Math.max(box.y2, 0), src.height - 1);
    if (x2 - x1 < 2 || y2 - y1 < 2) {
      return null;
    }
    return { x1, y1, x2, y2, score: box.score };
  }

  private decodeDeimv2(pred: TensorOut, src: FrameSource, absolute: boolean): HeadBox[] {
    const dims = pred.dims;
    if (dims[dims.length - 1] !== 6) {
      throw new Error(`Unexpected detector output shape [${dims.join(', ')}] (expected [1, Q, 6])`);
    }
    const rows = pred.data.length / 6;
    const heads: HeadBox[] = [];
    for (let i = 0; i < rows; i += 1) {
      const o = i * 6;
      const score = pred.data[o + 5];
      if (Math.round(pred.data[o]) !== HEAD_CLASS_ID || score < this.scoreThreshold) {
        continue;
      }
      const sx = absolute ? 1 : src.width;
      const sy = absolute ? 1 : src.height;
      const box = this.clip(
        { x1: pred.data[o + 1] * sx, y1: pred.data[o + 2] * sy, x2: pred.data[o + 3] * sx, y2: pred.data[o + 4] * sy, score },
        src,
      );
      if (box) {
        heads.push(box);
      }
    }
    return heads;
  }

  private decodeYoloRaw(outputs: Record<string, TensorOut>, src: FrameSource): HeadBox[] {
    // 最大の rank-3 出力を head として扱う([1, 4+C, A] か [1, A, 4+C]。チャネル軸は小さい方)
    let out: TensorOut | null = null;
    for (const name of this.engine.outputNames) {
      const o = outputs[name];
      if (o && o.dims.length === 3 && (out === null || o.data.length > out.data.length)) {
        out = o;
      }
    }
    if (out === null) {
      throw new Error(`Cannot interpret detector outputs (${this.engine.outputNames.join(', ')}): expected [1, 4+C, A]`);
    }
    const chFirst = out.dims[1] < out.dims[2];
    const ch = chFirst ? out.dims[1] : out.dims[2];
    const anchors = chFirst ? out.dims[2] : out.dims[1];
    const numClasses = ch - 4;
    if (numClasses <= HEAD_CLASS_ID) {
      throw new Error(`Detector output [${out.dims.join(', ')}] has ${numClasses} classes; head class ${HEAD_CLASS_ID} is not available`);
    }
    const data = out.data;
    const at = (c: number, a: number): number => (chFirst ? data[c * anchors + a] : data[a * ch + c]);
    const sx = src.width / this.inWidth;
    const sy = src.height / this.inHeight;
    const candidates: HeadBox[] = [];
    for (let a = 0; a < anchors; a += 1) {
      const score = at(4 + HEAD_CLASS_ID, a);
      if (score < this.scoreThreshold) {
        continue;
      }
      const cx = at(0, a);
      const cy = at(1, a);
      const hw = at(2, a) / 2;
      const hh = at(3, a) / 2;
      const box = this.clip({ x1: (cx - hw) * sx, y1: (cy - hh) * sy, x2: (cx + hw) * sx, y2: (cy + hh) * sy, score }, src);
      if (box) {
        candidates.push(box);
      }
    }
    return nms(candidates, NMS_IOU_THRESHOLD);
  }

  async detect(src: FrameSource): Promise<HeadBox[]> {
    this.preprocess(src);
    const feeds: Record<string, TensorIn> = {
      [this.engine.inputNames[0]]: { data: this.input, dims: [1, 3, this.inHeight, this.inWidth] },
    };
    const hasOrigSizes = this.engine.inputNames.includes('orig_target_sizes');
    if (hasOrigSizes) {
      feeds.orig_target_sizes = { data: Float32Array.from([src.width, src.height]), dims: [1, 2] };
    }
    const outputs = await this.engine.run(feeds);
    const heads = this.format === 'deimv2'
      ? this.decodeDeimv2(outputs.label_xyxy_score, src, hasOrigSizes)
      : this.decodeYoloRaw(outputs, src);
    heads.sort((a, b) => b.score - a.score);
    return heads;
  }
}
