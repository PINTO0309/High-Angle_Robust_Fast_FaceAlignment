// HRFFA(ONNX)による頭部クロップのランドマーク推定。学習・評価と同じ幾何:
// head bbox の中心を出力中心へ、長辺 × (1 + 2·pad) の正方領域を S×S へ相似変換(pad 0.05)。
// 入力は (x/255 − mean)/std、出力 `points` [N, K, 2] はクロップ比 [0,1]、`vis_logits` [N, K, 3]。
// canvas の drawImage は元画像外へはみ出す矩形を比例的に切り詰めるので、黒で塗った上に描けば
// cv2.warpPerspective(BORDER_CONSTANT 0)と同じ結果になる。

import { INPUT_NORMS, type InputNorm } from './constants';
import { create2dContext, type Any2DContext } from '../runtime/canvas';
import type { OrtModel, TensorOut } from '../runtime/engine';
import type { FrameSource, HeadBox, HeadResult } from './types';

interface CropGeometry {
  scale: number;
  cx: number;
  cy: number;
}

function concat(parts: Float32Array[]): Float32Array {
  let n = 0;
  for (const p of parts) {
    n += p.length;
  }
  const out = new Float32Array(n);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

export class FaceAligner {
  readonly size: number;
  readonly dynamicBatch: boolean;
  readonly inputNorm: InputNorm;
  cropPad: number;
  private engine: OrtModel;
  private ctx: Any2DContext;
  private mean: [number, number, number];
  private std: [number, number, number];

  constructor(engine: OrtModel, inputNorm: InputNorm, cropPad: number) {
    const dims = engine.inputDims;
    if (dims.length !== 4 || dims[1] !== 3 || dims[2] <= 0 || dims[2] !== dims[3]) {
      throw new Error(`Alignment model input must be [N, 3, S, S] with fixed S, got [${dims.join(', ')}]`);
    }
    for (const name of ['points', 'vis_logits']) {
      if (!engine.outputNames.includes(name)) {
        throw new Error(`Alignment model must output '${name}' (outputs: ${engine.outputNames.join(', ')})`);
      }
    }
    this.engine = engine;
    this.size = dims[2];
    this.dynamicBatch = dims[0] < 0;
    this.inputNorm = inputNorm;
    this.cropPad = cropPad;
    this.mean = INPUT_NORMS[inputNorm].mean;
    this.std = INPUT_NORMS[inputNorm].std;
    this.ctx = create2dContext(this.size, this.size);
  }

  private crop(src: FrameSource, box: HeadBox, dst: Float32Array, offset: number): CropGeometry {
    const S = this.size;
    const cx = (box.x1 + box.x2) / 2;
    const cy = (box.y1 + box.y2) / 2;
    const side = Math.max(box.x2 - box.x1, box.y2 - box.y1) * (1 + 2 * this.cropPad);
    const ctx = this.ctx;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, S, S);
    src.drawCrop(ctx, cx - side / 2, cy - side / 2, side, side, S, S);
    const rgba = ctx.getImageData(0, 0, S, S).data;
    const n = S * S;
    const [m0, m1, m2] = this.mean;
    const [s0, s1, s2] = this.std;
    for (let i = 0; i < n; i += 1) {
      dst[offset + i] = (rgba[i * 4] / 255 - m0) / s0;
      dst[offset + n + i] = (rgba[i * 4 + 1] / 255 - m1) / s1;
      dst[offset + 2 * n + i] = (rgba[i * 4 + 2] / 255 - m2) / s2;
    }
    return { scale: S / side, cx, cy };
  }

  async align(src: FrameSource, heads: HeadBox[]): Promise<HeadResult[]> {
    if (heads.length === 0) {
      return [];
    }
    const S = this.size;
    const per = 3 * S * S;
    const batch = new Float32Array(per * heads.length);
    const geometry = heads.map((box, i) => this.crop(src, box, batch, i * per));
    const inputName = this.engine.inputNames[0];

    let points: TensorOut;
    let vis: TensorOut;
    if (this.dynamicBatch) {
      const out = await this.engine.run({ [inputName]: { data: batch, dims: [heads.length, 3, S, S] } });
      points = out.points;
      vis = out.vis_logits;
    } else {
      const pts: Float32Array[] = [];
      const vs: Float32Array[] = [];
      let k = 0;
      let c = 0;
      for (let i = 0; i < heads.length; i += 1) {
        const out = await this.engine.run({
          [inputName]: { data: batch.slice(i * per, (i + 1) * per), dims: [1, 3, S, S] },
        });
        pts.push(out.points.data);
        vs.push(out.vis_logits.data);
        k = out.points.dims[1];
        c = out.vis_logits.dims[2];
      }
      points = { data: concat(pts), dims: [heads.length, k, 2] };
      vis = { data: concat(vs), dims: [heads.length, k, c] };
    }

    const K = points.dims[1];
    const C = vis.dims[2];
    return heads.map((box, i) => {
      const { scale, cx, cy } = geometry[i];
      const p = new Float32Array(K * 2);
      const v = new Uint8Array(K);
      for (let k = 0; k < K; k += 1) {
        const px = points.data[(i * K + k) * 2] * S;
        const py = points.data[(i * K + k) * 2 + 1] * S;
        p[k * 2] = (px - S / 2) / scale + cx;
        p[k * 2 + 1] = (py - S / 2) / scale + cy;
        let best = 0;
        let bestValue = Number.NEGATIVE_INFINITY;
        for (let c = 0; c < C; c += 1) {
          const value = vis.data[(i * K + k) * C + c];
          if (value > bestValue) {
            bestValue = value;
            best = c;
          }
        }
        v[k] = best;
      }
      return { box, points: p, visibility: v };
    });
  }
}
