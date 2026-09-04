// 頭の向き推定。既定は YawNet(SynthYaw の自作アーキテクチャ、center05 正規化・64 直接リサイズ・yawpose 規約)。
// BiternionNet 論文由来の旧モデル(46 入力、RGB [0,1]、正規化なし)にも対応する。
// 出力 [N, 2] = 単位 biternion (cos θ, sin θ)。前処理は元実装と同じく頭部 bbox を 50×50 に縮小してから中央 46×46。
// θ = atan2(sin, cos) [deg] は画像面内の向き: 0 = カメラ側(画像の下)、90 = 右、180 = 奥(上)、270 = 左。

import { create2dContext, type Any2DContext } from '../runtime/canvas';
import type { OrtModel } from '../runtime/engine';
import type { FrameSource, HeadBox } from './types';

export class HeadOrientation {
  readonly size: number;
  readonly dynamicBatch: boolean;
  private engine: OrtModel;
  private ctx: Any2DContext;
  private big: number;
  private margin: number;

  private yawnet: boolean;

  constructor(engine: OrtModel, modelName = '') {
    const dims = engine.inputDims;
    if (dims.length !== 4 || dims[1] !== 3 || dims[2] <= 0 || dims[2] !== dims[3]) {
      throw new Error(`Orientation model input must be [N, 3, S, S] with fixed S, got [${dims.join(', ')}]`);
    }
    this.engine = engine;
    this.size = dims[2];
    // 46 入力は元実装どおり 50×50 → 中央 46×46、それ以外(64 など)は直接リサイズ(resize_size == input_size で学習)
    this.margin = this.size === 46 ? 2 : 0;
    this.big = this.size + 2 * this.margin;
    this.dynamicBatch = dims[0] < 0;
    // SynthYaw YawNet: center05 正規化、出力は yawpose 規約(+90 = 画面左)→ 円環規約へ鏡像変換
    this.yawnet = /yawnet/i.test(modelName);
    this.ctx = create2dContext(this.big, this.big);
    // 頭部 bbox → 50×50 の縮小は品質重視(Python 側の Lanczos に近づける)
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.imageSmoothingQuality = 'high';
  }

  private crop(src: FrameSource, box: HeadBox, dst: Float32Array, offset: number): void {
    const S = this.size;
    const ctx = this.ctx;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, this.big, this.big);
    // bbox 全体を 50×50 に引き伸ばし(元実装の scale_all と同じ、縦横比は保たない)、中央 46×46 を読む
    src.drawCrop(ctx, box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1, this.big, this.big);
    const rgba = ctx.getImageData(this.margin, this.margin, S, S).data;
    const n = S * S;
    for (let i = 0; i < n; i += 1) {
      if (this.yawnet) {
        dst[offset + i] = (rgba[i * 4] / 255 - 0.5) / 0.5;
        dst[offset + n + i] = (rgba[i * 4 + 1] / 255 - 0.5) / 0.5;
        dst[offset + 2 * n + i] = (rgba[i * 4 + 2] / 255 - 0.5) / 0.5;
      } else {
        dst[offset + i] = rgba[i * 4] / 255;
        dst[offset + n + i] = rgba[i * 4 + 1] / 255;
        dst[offset + 2 * n + i] = rgba[i * 4 + 2] / 255;
      }
    }
  }

  // 頭部ごとの θ [deg]
  async estimate(src: FrameSource, heads: HeadBox[]): Promise<number[]> {
    if (heads.length === 0) {
      return [];
    }
    const S = this.size;
    const per = 3 * S * S;
    const batch = new Float32Array(per * heads.length);
    heads.forEach((box, i) => this.crop(src, box, batch, i * per));
    const inputName = this.engine.inputNames[0];
    const outputName = this.engine.outputNames[0];
    let data: Float32Array;
    if (this.dynamicBatch) {
      const out = await this.engine.run({ [inputName]: { data: batch, dims: [heads.length, 3, S, S] } });
      data = out[outputName].data;
    } else {
      const parts: Float32Array[] = [];
      for (let i = 0; i < heads.length; i += 1) {
        const out = await this.engine.run({ [inputName]: { data: batch.slice(i * per, (i + 1) * per), dims: [1, 3, S, S] } });
        parts.push(out[outputName].data);
      }
      data = new Float32Array(heads.length * 2);
      parts.forEach((p, i) => data.set(p.subarray(0, 2), i * 2));
    }
    return heads.map((_, i) => {
      const deg = ((Math.atan2(data[i * 2 + 1], data[i * 2]) * 180) / Math.PI + 360) % 360;
      return this.yawnet ? (360 - deg) % 360 : deg;
    });
  }
}
