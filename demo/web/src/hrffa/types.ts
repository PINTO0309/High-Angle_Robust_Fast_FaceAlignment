import type { Any2DContext } from '../runtime/canvas';

export interface HeadBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
}

export interface HeadResult {
  box: HeadBox;
  // [K * 2] 元画像座標(px)
  points: Float32Array;
  // [K] 0 = 画像外 / 1 = 遮蔽 / 2 = 可視
  visibility: Uint8Array;
  // 頭部 yaw θ [deg](YawNet、0 = カメラ側 = 画像の下、90 = 右、180 = 奥、270 = 左)。未推定なら undefined
  orientationDeg?: number;
}

export interface FrameStats {
  detectMs: number;
  alignMs: number;
  orientMs: number;
  nHeads: number;
}

export interface FrameOutput {
  heads: HeadResult[];
  stats: FrameStats;
}

// フレームの供給源: 検出器は全体を入力サイズへ引き伸ばして描き、整列器は頭部矩形を切り出して描く。
export interface FrameSource {
  width: number;
  height: number;
  draw: (ctx: Any2DContext, w: number, h: number) => void;
  drawCrop: (ctx: Any2DContext, sx: number, sy: number, sw: number, sh: number, dw: number, dh: number) => void;
}
