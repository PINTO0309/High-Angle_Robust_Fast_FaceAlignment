// canvas への描画: 頭部 bbox とランドマーク(予測のみ・単色緑・可視性による色分けなし)。

import { DRAW_COLOR, IBUG68_CHAINS } from './constants';
import type { HeadResult } from './types';

export interface DrawOptions {
  bbox: boolean;
  lines: boolean;
  pointRadius: number | null;
  // 頭部 yaw(YawNet): 右上の円環(最大の頭部)
  orientation: boolean;
}

const RING_COLOR = 'rgba(230, 230, 230, 0.95)';

// θ [deg] → 画像座標の単位ベクトル。0 = 下(カメラ側)、90 = 右、180 = 上、270 = 左
function orientationVector(deg: number): [number, number] {
  const rad = (deg * Math.PI) / 180;
  return [Math.sin(rad), Math.cos(rad)];
}

// 円環が追う頭部 = 向き推定のある頭部のうち bbox 面積が最大のもの
export function pickPrimaryOriented(heads: HeadResult[]): HeadResult | null {
  const withAngle = heads.filter((h) => h.orientationDeg !== undefined);
  if (withAngle.length === 0) {
    return null;
  }
  return withAngle.reduce((a, b) => ((b.box.x2 - b.box.x1) * (b.box.y2 - b.box.y1) > (a.box.x2 - a.box.x1) * (a.box.y2 - a.box.y1) ? b : a));
}

// ringDeg を渡すと扇形と中央の数字にその角度(平滑化済みの表示用角度)を使う
export function drawOrientation(ctx: CanvasRenderingContext2D, heads: HeadResult[], width: number, height: number, ringDeg?: number): void {
  const primary = pickPrimaryOriented(heads);
  if (primary === null) {
    return;
  }
  const deg = ringDeg ?? (primary.orientationDeg as number);
  ctx.save();
  // 右上の円環: 最大の頭部の向き(頭部ごとの矢印は描かない)。canvas の角度は +x から +y(下)へ正 → φ = 90° − θ
  const radius = Math.max(28, Math.round(Math.min(width, height) * 0.09));
  const band = Math.max(6, Math.round(radius / 4));
  // 外側のラベル(半径 + band + 9 の位置、10px フォント)が画像端で切れない余白
  const margin = band + 20;
  const cx = width - radius - margin;
  const cy = radius + margin;
  ctx.lineWidth = band;
  ctx.strokeStyle = 'rgba(40, 40, 40, 0.6)';
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();
  ctx.lineWidth = 1;
  ctx.strokeStyle = RING_COLOR;
  for (const r of [radius - band / 2, radius + band / 2]) {
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.fillStyle = RING_COLOR;
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (const [theta, label] of [[0, '0'], [90, '90'], [180, '180'], [270, '270']] as const) {
    const [dx, dy] = orientationVector(theta);
    ctx.beginPath();
    ctx.moveTo(cx + dx * (radius - band), cy + dy * (radius - band));
    ctx.lineTo(cx + dx * (radius + band / 2 + 2), cy + dy * (radius + band / 2 + 2));
    ctx.stroke();
    ctx.fillText(label, cx + dx * (radius + band + 9), cy + dy * (radius + band + 9));
  }
  const phi = ((90 - deg) * Math.PI) / 180;
  ctx.strokeStyle = DRAW_COLOR;
  ctx.lineWidth = band;
  ctx.lineCap = 'butt';
  ctx.beginPath();
  ctx.arc(cx, cy, radius, phi - (9 * Math.PI) / 180, phi + (9 * Math.PI) / 180);
  ctx.stroke();
  ctx.fillStyle = DRAW_COLOR;
  ctx.font = 'bold 13px sans-serif';
  ctx.fillText(`${Math.round(deg)}°`, cx, cy);
  ctx.restore();
}

export function drawHeads(ctx: CanvasRenderingContext2D, heads: HeadResult[], opts: DrawOptions): void {
  ctx.save();
  ctx.strokeStyle = DRAW_COLOR;
  ctx.fillStyle = DRAW_COLOR;
  ctx.lineJoin = 'round';
  for (const h of heads) {
    const side = Math.max(h.box.x2 - h.box.x1, h.box.y2 - h.box.y1);
    const radius = opts.pointRadius ?? Math.max(1, Math.round(side / 96));
    const thickness = Math.max(1, Math.round(side / 160));
    if (opts.bbox) {
      ctx.lineWidth = thickness;
      ctx.strokeRect(h.box.x1, h.box.y1, h.box.x2 - h.box.x1, h.box.y2 - h.box.y1);
    }
    const K = h.points.length / 2;
    if (opts.lines && K === 68) {
      ctx.lineWidth = Math.max(1, thickness / 2);
      for (const chain of IBUG68_CHAINS) {
        ctx.beginPath();
        chain.idx.forEach((k, j) => {
          const x = h.points[k * 2];
          const y = h.points[k * 2 + 1];
          if (j === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        });
        if (chain.closed) {
          ctx.closePath();
        }
        ctx.stroke();
      }
    }
    for (let k = 0; k < K; k += 1) {
      ctx.beginPath();
      ctx.arc(h.points[k * 2], h.points[k * 2 + 1], radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
}
