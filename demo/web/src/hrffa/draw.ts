// canvas への描画: 頭部 bbox とランドマーク(予測のみ・単色緑・可視性による色分けなし)。

import { DRAW_COLOR, IBUG68_CHAINS } from './constants';
import type { HeadResult } from './types';

export interface DrawOptions {
  bbox: boolean;
  lines: boolean;
  pointRadius: number | null;
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
