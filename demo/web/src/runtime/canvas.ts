// メインスレッド(HTMLCanvasElement)と Worker(OffscreenCanvas)の両方で使える 2D canvas ヘルパ。

export type Any2DContext = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;

export function create2dContext(width: number, height: number): Any2DContext {
  if (typeof document !== 'undefined') {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) {
      throw new Error('2D canvas context unavailable');
    }
    return ctx;
  }
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    throw new Error('2D canvas context unavailable');
  }
  return ctx as OffscreenCanvasRenderingContext2D;
}
