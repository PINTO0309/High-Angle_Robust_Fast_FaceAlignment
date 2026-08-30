// 検出器のクラス id・NMS と HRFFA の入力正規化・描画定数。

// DEIMv2-Wholebody49 と YOLOv9-Wholebody34 のどちらの語彙でも class 7 = head
export const HEAD_CLASS_ID = 7;
// YOLO 生出力に対する貪欲 NMS の IoU 閾値
export const NMS_IOU_THRESHOLD = 0.5;

export type InputNorm = 'center05' | 'imagenet';

export const INPUT_NORMS: Record<InputNorm, { mean: [number, number, number]; std: [number, number, number] }> = {
  imagenet: { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] },
  center05: { mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5] },
};

// モデル名から入力正規化を推定: vitl(教師)= imagenet、学生(vitt / hg0)= center05
export function inferInputNorm(modelName: string): InputNorm {
  const tokens = modelName.toLowerCase().split(/[^a-z0-9]+/).filter((t) => t.length > 0);
  return tokens.includes('vitl') ? 'imagenet' : 'center05';
}

// 予測のみ・単色(プロジェクトのレンダリング規約)
export const DRAW_COLOR = '#00ff00';

// ibug68 の連結(閉じる輪郭は最後に先頭へ戻す)
function range(a: number, b: number): number[] {
  const out: number[] = [];
  for (let i = a; i < b; i += 1) {
    out.push(i);
  }
  return out;
}

export const IBUG68_CHAINS: ReadonlyArray<{ idx: number[]; closed: boolean }> = [
  { idx: range(0, 17), closed: false }, // 顎
  { idx: range(17, 22), closed: false }, // 右眉(画像左)
  { idx: range(22, 27), closed: false }, // 左眉
  { idx: range(27, 31), closed: false }, // 鼻筋
  { idx: range(31, 36), closed: false }, // 鼻下
  { idx: range(36, 42), closed: true }, // 右目
  { idx: range(42, 48), closed: true }, // 左目
  { idx: range(48, 60), closed: true }, // 外唇
  { idx: range(60, 68), closed: true }, // 内唇
];
