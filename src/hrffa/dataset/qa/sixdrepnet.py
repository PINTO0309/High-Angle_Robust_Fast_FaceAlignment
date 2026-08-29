"""6DRepNet360(全周対応の頭部姿勢推定、ONNX)ラッパー。

前処理は PINTO_model_zoo #423 デモ準拠:
  head bbox を中心 1.2 倍拡張 → クロップ → 256x256 リサイズ → 中央 224x224 →
  RGB / 255 → ImageNet 正規化。出力は度単位の [yaw, pitch, roll]。
入力は [1,3,224,224] 固定のため 1 枚ずつ推論する。

符号規約は 300wlp_val GT との較正で確認する(angle_audit の docstring 参照)。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:  # venv では torch を先に import して CUDA/cuDNN を ORT から見えるようにする
    import torch  # noqa: F401
except ImportError:
    pass
import onnxruntime as ort

_MODEL = Path("data/models/sixdrepnet360_1x3x224x224_full.onnx")
_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


class SixDRepNet360:
    def __init__(self, model_path: Path = _MODEL, providers: list | None = None):
        so = ort.SessionOptions()
        so.log_severity_level = 4
        self.session = ort.InferenceSession(
            str(model_path), sess_options=so,
            providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])

    def infer(self, image_bgr: np.ndarray, head_bbox) -> tuple[float, float, float]:
        """head bbox から (yaw, pitch, roll)[deg] を推定する。"""
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = head_bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ew, eh = (x2 - x1) * 1.2, (y2 - y1) * 1.2
        ex1 = max(int(cx - ew / 2), 0)
        ex2 = min(int(cx + ew / 2), w)
        ey1 = max(int(cy - eh / 2), 0)
        ey2 = min(int(cy + eh / 2), h)
        crop = image_bgr[ey1:ey2, ex1:ex2]
        if crop.size == 0:
            return float("nan"), float("nan"), float("nan")
        resized = cv2.resize(crop, (256, 256))[16:240, 16:240]
        rgb = resized[..., ::-1].astype(np.float32) / 255.0
        x = ((rgb - _MEAN) / _STD).transpose(2, 0, 1)[None]
        (ypr,) = self.session.run(None, {"input": x.astype(np.float32)})
        yaw, pitch, roll = (float(v) for v in ypr[0])
        return yaw, pitch, roll


def pose_meta_for_source(unified: Path, source: str, splits: tuple,
                         model: "SixDRepNet360 | None" = None) -> dict:
    """ソースの全レコードに 6DRepNet 姿勢を付与(サイドカーキャッシュ付き)。

    returns: record_id -> [yaw, pitch, roll](度)
    """
    import json
    cache = unified / "annotations" / f"sixd_pose_{source}_{'_'.join(splits)}.jsonl"
    if cache.exists():
        return {r["record_id"]: r["ypr"] for r in
                (json.loads(l) for l in open(cache, encoding="utf-8"))}
    m = model or SixDRepNet360()
    out = {}
    with open(unified / "annotations" / f"{source}.jsonl", encoding="utf-8") as f, \
            open(cache, "w", encoding="utf-8") as fo:
        for line in f:
            rec = json.loads(line)
            if rec["split"] not in splits:
                continue
            img = cv2.imread(str(unified / rec["image_path"]))
            if img is None:
                continue
            yaw, pitch, roll = m.infer(img, rec["head_bbox"])
            ypr = [round(yaw, 1), round(pitch, 1), round(roll, 1)]
            out[rec["record_id"]] = ypr
            fo.write(json.dumps({"record_id": rec["record_id"], "ypr": ypr}) + "\n")
    return out
