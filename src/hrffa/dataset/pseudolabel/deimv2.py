"""DEIMv2-Wholebody49 (Apache-2.0) ONNX 推論ラッパー。

前処理はデモ実装と同一仕様(640x640 直リサイズ・BGR→RGB・/255、正規化なし。
正規化なしは実測比較でも head スコアが高いことを確認済み: history/003)。
出力 bbox は正規化座標 [0,1] のため元画像サイズを掛けて絶対座標へ戻す。

D2 と合成画像 QA で使うクラスのみ保持する:
  0=body, 7=head, 8-15=8方向(front..left_front), 16=face,
  17=eye, 18=nose, 19=mouth, 20=ear
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

try:  # venv では torch を先に import して CUDA/cuDNN を ORT から見えるようにする
    import torch  # noqa: F401
except ImportError:
    pass
import onnxruntime as ort

INPUT_SIZE = 640

CLASS_BODY = 0
CLASS_HEAD = 7
DIR8_CLASSES = {
    8: "front", 9: "right_front", 10: "right_side", 11: "right_back",
    12: "back", 13: "left_back", 14: "left_side", 15: "left_front",
}
CLASS_FACE = 16
KEEP_CLASSES = frozenset([
    CLASS_BODY, CLASS_HEAD, *DIR8_CLASSES.keys(), CLASS_FACE, 17, 18, 19, 20,
])


class Deimv2Detector:
    def __init__(self, model_path: Path, providers: list | None = None,
                 score_threshold: float = 0.35):
        self.score_threshold = score_threshold
        # postprocessor の Gather が要求するバッファはバッチ内容依存で数 GB まで
        # 変動し OOM が散発するが、_run_with_oom_fallback がバッチ分割で回復する
        # ため既定アリーナのままとする(kSameAsRequested は 11 img/s まで低下)。
        # OOM 時のカーネルエラーログは冗長なので fatal のみ表示にする。
        self._model_path = str(model_path)
        self._providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = self._make_session(self._providers)
        self._cpu_session: ort.InferenceSession | None = None

    def _make_session(self, providers: list) -> ort.InferenceSession:
        so = ort.SessionOptions()
        so.log_severity_level = 4
        return ort.InferenceSession(self._model_path, sess_options=so,
                                    providers=providers)

    @staticmethod
    def preprocess(image_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image_bgr, (INPUT_SIZE, INPUT_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb.transpose(2, 0, 1).astype(np.float32) / 255.0

    def infer_batch(self, images_bgr: list[np.ndarray]) -> list[list[list[float]]]:
        """画像リスト → 画像ごとの検出リスト [[cls, x1, y1, x2, y2, score], ...]。

        座標は各画像の絶対ピクセル。KEEP_CLASSES かつ score_threshold 以上のみ。
        GPU OOM 時はバッチを半分に割って再帰的に再試行する。
        """
        with ThreadPoolExecutor(max_workers=8) as ex:
            batch = np.stack(list(ex.map(self.preprocess, images_bgr)))
        out = self._run_with_oom_fallback(batch)
        assert len(out) == len(images_bgr)
        return self._postprocess(out, images_bgr)

    def _run_with_oom_fallback(self, batch: np.ndarray) -> np.ndarray:
        try:
            (out,) = self.session.run(["label_xyxy_score"], {"images": batch})
            return out
        except Exception as e:  # noqa: BLE001 - ORT は RuntimeException を投げる
            if "Failed to allocate memory" not in str(e):
                raise
            if len(batch) > 1:
                half = len(batch) // 2
                return np.concatenate([
                    self._run_with_oom_fallback(batch[:half]),
                    self._run_with_oom_fallback(batch[half:]),
                ])
            # バッチ 1 でも OOM: アリーナ断片化のためセッションを作り直して 1 回
            # 再試行し、それでも駄目ならその画像だけ CPU で処理する
            self.session = None
            self.session = self._make_session(self._providers)
            try:
                (out,) = self.session.run(["label_xyxy_score"], {"images": batch})
                return out
            except Exception as e2:  # noqa: BLE001
                if "Failed to allocate memory" not in str(e2):
                    raise
                if self._cpu_session is None:
                    self._cpu_session = self._make_session(["CPUExecutionProvider"])
                (out,) = self._cpu_session.run(["label_xyxy_score"], {"images": batch})
                return out

    def _postprocess(self, out: np.ndarray,
                     images_bgr: list[np.ndarray]) -> list[list[list[float]]]:
        results: list[list[list[float]]] = []
        for det, img in zip(out, images_bgr):
            h, w = img.shape[:2]
            keep = det[:, 5] >= self.score_threshold
            dets: list[list[float]] = []
            for cls, x1, y1, x2, y2, score in det[keep]:
                cls = int(cls)
                if cls not in KEEP_CLASSES:
                    continue
                dets.append([
                    cls,
                    round(float(np.clip(x1, 0, 1)) * w, 2),
                    round(float(np.clip(y1, 0, 1)) * h, 2),
                    round(float(np.clip(x2, 0, 1)) * w, 2),
                    round(float(np.clip(y2, 0, 1)) * h, 2),
                    round(float(score), 4),
                ])
            results.append(dets)
        return results
