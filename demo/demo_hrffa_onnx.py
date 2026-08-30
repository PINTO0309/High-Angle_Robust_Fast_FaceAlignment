"""HRFFA ONNX 推論デモ(頭部検出 → 頭部クロップのランドマーク推定 → 描画)。

2 段構成で、どちらのモデルもコマンドライン引数で差し替えられる:
  1. DEIMv2-Wholebody49(ONNX): 画像を 640×640 へ直リサイズ(BGR→RGB、/255、正規化なし)して
     `label_xyxy_score` [1, Q, 6] = (class, x1, y1, x2, y2, score) を得る。座標は [0,1] 正規化。
     class 7 = head の箱だけを使う。
  2. HRFFA(ONNX): 学習・評価と同じ幾何(head bbox の長辺 × (1 + 2·pad) の正方領域を out_size へ
     相似変換、pad 0.05)でクロップし、`(x/255 − mean)/std` で正規化して `points` [N, K, 2]
     (クロップ比 [0,1])と `vis_logits` [N, K, 3](0=画像外 / 1=遮蔽 / 2=可視)を得る。点はクロップ
     変換の逆行列で元画像座標へ戻す。入力正規化はモデル名から推定(vitl → imagenet、それ以外 → center05)
     し、`--input_norm` で上書きできる。バッチ軸が N のモデルは 1 フレームの全頭部を一括推論する。

描画は本プロジェクトの規約どおり予測のみ・単色(緑)・可視性による色分けなし。

使い方(既定モデル):
  python demo/demo_hrffa_onnx.py -i images_dir -o output_dir
  python demo/demo_hrffa_onnx.py -v 0 -o output_dir               # カメラ 0
  python demo/demo_hrffa_onnx.py -v input.mp4 -o output_dir -d cuda
  python demo/demo_hrffa_onnx.py -am data/models/hrffa_hg0_ibug68_1x3x96x96.onnx -i images -o out

キー操作(動画 / カメラ): ESC 終了、b = 頭部 bbox の表示切替、l = ランドマーク連結線の表示切替。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

try:  # CUDA / cuDNN を先にロードして onnxruntime-gpu から見えるようにする(venv 運用時)
    import torch  # noqa: F401
except ImportError:
    pass
import onnxruntime as ort

DEFAULT_DETECTOR = "data/models/deimv2_dinov3_s_wholebody49_ins_s08_maskhead256x3_center_1240query_masks.onnx"
DEFAULT_ALIGNER = "data/models/hrffa_vitt_ibug68_1x3x256x256.onnx"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HEAD_CLASS_ID = 7
DRAW_COLOR = (0, 255, 0)  # BGR。予測のみ・単色
INPUT_NORMS = {
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "center05": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}
# ibug68 の連結(閉じる輪郭は最後に先頭へ戻す)
IBUG68_CHAINS = (
    (list(range(0, 17)), False),    # 顎
    (list(range(17, 22)), False),   # 右眉(画像左)
    (list(range(22, 27)), False),   # 左眉
    (list(range(27, 31)), False),   # 鼻筋
    (list(range(31, 36)), False),   # 鼻下
    (list(range(36, 42)), True),    # 右目
    (list(range(42, 48)), True),    # 左目
    (list(range(48, 60)), True),    # 外唇
    (list(range(60, 68)), True),    # 内唇
)


@dataclass
class HeadBox:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


@dataclass
class HeadResult:
    box: HeadBox
    points: np.ndarray      # [K, 2] 元画像座標(px)
    visibility: np.ndarray  # [K] 0=画像外 / 1=遮蔽 / 2=可視


# ---------------------------------------------------------------------------
# onnxruntime
# ---------------------------------------------------------------------------
@dataclass
class TrtSettings:
    """TensorRT EP の精度とエンジンキャッシュの置き場所(起動時に 1 回だけ決める)。"""
    precision: str          # fp16 / bf16 / int8
    cache_dir: Path
    ort_version: str
    trt_version: str
    compute_capability: str


def _ort_version_tuple() -> tuple[int, int]:
    major, minor = (int(v) for v in ort.__version__.split(".")[:2])
    return major, minor


def _tensorrt_version() -> str:
    """libnvinfer の getInferLibVersion() から TensorRT の版を得る(tensorrt の Python パッケージは不要)。"""
    import ctypes
    import ctypes.util
    for name in ("nvinfer", "nvinfer.so.10", "nvinfer.so.8"):
        path = ctypes.util.find_library(name) if "." not in name else f"lib{name}"
        if not path:
            continue
        try:
            lib = ctypes.CDLL(path)
            lib.getInferLibVersion.restype = ctypes.c_int32
            v = int(lib.getInferLibVersion())
            return f"{v // 10000}.{(v // 100) % 100}.{v % 100}"
        except OSError:
            continue
    return "unknown"


def _compute_capability() -> str:
    """GPU の compute capability(例 "86")。torch が無ければ "unknown"。"""
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return f"{major}{minor}"
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def resolve_trt_precision(inference_type: str) -> str:
    """`auto`: onnxruntime >= 1.25(trt_bf16_enable あり)かつ Ampere 以降なら bf16、それ以外は fp16。

    明示指定は尊重するが、bf16 は onnxruntime < 1.25 では受理されないのでエラーにする。
    """
    version = _ort_version_tuple()
    bf16_available = version >= (1, 25)
    cc = _compute_capability()
    bf16_hw = cc == "unknown" or int(cc) >= 80
    if inference_type == "auto":
        if not bf16_available:
            print(f"onnxruntime-gpu >= 1.25.0 is recommended for TensorRT (BF16 via trt_bf16_enable); installed {ort.__version__}, "
                  "using fp16. Install the tensorrt dependency group and keep its flags on uv run: "
                  "uv sync --frozen --no-group ort --group tensorrt && uv run --no-group ort --group tensorrt python ...")
            return "fp16"
        if not bf16_hw:
            print(f"GPU compute capability {cc} does not support BF16 in TensorRT (needs >= 8.0); using fp16")
            return "fp16"
        return "bf16"
    if inference_type == "bf16" and not bf16_available:
        raise RuntimeError(f"--inference_type bf16 needs onnxruntime-gpu >= 1.25 (installed {ort.__version__}); "
                           "install the tensorrt dependency group and run with the same flags: "
                           "uv run --no-group ort --group tensorrt python ...")
    return inference_type


def prepare_trt_cache(cache_root: Path, precision: str) -> TrtSettings:
    """エンジンキャッシュを onnxruntime 版 / TensorRT 版 / 精度 / GPU ごとのディレクトリに分け、
    別の onnxruntime 版で作られたキャッシュは削除する(版が変わったエンジンは必ず作り直す)。"""
    import shutil
    ort_version = ort.__version__
    trt_version = _tensorrt_version()
    cc = _compute_capability()
    cache_root.mkdir(parents=True, exist_ok=True)
    prefix = f"ort-{ort_version}_"
    for entry in sorted(cache_root.iterdir()):
        if entry.is_dir() and entry.name.startswith("ort-") and not entry.name.startswith(prefix):
            shutil.rmtree(entry)
            print(f"Removed stale TensorRT engine cache built with another onnxruntime version: {entry}")
    cache_dir = cache_root / f"{prefix}trt-{trt_version}_{precision}_sm{cc}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return TrtSettings(precision=precision, cache_dir=cache_dir, ort_version=ort_version,
                       trt_version=trt_version, compute_capability=cc)


def build_providers(device: str | None, trt: TrtSettings | None = None) -> list:
    """参考デモ(DEIMv2-Wholebody49)と同じ規則で実行プロバイダを組む。"""
    available = set(ort.get_available_providers())
    requested = (device or "").lower()
    if requested.startswith("cuda"):
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider is not available in this onnxruntime build.")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if requested == "tensorrt":
        if "TensorrtExecutionProvider" not in available:
            raise RuntimeError("TensorrtExecutionProvider is not available in this onnxruntime build.")
        if trt is None:
            raise ValueError("TensorRT settings are required for the tensorrt device")
        if trt.precision == "fp16":
            type_params = {"trt_fp16_enable": True}
        elif trt.precision == "bf16":
            type_params = {"trt_bf16_enable": True}
        elif trt.precision == "int8":
            type_params = {"trt_fp16_enable": True, "trt_int8_enable": True,
                           "trt_int8_calibration_table_name": "calibration.flatbuffers"}
        else:
            raise ValueError(f"Unsupported inference type for TensorRT: {trt.precision}")
        providers: list = [(
            "TensorrtExecutionProvider",
            {"trt_engine_cache_enable": True,
             "trt_engine_cache_path": str(trt.cache_dir),
             "trt_timing_cache_enable": True,
             "trt_timing_cache_path": str(trt.cache_dir),
             "trt_op_types_to_exclude": "NonMaxSuppression,NonZero,RoiAlign"} | type_params,
        )]
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers
    if requested and requested != "cpu":
        raise ValueError(f"Unsupported device: {device}. Use cpu, cuda, cuda:0, or tensorrt.")
    if device is None and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


_ORT_LOG_LEVELS = {"verbose": 0, "info": 1, "warning": 2, "error": 3, "fatal": 4}


def set_ort_log_level(level: str) -> None:
    """onnxruntime のグローバル(Default)ロガーの重大度を設定する。

    TensorRT EP がエンジン構築中に出す `[W:onnxruntime:Default, tensorrt_execution_provider…]`
    (Int64 binding、timing cache 未作成など)はセッションの log_severity_level では抑制できず、
    こちらで決まる。既定は error(警告を出さない)。
    """
    ort.set_default_logger_severity(_ORT_LOG_LEVELS[level])


def make_session(model_path: Path, providers: list) -> ort.InferenceSession:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(model_path), sess_options=so, providers=providers)


# ---------------------------------------------------------------------------
# 1 段目: DEIMv2-Wholebody49(頭部検出)
# ---------------------------------------------------------------------------
class HeadDetector:
    def __init__(self, model_path: Path, providers: list, score_threshold: float):
        self.session = make_session(model_path, providers)
        self.providers = self.session.get_providers()
        self.score_threshold = score_threshold
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        if len(shape) != 4 or not all(isinstance(v, int) for v in shape[2:4]):
            raise ValueError(f"Detector input must be [N, 3, H, W] with fixed H/W, got {shape}")
        self.input_hw = (int(shape[2]), int(shape[3]))
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.last_inference_time = 0.0

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = self.input_hw
        resized = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb.transpose(2, 0, 1).astype(np.float32)[None] / 255.0

    def __call__(self, image_bgr: np.ndarray) -> list[HeadBox]:
        img_h, img_w = image_bgr.shape[:2]
        feed = {self.input_name: self.preprocess(image_bgr)}
        if "orig_target_sizes" in self.input_names:
            feed["orig_target_sizes"] = np.array([[img_w, img_h]], dtype=np.float32)
        t0 = time.perf_counter()
        (pred,) = self.session.run(["label_xyxy_score"], feed)
        self.last_inference_time = time.perf_counter() - t0
        pred = pred[0]  # [Q, 6]
        keep = (pred[:, 0].astype(np.int64) == HEAD_CLASS_ID) & (pred[:, 5] >= self.score_threshold)
        heads: list[HeadBox] = []
        for _, x1, y1, x2, y2, score in pred[keep]:
            if "orig_target_sizes" not in self.input_names:  # 正規化座標 → 画素
                x1, x2 = x1 * img_w, x2 * img_w
                y1, y2 = y1 * img_h, y2 * img_h
            x1, x2 = float(np.clip(x1, 0, img_w - 1)), float(np.clip(x2, 0, img_w - 1))
            y1, y2 = float(np.clip(y1, 0, img_h - 1)), float(np.clip(y2, 0, img_h - 1))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            heads.append(HeadBox(x1, y1, x2, y2, float(score)))
        heads.sort(key=lambda b: -b.score)
        return heads


# ---------------------------------------------------------------------------
# 2 段目: HRFFA(頭部クロップのランドマーク)
# ---------------------------------------------------------------------------
def infer_input_norm(model_path: Path) -> str:
    """モデル名から入力正規化を推定する: vitl(教師)= imagenet、学生(vitt / hg0)= center05。"""
    tokens = [t for t in re.split(r"[^a-z0-9]+", model_path.stem.lower()) if t]
    return "imagenet" if "vitl" in tokens else "center05"


def crop_transform(box: HeadBox, out_size: int, pad: float) -> np.ndarray:
    """学習・評価の crop_affine と同じ相似変換(3x3): bbox 中心を出力中心へ、長辺 × (1 + 2·pad) を out_size へ。"""
    cx, cy = (box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0
    side = max(box.x2 - box.x1, box.y2 - box.y1) * (1.0 + 2.0 * pad)
    s = out_size / side
    half = out_size / 2.0
    return np.array([[s, 0.0, half - s * cx],
                     [0.0, s, half - s * cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


class FaceAligner:
    def __init__(self, model_path: Path, providers: list, input_norm: str, crop_pad: float):
        self.session = make_session(model_path, providers)
        self.providers = self.session.get_providers()
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        if len(shape) != 4 or not isinstance(shape[2], int) or shape[2] != shape[3]:
            raise ValueError(f"Alignment model input must be [N, 3, S, S] with fixed S, got {shape}")
        self.out_size = int(shape[2])
        self.dynamic_batch = not isinstance(shape[0], int)
        self.output_names = [o.name for o in self.session.get_outputs()]
        for name in ("points", "vis_logits"):
            if name not in self.output_names:
                raise ValueError(f"Alignment model must output `{name}`, got {self.output_names}")
        self.input_norm = input_norm
        mean, std = INPUT_NORMS[input_norm]
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)
        self.crop_pad = crop_pad
        self.last_inference_time = 0.0

    def preprocess(self, image_bgr: np.ndarray, box: HeadBox) -> tuple[np.ndarray, np.ndarray]:
        T = crop_transform(box, self.out_size, self.crop_pad)
        crop = cv2.warpPerspective(image_bgr, T, (self.out_size, self.out_size),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1) / 255.0
        return (rgb - self.mean) / self.std, T

    def _run(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points, vis_logits = self.session.run(["points", "vis_logits"], {self.input_name: batch})
        return points, vis_logits

    def __call__(self, image_bgr: np.ndarray, heads: Sequence[HeadBox]) -> list[HeadResult]:
        if not heads:
            self.last_inference_time = 0.0
            return []
        crops, transforms = zip(*(self.preprocess(image_bgr, box) for box in heads))
        batch = np.stack(crops).astype(np.float32)
        t0 = time.perf_counter()
        if self.dynamic_batch:
            points, vis_logits = self._run(batch)
        else:
            outs = [self._run(batch[i:i + 1]) for i in range(len(batch))]
            points = np.concatenate([o[0] for o in outs])
            vis_logits = np.concatenate([o[1] for o in outs])
        self.last_inference_time = time.perf_counter() - t0

        results: list[HeadResult] = []
        for box, T, pts, vl in zip(heads, transforms, points, vis_logits):
            crop_xy = pts.astype(np.float64) * self.out_size          # クロップ画素座標
            homo = np.concatenate([crop_xy, np.ones((len(crop_xy), 1))], axis=1)
            img_xy = (np.linalg.inv(T) @ homo.T).T[:, :2]              # 元画像座標
            results.append(HeadResult(box=box, points=img_xy.astype(np.float32),
                                      visibility=vl.argmax(axis=1).astype(np.int64)))
        return results


# ---------------------------------------------------------------------------
# 描画・保存
# ---------------------------------------------------------------------------
def draw_results(image: np.ndarray, results: Sequence[HeadResult], draw_bbox: bool,
                 draw_lines: bool, point_radius: int | None) -> np.ndarray:
    out = image.copy()
    for r in results:
        side = max(r.box.x2 - r.box.x1, r.box.y2 - r.box.y1)
        radius = point_radius if point_radius is not None else max(1, int(round(side / 96.0)))
        thickness = max(1, int(round(side / 160.0)))
        if draw_bbox:
            cv2.rectangle(out, (int(round(r.box.x1)), int(round(r.box.y1))),
                          (int(round(r.box.x2)), int(round(r.box.y2))), DRAW_COLOR, thickness, cv2.LINE_AA)
        if draw_lines and len(r.points) == 68:
            for chain, closed in IBUG68_CHAINS:
                idx = chain + ([chain[0]] if closed else [])
                pts = np.round(r.points[idx]).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(out, [pts], False, DRAW_COLOR, max(1, thickness // 2), cv2.LINE_AA)
        for x, y in r.points:
            cv2.circle(out, (int(round(x)), int(round(y))), radius, DRAW_COLOR, -1, cv2.LINE_AA)
    return out


def put_text(image: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.7, thickness: int = 2) -> None:
    """黒帯の上に白文字を 1 回だけ描く。

    「太い白文字の上に細い赤文字を重ねて縁取りにする」方式は OpenCV 5 の文字描画では線の太さで
    文字送りが変わる(同じ文字列でも幅が 138 px と 146 px になる)ため 2 つの文字列がずれて見える。
    """
    x, y = org
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(image, (x - 4, y - th - 4), (x + tw + 4, y + base + 2), (0, 0, 0), -1)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def results_to_records(results: Sequence[HeadResult]) -> list[dict]:
    return [{
        "head_bbox": [round(r.box.x1, 2), round(r.box.y1, 2), round(r.box.x2, 2), round(r.box.y2, 2)],
        "score": round(r.box.score, 4),
        "points": [[round(float(x), 2), round(float(y), 2)] for x, y in r.points],
        "visibility": [int(v) for v in r.visibility],
    } for r in results]


def save_records(output_dir: Path, stem: str, results: Sequence[HeadResult]) -> None:
    (output_dir / f"{stem}.json").write_text(
        json.dumps({"heads": results_to_records(results)}, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------
class Pipeline:
    def __init__(self, args: argparse.Namespace):
        set_ort_log_level(args.ort_log_level)
        det_path, aln_path = Path(args.detector_model), Path(args.alignment_model)
        self.trt: TrtSettings | None = None
        if (args.device or "").lower() == "tensorrt":
            self.trt = prepare_trt_cache(Path(args.trt_cache_dir), resolve_trt_precision(args.inference_type))
            print(f"TensorRT: precision {self.trt.precision} (onnxruntime {self.trt.ort_version}, TensorRT {self.trt.trt_version}, "
                  f"sm{self.trt.compute_capability}), engine cache {self.trt.cache_dir}")
        providers = build_providers(args.device, self.trt)
        self.detector = HeadDetector(det_path, providers, args.head_score_threshold)
        input_norm = args.input_norm if args.input_norm != "auto" else infer_input_norm(aln_path)
        self.aligner = FaceAligner(aln_path, providers, input_norm, args.crop_pad)
        self.draw_bbox = not args.disable_bbox
        self.draw_lines = args.draw_lines
        self.point_radius = args.point_radius
        self.warm_up()

    def warm_up(self) -> None:
        """初回実行のカーネル選択・メモリ確保を計測から外すため、ダミー入力で両セッションを 1 回走らせる。"""
        h, w = self.detector.input_hw
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        self.detector(dummy)
        s = self.aligner.out_size
        self.aligner(np.zeros((s, s, 3), dtype=np.uint8), [HeadBox(0.0, 0.0, float(s - 1), float(s - 1), 1.0)])

    def run(self, image: np.ndarray) -> tuple[np.ndarray, list[HeadResult], float, float]:
        t0 = time.perf_counter()
        heads = self.detector(image)
        results = self.aligner(image, heads)
        infer_ms = (self.detector.last_inference_time + self.aligner.last_inference_time) * 1000.0
        rendered = draw_results(image, results, self.draw_bbox, self.draw_lines, self.point_radius)
        total_ms = (time.perf_counter() - t0) * 1000.0
        return rendered, results, infer_ms, total_ms


def list_image_paths(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def is_int(value: str) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


def create_video_writer(output_dir: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    safe_fps = fps if fps and math.isfinite(fps) and fps > 0 else 30.0
    return cv2.VideoWriter(str(output_dir / "output.mp4"), cv2.VideoWriter.fourcc(*"mp4v"), safe_fps, (width, height))


def process_images(pipeline: Pipeline, images_dir: Path, output_dir: Path, save_raw: bool) -> None:
    paths = list_image_paths(images_dir)
    if not paths:
        raise FileNotFoundError(f"No image files found in {images_dir}")
    print(f"Processing {len(paths)} images from {images_dir}")
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        rendered, results, infer_ms, total_ms = pipeline.run(image)
        cv2.imwrite(str(output_dir / path.name), rendered)
        if save_raw:
            save_records(output_dir, path.stem, results)
        print(f"{path.name}: heads={len(results)} infer={infer_ms:.1f} ms total={total_ms:.1f} ms")


def process_video(pipeline: Pipeline, video: str, output_dir: Path, save_raw: bool,
                  disable_video_writer: bool, disable_imshow: bool) -> None:
    cap = cv2.VideoCapture(int(video) if is_int(video) else video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {video}")
    if is_int(video):  # カメラ入力は VGA(640×480)に固定する
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"Processing video source: {video}")
    writer = None
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            rendered, results, infer_ms, total_ms = pipeline.run(frame)
            put_text(rendered, f"infer: {infer_ms:.2f} ms", (10, 30))
            put_text(rendered, f"total: {total_ms:.2f} ms", (10, 58))
            put_text(rendered, f"heads: {len(results)}", (10, 86))
            if writer is None and not disable_video_writer:
                writer = create_video_writer(output_dir, rendered.shape[1], rendered.shape[0], cap.get(cv2.CAP_PROP_FPS))
            if writer is not None:
                writer.write(rendered)
            if save_raw:
                save_records(output_dir, f"{frame_index:08d}", results)
            if not disable_imshow:
                cv2.imshow("HRFFA ONNX demo", rendered)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                if key == ord("b"):
                    pipeline.draw_bbox = not pipeline.draw_bbox
                elif key == ord("l"):
                    pipeline.draw_lines = not pipeline.draw_lines
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not disable_imshow:
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="HRFFA ONNX demo: DEIMv2-Wholebody49 head detection followed by HRFFA landmark estimation on each head crop.")
    ap.add_argument("-dm", "--detector_model", type=str, default=DEFAULT_DETECTOR,
                    help="DEIMv2-Wholebody49 ONNX (outputs label_xyxy_score; class 7 = head)")
    ap.add_argument("-am", "--alignment_model", type=str, default=DEFAULT_ALIGNER,
                    help="HRFFA ONNX (images [N,3,S,S] -> points, vis_logits); fixed-batch and N-batch graphs are both accepted")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("-i", "--images_dir", type=str, help="directory of input images")
    src.add_argument("-v", "--video", type=str, help="video file path or camera index")
    ap.add_argument("-o", "--output_dir", type=str, default="output")
    ap.add_argument("-d", "--device", type=str, default=None, help="cpu, cuda, cuda:0 or tensorrt (default: cuda if available)")
    ap.add_argument("--inference_type", type=str, choices=["auto", "fp16", "bf16", "int8"], default="auto",
                    help="TensorRT precision; auto = bf16 with onnxruntime-gpu >= 1.25 on Ampere or newer GPUs, otherwise fp16")
    ap.add_argument("--ort_log_level", type=str, choices=list(_ORT_LOG_LEVELS), default="error",
                    help="onnxruntime global log level (default: error, which hides the TensorRT engine-build warnings)")
    ap.add_argument("--trt_cache_dir", type=str, default="data/models/trt_cache",
                    help="TensorRT engine cache root; engines are kept per onnxruntime / TensorRT version, precision and GPU, "
                         "and caches built with another onnxruntime version are deleted at startup")
    ap.add_argument("--head_score_threshold", type=float, default=0.50)
    ap.add_argument("--crop_pad", type=float, default=0.05, help="head-bbox margin ratio of the square crop (training value: 0.05)")
    ap.add_argument("--input_norm", type=str, choices=["auto", "center05", "imagenet"], default="auto",
                    help="input normalization of the alignment model (auto: vitl -> imagenet, others -> center05)")
    ap.add_argument("--draw_lines", action="store_true", help="draw the ibug68 contour lines")
    ap.add_argument("--disable_bbox", action="store_true", help="do not draw head boxes")
    ap.add_argument("--point_radius", type=int, default=None, help="landmark radius in px (default: scaled by head size)")
    ap.add_argument("--disable_video_writer", action="store_true")
    ap.add_argument("--disable_imshow", action="store_true", help="do not open a window for video / camera input")
    ap.add_argument("--save_raw_predictions", action="store_true", help="save head boxes, points and visibility as JSON")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(args)
    print(f"Detector: {args.detector_model} (input {pipeline.detector.input_hw[1]}x{pipeline.detector.input_hw[0]}, "
          f"providers {pipeline.detector.providers})")
    print(f"Aligner: {args.alignment_model} (input {pipeline.aligner.out_size}x{pipeline.aligner.out_size}, "
          f"{'N-batch' if pipeline.aligner.dynamic_batch else 'batch 1'}, norm {pipeline.aligner.input_norm}, "
          f"crop pad {pipeline.aligner.crop_pad}, providers {pipeline.aligner.providers})")
    print(f"Output directory: {output_dir}")
    if args.images_dir is not None:
        images_dir = Path(args.images_dir)
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {images_dir}")
        process_images(pipeline, images_dir, output_dir, args.save_raw_predictions)
    else:
        process_video(pipeline, args.video, output_dir, args.save_raw_predictions,
                      args.disable_video_writer, args.disable_imshow)


if __name__ == "__main__":
    main()
