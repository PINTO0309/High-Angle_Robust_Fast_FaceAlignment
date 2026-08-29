"""D4b: 単眼深度(Depth Anything V2 small, Apache-2.0)による 3D 回転再投影。

head crop(apply_geometric の出力)に対して:
  1. DA-v2 で相対逆深度(disparity)を推定
  2. crop 内のロバスト百分位で擬似メトリック深度 z へ変換
     (中央値 → Z0 = focal、広がり → extent_ratio * out_size)
  3. 頭部中心 C = (0, 0, Z0) 回りにカメラをオービット回転(X' = R(X−C)+C)
     → 頭部が出力中心に保たれ、姿勢 GT は R' = R_cam @ R で厳密更新
  4. 2 パス warp: z を前方スプラット(z-buffer)→ 穴埋め → 逆方向 remap
  5. ランドマークは各点の深度で 3D に持ち上げて閉形式で再投影。
     z-buffer と比較して自己遮蔽になった点は可視性 1(遮蔽)へ更新

ホモグラフィ版(geometric.py)との違い: 深度による視差で「見た目も」回転する
(純カメラ回転はラベルのみ厳密で見た目は元視点の再投影に留まる)。
誤差源は深度推定のみ。disocclusion(裏側)は生成できず引き伸ばしになるため
実用域は cam_pitch ±40〜60° 程度。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:  # venv では torch を先に import して CUDA/cuDNN を ORT から見えるようにする
    import torch  # noqa: F401
except ImportError:
    pass
import onnxruntime as ort

from ..geometry import _rx, _ry

_DEFAULT_MODEL = Path("data/models/depth_anything_v2_small.onnx")
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class DepthAnythingV2:
    """ONNX 版 DA-v2 の薄いラッパー(相対逆深度を入力解像度で返す)。"""

    def __init__(self, model_path: Path = _DEFAULT_MODEL, infer_size: int = 392,
                 providers: list | None = None):
        self.infer_size = infer_size  # 14 の倍数であること
        so = ort.SessionOptions()
        so.log_severity_level = 4
        self.session = ort.InferenceSession(
            str(model_path), sess_options=so,
            providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])

    def infer(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = cv2.resize(rgb, (self.infer_size, self.infer_size),
                       interpolation=cv2.INTER_CUBIC)
        x = ((x - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1)[None]
        (d,) = self.session.run(None, {"pixel_values": x.astype(np.float32)})
        return cv2.resize(d[0], (w, h), interpolation=cv2.INTER_LINEAR)


@dataclass
class DepthWarpParams:
    cam_pitch_deg: float = 0.0
    cam_yaw_deg: float = 0.0
    focal_ratio: float = 1.2
    extent_ratio: float = 0.6    # 深度の広がり(out_size 比)
    occl_tol_ratio: float = 0.03  # 自己遮蔽判定の z 許容(Z0 比)


def disparity_to_z(disp: np.ndarray, out_size: int, p: DepthWarpParams) -> np.ndarray:
    """相対逆深度 → 擬似メトリック深度(px 単位、カメラ z 正方向)。"""
    z0 = p.focal_ratio * out_size
    d5, d95 = np.percentile(disp, [5, 95])
    med = float(np.median(disp))
    spread = max(float(d95 - d5), 1e-6)
    z = z0 - (disp - med) / spread * (p.extent_ratio * out_size)
    return np.clip(z, 0.3 * z0, 2.5 * z0).astype(np.float64)


def _fill_holes(arr: np.ndarray, hole: np.ndarray, iters: int = 64) -> np.ndarray:
    """マスク領域を近傍平均の反復で充填する(チャネル任意の float 配列)。"""
    out = arr.copy()
    valid = (~hole).astype(np.float32)
    for _ in range(iters):
        if not hole.any():
            break
        k = np.ones((3, 3), np.float32)
        # 穴は NaN のことがあるため必ず 0 化してから重み付き平均を取る
        safe = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        num = cv2.filter2D(safe * (valid[..., None] if out.ndim == 3 else valid),
                           -1, k, borderType=cv2.BORDER_REPLICATE)
        den = cv2.filter2D(valid, -1, k, borderType=cv2.BORDER_REPLICATE)
        fill = num / np.maximum(den[..., None] if out.ndim == 3 else den, 1e-6)
        newly = hole & (den > 0)
        if out.ndim == 3:
            out[newly] = fill[newly]
        else:
            out[newly] = fill[newly]
        valid[newly] = 1.0
        hole = hole & ~newly
    return out


def depth_reproject(
    crop_bgr: np.ndarray,
    disp: np.ndarray,
    points: np.ndarray,
    visibility: list[int],
    rotation: np.ndarray | None,
    p: DepthWarpParams,
) -> dict:
    """深度つき 3D 回転再投影。crop は正方(out_size)前提。"""
    s = crop_bgr.shape[0]
    z0 = p.focal_ratio * s
    f = z0
    c = s / 2.0
    K = np.array([[f, 0, c], [0, f, c], [0, 0, 1.0]])
    K_inv = np.linalg.inv(K)
    R_cam = _ry(np.radians(p.cam_yaw_deg)) @ _rx(np.radians(p.cam_pitch_deg))
    C = np.array([0.0, 0.0, z0])

    z = disparity_to_z(disp, s, p)

    # --- 前方パス: ソース全画素を 3D 化して回転・投影、z-buffer と逆マップをスプラット
    ys, xs = np.mgrid[0:s, 0:s].astype(np.float64)
    rays = np.stack([xs, ys, np.ones_like(xs)], axis=-1) @ K_inv.T  # (s,s,3)
    X = rays * z[..., None]
    Xp = (X - C) @ R_cam.T + C
    zp = Xp[..., 2]
    uvp = Xp @ K.T
    up = uvp[..., 0] / zp
    vp = uvp[..., 1] / zp

    order = np.argsort(-zp.ravel())  # 遠い順(近い画素が後勝ち)
    uf = up.ravel()[order]
    vf = vp.ravel()[order]
    src_x = xs.ravel()[order]
    src_y = ys.ravel()[order]
    zs = zp.ravel()[order]

    zbuf = np.full((s, s), np.inf)
    mapx = np.full((s, s), np.nan)
    mapy = np.full((s, s), np.nan)
    # 2x2 フットプリントでスプラット(点スプラットの 1px 隙間に背景が透けるのを防ぐ)
    u0 = np.floor(uf).astype(int)
    v0 = np.floor(vf).astype(int)
    for dx in (0, 1):
        for dy in (0, 1):
            uu, vv = u0 + dx, v0 + dy
            ok = (uu >= 0) & (uu < s) & (vv >= 0) & (vv < s)
            zbuf[vv[ok], uu[ok]] = zs[ok]
            mapx[vv[ok], uu[ok]] = src_x[ok]
            mapy[vv[ok], uu[ok]] = src_y[ok]

    hole = np.isnan(mapx)
    hole_ratio = float(hole.mean())
    maps = _fill_holes(np.stack([mapx, mapy], axis=-1), hole)
    # 2px 超の伸長部に残る孤立した背景サンプル(スペックル)を median で除去
    maps = np.stack([cv2.medianBlur(maps[..., 0].astype(np.float32), 5),
                     cv2.medianBlur(maps[..., 1].astype(np.float32), 5)], axis=-1)
    zbuf_f = _fill_holes(zbuf.copy(), np.isinf(zbuf))

    out = cv2.remap(crop_bgr, maps[..., 0].astype(np.float32),
                    maps[..., 1].astype(np.float32),
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # --- ランドマーク: 各点の深度で 3D 化 → 回転 → 再投影(閉形式)
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    z_lm = np.empty(n)
    for i, (x_, y_) in enumerate(pts):
        xi = int(np.clip(round(x_), 1, s - 2))
        yi = int(np.clip(round(y_), 1, s - 2))
        z_lm[i] = float(np.median(z[yi - 1:yi + 2, xi - 1:xi + 2]))
    Xl = np.concatenate([pts, np.ones((n, 1))], 1) @ K_inv.T * z_lm[:, None]
    Xlp = (Xl - C) @ R_cam.T + C
    zlp = Xlp[:, 2]
    uvl = Xlp @ K.T
    new_pts = uvl[:, :2] / zlp[:, None]

    vis = list(visibility)
    tol = p.occl_tol_ratio * z0
    for i, ((x_, y_), zl) in enumerate(zip(new_pts, zlp)):
        if x_ < 0 or y_ < 0 or x_ >= s or y_ >= s:
            vis[i] = 0
            continue
        if vis[i] == 0:
            continue
        zb = zbuf_f[int(y_), int(x_)]
        if np.isfinite(zb) and zb < zl - tol:
            vis[i] = 1  # 自己遮蔽

    R_new = R_cam @ rotation if rotation is not None else None
    return {"image": out, "points": new_pts, "visibility": vis,
            "rotation": R_new, "hole_ratio": hole_ratio, "zbuf": zbuf_f}
