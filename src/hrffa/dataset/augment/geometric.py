"""D4: 幾何拡張コア — head crop / Roll 360° / カメラ回転透視ワープ / 反転。

すべての幾何拡張は 1 つの 3x3 射影変換 T に合成し、画像は 1 回だけ warp する。
ランドマークは T で、頭部姿勢(回転行列)は対応する 3D 回転で厳密に更新する。

GT 更新の理論的根拠(規約は geometry.py 参照: 画像 x 右, y 下, カメラ z 奥):
  - **Roll 回転**(画像内回転 θ): 画像の 2D 回転はカメラの z 軸回り回転と等価。
      T に Rot2D(θ) を合成し、姿勢は R' = Rz(θ) @ R。ランドマーク・姿勢とも厳密。
  - **カメラ回転ワープ**(俯仰角 φ, 方位 ψ): 並進のない純カメラ回転による画像変化は
      シーンの 3D 形状に依存せずホモグラフィ H = K @ R_cam @ K^{-1} で厳密に表せる。
      姿勢は R' = R_cam @ R。ランドマーク・姿勢とも厳密(新たに見える面が無いという
      意味で見た目は元視点のままだが、幾何・ラベルは新カメラ姿勢に正確に一致する)。
      K は焦点距離 f = focal_ratio * out_size のピンホールを仮定(撮影実機の K は未知の
      ため近似。focal_ratio は SPIGA 等の慣例に倣い 1.0〜1.5 を想定)。
  - **水平反転**: 画像 x 反転。姿勢は R' = M @ R @ M(M = diag(-1,1,1))。
      Euler では pitch 不変・yaw / roll 符号反転に相当。点は scheme の flip_mapping で
      入れ替える。
  - スケール・平行移動は姿勢を変えない。

可視性: 変換後に出力クロップ外へ出た点は 0(画像外)へ更新する(元が -1 でも 0 に
落とす。遮蔽 1 は保持)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..geometry import _rx, _ry, _rz  # モジュール内共有の基本回転


@dataclass
class GeometricParams:
    """1 サンプル分の幾何拡張パラメータ(決定済みの値)。"""
    out_size: int = 256
    pad: float = 0.15            # head bbox の外周マージン(辺長比)
    roll_deg: float = 0.0        # 画像内回転(Roll 360° 対応の本体)
    cam_pitch_deg: float = 0.0   # カメラ俯仰(+で見上げ方向へ回す)
    cam_yaw_deg: float = 0.0     # カメラ方位
    scale: float = 1.0
    tx: float = 0.0              # 出力サイズ比の平行移動
    ty: float = 0.0
    hflip: bool = False
    focal_ratio: float = 1.2


@dataclass
class GeometricPolicy:
    """サンプリング範囲(学習設定)。roll_mode: 'full360' | 'small'"""
    out_size: int = 256
    pad: float = 0.15
    roll_mode: str = "full360"
    roll_small_deg: float = 30.0
    cam_pitch_deg: float = 25.0
    cam_yaw_deg: float = 15.0
    scale_range: tuple[float, float] = (0.9, 1.1)
    translate: float = 0.05
    hflip_prob: float = 0.5
    focal_ratio: float = 1.2

    def sample(self, rng: np.random.Generator) -> GeometricParams:
        roll = (float(rng.uniform(0.0, 360.0)) if self.roll_mode == "full360"
                else float(rng.uniform(-self.roll_small_deg, self.roll_small_deg)))
        return GeometricParams(
            out_size=self.out_size,
            pad=self.pad,
            roll_deg=roll,
            cam_pitch_deg=float(rng.uniform(-self.cam_pitch_deg, self.cam_pitch_deg)),
            cam_yaw_deg=float(rng.uniform(-self.cam_yaw_deg, self.cam_yaw_deg)),
            scale=float(rng.uniform(*self.scale_range)),
            tx=float(rng.uniform(-self.translate, self.translate)),
            ty=float(rng.uniform(-self.translate, self.translate)),
            hflip=bool(rng.random() < self.hflip_prob),
            focal_ratio=self.focal_ratio,
        )


def crop_affine(head_bbox: list[float], p: GeometricParams) -> np.ndarray:
    """head bbox(+pad)を out_size 正方へ写す相似変換(3x3)。

    回転・スケール・平行移動もクロップ中心基準でここに合成する。
    """
    x1, y1, x2, y2 = head_bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * p.pad)
    s = p.out_size / side * p.scale
    theta = math.radians(p.roll_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    half = p.out_size / 2
    # 平行移動 → 回転+スケール → 出力中心へ
    T = np.array([
        [s * cos_t, -s * sin_t, 0.0],
        [s * sin_t, s * cos_t, 0.0],
        [0.0, 0.0, 1.0],
    ])
    T[0, 2] = half + p.tx * p.out_size - (T[0, 0] * cx + T[0, 1] * cy)
    T[1, 2] = half + p.ty * p.out_size - (T[1, 0] * cx + T[1, 1] * cy)
    return T


def camera_homography(p: GeometricParams) -> tuple[np.ndarray, np.ndarray]:
    """純カメラ回転のホモグラフィ H(出力クロップ座標系)と R_cam を返す。"""
    phi = math.radians(p.cam_pitch_deg)
    psi = math.radians(p.cam_yaw_deg)
    R_cam = _ry(psi) @ _rx(phi)
    f = p.focal_ratio * p.out_size
    c = p.out_size / 2
    K = np.array([[f, 0, c], [0, f, c], [0, 0, 1.0]])
    H = K @ R_cam @ np.linalg.inv(K)
    return H, R_cam


def flip_matrix(out_size: int) -> np.ndarray:
    return np.array([[-1.0, 0.0, out_size - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


_M_MIRROR = np.diag([-1.0, 1.0, 1.0])


def apply_geometric(
    image: np.ndarray,
    points: np.ndarray,
    visibility: list[int],
    rotation: np.ndarray | None,
    head_bbox: list[float],
    p: GeometricParams,
    flip_mapping: list[list[int]] | None = None,
) -> dict:
    """幾何拡張を適用し、画像・点・可視性・姿勢を一括更新して返す。"""
    T = crop_affine(head_bbox, p)
    R_new = rotation.copy() if rotation is not None else None
    theta = math.radians(p.roll_deg)
    if R_new is not None:
        R_new = _rz(theta) @ R_new

    H, R_cam = camera_homography(p)
    if abs(p.cam_pitch_deg) > 1e-9 or abs(p.cam_yaw_deg) > 1e-9:
        T = H @ T
        if R_new is not None:
            R_new = R_cam @ R_new
        # カメラ回転は f·tanφ 級の平行移動成分を持つため、頭部中心を出力中心へ
        # 戻す(クロップ窓の平行移動であり姿勢 GT には影響しない)
        cx, cy = (head_bbox[0] + head_bbox[2]) / 2, (head_bbox[1] + head_bbox[3]) / 2
        m = T @ np.array([cx, cy, 1.0])
        mx, my = m[0] / m[2], m[1] / m[2]
        half = p.out_size / 2
        T = np.array([[1.0, 0.0, half + p.tx * p.out_size - mx],
                      [0.0, 1.0, half + p.ty * p.out_size - my],
                      [0.0, 0.0, 1.0]]) @ T

    if p.hflip:
        T = flip_matrix(p.out_size) @ T
        if R_new is not None:
            R_new = _M_MIRROR @ R_new @ _M_MIRROR

    out = cv2.warpPerspective(
        image, T, (p.out_size, p.out_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    pts_h = np.concatenate([points, np.ones((len(points), 1))], axis=1) @ T.T
    pts = pts_h[:, :2] / pts_h[:, 2:3]

    vis = list(visibility)
    if p.hflip and flip_mapping:
        pts = pts.copy()
        for a, b in flip_mapping:
            pts[[a, b]] = pts[[b, a]]
            vis[a], vis[b] = vis[b], vis[a]

    vis = [
        0 if (x < 0 or y < 0 or x >= p.out_size or y >= p.out_size) else v
        for (x, y), v in zip(pts, vis)
    ]
    return {"image": out, "points": pts, "visibility": vis,
            "rotation": R_new, "transform": T}
