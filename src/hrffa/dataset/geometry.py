"""座標補正・回転表現・bbox 生成のユーティリティ。

回転の規約(本プロジェクト統一):
  - 画像座標系: x 右, y 下(カメラ z は画面奥)
  - 300W-LP の Pose_Para = [pitch, yaw, roll, ...](ラジアン)に対し
        R = Rx(pitch) @ Ry(-yaw) @ Rz(roll)
    と定義する。R の列ベクトルが頭部座標軸(X: 被写体の左方向,
    Y: 頭部下方向, Z: 顔前方)の画像フレームでの向きになる。
    この規約は一般的な頭部姿勢の軸描画式
    (x1,y1)=(cos y cos r, cos p sin r + cos r sin p sin y) 等と一致する
    (yaw は符号反転後の値)。QA 可視化で軸描画に用いて整合性を確認できる。
"""

from __future__ import annotations

import math

import numpy as np

# 300W_LP_w_masked の保存画像は 450x450 フレームからの平行移動クロップ。
# クロップ原点は pt2d から k=0.2 系の式で復元できる(検証済み: history/001 §1.2)。
_CROP_K2 = 0.4  # 2*k, k=0.2


def correct_300wlp_pt2d(pt2d: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """450x450 フレーム座標の pt2d (2,68) を保存画像座標へ平行移動補正する。

    Returns:
        (corrected (2,68), (ox, oy))
    """
    w = float(pt2d[0].max() - pt2d[0].min())
    h = float(pt2d[1].max() - pt2d[1].min())
    ox = float(pt2d[0].min()) - _CROP_K2 * w
    oy = float(pt2d[1].min()) - _CROP_K2 * h
    out = pt2d.astype(np.float64).copy()
    out[0] -= ox
    out[1] -= oy
    return out, (ox, oy)


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def euler300wlp_to_rotmat(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """300W-LP の Euler 角(ラジアン)→ 回転行列(モジュール規約)。"""
    return _rx(pitch) @ _ry(-yaw) @ _rz(roll)


def rotmat_to_euler300wlp(R: np.ndarray) -> tuple[float, float, float]:
    """euler300wlp_to_rotmat の逆変換(ラジアン)。検証用。

    R = Rx(p) @ Ry(y') @ Rz(r), y' = -yaw を展開すると
        R[0,2] = sin y',  R[1,2] = -sin p cos y',  R[2,2] = cos p cos y'
        R[0,0] = cos y' cos r,  R[0,1] = -cos y' sin r
    となるため、これらから閉形式で解く。
    """
    R = np.asarray(R, dtype=float)
    yp = math.asin(max(-1.0, min(1.0, float(R[0, 2]))))  # y' = -yaw
    if abs(math.cos(yp)) > 1e-8:
        pitch = math.atan2(-float(R[1, 2]), float(R[2, 2]))
        roll = math.atan2(-float(R[0, 1]), float(R[0, 0]))
    else:  # ジンバルロック(y' = ±90°): roll を 0 とみなし pitch に吸収
        pitch = math.atan2(float(R[2, 1]), float(R[1, 1]))
        roll = 0.0
    return pitch, -yp, roll


# ランドマーク bbox → 頭部全体 bbox の外挿係数(D1 プレースホルダ。
# D2 で DEIMv2 擬似ラベルに置き換える。roll≈0 前提の近似)。
_HEAD_EXPAND_X = 0.30   # 左右に w の 30% ずつ
_HEAD_EXPAND_TOP = 0.90  # 上に h の 90%(頭髪・頭頂)
_HEAD_EXPAND_BOT = 0.15  # 下に h の 15%(顎下)


def head_bbox_from_landmarks(points: np.ndarray) -> list[float]:
    """ランドマーク (N,2) から頭部全体 bbox [x1,y1,x2,y2] を外挿する。

    画像外にはみ出してもクリップしない(クロップ時にパディングで対応)。
    """
    x1, y1 = float(points[:, 0].min()), float(points[:, 1].min())
    x2, y2 = float(points[:, 0].max()), float(points[:, 1].max())
    w, h = x2 - x1, y2 - y1
    return [
        x1 - _HEAD_EXPAND_X * w,
        y1 - _HEAD_EXPAND_TOP * h,
        x2 + _HEAD_EXPAND_X * w,
        y2 + _HEAD_EXPAND_BOT * h,
    ]


def count_points_outside(points: np.ndarray, image_size: tuple[int, int]) -> tuple[int, float]:
    """画像外の点数と最大はみ出し量(px)を返す。"""
    w, h = image_size
    dx = np.maximum(np.maximum(-points[:, 0], points[:, 0] - w), 0.0)
    dy = np.maximum(np.maximum(-points[:, 1], points[:, 1] - h), 0.0)
    over = np.maximum(dx, dy)
    n_out = int((over > 0).sum())
    return n_out, float(over.max()) if len(over) else 0.0
