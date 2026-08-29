"""幾何拡張の数値セルフチェック。

検証項目:
  1. Roll 回転の厳密性: 変換後ランドマークと、姿勢軸の画像投影が同じ Rot2D(θ) で
     回ること(相対誤差 < 1e-9)
  2. hflip の Euler 整合: pitch 不変・yaw/roll 符号反転
  3. 変換の往復: T ののち T^{-1} で元座標に戻る(誤差 < 1e-6 px)
  4. 回転行列の直交性維持
  5. カメラ回転ホモグラフィの整合: H による 2D 変換が、3D 光線を R_cam で回して
     再投影した結果と一致する(誤差 < 1e-6 px)

使い方: PYTHONPATH=src python3 -m hrffa.dataset.augment.selfcheck
"""

from __future__ import annotations

import math

import numpy as np

from ..geometry import euler300wlp_to_rotmat, rotmat_to_euler300wlp
from .geometric import GeometricParams, apply_geometric, camera_homography


def main() -> None:
    rng = np.random.default_rng(0)
    img = np.zeros((300, 280, 3), dtype=np.uint8)
    head_bbox = [40.0, 30.0, 240.0, 260.0]

    # --- 1. Roll 回転の厳密性 ---
    max_err = 0.0
    for _ in range(300):
        pts = rng.uniform(50, 230, size=(68, 2))
        p_euler = rng.uniform(-1.2, 1.2, size=3)
        R = euler300wlp_to_rotmat(*p_euler)
        theta = float(rng.uniform(0, 360))
        base = GeometricParams(roll_deg=0.0)
        rot = GeometricParams(roll_deg=theta)
        out0 = apply_geometric(img, pts, [-1] * 68, R, head_bbox, base)
        out1 = apply_geometric(img, pts, [-1] * 68, R, head_bbox, rot)
        # 出力中心回りの Rot2D(θ) で out0 の点を回すと out1 に一致するはず
        t = math.radians(theta)
        c, s = math.cos(t), math.sin(t)
        Rot = np.array([[c, -s], [s, c]])
        center = np.array([128.0, 128.0])
        mapped = (out0["points"] - center) @ Rot.T + center
        max_err = max(max_err, float(np.abs(mapped - out1["points"]).max()))
        # 姿勢軸の画像投影も同じ 2D 回転に従う
        ax0 = out0["rotation"][:2, :]
        ax1 = out1["rotation"][:2, :]
        max_err = max(max_err, float(np.abs(Rot @ ax0 - ax1).max()))
    print(f"1. roll exactness: max_err={max_err:.2e}  {'PASS' if max_err < 1e-9 else 'FAIL'}")

    # --- 2. hflip の Euler 整合 ---
    max_err = 0.0
    for _ in range(300):
        p0, y0, r0 = rng.uniform(-1.2, 1.2, size=3)
        R = euler300wlp_to_rotmat(p0, y0, r0)
        out = apply_geometric(img, rng.uniform(50, 230, (68, 2)), [-1] * 68, R,
                              head_bbox, GeometricParams(hflip=True))
        p1, y1, r1 = rotmat_to_euler300wlp(out["rotation"])
        err = max(abs(p1 - p0), abs(y1 + y0), abs(r1 + r0))
        max_err = max(max_err, err)
    print(f"2. hflip euler (p,+y,+r inv): max_err={max_err:.2e}  "
          f"{'PASS' if max_err < 1e-9 else 'FAIL'}")

    # --- 3. 往復変換 ---
    max_err = 0.0
    for _ in range(300):
        pts = rng.uniform(50, 230, size=(68, 2))
        prm = GeometricParams(
            roll_deg=float(rng.uniform(0, 360)),
            cam_pitch_deg=float(rng.uniform(-25, 25)),
            cam_yaw_deg=float(rng.uniform(-15, 15)),
            scale=float(rng.uniform(0.9, 1.1)),
            tx=float(rng.uniform(-0.05, 0.05)), ty=float(rng.uniform(-0.05, 0.05)))
        out = apply_geometric(img, pts, [-1] * 68, None, head_bbox, prm)
        T = out["transform"]
        back_h = np.concatenate([out["points"], np.ones((68, 1))], 1) @ np.linalg.inv(T).T
        back = back_h[:, :2] / back_h[:, 2:3]
        max_err = max(max_err, float(np.abs(back - pts).max()))
    print(f"3. roundtrip: max_err={max_err:.2e} px  {'PASS' if max_err < 1e-6 else 'FAIL'}")

    # --- 4. 直交性 ---
    max_err = 0.0
    for _ in range(300):
        R = euler300wlp_to_rotmat(*rng.uniform(-1.2, 1.2, size=3))
        prm = GeometricParams(
            roll_deg=float(rng.uniform(0, 360)),
            cam_pitch_deg=float(rng.uniform(-25, 25)),
            cam_yaw_deg=float(rng.uniform(-15, 15)),
            hflip=bool(rng.random() < 0.5))
        out = apply_geometric(img, rng.uniform(50, 230, (68, 2)), [-1] * 68, R,
                              head_bbox, prm)
        Rn = out["rotation"]
        max_err = max(max_err, float(np.abs(Rn @ Rn.T - np.eye(3)).max()),
                      abs(float(np.linalg.det(Rn)) - 1.0))
    print(f"4. rotation orthonormality: max_err={max_err:.2e}  "
          f"{'PASS' if max_err < 1e-9 else 'FAIL'}")

    # --- 5. カメラ回転ホモグラフィの光線整合 ---
    max_err = 0.0
    for _ in range(300):
        prm = GeometricParams(
            cam_pitch_deg=float(rng.uniform(-25, 25)),
            cam_yaw_deg=float(rng.uniform(-15, 15)))
        H, R_cam = camera_homography(prm)
        f = prm.focal_ratio * prm.out_size
        c = prm.out_size / 2
        K = np.array([[f, 0, c], [0, f, c], [0, 0, 1.0]])
        uv = rng.uniform(0, 256, size=(50, 2))
        rays = np.linalg.inv(K) @ np.concatenate([uv, np.ones((50, 1))], 1).T
        reproj = K @ (R_cam @ rays)
        reproj = (reproj[:2] / reproj[2]).T
        via_h = np.concatenate([uv, np.ones((50, 1))], 1) @ H.T
        via_h = via_h[:, :2] / via_h[:, 2:3]
        max_err = max(max_err, float(np.abs(reproj - via_h).max()))
    print(f"5. homography ray consistency: max_err={max_err:.2e} px  "
          f"{'PASS' if max_err < 1e-6 else 'FAIL'}")


if __name__ == "__main__":
    main()
