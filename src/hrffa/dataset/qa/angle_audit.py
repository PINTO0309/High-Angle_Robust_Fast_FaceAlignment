"""生成画像の頭部角度忠実度の自動監査 CLI。

ファイル名の意図角度(pitch{P}_yaw{Y}_cam{E}_*)と、姿勢推定器による実測値を
突き合わせ、ビン別の誤差統計を出す。

3 推定器の役割分担(較正結果に基づく。history/015 §6.5):
  - 教師(abl_pose_on): pitch ±45°・カメラ水平で誤差 2-3° — 中間域の精密判定
  - 6DRepNet360: yaw 全域(p50 5.1°)・roll(p50 4.1°)・極端 pitch の崩壊検出
    (±40° 超は p50 20° とノイジーだが「正面に潰れている」ことの検出には十分)
  - DEIMv2 顔パーツ: 極端域のカテゴリカル裏取り

2 段構成:
  1. |pitch| <= 45° かつカメラ水平(|cam| <= 15): 姿勢監督つき教師
     (abl_pose_on、geodesic 約 3.9°)による度数比較。学習分布外(|pitch|>45)では
     平均回帰・飽和し、カメラ俯仰の透視は P−E 近似が崩れるため対象外とする
  2. |pitch_target| > 60°(極端域): DEIMv2 の顔パーツ検出によるカテゴリカル判定
     - 下向き極端(<= -60): 頭頂部が写るはず → 鼻・口が head 内で未検出なら合格
     - 上向き極端(>= +60): 顎下から顔が見えるはず → 鼻または口が検出されたら合格

意図値の解釈: 推定器は「カメラに対する頭部姿勢」を出すため、カメラ仰俯角 E がある
画像は pitch_target = P − E(カメラが上にあるほど頭部は相対的に下向きに見える近似)。
yaw は符号規約の差異を避けるため絶対値で比較する。
pitch は Euler 分解(yaw±90° でジンバルロック)を避け、頭部 Y 軸の前後傾
atan2(R[2,1], R[1,1]) で抽出する(roll≈0 の生成仕様が前提)。

使い方:
    uv run python -m hrffa.dataset.qa.angle_audit \
        --images-dir data/imagegen/.../images \
        [--pose-ckpt runs/abl_pose_on_96gb/abl_pose_on_96gb_best_e0008_0.015890.pt]
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from ...model.losses import rot6d_to_matrix  # noqa: F401(モデル内部で使用)
from ...model.teacher import TeacherModel
from ...train.config import get_config
from ..augment.geometric import GeometricParams, apply_geometric
from ..geometry import rotmat_to_euler300wlp
from ..pseudolabel.deimv2 import CLASS_HEAD, Deimv2Detector
from .sixdrepnet import SixDRepNet360

_NOSE, _MOUTH = 18, 19

_NAME_RE = re.compile(r"pitch([+-]\d+)_yaw([+-]\d+)_cam([+-]\d+)_")
_DEIMV2 = Path("data/models/deimv2_wholebody49_boxes_only.onnx")

# 集計ビン(意図 pitch_target 基準、度)
_PITCH_BINS = [(-180, -90), (-90, -45), (-45, -20), (-20, 20), (20, 45), (45, 90), (90, 180)]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Automated audit of head-angle fidelity for generated images.")
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--pose-ckpt", type=Path,
                    default=Path("runs/abl_pose_on_96gb/abl_pose_on_96gb_best_e0008_0.015890.pt"))
    ap.add_argument("--preset", default="teacher_vitl_96gb")
    ap.add_argument("--off-bin-tol", type=float, default=20.0,
                    help="errors (deg) above this count as off-target")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-teacher", action="store_true",
                    help="skip the teacher estimator (faster on CPU; 6DRepNet always runs)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config(args.preset)
    model = TeacherModel.from_config(cfg).to(device).eval()
    ck = torch.load(args.pose_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("ema") or ck["model"])
    det = Deimv2Detector(_DEIMV2)
    sixd = SixDRepNet360()

    paths = sorted(p for p in args.images_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    if args.limit:
        paths = paths[: args.limit]

    rows = []
    n_nohead = n_noname = 0
    for p in paths:
        m = _NAME_RE.search(p.name)
        if not m:
            n_noname += 1
            continue
        pi, yi, ei = (int(m.group(k)) for k in (1, 2, 3))
        pitch_target = pi - ei  # カメラ相対への補正(近似)
        img = cv2.imread(str(p))
        if img is None:
            continue
        dets = det.infer_batch([img])[0]
        heads = [d for d in dets if int(d[0]) == CLASS_HEAD]
        if not heads:
            n_nohead += 1
            continue
        hb = max(heads, key=lambda d: d[5])[1:5]
        # head 内の鼻・口検出(中心が head bbox 内にあるか)
        def _inside(d):
            cx, cy = (d[1] + d[3]) / 2, (d[2] + d[4]) / 2
            return hb[0] <= cx <= hb[2] and hb[1] <= cy <= hb[3]
        has_nose = any(int(d[0]) == _NOSE and d[5] >= 0.6 and _inside(d) for d in dets)
        has_mouth = any(int(d[0]) == _MOUTH and d[5] >= 0.6 and _inside(d) for d in dets)
        s_yaw, s_pitch, s_roll = sixd.infer(img, hb)
        if args.no_teacher:
            pe_deg, ye_deg = float("nan"), float("nan")
        else:
            crop = apply_geometric(img, np.zeros((68, 2)), [-1] * 68, None, hb,
                                   GeometricParams(out_size=cfg.out_size))
            from ...model.backbone import IMAGE_MEAN, IMAGE_STD
            x = cv2.cvtColor(crop["image"], cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            x = (x - np.array(IMAGE_MEAN, np.float32)) / np.array(IMAGE_STD, np.float32)
            t = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out = model(t, "ibug68")
            R = out["rot"][0].float().cpu().numpy()
            _, ye, _ = rotmat_to_euler300wlp(R)
            # pitch は Euler 分解だと yaw±90° でジンバルロックするため、頭部 Y 軸
            # (R の第 2 列)の前後傾から直接取る
            pe = math.atan2(float(R[2, 1]), float(R[1, 1]))
            pe_deg, ye_deg = math.degrees(pe), math.degrees(ye)
        rows.append({"file": p.name, "pitch_target": pitch_target,
                     "pitch_intent": pi, "cam": ei, "yaw_intent": yi,
                     "pitch_est": round(pe_deg, 1),
                     "yaw_est": round(ye_deg, 1),
                     "pitch_err": round(pe_deg - pitch_target, 1),
                     "abs_yaw_err": round(abs(abs(ye_deg) - abs(yi)), 1),
                     "sixd_pitch": round(s_pitch, 1), "sixd_yaw": round(s_yaw, 1),
                     "sixd_roll": round(s_roll, 1),
                     "sixd_yaw_err": round(abs(abs(s_yaw) - abs(yi)), 1),
                     "has_nose": has_nose, "has_mouth": has_mouth})

    print(f"audited {len(rows)} / {len(paths)} images "
          f"(no-head {n_nohead}, name-mismatch {n_noname})\n")
    print("== quantitative audit (|pitch| <= 45 deg, camera level only = estimator validity range) ==")
    print(f"{'bin':>12} {'n':>5} {'|pitch_err| mean':>17} {'p50':>7} "
          f"{'off>{:.0f}deg'.format(args.off_bin_tol):>10}")
    binned = defaultdict(list)
    for r in rows:
        if abs(r["pitch_intent"]) > 45 or abs(r["cam"]) > 15:
            continue
        for lo, hi in _PITCH_BINS:
            if lo <= r["pitch_target"] < hi:
                binned[(lo, hi)].append(r)
                break
    for (lo, hi) in _PITCH_BINS:
        rs = binned.get((lo, hi), [])
        if not rs:
            continue
        errs = np.abs([r["pitch_err"] for r in rs])
        off = float((errs > args.off_bin_tol).mean())
        print(f"{lo:>5}..{hi:<5} {len(rs):>5} {errs.mean():>17.1f} "
              f"{np.median(errs):>7.1f} {off:>9.1%}")
    # 極端域のカテゴリカル判定
    # 6DRepNet360 による極端 pitch 崩壊検出 + roll 遵守 + yaw
    print("\n== 6DRepNet360 audit ==")
    for lo, hi in [(-120, -60), (60, 120)]:
        ext = [r for r in rows if lo <= r["pitch_intent"] < hi and abs(r["cam"]) <= 15]
        if not ext:
            continue
        est = np.array([r["sixd_pitch"] for r in ext])
        collapse = float((np.abs(est) < 20).mean())
        med = float(np.median(est))
        print(f"  intended pitch {lo}..{hi}: n={len(ext)} estimated median {med:+.0f} deg "
              f"frontal-collapse rate (|est|<20 deg) {collapse:.1%}")
    rolls = np.abs([r["sixd_roll"] for r in rows if not np.isnan(r["sixd_roll"])])
    print(f"  roll compliance (spec ~0 deg): p50 {np.median(rolls):.1f} deg / p90 "
          f"{np.percentile(rolls, 90):.1f}° / >15° {float((rolls > 15).mean()):.1%}")
    syaw = np.array([r["sixd_yaw_err"] for r in rows if not np.isnan(r["sixd_yaw"])])
    print(f"  |yaw| error (6DRepNet): mean {syaw.mean():.1f} deg / p50 {np.median(syaw):.1f} deg")

    # カテゴリカル判定は subject の姿勢(pitch_intent)基準。カメラ俯仰の大きい
    # 画像は「カメラが上/下から顔を見る」ため除外する
    flat_cam = [r for r in rows if abs(r["cam"]) <= 15]
    lo_ext = [r for r in flat_cam if r["pitch_intent"] <= -60]
    hi_ext = [r for r in flat_cam if r["pitch_intent"] >= 60]
    print("\n== categorical audit of the extreme range (face-part detection) ==")
    if lo_ext:
        ok = sum(1 for r in lo_ext if not r["has_mouth"])
        print(f"  downward <= -60 deg (expect looking down: mouth not detected): {ok}/{len(lo_ext)} passed")
        for r in lo_ext:
            if r["has_mouth"]:
                print(f"    needs review: {r['file']}")
    if hi_ext:
        ok = sum(1 for r in hi_ext if r["has_nose"] or r["has_mouth"])
        print(f"  upward >= +60 deg (expect face from below the chin: nose or mouth present): {ok}/{len(hi_ext)} passed")
        for r in hi_ext:
            if not (r["has_nose"] or r["has_mouth"]):
                print(f"    needs review: {r['file']}")

    yaw_errs = np.array([r["abs_yaw_err"] for r in rows])
    print(f"\n|yaw| error: mean {yaw_errs.mean():.1f} deg / p50 {np.median(yaw_errs):.1f} deg")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            wr = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"csv -> {args.csv}")


if __name__ == "__main__":
    main()
