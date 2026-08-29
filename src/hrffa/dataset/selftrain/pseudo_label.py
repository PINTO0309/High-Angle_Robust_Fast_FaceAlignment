"""T2: 教師モデルによる擬似ラベル付与 CLI(Roll/flip-TTA 一貫性フィルタ付き)。

各候補頭部に対し 8 ビュー(roll 0/90/180/270° × flip なし/あり)で教師推論し、
すべて元画像座標系へ逆変換して集約する:
  - 擬似ランドマーク = 8 推定の平均
  - 信頼度 = 点ごとの標準偏差の平均を head bbox 対角で正規化(tta_std_norm)。
    教師はランドマークが Roll 等変(014: 1.9px)なので、ばらつきは教師自身の
    不確実性の角度不変な推定になる
  - 可視性 = 8 ビューの多数決
--max-std-norm 以下のサンプルのみ統合フォーマットとして出力する。

--crosscheck 指定時は 6DRepNet360(独立モデル)による合意ゲートを追加する
(確証バイアス対策。history/015 §6.7):
  - yaw: 6DRepNet yaw と DEIMv2 dir8 セクタ中心の円周差が 60° 以内
  - pitch: 6DRepNet pitch <= -50° なのに教師が口領域(48-67)の 7 割以上を
    可視と判定 → 矛盾として棄却(俯きなら口は隠れるはず)。+50° 以上で
    口領域の 7 割以上が画像外判定 → 矛盾として棄却
採用レコードには attributes.sixd_ypr(参考メタデータ)を付与する。

使い方:
    uv run python -m hrffa.dataset.selftrain.pseudo_label \
        --candidates datasets/unified/annotations/selftrain_candidates_v1.jsonl \
        --ckpt <baseline_v2.pt> --preset abl_pose_off_scratch_96gb --name v1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from ...model.teacher import TeacherModel
from ...train.config import get_config
from ..augment.geometric import GeometricParams, apply_geometric
from ..converters.base import JsonlWriter, make_record, write_stats
from ..qa.sixdrepnet import SixDRepNet360
from ...model.losses import DIR8_YAW_DEG

_VIEWS = [(roll, flip) for roll in (0.0, 90.0, 180.0, 270.0) for flip in (False, True)]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="T2: pseudo-label with the teacher model (roll/flip-TTA consistency filter).")
    ap.add_argument("--unified", type=Path, default=Path("datasets/unified"))
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--preset", default="abl_pose_off_scratch_96gb")
    ap.add_argument("--name", required=True, help="suffix of the output source name (e.g. v1)")
    ap.add_argument("--max-std-norm", type=float, default=0.03,
                    help="acceptance threshold (TTA std as a ratio of the head bbox diagonal)")
    ap.add_argument("--batch-heads", type=int, default=4, help="number of heads processed at once")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--crosscheck", action="store_true",
                    help="enable the independent 6DRepNet360 cross-check gate")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config(args.preset)
    model = TeacherModel.from_config(cfg).to(device).eval()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("ema") or ck["model"])

    meta = json.loads((args.unified / "annotations" / "meta" / "ibug68.json").read_text())
    flip_map = meta["flip_mapping"]
    swap = np.arange(68)
    for a, b in flip_map:
        swap[a], swap[b] = b, a

    from ...model.backbone import IMAGE_MEAN, IMAGE_STD
    mean = np.array(IMAGE_MEAN, np.float32)
    std = np.array(IMAGE_STD, np.float32)
    s = cfg.out_size

    sixd = SixDRepNet360() if args.crosscheck else None

    cands = [json.loads(l) for l in open(args.candidates, encoding="utf-8")]
    if args.limit:
        cands = cands[: args.limit]

    out_path = args.unified / "annotations" / f"selftrain_{args.name}.jsonl"
    n_kept = n_drop = 0
    n_xc_yaw = n_xc_pitch = 0
    std_all = []
    from tqdm import tqdm
    with JsonlWriter(out_path) as w:
        for i0 in tqdm(range(0, len(cands), args.batch_heads), unit="batch"):
            chunk = cands[i0:i0 + args.batch_heads]
            tensors, invs, metas_c = [], [], []
            for c in chunk:
                img = cv2.imread(str(args.unified / c["image_path"]))
                if img is None:
                    metas_c.append(None)
                    continue
                dummy = np.zeros((68, 2))
                views = []
                for roll, flip in _VIEWS:
                    prm = GeometricParams(out_size=s, roll_deg=roll, hflip=flip)
                    out = apply_geometric(img, dummy, [-1] * 68, None,
                                          c["head_bbox"], prm)
                    x = cv2.cvtColor(out["image"], cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                    x = (x - mean) / std
                    tensors.append(torch.from_numpy(x.transpose(2, 0, 1)))
                    views.append((out["transform"], flip))
                invs.append(views)
                metas_c.append((c, img.shape[1], img.shape[0]))
            valid = [m for m in metas_c if m is not None]
            if not valid:
                continue
            batch = torch.stack(tensors).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                pred = model(batch, "ibug68")
            pts = pred["points"].float().cpu().numpy() * s      # (B*8, 68, 2)
            vis = pred["vis_logits"].argmax(-1).cpu().numpy()   # (B*8, 68)

            k = 0
            for m in metas_c:
                if m is None:
                    continue
                c, iw, ih = m
                views = invs.pop(0)
                est, vises = [], []
                for (T, flip) in views:
                    p = pts[k]
                    v = vis[k]
                    k += 1
                    if flip:
                        p = p[swap]
                        v = v[swap]
                    ph = np.concatenate([p, np.ones((68, 1))], 1) @ np.linalg.inv(T).T
                    est.append(ph[:, :2] / ph[:, 2:3])
                    vises.append(v)
                est = np.stack(est)                      # (8, 68, 2)
                hb = c["head_bbox"]
                diag = float(np.hypot(hb[2] - hb[0], hb[3] - hb[1]))
                std_norm = float(est.std(axis=0).mean() / max(diag, 1e-6))
                std_all.append(std_norm)
                if std_norm > args.max_std_norm:
                    n_drop += 1
                    continue
                mean_pts = est.mean(axis=0)
                vis_maj = [int(Counter(v[j] for v in vises).most_common(1)[0][0])
                           for j in range(68)]

                sixd_ypr = None
                if sixd is not None:
                    img_full = cv2.imread(str(args.unified / c["image_path"]))
                    s_yaw, s_pitch, s_roll = sixd.infer(img_full, c["head_bbox"])
                    sixd_ypr = [round(s_yaw, 1), round(s_pitch, 1), round(s_roll, 1)]
                    # yaw 合意(dir8 がある場合のみ)
                    d8 = c.get("dir8")
                    if d8 in DIR8_YAW_DEG:
                        diff = abs(s_yaw - DIR8_YAW_DEG[d8])
                        diff = min(diff, 360 - diff)
                        if diff > 60:
                            n_xc_yaw += 1
                            continue
                    # pitch と教師可視性の整合(口領域 48-67)
                    mouth_vis = vis_maj[48:68]
                    n_visible = sum(1 for v in mouth_vis if v == 2)
                    n_out = sum(1 for v in mouth_vis if v == 0)
                    if s_pitch <= -50 and n_visible >= 14:
                        n_xc_pitch += 1
                        continue
                    if s_pitch >= 50 and n_out >= 14:
                        n_xc_pitch += 1
                        continue
                w.write(make_record(
                    record_id=f"selftrain/{args.name}/{n_kept:06d}",
                    image_path=c["image_path"],
                    image_size=(iw, ih),
                    source_dataset=f"selftrain_{args.name}",
                    license_tag="research_only",
                    split="train",
                    head_bbox=hb,
                    head_bbox_source="deimv2_pseudo",
                    face_bbox=None,
                    scheme="ibug68",
                    points=mean_pts,
                    visibility=vis_maj,
                    pose=None,
                    direction8=c.get("dir8"),
                    attributes={"pseudo_teacher": args.ckpt.name,
                                "tta_std_norm": round(std_norm, 5),
                                **({"sixd_ypr": sixd_ypr} if sixd_ypr else {})},
                    quality={"deimv2": "matched",
                             "deimv2_head_score": c.get("head_score")},
                ))
                n_kept += 1
    stats = {"kept": n_kept, "dropped": n_drop,
             "crosscheck_reject_yaw": n_xc_yaw,
             "crosscheck_reject_pitch": n_xc_pitch,
             "std_norm_p50": round(float(np.percentile(std_all, 50)), 5),
             "std_norm_p90": round(float(np.percentile(std_all, 90)), 5),
             "max_std_norm": args.max_std_norm}
    write_stats(args.unified / "annotations" / f"selftrain_{args.name}.stats.json", stats)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
