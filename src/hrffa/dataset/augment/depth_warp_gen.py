"""D4b: 深度再投影による極端 pitch 合成データ生成 CLI。

300wlp の GT 姿勢つきレコードから、カメラ pitch を |合成後 pitch| が増える方向へ
回して合成レコードを生成する。出力:
  - datasets/unified/images/synth_dwarp/<parent_id>_p<deg>.jpg(256x256 クロップ)
  - datasets/unified/annotations/synth_dwarp.jsonl(スキーマ準拠、pose は厳密更新)

選択方針: |pitch| GT が大きいレコードを優先(warp 量を抑えて品質を保ちつつ
45-90° 帯に届かせる)。warp は ±(--warp-min..--warp-max)°、符号は合成後の
|pitch| が最大になる方を選ぶ。

使い方:
    PYTHONPATH=src python3 -m hrffa.dataset.augment.depth_warp_gen \
        --n 2000 --min-src-pitch 20 --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ..converters.base import JsonlWriter, make_record, write_stats
from ..geometry import rotmat_to_euler300wlp
from .depth_warp import DepthAnythingV2, DepthWarpParams, depth_reproject
from .geometric import GeometricParams, apply_geometric


def main() -> None:
    ap = argparse.ArgumentParser(description="D4b: generate extreme-pitch synthetic data by depth reprojection.")
    ap.add_argument("--unified", type=Path, default=Path("datasets/unified"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--min-src-pitch", type=float, default=20.0,
                    help="prefer GT records with |pitch| at or above this value")
    ap.add_argument("--warp-min", type=float, default=20.0)
    ap.add_argument("--warp-max", type=float, default=45.0)
    ap.add_argument("--max-hole", type=float, default=0.45,
                    help="discard outputs whose hole ratio exceeds this value")
    ap.add_argument("--out-size", type=int, default=256)
    ap.add_argument("--out-name", default="synth_dwarp",
                    help="output JSONL name (images are shared under images/synth_dwarp/)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # 候補収集: offset pass + deimv2 matched + 非マスク優先/マスク版も可
    strong: list[dict] = []
    weak: list[dict] = []
    with open(args.unified / "annotations" / "300wlp.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["quality"].get("offset_check") == "fail":
                continue
            if rec["quality"].get("deimv2") != "matched":
                continue
            (strong if abs(rec["pose"]["euler_deg"]["pitch"]) >= args.min_src_pitch
             else weak).append(rec)
    rng.shuffle(strong)
    rng.shuffle(weak)
    picks = (strong + weak)[: args.n]
    print(f"candidates: strong={len(strong)} weak={len(weak)} -> generating {len(picks)}")

    img_dir = args.unified / "images" / "synth_dwarp"
    img_dir.mkdir(parents=True, exist_ok=True)
    depth_model = DepthAnythingV2()

    n_drop = 0
    pitch_out: list[float] = []
    out_path = args.unified / "annotations" / f"{args.out_name}.jsonl"
    with JsonlWriter(out_path) as w:
        for rec in tqdm(picks, unit="rec"):
            img = cv2.imread(str(args.unified / rec["image_path"]))
            if img is None:
                continue
            pts = np.asarray(rec["landmarks"]["points"], dtype=np.float64)
            R = np.asarray(rec["pose"]["rotation_matrix"], dtype=np.float64)
            base = apply_geometric(
                img, pts, rec["landmarks"]["visibility"], R, rec["head_bbox"],
                GeometricParams(out_size=args.out_size))
            disp = depth_model.infer(base["image"])

            mag = float(rng.uniform(args.warp_min, args.warp_max))
            # 合成後 |pitch| が大きくなる符号を選ぶ
            best = None
            for sign in (1.0, -1.0):
                prm = DepthWarpParams(cam_pitch_deg=sign * mag)
                out = depth_reproject(base["image"], disp, base["points"],
                                      base["visibility"], base["rotation"], prm)
                p_new = math.degrees(rotmat_to_euler300wlp(out["rotation"])[0])
                if best is None or abs(p_new) > abs(best[1]):
                    best = (out, p_new, sign * mag)
            out, p_new, warp_deg = best
            if out["hole_ratio"] > args.max_hole:
                n_drop += 1
                continue

            p_r, y_r, r_r = rotmat_to_euler300wlp(out["rotation"])
            pose = {
                "rotation_matrix": [[round(float(v), 6) for v in row]
                                    for row in out["rotation"]],
                "euler_deg": {"pitch": round(math.degrees(p_r), 3),
                              "yaw": round(math.degrees(y_r), 3),
                              "roll": round(math.degrees(r_r), 3)},
                "source": "300wlp_gt_warped",
            }
            # record_id = "300wlp/<folder>/<stem>"。Flip フォルダ違いで同名 stem が
            # 存在するため、フォルダ名も含めて一意にする
            _, folder, stem = rec["record_id"].split("/")
            name = f"{folder}_{stem}_p{warp_deg:+.0f}".replace("+", "p").replace("-", "m")
            cv2.imwrite(str(img_dir / f"{name}.jpg"), out["image"],
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

            pts_new = np.asarray(out["points"])
            lx1, ly1 = pts_new.min(axis=0)
            lx2, ly2 = pts_new.max(axis=0)
            wl, hl = lx2 - lx1, ly2 - ly1
            head_bbox = [lx1 - 0.3 * wl, ly1 - 0.9 * hl, lx2 + 0.3 * wl, ly2 + 0.15 * hl]

            w.write(make_record(
                record_id=f"synth_dwarp/{name}",
                image_path=f"images/synth_dwarp/{name}.jpg",
                image_size=(args.out_size, args.out_size),
                source_dataset="synth_dwarp_300wlp",
                license_tag="research_only",
                split="train",
                head_bbox=head_bbox,
                head_bbox_source="landmark_derived",
                face_bbox=None,
                scheme="ibug68",
                points=pts_new,
                visibility=out["visibility"],
                pose=pose,
                attributes={"synthetic": "depth_warp",
                            "parent_record": rec["record_id"],
                            "warp_pitch_deg": round(warp_deg, 1),
                            "mask_worn": rec["attributes"].get("mask_worn", False)},
                quality={"hole_ratio": round(out["hole_ratio"], 3)},
            ))
            pitch_out.append(math.degrees(p_r))

    a = np.abs(np.array(pitch_out))
    bins = [(0, 30), (30, 45), (45, 60), (60, 90), (90, 180)]
    hist = {f"{lo}-{hi}": int(((a >= lo) & (a < hi)).sum()) for lo, hi in bins}
    stats = {"generated": len(pitch_out), "dropped_hole": n_drop,
             "abs_pitch_hist": hist}
    write_stats(args.unified / "annotations" / f"{args.out_name}.stats.json", stats)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
