"""D4b 深度再投影の QA プレビュー。

各サンプルにつき「元クロップ / 深度マップ / 深度再投影(複数角度)」を並べ、
ランドマーク・可視性・姿勢軸・hole率を描画する。--compare-homography で
同角度の純ホモグラフィ版(D4)も並べて視差の効果を比較できる。

使い方:
    PYTHONPATH=src python3 -m hrffa.dataset.qa.depth_warp_preview \
        --source 300wlp --n 4 --angles -40 -25 25 40 --out /tmp/qa_dw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ..augment.depth_warp import DepthAnythingV2, DepthWarpParams, depth_reproject
from ..augment.geometric import GeometricParams, apply_geometric
from .augment_preview import _draw
from .visualize import _load_meta, sample_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="QA preview of the D4b depth reprojection.")
    ap.add_argument("--unified", type=Path, default=Path("datasets/unified"))
    ap.add_argument("--source", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--angles", type=float, nargs="+", default=[-40, -25, 25, 40])
    ap.add_argument("--axis", choices=["pitch", "yaw"], default="pitch")
    ap.add_argument("--compare-homography", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    recs = sample_jsonl(args.unified / "annotations" / f"{args.source}.jsonl",
                        args.n, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    depth_model = DepthAnythingV2()
    meta_cache: dict = {}

    rows = []
    for rec in recs:
        scheme = rec["landmarks"]["scheme"]
        if scheme not in meta_cache:
            meta_cache[scheme] = _load_meta(args.unified, scheme)
        meta = meta_cache[scheme]
        img = cv2.imread(str(args.unified / rec["image_path"]))
        pts = np.asarray(rec["landmarks"]["points"], dtype=np.float64)
        vis = rec["landmarks"]["visibility"]
        R = (np.asarray(rec["pose"]["rotation_matrix"])
             if rec["pose"] and rec["pose"].get("rotation_matrix") else None)

        base = apply_geometric(img, pts, vis, R, rec["head_bbox"],
                               GeometricParams(out_size=256))
        disp = depth_model.infer(base["image"])
        dn = ((disp - disp.min()) / (np.ptp(disp) + 1e-9) * 255).astype(np.uint8)

        tiles = [_draw(base, meta, "orig crop"),
                 cv2.applyColorMap(dn, cv2.COLORMAP_INFERNO)]
        for a in args.angles:
            prm = DepthWarpParams(
                cam_pitch_deg=a if args.axis == "pitch" else 0.0,
                cam_yaw_deg=a if args.axis == "yaw" else 0.0)
            out = depth_reproject(base["image"], disp, base["points"],
                                  base["visibility"], base["rotation"], prm)
            tiles.append(_draw(out, meta,
                               f"dw {args.axis}{a:+.0f} h{out['hole_ratio']:.2f}"))
            if args.compare_homography:
                hprm = GeometricParams(
                    out_size=256,
                    cam_pitch_deg=a if args.axis == "pitch" else 0.0,
                    cam_yaw_deg=a if args.axis == "yaw" else 0.0)
                hout = apply_geometric(img, pts, vis, R, rec["head_bbox"], hprm)
                tiles.append(_draw(hout, meta, f"H {args.axis}{a:+.0f}"))
        rows.append(np.concatenate(tiles, axis=1))
    grid = np.concatenate(rows, axis=0)
    name = f"dw_{args.source}_{args.axis}.jpg"
    cv2.imwrite(str(args.out / name), grid)
    print(f"wrote {args.out / name}")


if __name__ == "__main__":
    main()
