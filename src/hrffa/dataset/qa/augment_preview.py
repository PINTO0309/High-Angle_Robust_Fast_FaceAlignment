"""D4 拡張の QA プレビュー: 実レコードにランダム幾何拡張を適用して重畳描画する。

各サンプルにつき「元クロップ + 拡張 K 変種」を 1 行に並べ、ランドマーク・
可視性・姿勢軸(300wlp のみ)・適用パラメータを描画する。

使い方:
    PYTHONPATH=src python3 -m hrffa.dataset.qa.augment_preview \
        --unified datasets/unified --source 300wlp --n 4 --k 3 --seed 0 \
        --out /tmp/qa_aug
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ..augment.geometric import GeometricParams, GeometricPolicy, apply_geometric
from .visualize import _VIS_COLORS, _load_meta, sample_jsonl


def _draw(out: dict, meta: dict, label: str) -> np.ndarray:
    img = out["image"].copy()
    pts = out["points"].astype(np.float32)
    for edge in meta["edges"]:
        idx = edge["indices"]
        chain = idx + [idx[0]] if edge.get("closed") else idx
        for a, b in zip(chain[:-1], chain[1:]):
            cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                     (255, 200, 0), 1, cv2.LINE_AA)
    for p, v in zip(pts, out["visibility"]):
        cv2.circle(img, tuple(p.astype(int)), 2, _VIS_COLORS[int(v)], -1, cv2.LINE_AA)
    if out["rotation"] is not None:
        R = out["rotation"]
        c = pts[np.array(out["visibility"]) != 0].mean(axis=0) \
            if (np.array(out["visibility"]) != 0).any() else pts.mean(axis=0)
        L = img.shape[0] * 0.18
        for k, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            tip = (int(c[0] + L * R[0, k]), int(c[1] + L * R[1, k]))
            cv2.line(img, tuple(c.astype(int)), tip, color, 2, cv2.LINE_AA)
    cv2.putText(img, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="QA preview of the D4 augmentation: apply random geometric augmentation to real records and render overlays.")
    ap.add_argument("--unified", type=Path, default=Path("datasets/unified"))
    ap.add_argument("--source", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    policy = GeometricPolicy()
    recs = sample_jsonl(args.unified / "annotations" / f"{args.source}.jsonl",
                        args.n, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

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

        tiles = []
        base = apply_geometric(img, pts, vis, R, rec["head_bbox"],
                               GeometricParams(out_size=256, pad=policy.pad))
        tiles.append(_draw(base, meta, "orig crop"))
        for _ in range(args.k):
            prm = policy.sample(rng)
            out = apply_geometric(img, pts, vis, R, rec["head_bbox"], prm,
                                  flip_mapping=meta["flip_mapping"])
            label = (f"r{prm.roll_deg:.0f} cp{prm.cam_pitch_deg:+.0f} "
                     f"cy{prm.cam_yaw_deg:+.0f}{' F' if prm.hflip else ''}")
            tiles.append(_draw(out, meta, label))
        rows.append(np.concatenate(tiles, axis=1))
    grid = np.concatenate(rows, axis=0)
    cv2.imwrite(str(args.out / f"aug_{args.source}.jpg"), grid)
    print(f"wrote {args.out / f'aug_{args.source}.jpg'}")


if __name__ == "__main__":
    main()
