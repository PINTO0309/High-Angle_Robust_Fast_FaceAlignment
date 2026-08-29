"""QA 可視化: 統合フォーマットのレコードにランドマーク・bbox・姿勢軸を重畳描画する。

使い方:
    python3 -m hrffa.dataset.qa.visualize --unified datasets/unified \
        --source 300wlp --n 12 --seed 0 --out /tmp/qa_300wlp

出力: 個別画像 + コンタクトシート(grid.jpg)。
可視性の色: 緑=可視(2) 橙=遮蔽(1) 赤=画像外(0) 灰=不明(-1)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

_VIS_COLORS = {2: (0, 200, 0), 1: (0, 160, 255), 0: (0, 0, 255), -1: (160, 160, 160)}


def _load_meta(unified: Path, scheme: str) -> dict:
    return json.loads((unified / "annotations" / "meta" / f"{scheme}.json").read_text())


def draw_record(unified: Path, rec: dict, meta_cache: dict) -> np.ndarray:
    img_path = unified / rec["image_path"]
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)

    scheme = rec["landmarks"]["scheme"]
    if scheme not in meta_cache:
        meta_cache[scheme] = _load_meta(unified, scheme)
    meta = meta_cache[scheme]

    pts = np.asarray(rec["landmarks"]["points"], dtype=np.float32)
    vis = rec["landmarks"]["visibility"]

    # スケールに応じた描画サイズ
    hb = rec["head_bbox"]
    ref = max(8.0, (hb[2] - hb[0] + hb[3] - hb[1]) / 2)
    r = max(1, int(round(ref / 120)))
    th = max(1, int(round(ref / 180)))

    # edge
    for edge in meta["edges"]:
        idx = edge["indices"]
        if len(idx) < 2:
            continue
        chain = idx + [idx[0]] if edge.get("closed") else idx
        for a, b in zip(chain[:-1], chain[1:]):
            cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                     (255, 200, 0), th, cv2.LINE_AA)
    # points
    for p, v in zip(pts, vis):
        cv2.circle(img, tuple(p.astype(int)), r, _VIS_COLORS[int(v)], -1, cv2.LINE_AA)

    # bbox
    cv2.rectangle(img, (int(hb[0]), int(hb[1])), (int(hb[2]), int(hb[3])),
                  (255, 0, 0), th + 1)
    if rec["face_bbox"] is not None:
        fb = rec["face_bbox"]
        cv2.rectangle(img, (int(fb[0]), int(fb[1])), (int(fb[2]), int(fb[3])),
                      (255, 255, 0), th)

    # 姿勢軸(回転行列の列 = 頭部 X/Y/Z 軸の画像投影)
    pose = rec.get("pose")
    if pose and pose.get("rotation_matrix"):
        R = np.asarray(pose["rotation_matrix"], dtype=np.float64)
        c = pts.mean(axis=0)
        L = ref * 0.35
        for k, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):  # X赤 Y緑 Z青
            tip = (c[0] + L * R[0, k], c[1] + L * R[1, k])
            cv2.line(img, tuple(c.astype(int)), (int(tip[0]), int(tip[1])),
                     color, th + 1, cv2.LINE_AA)
        e = pose.get("euler_deg")
        if e:
            cv2.putText(img, f"p{e['pitch']:+.0f} y{e['yaw']:+.0f} r{e['roll']:+.0f}",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
    return img


def make_grid(images: list[np.ndarray], cell: int = 320, cols: int = 4) -> np.ndarray:
    tiles = []
    for img in images:
        h, w = img.shape[:2]
        s = cell / max(h, w)
        resized = cv2.resize(img, (int(w * s), int(h * s)))
        tile = np.zeros((cell, cell, 3), dtype=np.uint8)
        y0 = (cell - resized.shape[0]) // 2
        x0 = (cell - resized.shape[1]) // 2
        tile[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
        tiles.append(tile)
    rows = (len(tiles) + cols - 1) // cols
    grid = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        rr, cc = divmod(i, cols)
        grid[rr * cell:(rr + 1) * cell, cc * cell:(cc + 1) * cell] = tile
    return grid


def sample_jsonl(path: Path, n: int, seed: int) -> list[dict]:
    """reservoir sampling で JSONL から n 件抽出する。"""
    rng = random.Random(seed)
    sample: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if len(sample) < n:
                sample.append(json.loads(line))
            else:
                j = rng.randint(0, i)
                if j < n:
                    sample[j] = json.loads(line)
    return sample


def main() -> None:
    ap = argparse.ArgumentParser(description="QA visualization: render landmarks, bbox and pose axes on unified-format records.")
    ap.add_argument("--unified", type=Path, default=Path("datasets/unified"))
    ap.add_argument("--source", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    jsonl = args.unified / "annotations" / f"{args.source}.jsonl"
    recs = sample_jsonl(jsonl, args.n, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    meta_cache: dict = {}
    drawn = []
    for rec in recs:
        img = draw_record(args.unified, rec, meta_cache)
        name = rec["record_id"].replace("/", "_") + ".jpg"
        cv2.imwrite(str(args.out / name), img)
        drawn.append(img)
    grid = make_grid(drawn)
    cv2.imwrite(str(args.out / "grid.jpg"), grid)
    print(f"wrote {len(drawn)} overlays + grid.jpg -> {args.out}")


if __name__ == "__main__":
    main()
