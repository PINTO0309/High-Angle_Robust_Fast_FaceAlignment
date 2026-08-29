"""DEIMv2 検出キャッシュを統合アノテーションへ適用する CLI。

- head(cls 7): ランドマーク包含率が最大の box を選択(同率はスコア優先)。
  包含率 >= --containment(既定 0.6)で採用し head_bbox を置換、
  head_bbox_source = "deimv2_pseudo"。不採用時は landmark_derived のまま。
- direction8(cls 8-15): 採用 head と IoU >= 0.5 の中で最高スコアのものを付与。
- face(cls 16): face_bbox が null のレコード(300wlp)のみ、ランドマーク bbox と
  IoU >= 0.3 の最高スコア box で補完(face_bbox_source を quality に記録)。
- 300wlp の masked レコードは base 画像の検出結果を共有する(幾何は同一)。

使い方:
    PYTHONPATH=src python3 -m hrffa.dataset.pseudolabel.apply \
        --unified datasets/unified --source 300wlp
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .deimv2 import CLASS_FACE, CLASS_HEAD, DIR8_CLASSES
from ..converters.base import write_stats


def _iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _containment(points: np.ndarray, box: list[float]) -> float:
    inside = (
        (points[:, 0] >= box[0]) & (points[:, 0] <= box[2])
        & (points[:, 1] >= box[1]) & (points[:, 1] <= box[3])
    )
    return float(inside.mean())


def apply_to_record(rec: dict, dets: list[list[float]], containment_thr: float) -> str:
    """レコードを in-place 更新し、適用結果種別を返す。"""
    points = np.asarray(rec["landmarks"]["points"], dtype=np.float64)

    heads = [d for d in dets if int(d[0]) == CLASS_HEAD]
    best, best_key = None, (-1.0, -1.0)
    for d in heads:
        key = (_containment(points, d[1:5]), d[5])
        if key > best_key:
            best, best_key = d, key
    cont = best_key[0]

    if best is None or cont < containment_thr:
        rec["quality"]["deimv2"] = "no_head" if best is None else "low_containment"
        rec["quality"]["deimv2_containment"] = round(cont, 3) if best is not None else None
        return rec["quality"]["deimv2"]

    head_box = [float(v) for v in best[1:5]]
    rec["head_bbox"] = [round(v, 2) for v in head_box]
    rec["head_bbox_source"] = "deimv2_pseudo"
    rec["quality"]["deimv2"] = "matched"
    rec["quality"]["deimv2_containment"] = round(cont, 3)
    rec["quality"]["deimv2_head_score"] = best[5]

    # direction8
    dir_best = None
    for d in dets:
        if int(d[0]) in DIR8_CLASSES and _iou(head_box, d[1:5]) >= 0.5:
            if dir_best is None or d[5] > dir_best[5]:
                dir_best = d
    if dir_best is not None:
        rec["direction8"] = DIR8_CLASSES[int(dir_best[0])]
        rec["quality"]["direction8_score"] = dir_best[5]

    # face bbox の補完(300wlp のみ null)
    if rec["face_bbox"] is None:
        lx1, ly1 = points.min(axis=0)
        lx2, ly2 = points.max(axis=0)
        lmk_box = [float(lx1), float(ly1), float(lx2), float(ly2)]
        face_best, face_iou = None, 0.0
        for d in dets:
            if int(d[0]) == CLASS_FACE:
                iou = _iou(lmk_box, d[1:5])
                if iou > face_iou:
                    face_best, face_iou = d, iou
        if face_best is not None and face_iou >= 0.3:
            rec["face_bbox"] = [round(float(v), 2) for v in face_best[1:5]]
            rec["quality"]["face_bbox_source"] = "deimv2_pseudo"
            rec["quality"]["face_bbox_iou_lmk"] = round(face_iou, 3)

    return "matched"


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply the DEIMv2 detection cache to the unified annotations.")
    ap.add_argument("--unified", type=Path, default=Path("datasets/unified"))
    ap.add_argument("--source", required=True,
                    choices=["300wlp", "wflw", "300w", "cofw"])
    ap.add_argument("--containment", type=float, default=0.6)
    args = ap.parse_args()

    ann_path = args.unified / "annotations" / f"{args.source}.jsonl"
    cache_path = args.unified / "annotations" / f"{args.source}.deimv2.jsonl"

    cache: dict[str, list[list[float]]] = {}
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "dets" in row:
                cache[row["image_path"]] = row["dets"]

    counter: Counter = Counter()
    dir8_counter: Counter = Counter()
    tmp_path = ann_path.with_suffix(".jsonl.tmp")
    with open(ann_path, encoding="utf-8") as fin, \
            open(tmp_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, unit="rec", desc=f"apply:{args.source}"):
            rec = json.loads(line)
            key = rec["image_path"]
            # masked は base 画像の検出を共有
            if rec["source_dataset"] == "300W_LP_w_masked" and rec["attributes"].get("mask_worn"):
                key = key.replace("_masked.jpg", ".jpg")
            dets = cache.get(key)
            if dets is None:
                rec["quality"]["deimv2"] = "no_cache"
                counter["no_cache"] += 1
            else:
                counter[apply_to_record(rec, dets, args.containment)] += 1
            if rec["direction8"]:
                dir8_counter[rec["direction8"]] += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_path.replace(ann_path)

    stats = {
        "source": args.source,
        "apply_result": dict(counter),
        "direction8": dict(dir8_counter),
        "containment_threshold": args.containment,
    }
    write_stats(args.unified / "annotations" / f"{args.source}.deimv2_apply.stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
