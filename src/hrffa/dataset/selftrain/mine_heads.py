"""T2: self-training 候補となる未ラベル実写頭部のマイニング CLI。

2 つの入力経路を持つ(出力形式は共通の候補 JSONL):

1. 既存 DEIMv2 キャッシュから(--source wflw/300w/cofw/all):
   アノテーション済みレコードに使われなかった head 検出を発掘する
   (WFLW の群衆シーン等。IoU > 0.4 で既使用 head を除外)
2. 外部画像ディレクトリから(--images-dir):
   DEIMv2 を推論して head を検出し、画像を datasets/unified/images/selftrain_<name>/
   へ実コピーする(ユーザー収集の極端姿勢画像の投入口)

候補 JSONL(1 行 = 1 頭部): {image_path, head_bbox, dir8, head_score}
既定で dir8 == back は除外(ランドマークが存在しないため)。

使い方:
    uv run python -m hrffa.dataset.selftrain.mine_heads --source all --name v1
    uv run python -m hrffa.dataset.selftrain.mine_heads \
        --images-dir /path/to/wild_images --name ext1
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from tqdm import tqdm

from ..pseudolabel.deimv2 import CLASS_HEAD, DIR8_CLASSES

_MODEL = Path("data/models/deimv2_wholebody49_boxes_only.onnx")


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1])
                    + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _dir8_of(head_box, dets) -> tuple[str | None, float]:
    best, score = None, 0.0
    for d in dets:
        if int(d[0]) in DIR8_CLASSES and _iou(head_box, d[1:5]) >= 0.5 and d[5] > score:
            best, score = DIR8_CLASSES[int(d[0])], d[5]
    return best, score


def mine_from_caches(unified: Path, sources: list[str], min_size: float,
                     min_score: float, exclude_dirs: set[str]):
    for src in sources:
        used: dict[str, list] = {}
        with open(unified / "annotations" / f"{src}.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                used.setdefault(r["image_path"], []).append(r["head_bbox"])
        with open(unified / "annotations" / f"{src}.deimv2.jsonl", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if "dets" not in row:
                    continue
                for d in row["dets"]:
                    if int(d[0]) != CLASS_HEAD or d[5] < min_score:
                        continue
                    hb = d[1:5]
                    if min(hb[2] - hb[0], hb[3] - hb[1]) < min_size:
                        continue
                    if any(_iou(hb, ub) > 0.4 for ub in used.get(row["image_path"], [])):
                        continue
                    dname, _ = _dir8_of(hb, row["dets"])
                    if dname in exclude_dirs:
                        continue
                    yield {"image_path": row["image_path"],
                           "head_bbox": [round(v, 2) for v in hb],
                           "dir8": dname, "head_score": d[5]}


def mine_from_dir(unified: Path, images_dir: Path, name: str, min_size: float,
                  min_score: float, exclude_dirs: set[str]):
    import cv2
    from ..pseudolabel.deimv2 import Deimv2Detector

    det = Deimv2Detector(_MODEL, score_threshold=min_score)
    out_img = unified / "images" / f"selftrain_{name}"
    out_img.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in images_dir.rglob("*")
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    for p in tqdm(paths, desc="detect", unit="img"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        dets = det.infer_batch([img])[0]
        heads = [d for d in dets if int(d[0]) == CLASS_HEAD and d[5] >= min_score]
        if not heads:
            continue
        # PNG 等は JPEG q92 に再エンコードして取り込む(ストレージ抑制。
        # 写真的コンテンツで視覚的無劣化、実データ側と形式も揃う)
        if p.suffix.lower() in (".jpg", ".jpeg"):
            rel = f"images/selftrain_{name}/{p.stem}{p.suffix.lower()}"
            if not (unified / rel).exists():
                shutil.copy2(p, unified / rel)
        else:
            rel = f"images/selftrain_{name}/{p.stem}.jpg"
            if not (unified / rel).exists():
                cv2.imwrite(str(unified / rel), img,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
        for d in heads:
            hb = d[1:5]
            if min(hb[2] - hb[0], hb[3] - hb[1]) < min_size:
                continue
            dname, _ = _dir8_of(hb, dets)
            if dname in exclude_dirs:
                continue
            yield {"image_path": rel, "head_bbox": [round(v, 2) for v in hb],
                   "dir8": dname, "head_score": d[5]}


def main() -> None:
    ap = argparse.ArgumentParser(description="T2: mine unlabeled real head crops as self-training candidates.")
    ap.add_argument("--unified", type=Path, default=Path("datasets/unified"))
    ap.add_argument("--source", default=None,
                    choices=["all", "wflw", "300w", "cofw"])
    ap.add_argument("--images-dir", type=Path, default=None)
    ap.add_argument("--name", required=True, help="candidate set name (output file name)")
    ap.add_argument("--min-size", type=float, default=48.0)
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--exclude-dirs", nargs="*", default=["back"])
    args = ap.parse_args()
    if (args.source is None) == (args.images_dir is None):
        raise SystemExit("specify exactly one of --source or --images-dir")

    if args.source:
        sources = ["wflw", "300w", "cofw"] if args.source == "all" else [args.source]
        gen = mine_from_caches(args.unified, sources, args.min_size,
                               args.min_score, set(args.exclude_dirs))
    else:
        gen = mine_from_dir(args.unified, args.images_dir, args.name,
                            args.min_size, args.min_score, set(args.exclude_dirs))

    out = args.unified / "annotations" / f"selftrain_candidates_{args.name}.jsonl"
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in gen:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"candidates: {n} -> {out}")


if __name__ == "__main__":
    main()
