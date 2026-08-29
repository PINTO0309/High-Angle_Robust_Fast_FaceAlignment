"""DEIMv2 擬似ラベルの推論実行 CLI(キャッシュ生成)。

統合アノテーション JSONL からユニーク画像を列挙して推論し、
検出結果キャッシュ(1 行 = 1 画像)を追記型 JSONL で書き出す。
中断しても再実行すれば未処理分だけ推論する(resumable)。

300wlp は base 画像のみ推論する(masked はアノテーション共有のため
適用時に base の結果を流用する)。

使い方:
    PYTHONPATH=src python3 -m hrffa.dataset.pseudolabel.run \
        --unified datasets/unified --source 300wlp --batch 16
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from tqdm import tqdm

from .deimv2 import Deimv2Detector

_DEFAULT_MODEL = Path(
    "data/models/deimv2_dinov3_x_wholebody49_ins_s08_maskhead256x3_center_1240query_masks_n_batch.onnx"
)


def unique_images(jsonl: Path) -> list[str]:
    """推論対象のユニーク image_path(300wlp は masked を除外)。"""
    seen: dict[str, None] = {}
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            path = rec["image_path"]
            if rec["source_dataset"] == "300W_LP_w_masked" and rec["attributes"].get("mask_worn"):
                continue
            seen.setdefault(path, None)
    return list(seen)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run DEIMv2 pseudo-label inference (build the detection cache).")
    ap.add_argument("--unified", type=Path, default=Path("datasets/unified"))
    ap.add_argument("--source", required=True,
                    choices=["300wlp", "wflw", "300w", "cofw"])
    ap.add_argument("--model", type=Path, default=_DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--score-threshold", type=float, default=0.35)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cache_path = args.unified / "annotations" / f"{args.source}.deimv2.jsonl"
    done: set[str] = set()
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            done = {json.loads(line)["image_path"] for line in f}

    images = unique_images(args.unified / "annotations" / f"{args.source}.jsonl")
    if args.limit is not None:
        images = images[: args.limit]
    todo = [p for p in images if p not in done]
    print(f"[{args.source}] unique={len(images)} cached={len(done)} todo={len(todo)}")
    if not todo:
        return

    detector = Deimv2Detector(args.model, score_threshold=args.score_threshold)
    t0 = time.time()
    n_done = 0
    with open(cache_path, "a", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=8) as loader:
        for i in tqdm(range(0, len(todo), args.batch), unit="batch"):
            chunk = todo[i:i + args.batch]
            imgs = list(loader.map(
                lambda p: cv2.imread(str(args.unified / p)), chunk))
            valid = [(p, im) for p, im in zip(chunk, imgs) if im is not None]
            for p, im in zip(chunk, imgs):
                if im is None:
                    out.write(json.dumps({"image_path": p, "error": "imread_failed"}) + "\n")
            if not valid:
                continue
            results = detector.infer_batch([im for _, im in valid])
            for (p, _), dets in zip(valid, results):
                out.write(json.dumps({"image_path": p, "dets": dets}) + "\n")
            n_done += len(valid)
            if i % (args.batch * 50) == 0:
                out.flush()
    dt = time.time() - t0
    print(f"[{args.source}] inferred {n_done} images in {dt:.1f}s "
          f"({n_done / max(dt, 1e-9):.1f} img/s) -> {cache_path}")


if __name__ == "__main__":
    main()
