"""コンバータ共通処理: レコード生成・JSONL 書き出し・統計・画像リンク。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..schema import SCHEMA_VERSION, validate_record


def make_record(
    *,
    record_id: str,
    image_path: str,
    image_size: tuple[int, int],
    source_dataset: str,
    license_tag: str,
    split: str,
    head_bbox: list[float],
    head_bbox_source: str,
    face_bbox: list[float] | None,
    scheme: str,
    points: np.ndarray,
    visibility: list[int],
    pose: dict | None = None,
    direction8: str | None = None,
    attributes: dict | None = None,
    paired_record: str | None = None,
    quality: dict | None = None,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id,
        "image_path": image_path,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "source_dataset": source_dataset,
        "license_tag": license_tag,
        "split": split,
        "head_bbox": [round(float(v), 2) for v in head_bbox],
        "head_bbox_source": head_bbox_source,
        "face_bbox": None if face_bbox is None else [round(float(v), 2) for v in face_bbox],
        "landmarks": {
            "scheme": scheme,
            "points": [[round(float(x), 3), round(float(y), 3)] for x, y in np.asarray(points)],
            "visibility": [int(v) for v in visibility],
        },
        "pose": pose,
        "direction8": direction8,
        "attributes": attributes or {},
        "paired_record": paired_record,
        "quality": quality or {},
    }


class JsonlWriter:
    """JSONL 書き出し + バリデーション + 統計集計。"""

    def __init__(self, out_path: Path, max_errors: int = 50):
        self.out_path = out_path
        self.max_errors = max_errors
        self.n_written = 0
        self.errors: list[str] = []
        self.split_counter: Counter = Counter()
        self.vis_counter: Counter = Counter()
        self.quality_counter: Counter = Counter()
        self.pose_available = 0
        self._fh = None

    def __enter__(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.out_path, "w", encoding="utf-8")
        return self

    def __exit__(self, *exc):
        self._fh.close()
        return False

    def write(self, rec: dict) -> None:
        errs = validate_record(rec)
        if errs:
            if len(self.errors) < self.max_errors:
                self.errors.extend(errs)
            raise ValueError(f"schema validation failed: {errs[:3]}")
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.n_written += 1
        self.split_counter[rec["split"]] += 1
        for v in rec["landmarks"]["visibility"]:
            self.vis_counter[v] += 1
        if rec["pose"] is not None:
            self.pose_available += 1
        oc = rec["quality"].get("offset_check")
        if oc is not None:
            self.quality_counter[oc] += 1

    def stats(self) -> dict[str, Any]:
        return {
            "n_records": self.n_written,
            "splits": dict(self.split_counter),
            "visibility_histogram": {str(k): v for k, v in sorted(self.vis_counter.items())},
            "pose_available": self.pose_available,
            "offset_check": dict(self.quality_counter),
            "validation_errors": self.errors,
        }


def write_stats(out_path: Path, stats: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def batched(it: Iterable, n: int) -> Iterable[list]:
    buf: list = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf
