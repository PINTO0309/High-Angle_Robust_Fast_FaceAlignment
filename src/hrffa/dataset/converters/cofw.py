"""ORFormer/COFW(MMPose COCO 形式 JSON + 展開済み images/)→ 統合フォーマット変換。

- 29 点。keypoints の v: 2=可視 / 1=遮蔽(統合表現と同値なのでそのまま写像)。
  train/test とも遮蔽情報は原本 COFW_*_color.mat の occlusion ビットと
  全数一致することを検証済み(history/004)。
- mirror 画像は JSON に含まれない。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..geometry import head_bbox_from_landmarks
from .base import make_record

_SPLIT_FILES = [
    ("train", "annotations/cofw_train.json"),
    ("test", "annotations/cofw_test.json"),
]


def iter_records(root: Path, limit: int | None = None, workers: int = 0):
    for split, fname in _SPLIT_FILES:
        data = json.loads((root / fname).read_text())
        images = {im["id"]: im for im in data["images"]}
        anns = data["annotations"]
        if limit is not None:
            anns = anns[:limit]
        for ann in anns:
            im = images[ann["image_id"]]
            kp = np.asarray(ann["keypoints"], dtype=np.float64).reshape(29, 3)
            points = kp[:, :2]
            visibility = [int(v) for v in kp[:, 2]]
            x, y, bw, bh = (float(v) for v in ann["bbox"])
            yield make_record(
                record_id=f"cofw/{split}/{ann['image_id']:06d}",
                image_path=f"images/cofw/{im['file_name']}",
                image_size=(int(im["width"]), int(im["height"])),
                source_dataset="ORFormer_COFW",
                license_tag="research_only",
                split=split,
                head_bbox=head_bbox_from_landmarks(points),
                head_bbox_source="landmark_derived",
                face_bbox=[x, y, x + bw, y + bh],
                scheme="cofw29",
                points=points,
                visibility=visibility,
                pose=None,
                attributes={},
                quality={},
            )
