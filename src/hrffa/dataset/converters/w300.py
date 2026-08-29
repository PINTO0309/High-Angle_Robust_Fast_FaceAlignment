"""ORFormer/300W(MMPose COCO 形式 JSON)→ 統合フォーマット変換。

- train / valid_common / valid_challenge の 3 JSON を正とする
  (valid は common+challenge の和、test は画像実体が無いため対象外)。
- mirror 画像は JSON に含まれない(反転は学習時拡張で行う方針)。
- keypoints の v は常に 1(=ラベル有りの意)で可視性情報ではないため
  visibility は不明(-1)とする。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..geometry import head_bbox_from_landmarks
from ..schema import VIS_UNKNOWN
from .base import make_record

_SPLIT_FILES = [
    ("train", "face_landmarks_300w_train.json"),
    ("valid_common", "face_landmarks_300w_valid_common.json"),
    ("valid_challenge", "face_landmarks_300w_valid_challenge.json"),
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
            kp = np.asarray(ann["keypoints"], dtype=np.float64).reshape(68, 3)
            points = kp[:, :2]
            x, y, bw, bh = (float(v) for v in ann["bbox"])
            yield make_record(
                record_id=f"300w/{split}/{ann['image_id']:04d}",
                image_path=f"images/300w/{im['file_name']}",
                image_size=(int(im["width"]), int(im["height"])),
                source_dataset="ORFormer_300W",
                license_tag="research_only",
                split=split,
                head_bbox=head_bbox_from_landmarks(points),
                head_bbox_source="landmark_derived",
                face_bbox=[x, y, x + bw, y + bh],
                scheme="ibug68",
                points=points,
                visibility=[VIS_UNKNOWN] * 68,
                pose=None,
                attributes={
                    "center": [float(c) for c in ann.get("center", [])] or None,
                    "scale": float(ann["scale"]) if "scale" in ann else None,
                },
                quality={},
            )
