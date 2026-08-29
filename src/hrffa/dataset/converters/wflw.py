"""ORFormer/WFLW(原寸画像 + 98 点 txt)→ 統合フォーマット変換。

1 行 = 196 座標 + 検出矩形 4 + 属性 6(pose/expression/illumination/make-up/
occlusion/blur)+ 画像相対パス。座標は原寸画像の絶対ピクセル。
1 画像に複数顔があるため 1 行 = 1 レコード。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..geometry import head_bbox_from_landmarks
from ..schema import VIS_UNKNOWN
from .base import make_record

_ATTR_NAMES = ["pose", "expression", "illumination", "makeup", "occlusion", "blur"]


def iter_records(root: Path, limit: int | None = None, workers: int = 0):
    ann_dir = root / "WFLW_annotations" / "list_98pt_rect_attr_train_test"
    img_root = root / "WFLW_images"
    size_cache: dict[str, tuple[int, int]] = {}

    for split, fname in [
        ("train", "list_98pt_rect_attr_train.txt"),
        ("test", "list_98pt_rect_attr_test.txt"),
    ]:
        lines = (ann_dir / fname).read_text().splitlines()
        if limit is not None:
            lines = lines[:limit]
        for i, line in enumerate(lines):
            tok = line.split()
            assert len(tok) == 207, f"unexpected field count {len(tok)} at {fname}:{i+1}"
            coords = np.array([float(v) for v in tok[:196]], dtype=np.float64).reshape(98, 2)
            rect = [float(v) for v in tok[196:200]]  # x_min y_min x_max y_max
            attrs = {name: bool(int(v)) for name, v in zip(_ATTR_NAMES, tok[200:206])}
            rel_path = tok[206]

            if rel_path not in size_cache:
                with Image.open(img_root / rel_path) as im:
                    size_cache[rel_path] = im.size
            w, h = size_cache[rel_path]

            yield make_record(
                record_id=f"wflw/{split}/{i:05d}",
                image_path=f"images/wflw/{rel_path}",
                image_size=(w, h),
                source_dataset="ORFormer_WFLW",
                license_tag="research_only",
                split=split,
                head_bbox=head_bbox_from_landmarks(coords),
                head_bbox_source="landmark_derived",
                face_bbox=rect,
                scheme="wflw98",
                points=coords,
                visibility=[VIS_UNKNOWN] * 98,
                pose=None,
                attributes=attrs,
                quality={},
            )
