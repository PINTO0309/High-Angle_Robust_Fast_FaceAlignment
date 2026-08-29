"""300W_LP_w_masked → 統合フォーマット変換。

- .mat の pt2d は 450x450 フレーム座標のため平行移動補正する(geometry 参照)。
- X_masked.mat は X.mat と同一(検証済み)なので base の .mat のみ読む。
- base / masked を別レコードとし paired_record で相互リンクする。
- 姿勢 GT は Pose_Para[0:3](rad)。回転行列はモジュール規約で生成。
- Flip フォルダの pt2d は座標ミラーのみで左右番号入替がされていない(history/020)。
  ibug68 の flip_mapping で並べ替えて意味を揃える(Pose_Para は 300W-LP 側で
  反転済みのためそのまま)。
"""

from __future__ import annotations

import json
import math
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

from ..geometry import (
    correct_300wlp_pt2d,
    count_points_outside,
    euler300wlp_to_rotmat,
    head_bbox_from_landmarks,
)
from ..schema import VIS_UNKNOWN
from .base import make_record

_FOLDER_RE = re.compile(r"^(AFW|HELEN|IBUG|LFPW)(_Flip)?_\d{2}$")

# 補正後ランドマークのはみ出し判定:
#   pass    : はみ出し <= _PASS_PX
#   clipped : <= _CLIPPED_PX。クロップが元画像境界で切れ顎等が枠外に出た正常
#             ケース(目視確認済み。可視部分の整合は保たれている)
#   fail    : それ超。系統的オフセット誤り(~150px)の疑い
_PASS_PX = 8.0
_CLIPPED_PX = 60.0


def ibug68_flip_perm() -> np.ndarray:
    """ibug68 の左右入替 permutation(new[i] = old[perm[i]])。"""
    meta = json.loads(
        (Path(__file__).resolve().parent.parent / "meta" / "ibug68.json").read_text()
    )
    perm = np.arange(68)
    for a, b in meta["flip_mapping"]:
        perm[a], perm[b] = b, a
    return perm


_FLIP_PERM = ibug68_flip_perm()


def list_folders(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and _FOLDER_RE.match(p.name))


def _convert_one_stem(folder: Path, stem: str) -> list[dict]:
    """1 つの base stem から base / masked の最大 2 レコードを作る。"""
    mat_path = folder / f"{stem}.mat"
    base_jpg = folder / f"{stem}.jpg"
    masked_jpg = folder / f"{stem}_masked.jpg"
    if not base_jpg.exists():
        return []

    mat = loadmat(str(mat_path), variable_names=["pt2d", "Pose_Para"])
    pt2d = np.asarray(mat["pt2d"], dtype=np.float64)          # (2, 68)
    pose_para = np.asarray(mat["Pose_Para"], dtype=np.float64).reshape(-1)
    pitch, yaw, roll = (float(a) for a in pose_para[:3])

    with Image.open(base_jpg) as im:
        w, h = im.size

    pts_f, (ox, oy) = correct_300wlp_pt2d(pt2d)
    points = pts_f.T  # (68, 2)
    flip_baked = "_Flip_" in folder.name
    if flip_baked:
        points = points[_FLIP_PERM]
    n_out, max_over = count_points_outside(points, (w, h))
    if max_over <= _PASS_PX:
        offset_check = "pass"
    elif max_over <= _CLIPPED_PX:
        offset_check = "clipped"
    else:
        offset_check = "fail"
    # 枠外に出た点は可視性 0(画像外)とする
    visibility = [
        0 if (x < 0 or y < 0 or x >= w or y >= h) else VIS_UNKNOWN
        for x, y in points
    ]

    R = euler300wlp_to_rotmat(pitch, yaw, roll)
    pose = {
        "rotation_matrix": [[round(float(v), 6) for v in row] for row in R],
        "euler_deg": {
            "pitch": round(math.degrees(pitch), 3),
            "yaw": round(math.degrees(yaw), 3),
            "roll": round(math.degrees(roll), 3),
        },
        "source": "300wlp_gt",
    }
    head_bbox = head_bbox_from_landmarks(points)
    subset = folder.name.split("_")[0]
    quality = {
        "offset_check": offset_check,
        "n_points_outside": n_out,
        "max_overshoot_px": round(max_over, 2),
        "crop_offset": [round(ox, 2), round(oy, 2)],
    }

    base_id = f"300wlp/{folder.name}/{stem}"
    masked_id = f"{base_id}_masked" if masked_jpg.exists() else None

    common = dict(
        image_size=(w, h),
        source_dataset="300W_LP_w_masked",
        license_tag="research_only",
        split="train",
        head_bbox=head_bbox,
        head_bbox_source="landmark_derived",
        face_bbox=None,
        scheme="ibug68",
        points=points,
        visibility=visibility,
        pose=pose,
        quality=quality,
    )
    records = [
        make_record(
            record_id=base_id,
            image_path=f"images/300wlp/{folder.name}/{stem}.jpg",
            attributes={"subset": subset, "flip_baked": flip_baked, "mask_worn": False},
            paired_record=masked_id,
            **common,
        )
    ]
    if masked_id is not None:
        with Image.open(masked_jpg) as im:
            mw, mh = im.size
        if (mw, mh) != (w, h):
            # 画像サイズ不一致はアノテーション共有の前提が崩れるため除外して記録
            rec = records[0]
            rec["quality"]["masked_size_mismatch"] = [mw, mh]
            rec["paired_record"] = None
            return records
        records.append(
            make_record(
                record_id=masked_id,
                image_path=f"images/300wlp/{folder.name}/{stem}_masked.jpg",
                attributes={"subset": subset, "flip_baked": flip_baked, "mask_worn": True},
                paired_record=base_id,
                **common,
            )
        )
    return records


def convert_folder(folder: Path, limit: int | None = None) -> list[dict]:
    stems = sorted(
        p.stem for p in folder.glob("*.mat") if not p.stem.endswith("_masked")
    )
    if limit is not None:
        stems = stems[:limit]
    out: list[dict] = []
    for stem in stems:
        out.extend(_convert_one_stem(folder, stem))
    return out


def iter_records(root: Path, limit: int | None = None, workers: int = 8):
    """全フォルダを並列変換しレコードを順に返す。limit はフォルダごとの上限。"""
    folders = list_folders(root)
    if workers <= 1:
        for folder in folders:
            yield from convert_folder(folder, limit)
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for recs in ex.map(convert_folder, folders, [limit] * len(folders)):
            yield from recs
