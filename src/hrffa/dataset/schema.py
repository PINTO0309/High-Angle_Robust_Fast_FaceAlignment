"""統合データセットスキーマ v0.1 とバリデータ。

仕様の出典: history/001_implementation_plan.md §5(ドラフト)に対し、
本実装で `split` フィールドを追加した(v0.1)。変更点は history/002 に記録。

1 レコード = 1 頭部インスタンス。JSONL で保存する。
"""

from __future__ import annotations

import math
from typing import Any

SCHEMA_VERSION = "0.1"

# 点 ID 名前空間(scheme)と点数。相互変換はデータセット側では行わない。
SCHEMES: dict[str, int] = {
    "ibug68": 68,
    "wflw98": 98,
    "cofw29": 29,
}

# 可視性: 2=可視 1=遮蔽 0=画像外 -1=不明
VIS_VISIBLE = 2
VIS_OCCLUDED = 1
VIS_OUT_OF_IMAGE = 0
VIS_UNKNOWN = -1
VIS_VALUES = {VIS_VISIBLE, VIS_OCCLUDED, VIS_OUT_OF_IMAGE, VIS_UNKNOWN}

LICENSE_TAGS = {
    "research_only",   # 300W-LP / WFLW / 300W / COFW 系(配布元が研究用途前提)
    "apache-2.0",
    "mit",
    "bsd-3-clause",
    "unknown",
}

HEAD_BBOX_SOURCES = {
    "deimv2_pseudo",     # D2 で付与(主系統)
    "landmark_derived",  # ランドマークからの外挿(D1 時点のプレースホルダ/フォールバック)
    "manual",
}

POSE_SOURCES = {
    "300wlp_gt",
    "300wlp_gt_warped",  # 深度再投影による厳密回転更新(D4b)
    "deimv2_dir8_weak",
    "teacher_pseudo",
}

SPLITS = {"train", "test", "valid", "valid_common", "valid_challenge"}

DIRECTION8 = {
    "front", "right_front", "right_side", "right_back",
    "back", "left_back", "left_side", "left_front",
}

SOURCE_DATASETS = {
    "300W_LP_w_masked",
    "ORFormer_WFLW",
    "ORFormer_300W",
    "ORFormer_COFW",
    "synth_dwarp_300wlp",  # D4b 深度再投影による合成(親は 300W_LP_w_masked)
    "selftrain_v1",        # T2 教師 self-training(実写未ラベル頭部への擬似ラベル)
    "selftrain_v2",        # T2 生成画像(gpt-image)への教師擬似ラベル
    "selftrain_lookup",    # T2 見上げ特化生成画像(synthetic_lookup)への教師擬似ラベル
}

_REQUIRED_KEYS = [
    "record_id", "image_path", "image_size", "source_dataset", "license_tag",
    "split", "head_bbox", "head_bbox_source", "face_bbox",
    "landmarks", "pose", "direction8", "attributes", "paired_record", "quality",
]


def _is_bbox(v: Any) -> bool:
    return (
        isinstance(v, (list, tuple)) and len(v) == 4
        and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)
        and v[2] > v[0] and v[3] > v[1]
    )


def validate_record(rec: dict) -> list[str]:
    """1 レコードを検証しエラーメッセージのリストを返す(空なら合格)。"""
    errs: list[str] = []
    rid = rec.get("record_id", "<no-id>")

    for k in _REQUIRED_KEYS:
        if k not in rec:
            errs.append(f"{rid}: missing key '{k}'")
    if errs:
        return errs

    if not isinstance(rec["record_id"], str) or not rec["record_id"]:
        errs.append(f"{rid}: record_id must be non-empty str")
    if not isinstance(rec["image_path"], str) or not rec["image_path"]:
        errs.append(f"{rid}: image_path must be non-empty str")

    size = rec["image_size"]
    if not (isinstance(size, (list, tuple)) and len(size) == 2
            and all(isinstance(x, int) and x > 0 for x in size)):
        errs.append(f"{rid}: image_size must be [w, h] positive ints")

    if rec["source_dataset"] not in SOURCE_DATASETS:
        errs.append(f"{rid}: unknown source_dataset '{rec['source_dataset']}'")
    if rec["license_tag"] not in LICENSE_TAGS:
        errs.append(f"{rid}: unknown license_tag '{rec['license_tag']}'")
    if rec["split"] not in SPLITS:
        errs.append(f"{rid}: unknown split '{rec['split']}'")

    if not _is_bbox(rec["head_bbox"]):
        errs.append(f"{rid}: head_bbox invalid: {rec['head_bbox']}")
    if rec["head_bbox_source"] not in HEAD_BBOX_SOURCES:
        errs.append(f"{rid}: unknown head_bbox_source '{rec['head_bbox_source']}'")
    if rec["face_bbox"] is not None and not _is_bbox(rec["face_bbox"]):
        errs.append(f"{rid}: face_bbox invalid: {rec['face_bbox']}")

    lm = rec["landmarks"]
    if not isinstance(lm, dict):
        errs.append(f"{rid}: landmarks must be dict")
    else:
        scheme = lm.get("scheme")
        if scheme not in SCHEMES:
            errs.append(f"{rid}: unknown landmark scheme '{scheme}'")
        else:
            n = SCHEMES[scheme]
            pts = lm.get("points")
            vis = lm.get("visibility")
            if not (isinstance(pts, list) and len(pts) == n
                    and all(isinstance(p, (list, tuple)) and len(p) == 2
                            and all(isinstance(c, (int, float)) and math.isfinite(c) for c in p)
                            for p in pts)):
                errs.append(f"{rid}: landmarks.points must be {n}x2 finite floats")
            if not (isinstance(vis, list) and len(vis) == n
                    and all(v in VIS_VALUES for v in vis)):
                errs.append(f"{rid}: landmarks.visibility must be {n} values in {sorted(VIS_VALUES)}")

    pose = rec["pose"]
    if pose is not None:
        if not isinstance(pose, dict):
            errs.append(f"{rid}: pose must be dict or null")
        else:
            if pose.get("source") not in POSE_SOURCES:
                errs.append(f"{rid}: unknown pose.source '{pose.get('source')}'")
            R = pose.get("rotation_matrix")
            if R is not None:
                ok = (isinstance(R, list) and len(R) == 3
                      and all(isinstance(r, list) and len(r) == 3 for r in R))
                if ok:
                    # 直交性と det=+1 を許容誤差付きで検証
                    import numpy as np
                    Rm = np.asarray(R, dtype=float)
                    if not np.allclose(Rm @ Rm.T, np.eye(3), atol=1e-4):
                        errs.append(f"{rid}: rotation_matrix not orthonormal")
                    elif abs(float(np.linalg.det(Rm)) - 1.0) > 1e-4:
                        errs.append(f"{rid}: rotation_matrix det != +1")
                else:
                    errs.append(f"{rid}: rotation_matrix must be 3x3")
            e = pose.get("euler_deg")
            if e is not None and not (
                isinstance(e, dict) and all(
                    isinstance(e.get(k), (int, float)) and math.isfinite(e[k])
                    for k in ("pitch", "yaw", "roll"))
            ):
                errs.append(f"{rid}: euler_deg must have finite pitch/yaw/roll")

    if rec["direction8"] is not None and rec["direction8"] not in DIRECTION8:
        errs.append(f"{rid}: unknown direction8 '{rec['direction8']}'")
    if not isinstance(rec["attributes"], dict):
        errs.append(f"{rid}: attributes must be dict")
    if rec["paired_record"] is not None and not isinstance(rec["paired_record"], str):
        errs.append(f"{rid}: paired_record must be str or null")
    if not isinstance(rec["quality"], dict):
        errs.append(f"{rid}: quality must be dict")

    return errs
