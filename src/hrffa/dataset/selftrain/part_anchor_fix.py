"""DEIMv2 部位アンカーによる擬似ラベルの自動補正・整合フラグ付与(history/033 §6.3)。

対象: selftrain 系 jsonl(lookup / v2 / v1)。マスク遮蔽を考慮した設計:
  - 目(17): 位置補正アンカー(マスク頑健 96.3%)。クラスタ重心を検出中心へ
    保守的に平行移動(閾値超のみ・上限あり・形状保持)。補正後に眉との順序が
    崩れる側は眉クラスタも同量移動
  - 口(19): 「検出されたときだけ」位置整合フラグ。未検出は棄却理由にしない
    (マスクで 3.3% まで低下するため)。高い分離度(99.0% vs 3.3%)を可視性補正に
    転用: 顔・目あり+口未検出 → 口を遮蔽へ / 口が高スコア検出+全点遮蔽 → 可視へ
  - 鼻(18): 検出時の横位置フラグのみ(マスク下 40.3% の曖昧領域のため存在有無は不使用。
    縦方向は意味論的オフセットがあるため使用しない)
  - 耳(20): 本ツールでは記録のみ(yaw 符号ゲートは姿勢推定を持つ工程で使用)

検出結果は annotations/<source>.part_anchor.dets.jsonl にキャッシュ(再実行高速)。
補正は attributes.part_anchor マーカーで冪等。

usage:
  uv run python -m hrffa.dataset.selftrain.part_anchor_fix \\
    --sources selftrain_lookup selftrain_v2 selftrain_v1 [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

EYE, NOSE, MOUTH, EAR, FACE, HEAD = 17, 18, 19, 20, 16, 7
EYE_L = list(range(36, 42))   # 被写体右目(画像左寄り想定はしない — 位置で対応付け)
EYE_R = list(range(42, 48))
BROW_L = list(range(17, 22))
BROW_R = list(range(22, 27))
MOUTH_LM = list(range(48, 68))

MARK = "part_anchor"
VERSION = 1


def _detector(score_thr: float):
    from ..pseudolabel.deimv2 import Deimv2Detector
    model = Path("data/models/deimv2_wholebody49_boxes_only.onnx")
    try:
        return Deimv2Detector(model, score_threshold=score_thr)
    except Exception:
        print("(GPU init failed -> CPU fallback)")
        return Deimv2Detector(model, providers=["CPUExecutionProvider"],
                              score_threshold=score_thr)


def _load_dets(unified: Path, source: str, records: list[dict],
               score_thr: float, batch: int = 4) -> dict[str, list]:
    """image_path → dets。キャッシュがあれば利用し、不足分のみ推論して追記する。"""
    cache = unified / "annotations" / f"{source}.part_anchor.dets.jsonl"
    dets: dict[str, list] = {}
    if cache.exists():
        for line in open(cache, encoding="utf-8"):
            r = json.loads(line)
            dets[r["image_path"]] = r["dets"]
    todo = [r["image_path"] for r in records if r["image_path"] not in dets]
    if todo:
        det = _detector(score_thr)
        with open(cache, "a", encoding="utf-8") as f:
            from tqdm import tqdm
            for i0 in tqdm(range(0, len(todo), batch), desc=f"detect {source}",
                           dynamic_ncols=True):
                paths = todo[i0:i0 + batch]
                imgs = [cv2.imread(str(unified / p)) for p in paths]
                try:
                    outs = det.infer_batch(imgs)
                except Exception:
                    from ..pseudolabel.deimv2 import Deimv2Detector
                    model = Path("data/models/deimv2_wholebody49_boxes_only.onnx")
                    if not getattr(det, "_gpu_retried", False):
                        # まず GPU セッションを 1 回作り直して継続を試みる
                        print("(inference failed -> retrying with a new GPU session)")
                        try:
                            det = Deimv2Detector(model, score_threshold=score_thr)
                            det._gpu_retried = True
                            outs = det.infer_batch(imgs)
                        except Exception:
                            print("(GPU retry failed too -> continuing on CPU)")
                            det = Deimv2Detector(
                                model, providers=["CPUExecutionProvider"],
                                score_threshold=score_thr)
                            outs = det.infer_batch(imgs)
                    else:
                        print("(inference failed -> continuing on CPU)")
                        det = Deimv2Detector(
                            model, providers=["CPUExecutionProvider"],
                            score_threshold=score_thr)
                        outs = det.infer_batch(imgs)
                for p, d in zip(paths, outs):
                    dets[p] = d
                    f.write(json.dumps({"image_path": p, "dets": d}) + "\n")
                f.flush()
    return dets


def _center(box) -> np.ndarray:
    return np.array([(box[1] + box[3]) / 2, (box[2] + box[4]) / 2])


def fix_record(rec: dict, dets: list, eye_min: float, eye_cap: float,
               mouth_flag_thr: float, nose_flag_thr: float,
               mouth_score_vis: float) -> dict | None:
    """1 レコードを補正し、変更概要(なければ None)を返す。points/visibility を書換える。"""
    x1, y1, x2, y2 = rec["head_bbox"]
    side = max(x2 - x1, y2 - y1)
    pts = np.asarray(rec["landmarks"]["points"], dtype=np.float64)
    vis = list(rec["landmarks"]["visibility"])
    def _inside(b, margin=0.2):
        bx, by = (b[1] + b[3]) / 2, (b[2] + b[4]) / 2
        return (x1 - side * margin <= bx <= x2 + side * margin and
                y1 - side * margin <= by <= y2 + side * margin)

    by_cls: dict[int, list] = {}
    for d in dets:
        if _inside(d):   # 対象頭部 bbox 内の検出のみ使用(複数人クロップ対策)
            by_cls.setdefault(int(d[0]), []).append(d)

    info: dict = {"version": VERSION}
    changed = False

    # --- 目の位置補正(+必要時に同側の眉も追従)---
    eyes = by_cls.get(EYE, [])
    for eye_lm, brow_lm, tag in ((EYE_L, BROW_L, "l"), (EYE_R, BROW_R, "r")):
        if not eyes:
            break
        c = pts[eye_lm].mean(0)
        best = min(eyes, key=lambda d: float(((_center(d) - c) ** 2).sum()))
        delta = _center(best) - c
        if abs(delta[0]) > side * 0.25 or abs(delta[1]) > side * 0.12:
            info[f"eye_{tag}_flag"] = "far"      # 対応付け不能 — 補正せずフラグ
            continue
        mag = float(np.hypot(*delta))
        if mag <= side * eye_min:
            continue
        if mag > side * eye_cap:
            delta = delta * (side * eye_cap / mag)
        pts[eye_lm] += delta
        info[f"eye_{tag}_dxy"] = [round(float(delta[0]) / side, 4),
                                  round(float(delta[1]) / side, 4)]
        changed = True
        # 眉が補正後の目と交差する場合のみ同量移動(順序保持)
        if pts[brow_lm].mean(0)[1] > pts[eye_lm].mean(0)[1] - side * 0.005:
            pts[brow_lm] += delta
            info[f"brow_{tag}_follow"] = True

    # --- 口: 検出時のみ位置フラグ / 可視性の双方向補正 ---
    mouths = by_cls.get(MOUTH, [])
    has_context = bool(by_cls.get(FACE) or by_cls.get(HEAD)) and bool(eyes)
    mc = pts[MOUTH_LM].mean(0)
    if mouths:
        best = min(mouths, key=lambda d: float(((_center(d) - mc) ** 2).sum()))
        dm = float(np.hypot(*(_center(best) - mc)))
        if dm > side * mouth_flag_thr:
            info["mouth_flag"] = round(dm / side, 4)
        if dm <= side * 0.35 and best[5] >= mouth_score_vis \
                and all(vis[j] <= 1 for j in MOUTH_LM):
            for j in MOUTH_LM:
                vis[j] = 2
            info["mouth_vis"] = "to_visible"
            changed = True
    elif has_context:
        flipped = [j for j in MOUTH_LM if vis[j] == 2]
        if flipped:
            for j in flipped:
                vis[j] = 1
            info["mouth_vis"] = "to_occluded"
            changed = True

    # --- 鼻: 検出時の横位置フラグのみ ---
    noses = by_cls.get(NOSE, [])
    if noses:
        nc = pts[list(range(31, 36))].mean(0)
        best = min(noses, key=lambda d: float(((_center(d) - nc) ** 2).sum()))
        if abs(_center(best)[0] - nc[0]) > side * nose_flag_thr:
            info["nose_flag_dx"] = round(float(_center(best)[0] - nc[0]) / side, 4)

    if not changed and len(info) <= 1:
        return None
    rec["landmarks"]["points"] = [[round(float(x), 2), round(float(y), 2)]
                                  for x, y in pts]
    rec["landmarks"]["visibility"] = vis
    rec["attributes"][MARK] = info
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-correct pseudo labels with DEIMv2 part anchors and attach consistency flags (history/033 section 6.3).")
    ap.add_argument("--unified", type=Path, default=Path("datasets/unified"))
    ap.add_argument("--sources", nargs="+",
                    default=["selftrain_lookup", "selftrain_v2", "selftrain_v1"])
    ap.add_argument("--score-thr", type=float, default=0.35)
    ap.add_argument("--batch", type=int, default=4,
                    help="detection batch size (smaller is more stable for the GPU arena)")
    ap.add_argument("--repair-bbox", action="store_true",
                    help="revert mouth-visibility flips that were based on out-of-bbox detections to 1 (occluded) "
                         "and drop the nose/mouth flags derived from out-of-bbox detections (history/033 §7.4)")
    ap.add_argument("--revert-eyes", action="store_true",
                    help="revert only the eye/brow position corrections (keep the mouth-visibility corrections)")
    ap.add_argument("--keep-pitch-over", type=float, default=None,
                    help="with --revert-eyes, keep the corrections for records whose "
                         "file-name |pitch| is at least this value")
    ap.add_argument("--eye-min", type=float, default=0.01,
                    help="minimum offset (ratio of head side) that triggers a correction")
    ap.add_argument("--eye-cap", type=float, default=0.06,
                    help="upper bound of the correction (ratio of head side)")
    ap.add_argument("--mouth-flag-thr", type=float, default=0.08)
    ap.add_argument("--nose-flag-thr", type=float, default=0.10)
    ap.add_argument("--mouth-score-vis", type=float, default=0.60,
                    help="detection score at or above which all-occluded labels are reverted to visible")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import re as _re
    _pitch_pat = _re.compile(r"pitch([+-]\d+)")

    for source in args.sources:
        path = args.unified / "annotations" / f"{source}.jsonl"
        if not path.exists():
            print(f"{source}: no jsonl (skipped)")
            continue
        records = [json.loads(l) for l in open(path, encoding="utf-8")]
        if args.repair_bbox:
            dets_map = _load_dets(args.unified, source, [], args.score_thr)
            fixed_vis = cleared = 0
            for r in records:
                pa = r["attributes"].get(MARK)
                if not pa:
                    continue
                x1, y1, x2, y2 = r["head_bbox"]
                side = max(x2 - x1, y2 - y1)
                d = dets_map.get(r["image_path"], [])

                def _ok(b, mc=None, near=None):
                    bx, by = (b[1] + b[3]) / 2, (b[2] + b[4]) / 2
                    if not (x1 - side*0.2 <= bx <= x2 + side*0.2 and
                            y1 - side*0.2 <= by <= y2 + side*0.2):
                        return False
                    if mc is not None and near is not None:
                        return float(np.hypot(bx - mc[0], by - mc[1])) <= side * near
                    return True

                pts = np.asarray(r["landmarks"]["points"], float)
                mc = pts[MOUTH_LM].mean(0)
                if pa.get("mouth_vis") == "to_visible":
                    good = [b for b in d if int(b[0]) == MOUTH and b[5] >= 0.6
                            and _ok(b, mc, 0.35)]
                    if not good:
                        vis = list(r["landmarks"]["visibility"])
                        for j in MOUTH_LM:
                            vis[j] = 1   # 近似復元(旧値 0/1 の別は失われている)
                        r["landmarks"]["visibility"] = vis
                        pa["mouth_vis"] = "reverted_bbox"
                        fixed_vis += 1
                if "nose_flag_dx" in pa:
                    noses = [b for b in d if int(b[0]) == NOSE and b[5] >= args.score_thr
                             and _ok(b)]
                    if not noses:
                        pa.pop("nose_flag_dx"); cleared += 1
                if "mouth_flag" in pa:
                    mo = [b for b in d if int(b[0]) == MOUTH and b[5] >= args.score_thr
                          and _ok(b)]
                    if not mo:
                        pa.pop("mouth_flag"); cleared += 1
                pa["repair"] = "bbox_filter_v2"
            print(f"{source}: mouth-visibility reverts {fixed_vis} / invalid flags removed {cleared}")
            if not args.dry_run:
                tmp = path.with_suffix(".jsonl.tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    for r in records:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                tmp.replace(path)
            continue
        if args.revert_eyes:
            reverted = kept = 0
            for r in records:
                pa = r["attributes"].get(MARK)
                if not pa or not any(k in pa for k in
                                     ("eye_l_dxy", "eye_r_dxy")):
                    continue
                if args.keep_pitch_over is not None:
                    m = _pitch_pat.search(r["image_path"])
                    if m and abs(int(m.group(1))) >= args.keep_pitch_over:
                        kept += 1
                        continue
                x1, y1, x2, y2 = r["head_bbox"]
                side = max(x2 - x1, y2 - y1)
                pts = np.asarray(r["landmarks"]["points"], dtype=np.float64)
                for tag, eye_lm, brow_lm in (("l", EYE_L, BROW_L),
                                             ("r", EYE_R, BROW_R)):
                    d = pa.pop(f"eye_{tag}_dxy", None)
                    if d is None:
                        continue
                    delta = np.array(d, dtype=np.float64) * side
                    pts[eye_lm] -= delta
                    if pa.pop(f"brow_{tag}_follow", None):
                        pts[brow_lm] -= delta
                r["landmarks"]["points"] = [[round(float(x), 2), round(float(y), 2)]
                                            for x, y in pts]
                reverted += 1
            print(f"{source}: eye corrections reverted {reverted} (kept {kept})")
            if not args.dry_run:
                tmp = path.with_suffix(".jsonl.tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    for r in records:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                tmp.replace(path)
            continue
        pending = [r for r in records
                   if r["attributes"].get(MARK, {}).get("version") != VERSION]
        dets = _load_dets(args.unified, source, pending, args.score_thr,
                          batch=args.batch)
        stats = {"eye_fixed": 0, "eye_flag": 0, "brow_follow": 0,
                 "mouth_to_occluded": 0, "mouth_to_visible": 0,
                 "mouth_flag": 0, "nose_flag": 0}
        shifts = []
        for r in pending:
            info = fix_record(r, dets.get(r["image_path"], []),
                              args.eye_min, args.eye_cap, args.mouth_flag_thr,
                              args.nose_flag_thr, args.mouth_score_vis)
            if not info:
                r["attributes"][MARK] = {"version": VERSION}
                continue
            for tag in ("l", "r"):
                if f"eye_{tag}_dxy" in info:
                    stats["eye_fixed"] += 1
                    shifts.append(np.hypot(*info[f"eye_{tag}_dxy"]))
                if f"eye_{tag}_flag" in info:
                    stats["eye_flag"] += 1
                if info.get(f"brow_{tag}_follow"):
                    stats["brow_follow"] += 1
            if info.get("mouth_vis") == "to_occluded":
                stats["mouth_to_occluded"] += 1
            if info.get("mouth_vis") == "to_visible":
                stats["mouth_to_visible"] += 1
            if "mouth_flag" in info:
                stats["mouth_flag"] += 1
            if "nose_flag_dx" in info:
                stats["nose_flag"] += 1
        tag = "(dry-run)" if args.dry_run else ""
        mean_shift = float(np.mean(shifts)) if shifts else 0.0
        print(f"{source}: processed {len(pending)}/{len(records)} {tag}\n"
              f"  eye corrections {stats['eye_fixed']} (mean shift {mean_shift:.3f} x side)"
              f" unfixable flags {stats['eye_flag']} brow follow {stats['brow_follow']}\n"
              f"  mouth visibility: to occluded {stats['mouth_to_occluded']} / to visible {stats['mouth_to_visible']}"
              f" mouth-position flags {stats['mouth_flag']} nose-lateral flags {stats['nose_flag']}")
        if not args.dry_run:
            tmp = path.with_suffix(".jsonl.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmp.replace(path)


if __name__ == "__main__":
    main()
