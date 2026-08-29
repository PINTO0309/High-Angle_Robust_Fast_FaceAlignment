"""左右整合監査(history/033 §6.4)— キャッシュのみで回るデータ健全性モニタ。

2 変種の符号整合チェック(大 |yaw| レコード対象):
  ① 耳側整合: 検出された耳が頭部中心の左右どちらにあるか vs yaw 符号
     → 姿勢符号系の故障(変換器の符号ミス・EXIF 回転・ミラー混入)を検出
  ② 目番号整合: 検出された可視の目に近い番号クラスタ(36-41 / 42-47)vs yaw 符号
     → ランドマーク左右意味論の故障(020 型の Flip 番号入替漏れ)を検出

規約(D2 で GT 検証済み): yaw > 0 = 被写体の右側が可視 = 可視目は 36-41、
被写体は画像左を向く = 可視耳は頭部中心より画像右。

yaw の解決順: pose GT euler → 6DRepNet サイドカー → ファイル名 yaw±(生成系)。
検出は <source>.part_anchor.dets.jsonl / <source>.deimv2.jsonl キャッシュを使用
(キャッシュがないソースはスキップ。GPU 不要)。

usage:
  uv run python -m hrffa.dataset.qa.lr_consistency_audit
  uv run python -m hrffa.dataset.qa.lr_consistency_audit --simulate-020   # 感度自己検証
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

EYE, EAR = 17, 20
# scheme 別の目クラスタ(R = 被写体右目 = yaw>0 で可視)
EYE_IDX = {
    "ibug68": (list(range(36, 42)), list(range(42, 48))),
    "wflw98": (list(range(60, 68)), list(range(68, 76))),
    "cofw29": ([8, 10, 12, 13], [9, 11, 14, 15]),
}
_FNAME_YAW = re.compile(r"yaw([+-]\d+)")
DIR8_RIGHT = {9, 10, 11}   # right_front / right_side / right_back(D2 検証済み規約)
DIR8_LEFT = {13, 14, 15}

# ibug68 の左右入替(--simulate-020 用)
_FLIP68 = (list(range(16, -1, -1)) + list(range(26, 16, -1)) + [27, 28, 29, 30]
           + [35, 34, 33, 32, 31] + [45, 44, 43, 42, 47, 46]
           + [39, 38, 37, 36, 41, 40]
           + [54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55,
              64, 63, 62, 61, 60, 67, 66, 65])


def _load_dets(unified: Path, source: str) -> dict[str, list] | None:
    for name in (f"{source}.part_anchor.dets.jsonl", f"{source}.deimv2.jsonl"):
        p = unified / "annotations" / name
        if p.exists():
            out = {}
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                out[r["image_path"]] = r["dets"]
            return out
    return None


def _sixd_meta(unified: Path, source: str) -> dict[str, list] | None:
    """6DRepNet サイドカー(record_id → [yaw, pitch, roll])。キャッシュのみ参照。"""
    hits = list((unified / "annotations").glob(f"sixd_pose_{source}_*.jsonl"))
    if not hits:
        return None
    out = {}
    for p in hits:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            out[r["record_id"]] = r["ypr"]
    return out


def _resolve_side(rec: dict, sixd: dict | None, dets: list) -> tuple[float, bool] | None:
    """(|yaw|, 右側可視か) を返す。生成系はファイル名符号が信頼できないため、
    dir8 検出クラス(画像由来)から側を決め、ファイル名 yaw は帯域判定のみに使う。"""
    pose = rec.get("pose") or {}
    if pose.get("euler_deg"):
        yaw = float(pose["euler_deg"]["yaw"])
        return abs(yaw), yaw > 0
    if sixd and rec["record_id"] in sixd:
        yaw = float(sixd[rec["record_id"]][0])
        return abs(yaw), yaw > 0
    m = _FNAME_YAW.search(rec["image_path"])
    if m:
        mag = abs(float(m.group(1)))
        d8 = {int(b[0]) for b in dets if int(b[0]) in DIR8_RIGHT | DIR8_LEFT
              and b[5] >= 0.4}
        right = bool(d8 & DIR8_RIGHT)
        left = bool(d8 & DIR8_LEFT)
        if right != left:          # 片側のみ確定したときだけ判定に使う
            return mag, right
    return None


def audit_source(unified: Path, source: str, yaw_min: float, yaw_max: float,
                 score_thr: float, simulate_020: bool, max_n: int | None) -> None:
    path = unified / "annotations" / f"{source}.jsonl"
    if not path.exists():
        print(f"{source:18s} no jsonl (skipped)")
        return
    dets = _load_dets(unified, source)
    if dets is None:
        print(f"{source:18s} no detection cache (skipped)")
        return
    sixd = _sixd_meta(unified, source)

    n1 = bad1 = n2 = bad2 = 0
    bad_ids: list[str] = []
    seen = 0
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        d = dets.get(rec["image_path"])
        if d is None:
            continue
        resolved = _resolve_side(rec, sixd, d)
        if resolved is None:
            continue
        yaw_abs, expect_right_visible = resolved
        if not (yaw_min <= yaw_abs <= yaw_max):
            continue
        seen += 1
        if max_n and seen > max_n:
            break
        scheme = rec["landmarks"]["scheme"]
        if scheme not in EYE_IDX:
            continue
        r_idx, l_idx = EYE_IDX[scheme]
        x1, y1, x2, y2 = rec["head_bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(x2 - x1, y2 - y1)
        pts = np.asarray(rec["landmarks"]["points"], dtype=np.float64)
        if simulate_020 and rec["attributes"].get("flip_baked"):
            pts = pts[_FLIP68]

        def inside(b, margin=0.2):
            bx, by = (b[1] + b[3]) / 2, (b[2] + b[4]) / 2
            return (x1 - side * margin <= bx <= x2 + side * margin and
                    y1 - side * margin <= by <= y2 + side * margin)

        # ① 耳側整合(頭部 bbox 内の検出のみ — 群衆画像の他人を除外)
        ears = [b for b in d if int(b[0]) == EAR and b[5] >= score_thr and inside(b)]
        if ears:
            best = max(ears, key=lambda b: b[5])
            ex = (best[1] + best[3]) / 2 - cx
            if abs(ex) > side * 0.05:      # 中央付近は判定除外
                n1 += 1
                if (ex > 0) != expect_right_visible:
                    bad1 += 1

        # ② 目番号整合(|yaw|≥60 の真の片目域のみ。40-60 は両目可視で前提が崩れる)
        eyes = [b for b in d if int(b[0]) == EYE and b[5] >= score_thr and inside(b)]
        if yaw_abs >= 60 and len(eyes) == 1:
            b = eyes[0]
            ec = np.array([(b[1] + b[3]) / 2, (b[2] + b[4]) / 2])
            d_r = float(np.linalg.norm(pts[r_idx].mean(0) - ec))
            d_l = float(np.linalg.norm(pts[l_idx].mean(0) - ec))
            if abs(d_r - d_l) > side * 0.10:   # 曖昧対応は判定除外
                n2 += 1
                if (d_r < d_l) != expect_right_visible:
                    bad2 += 1
                    if len(bad_ids) < 10:
                        bad_ids.append(rec["record_id"])

    r1 = bad1 / n1 if n1 else 0.0
    r2 = bad2 / n2 if n2 else 0.0
    a1 = " ⚠" if r1 > 0.02 else ""
    a2 = " ⚠" if r2 > 0.02 else ""
    print(f"{source:18s} (1) ear side {bad1:>5}/{n1:<6} ({r1:6.2%}){a1}  "
          f"(2) eye index {bad2:>5}/{n2:<6} ({r2:6.2%}){a2}")
    if bad_ids:
        print(f"{'':18s} (2) conflicting examples: {', '.join(bad_ids[:5])}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Left/right consistency audit (history/033 section 6.4) - a cache-only data-health monitor.")
    ap.add_argument("--unified", type=Path, default=Path("datasets/unified"))
    ap.add_argument("--sources", nargs="+",
                    default=["300wlp", "wflw", "300w", "cofw",
                             "selftrain_lookup", "selftrain_v2", "selftrain_v1"])
    ap.add_argument("--yaw-min", type=float, default=40.0)
    ap.add_argument("--yaw-max", type=float, default=100.0)
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--max-n", type=int, default=None,
                    help="max records judged per source (sampling)")
    ap.add_argument("--simulate-020", action="store_true",
                    help="re-inject the 020 defect (missing index swap) into the 300wlp flip records "
                         "in memory to self-check the audit's sensitivity")
    args = ap.parse_args()

    print(f"left/right consistency audit |yaw| in [{args.yaw_min:.0f}, {args.yaw_max:.0f}]"
          f"{' (020 defect injected)' if args.simulate_020 else ''}")
    for source in args.sources:
        audit_source(args.unified, source, args.yaw_min, args.yaw_max,
                     args.score_thr, args.simulate_020, args.max_n)


if __name__ == "__main__":
    main()
