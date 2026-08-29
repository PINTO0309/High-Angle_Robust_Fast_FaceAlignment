"""【廃止 — 使用禁止】(history/037 §9・036 §10)

本ツールの前提「可視眼は常に L クラスタ」は誤りだった(実写規約は鏡映等変:
可視は yaw+ で L / yaw− で R、幻は常に鼻側)。2026-08-26 に全修復を revert 済み。
det 監査の設計・誤爆防止ゲート較正の記録として残置する。

--- 以下は当時の記述(前提が誤り) ---
selftrain プール擬似ラベルの鏡像破れ外科修復(history/037)。

破れ: 強横顔で「可視眼が R クラスタ + 幻 L が鼻側」(統一規約は可視 = 常に L、
幻 R = 可視 − fold×hb 幅)。034 v1 後の教師の綱引き出力が焼き付いたもの。

判定(側非依存 — 統一規約の「可視 = L」は両側共通のため側解決不要。
v1 のような side 未解決ソースにもそのまま適用可能):
  1. head_bbox 内(margin 0.2×side)の眼 det(score ≥ --score-thr)がちょうど 1 個
  2. 眼クラスタ分離 |R−L|x < sep 上限 × hb 幅(折り畳み幾何の存在)。
     上限は鼻 det の側方性 3 ゾーンで切替(実測: 真横顔の鼻off 0.23〜0.38、
     3/4 顔 0.10〜0.20、閉眼・細目の片目 det 落ちが 3/4 に多発):
       鼻off ≥ 0.20(真横顔確定)→ 0.26 / 0.10〜0.20 と鼻なし → --sep-max(0.17)
  3. 鼻 det が中央(≤--nose-lat=0.10)なら正面〜3/4 と判断して対象外
  4. det 中心の最近傍クラスタ判定が曖昧でない(距離差 > 0.10×side)
  5. 最近傍が R クラスタ = 破れ → 修復

修復(fix_yaw_semantics.convert と同型):
  - 可視座標(旧 R)と vis を L index(42-47 / 22-26)へ移す
  - R index は「新 L を −x へ fold_hb×hb 幅」の折り畳み、vis=1(画像左端 2px クランプ)
  - 旧 L(鼻側の幻)は破棄、attributes にバックアップ(--revert で完全復元)

冪等: attributes.pool_mirror_fix でスキップ。

usage:
  uv run python -m hrffa.dataset.selftrain.pool_mirror_fix --source selftrain_v2
      [--fold-hb 0.15] [--score-thr 0.5] [--sep-max 0.22] [--dry-run | --revert]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EYE = 17
NOSE = 18
EYE_R = list(range(36, 42))
EYE_L = list(range(42, 48))
BROW_R = list(range(17, 22))
BROW_L = list(range(22, 27))
MARK = "pool_mirror_fix"
VALUE = "unified_v1"


def is_broken(rec: dict, dets: list, score_thr: float, sep_max: float,
              nose_lat: float = 0.10) -> bool:
    x1, y1, x2, y2 = rec["head_bbox"]
    side = max(x2 - x1, y2 - y1)
    hbw = x2 - x1
    eyes, noses = [], []
    for b in dets:
        cls = int(b[0])
        if cls not in (EYE, NOSE) or b[5] < score_thr:
            continue
        cx, cy = (b[1] + b[3]) / 2, (b[2] + b[4]) / 2
        if (x1 - side * 0.2 <= cx <= x2 + side * 0.2
                and y1 - side * 0.2 <= cy <= y2 + side * 0.2):
            (eyes if cls == EYE else noses).append((cx, cy))
    if len(eyes) != 1:
        return False
    profile_nose = False
    if noses:
        # 最も中央寄りの鼻で判定(複数人の写り込み時に保守側へ倒す)
        noff = min(abs((min(noses, key=lambda t: abs(t[0] - (x1 + x2) / 2))[0])
                       - (x1 + x2) / 2), 10**9) / hbw
        if noff <= nose_lat:
            return False                 # 鼻が中央 = 正面〜3/4 → 対象外
        profile_nose = noff >= 0.20      # 真横顔確定(3/4 の 0.10-0.20 は除く)
    pts = np.asarray(rec["landmarks"]["points"], dtype=np.float64)
    rc, lc = pts[EYE_R].mean(0), pts[EYE_L].mean(0)
    cap = 0.26 if profile_nose else sep_max   # 真横顔時のみ幅広 fold の破れも対象
    if abs(rc[0] - lc[0]) >= cap * hbw:
        return False                     # 折り畳み幾何なし(正面系)→ 対象外
    c = np.asarray(eyes[0])
    d_r, d_l = np.linalg.norm(c - rc), np.linalg.norm(c - lc)
    if abs(d_r - d_l) < side * 0.10:
        return False                     # 曖昧対応
    return d_r < d_l                     # 可視眼が R = 鏡像破れ


def repair(rec: dict, fold_hb: float) -> bool:
    """可視(旧 R)を L へ移し、R を fold で再配置。クランプ時 True。"""
    pts = np.asarray(rec["landmarks"]["points"], dtype=np.float64)
    vis = list(rec["landmarks"]["visibility"])
    backup = {"eye_l": pts[EYE_L].round(2).tolist(),
              "brow_l": pts[BROW_L].round(2).tolist(),
              "vis_eye": [vis[j] for j in EYE_R + EYE_L],
              "vis_brow": [vis[j] for j in BROW_R + BROW_L]}
    pts[EYE_L] = pts[EYE_R].copy()
    pts[BROW_L] = pts[BROW_R].copy()
    for dst, src in ((EYE_L, EYE_R), (BROW_L, BROW_R)):
        for d_j, s_j in zip(dst, src):
            vis[d_j] = rec["landmarks"]["visibility"][s_j]
    eye_vis = pts[EYE_L].copy()
    brow_vis = pts[BROW_L].copy()
    bb = rec["head_bbox"]
    mag = fold_hb * float(bb[2] - bb[0])
    vis_min_x = float(min(eye_vis[:, 0].min(), brow_vis[:, 0].min()))
    allowed = max(vis_min_x - 2.0, 0.0)
    clamped = mag > allowed
    mag = min(mag, allowed)
    delta = np.array([-mag, 0.0])
    pts[EYE_R] = eye_vis + delta
    pts[BROW_R] = brow_vis + delta
    for j in EYE_R + BROW_R:
        vis[j] = 1
    rec["landmarks"]["points"] = [[round(float(x), 2), round(float(y), 2)]
                                  for x, y in pts]
    rec["landmarks"]["visibility"] = vis
    rec["attributes"][MARK] = VALUE
    rec["attributes"][MARK + "_backup"] = backup
    rec["attributes"][MARK + "_fold"] = fold_hb
    return clamped


def revert(rec: dict) -> bool:
    if rec["attributes"].get(MARK) != VALUE:
        return False
    b = rec["attributes"].pop(MARK + "_backup")
    rec["attributes"].pop(MARK)
    rec["attributes"].pop(MARK + "_fold", None)
    pts = np.asarray(rec["landmarks"]["points"], dtype=np.float64)
    vis = list(rec["landmarks"]["visibility"])
    eye_vis = pts[EYE_L].copy()          # 修復後 L = 可視クラスタ(不変)
    brow_vis = pts[BROW_L].copy()
    pts[EYE_R] = eye_vis
    pts[EYE_L] = np.asarray(b["eye_l"])
    pts[BROW_R] = brow_vis
    pts[BROW_L] = np.asarray(b["brow_l"])
    for i, j in enumerate(EYE_R + EYE_L):
        vis[j] = b["vis_eye"][i]
    for i, j in enumerate(BROW_R + BROW_L):
        vis[j] = b["vis_brow"][i]
    rec["landmarks"]["points"] = [[round(float(x), 2), round(float(y), 2)]
                                  for x, y in pts]
    rec["landmarks"]["visibility"] = vis
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="[DEPRECATED - do not use] (history/037 section 9, 036 section 10).")
    ap.add_argument("--unified", type=Path, default=Path("datasets/unified"))
    ap.add_argument("--source", required=True)
    ap.add_argument("--fold-hb", type=float, default=0.15)
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--sep-max", type=float, default=0.17)
    ap.add_argument("--nose-lat", type=float, default=0.10)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    path = args.unified / "annotations" / f"{args.source}.jsonl"
    dets_path = args.unified / "annotations" / f"{args.source}.part_anchor.dets.jsonl"
    dets_map = {r["image_path"]: r["dets"] for r in
                (json.loads(l) for l in open(dets_path, encoding="utf-8"))}
    records = [json.loads(l) for l in open(path, encoding="utf-8")]
    n_fix = n_clamp = 0
    for rec in records:
        if args.revert:
            n_fix += int(revert(rec))
            continue
        if rec["attributes"].get(MARK):
            continue
        d = dets_map.get(rec["image_path"])
        if not d or not is_broken(rec, d, args.score_thr, args.sep_max,
                                  args.nose_lat):
            continue
        if args.dry_run:
            n_fix += 1
        else:
            n_clamp += int(repair(rec, args.fold_hb))
            n_fix += 1
    mode = "revert" if args.revert else "dry-run" if args.dry_run else "repair"
    print(f"{args.source}: {mode} {n_fix:,} records / clamped {n_clamp:,}"
          f" (total {len(records):,}, fold={args.fold_hb} x hb)")
    if not args.dry_run:
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(path)


if __name__ == "__main__":
    main()
