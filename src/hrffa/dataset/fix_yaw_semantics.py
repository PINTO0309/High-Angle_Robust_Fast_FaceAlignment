"""300wlp の |yaw|>60° レコードの目・眉を空間順序規約へ変換(history/034 v1 + 036 対称化)。

背景: 横顔で 300wlp は解剖学規約(index は常に解剖学的左右、遮蔽目 = 頭部内 3D 投影)、
実写 GT(wflw)は空間順序規約(可視座標は L index、R ブロックは可視クラスタの
画像左脇に折り畳み)を使い、教師信号が矛盾する。

v1(034)は yaw>+thr のみ変換した(可視クラスタの index 検証では yaw− は両規約一致)。
しかし 036 の帯別診断で「遮蔽側ブロックの配置様式(3D 投影 vs 可視側折り畳み)」は
yaw− でも相違することが実証された(片側変換が左右非対称ラベルを生み勾配競合)。
本版はこれを対称化し、折り畳み量も実写 GT に合わせる:

  yaw > +thr(v1 由来):
    - 可視ブロック(旧 R 座標)を L インデックス(42-47 / 22-26)へ移す
    - R インデックス(36-41 / 17-21)は「可視ブロックを −x へ折り畳み量だけ平行移動」
      に再配置し、可視性を 1(遮蔽)にする(実写 GT の様式を模倣)
    - 旧 L 座標(耳側投影)は破棄。ただし attributes にバックアップを保存(可逆)
  yaw < −thr(036 v2 — 2026-08-26 に方向を訂正):
    - 実写 GT の実測(wflw det 検証: 右向き 可視=L 92%/左向き 可視=R 94%、
      300w 独立確認: 100%/100%)により、**幻ブロックは常に「鼻側」**で
      index は空間順(可視眼は yaw+ で L、yaw− で R)= 鏡映等変が真の規約。
      旧 v1_neg(可視=L のまま R を −x 側へ捏造)は逆向きだったため廃止・自動移行
    - 可視座標(旧 L)と vis を R インデックス(36-41 / 17-21)へ移す
    - L インデックスは「可視ブロックを +x(鼻側)へ折り畳み量平行移動」で再配置し
      可視性 1(画像右端 2px クランプ)
    - 旧 R 座標(解剖学投影)は attributes にバックアップ(可逆)

折り畳み量(036 実測・head_bbox 幅基準): WFLW 実写横顔の R-L 分離は
head_bbox 幅比 0.134〜0.170(姿勢極端化で漸減)。眼幅基準の較正は
データセット間の眼幅定義差(投影点列 vs 手付け)で 2 倍過大になったため撤回し、
検出器由来で全ソース同一定義の head_bbox 幅 × --fold-hb(既定 0.15 = 実写帯の中庸)
を採用。変換済みレコードも目標値へ再折り畳み(refold)する。捏造点が画像外へ
出ないよう画像左端 2px でクランプ(発生率は実行時に報告)。

閾値 60° の妥当性(036): 300W-LP のワープ描画では |yaw|≈62 で遠目が画素上ほぼ消失
することを目視確認済み(遠目が見えている間は元ラベル = 描画に一致する真の投影位置
なので触らない。規約の自由度は不可視の幻ブロックのみ)。

対象ソース: --source(既定 300wlp)。synth_dwarp(ibug68 + pose あり)にも適用可。

冪等: attributes.yaw_semantics(spatial_v1 / spatial_v1_neg)+ yaw_semantics_fold が
目標値と一致すればスキップ。--revert で両方向ともバックアップから元規約へ完全復元。

usage:
  uv run python -m hrffa.dataset.fix_yaw_semantics [--source 300wlp] [--yaw-thr 60]
                                                   [--fold-hb 0.15] [--dry-run | --revert]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EYE_R = list(range(36, 42))
EYE_L = list(range(42, 48))
BROW_R = list(range(17, 22))
BROW_L = list(range(22, 27))
MARK = "yaw_semantics"
VALUE = "spatial_v1"          # yaw > +thr(034 v1)
VALUE_NEG = "spatial_v1_neg"   # 廃止(逆向きだった旧変換。検出したら自動移行)
VALUE_NEG2 = "spatial_v2_neg"  # yaw < −thr(036 v2: 可視→R + 幻 L を鼻側 +x)
FOLD_HB = 0.15                # 折り畳み量(head_bbox 幅比、036 実測 0.134〜0.170 の中庸)


def _fold(pts: np.ndarray, vis: list, rec: dict, fold_hb: float) -> bool:
    """L(可視)クラスタから R ブロックを −x 折り畳みで再配置。クランプ時 True。"""
    eye_vis = pts[EYE_L].copy()
    brow_vis = pts[BROW_L].copy()
    bb = rec["head_bbox"]
    mag = fold_hb * float(bb[2] - bb[0])
    # 画像内クランプ: 捏造点の最小 x が 2px を割らない範囲まで
    vis_min_x = float(min(eye_vis[:, 0].min(), brow_vis[:, 0].min()))
    allowed = max(vis_min_x - 2.0, 0.0)
    clamped = mag > allowed
    mag = min(mag, allowed)
    delta = np.array([-mag, 0.0])
    pts[EYE_R] = eye_vis + delta
    pts[BROW_R] = brow_vis + delta
    for j in EYE_R + BROW_R:
        vis[j] = 1                                     # 折り畳み側は遮蔽
    return clamped


def _writeback(rec: dict, pts: np.ndarray, vis: list) -> None:
    rec["landmarks"]["points"] = [[round(float(x), 2), round(float(y), 2)]
                                  for x, y in pts]
    rec["landmarks"]["visibility"] = vis


def convert(rec: dict, fold_hb: float) -> bool:
    """yaw > +thr: 可視(旧 R)を L へ移してから折り畳み。"""
    pts = np.asarray(rec["landmarks"]["points"], dtype=np.float64)
    vis = list(rec["landmarks"]["visibility"])
    backup = {"eye_l": pts[EYE_L].round(2).tolist(),
              "brow_l": pts[BROW_L].round(2).tolist(),
              "vis_eye": [vis[j] for j in EYE_R + EYE_L],
              "vis_brow": [vis[j] for j in BROW_R + BROW_L]}
    # 可視(旧 R)座標と可視性を L index へ
    pts[EYE_L] = pts[EYE_R].copy()
    pts[BROW_L] = pts[BROW_R].copy()
    for dst, src in ((EYE_L, EYE_R), (BROW_L, BROW_R)):
        for d_j, s_j in zip(dst, src):
            vis[d_j] = rec["landmarks"]["visibility"][s_j]
    clamped = _fold(pts, vis, rec, fold_hb)
    _writeback(rec, pts, vis)
    rec["attributes"][MARK] = VALUE
    rec["attributes"][MARK + "_backup"] = backup
    rec["attributes"][MARK + "_fold"] = fold_hb
    return clamped


def _fold_neg(pts: np.ndarray, vis: list, rec: dict, fold_hb: float) -> bool:
    """R(可視)クラスタから L ブロックを +x(鼻側)折り畳みで再配置。"""
    eye_vis = pts[EYE_R].copy()
    brow_vis = pts[BROW_R].copy()
    bb = rec["head_bbox"]
    mag = fold_hb * float(bb[2] - bb[0])
    W = float(rec["image_size"][0])
    vis_max_x = float(max(eye_vis[:, 0].max(), brow_vis[:, 0].max()))
    allowed = max(W - 2.0 - vis_max_x, 0.0)
    clamped = mag > allowed
    mag = min(mag, allowed)
    delta = np.array([mag, 0.0])
    pts[EYE_L] = eye_vis + delta
    pts[BROW_L] = brow_vis + delta
    for j in EYE_L + BROW_L:
        vis[j] = 1                                     # 折り畳み側は遮蔽
    return clamped


def convert_neg(rec: dict, fold_hb: float) -> bool:
    """yaw < −thr(v2): 可視(旧 L)を R へ移し、幻 L を鼻側 +x へ折り畳み。"""
    pts = np.asarray(rec["landmarks"]["points"], dtype=np.float64)
    vis = list(rec["landmarks"]["visibility"])
    backup = {"eye_r": pts[EYE_R].round(2).tolist(),
              "brow_r": pts[BROW_R].round(2).tolist(),
              "vis_eye": [vis[j] for j in EYE_R + EYE_L],
              "vis_brow": [vis[j] for j in BROW_R + BROW_L]}
    # 可視(旧 L)座標と可視性を R index へ
    pts[EYE_R] = pts[EYE_L].copy()
    pts[BROW_R] = pts[BROW_L].copy()
    for dst, src in ((EYE_R, EYE_L), (BROW_R, BROW_L)):
        for d_j, s_j in zip(dst, src):
            vis[d_j] = rec["landmarks"]["visibility"][s_j]
    clamped = _fold_neg(pts, vis, rec, fold_hb)
    _writeback(rec, pts, vis)
    rec["attributes"][MARK] = VALUE_NEG2
    rec["attributes"][MARK + "_backup"] = backup
    rec["attributes"][MARK + "_fold"] = fold_hb
    return clamped


def refold(rec: dict, fold_hb: float) -> bool:
    """変換済みレコードの幻ブロックを新しい折り畳み量で再配置(可視側は不変)。"""
    pts = np.asarray(rec["landmarks"]["points"], dtype=np.float64)
    vis = list(rec["landmarks"]["visibility"])
    if rec["attributes"].get(MARK) == VALUE_NEG2:
        clamped = _fold_neg(pts, vis, rec, fold_hb)
    else:
        clamped = _fold(pts, vis, rec, fold_hb)
    _writeback(rec, pts, vis)
    rec["attributes"][MARK + "_fold"] = fold_hb
    return clamped


def revert(rec: dict) -> bool:
    mark = rec["attributes"].get(MARK)
    if mark not in (VALUE, VALUE_NEG, VALUE_NEG2):
        return False
    b = rec["attributes"].pop(MARK + "_backup")
    rec["attributes"].pop(MARK)
    rec["attributes"].pop(MARK + "_fold", None)
    pts = np.asarray(rec["landmarks"]["points"], dtype=np.float64)
    vis = list(rec["landmarks"]["visibility"])
    if mark == VALUE_NEG:
        pts[EYE_R] = np.asarray(b["eye_r"])
        pts[BROW_R] = np.asarray(b["brow_r"])
        for i, j in enumerate(EYE_R):
            vis[j] = b["vis_eye_r"][i]
        for i, j in enumerate(BROW_R):
            vis[j] = b["vis_brow_r"][i]
    elif mark == VALUE_NEG2:
        eye_vis = pts[EYE_R].copy()           # v2 変換後 R = 可視クラスタ
        brow_vis = pts[BROW_R].copy()
        pts[EYE_L] = eye_vis
        pts[EYE_R] = np.asarray(b["eye_r"])
        pts[BROW_L] = brow_vis
        pts[BROW_R] = np.asarray(b["brow_r"])
        for i, j in enumerate(EYE_R + EYE_L):
            vis[j] = b["vis_eye"][i]
        for i, j in enumerate(BROW_R + BROW_L):
            vis[j] = b["vis_brow"][i]
    else:
        eye_vis = pts[EYE_L].copy()           # 変換後 L = 可視クラスタ
        brow_vis = pts[BROW_L].copy()
        pts[EYE_R] = eye_vis
        pts[EYE_L] = np.asarray(b["eye_l"])
        pts[BROW_R] = brow_vis
        pts[BROW_L] = np.asarray(b["brow_l"])
        for i, j in enumerate(EYE_R + EYE_L):
            vis[j] = b["vis_eye"][i]
        for i, j in enumerate(BROW_R + BROW_L):
            vis[j] = b["vis_brow"][i]
    _writeback(rec, pts, vis)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert eye/brow labels of 300wlp |yaw|>60 deg records to the spatial-order convention (history/034 + 036).")
    ap.add_argument("--unified", type=Path, default=Path("datasets/unified"))
    ap.add_argument("--source", default="300wlp")
    ap.add_argument("--yaw-thr", type=float, default=60.0)
    ap.add_argument("--fold-hb", type=float, default=FOLD_HB)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    path = args.unified / "annotations" / f"{args.source}.jsonl"
    records = [json.loads(l) for l in open(path, encoding="utf-8")]
    n_pos = n_neg = n_refold = n_clamp = 0
    for rec in records:
        if args.revert:
            mark = rec["attributes"].get(MARK)
            if revert(rec):
                n_pos += int(mark == VALUE)
                n_neg += int(mark == VALUE_NEG)
            continue
        if rec["attributes"].get(MARK) == VALUE_NEG:
            # 旧(逆向き)neg 変換 → 自動移行: 復元してから v2 で再変換
            if args.dry_run:
                n_neg += 1
                continue
            revert(rec)
            n_clamp += int(convert_neg(rec, args.fold_hb))
            n_neg += 1
            continue
        if rec["attributes"].get(MARK):
            if rec["attributes"].get(MARK + "_fold") != args.fold_hb:
                if not args.dry_run:
                    n_clamp += int(refold(rec, args.fold_hb))
                n_refold += 1
            continue
        yaw = (rec.get("pose") or {}).get("euler_deg", {}).get("yaw")
        if yaw is None or abs(yaw) <= args.yaw_thr:
            continue
        if args.dry_run:
            n_pos += int(yaw > 0)
            n_neg += int(yaw < 0)
        elif yaw > 0:
            n_clamp += int(convert(rec, args.fold_hb))
            n_pos += 1
        else:
            n_clamp += int(convert_neg(rec, args.fold_hb))
            n_neg += 1
    mode = "revert" if args.revert else "dry-run" if args.dry_run else "convert"
    print(f"{args.source}: {mode} yaw>+{args.yaw_thr:.0f} deg {n_pos:,} / "
          f"yaw<-{args.yaw_thr:.0f} deg {n_neg:,} / refold {n_refold:,} / "
          f"clamped {n_clamp:,} (total {len(records):,}, fold={args.fold_hb} x hb width)")
    if not args.dry_run:
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(path)


if __name__ == "__main__":
    main()
