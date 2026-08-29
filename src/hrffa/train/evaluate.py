"""教師モデル評価 CLI: head-NME / 可視性精度 / 姿勢誤差 / Roll360 一貫性。

- head-NME: 決定的 head crop(256px)上の平均 L2 誤差をクロップ辺長で正規化。
  interocular ではなく頭部基準(001 §2.1 の決定)。
- Roll360 一貫性: 入力を θ=0..330° (30° 刻み) で回転し、
  geodesic(R̂(θ), Rz(θ) R̂(0)) の平均(度)と、ランドマークの 2D 回転一貫性(px)。

使い方:
    PYTHONPATH=src python3 -m hrffa.train.evaluate \
        --ckpt ckpts/teacher/smoke_8gb_last.pt --preset smoke_8gb \
        --sources 300w:valid_common wflw:test cofw:test --n 200
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from ..data.dataset import SourceDataset, SourceSpec, collate
from ..dataset.augment.geometric import GeometricPolicy
from ..model.losses import geodesic_loss
from ..model.teacher import TeacherModel
from .config import get_config


def _rz(theta: float) -> torch.Tensor:
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)


@torch.no_grad()
def eval_nme(model, ds, device, n, batch: int = 32) -> dict:
    idxs = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    errs, vis_ok, vis_n, pose_errs = [], 0, 0, []
    for i in range(0, len(idxs), batch):
        items = [ds[j] for j in idxs[i:i + batch]]
        b = collate(items)
        out = model(b["image"].to(device), b["scheme"])
        pred = out["points"].float().cpu()
        errs.append(torch.linalg.norm(pred - b["points"], dim=-1).mean(dim=1))
        pv = out["vis_logits"].argmax(-1).cpu()
        mask = b["vis"] >= 0
        vis_ok += int((pv[mask] == b["vis"][mask]).sum())
        vis_n += int(mask.sum())
        if b["has_rot"].any():
            pose_errs.append(geodesic_loss(
                out["rot"].float().cpu()[b["has_rot"]], b["rot"][b["has_rot"]]))
    res = {"head_nme": float(torch.cat(errs).mean()),
           "vis_acc": vis_ok / max(vis_n, 1)}
    if pose_errs:
        res["pose_err_deg"] = float(torch.rad2deg(torch.cat(pose_errs)).mean())
    return res


@torch.no_grad()
def eval_roll360(model, ds, device, n: int = 32) -> dict:
    from ..dataset.augment.geometric import GeometricParams, apply_geometric
    import cv2
    import json
    rot_errs, pt_errs = [], []
    idxs = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    for j in idxs:
        rec = ds.records[j]
        img = cv2.imread(str(ds.unified / rec["image_path"]))
        pts = np.asarray(rec["landmarks"]["points"])
        base_R = None
        outs = {}
        for deg in range(0, 360, 30):
            prm = GeometricParams(out_size=ds.out_size, roll_deg=float(deg))
            aug = apply_geometric(img, pts, rec["landmarks"]["visibility"],
                                  None, rec["head_bbox"], prm)
            x = cv2.cvtColor(aug["image"], cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            x = (x - ds.norm_mean) / ds.norm_std
            t = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)
            out = model(t, ds.scheme)
            outs[deg] = (out["rot"][0].float().cpu(), out["points"][0].float().cpu())
        R0, p0 = outs[0]
        for deg in range(30, 360, 30):
            th = math.radians(deg)
            Rd, pd = outs[deg]
            rot_errs.append(math.degrees(float(
                geodesic_loss((_rz(th) @ R0)[None], Rd[None])[0])))
            c, s = math.cos(th), math.sin(th)
            Rot = torch.tensor([[c, -s], [s, c]])
            center = torch.tensor([0.5, 0.5])
            mapped = (p0 - center) @ Rot.T + center
            pt_errs.append(float(torch.linalg.norm(
                mapped - pd, dim=-1).mean()) * ds.out_size)
    return {"roll360_rot_err_deg": float(np.mean(rot_errs)),
            "roll360_pt_err_px": float(np.mean(pt_errs))}


@torch.no_grad()
def render_val_samples(model, val_sets, device, out_dir: Path, n: int = 20,
                       draw_axes: bool = True) -> None:
    """検証サンプルの予測を重畳レンダリングして out_dir に保存する。

    out_dir は毎回丸ごと作り直す(最新 best のみ保持)。
    描画: 予測のみ・単色(可視性の色分けはしない)。GT は描画しない。
    per-sample NME はヘッダに表示。
    draw_axes=False で姿勢軸を省略(姿勢監督なしの学習ではヘッドが未学習のため)。
    """
    import shutil

    import cv2

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_set = max(1, n // len(val_sets))
    color = (0, 200, 0)  # 可視性の色分けは行わない(2026-08-25 ユーザー方針)
    for name, ds in val_sets:
        idxs = np.linspace(0, len(ds) - 1, per_set).astype(int)
        items = [ds[int(j)] for j in idxs]
        b = collate(items)
        out = model(b["image"].to(device), b["scheme"])
        pred = out["points"].float().cpu().numpy()
        R = out["rot"].float().cpu().numpy()
        s = ds.out_size
        mean, std = ds.norm_mean, ds.norm_std
        for k, item in enumerate(items):
            img = item["image"].numpy().transpose(1, 2, 0) * std + mean
            img = cv2.cvtColor((img.clip(0, 1) * 255).astype(np.uint8),
                               cv2.COLOR_RGB2BGR).copy()
            for (x, y) in pred[k] * s:
                cv2.circle(img, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)
            if draw_axes:
                c = pred[k].mean(axis=0) * s
                for a, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                    tip = (int(c[0] + 0.15 * s * R[k][0, a]),
                           int(c[1] + 0.15 * s * R[k][1, a]))
                    cv2.line(img, (int(c[0]), int(c[1])), tip, color, 2, cv2.LINE_AA)
            nme = float(np.linalg.norm(pred[k] - item["points"].numpy(), axis=-1).mean())
            cv2.putText(img, f"{name} nme={nme:.4f}", (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(out_dir / f"{name}_{int(idxs[k]):05d}.jpg"), img)


# 公式プロトコルの inter-ocular 正規化点(外眼角)
_INTEROCULAR = {"ibug68": (36, 45), "wflw98": (60, 72), "cofw29": (8, 9)}
_WFLW_ATTRS = ["pose", "expression", "illumination", "makeup", "occlusion", "blur"]


@torch.no_grad()
def _collect_io_nme(model, ds, device, batch: int = 32) -> np.ndarray:
    """全件の inter-ocular 正規化 NME を返す(相似変換不変なので crop 座標系で計算)。"""
    a, b = _INTEROCULAR[ds.scheme]
    out_all = []
    for i in range(0, len(ds), batch):
        items = [ds[j] for j in range(i, min(i + batch, len(ds)))]
        bt = collate(items)
        pred = model(bt["image"].to(device), bt["scheme"])["points"].float().cpu()
        gt = bt["points"]
        iod = torch.linalg.norm(gt[:, a] - gt[:, b], dim=-1).clamp_min(1e-6)
        nme = torch.linalg.norm(pred - gt, dim=-1).mean(dim=1) / iod
        out_all.append(nme)
    return torch.cat(out_all).numpy()


def _summary(nme: np.ndarray, thr: float = 0.10) -> dict:
    ced_x = np.linspace(0, thr, 201)
    ced = [(nme <= x).mean() for x in ced_x]
    return {"n": len(nme), "nme_pct": round(float(nme.mean()) * 100, 3),
            "fr@0.1": round(float((nme > thr).mean()), 4),
            "auc@0.1": round(float(np.trapezoid(ced, ced_x) / thr), 4)}


def eval_official(model, cfg, device) -> None:
    """公式データセット基準(inter-ocular NME / FR@0.1 / AUC@0.1)での全件評価。

    注意: 入力クロップは本プロジェクトの head bbox 規約(文献の face bbox とは異なる)
    のため、文献値との比較は参考値として扱うこと。
    """
    policy = GeometricPolicy(out_size=cfg.out_size, pad=cfg.crop_pad)

    # WFLW: full + 6 サブセット
    ds = SourceDataset(cfg.unified, SourceSpec("wflw", 1.0, ("test",)),
                       cfg.out_size, train=False, policy=policy,
                       input_norm=cfg.input_norm)
    nme = _collect_io_nme(model, ds, device)
    print(f"[WFLW test full]      " + json.dumps(_summary(nme)))
    for attr in _WFLW_ATTRS:
        idx = [i for i, r in enumerate(ds.records) if r["attributes"].get(attr)]
        if idx:
            print(f"[WFLW {attr:12s}] " + json.dumps(_summary(nme[idx])))

    # 300W: common / challenge / full
    parts = {}
    for split in ("valid_common", "valid_challenge"):
        ds = SourceDataset(cfg.unified, SourceSpec("300w", 1.0, (split,)),
                           cfg.out_size, train=False, policy=policy,
                           input_norm=cfg.input_norm)
        parts[split] = _collect_io_nme(model, ds, device)
        name = "common" if split == "valid_common" else "challenge"
        print(f"[300W {name:9s}]      " + json.dumps(_summary(parts[split])))
    print(f"[300W full]           " + json.dumps(_summary(np.concatenate(list(parts.values())))))

    # COFW test
    ds = SourceDataset(cfg.unified, SourceSpec("cofw", 1.0, ("test",)),
                       cfg.out_size, train=False, policy=policy,
                       input_norm=cfg.input_norm)
    print(f"[COFW test]           " + json.dumps(_summary(_collect_io_nme(model, ds, device))))


@torch.no_grad()
def eval_pose_stress(model, cfg, device, batch: int = 16,
                     n_per_set: int = 300) -> None:
    """大角度ストレス評価(history/024)。

    文献ベンチ(顔クロップ・正面偏重)では測れない大 Yaw/Pitch/Roll 域の「精度」を、
    実写公式セットに厳密 GT 幾何変換(Roll 360°・カメラ回転ホモグラフィ)を適用して測る。
    GT は同一の 3x3 変換で厳密に追従するため、Roll360 一貫性(自己整合のみ)と違い
    真の inter-ocular NME を報告できる。実効姿勢ビンは 6DRepNet 基準姿勢(キャッシュ)に
    適用回転を合成した回転行列から算出する。
    制約: カメラ回転は同一画素の再投影であり新しい自己遮蔽は生まない
    (検証済み範囲 pitch ±25° / yaw ±15°、history/005)。
    """
    import cv2

    from ..dataset.augment.geometric import GeometricParams, apply_geometric
    from ..dataset.geometry import euler300wlp_to_rotmat, rotmat_to_euler300wlp
    from ..dataset.qa.sixdrepnet import pose_meta_for_source

    s = cfg.out_size

    # (ラベル, roll, cam_pitch, cam_yaw)。base 以外が PS 集計対象
    configs = ([("base", 0, 0, 0)]
               + [(f"roll{r:+04d}", r, 0, 0) for r in (45, 90, 135, 180, 225, 270, 315)]
               + [(f"cam_p{p:+03d}", 0, p, 0) for p in (-25, -15, 15, 25)]
               + [(f"cam_y{y:+03d}", 0, 0, y) for y in (-15, 15)]
               + [("cam_p+25_y+15", 0, 25, 15), ("cam_p-25_y-15", 0, -25, -15)])
    defs = [("wflw", ("test",)), ("300w", ("valid_common", "valid_challenge")),
            ("cofw", ("test",))]
    policy = GeometricPolicy(out_size=s)

    all_rows = []          # (set, config, nme配列)
    eff_bins = []          # cam 系のみ: (eff_yaw_deg, eff_pitch_deg, nme)
    for source, splits in defs:
        ds = SourceDataset(cfg.unified, SourceSpec(source, 1.0, splits),
                           s, train=False, policy=policy,
                           input_norm=cfg.input_norm)
        meta = pose_meta_for_source(cfg.unified, source, splits, None)
        mean, std = ds.norm_mean, ds.norm_std
        a, b = _INTEROCULAR[ds.scheme]
        idxs = np.linspace(0, len(ds) - 1, min(n_per_set, len(ds))).astype(int)
        recs = [ds.records[int(j)] for j in idxs]
        base_R = []
        for rec in recs:
            ypr = meta.get(rec["record_id"])
            base_R.append(None if ypr is None else euler300wlp_to_rotmat(
                math.radians(ypr[1]), math.radians(ypr[0]), math.radians(ypr[2])))
        for label, roll, cp, cy in configs:
            prm = GeometricParams(out_size=s, pad=policy.pad, roll_deg=float(roll),
                                  cam_pitch_deg=float(cp), cam_yaw_deg=float(cy))
            nmes = []
            for i0 in range(0, len(recs), batch):
                chunk = recs[i0:i0 + batch]
                imgs, gts, Rs = [], [], []
                for k, rec in enumerate(chunk):
                    img = cv2.imread(str(ds.unified / rec["image_path"]))
                    pts = np.asarray(rec["landmarks"]["points"], dtype=np.float64)
                    out = apply_geometric(img, pts, rec["landmarks"]["visibility"],
                                          base_R[i0 + k], rec["head_bbox"], prm,
                                          flip_mapping=ds.flip_mapping)
                    x = cv2.cvtColor(out["image"], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    imgs.append(torch.from_numpy(((x - mean) / std).transpose(2, 0, 1).copy()))
                    gts.append(torch.from_numpy(out["points"].astype(np.float32)))
                    Rs.append(out["rotation"])
                pred = model(torch.stack(imgs).to(device), ds.scheme)["points"].float().cpu() * s
                gt = torch.stack(gts)
                iod = torch.linalg.norm(gt[:, a] - gt[:, b], dim=-1).clamp_min(1e-6)
                nme = (torch.linalg.norm(pred - gt, dim=-1).mean(dim=1) / iod).numpy()
                nmes.append(nme)
                if label.startswith("cam"):
                    for R_eff, v in zip(Rs, nme):
                        if R_eff is not None:
                            p_e, y_e, _ = rotmat_to_euler300wlp(R_eff)
                            eff_bins.append((abs(math.degrees(y_e)),
                                             math.degrees(p_e), float(v)))
            all_rows.append((source, label, np.concatenate(nmes)))

    # ---- 集計 ----
    print(f"[pose-stress] n/set={n_per_set} configs={len(configs)}")
    for source, _ in defs:
        rows = [(l, v) for src, l, v in all_rows if src == source]
        base = dict(rows)["base"].mean()
        worst_roll = max((v.mean() for l, v in rows if l.startswith("roll")))
        worst_cam = max((v.mean() for l, v in rows if l.startswith("cam")))
        ps = np.concatenate([v for l, v in rows if l != "base"])
        print(f"  {source:5s} base={base*100:.3f} worst_roll={worst_roll*100:.3f} "
              f"worst_cam={worst_cam*100:.3f} PS-NME%={ps.mean()*100:.3f} "
              f"PS-FR@0.1={(ps>0.1).mean():.4f} degradation={(ps.mean()/base-1)*100:+.1f}%")
        for l, v in rows:
            print(f"    {source:5s} {l:14s} nme%={v.mean()*100:.3f} fr={(v>0.1).mean():.4f}")

    if eff_bins:
        ya = np.array([e[0] for e in eff_bins])
        pa = np.array([e[1] for e in eff_bins])
        va = np.array([e[2] for e in eff_bins])
        print("  effective pose bins (cam configs, 6DRepNet estimate + applied rotation):")
        for lo, hi in [(0, 30), (30, 60), (60, 95), (95, 130)]:
            m = (ya >= lo) & (ya < hi)
            if m.any():
                print(f"    |yaw|  {lo:>3}..{hi:<3}: n={int(m.sum()):>5} nme%={va[m].mean()*100:.3f}")
        for lo, hi in [(-95, -45), (-45, -15), (-15, 15), (15, 45), (45, 95)]:
            m = (pa >= lo) & (pa < hi)
            if m.any():
                print(f"    pitch {lo:>4}..{hi:<4}: n={int(m.sum()):>5} nme%={va[m].mean()*100:.3f}")


# スタイルシフト摂動スイート(決定的)。warm/cool のチャンネルゲインは学習
# photometric 拡張(明度・コントラスト・ガンマ・グレー・ノイズ・JPEG)に無い
# 色温度軸で、InstanceNorm のスタイル頑健化(017 §5.4 → 021)の主検証対象。
def _style_shifts():
    import cv2

    def gains(b, g, r):
        def f(img):
            x = img.astype(np.float32) * np.array([b, g, r], np.float32)
            return x.clip(0, 255).astype(np.uint8)
        return f

    def gamma(g):
        def f(img):
            return (255.0 * (img.astype(np.float32) / 255.0) ** g).astype(np.uint8)
        return f

    def gray(img):
        return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    def jpeg30(img):
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 30])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    def mblur(length, angle_deg=30.0):
        k = np.zeros((length, length), np.float32)
        c = length // 2
        th = np.deg2rad(angle_deg)
        for t in np.linspace(-c, c, length * 4):
            x, y = int(round(c + t * np.cos(th))), int(round(c + t * np.sin(th)))
            if 0 <= x < length and 0 <= y < length:
                k[y, x] = 1.0
        k /= max(k.sum(), 1.0)
        def f(img):
            return cv2.filter2D(img, -1, k)
        return f

    return [("clean", None),
            ("mblur9", mblur(9)), ("mblur21", mblur(21)),
            ("warm", gains(0.80, 1.00, 1.20)),
            ("cool", gains(1.20, 1.00, 0.80)),
            ("gamma0.6", gamma(0.6)),
            ("gamma1.6", gamma(1.6)),
            ("gray", gray),
            ("jpeg30", jpeg30)]


def _perturb_item_image(img_t: torch.Tensor, fn, mean: np.ndarray,
                        std: np.ndarray) -> torch.Tensor:
    """正規化済み RGB テンソル (C,H,W) に BGR uint8 摂動 fn を適用して再正規化する。"""
    import cv2
    x = img_t.numpy().transpose(1, 2, 0) * std + mean
    bgr = cv2.cvtColor((x.clip(0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    bgr = fn(bgr)
    x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((x - mean) / std).transpose(2, 0, 1).copy())


@torch.no_grad()
def eval_style_shift(model, cfg, device, batch: int = 32,
                     max_per_set: int | None = None) -> None:
    """スタイルシフト耐性評価: 実写公式セットのクロップに決定的な色調摂動を加え、
    inter-ocular NME の劣化率(vs clean)を測る。innorm 再検証(history/021)用。"""
    policy = GeometricPolicy(out_size=cfg.out_size, pad=cfg.crop_pad)
    defs = [("wflw_test", "wflw", ("test",)),
            ("300w_valid", "300w", ("valid_common", "valid_challenge")),
            ("cofw_test", "cofw", ("test",))]
    shifts = _style_shifts()
    print(f"[style-shift] shifts={[n for n, _ in shifts]}")
    worst = {}
    for label, source, splits in defs:
        ds = SourceDataset(cfg.unified, SourceSpec(source, 1.0, splits),
                           cfg.out_size, train=False, policy=policy,
                           input_norm=cfg.input_norm)
        n = len(ds) if max_per_set is None else min(max_per_set, len(ds))
        idxs = np.linspace(0, len(ds) - 1, n).astype(int)
        a, b = _INTEROCULAR[ds.scheme]
        base = None
        for sname, fn in shifts:
            vals = []
            for i in range(0, len(idxs), batch):
                items = [ds[int(j)] for j in idxs[i:i + batch]]
                if fn is not None:
                    for it in items:
                        it["image"] = _perturb_item_image(it["image"], fn,
                                                          ds.norm_mean, ds.norm_std)
                bt = collate(items)
                pred = model(bt["image"].to(device), bt["scheme"])["points"].float().cpu()
                gt = bt["points"]
                iod = torch.linalg.norm(gt[:, a] - gt[:, b], dim=-1).clamp_min(1e-6)
                vals.append(torch.linalg.norm(pred - gt, dim=-1).mean(dim=1) / iod)
            m = float(torch.cat(vals).mean()) * 100
            if sname == "clean":
                base = m
                print(f"  {label:11s} clean    nme%={m:.3f} (n={n})")
            else:
                rel = (m - base) / base * 100
                worst[label] = max(worst.get(label, 0.0), rel)
                print(f"  {label:11s} {sname:8s} nme%={m:.3f} ({rel:+.1f}%)")
    for label, rel in worst.items():
        print(f"[style-shift] {label} worst degradation: +{rel:.1f}%")


@torch.no_grad()
def eval_stratified(model, cfg, device, batch: int = 32) -> None:
    """姿勢角層別のランドマーク NME(head crop 正規化)。

    姿勢 GT を持つ 300wlp_val + synth_dwarp val(ホールドアウト)を対象に、
    GT の |pitch| / |yaw| ビンごとの NME を報告する。姿勢マルチタスクが
    「極端姿勢のランドマーク」に効くかの検証(history/012)に用いる。
    """
    policy = GeometricPolicy(out_size=cfg.out_size, pad=cfg.crop_pad)
    dss = [SourceDataset(cfg.unified, SourceSpec(n, 1.0, holdout="val"),
                         cfg.out_size, train=False, policy=policy,
                         input_norm=cfg.input_norm)
           for n in ("300wlp", "synth_dwarp")]
    nmes, pitches, yaws = [], [], []
    for ds in dss:
        for i in range(0, len(ds), batch):
            items = [ds[j] for j in range(i, min(i + batch, len(ds)))]
            b = collate(items)
            pred = model(b["image"].to(device), b["scheme"])["points"].float().cpu()
            nmes.append(torch.linalg.norm(pred - b["points"], dim=-1).mean(dim=1))
            for it, rec in zip(items, ds.records[i:i + batch]):
                e = rec["pose"]["euler_deg"]
                pitches.append(abs(e["pitch"]))
                yaws.append(abs(e["yaw"]))
    nme = torch.cat(nmes).numpy()
    pitches, yaws = np.array(pitches), np.array(yaws)
    print(f"[stratified] total n={len(nme)} mean_nme={nme.mean():.5f}")
    for name, arr, bins in [("|pitch|", pitches, [0, 15, 30, 45, 180]),
                            ("|yaw|", yaws, [0, 30, 60, 95])]:
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (arr >= lo) & (arr < hi)
            if m.any():
                print(f"  {name} {lo:>3}-{hi:<3}: n={int(m.sum()):>5} "
                      f"nme={float(nme[m].mean()):.5f}")


@torch.no_grad()
def eval_stratified_real(model, cfg, device, batch: int = 32) -> None:
    """実写ベンチ(wflw test / 300w valid / cofw test)を 6DRepNet 推定姿勢で
    層別した head-NME。姿勢 GT のない実写での姿勢別性能分析(history/015 §6.7)。
    姿勢メタデータはサイドカーにキャッシュされ再計算不要。"""
    from ..dataset.qa.sixdrepnet import SixDRepNet360, pose_meta_for_source
    from collections import defaultdict

    sixd = SixDRepNet360()
    policy = GeometricPolicy(out_size=cfg.out_size, pad=cfg.crop_pad)
    defs = [("wflw", ("test",)), ("300w", ("valid_common", "valid_challenge")),
            ("cofw", ("test",))]
    nmes, yaws, pitches = [], [], []
    for source, splits in defs:
        meta = pose_meta_for_source(cfg.unified, source, splits, sixd)
        ds = SourceDataset(cfg.unified, SourceSpec(source, 1.0, splits),
                           cfg.out_size, train=False, policy=policy,
                           input_norm=cfg.input_norm)
        for i in range(0, len(ds), batch):
            items = [ds[j] for j in range(i, min(i + batch, len(ds)))]
            bt = collate(items)
            pred = model(bt["image"].to(device), bt["scheme"])["points"].float().cpu()
            per = torch.linalg.norm(pred - bt["points"], dim=-1).mean(dim=1)
            for k, j in enumerate(range(i, min(i + batch, len(ds)))):
                rid = ds.records[j]["record_id"]
                if rid not in meta:
                    continue
                yaw, pitch, _ = meta[rid]
                nmes.append(float(per[k]))
                yaws.append(abs(yaw))
                pitches.append(pitch)
    nmes = np.array(nmes)
    yaws = np.array(yaws)
    pitches = np.array(pitches)
    print(f"[stratify-real] n={len(nmes)} mean_nme={nmes.mean():.5f} (stratified by 6DRepNet-estimated pose)")
    for name, arr, bins in [("|yaw|", yaws, [0, 30, 60, 95, 180]),
                            ("pitch", pitches, [-90, -30, -10, 10, 30, 90])]:
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (arr >= lo) & (arr < hi)
            if m.any():
                print(f"  {name} {lo:>4}..{hi:<4}: n={int(m.sum()):>5} "
                      f"nme={float(nmes[m].mean()):.5f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint: head-NME / visibility / official protocol / stratified / pose-stress / style-shift.")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--preset", default="smoke_8gb")
    ap.add_argument("--use-ema", action="store_true")
    ap.add_argument("--official", action="store_true",
                    help="full evaluation with the official dataset protocol (inter-ocular NME/FR/AUC)")
    ap.add_argument("--stratify", action="store_true",
                    help="landmark NME stratified by pose angle (|pitch|/|yaw|)")
    ap.add_argument("--stratify-real", action="store_true",
                    help="NME on the real-image benchmarks stratified by 6DRepNet-estimated pose")
    ap.add_argument("--style-shift", action="store_true",
                    help="NME degradation under photometric shifts (color temperature / gamma / gray / JPEG)")
    ap.add_argument("--style-n", type=int, default=None,
                    help="max samples per set for --style-shift (default: all)")
    ap.add_argument("--pose-stress", action="store_true",
                    help="accuracy at large yaw/pitch/roll via exact-GT geometric transforms of real images")
    ap.add_argument("--stress-n", type=int, default=300,
                    help="samples per set for --pose-stress")
    ap.add_argument("--sources", nargs="+",
                    default=["300w:valid_common", "wflw:test", "cofw:test"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--roll360-n", type=int, default=16)
    args = ap.parse_args()

    cfg = get_config(args.preset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = TeacherModel.from_config(cfg).to(device).eval()
    # best ckpt は ema のみ保持するため自動フォールバック
    key = "ema" if (args.use_ema or "model" not in ck) else "model"
    model.load_state_dict(ck[key])

    if args.official:
        eval_official(model, cfg, device)
        return
    if args.stratify:
        eval_stratified(model, cfg, device)
        return
    if args.stratify_real:
        eval_stratified_real(model, cfg, device)
        return
    if args.style_shift:
        eval_style_shift(model, cfg, device, max_per_set=args.style_n)
        return
    if args.pose_stress:
        eval_pose_stress(model, cfg, device, n_per_set=args.stress_n)
        return

    policy = GeometricPolicy(out_size=cfg.out_size, pad=cfg.crop_pad)
    for spec in args.sources:
        name, split = spec.split(":")
        ds = SourceDataset(cfg.unified, SourceSpec(name, 1.0, (split,)),
                           cfg.out_size, train=False, policy=policy,
                           input_norm=cfg.input_norm)
        res = eval_nme(model, ds, device, args.n)
        if name == "wflw":
            res |= eval_roll360(model, ds, device, args.roll360_n)
        print(f"[{name}:{split}] " + " ".join(f"{k}={v:.4f}" for k, v in res.items()))


if __name__ == "__main__":
    main()
