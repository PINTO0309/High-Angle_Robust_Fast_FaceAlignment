"""300wlp: yaw 帯別のモデル-ラベル不一致(目眉ブロック vs その他)。

036 の規約統一の受入計測器。左右対称な規約なら学習が進むにつれ全帯の比が
~1.0 に収束するはず。左右非対称の残存(yaw− ≫ yaw+)や高止まりは競合の残存を示す。
基準値: runs/abl_v8_datafix/qa_yaw_band_r2init_baseline.json(r2 init 時点 = v8 best)

usage:
  uv run python -m hrffa.dataset.qa.yaw_band_conflict \
      [--preset abl_v8_datafix] [--ckpt-glob "runs/abl_v8_datafix/*_best_*.pt"] [--out out.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hrffa.train.config import get_config
from hrffa.train.evaluate import GeometricPolicy
from hrffa.data.dataset import SourceDataset, SourceSpec, collate
from hrffa.model.teacher import TeacherModel

ap = argparse.ArgumentParser()
ap.add_argument("--preset", default="abl_v8_datafix")
ap.add_argument("--ckpt-glob", default="runs/abl_v8_datafix/*_best_*.pt")
ap.add_argument("--out", default="runs/abl_v8_datafix/qa_yaw_band_conflict.json")
args = ap.parse_args()

cfg = get_config(args.preset)
cfg.num_workers = 0
ds = SourceDataset(cfg.unified, SourceSpec("300wlp", 1.0, holdout=None),
                   cfg.out_size, train=False,
                   policy=GeometricPolicy(out_size=cfg.out_size, pad=cfg.crop_pad),
                   input_norm=cfg.input_norm)

EYEBROW = list(range(17, 27)) + list(range(36, 48))   # 眉+目(変換対象ブロック)
REST = [i for i in range(68) if i not in EYEBROW]
FAB_R = list(range(17, 22)) + list(range(36, 42))     # 変換時に捏造される R 側
VIS_L = list(range(22, 27)) + list(range(42, 48))     # 実座標が入る L 側

BANDS = [(-95,-75),(-75,-65),(-65,-60),(-60,-55),(-55,-50),(-50,-30),
         (30,50),(50,55),(55,60),(60,65),(65,75),(75,95)]
PER_BAND = 100

rng = np.random.default_rng(0)
yaws = np.array([float((r.get("pose") or {}).get("euler_deg", {}).get("yaw", np.nan))
                 for r in ds.records])
picks = {}
for lo, hi in BANDS:
    idx = np.where((yaws > lo) & (yaws <= hi))[0]
    picks[(lo, hi)] = rng.choice(idx, min(PER_BAND, len(idx)), replace=False)

import glob as _glob
ck_path = sorted(_glob.glob(args.ckpt_glob))[-1]
print(f"ckpt: {ck_path}")
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
model = TeacherModel.from_config(cfg)
model.load_state_dict(ck["ema"])
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

print(f"{'yaw band':>10s} {'n':>4s} {'eye/brow':>8s} {'other':>7s} {'eb/other':>8s} {'R side':>7s} {'L side':>7s}")
rows_out = []
with torch.no_grad():
    for (lo, hi), idx in picks.items():
        eb, rest, fabr, visl = [], [], [], []
        for s in range(0, len(idx), 16):
            items = [ds[int(j)] for j in idx[s:s+16]]
            b = collate(items)
            out = model(b["image"].to(device), b["scheme"])
            pred = out["points"].float().cpu()
            d = torch.linalg.norm(pred - b["points"], dim=-1)  # (B,68) crop 正規化
            eb.append(d[:, EYEBROW].mean(1)); rest.append(d[:, REST].mean(1))
            fabr.append(d[:, FAB_R].mean(1)); visl.append(d[:, VIS_L].mean(1))
        eb = torch.cat(eb).numpy(); rest = torch.cat(rest).numpy()
        fabr = torch.cat(fabr).numpy(); visl = torch.cat(visl).numpy()
        print(f"({lo:+d},{hi:+d}] {len(idx):4d} {eb.mean():.5f} {rest.mean():.5f} "
              f"{eb.mean()/rest.mean():6.2f} {fabr.mean():.5f} {visl.mean():.5f}")
        rows_out.append(dict(band=[lo, hi], n=int(len(idx)), eyebrow=float(eb.mean()),
                             rest=float(rest.mean()), fab_r=float(fabr.mean()),
                             vis_l=float(visl.mean())))
json.dump(rows_out, open(args.out, "w"))
print(f"saved: {args.out}")
