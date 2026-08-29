"""Roll 等変性の目視検証グリッド生成(history/024)。

1 サンプルに Roll を等間隔 N 方位で適用し、各方位のモデル予測(単色)と厳密 GT に対する
inter-ocular NME をタイルに焼き込み、正方グリッド画像を出力する。

usage:
  uv run python -m hrffa.dataset.qa.roll_grid \\
    --preset abl_v5_innorm_96gb \\
    --ckpt "runs/abl_v5_innorm_96gb/abl_v5_innorm_96gb_best_*.pt" \\
    --source wflw:test --idx 42 --out roll_grid_3x3.jpg
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
import torch

from ...data.dataset import SourceDataset, SourceSpec
from ...model.teacher import TeacherModel
from ...train.config import get_config
from ..augment.geometric import GeometricParams, GeometricPolicy, apply_geometric

_INTEROCULAR = {"ibug68": (36, 45), "wflw98": (60, 72), "cofw29": (8, 9)}
_COLOR = (0, 200, 0)  # 単色描画(可視性の色分けはしない)


def render_roll_grid(model, ds: SourceDataset, idx: int, device: str,
                     n_tiles: int = 9) -> tuple[np.ndarray, list[float]]:
    """(グリッド画像 BGR, 方位別 io-NME) を返す。n_tiles は平方数であること。"""
    side = int(round(n_tiles ** 0.5))
    assert side * side == n_tiles, "n_tiles must be a square number"
    s = ds.out_size
    rec = ds.records[idx]
    a, b = _INTEROCULAR[ds.scheme]
    img0 = cv2.imread(str(ds.unified / rec["image_path"]))
    if img0 is None:
        raise FileNotFoundError(ds.unified / rec["image_path"])
    pts0 = np.asarray(rec["landmarks"]["points"], dtype=np.float64)
    vis0 = rec["landmarks"]["visibility"]
    mean, std = ds.norm_mean, ds.norm_std

    tiles, nmes = [], []
    with torch.no_grad():
        for k in range(n_tiles):
            roll = 360.0 * k / n_tiles
            prm = GeometricParams(out_size=s, pad=ds.policy.pad, roll_deg=roll)
            out = apply_geometric(img0, pts0, vis0, None, rec["head_bbox"], prm,
                                  flip_mapping=ds.flip_mapping)
            x = cv2.cvtColor(out["image"], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t = torch.from_numpy(((x - mean) / std).transpose(2, 0, 1).copy())
            o = model(t[None].to(device), ds.scheme)
            pred = o["points"].float().cpu().numpy()[0] * s
            gt = out["points"]
            iod = max(float(np.linalg.norm(gt[a] - gt[b])), 1e-6)
            nme = float(np.linalg.norm(pred - gt, axis=-1).mean()) / iod
            nmes.append(nme)
            tile = out["image"].copy()
            for (px, py) in pred:
                cv2.circle(tile, (int(px), int(py)), 2, _COLOR, -1, cv2.LINE_AA)
            cv2.rectangle(tile, (0, 0), (s - 1, 22), (0, 0, 0), -1)
            cv2.putText(tile, f"roll={roll:5.1f}  io-nme={nme*100:.2f}%", (6, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)
            tiles.append(tile)
    grid = np.vstack([np.hstack(tiles[i:i + side]) for i in range(0, n_tiles, side)])
    return grid, nmes


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a visual grid for checking roll equivariance (history/024).")
    ap.add_argument("--preset", required=True)
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint path (glob allowed; the last match = newest)")
    ap.add_argument("--source", default="wflw:test", help="e.g. wflw:test / 300w:valid_common")
    ap.add_argument("--idx", type=int, default=42)
    ap.add_argument("--n-tiles", type=int, default=9, help="square number (9=3x3, 16=4x4)")
    ap.add_argument("--out", type=Path, default=Path("roll_grid_3x3.jpg"))
    args = ap.parse_args()

    cfg = get_config(args.preset)
    name, split = args.source.split(":")
    ds = SourceDataset(cfg.unified, SourceSpec(name, 1.0, (split,)), cfg.out_size,
                       train=False, policy=GeometricPolicy(out_size=cfg.out_size, pad=cfg.crop_pad),
                       input_norm=cfg.input_norm)
    matches = sorted(glob.glob(str(args.ckpt)))
    if not matches:
        raise FileNotFoundError(f"no file matches --ckpt: {args.ckpt}")
    ck = torch.load(matches[-1], map_location="cpu", weights_only=False)
    model = TeacherModel.from_config(cfg)
    model.load_state_dict(ck.get("ema") or ck.get("model") or ck)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    grid, nmes = render_roll_grid(model, ds, args.idx, device, args.n_tiles)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    rid = ds.records[args.idx]["record_id"]
    print(f"sample={rid} nme%={[round(v*100, 2) for v in nmes]}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
