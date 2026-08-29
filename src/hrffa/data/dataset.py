"""統合データセット → 教師学習用 Dataset / サンプラ。

- 1 ソース(jsonl)= 1 SourceDataset(scheme は単一)
- 学習時: D4 幾何拡張(GeometricPolicy)+ 軽量 photometric 拡張
- 評価時: 決定的クロップ(拡張なし)
- バッチは scheme 同種で構成する(SchemeBatchSampler がソース重みで選択)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..dataset.augment.geometric import GeometricParams, GeometricPolicy, apply_geometric
from ..model.backbone import IMAGE_MEAN, IMAGE_STD
from ..model.losses import DIR8_YAW_RAD


def _photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """軽量 photometric 拡張(BGR uint8 → BGR uint8)。"""
    x = img.astype(np.float32)
    if rng.random() < 0.8:  # 明度・コントラスト・ガンマ
        x = x * rng.uniform(0.6, 1.4) + rng.uniform(-25, 25)
        x = 255.0 * (x.clip(0, 255) / 255.0) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.2:  # グレースケール化
        g = cv2.cvtColor(x.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        x = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR).astype(np.float32)
    if rng.random() < 0.3:  # ガウスノイズ
        x = x + rng.normal(0, rng.uniform(3, 12), x.shape)
    if rng.random() < 0.3:  # ぼかし
        k = int(rng.choice([3, 5]))
        x = cv2.GaussianBlur(x, (k, k), 0)
    x = x.clip(0, 255).astype(np.uint8)
    if rng.random() < 0.3:  # JPEG 劣化
        q = int(rng.integers(35, 85))
        _, enc = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, q])
        x = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return x


@dataclass
class SourceSpec:
    name: str          # jsonl 名(300wlp / wflw / 300w / cofw / synth_dwarp)
    weight: float      # バッチ抽選の重み
    splits: tuple[str, ...] = ("train",)
    # 決定的ホールドアウト: record_id の CRC32 % holdout_mod == 0 を検証用に確保。
    # holdout="train" は該当レコードを除外、"val" は該当レコードのみ、None は全件。
    holdout: str | None = None
    holdout_mod: int = 50  # ≒2%
    # 姿勢 GT を持つソースの |yaw| 上限(None = 制限なし)。history/039: 300W-LP 系は
    # |yaw|>20 で投影ラベルが描画画素からずれるため、クリーン学習では除外する
    yaw_max: float | None = None


def _motion_blur(img: np.ndarray, rng: np.random.Generator,
                 max_frac: float = 0.06) -> np.ndarray:
    """線形モーションブラー(長さ = 辺の 1〜max_frac、角度一様)。GT は不変。"""
    h, w = img.shape[:2]
    length = int(max(3, rng.uniform(0.01, max_frac) * max(h, w)))
    length += (length + 1) % 2
    k = np.zeros((length, length), np.float32)
    c = length // 2
    th = rng.uniform(0, np.pi)
    for t in np.linspace(-c, c, length * 4):
        x, y = int(round(c + t * np.cos(th))), int(round(c + t * np.sin(th)))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1.0
    k /= max(k.sum(), 1.0)
    return cv2.filter2D(img, -1, k)


def _random_erase(img: np.ndarray, rng: np.random.Generator,
                  n_max: int = 2) -> np.ndarray:
    """矩形領域をノイズ/平均色で消去する遮蔽拡張(ランドマーク GT は不変。
    可視性 GT も更新しない = 合成遮蔽下でも位置を当てる学習を意図)。"""
    h, w = img.shape[:2]
    out = img.copy()
    for _ in range(int(rng.integers(1, n_max + 1))):
        ew = int(w * rng.uniform(0.10, 0.28))
        eh = int(h * rng.uniform(0.10, 0.28))
        x0 = int(rng.integers(0, max(w - ew, 1)))
        y0 = int(rng.integers(0, max(h - eh, 1)))
        if rng.random() < 0.5:
            out[y0:y0 + eh, x0:x0 + ew] = rng.integers(
                0, 256, (eh, ew, 3), dtype=np.uint8)
        else:
            out[y0:y0 + eh, x0:x0 + ew] = out[y0:y0 + eh, x0:x0 + ew].mean(
                axis=(0, 1), keepdims=True).astype(np.uint8)
    return out


class SourceDataset(Dataset):
    def __init__(self, unified: Path, spec: SourceSpec, out_size: int,
                 train: bool, policy: GeometricPolicy, seed: int = 0,
                 erase_prob: float = 0.0, input_norm: str = "imagenet",
                 motion_blur_prob: float = 0.0):
        from ..model.backbone import norm_constants
        self.unified = unified
        self.spec = spec
        self.out_size = out_size
        self.train = train
        self.policy = policy
        self.input_norm = input_norm
        mean, std = norm_constants(input_norm)
        self.norm_mean = np.array(mean, np.float32)
        self.norm_std = np.array(std, np.float32)
        self.records: list[dict] = []
        import zlib
        with open(unified / "annotations" / f"{spec.name}.jsonl", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if spec.yaw_max is not None:
                    _yaw = ((rec.get("pose") or {}).get("euler_deg") or {}).get("yaw")
                    if _yaw is not None and abs(float(_yaw)) > spec.yaw_max:
                        continue
                if rec["split"] not in spec.splits:
                    continue
                if spec.holdout is not None:
                    # 合成データは親レコード基準で判定(親が検証側なら子も検証側 →
                    # 学習への同一人物リーク防止)
                    key = rec["attributes"].get("parent_record", rec["record_id"])
                    held = zlib.crc32(key.encode()) % spec.holdout_mod == 0
                    if (spec.holdout == "train" and held) or \
                       (spec.holdout == "val" and not held):
                        continue
                self.records.append(rec)
        self.scheme = self.records[0]["landmarks"]["scheme"]
        meta = json.loads(
            (unified / "annotations" / "meta" / f"{self.scheme}.json").read_text())
        self.flip_mapping = meta["flip_mapping"]
        self.seed = seed
        self.erase_prob = erase_prob
        self.motion_blur_prob = motion_blur_prob

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        rng = np.random.default_rng(None if self.train else self.seed + idx)
        img = cv2.imread(str(self.unified / rec["image_path"]))
        pts = np.asarray(rec["landmarks"]["points"], dtype=np.float64)
        vis = rec["landmarks"]["visibility"]
        R = (np.asarray(rec["pose"]["rotation_matrix"], dtype=np.float64)
             if rec["pose"] and rec["pose"].get("rotation_matrix") else None)

        prm = (self.policy.sample(rng) if self.train
               else GeometricParams(out_size=self.out_size, pad=self.policy.pad))
        prm.out_size = self.out_size
        out = apply_geometric(img, pts, vis, R, rec["head_bbox"], prm,
                              flip_mapping=self.flip_mapping)
        crop = out["image"]
        if self.train:
            crop = _photometric(crop, rng)
            if self.motion_blur_prob > 0 and rng.random() < self.motion_blur_prob:
                crop = _motion_blur(crop, rng)
            if self.erase_prob > 0 and rng.random() < self.erase_prob:
                crop = _random_erase(crop, rng)

        x = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - self.norm_mean) / self.norm_std

        s = float(self.out_size)
        target_pts = np.asarray(out["points"], dtype=np.float32) / s

        dir8 = rec.get("direction8")
        # flip 時は direction8 の左右も反転する
        if dir8 is not None and prm.hflip:
            dir8 = dir8.replace("left", "TMP").replace("right", "left").replace("TMP", "right")
        yaw_weak = DIR8_YAW_RAD.get(dir8, 0.0) if dir8 else 0.0

        return {
            "image": torch.from_numpy(x.transpose(2, 0, 1)),
            "points": torch.from_numpy(target_pts),
            "vis": torch.tensor(out["visibility"], dtype=torch.long),
            "rot": torch.tensor(out["rotation"], dtype=torch.float32)
                   if out["rotation"] is not None else torch.zeros(3, 3),
            "has_rot": torch.tensor(out["rotation"] is not None),
            "yaw_weak": torch.tensor(yaw_weak, dtype=torch.float32),
            "has_dir8": torch.tensor(dir8 is not None and out["rotation"] is None),
            "scheme_name": self.scheme,
        }


def collate(batch: list[dict]) -> dict:
    out = {k: torch.stack([b[k] for b in batch]) for k in
           ("image", "points", "vis", "rot", "has_rot", "yaw_weak", "has_dir8")}
    out["scheme"] = batch[0]["scheme_name"]
    return out


class SchemeBatchSampler(Sampler):
    """ソース重みで 1 ソースを選び、そのソース内からバッチを作る(scheme 同種)。"""

    def __init__(self, sizes: list[int], weights: list[float], batch_size: int,
                 steps_per_epoch: int, seed: int = 0):
        self.sizes = sizes
        self.p = np.asarray(weights, dtype=np.float64)
        self.p /= self.p.sum()
        self.batch_size = batch_size
        self.steps = steps_per_epoch
        self.seed = seed
        self.epoch = 0
        self.offsets = np.cumsum([0] + sizes[:-1])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.steps):
            src = int(rng.choice(len(self.sizes), p=self.p))
            idx = rng.integers(0, self.sizes[src], size=self.batch_size)
            yield (self.offsets[src] + idx).tolist()
