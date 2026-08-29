"""POPoS 型サブトークンデコード(history/001 §、045 D8)の自前実装。

各ランドマーク(クエリ)がメモリ格子の各セルまでの距離 d(セル単位)を予測し、
予測距離が最小の top-K セルをアンカーとして multilateration(閉形式最小二乗)で座標を復元する。
低解像度(16×16)の格子でもサブセル精度の座標が出せる。

座標系: 正規化座標 (x, y) ∈ [0,1](クロップ辺基準)。格子 (h, w) のセル (j, i) の中心は
セル単位で (i + 0.5, j + 0.5)、正規化では ((i + 0.5)/w, (j + 0.5)/h)。距離はセル単位。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def grid_centers_cells(h: int, w: int, device) -> torch.Tensor:
    """(hw, 2) のセル中心座標(セル単位、x=列, y=行)。"""
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    return torch.stack([xs.reshape(-1) + 0.5, ys.reshape(-1) + 0.5], dim=-1)


def multilaterate(dist: torch.Tensor, h: int, w: int, k: int = 6,
                  ridge: float = 1e-3) -> torch.Tensor:
    """予測距離マップ (B,N,hw) から top-K アンカーの multilateration で正規化座標 (B,N,2) を復元。

    各アンカー i について ||x - p_i||^2 = d_i^2。基準アンカー 0 との差をとると x について線形:
      2 (p_i - p_0)·x = (||p_i||^2 - ||p_0||^2) - (d_i^2 - d_0^2)
    これを最小二乗(2×2 正規方程式、閉形式)で解く。top-K の選択は非微分だが、選択後の解は
    d_i について微分可能(座標損失が距離予測に流れる)。ONNX: TopK / Gather / MatMul のみ。
    """
    b, n, m = dist.shape
    centers = grid_centers_cells(h, w, dist.device)                      # (hw,2)
    d_k, idx = torch.topk(dist, k, dim=-1, largest=False)                # (B,N,k)
    p = centers[idx]                                                     # (B,N,k,2)
    p0, d0 = p[..., :1, :], d_k[..., :1]
    A = 2.0 * (p[..., 1:, :] - p0)                                       # (B,N,k-1,2)
    rhs = ((p[..., 1:, :] ** 2).sum(-1) - (p0 ** 2).sum(-1)
           - (d_k[..., 1:] ** 2 - d0 ** 2))                              # (B,N,k-1)
    At = A.transpose(-1, -2)                                             # (B,N,2,k-1)
    AtA = At @ A                                                         # (B,N,2,2)
    Atb = (At @ rhs.unsqueeze(-1)).squeeze(-1)                           # (B,N,2)
    a, bb = AtA[..., 0, 0] + ridge, AtA[..., 0, 1]
    c, d = AtA[..., 1, 0], AtA[..., 1, 1] + ridge
    det = a * d - bb * c
    x = (d * Atb[..., 0] - bb * Atb[..., 1]) / det
    y = (-c * Atb[..., 0] + a * Atb[..., 1]) / det
    scale = torch.tensor([float(w), float(h)], device=dist.device)
    return torch.stack([x, y], dim=-1) / scale


def distance_targets(points: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """GT 正規化座標 (B,N,2) → 各セル中心までの距離 (B,N,hw)(セル単位)。"""
    centers = grid_centers_cells(h, w, points.device)                    # (hw,2)
    scale = torch.tensor([float(w), float(h)], device=points.device)
    gt = points * scale                                                  # (B,N,2) セル単位
    return ((gt[:, :, None, :] - centers[None, None]) ** 2).sum(-1).clamp_min(0).sqrt()


def distance_loss(dist_pred: torch.Tensor, points: torch.Tensor, vis: torch.Tensor,
                  h: int, w: int, radius: float = 6.0, out_weight: float = 0.5) -> torch.Tensor:
    """距離マップ損失: GT 点から radius セル以内は L1、外側は hinge(radius 未満を予測したら罰)。

    点ごとの重みは coord_loss と同じ(vis==0 の画像外点は out_weight)。
    """
    d_gt = distance_targets(points.float(), h, w)                        # (B,N,hw)
    near = d_gt <= radius
    l_near = (F.l1_loss(dist_pred, d_gt, reduction="none") * near).sum(-1) / near.sum(-1).clamp_min(1)
    far = ~near
    l_far = (F.relu(radius - dist_pred) * far).sum(-1) / far.sum(-1).clamp_min(1)
    per_point = l_near + l_far                                           # (B,N)
    wgt = torch.where(vis == 0, torch.full_like(per_point, out_weight), torch.ones_like(per_point))
    return (per_point * wgt).sum() / wgt.sum().clamp_min(1.0)
