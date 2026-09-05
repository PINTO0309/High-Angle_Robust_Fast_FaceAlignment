"""教師モデルの損失・回転表現ユーティリティ。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """6D 回転表現 (B,6) → 回転行列 (B,3,3)(Gram-Schmidt)。"""
    a1, a2 = x[:, :3], x[:, 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # 列ベクトルとして配置


def geodesic_loss(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """回転行列間の測地距離(rad)。(B,3,3)x2 → (B,)"""
    m = R_pred.transpose(1, 2) @ R_gt
    tr = m.diagonal(dim1=1, dim2=2).sum(-1)
    cos = ((tr - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos)


def yaw_from_matrix(R: torch.Tensor) -> torch.Tensor:
    """モジュール規約(geometry.py)での yaw(rad)。R[0,2] = sin(-yaw)。"""
    return -torch.asin(R[:, 0, 2].clamp(-1 + 1e-6, 1 - 1e-6))


# direction8 → yaw セクタ中心角(度)。003 §6 で確定した対応
# (yaw 正 ↔ right_*)。back 系は 300W-LP 域外だが定義は置く。
DIR8_YAW_DEG = {
    "front": 0.0, "right_front": 45.0, "right_side": 90.0, "right_back": 135.0,
    "back": 180.0, "left_back": -135.0, "left_side": -90.0, "left_front": -45.0,
}


def von_mises_yaw_loss(yaw_pred: torch.Tensor, yaw_target: torch.Tensor,
                       kappa: float = 2.0) -> torch.Tensor:
    """von Mises NLL(定数項除く)= kappa * (1 - cos(dθ))。セクタ境界の曖昧さは
    kappa を小さくして表現する(003 §6: DEIMv2 は境界が早めに *_side に倒れる)。"""
    return kappa * (1.0 - torch.cos(yaw_pred - yaw_target))


def coord_loss(pred: torch.Tensor, gt: torch.Tensor, vis: torch.Tensor,
               out_weight: float = 0.5) -> torch.Tensor:
    """正規化座標の smooth-L1。(B,N,2)。

    vis: 2=可視 1=遮蔽 0=画像外 -1=不明。画像外(0)は座標自体は正しいので
    重み out_weight で学習に含める。-1(不明)は通常の重み 1(座標は GT)。
    """
    l1 = F.smooth_l1_loss(pred, gt, beta=0.01, reduction="none").sum(-1)  # (B,N)
    w = torch.where(vis == 0, torch.full_like(l1, out_weight), torch.ones_like(l1))
    return (l1 * w).sum() / w.sum().clamp_min(1.0)


def visibility_loss(logits: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
    """3 クラス CE(0=画像外, 1=遮蔽, 2=可視)。-1(不明)は除外。"""
    mask = vis >= 0
    # Boolean selection is faster on CPU; the dense path below avoids the
    # host synchronization and dynamic gathers that are expensive on CUDA.
    if logits.device.type != "cuda":
        if not mask.any():
            return logits.sum() * 0.0
        return F.cross_entropy(logits[mask], vis[mask].long())
    targets = vis.long().masked_fill(~mask, -1)
    # A summed loss with an explicit denominator stays differentiable when every
    # label is ignored, without a GPU-to-host mask.any() or boolean-index gathers.
    # Accumulate explicit half/bfloat16 inputs in float32 to avoid sum overflow.
    # Leave autocast's CE policy intact: some PyTorch CUDA versions compute
    # log-softmax in the input dtype before accumulating NLL in float32.
    low_precision = logits.dtype in (torch.float16, torch.bfloat16)
    autocast_enabled = torch.is_autocast_enabled(logits.device.type)
    # ignore_index alone still evaluates log-softmax on ignored rows. Mask them
    # first so NaN/Inf values cannot create NaN gradients in excluded logits.
    loss_logits = logits.masked_fill(~mask.unsqueeze(-1), 0.0)
    if low_precision and not autocast_enabled:
        loss_logits = loss_logits.float()
    loss = F.cross_entropy(loss_logits.reshape(-1, logits.shape[-1]),
                           targets.reshape(-1), ignore_index=-1, reduction="sum")
    count = mask.sum()
    # Keep the legacy all-ignored result (including NaN for nonfinite logits)
    # and its zero gradients, without reading the condition back to the CPU.
    loss = torch.where(count > 0, loss / count.clamp_min(1), logits.sum() * 0.0)
    if low_precision and not autocast_enabled:
        loss = loss.to(logits.dtype)
    return loss


def roll_biternion_loss(bit_pred: torch.Tensor, roll_gt: torch.Tensor) -> torch.Tensor:
    """cosine 損失: 1 - <bit_pred, (cos r, sin r)>。bit_pred は L2 正規化済み前提。"""
    target = torch.stack([torch.cos(roll_gt), torch.sin(roll_gt)], dim=-1)
    return 1.0 - (bit_pred * target).sum(-1)


def roll_from_matrix(R: torch.Tensor) -> torch.Tensor:
    """モジュール規約での roll(rad)。R[0,1] = -cos(y')sin(r), R[0,0] = cos(y')cos(r)。"""
    return torch.atan2(-R[:, 0, 1], R[:, 0, 0])


DIR8_YAW_RAD = {k: math.radians(v) for k, v in DIR8_YAW_DEG.items()}
