"""教師モデル本体: DINOv3 backbone + 点クエリ Transformer デコーダ。

設計要件(history/004 §7・005 §5 で確定):
  - 点クエリ方式: scheme(ibug68/wflw98/cofw29)ごとの学習可能クエリ埋め込み。
    将来の点数拡張はクエリ追加のみで可能。
  - 座標は線形出力(sigmoid なし)。クロップ外の点(可視性 0)も座標を持つため。
    正規化はクロップ辺長基準(0..1 が枠内)。
  - 可視性 3 クラス(0=画像外, 1=遮蔽, 2=可視)を点ごとに出力。
  - 姿勢: 6D 回転表現(ジンバルロック回避)+ Roll Biternion 補助ヘッド。
  - 蒸留用にデコーダ出力トークンとメモリ特徴を返す。
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .backbone import Dinov3Backbone
from .losses import rot6d_to_matrix
from .popos import multilaterate

SCHEMES = {"ibug68": 68, "wflw98": 98, "cofw29": 29}
# export_modules.to_export_model(static=True) が True にする(Reshape を定数形状で書き出す)
STATIC_SHAPES = False


def sincos_pos_embed_2d(d_model: int, h: int, w: int, device) -> torch.Tensor:
    """(h*w, d_model) の 2D sin-cos 位置埋め込み。"""
    assert d_model % 4 == 0
    quarter = d_model // 4
    omega = torch.arange(quarter, device=device) / quarter
    omega = 1.0 / (10000 ** omega)
    ys, xs = torch.meshgrid(torch.arange(h, device=device),
                            torch.arange(w, device=device), indexing="ij")
    out = []
    for pos in (xs, ys):
        ang = pos.reshape(-1, 1).float() * omega[None]
        out += [torch.sin(ang), torch.cos(ang)]
    return torch.cat(out, dim=1)  # (h*w, d)


class TeacherModel(nn.Module):
    def __init__(
        self,
        backbone_variant: str = "vitl16",
        ckpt_dir: Path = Path("ckpts"),
        d_model: int = 256,
        dec_layers: int = 4,
        n_heads: int = 8,
        ffn_dim: int = 1024,
        schemes: dict[str, int] | None = None,
        patch_instance_norm: bool = False,
        input_norm: str = "imagenet",
        local_conv: bool = False,
        feat_layers: tuple[int, ...] = (),
        dec_local_iters: int = 0,
        head: str = "regress",
        popos_topk: int = 6,
        feat_stride: int = 16,
        cnn_feat_ch: int = 128,
    ):
        super().__init__()
        self.schemes = schemes or dict(SCHEMES)
        self.backbone = Dinov3Backbone(backbone_variant, ckpt_dir,
                                       patch_instance_norm=patch_instance_norm,
                                       input_norm=input_norm,
                                       local_conv=local_conv, feat_layers=tuple(feat_layers),
                                       feat_stride=feat_stride, cnn_feat_ch=cnn_feat_ch)
        c = self.backbone.embed_dim
        if head not in ("regress", "popos"):
            raise ValueError(f"head must be regress | popos: {head!r}")
        if head == "popos" and dec_local_iters > 0:
            raise ValueError("head=popos cannot be combined with dec_local_iters>0 (045: one factor per arm)")
        self.head, self.dec_local_iters, self.popos_topk = head, int(dec_local_iters), int(popos_topk)
        self.input_proj = nn.Linear(c, d_model)
        self.cls_proj = nn.Linear(c, d_model)

        self.queries = nn.ParameterDict({
            name: nn.Parameter(torch.randn(n, d_model) * 0.02)
            for name, n in self.schemes.items()
        })
        layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward=ffn_dim, dropout=0.0,
            batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, dec_layers,
                                             norm=nn.LayerNorm(d_model))
        self.coord_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.vis_head = nn.Linear(d_model, 3)
        self.pose_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 8))
        # 6D 回転の初期値を単位行列近傍にする
        nn.init.zeros_(self.pose_head[-1].bias)
        with torch.no_grad():
            self.pose_head[-1].bias[:6] = torch.tensor([1., 0., 0., 0., 1., 0.])

        # D6(history/045): デコーダ局所項。推定座標で memory を採取した局所特徴を、
        # ゲート(LESA の動的融合)でクエリに混ぜてデコーダを再適用し、座標を残差で精密化する。
        # delta_head の最終層を zero-init するため、追加直後は反復なしと同一出力
        if self.dec_local_iters > 0:
            self.local_proj = nn.Linear(d_model, d_model)
            self.local_gate = nn.Sequential(
                nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            self.delta_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
            nn.init.zeros_(self.delta_head[-1].weight)
            nn.init.zeros_(self.delta_head[-1].bias)
        # D8(history/045): POPoS 型。クエリ × memory の双線形形式で距離マップ(セル単位)を予測し、
        # top-K アンカーの multilateration で座標を復元(coord_head は使わない)
        if self.head == "popos":
            self.dist_q = nn.Linear(d_model, d_model)
            self.dist_m = nn.Linear(d_model, d_model)

        self._pos_cache: dict[tuple, torch.Tensor] = {}

    @classmethod
    def from_config(cls, cfg, ckpt_dir: Path = Path("ckpts")) -> "TeacherModel":
        """TrainConfig からモデルを構築する(全呼び出し箇所で共通に使う)。"""
        return cls(cfg.backbone, ckpt_dir, d_model=cfg.d_model, dec_layers=cfg.dec_layers,
                   n_heads=getattr(cfg, "dec_heads", 8), ffn_dim=getattr(cfg, "dec_ffn", 1024),
                   patch_instance_norm=cfg.patch_instance_norm, input_norm=cfg.input_norm,
                   feat_stride=getattr(cfg, "feat_stride", 16), cnn_feat_ch=getattr(cfg, "cnn_feat_ch", 128),
                   local_conv=getattr(cfg, "local_conv", False),
                   feat_layers=tuple(getattr(cfg, "feat_layers", ()) or ()),
                   dec_local_iters=getattr(cfg, "dec_local_iters", 0),
                   head=getattr(cfg, "head", "regress"),
                   popos_topk=getattr(cfg, "popos_topk", 6))

    def _refine(self, dec, coords, mem, h, w):
        b, hw, d = mem.shape
        if STATIC_SHAPES:
            b, d = int(b), int(d)
        mem_map = mem.transpose(1, 2).reshape(b, d, h, w)
        grid = (coords * 2.0 - 1.0).unsqueeze(2)                          # (B,N,1,2) x,y ∈ [-1,1]
        local = F.grid_sample(mem_map, grid, mode="bilinear", padding_mode="zeros",
                              align_corners=False).squeeze(-1).transpose(1, 2)  # (B,N,d)
        gate = torch.sigmoid(self.local_gate(torch.cat([dec, local], dim=-1)))
        q2 = dec + gate * self.local_proj(local)
        dec2 = self.decoder(q2, mem)
        return dec2, coords + self.delta_head(dec2)

    def _popos(self, dec, mem, h, w):
        d = dec.shape[-1]
        s = torch.einsum("bnd,bmd->bnm", self.dist_q(dec), self.dist_m(mem)) / math.sqrt(d)
        dist_pred = F.softplus(s)                                         # (B,N,hw) セル単位
        return multilaterate(dist_pred, h, w, self.popos_topk), dist_pred

    def _pos(self, h: int, w: int, d: int, device) -> torch.Tensor:
        key = (h, w, d, str(device))
        if key not in self._pos_cache:
            self._pos_cache[key] = sincos_pos_embed_2d(d, h, w, device)
        return self._pos_cache[key]

    def forward(self, images: torch.Tensor, scheme: str) -> dict:
        patch, cls = self.backbone(images)          # (B,C,h,w), (B,C)
        b, c, h, w = patch.shape
        if STATIC_SHAPES:
            b, c, h, w = int(b), int(c), int(h), int(w)
        mem = self.input_proj(patch.reshape(b, c, h * w).transpose(1, 2))  # (B,hw,d)
        mem = mem + self._pos(h, w, mem.shape[-1], mem.device)[None]

        if STATIC_SHAPES:                                        # 静的 export: 入力値依存の複製(定数畳み込み防止)
            q = mem[:, :1, :1] * 0 + self.queries[scheme][None]
        else:
            q = self.queries[scheme][None].expand(b, -1, -1)     # (B,N,d)
        dec = self.decoder(q, mem)                               # (B,N,d)

        dist_pred = None
        if self.head == "popos":
            coords, dist_pred = self._popos(dec, mem, h, w)
        else:
            coords = self.coord_head(dec)                        # 正規化座標(線形)
            dec_r = dec
            for _ in range(self.dec_local_iters):               # D6: 座標のみ精密化(vis / KD トークンは 1 回目)
                dec_r, coords = self._refine(dec_r, coords, mem, h, w)
        vis_logits = self.vis_head(dec)

        g = torch.cat([self.cls_proj(cls), mem.mean(dim=1)], dim=-1)
        pose = self.pose_head(g)                                 # (B,8)
        rot6d, roll_bit = pose[:, :6], pose[:, 6:]
        roll_bit = nn.functional.normalize(roll_bit, dim=-1)
        R = rot6d_to_matrix(rot6d)

        out = {"points": coords, "vis_logits": vis_logits,
               "rot": R, "roll_bit": roll_bit,
               "dec_tokens": dec, "memory": mem}
        if dist_pred is not None:
            out["dist_pred"] = dist_pred
            out["grid_hw"] = (h, w)
        return out

    def param_groups(self, lr_backbone: float, lr_head: float, weight_decay: float):
        """凍結中(requires_grad=False)のパラメータも登録する。凍結中は grad が
        None のため AdamW の step で更新されず、解凍後は自動的に更新が始まる。
        (requires_grad で除外すると解凍後も永久に更新されないバグになる)"""
        bb, head = [], []
        for n, p in self.named_parameters():
            (bb if n.startswith("backbone.") else head).append(p)
        return [
            {"params": bb, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": head, "lr": lr_head, "weight_decay": weight_decay},
        ]

    def freeze_backbone(self, freeze: bool = True) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
