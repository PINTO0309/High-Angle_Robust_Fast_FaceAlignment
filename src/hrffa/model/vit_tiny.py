"""学生 S 用バックボーン: vitt_distill(DINOv3 系 RoPE の最小構成 ViT-T/16)。

state dict 構造(実測、history/026):
    cls_token, patch_embed.proj.{weight,bias}, rope_embed.periods,
    blocks.N.{norm1,attn.qkv,attn.proj,norm2,mlp.fc1,mlp.fc2}.{weight,bias}
storage/mask トークン・最終 norm・layerscale は持たない。

RoPE は DINOv3 公開仕様に合わせた自前実装(コード流用はしない方針: history/008):
  - パッチ中心座標を軸ごとに [0,1] 正規化("separate")→ [-1,1]
  - angles = 2π · coord / periods(periods は checkpoint のバッファをロード)
  - 軸順 (h, w) で D_head/2 を構成し 2 回タイル → rotate-half 適用
  - cls トークンには適用しない
RoPE のため入力解像度(256/192/128)の変更に位置埋め込み補間は不要。
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class Rope2D(nn.Module):
    """軸分離 2D RoPE。periods は checkpoint からロードするバッファ。"""

    def __init__(self, head_dim: int):
        super().__init__()
        assert head_dim % 4 == 0
        self.head_dim = head_dim
        self.register_buffer("periods", torch.empty(head_dim // 4),
                             persistent=True)
        self._cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def forward(self, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (h, w, str(self.periods.device))
        if key not in self._cache:
            dd = {"device": self.periods.device, "dtype": torch.float32}
            ch = torch.arange(0.5, h, **dd) / h
            cw = torch.arange(0.5, w, **dd) / w
            coords = torch.stack(torch.meshgrid(ch, cw, indexing="ij"),
                                 dim=-1).flatten(0, 1)          # (hw, 2)
            coords = 2.0 * coords - 1.0
            ang = 2 * math.pi * coords[:, :, None] / self.periods.float()[None, None]
            ang = ang.flatten(1, 2).tile(2)                      # (hw, D_head)
            self._cache[key] = (torch.sin(ang), torch.cos(ang))
        return self._cache[key]


def _rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def _rope_apply(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    dt = x.dtype
    x = x.float()
    return ((x * cos) + (_rope_rotate_half(x) * sin)).to(dt)


# export_modules.to_export_model(static=True) が True にする。形状を Python int に落として
# Reshape を定数化する(ONNX でバッチ軸を先頭に持つ定数 shape になる)。学習時は False
STATIC_SHAPES = False


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor],
                prefix: int) -> torch.Tensor:
        b, n, c = x.shape
        if STATIC_SHAPES:
            b, n, c = int(b), int(n), int(c)
        # qkv の分割は 4 次元テンソルだけで行う(2026-08-29): 旧実装の reshape(B,N,3,H,Dh) → permute(2,0,3,1,4) → unbind は
        # ONNX に 5 次元の Reshape / Transpose / Split を残す(ブロックあたり 5 テンソル)。チャネル順は [3][H][Dh] なので、
        # 末尾軸で 3 分割 → (B,N,H,Dh) → transpose(1,2) は数値的に同一(単体テストで確認)
        q, k, v = (t.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
                   for t in self.qkv(x).split(c, dim=-1))   # (B, heads, N, Dh)
        sin, cos = rope
        q = torch.cat([q[:, :, :prefix], _rope_apply(q[:, :, prefix:], sin, cos)], dim=2)
        k = torch.cat([k[:, :, :prefix], _rope_apply(k[:, :, prefix:], sin, cos)], dim=2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _LocalConv(nn.Module):
    """LESA 由来の局所項(history/045 D7): 3×3 depthwise conv → GELU → 1×1(zero-init)。

    注意(文脈項)と並列に同じ正規化入力へ適用する。1×1 を zero-init、ゲートを 1 で初期化するため
    追加直後は出力が 0 で、事前学習重みの挙動を変えない。prefix(cls)トークンには適用しない。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.GELU()
        self.pw = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.pw.bias)
        self.gate = nn.Parameter(torch.ones(dim))

    def forward(self, tokens: torch.Tensor, h: int, w: int, prefix: int) -> torch.Tensor:
        b, n, c = tokens.shape
        z = tokens[:, prefix:].transpose(1, 2).reshape(b, c, h, w)
        z = self.pw(self.act(self.dw(z))) * self.gate.view(1, -1, 1, 1)
        z = z.flatten(2).transpose(1, 2)
        return torch.cat([tokens.new_zeros(b, prefix, c), z], dim=1)


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 local_conv: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))
        self.local = _LocalConv(dim) if local_conv else None

    def forward(self, x, rope, prefix, hw=None):
        y = self.norm1(x)
        x = x + self.attn(y, rope, prefix)
        if self.local is not None:
            x = x + self.local(y, hw[0], hw[1], prefix)
        return x + self.mlp(self.norm2(x))


class VitTiny(nn.Module):
    """vitt_distill 互換の最小 ViT。forward は (patch BCHW, cls BC) を返す。"""

    def __init__(self, embed_dim: int = 192, depth: int = 12, num_heads: int = 3,
                 patch_size: int = 16, patch_instance_norm: bool = False,
                 local_conv: bool = False, feat_layers: tuple[int, ...] = ()):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = nn.ModuleDict(
            {"proj": nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)})
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.rope_embed = Rope2D(embed_dim // num_heads)
        self.blocks = nn.ModuleList(
            [_Block(embed_dim, num_heads, local_conv=local_conv) for _ in range(depth)])
        # 教師と同じ思想の入口 IN(023 §4)。checkpoint 由来キーとは独立の新設枝
        self.patch_in = (nn.InstanceNorm2d(embed_dim, affine=True)
                         if patch_instance_norm else None)
        # 多層特徴の融合(history/045 D5): 指定ブロックの出力をチャネル結合して返す。
        # 最終ブロック以外には新設の LayerNorm をかける(最終ブロックは従来どおり無正規化)
        feat_layers = tuple(sorted(set(int(i) for i in feat_layers)))
        for i in feat_layers:
            if not 0 <= i < depth:
                raise ValueError(f"feat_layers index {i} is out of range (depth={depth})")
        self.feat_layers = feat_layers
        self.feat_norms = nn.ModuleList(
            [nn.Identity() if i == depth - 1 else nn.LayerNorm(embed_dim) for i in feat_layers])
        self.embed_dim_out = embed_dim * max(len(feat_layers), 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.patch_embed["proj"](x)                  # (B, C, h, w)
        if self.patch_in is not None:
            z = self.patch_in(z)
        b, c, h, w = z.shape
        if STATIC_SHAPES:
            # 静的 export: 形状を定数化し、cls の複製は入力値に依存させて(×0 + cls)定数畳み込みを防ぐ
            # (バッチ次元を後から書き換えても cls / Reshape が追従する)
            tokens = z.reshape(int(b), int(c), int(h) * int(w)).transpose(1, 2)
            cls = tokens[:, :1, :1] * 0 + self.cls_token
        else:
            tokens = z.flatten(2).transpose(1, 2)        # (B, hw, C)
            cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        rope = self.rope_embed(h, w)
        feats = []
        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens, rope, prefix=1, hw=(h, w))
            if i in self.feat_layers:
                feats.append(tokens)
        if self.feat_layers:
            tokens = torch.cat([norm(t) for t, norm in zip(feats, self.feat_norms)], dim=-1)
        c_out = tokens.shape[-1]
        if STATIC_SHAPES:
            b, c_out, h, w = int(b), int(c_out), int(h), int(w)
        cls, patch = tokens[:, 0], tokens[:, 1:]
        patch = patch.transpose(1, 2).reshape(b, c_out, h, w)
        return patch, cls

    def load_flat_ckpt(self, path: Path, input_norm: str = "imagenet") -> None:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd.get("state_dict", sd))
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()
        # ModuleDict 経由の patch_embed はキーが一致する(patch_embed.proj.*)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        # patch_in(新設)以外の欠落・余剰は構造不一致なのでエラーにする
        # 新設枝(patch_in / feat_norms / blocks.N.local)は checkpoint に無くてよい
        bad_missing = [k for k in missing
                       if not (k.startswith(("patch_in", "feat_norms")) or ".local." in k)]
        if bad_missing or unexpected:
            raise RuntimeError(
                f"vitt checkpoint structure mismatch: missing={bad_missing} unexpected={list(unexpected)}")
        if input_norm != "imagenet":
            self._fold_input_norm(input_norm)

    @torch.no_grad()
    def _fold_input_norm(self, input_norm: str) -> None:
        """事前学習重み(ImageNet 正規化前提)を別の入力正規化仕様へ厳密変換する。

        z_imagenet = a_c * z_new + b_c(チャネル毎アフィン)なので、
        patch embed conv の重みスケールとバイアス補正に折り込める(無劣化)。
        """
        from .backbone import IMAGE_MEAN, IMAGE_STD, norm_constants
        mean_new, std_new = norm_constants(input_norm)
        conv = self.patch_embed["proj"]
        for c in range(3):
            a = std_new[c] / IMAGE_STD[c]
            b = (mean_new[c] - IMAGE_MEAN[c]) / IMAGE_STD[c]
            conv.bias += conv.weight[:, c].sum(dim=(1, 2)) * b
            conv.weight[:, c] *= a
