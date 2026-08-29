"""ONNX export 用の等価モジュール(history/045 §6 追記 2026-08-28)。

torch.nn.MultiheadAttention / TransformerDecoder は内部で (L, B·H, D) のようにバッチ軸をヘッド軸と
結合して Reshape するため、書き出した ONNX ではバッチ次元が潰れた Reshape が並ぶ(意味は正しいが、
後からバッチ次元を書き換える運用ができない)。ここでは同じ重みで同じ計算を (B, H, L, D) レイアウトで
行う実装に差し替え、全ての Reshape でバッチ軸を先頭に独立した次元として維持する。
学習側のモジュールは変更しない(export 時にのみモデルの decoder をその場で差し替える)。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


# 静的 export(--dynamic なし)のときに True。形状を Python int に落として Reshape を定数化する
# (先頭要素がバッチ = 1 の定数 shape になり、後からバッチ次元を書き換えるツールが扱いやすい)。
# 動的 export では False のまま(Shape 由来の形状で任意バッチに追従)
STATIC_SHAPES = False


def _dims(*vals):
    return tuple(int(v) for v in vals) if STATIC_SHAPES else vals


class ExportMHA(nn.Module):
    """nn.MultiheadAttention(batch_first=True)と同一計算。Reshape は常に (B, L, H, D) 形。"""

    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        assert mha.batch_first and mha._qkv_same_embed_dim
        self.h = mha.num_heads
        self.e = mha.embed_dim
        self.d = self.e // self.h
        self.in_proj_weight = mha.in_proj_weight
        self.in_proj_bias = mha.in_proj_bias
        self.out_proj = mha.out_proj

    def forward(self, q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor) -> torch.Tensor:
        e = self.e
        w, b = self.in_proj_weight, self.in_proj_bias
        q = F.linear(q_in, w[:e], b[:e])
        k = F.linear(k_in, w[e:2 * e], b[e:2 * e])
        v = F.linear(v_in, w[2 * e:], b[2 * e:])
        bsz, lq, lk = _dims(q.shape[0], q.shape[1], k.shape[1])
        q = q.reshape(bsz, lq, self.h, self.d).transpose(1, 2)        # (B, H, Lq, D)
        k = k.reshape(bsz, lk, self.h, self.d).transpose(1, 2)        # (B, H, Lk, D)
        v = v.reshape(bsz, lk, self.h, self.d).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d)  # (B, H, Lq, Lk)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)                                    # (B, H, Lq, D)
        out = out.transpose(1, 2).reshape(bsz, lq, e)                  # (B, Lq, E)
        return self.out_proj(out)


class ExportDecoderLayer(nn.Module):
    """nn.TransformerDecoderLayer(norm_first=True, dropout=0, activation=relu)と同一計算。"""

    def __init__(self, layer: nn.TransformerDecoderLayer):
        super().__init__()
        assert layer.norm_first, "only norm_first=True is supported"
        self.self_attn = ExportMHA(layer.self_attn)
        self.cross_attn = ExportMHA(layer.multihead_attn)
        self.norm1, self.norm2, self.norm3 = layer.norm1, layer.norm2, layer.norm3
        self.linear1, self.linear2 = layer.linear1, layer.linear2
        act = layer.activation
        self.activation = act if callable(act) and not isinstance(act, str) else F.relu

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        x = x + self.self_attn(y, y, y)
        y = self.norm2(x)
        x = x + self.cross_attn(y, mem, mem)
        y = self.norm3(x)
        return x + self.linear2(self.activation(self.linear1(y)))


class ExportDecoder(nn.Module):
    def __init__(self, decoder: nn.TransformerDecoder):
        super().__init__()
        self.layers = nn.ModuleList([ExportDecoderLayer(layer) for layer in decoder.layers])
        self.norm = decoder.norm

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mem)
        return self.norm(x) if self.norm is not None else x


def to_export_model(model: nn.Module, static: bool = False) -> nn.Module:
    """TeacherModel の decoder を export 用実装に差し替えて返す(その場で置換、重みは共有)。

    static=True(--dynamic なし)では、本モジュールと vit_tiny / teacher の Reshape を Python int の
    定数形状で書き出す(先頭 = バッチ 1)。
    """
    global STATIC_SHAPES
    from . import teacher as _teacher
    from . import vit_tiny as _vit
    STATIC_SHAPES = static
    _vit.STATIC_SHAPES = static
    _teacher.STATIC_SHAPES = static
    # deepcopy はしない: 教師の patch_in は hub モデルへの forward hook(閉包が元の backbone を参照)で
    # 適用されるため、コピーでは hook が元のパラメータを掴んだままになり export が失敗する。
    # export 専用プロセスで呼ぶ前提で、モデルをその場で差し替える(冪等)
    m = model.eval()
    for prm in m.parameters():          # export に勾配は不要
        prm.requires_grad_(False)
    if not isinstance(m.decoder, ExportDecoder):
        m.decoder = ExportDecoder(m.decoder)
    return m
