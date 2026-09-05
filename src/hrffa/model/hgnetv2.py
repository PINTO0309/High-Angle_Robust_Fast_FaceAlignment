"""PP-HGNetV2(B0)の自前実装(history/049: ≤2.5M の軽量 CNN 学生)。

DEIMv2 が配布する ImageNet 事前学習重み `ckpts/PPHGNetV2_B0_stage1.pth`(state_dict のキー名に一致させ
strict にロードする)。構造は PaddleDetection の PPHGNetV2:
  stem: stem1(3×3 s2)→ [pad → stem2a(2×2) → pad → stem2b(2×2)] ∥ maxpool(2, s1) → concat → stem3(3×3 s2) → stem4(1×1)
  stage i: [downsample(dw 3×3 s2、act なし)] → HG_Block × n
  HG_Block: layers(3×3 conv または LightConv = 1×1(act なし)+ dw k×k)× 3 を入力と concat →
            aggregation(1×1 → out/2 → 1×1 → out)(+ residual)
  ConvBNAct = conv(bias なし)→ BN → ReLU → LAB(学習可能な scalar scale / bias、act があるときのみ)
出力 stride: stem 4 / stage0 4(64ch)/ stage1 8(256)/ stage2 16(512)/ stage3 32(1024)。

学生用ラッパー HGNetV2Backbone は stage2(stride 16)までを使い(0.75M)、feat_stride=8 のときは
stage1(stride 8)と FPN 風に融合した 32×32 のメモリを返す。cls は stride16 特徴の GAP。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class LAB(nn.Module):
    """Learnable Affine Block: y = scale * x + bias(scalar)。"""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.scale + self.bias


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k, stride=1, groups=1, use_act=True, use_lab=True, padding=None):
        super().__init__()
        if padding is None:
            padding = (k - 1) // 2
        self.conv = nn.Conv2d(cin, cout, k, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.use_act = use_act
        self.act = nn.ReLU(inplace=True) if use_act else nn.Identity()
        self.lab = LAB() if (use_act and use_lab) else None

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return self.lab(x) if self.lab is not None else x


class LightConvBNAct(nn.Module):
    def __init__(self, cin, cout, k, use_lab=True):
        super().__init__()
        self.conv1 = ConvBNAct(cin, cout, 1, use_act=False, use_lab=use_lab)
        self.conv2 = ConvBNAct(cout, cout, k, groups=cout, use_act=True, use_lab=use_lab)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class StemBlock(nn.Module):
    def __init__(self, cin, cmid, cout, use_lab=True):
        super().__init__()
        self.stem1 = ConvBNAct(cin, cmid, 3, 2, use_lab=use_lab)
        self.stem2a = ConvBNAct(cmid, cmid // 2, 2, 1, padding=0, use_lab=use_lab)
        self.stem2b = ConvBNAct(cmid // 2, cmid, 2, 1, padding=0, use_lab=use_lab)
        self.stem3 = ConvBNAct(cmid * 2, cmid, 3, 2, use_lab=use_lab)
        self.stem4 = ConvBNAct(cmid, cout, 1, 1, use_lab=use_lab)
        self.pool = nn.MaxPool2d(2, stride=1, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        return self.stem4(self.stem3(x))


class HGBlock(nn.Module):
    def __init__(self, cin, cmid, cout, layer_num=3, k=3, residual=False, light=False, use_lab=True):
        super().__init__()
        self.residual = residual
        layers = []
        for i in range(layer_num):
            c = cin if i == 0 else cmid
            layers.append(LightConvBNAct(c, cmid, k, use_lab) if light else ConvBNAct(c, cmid, k, use_lab=use_lab))
        self.layers = nn.ModuleList(layers)
        total = cin + layer_num * cmid
        self.aggregation = nn.Sequential(ConvBNAct(total, cout // 2, 1, use_lab=use_lab),
                                         ConvBNAct(cout // 2, cout, 1, use_lab=use_lab))

    def forward(self, x):
        identity = x if self.residual else None
        outs = [x]
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        x = torch.cat(outs, dim=1)
        # Concatenation owns the copied features. Release their Python references
        # before aggregation allocates its outputs (autograd keeps what it needs).
        del outs
        x = self.aggregation(x)
        return x + identity if self.residual else x


class HGStage(nn.Module):
    def __init__(self, cin, cmid, cout, blocks, layer_num, k, light, downsample, use_lab=True):
        super().__init__()
        self.downsample = (ConvBNAct(cin, cin, 3, 2, groups=cin, use_act=False, use_lab=use_lab)
                           if downsample else nn.Identity())
        self.blocks = nn.ModuleList(
            [HGBlock(cin if i == 0 else cout, cmid, cout, layer_num, k, residual=(i > 0), light=light, use_lab=use_lab)
             for i in range(blocks)])

    def forward(self, x):
        x = self.downsample(x)
        for b in self.blocks:
            x = b(x)
        return x


# variant -> (stem (in, mid, out), stages [(cin, cmid, cout, blocks, layer_num, k, light, downsample)])
HGNETV2_ARCH = {
    "hgnetv2_b0": ((3, 16, 16), [(16, 16, 64, 1, 3, 3, False, False),
                                 (64, 32, 256, 1, 3, 3, False, True),
                                 (256, 64, 512, 2, 3, 5, True, True),
                                 (512, 128, 1024, 1, 3, 5, True, True)]),
}
HGNETV2_CKPT = {"hgnetv2_b0": "PPHGNetV2_B0_stage1.pth"}


class HGNetV2(nn.Module):
    """stage 0..3 の特徴 (stride 4/8/16/32) を返す。state_dict のキーは配布重みと一致。"""

    def __init__(self, variant: str = "hgnetv2_b0", use_lab: bool = True):
        super().__init__()
        stem, stages = HGNETV2_ARCH[variant]
        self.stem = StemBlock(*stem, use_lab=use_lab)
        self.stages = nn.ModuleList([HGStage(*s, use_lab=use_lab) for s in stages])
        self.out_channels = [s[2] for s in stages]

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for st in self.stages:
            x = st(x)
            outs.append(x)
        return outs

    def load_pretrained(self, path: Path) -> None:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd.get("state_dict", sd))
        self.load_state_dict(sd, strict=True)


class HGNetV2Backbone(nn.Module):
    """学生用ラッパー: (patch (B, feat_ch, h, w), cls (B, feat_ch)) を返す。

    feat_stride=16: stage2(512ch)を 1×1 で feat_ch へ。feat_stride=8: stage1(256ch, 32×32)と
    stage2 を FPN 風に融合(1×1 → 加算 → 3×3)して 32×32 のメモリにする。stage3(1.1M)は使わない。
    """

    def __init__(self, variant: str, ckpt_path: Path | None, feat_stride: int = 8, feat_ch: int = 128,
                 input_norm: str = "imagenet", patch_instance_norm: bool = False):
        super().__init__()
        assert feat_stride in (8, 16)
        self.net = HGNetV2(variant)
        if ckpt_path is not None:
            self.net.load_pretrained(ckpt_path)
        if input_norm != "imagenet":
            self._fold_input_norm(input_norm)
        # vitt の patch_in(patch embed 直後の IN、017 F / 023 §4)に相当: stem(stride 4 の埋め込み段)の出力に
        # チャネル毎 InstanceNorm を 1 ノード挿入し、画像ごとのスタイル統計(照明・色被り)を除去する。
        # 事前学習キーとは独立の新設枝(state_dict キー: patch_in.*)
        stem_out = self.net.stem.stem4.conv.out_channels
        self.patch_in = nn.InstanceNorm2d(stem_out, affine=True) if patch_instance_norm else None
        self.net.stages = self.net.stages[:3]           # stage3(stride 32)は捨てる
        c8, c16 = self.net.out_channels[1], self.net.out_channels[2]
        self.feat_stride, self.embed_dim = feat_stride, feat_ch
        self.lat16 = nn.Sequential(nn.Conv2d(c16, feat_ch, 1, bias=False), nn.BatchNorm2d(feat_ch), nn.ReLU(inplace=True))
        if feat_stride == 8:
            self.lat8 = nn.Sequential(nn.Conv2d(c8, feat_ch, 1, bias=False), nn.BatchNorm2d(feat_ch), nn.ReLU(inplace=True))
            self.smooth = nn.Sequential(nn.Conv2d(feat_ch, feat_ch, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(feat_ch), nn.ReLU(inplace=True))

    @torch.no_grad()
    def _fold_input_norm(self, input_norm: str) -> None:
        """事前学習重み(ImageNet 正規化前提)を別の入力正規化へ折り込む(入口の演算を増やさない。049 追記)。

        z_imagenet = a·z_new + b(チャネル毎アフィン)なので stem1 conv(bias なし、BN 付き)は
          W * z_imagenet = (W·a) * z_new + Σ_{c,k} W[o,c,k]·b_c
        となる。第 1 項は conv 重みのスケール、第 2 項(出力チャネル毎の定数 c_o)は後続 BN の
        running_mean から差し引く(BN の学習時は batch 統計で定数が消えるため無調整で等価)。
        vitt(patch 16、padding なし)と異なり stem1 は padding 付き 3×3 のため、ゼロ padding された
        境界 1 画素の帯だけは厳密には等価でない(内部画素は厳密一致)。
        """
        from .backbone import IMAGE_MEAN, IMAGE_STD, norm_constants
        mean_new, std_new = norm_constants(input_norm)
        conv, bn = self.net.stem.stem1.conv, self.net.stem.stem1.bn
        w = conv.weight.clone()
        c = torch.zeros(w.shape[0], dtype=w.dtype, device=w.device)
        for ch in range(3):
            a = std_new[ch] / IMAGE_STD[ch]
            b = (mean_new[ch] - IMAGE_MEAN[ch]) / IMAGE_STD[ch]
            c += w[:, ch].sum(dim=(1, 2)) * b
            conv.weight[:, ch] *= a
        bn.running_mean -= c

    def forward(self, x):
        x = self.net.stem(x)
        if self.patch_in is not None:
            x = self.patch_in(x)
        f8 = None
        for i, st in enumerate(self.net.stages):
            x = st(x)
            if i == 1 and self.feat_stride == 8:
                f8 = x
        # Keep only the stride-8 feature needed by the FPN. Retaining every
        # stage output also kept unused activations alive throughout inference.
        p16 = self.lat16(x)
        cls = p16.mean(dim=(2, 3))
        if self.feat_stride == 16:
            return p16, cls
        # 動的 export: scale_factor(stride 16 → 8 は厳密に 2 倍、解像度に追従)。
        # 静的 export(STATIC_SHAPES): sizes を定数で与える(Resize の出力形状が推論可能になり、固定/N ペアの証明に必要)
        # scale_factor(Resize scales=[1,1,2,2])はバッチ非依存。sizes 指定はバッチ次元を定数 1 に固定するため使わない
        p8 = self.lat8(f8) + F.interpolate(p16, scale_factor=2.0, mode="nearest")
        return self.smooth(p8), cls
