"""DINOv3 backbone ローダ。

公式実装(torch.hub キャッシュ)を実行時 import して使う。コードはリポジトリに
取り込まない(DINOv3 License のコード/重みを配布物に含めない方針: history/008)。
hubconf.py は壊れた依存(torchmetrics→transformers)を巻き込むため、
`dinov3.hub.backbones` モジュールを直接 import する。

初回のみネットワークが必要(git clone)。以後は ~/.cache/torch/hub を使用。
重みはローカル ckpts/ のファイルを指定する(再配布禁止のため)。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

_DINOV3_GIT = "https://github.com/facebookresearch/dinov3"

# variant -> (hub 関数名, 既定 ckpt ファイル名, 埋め込み次元)
DINOV3_VARIANTS = {
    "vits16": ("dinov3_vits16", "dinov3_vits16_pretrain_lvd1689m-08c60483.pth", 384),
    "vits16plus": ("dinov3_vits16plus", "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth", 384),
    "vitb16": ("dinov3_vitb16", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", 768),
    "vitl16": ("dinov3_vitl16", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", 1024),
    "vith16plus": ("dinov3_vith16plus", "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth", 1280),
}

# 学生 S 用の軽量変種(vit_tiny.py の自前実装でロード。history/026)
# variant -> (ckpt ファイル名, 埋め込み次元, 深さ, ヘッド数)
TINY_VARIANTS = {
    "vitt": ("vitt_distill.pt", 192, 12, 3),
}

# 軽量 CNN 学生(history/049)。重みは ckpts/ の PP-HGNetV2 stage1(DEIMv2 配布、ImageNet 事前学習)
CNN_VARIANTS = ("hgnetv2_b0",)

PATCH_SIZE = 16
# DINOv3 標準の入力正規化(ImageNet)。教師系はこちら
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

# 入力正規化仕様(history/026: 学生系は center05 = ((x/255)-0.5)/0.5 に統一。
# ImageNet 正規化とはチャネル毎の固定アフィン差であり、vitt 事前学習重みへは
# patch embed conv への厳密折り込みで無劣化に変換できる)
INPUT_NORMS = {
    "imagenet": (IMAGE_MEAN, IMAGE_STD),
    "center05": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}


def norm_constants(input_norm: str) -> tuple[tuple, tuple]:
    return INPUT_NORMS[input_norm]


def _ensure_hub_code() -> None:
    """DINOv3 公式実装を torch.hub キャッシュ位置に用意して import path に載せる。

    torch.hub の私的 API はバージョン間で壊れる(2.11 で実際に破損)ため使わず、
    初回のみ素の git clone を行う。TORCH_HOME 環境変数を尊重する。
    """
    target = Path(torch.hub.get_dir()) / "facebookresearch_dinov3_main"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", _DINOV3_GIT, str(target)],
            check=True)
    if str(target) not in sys.path:
        sys.path.insert(0, str(target))


class Dinov3Backbone(nn.Module):
    """パッチ特徴 (B, C, H/16, W/16) と CLS トークン (B, C) を返すラッパー。

    patch_instance_norm=True で patch embedding 直後に token 軸 InstanceNorm を
    1 ノード挿入する(PersonViT の知見: 画像ごとのスタイル統計を除去し
    照明・色被り・カメラトーンへの耐性を上げる。history/017 アーム F)。
    既存重みのキーを変えないため forward hook で適用し、IN パラメータは
    backbone.patch_in.* に置く。
    """

    def __init__(self, variant: str, ckpt_dir: Path = Path("ckpts"),
                 patch_instance_norm: bool = False, input_norm: str = "imagenet",
                 local_conv: bool = False, feat_layers: tuple[int, ...] = (),
                 feat_stride: int = 16, cnn_feat_ch: int = 128):
        super().__init__()
        if variant in CNN_VARIANTS:
            # 軽量 CNN 学生(history/049)。ImageNet 事前学習(DEIMv2 配布)を strict ロード
            from .hgnetv2 import HGNETV2_CKPT, HGNetV2Backbone  # noqa: PLC0415
            assert not (local_conv or feat_layers), "local_conv / feat_layers are not supported with CNN backbones"
            self._tiny = True                      # forward は inner(x) をそのまま返す
            self.inner = HGNetV2Backbone(variant, ckpt_dir / HGNETV2_CKPT[variant],
                                         feat_stride=feat_stride, feat_ch=cnn_feat_ch,
                                         input_norm=input_norm,   # center05 は stem1 conv + BN へ折り込み(入口の演算なし)
                                         patch_instance_norm=patch_instance_norm)   # stem 出力に IN(vitt の patch_in 相当)
            self.embed_dim = self.inner.embed_dim
            return
        assert feat_stride == 16, "feat_stride=8 is only for CNN backbones"
        if variant in TINY_VARIANTS:
            from .vit_tiny import VitTiny  # noqa: PLC0415
            ckpt_name, dim, depth, heads = TINY_VARIANTS[variant]
            self._tiny = True
            self.inner = VitTiny(dim, depth, heads,
                                 patch_instance_norm=patch_instance_norm,
                                 local_conv=local_conv, feat_layers=feat_layers)
            self.inner.load_flat_ckpt(ckpt_dir / ckpt_name, input_norm=input_norm)
            self.embed_dim = self.inner.embed_dim_out   # 多層融合時は dim × 層数
            return
        assert not local_conv and not feat_layers, \
            "local_conv / feat_layers are only for vitt (student); hub backbones are not supported"
        assert input_norm == "imagenet", \
            "hub backbones (teacher) support ImageNet normalization only"
        self._tiny = False
        hub_fn, ckpt_name, dim = DINOV3_VARIANTS[variant]
        self.embed_dim = dim
        _ensure_hub_code()
        from dinov3.hub import backbones  # noqa: PLC0415
        self.inner = getattr(backbones, hub_fn)(
            pretrained=True, weights=str(ckpt_dir / ckpt_name))
        if patch_instance_norm:
            self.patch_in = nn.InstanceNorm2d(dim, affine=True)

            def _hook(_mod, _inp, out):
                # patch_embed 出力 (B, H, W, C) → チャネルごとに空間軸で IN
                z = out.permute(0, 3, 1, 2)
                z = self.patch_in(z)
                return z.permute(0, 2, 3, 1)

            self.inner.patch_embed.register_forward_hook(_hook)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._tiny:
            return self.inner(x)
        # get_intermediate_layers は最終層の (patch, cls) を返せる
        feats = self.inner.get_intermediate_layers(
            x, n=1, reshape=True, return_class_token=True)
        patch, cls = feats[0]
        return patch, cls
