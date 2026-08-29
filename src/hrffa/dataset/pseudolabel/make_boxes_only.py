"""DEIMv2-Wholebody49 ONNX から masks 出力を除去した boxes-only モデルを生成する。

D2 の擬似ラベル付与では bbox のみ必要なため、グラフ出力を label_xyxy_score に
絞ることで ORT のグラフ枝刈りを効かせる(マスクデコーダ分のメモリ・計算を削減)。

使い方:
    PYTHONPATH=src python3 -m hrffa.dataset.pseudolabel.make_boxes_only \
        --src data/models/deimv2_dinov3_x_wholebody49_ins_s08_maskhead256x3_center_1240query_masks_n_batch.onnx \
        --dst data/models/deimv2_wholebody49_boxes_only.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a boxes-only DEIMv2-Wholebody49 ONNX model (masks output removed).")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--keep-output", default="label_xyxy_score")
    args = ap.parse_args()

    model = onnx.load(str(args.src))
    keep = [o for o in model.graph.output if o.name == args.keep_output]
    if not keep:
        raise SystemExit(f"output '{args.keep_output}' not found in {args.src}")
    del model.graph.output[:]
    model.graph.output.extend(keep)
    onnx.save(model, str(args.dst))
    print(f"saved boxes-only model -> {args.dst}")


if __name__ == "__main__":
    main()
