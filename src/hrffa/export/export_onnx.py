"""教師モデルの ONNX エクスポート CLI。

- opset 17(既定)
- 入力解像度は --size で指定可能。既定は学習時解像度(preset の out_size)
- --dynamic 指定時は N x 3 x H x W のダイナミック軸で生成
- 出力後に onnxslim(1 回、Gemm 融合なし)→ onnxsim(1 回、定数畳み込みのみ)で最適化し、onnxruntime(グラフ最適化なし)で数値一致を検証する。静的 export は続けて N バッチ化(`<stem>_n.onnx`)する

入力は学習時と同じ前処理済みテンソル(RGB / 255 → preset の input_norm による正規化。
教師系 imagenet = (x − mean) / std、学生系 center05 = (x − 0.5) / 0.5)を想定する。
学生(vitt)は正規化を patch embed の conv に折り込んで学習するため ONNX 内に正規化演算はなく、
呼び出し側が preset どおりの正規化を行う。
scheme はエクスポート時に固定する(--scheme、既定 ibug68)。

使い方:
    uv run python -m hrffa.export.export_onnx \
        --ckpt runs/teacher/teacher_vitl_96gb_best_e0009_0.023980.pt \
        --preset teacher_vitl_96gb [--size 256] [--dynamic] [--skip-sim]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnxsim import simplify
from torch import nn

from ..model.teacher import TeacherModel
from ..train.config import get_config


class ExportWrapper(nn.Module):
    """scheme を固定し、デプロイに必要な出力のみ返すラッパー(points / vis_logits の 2 本)。

    rotation / roll_bit は 2026-08-28 に契約から外した: 教師 clean_v* と全学生は w_rot = w_roll = 0(012 §6.4)で
    姿勢ヘッドが未更新のため無意味な定数出力だった(047 §2.2)。姿勢が必要なら後付けヘッドか PnP。"""

    def __init__(self, model: TeacherModel, scheme: str):
        super().__init__()
        self.model = model
        self.scheme = scheme

    def forward(self, images: torch.Tensor):
        out = self.model(images, self.scheme)
        return out["points"], out["vis_logits"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a checkpoint to ONNX: fixed batch-1 graph (onnxslim -> onnxsim, canonicalized, parity-checked) and its N-batch variant.")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--preset", default="teacher_vitl_96gb")
    ap.add_argument("--scheme", default="ibug68",
                    choices=["ibug68", "wflw98", "cofw29"])
    ap.add_argument("--size", type=int, default=None,
                    help="input resolution (default: the preset's training resolution)")
    ap.add_argument("--dynamic", action="store_true",
                    help="export with dynamic axes (N x 3 x H x W)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--skip-sim", action="store_true",
                    help="skip both onnxslim and onnxsim (the raw graph becomes the artifact)")
    ap.add_argument("--no-n-batch", action="store_true",
                    help="do not derive the N-batch graph (<stem>_n.onnx) from the fixed batch-1 graph")
    args = ap.parse_args()

    cfg = get_config(args.preset)
    size = args.size or cfg.out_size

    model = TeacherModel.from_config(cfg)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    key = "ema" if "ema" in ck else "model"
    model.load_state_dict(ck[key])
    model.eval()
    # バッチ軸を潰さない Reshape で書き出すため、デコーダを export 用の等価実装に差し替える(export_modules)
    from ..model.export_modules import to_export_model
    model = to_export_model(model, static=not args.dynamic)
    wrapper = ExportWrapper(model, args.scheme).eval()

    if args.output is None:
        tag = "dynamic" if args.dynamic else f"{size}x{size}"
        args.output = Path("runs/export") / \
            f"{args.ckpt.stem}_{args.scheme}_{tag}.onnx"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, size, size)
    dynamic_axes = None
    # CNN 学生(hgnetv2_*、history/049)は TorchScript export でメモリ長が定数化されるため
    # 解像度は固定(バッチのみ可変)。ViT 系は H/W も可変
    from ..model.backbone import CNN_VARIANTS
    hw_dynamic = cfg.backbone not in CNN_VARIANTS
    if args.dynamic:
        img_axes = {0: "N", 2: "H", 3: "W"} if hw_dynamic else {0: "N"}
        dynamic_axes = {"images": img_axes,
                        "points": {0: "N"}, "vis_logits": {0: "N"}}
    torch.onnx.export(
        wrapper, (dummy,), str(args.output),
        opset_version=args.opset,
        input_names=["images"],
        output_names=["points", "vis_logits"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    # 最適化の流れ(ユーザー指定 2026-08-28): onnxslim を 1 回 → onnxsim を 1 回(--skip-sim で両方省略)。
    # onnxsim 0.7.x は DINOv3 の RoPE サブグラフの定数畳み込みで失敗(生 graph では RuntimeError、
    # onnxslim 出力では segfault)するため、onnxsim は子プロセスで実行し、通常 → folding なし → 断念の順に
    # フォールバックする。ORT で読めない graph も採用しない。成果物は sim > slim > raw の順で決め、--output に上書きする(中間ファイルは残さない)
    import subprocess

    raw_path = args.output
    slim_path = args.output.with_name(args.output.stem + "_slim.onnx")
    sim_path = args.output.with_name(args.output.stem + "_sim.onnx")
    candidates = [raw_path]
    # onnxslim も子プロセスで実行する: 同一プロセスで slim() を呼ぶと、その後の ORT セッション生成で
    # "Exception during initialization ... !utils::HasExternalDataInMemory(tensor_proto) was false" が
    # E ログに 4 回出る(セッションは成功する = 無害だが紛らわしい。onnxslim が ORT のグローバル状態を変える)
    if args.skip_sim:
        print("  onnxslim / onnxsim: skipped (--skip-sim)")
    else:
        # Gemm 融合(FusionGemm*)はバッチ×トークンを 2-D に融合した Reshape を作るため無効化する
        # (バッチ軸を常に先頭に保つ。~/.codex/skills/optimize-onnx-batches の INV-RESHAPE-001)
        code = ("import sys, onnx, onnxslim; m = onnx.load(sys.argv[1]); "
                "s = onnxslim.slim(m, skip_fusion_patterns=['FusionGemm', 'FusionGemmAdd', 'FusionGemmMul']); "
                "onnx.save(s, sys.argv[2]); print(len(m.graph.node), '->', len(s.graph.node))")
        r = subprocess.run([sys.executable, "-c", code, str(raw_path), str(slim_path)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  onnxslim: nodes {r.stdout.strip().splitlines()[-1]}")
            candidates.insert(0, slim_path)
        else:
            print(f"  onnxslim failed: rc={r.returncode} {r.stderr.strip().splitlines()[-1][:80] if r.stderr.strip() else ''}")

    def _run_onnxsim(src: Path, dst: Path, skip_folding: bool) -> bool:
        # perform_optimization=False: onnxsim の最適化パス(MatMul+Add → 2-D Gemm 化を含む)は使わず、
        # 定数畳み込み・形状推論・検証のみ行う(Gemm 化はバッチ×トークンを融合した Reshape を作り、
        # 「バッチ軸は常に先頭」の不変条件を壊す。skipped_optimizers では止められない)
        code = ("import sys, onnx; from onnxsim import simplify; m = onnx.load(sys.argv[1]); "
                f"s, ok = simplify(m, skip_constant_folding={skip_folding}, perform_optimization=False); "
                "assert ok; onnx.save(s, sys.argv[2]); print(len(m.graph.node), '->', len(s.graph.node))")
        r = subprocess.run([sys.executable, "-c", code, str(src), str(dst)],
                           capture_output=True, text=True)
        tag = "onnxsim(no constant folding)" if skip_folding else "onnxsim"
        if r.returncode == 0:
            print(f"  {tag}: nodes {r.stdout.strip().splitlines()[-1]}")
            return True
        print(f"  {tag} failed: rc={r.returncode} {r.stderr.strip().splitlines()[-1][:80] if r.stderr.strip() else ''}")
        return False

    src = candidates[0]
    if not args.skip_sim and (_run_onnxsim(src, sim_path, False) or _run_onnxsim(src, sim_path, True)):
        candidates.insert(0, sim_path)
    # 固定 graph の正準化(nbatch.canonicalize_fixed_graph): 共有 shape 定数の私有化と Reshape 定数の -1 の明示
    if not args.dynamic:
        from .nbatch import canonicalize_fixed_graph, eliminate_qkv_rank5
        for cand in candidates:
            m_c = onnx.load(str(cand))
            # qkv 分割の 5 次元テンソル(hub DINOv3 の reshape(B,N,3,H,D) 系)を 4 次元の Split/Reshape に書き換える
            # (045 §6 追記 4。数値同一。学生 vit_tiny は実装側で 4 次元化済みなので通常 0 件)
            n_qkv = eliminate_qkv_rank5(m_c)
            info_c = canonicalize_fixed_graph(m_c, input_name="images")
            onnx.save(m_c, str(cand))
            print(f"  canonicalize[{cand.name}]: rank-5 qkv blocks rewritten {n_qkv}, shape constants de-aliased "
                  f"{info_c['shape_dealiased']}, Reshape -1 made explicit {info_c['reshape_explicit']}")

    # 数値検証(torch と ORT の一致 + dynamic 時はバッチ/解像度可変の動作確認)。
    # parity は「graph そのもの」を検証するため ORT のグラフ最適化を無効にして行う(ORT の fusion は
    # 大型 ViT の静的 graph で 1e-3 級の誤差を出すことがある。ランタイム側の問題で graph は正しい)
    import onnxruntime as ort
    so_exact = ort.SessionOptions()
    so_exact.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = chosen = None
    for cand in candidates:
        try:
            sess = ort.InferenceSession(str(cand), so_exact, providers=["CPUExecutionProvider"])
            chosen = cand
            break
        except Exception as e:  # noqa: BLE001
            print(f"  {cand.name} rejected: onnxruntime cannot load it ({str(e)[:80]}...)")
            cand.unlink()
    assert sess is not None, "no graph loadable by onnxruntime"
    # 成果物は --output に 1 本だけ(最適化後の graph で生 graph を上書き、中間ファイルは削除)
    stage = {sim_path: "onnxslim+onnxsim", slim_path: "onnxslim only", raw_path: "raw graph"}[chosen]
    if chosen != raw_path:
        del sess
        chosen.replace(raw_path)
        sess = ort.InferenceSession(str(raw_path), so_exact, providers=["CPUExecutionProvider"])
    for leftover in (slim_path, sim_path):
        if leftover.exists():
            leftover.unlink()
    final_path = raw_path
    print(f"  artifact: {final_path} ({stage})")
    with torch.no_grad():
        ref = wrapper(dummy)
    got = sess.run(None, {"images": dummy.numpy()})
    # tolerance per output: points are normalized coordinates (5e-4 = 0.13 px at 256); vis_logits are unbounded
    # logits (argmax over 3 classes), so allow 5e-3 absolute (BN folding by onnxslim changes fp32 rounding)
    tol = {"points": 5e-4, "vis_logits": 5e-3}
    for name, r, g in zip(["points", "vis_logits"], ref, got):
        err = float(np.abs(r.numpy() - g).max())
        print(f"  parity {name}: max_err={err:.2e} (|ref| max {float(r.abs().max()):.2f}, tol {tol[name]:.0e})")
        assert err < tol[name], f"parity check failed for {name}: {err:.2e} >= {tol[name]:.0e}"
    if args.dynamic:
        checks = [(2, size), (1, size + 32)] if hw_dynamic else [(2, size)]
        if not hw_dynamic:
            print(f"  (CNN backbone: resolution fixed at {size}, only the batch axis is dynamic)")
        for n, s in checks:
            x = np.random.randn(n, 3, s, s).astype(np.float32)
            try:
                out = sess.run(None, {"images": x})
                print(f"  dynamic check N={n} HxW={s}: ok points={out[0].shape}")
            except Exception as e:  # noqa: BLE001
                print(f"  dynamic check N={n} HxW={s}: FAILED ({str(e)[:120]})")

    mb = final_path.stat().st_size / 1e6
    print(f"exported: {final_path} ({stage}, {mb:.1f} MB)")

    # 固定バッチ 1 の graph から、バッチ軸だけ記号 N の graph を派生させる(nbatch.py、PersonViT 方式)
    if not args.dynamic and not args.no_n_batch:
        from .nbatch import convert_fixed_batch_to_n
        n_path = final_path.with_name(final_path.stem + "_n.onnx")
        info = convert_fixed_batch_to_n(final_path, n_path, input_name="images")
        print(f"n-batch: {n_path} (Reshape leading 1->-1: {info['reshape_rewritten']}, Concat broadcasts restored: "
              f"{info['concat_broadcast']}, symbolic dims set to N: {info['symbols_canonicalized']}, "
              f"batch 1/2/3 check max_err={info['max_err']:.2e})")


if __name__ == "__main__":
    main()
