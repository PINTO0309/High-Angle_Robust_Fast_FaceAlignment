"""ONNX graph(DEIMv2-Wholebody49 検出器など)を onnxruntime-web(WebGPU / WASM)向けに変換・最適化する。

処理の順序(すべて意味を変えない書き換え。出力は元 graph と onnxruntime で突き合わせて検証する):
  1. fix_batch_to_one       N バッチ export の先頭次元を 1 に固定(web デモはバッチ 1。正準化は固定 graph が前提)
  2. convert_double_to_float `Cast(FLOAT → DOUBLE)` と DOUBLE 定数を float32 に(onnxruntime-web の WASM カーネルは double を
                            含まない縮小ビルドで、セッション作成が "Could not find an implementation for Cast(13)" で失敗する)
  3. simplify_slim_sim       ONNX 最適化スキル(~/.claude/skills/onnx-export-optimize)の流れ: onnxslim(Gemm 融合なし)→
                            onnxsim(最適化パスなし = 定数畳み込みのみ)。子プロセス実行、ORT で読めない graph は不採用
  4. normalize_negative_axes 負の axis / axes を正の値に(axes 入力の定数は共有されうるのでノードごとに複製)
  5. fix_matmul_1d           1 次元オペランドの MatMul(DFL 積分 [Q, 33] × [33])を Unsqueeze → MatMul → Squeeze に
                            (WebGPU EP が "Invalid dimension of 4294967295 for SizeToDimension" で落ちる)
  6. rewrite_isinf_isnan     IsNaN → Not(Equal(x, x))、IsInf → Greater(Abs(x), FLT_MAX)(WebGPU EP に無く CPU に落ち、
                            デコーダ途中で GPU↔CPU 同期を起こす)
  7. prune_unreachable       出力から到達できないノード・initializer を削除(boxes-only export に残るマスクヘッド)
  8. canonicalize_with_skill eliminate_qkv_rank5(ViT の qkv 分割の 5 次元テンソル排除)+ canonicalize_fixed_graph
                            (shape 定数の私有化、Reshape の -1 明示)= hrffa.export.nbatch
  9. 保存(一時ファイル → checker → 置換)と検証: 元 graph と同じ入力(--check-image または乱数)を graph 最適化 OFF の
     onnxruntime CPU で流し、検出行を対応付けて bbox / score の最大差を報告する(丸め差で同点行が入れ替わるだけ)。

残る監査違反(上流 export の Gemm 融合 = rank-2 Reshape、(B·H, L, D) レイアウト)は ONNX だけでは戻せず、バッチ 1 では無害。
後処理の int64 添字演算(TopK index の Div / Mul / Sub など 9 ノード)は WGSL に i64 が無いため CPU に残る。

出力名は既定で短縮し `_webgpu` を付ける(例: deimv2_dinov3_s_wholebody49_ins_s08_maskhead256x3_center_1240query_masks.onnx
→ deimv2_dinov3_s_wholebody49_webgpu.onnx。旧名 deimv2_wholebody49_boxes_only.onnx は X 由来なので
deimv2_dinov3_x_wholebody49_boxes_only_webgpu.onnx)。`--keep-name` で元の名前のまま出力する。

使い方:
  uv run python scripts/onnx_web_compat.py data/models/deimv2_*.onnx --out-dir demo/web/models
  uv run python scripts/onnx_web_compat.py model.onnx --out-dir out --check-image image.jpg
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper, shape_inference


def convert_double_to_float(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, int]]:
    """double を使う箇所を float32 に置き換えた新しい ModelProto と変更数を返す。"""
    g = model.graph
    stats = {"cast": 0, "constant": 0, "initializer": 0, "value_info": 0}
    for n in g.node:
        if n.op_type == "Cast":
            for a in n.attribute:
                if a.name == "to" and a.i == TensorProto.DOUBLE:
                    a.i = TensorProto.FLOAT
                    stats["cast"] += 1
        elif n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value" and a.t.data_type == TensorProto.DOUBLE:
                    arr = numpy_helper.to_array(a.t).astype(np.float32)
                    a.t.CopyFrom(numpy_helper.from_array(arr, a.t.name))
                    stats["constant"] += 1
    for init in g.initializer:
        if init.data_type == TensorProto.DOUBLE:
            arr = numpy_helper.to_array(init).astype(np.float32)
            init.CopyFrom(numpy_helper.from_array(arr, init.name))
            stats["initializer"] += 1
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        if vi.type.tensor_type.elem_type == TensorProto.DOUBLE:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT
            stats["value_info"] += 1
    # 型が変わったので推論済みの value_info は作り直す
    del g.value_info[:]
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    remaining = [n.name for n in model.graph.node if n.op_type == "Cast"
                 and any(a.name == "to" and a.i == TensorProto.DOUBLE for a in n.attribute)]
    assert not remaining, f"double Cast nodes remain: {remaining}"
    return model, stats


# 負の axis を持ちうる属性と、その axis が参照する入力(rank の基準)。Unsqueeze は出力 rank 基準なので対象外
_AXIS_OPS = {"Concat", "Split", "Softmax", "LogSoftmax", "LayerNormalization", "Flatten", "Gather", "GatherElements",
             "ScatterElements", "TopK", "ReduceMax", "ReduceMin", "ReduceMean", "ReduceSum", "ReduceProd", "ReduceL2",
             "ArgMax", "ArgMin", "CumSum", "Squeeze", "Hardmax"}
_AXES_INPUT_OPS = {"ReduceSum", "Squeeze"}  # opset 13+: axes は第 2 入力(定数の場合のみ正規化)


def normalize_negative_axes(model: onnx.ModelProto) -> dict[str, int]:
    """負の axis / axes を、形状推論で rank が分かる入力に対して正の値へ正規化する(意味は不変)。

    onnxruntime-web(JSEP)の一部カーネルは負の axis を正規化せずに扱うため、実行時に
    "Invalid dimension of 4294967295 for SizeToDimension" で失敗することがある。
    """
    inferred = shape_inference.infer_shapes(model)
    rank: dict[str, int] = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        if vi.type.tensor_type.HasField("shape"):
            rank[vi.name] = len(vi.type.tensor_type.shape.dim)
    consts: dict[str, onnx.TensorProto] = {init.name: init for init in model.graph.initializer}
    for n in model.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    consts[n.output[0]] = a.t
        rank.update({init.name: len(init.dims) for init in model.graph.initializer})
    stats = {"attr": 0, "axes_input": 0, "unknown_rank": 0}
    for n in model.graph.node:
        if n.op_type not in _AXIS_OPS or not n.input:
            continue
        r = rank.get(n.input[0])
        for a in n.attribute:
            if a.name == "axis" and a.i < 0:
                if r is None:
                    stats["unknown_rank"] += 1
                else:
                    a.i += r
                    stats["attr"] += 1
            elif a.name == "axes" and any(v < 0 for v in a.ints):
                if r is None:
                    stats["unknown_rank"] += 1
                else:
                    vals = [v + r if v < 0 else v for v in a.ints]
                    del a.ints[:]
                    a.ints.extend(vals)
                    stats["attr"] += 1
        if n.op_type in _AXES_INPUT_OPS and len(n.input) > 1 and n.input[1] in consts:
            arr = numpy_helper.to_array(consts[n.input[1]])
            if arr.size and (arr < 0).any():
                if r is None:
                    stats["unknown_rank"] += 1
                else:
                    # axes 定数は他ノード(Unsqueeze など、出力 rank 基準)と共有されうるので、このノード専用の
                    # initializer を新設して正規化する(共有元は不変)
                    fixed = np.where(arr < 0, arr + r, arr).astype(arr.dtype)
                    name = f"{n.input[1]}__axes_pos__{stats['axes_input']}"
                    model.graph.initializer.append(numpy_helper.from_array(fixed, name))
                    n.input[1] = name
                    stats["axes_input"] += 1
    return stats


def _tensor_ranks(model: onnx.ModelProto) -> dict[str, int]:
    """形状推論 + initializer / Constant / Identity 追跡で分かる範囲のテンソル rank。"""
    inferred = shape_inference.infer_shapes(model)
    rank: dict[str, int] = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        if vi.type.tensor_type.HasField("shape"):
            rank[vi.name] = len(vi.type.tensor_type.shape.dim)
    for init in model.graph.initializer:
        rank[init.name] = len(init.dims)
    for n in model.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    rank[n.output[0]] = len(a.t.dims)
    for _ in range(3):  # Identity の連鎖
        for n in model.graph.node:
            if n.op_type == "Identity" and n.input[0] in rank and n.output[0] not in rank:
                rank[n.output[0]] = rank[n.input[0]]
    return rank


def fix_matmul_1d(model: onnx.ModelProto) -> int:
    """1 次元オペランドを持つ MatMul を 2 次元 MatMul + Squeeze に書き換える(意味は不変)。

    onnxruntime-web の WebGPU(JSEP)MatMul は 1 次元オペランド(例: DFL 積分の [4960, 33] × [33])を
    "Invalid dimension of 4294967295 for SizeToDimension" で落とす。A [.., K] × b [K] は
    A × unsqueeze(b, 1) → [.., 1] → squeeze(-1)、a [K] × B [K, N] は unsqueeze(a, 0) × B → [1, N] → squeeze(0)。
    """
    rank = _tensor_ranks(model)
    g = model.graph
    new_nodes = []
    fixed = 0
    for n in g.node:
        if n.op_type == "MatMul" and len(n.input) == 2:
            ra, rb = rank.get(n.input[0]), rank.get(n.input[1])
            if (ra == 1) != (rb == 1) and ra is not None and rb is not None and max(ra, rb) >= 2:
                side = 1 if rb == 1 else 0                       # 1 次元側
                unsq_axis = 1 if side == 1 else 0                # b → [K, 1] / a → [1, K]
                base = n.name or n.output[0]
                axes_u = numpy_helper.from_array(np.array([unsq_axis], dtype=np.int64), f"{base}/web_unsq_axes")
                axes_s = numpy_helper.from_array(np.array([-1 if side == 1 else 0], dtype=np.int64), f"{base}/web_sq_axes")
                g.initializer.extend([axes_u, axes_s])
                unsq_out = f"{n.input[side]}/web_2d/{fixed}"
                mm_out = f"{n.output[0]}/web_2d"
                new_nodes.append(onnx.helper.make_node("Unsqueeze", [n.input[side], axes_u.name], [unsq_out], name=f"{base}/web_unsqueeze"))
                new_nodes.append(onnx.helper.make_node("MatMul", [n.input[0] if side == 1 else unsq_out, unsq_out if side == 1 else n.input[1]],
                                                       [mm_out], name=f"{base}/web_matmul"))
                new_nodes.append(onnx.helper.make_node("Squeeze", [mm_out, axes_s.name], [n.output[0]], name=f"{base}/web_squeeze"))
                fixed += 1
                continue
        new_nodes.append(n)
    if fixed:
        del g.node[:]
        g.node.extend(new_nodes)
    return fixed


_LONG_SUFFIX = re.compile(r"_ins_s\d+_maskhead\d+x\d+_center_\d+query_masks")
# 旧名の boxes-only(make_boxes_only.py で dinov3_x の N バッチ export から作ったもの。擬似ラベル系が
# この名前を参照するので data/models 側は改名しない)には、配布名でバックボーンを明示する
_LEGACY_NAMES = {"deimv2_wholebody49_boxes_only": "deimv2_dinov3_x_wholebody49_boxes_only"}


def web_name(src: Path, suffix: str = "_webgpu") -> str:
    """配布名を短縮して用途サフィックスを付ける: 長い export 修飾子を落とし、末尾に `_webgpu` を足す。"""
    stem = _LONG_SUFFIX.sub("", src.stem)
    stem = _LEGACY_NAMES.get(stem, stem)
    return f"{stem}{suffix}{src.suffix}"


def prune_unreachable(model: onnx.ModelProto) -> dict[str, int]:
    """graph 出力から到達できないノード・initializer を削除する(boxes-only export はマスクヘッドの重みを
    ファイルに残したままなので、配布サイズと読み込み時間を減らす。意味は不変)。"""
    g = model.graph
    producer = {o: n for n in g.node for o in n.output if o}
    needed: set[str] = set(o.name for o in g.output)
    stack = list(needed)
    keep_nodes: set[int] = set()
    while stack:
        t = stack.pop()
        n = producer.get(t)
        if n is None or id(n) in keep_nodes:
            continue
        keep_nodes.add(id(n))
        for i in n.input:
            if i and i not in needed:
                needed.add(i)
                stack.append(i)
        for a in n.attribute:  # サブグラフ(If / Loop)が外側のテンソルを参照する場合
            for sub in ([a.g] if a.type == onnx.AttributeProto.GRAPH else list(a.graphs)):
                for sn in sub.node:
                    for i in sn.input:
                        if i and i not in needed:
                            needed.add(i)
                            stack.append(i)
    nodes_before, inits_before = len(g.node), len(g.initializer)
    kept_nodes = [n for n in g.node if id(n) in keep_nodes]
    del g.node[:]
    g.node.extend(kept_nodes)
    kept_inits = [i for i in g.initializer if i.name in needed]
    del g.initializer[:]
    g.initializer.extend(kept_inits)
    kept_inputs = [i for i in g.input if i.name in needed or i.name not in {x.name for x in kept_inits}]
    del g.input[:]
    g.input.extend(kept_inputs)
    return {"nodes": nodes_before - len(g.node), "initializers": inits_before - len(g.initializer)}


# ---------------------------------------------------------------------------
# ONNX 最適化スキル(~/.claude/skills/onnx-export-optimize)の流れ:
#   onnxslim(Gemm 融合なし)→ onnxsim(最適化パスなし = 定数畳み込みのみ)→ qkv の 5 次元排除 → 固定 graph の正準化
#   (shape 定数の私有化、Reshape の -1 明示)→ ORT parity(graph 最適化 OFF)。onnxslim / onnxsim は子プロセスで実行し、
#   onnxsim は 通常 → skip_constant_folding → 断念 の順にフォールバック、ORT で読めない graph は採用しない(採用順 sim > slim > raw)。
# ---------------------------------------------------------------------------
_SLIM = ("import sys, onnx, onnxslim; m = onnx.load(sys.argv[1]); "
         "s = onnxslim.slim(m, skip_fusion_patterns=['FusionGemm', 'FusionGemmAdd', 'FusionGemmMul']); "
         "onnx.save(s, sys.argv[2]); print(len(m.graph.node), '->', len(s.graph.node))")
_SIM = ("import sys, onnx; from onnxsim import simplify; m = onnx.load(sys.argv[1]); "
        "s, ok = simplify(m, skip_constant_folding={skip_folding}, perform_optimization=False); "
        "assert ok; onnx.save(s, sys.argv[2]); print(len(m.graph.node), '->', len(s.graph.node))")


def _run_subprocess(code: str, src: Path, dst: Path, tag: str) -> bool:
    import subprocess
    import sys as _sys
    r = subprocess.run([_sys.executable, "-c", code, str(src), str(dst)], capture_output=True, text=True)
    if r.returncode == 0:
        print(f"  {tag}: nodes {r.stdout.strip().splitlines()[-1]}")
        return True
    err = r.stderr.strip().splitlines()[-1][:120] if r.stderr.strip() else ""
    print(f"  {tag} failed: rc={r.returncode} {err}")
    return False


def _loads_in_ort(path: Path) -> bool:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  {path.name} does not load in ORT: {str(e)[:100]}")
        return False


def fix_batch_to_one(model: onnx.ModelProto) -> int:
    """入力・出力の先頭(バッチ)次元が記号 / 0 の N バッチ export を固定バッチ 1 にする。

    web デモの検出器はバッチ 1 でしか走らせず、スキルの正準化(Reshape の -1 明示 = ORT 参照実行)は固定 graph が前提。
    バッチを固めると onnxsim の定数畳み込みで Shape 由来のノードも静的になる。変更した次元の数を返す。
    """
    changed = 0
    for vi in list(model.graph.input) + list(model.graph.output):
        shape = vi.type.tensor_type.shape
        if len(shape.dim) >= 1 and shape.dim[0].dim_value <= 0:
            shape.dim[0].ClearField("dim_param")
            shape.dim[0].dim_value = 1
            changed += 1
    if changed:
        del model.graph.value_info[:]
    return changed


def simplify_slim_sim(model: onnx.ModelProto, work_dir: Path, stem: str) -> tuple[onnx.ModelProto, str]:
    """onnxslim(Gemm 融合なし)→ onnxsim(最適化なし)。採用した graph と採用元のタグを返す。"""
    raw = work_dir / f"{stem}.raw.onnx"
    slim = work_dir / f"{stem}.slim.onnx"
    sim = work_dir / f"{stem}.sim.onnx"
    onnx.save(model, str(raw))
    candidates: list[tuple[Path, str]] = [(raw, "raw")]
    if _run_subprocess(_SLIM, raw, slim, "onnxslim"):
        candidates.insert(0, (slim, "onnxslim"))
    src = candidates[0][0]
    if _run_subprocess(_SIM.format(skip_folding=False), src, sim, "onnxsim") or \
            _run_subprocess(_SIM.format(skip_folding=True), src, sim, "onnxsim(no constant folding)"):
        candidates.insert(0, (sim, "onnxslim+onnxsim"))
    chosen = next(((p, t) for p, t in candidates if _loads_in_ort(p)), (raw, "raw"))
    out = onnx.load(str(chosen[0]))
    for p in (raw, slim, sim):
        p.unlink(missing_ok=True)
    return out, chosen[1]


def canonicalize_with_skill(model: onnx.ModelProto, input_name: str) -> dict:
    """qkv の 5 次元排除と固定 graph の正準化(リポジトリの hrffa.export.nbatch = スキルの fixed_to_nbatch と同じ実装)。"""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hrffa.export.nbatch import canonicalize_fixed_graph, eliminate_qkv_rank5
    n_qkv = eliminate_qkv_rank5(model)
    info = canonicalize_fixed_graph(model, input_name)
    return {"qkv_rank5": n_qkv, **info}


def graph_stats(model: onnx.ModelProto) -> str:
    from collections import Counter
    ops = Counter(n.op_type for n in model.graph.node)
    consts = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    for n in model.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    consts[n.output[0]] = numpy_helper.to_array(a.t)
    rank2 = neg1 = 0
    for n in model.graph.node:
        if n.op_type == "Reshape" and len(n.input) > 1 and n.input[1] in consts:
            t = consts[n.input[1]]
            if t.ndim == 1:
                rank2 += int(t.size == 2)
                neg1 += int(bool((t[1:] == -1).any())) if t.size > 1 else 0
    rank5 = sum(1 for vi in model.graph.value_info if len(vi.type.tensor_type.shape.dim) == 5)
    return (f"nodes {len(model.graph.node)}, Gemm {ops.get('Gemm', 0)}, MatMul {ops.get('MatMul', 0)}, "
            f"Reshape {ops.get('Reshape', 0)} (rank-2 {rank2}, non-leading -1 {neg1}), rank-5 value_info {rank5}")


def rewrite_isinf_isnan(model: onnx.ModelProto) -> dict[str, int]:
    """IsNaN / IsInf を WebGPU EP にカーネルがある op で等価に書き換える(意味は不変)。

    onnxruntime-web 1.27 のネイティブ WebGPU EP は IsNaN / IsInf を CPU に割り当てるため、デコーダ各層の
    NaN/Inf 無害化(Where(IsNaN|IsInf, …))のたびに GPU → CPU → GPU の同期が入る。
      IsNaN(x) = Not(Equal(x, x))、IsInf(x) = Greater(Abs(x), FLT_MAX)(片側指定なら Greater(x, FLT_MAX) / Less(x, -FLT_MAX))。
    """
    g = model.graph
    stats = {"isnan": 0, "isinf": 0}
    new_nodes = []
    fmax = np.array(np.finfo(np.float32).max, dtype=np.float32)
    for n in g.node:
        base = n.name or n.output[0]
        if n.op_type == "IsNaN":
            eq = f"{base}/web_eq"
            new_nodes.append(onnx.helper.make_node("Equal", [n.input[0], n.input[0]], [eq], name=f"{base}/web_equal"))
            new_nodes.append(onnx.helper.make_node("Not", [eq], [n.output[0]], name=f"{base}/web_not"))
            stats["isnan"] += 1
            continue
        if n.op_type == "IsInf":
            attrs = {a.name: a.i for a in n.attribute}
            pos, neg = attrs.get("detect_positive", 1), attrs.get("detect_negative", 1)
            if pos and neg:
                g.initializer.append(numpy_helper.from_array(fmax, f"{base}/web_fmax"))
                absn = f"{base}/web_abs"
                new_nodes.append(onnx.helper.make_node("Abs", [n.input[0]], [absn], name=f"{base}/web_abs_node"))
                new_nodes.append(onnx.helper.make_node("Greater", [absn, f"{base}/web_fmax"], [n.output[0]], name=f"{base}/web_greater"))
            elif pos:
                g.initializer.append(numpy_helper.from_array(fmax, f"{base}/web_fmax"))
                new_nodes.append(onnx.helper.make_node("Greater", [n.input[0], f"{base}/web_fmax"], [n.output[0]], name=f"{base}/web_greater"))
            elif neg:
                g.initializer.append(numpy_helper.from_array(-fmax, f"{base}/web_fmin"))
                new_nodes.append(onnx.helper.make_node("Less", [n.input[0], f"{base}/web_fmin"], [n.output[0]], name=f"{base}/web_less"))
            else:  # 何も検出しない指定: 常に false
                g.initializer.append(numpy_helper.from_array(fmax, f"{base}/web_fmax"))
                absn = f"{base}/web_abs"
                new_nodes.append(onnx.helper.make_node("Abs", [n.input[0]], [absn], name=f"{base}/web_abs_node"))
                new_nodes.append(onnx.helper.make_node("Greater", [f"{base}/web_fmax", absn], [f"{base}/web_gt"], name=f"{base}/web_greater"))
                new_nodes.append(onnx.helper.make_node("Not", [f"{base}/web_gt"], [n.output[0]], name=f"{base}/web_not"))
                # Abs(x) > FLT_MAX でも "Not(FLT_MAX > Abs)" が true になり得るが、両側とも無効な IsInf は実用上現れない
            stats["isinf"] += 1
            continue
        new_nodes.append(n)
    if sum(stats.values()):
        del g.node[:]
        g.node.extend(new_nodes)
    return stats


def _detector_input(model: onnx.ModelProto, image_path: Path | None) -> np.ndarray:
    inp = model.graph.input[0]
    dims = [d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim]
    if image_path is None:
        return np.random.default_rng(0).random(dims, dtype=np.float32)
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    resized = cv2.resize(img, (dims[3], dims[2]), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.transpose(2, 0, 1).astype(np.float32)[None] / 255.0


def verify(original: Path, converted: Path, image_path: Path | None) -> str:
    """元 graph と変換後を onnxruntime CPU で実行し、出力ごとの最大差を文字列で返す。"""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    # スキルの流儀: parity は graph 最適化 OFF で測る(graph そのものの検証)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_a = ort.InferenceSession(str(original), so, providers=["CPUExecutionProvider"])
    sess_b = ort.InferenceSession(str(converted), so, providers=["CPUExecutionProvider"])
    x = _detector_input(onnx.load(str(original), load_external_data=False), image_path)
    feed = {sess_a.get_inputs()[0].name: x}
    outs_a = sess_a.run(None, feed)
    outs_b = sess_b.run(None, feed)
    parts = []
    for meta, a, b in zip(sess_a.get_outputs(), outs_a, outs_b):
        a64, b64 = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if meta.name == "label_xyxy_score":
            # 丸め差でスコア同点の行が入れ替わることがあるので、行を対応付けて比較する
            # (score >= 0.3 の各検出について、同クラスで bbox が最も近い行との差)
            ra, rb = a64[0], b64[0]
            keep = ra[:, 5] >= 0.3
            worst_box, worst_score = 0.0, 0.0
            for row in ra[keep]:
                same = rb[rb[:, 0] == row[0]]
                if len(same) == 0:
                    worst_box = float("inf")
                    continue
                j = np.abs(same[:, 1:5] - row[1:5]).sum(axis=1).argmin()
                worst_box = max(worst_box, float(np.abs(same[j, 1:5] - row[1:5]).max()))
                worst_score = max(worst_score, float(abs(same[j, 5] - row[5])))
            heads_a = int(((ra[:, 0] == 7) & (ra[:, 5] >= 0.5)).sum())
            heads_b = int(((rb[:, 0] == 7) & (rb[:, 5] >= 0.5)).sum())
            permuted = int((np.abs(ra - rb).max(axis=1) > 1e-6).sum())
            parts.append(f"{meta.name}: matched {int(keep.sum())} detections (score>=0.3) max box diff={worst_box:.3e} "
                         f"max score diff={worst_score:.3e}, rows reordered={permuted}, heads>=0.5 {heads_a}/{heads_b}")
        else:
            parts.append(f"{meta.name} max_abs_diff={np.abs(a64 - b64).max():.3e}")
    return ", ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert ONNX graphs for onnxruntime-web (WebGPU / WASM): double -> float32, onnxslim / onnxsim, negative axes, 1-D MatMul, IsNaN / IsInf, dead-code pruning, rank-5 qkv rewrite and canonicalization; outputs are verified against the originals.")
    ap.add_argument("inputs", nargs="+", type=Path, help="source ONNX files (e.g. data/models/deimv2_*.onnx)")
    ap.add_argument("--out-dir", type=Path, required=True, help="output directory (file names are shortened and tagged with the suffix unless --keep-name)")
    ap.add_argument("--check-image", type=Path, default=None, help="image used for the numerical check (default: random input)")
    ap.add_argument("--no-verify", action="store_true", help="skip the numerical comparison with the original graph")
    ap.add_argument("--keep-negative-axes", action="store_true", help="do not rewrite negative axis attributes")
    ap.add_argument("--keep-name", action="store_true", help="keep the original file name (default: shortened name + _webgpu)")
    ap.add_argument("--suffix", default="_webgpu", help="suffix appended to the shortened name (default: _webgpu)")
    ap.add_argument("--no-simplify", action="store_true",
                    help="skip the onnxslim / onnxsim / canonicalization stage of the ONNX optimization skill")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for src in args.inputs:
        model = onnx.load(str(src))
        print(f"{src.name}: before: {graph_stats(model)}")
        batch_fixed = fix_batch_to_one(model)
        if batch_fixed:
            print(f"  batch dimension fixed to 1 on {batch_fixed} graph input(s) / output(s)")
        model, stats = convert_double_to_float(model)
        simplified_by = "skipped"
        if not args.no_simplify:
            model, simplified_by = simplify_slim_sim(model, args.out_dir, src.stem)
        axes = {"attr": 0, "axes_input": 0, "unknown_rank": 0} if args.keep_negative_axes else normalize_negative_axes(model)
        matmul_1d = fix_matmul_1d(model)
        nan_inf = rewrite_isinf_isnan(model)
        pruned = prune_unreachable(model)
        del model.graph.value_info[:]
        model = shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        canon = {"qkv_rank5": 0, "shape_dealiased": 0, "reshape_explicit": 0, "runtime_probed": 0}
        if not args.no_simplify:
            canon = canonicalize_with_skill(model, model.graph.input[0].name)
            del model.graph.value_info[:]
            model = shape_inference.infer_shapes(model)
            onnx.checker.check_model(model)
        dst = args.out_dir / (src.name if args.keep_name else web_name(src, args.suffix))
        tmp = dst.with_name(dst.name + ".tmp")
        onnx.save(model, str(tmp))
        onnx.checker.check_model(str(tmp))
        tmp.replace(dst)
        print(f"  double->float: cast {stats['cast']}, constant {stats['constant']}, initializer {stats['initializer']}; "
              f"simplified by {simplified_by}; negative axes {axes['attr']} attr + {axes['axes_input']} input "
              f"({axes['unknown_rank']} left); 1-D MatMul {matmul_1d}; IsNaN/IsInf rewritten {nan_inf['isnan']}/{nan_inf['isinf']}; "
              f"pruned {pruned['nodes']} nodes / {pruned['initializers']} inits; "
              f"qkv rank-5 {canon['qkv_rank5']}; shape constants de-aliased {canon['shape_dealiased']}; "
              f"Reshape -1 made explicit {canon['reshape_explicit']} (runtime probed {canon['runtime_probed']})")
        print(f"  after: {graph_stats(model)} -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
        if not args.no_verify:
            print(f"  verify (ORT, graph optimization off): {verify(src, dst, args.check_image)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
