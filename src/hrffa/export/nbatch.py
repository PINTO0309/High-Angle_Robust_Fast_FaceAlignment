"""固定バッチ 1 の ONNX を、バッチ軸だけ記号 N にした ONNX へ変換する(history/045 §6 追記 2026-08-28)。

方針は PINTO0309/PersonViT の export_onnx.py に倣う: まずバッチ 1 で完全に静的な graph を書き出し
(全 Reshape の target が明示定数)、その後に
  1. 入力/出力のバッチ軸を記号 N にする
  2. バッチ軸を先頭に持つテンソルを入力から辿り、それを受ける Reshape の定数 target の先頭要素 1 を -1 に
     書き換える(-1 は先頭のバッチ軸にだけ予約。0(コピー)は使わない。他の軸は明示のまま)
  3. onnxsim / 定数畳み込みでバッチ 1 に固定された定数(cls / register トークンなど)がバッチ依存テンソルと
     Concat される箇所は、隣のバッチ依存テンソルから Shape で N を取り、ConstantOfShape(1.0) との Mul で
     [N, …] にブロードキャストし直す
  4. 形状推論・checker を通し、ORT で「N 枚一括」と「固定 graph の 1 枚ずつ」が一致することを数値検証する
     (一致しなければ失敗として扱い、_n.onnx は出力しない)

Reshape の先頭次元の扱いが要点: 固定 graph の時点で先頭がバッチ(1)である Reshape だけを -1 にし、
バッチとトークンを融合した軸(Gemm 形の flatten)は本プロジェクトの graph には存在しない前提で検証する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

_SHAPE_OPS = {"Shape", "Size", "Constant", "ConstantOfShape"}
_REDUCE_OPS = {"ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL2",
               "ArgMax", "ArgMin"}


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default


def _const_map(model):
    """名前 -> (定数配列, 更新関数)。initializer と Constant ノードの両方を扱う。"""
    out = {}
    for init in model.graph.initializer:
        def _set(arr, init=init):
            init.CopyFrom(numpy_helper.from_array(arr, init.name))
        out[init.name] = (numpy_helper.to_array(init), _set)
    for node in model.graph.node:
        if node.op_type == "Constant" and node.attribute and node.attribute[0].name == "value":
            def _set(arr, node=node):
                node.attribute[0].t.CopyFrom(numpy_helper.from_array(arr))
            out[node.output[0]] = (numpy_helper.to_array(node.attribute[0].t), _set)
    return out


def _infer_model(model):
    try:                                                 # data_prop: Shape→Concat 由来の Reshape target も評価する
        return shape_inference.infer_shapes(model, data_prop=True)
    except Exception:
        return shape_inference.infer_shapes(model)


def _infer_dims(model):
    inferred = _infer_model(model)
    dims = {}
    for vi in list(inferred.graph.input) + list(inferred.graph.output) + list(inferred.graph.value_info):
        dims[vi.name] = [d.dim_value if d.HasField("dim_value") else None for d in vi.type.tensor_type.shape.dim]
    return dims


def _track_axes(model, consts, dims, input_name):
    """入力から前向きに辿り、(a) 各テンソルのバッチ軸の位置(名前 -> 軸番号)と、(b) Gemm 形でバッチ×トークンを
    融合したテンソル(名前 -> 融合前のトークン数 L)を求める。

    素の torch graph では vitt の注意が permute(2,0,3,1,4) → Split(axis 0) → Squeeze のようにバッチ軸を一時的に
    先頭以外へ動かすため、軸の位置を追跡する(Transpose / Squeeze / Unsqueeze / Split / Reduce で更新)。
    onnxsim は 3-D 以上の Linear を [1, L(, H), C] --Reshape[L, C] or [-1, C]--> Gemm --Reshape[1, L, C']--> に
    書き換えるため、その flatten / unflatten を融合軸として追跡する。
    """
    axis = {input_name: 0}          # name -> バッチ軸の位置
    fused = {}                       # name -> 融合前のトークン数 L
    for node in model.graph.node:
        if node.op_type in _SHAPE_OPS:
            continue
        ins_b = [i for i in node.input if i in axis]
        ins_f = [i for i in node.input if i in fused]
        if not ins_b and not ins_f:
            continue
        op = node.op_type
        outs = [o for o in node.output if o]
        if not outs:
            continue
        if op == "Reshape":
            src, out = node.input[0], outs[0]
            entry = consts.get(node.input[1])
            d_src, d_out = dims.get(src), dims.get(out)
            if src in axis and axis[src] == 0:
                if entry is None:                                # Shape 由来: 実行時に追従
                    if d_out and d_out[0] == 1:
                        axis[out] = 0
                    continue
                arr = entry[0]
                if arr.ndim != 1 or arr.size < 2:
                    continue
                # 融合 flatten: [1, d1, ..., dk] -> [K, C](C は末尾側の次元を吸収しうる)。K = Π(d1..dk) / C
                k_fused = None
                if arr.size == 2 and d_src and len(d_src) >= 3 and all(d is not None for d in d_src) and arr[1] > 0:
                    rest = int(np.prod(d_src[1:]))
                    if rest % int(arr[1]) == 0:
                        k_fused = rest // int(arr[1])
                if k_fused is not None and arr[0] in (-1, k_fused):
                    fused[out] = k_fused                         # 融合 flatten を先に判定
                elif arr[0] == 1 or (arr[0] == -1 and not np.any(arr[1:] == -1)):
                    axis[out] = 0
            elif src in fused:
                if entry is not None and entry[0].ndim == 1 and entry[0].size >= 2 and entry[0][0] == 1:
                    axis[out] = 0                                # unflatten
            continue
        k = axis[ins_b[0]] if ins_b else None
        new_k = k
        if op == "Transpose" and k is not None:
            perm = _attr(node, "perm")
            new_k = perm.index(k) if perm is not None else None
        elif op == "Concat" and k is not None:
            if _attr(node, "axis", 0) == k:
                new_k = None
        elif op == "Gather" and k is not None:
            if _attr(node, "axis", 0) == k:
                new_k = None
        elif op == "Flatten" and k is not None:
            new_k = 0 if (k == 0 and _attr(node, "axis", 1) >= 1) else None
        elif op in ("Squeeze", "Unsqueeze") and k is not None:
            ax = _attr(node, "axes")
            if ax is None and len(node.input) > 1 and node.input[1] in consts:
                ax = consts[node.input[1]][0]
            if ax is None:
                new_k = None
            else:
                ax = sorted(int(a) for a in np.atleast_1d(ax))
                if op == "Squeeze":
                    new_k = None if k in ax else k - sum(1 for a in ax if a < k)
                else:
                    new_k = k + sum(1 for a in ax if a <= k)
        elif op in _REDUCE_OPS and k is not None:
            ax = _attr(node, "axes")
            if ax is None and len(node.input) > 1 and node.input[1] in consts:
                ax = consts[node.input[1]][0]
            keep = _attr(node, "keepdims", 1)
            if ax is None:
                new_k = None
            else:
                ax = [int(a) for a in np.atleast_1d(ax)]
                new_k = None if k in ax else (k if keep else k - sum(1 for a in ax if a < k))
        elif op == "Gemm":
            new_k = None
        for o in outs:
            d = dims.get(o)
            if ins_f and d and d[0] in {fused[i] for i in ins_f}:
                fused[o] = d[0]                                  # Gemm / Add / Relu 等は融合軸を保つ
            elif ins_f and not ins_b and not d:
                fused[o] = fused[ins_f[0]]
            elif new_k is not None and (not d or new_k < len(d) and d[new_k] == 1):
                axis[o] = new_k
    batch = {n for n, k in axis.items() if k == 0}
    return batch, fused


def _consumers(model):
    cons = {}
    for node in model.graph.node:
        for i in node.input:
            cons.setdefault(i, []).append(node)
    return cons


def _dealias_reshape_shapes(model: onnx.ModelProto) -> int:
    """複数の Reshape で共有されている shape 定数を Reshape ごとに複製する(PAT-CNN-RESHAPE-001)。

    onnxsim は [-1, C] のような shape 初期化子を複数の Reshape で共有させることがあり、その場で書き換えると
    別の Reshape の意味が壊れる。target を書き換える前に必ず私有化する。
    """
    users = {}
    for n in model.graph.node:
        if n.op_type == "Reshape" and len(n.input) > 1:
            users.setdefault(n.input[1], []).append(n)
    inits = {i.name: i for i in model.graph.initializer}
    const_nodes = {n.output[0]: n for n in model.graph.node if n.op_type == "Constant"}
    cloned = 0
    for name, nodes in users.items():
        if len(nodes) < 2:
            continue
        for k, n in enumerate(nodes[1:], start=1):
            new_name = f"{name}_dealias{k}"
            if name in inits:
                t = onnx.TensorProto(); t.CopyFrom(inits[name]); t.name = new_name
                model.graph.initializer.append(t)
            elif name in const_nodes:
                c = onnx.NodeProto(); c.CopyFrom(const_nodes[name]); c.output[0] = new_name; c.name = f"{c.name}_dealias{k}"
                model.graph.node.insert(list(model.graph.node).index(const_nodes[name]) + 1, c)
            else:
                continue                                    # 動的 shape は対象外
            n.input[1] = new_name; cloned += 1
    return cloned


def _runtime_dims(model, names, input_name="images"):
    """固定 graph を ORT で 1 回実行して、指定テンソルの実形状を得る(形状推論が解決できない場合の参照実行)。"""
    import onnxruntime as ort
    probe = onnx.ModelProto(); probe.CopyFrom(model)
    existing = {o.name for o in probe.graph.output}
    for name in names:
        if name not in existing:
            probe.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.UNDEFINED, None))
    sess = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
    shape = [d.dim_value for d in probe.graph.input[0].type.tensor_type.shape.dim]
    x = np.random.default_rng(0).uniform(-1, 1, shape).astype(np.float32)
    outs = sess.run(list(names), {input_name: x})
    return {name: list(o.shape) for name, o in zip(names, outs)}


def canonicalize_fixed_graph(model: onnx.ModelProto, input_name: str = "images") -> dict:
    """固定バッチ 1 の graph の正準化(Phase A)。

    - 複数の Reshape で共有された shape 定数を私有化する(PAT-CNN-RESHAPE-001)
    - Reshape 定数の -1(hub の [1,1024,-1] / [-1,20,20,1024] など)を固定 graph の推論形状で明示し、固定 graph の
      target に -1 を残さない(INV-RESHAPE-001)
    バッチ×トークンを融合する Gemm 形は、export 側で onnxslim(FusionGemm*)/ onnxsim(fuse_matmul_add_bias_into_gemm)
    の融合を無効化して発生させない(Linear は 3-D の MatMul + Add のまま = バッチ軸が常に先頭)。
    """
    dealiased = _dealias_reshape_shapes(model)
    consts = _const_map(model)
    dims = _infer_dims(model)
    explicit = 0
    targets = []
    for n in model.graph.node:
        if n.op_type != "Reshape":
            continue
        entry = consts.get(n.input[1])
        if entry is not None and entry[0].ndim == 1 and np.any(entry[0] == -1):
            targets.append((n, entry))
    unresolved = [n.output[0] for n, entry in targets
                  if not dims.get(n.output[0]) or any(d is None for d in dims[n.output[0]])]
    if unresolved:                                       # 推論で決まらない出力は参照実行(ORT, バッチ 1)で実形状を得る
        dims.update(_runtime_dims(model, unresolved, input_name))
    for n, (arr, setter) in targets:
        d_out = dims.get(n.output[0])
        if not d_out or any(d is None for d in d_out) or len(d_out) != arr.size:
            raise RuntimeError(f"cannot make -1 explicit for Reshape {n.name} of the fixed graph: {arr.tolist()} dims={d_out}")
        setter(np.asarray(d_out, dtype=np.int64)); explicit += 1
    onnx.checker.check_model(model)
    return {"reshape_explicit": explicit, "shape_dealiased": dealiased, "runtime_probed": len(unresolved)}


def _all_inputs(graph) -> set:
    """graph 内の全ノード入力(If / Loop / Scan のサブグラフが外側スコープから参照する名前も含む)。"""
    used = set()
    for n in graph.node:
        used.update(i for i in n.input if i)
        for a in n.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                used |= _all_inputs(a.g)
            elif a.type == onnx.AttributeProto.GRAPHS:
                for g in a.graphs:
                    used |= _all_inputs(g)
    used.update(o.name for o in graph.output)
    return used


def _remove_unused_initializers(model):
    used = _all_inputs(model.graph)
    keep = [t for t in model.graph.initializer if t.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)


def convert_fixed_batch_to_n(fixed_path: Path, out_path: Path, input_name: str = "images",
                             n_check: int = 3, atol: float = 1e-4, strict: bool = True) -> dict:
    """固定バッチ 1 の ONNX を N バッチ化して out_path に保存し、数値検証の結果を返す。

    strict=True では、形状推論が残した記号次元をすべてバッチ由来(N)と証明できなければ失敗する
    (onnxslim 済み graph はこれを満たす。生 export は RoPE の If 由来の未解決記号が残るため strict=False)。
    atol は float32 の MatMul 累積順序差(バッチ 1 と N でカーネルが異なる)を許容する 1e-4。"""
    import onnxruntime as ort

    model = onnx.load(str(fixed_path))
    _dealias_reshape_shapes(model)                     # 共有 shape 定数を私有化してから書き換える
    consts = _const_map(model)
    dims = _infer_dims(model)
    batch, fused = _track_axes(model, consts, dims, input_name)

    # 2) Reshape の先頭要素を -1 に(バッチ軸、または Gemm 形の融合軸 = バッチ×トークン)
    n_reshape = 0
    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue
        src = node.input[0]
        entry = consts.get(node.input[1])
        if entry is None or (src not in batch and src not in fused):
            continue                                   # Shape 由来(実行時にバッチへ追従)はそのまま
        arr, setter = entry
        if arr.ndim != 1 or arr.size < 2:
            continue
        d_src, d_out = dims.get(src), dims.get(node.output[0])
        is_batch_head = src in batch and arr[0] == 1
        k_fused = None
        if src in batch and arr.size == 2 and d_src and len(d_src) >= 3 and all(d is not None for d in d_src) and arr[1] > 0:
            rest = int(np.prod(d_src[1:]))
            if rest % int(arr[1]) == 0:
                k_fused = rest // int(arr[1])
        is_fused_flatten = k_fused is not None and arr[0] in (-1, k_fused)
        is_unflatten = src in fused and arr[0] == 1
        if is_batch_head or is_fused_flatten or is_unflatten:
            new = arr.copy()
            if np.any(new[1:] == -1):
                # 末尾側の -1(例: hub の flatten [1, 1024, -1])は固定 graph の推論形状で明示し、-1 は先頭に予約する
                if not d_out or any(d is None for d in d_out) or len(d_out) != new.size:
                    raise RuntimeError(f"Reshape {node.name}: cannot resolve a non-leading -1 {arr.tolist()} dims={d_out}")
                for i in range(1, new.size):
                    if new[i] == -1:
                        new[i] = d_out[i]
            if new[0] != -1:
                new[0] = -1
                setter(new); n_reshape += 1
            elif not np.array_equal(new, arr):
                setter(new); n_reshape += 1

    # 3) バッチ 1 に畳まれた定数を Concat(axis≠0)で結合している箇所をブロードキャストに戻す
    n_concat = 0
    new_nodes = []
    known_dims = {}                                      # 復元ノードの出力形状(形状推論では決まらないので明示する)
    for idx, node in enumerate(list(model.graph.node)):
        if node.op_type != "Concat" or _attr(node, "axis", 0) == 0:
            continue
        others = [i for i in node.input if i in batch]
        folded = [i for i in node.input if i in consts and i not in batch and consts[i][0].ndim >= 2 and consts[i][0].shape[0] == 1]
        if not others or not folded:
            continue
        ref = others[0]
        for k, name in enumerate(folded):
            arr, _ = consts[name]
            prefix = f"{node.name}/nbatch_{k}"
            src = f"{name}_batch1"
            # 定数の名前を付け替え(initializer / Constant のどちらでも)
            renamed = False
            for init in model.graph.initializer:
                if init.name == name:
                    init.name = src; renamed = True
            if not renamed:
                for c in model.graph.node:
                    if c.op_type == "Constant" and c.output[0] == name:
                        c.output[0] = src; renamed = True
            assert renamed
            tail = numpy_helper.from_array(np.ones(arr.ndim - 1, dtype=np.int64), name=f"{prefix}/tail")
            model.graph.initializer.append(tail)
            known_dims[f"{prefix}/ones"] = ["N"] + [1] * (arr.ndim - 1)
            known_dims[name] = ["N"] + [int(v) for v in arr.shape[1:]]
            new_nodes.append((idx, [
                onnx.helper.make_node("Shape", [ref], [f"{prefix}/bshape"], name=f"{prefix}/Shape", start=0, end=1),
                onnx.helper.make_node("Concat", [f"{prefix}/bshape", tail.name], [f"{prefix}/shape"], name=f"{prefix}/Concat", axis=0),
                onnx.helper.make_node("ConstantOfShape", [f"{prefix}/shape"], [f"{prefix}/ones"], name=f"{prefix}/ConstantOfShape",
                                      value=numpy_helper.from_array(np.asarray([1.0], dtype=np.float32))),
                onnx.helper.make_node("Mul", [src, f"{prefix}/ones"], [name], name=f"{prefix}/Mul"),
            ]))
            n_concat += 1
    for idx, nodes in sorted(new_nodes, key=lambda t: -t[0]):
        for off, nd in enumerate(nodes):
            model.graph.node.insert(idx + off, nd)

    # 1) 入力/出力のバッチ軸を N に、古い ValueInfo を捨てて再推論
    for vi in list(model.graph.input) + list(model.graph.output):
        d = vi.type.tensor_type.shape.dim
        if len(d):
            d[0].ClearField("dim_value"); d[0].dim_param = "N"
    del model.graph.value_info[:]
    model = _infer_model(model)
    if known_dims:                                       # 復元ノードの形状を与えて下流(Concat のトークン軸など)を再推論
        vimap = {vi.name: vi for vi in model.graph.value_info}
        for name, dims_k in known_dims.items():
            vi = vimap.get(name)
            if vi is None:
                model.graph.value_info.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, dims_k))
            else:
                vi.type.tensor_type.shape.CopyFrom(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, dims_k).type.tensor_type.shape)
        model = _infer_model(model)
    onnx.checker.check_model(model)

    # 5) ValueInfo の具体化(INV-VALUE-INFO-001): 推論が作った unk* 記号を、N graph 上の軸追跡(バッチ軸の位置)と
    #    固定 graph の同名テンソル(該当軸 = 1)で証明できるものだけ N に置き換え、初期化子の型/形状も ValueInfo に載せる
    fixed_dims = _infer_dims(onnx.load(str(fixed_path)))
    n_consts = _const_map(model)
    n_dims = _infer_dims(model)
    n_batch, n_fused = _track_axes(model, n_consts, n_dims, input_name)
    n_axis = {}
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        n_axis[vi.name] = 0 if vi.name in n_batch else None
    replaced, unproven = 0, []
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        for k, d in enumerate(vi.type.tensor_type.shape.dim):
            if not d.dim_param or d.dim_param == "N":
                continue
            fd = fixed_dims.get(vi.name)
            proof = (k == 0 and vi.name in n_batch) or (fd is not None and len(fd) == len(vi.type.tensor_type.shape.dim) and fd[k] == 1) \
                or (vi.name in n_fused and k == 0)
            if proof:
                d.dim_param = "N" if not (vi.name in n_fused and k == 0) else f"{n_fused[vi.name]}N"
                replaced += 1
            else:
                unproven.append((vi.name, k, d.dim_param))
    if unproven and strict:
        raise RuntimeError(f"symbolic dims that cannot be proven batch-derived remain: {unproven[:5]}")
    known = {vi.name for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)}
    for init in model.graph.initializer:
        if init.name not in known:
            model.graph.value_info.append(onnx.helper.make_tensor_value_info(init.name, init.data_type, list(init.dims)))
    node_outputs = {o for n in model.graph.node for o in n.output if o}
    missing = sorted(node_outputs - {vi.name for vi in list(model.graph.output) + list(model.graph.value_info)})
    if missing:
        raise RuntimeError(f"node outputs without ValueInfo: {missing[:5]}")
    # 6) 固定/N の Reshape・Transpose の対応検査(INV-RESHAPE-001 / INV-TRANSPOSE-001):
    #    参照は正準化済み固定 graph(冪等なのでメモリ上で再適用)。N target は先頭のみ -1、残りは固定と同一
    fixed_model = onnx.load(str(fixed_path))
    canonicalize_fixed_graph(fixed_model)
    f_consts, d_consts = _const_map(fixed_model), _const_map(model)
    f_rs = {n.name: n for n in fixed_model.graph.node if n.op_type == "Reshape"}
    d_rs = {n.name: n for n in model.graph.node if n.op_type == "Reshape"}
    if f_rs.keys() != d_rs.keys():
        raise RuntimeError(f"Reshape node sets differ between fixed and N graphs: {sorted(f_rs.keys() ^ d_rs.keys())[:5]}")
    for name, fn in f_rs.items():
        fa, da = f_consts.get(fn.input[1]), d_consts.get(d_rs[name].input[1])
        if fa is None or da is None:
            continue                                    # Shape 由来(データ依存)
        fa, da = fa[0].reshape(-1), da[0].reshape(-1)
        if np.any(fa == 0) or np.any(da == 0):
            raise RuntimeError(f"Reshape {name}: target contains 0 (fixed {fa.tolist()} / N {da.tolist()})")
        if strict and np.any(fa == -1):
            raise RuntimeError(f"Reshape {name}: fixed target still contains -1 {fa.tolist()}")
        if any(v == -1 and i != 0 for i, v in enumerate(da.tolist())):
            raise RuntimeError(f"Reshape {name}: N target has -1 outside the leading axis {da.tolist()}")
        exp = fa.copy()
        if da[0] == -1:
            exp[0] = -1
        if len(da) != len(exp) or not np.array_equal(da, exp):
            raise RuntimeError(f"Reshape {name}: fixed {fa.tolist()} and N {da.tolist()} do not correspond")
    for fn in fixed_model.graph.node:
        if fn.op_type == "Transpose":
            dn = next((n for n in model.graph.node if n.name == fn.name), None)
            if dn is None or list(_attr(fn, "perm")) != list(_attr(dn, "perm")):
                raise RuntimeError(f"Transpose {fn.name}: perm differs between fixed and N graphs")
    onnx.checker.check_model(model)

    # 7) 数値検証: 固定 graph を 1 枚ずつ vs N graph を一括(バッチ 1, 2, n_check)
    fixed_sess = ort.InferenceSession(str(fixed_path), providers=["CPUExecutionProvider"])
    n_sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    shape = [d.dim_value for d in onnx.load(str(fixed_path)).graph.input[0].type.tensor_type.shape.dim]
    rng = np.random.default_rng(0)
    max_err = 0.0
    for nb in sorted({1, 2, n_check}):
        shape[0] = nb
        x = rng.uniform(-1, 1, shape).astype(np.float32)
        ref = [fixed_sess.run(None, {input_name: x[i:i + 1]}) for i in range(nb)]
        got = n_sess.run(None, {input_name: x})
        for j, g in enumerate(got):
            if g.shape[0] != nb:
                raise RuntimeError(f"output {j} has batch size {g.shape[0]} (expected {nb})")
            max_err = max(max_err, max(float(np.abs(g[i] - ref[i][j][0]).max()) for i in range(nb)))
    if not np.isfinite(max_err) or max_err > atol:
        raise RuntimeError(f"numeric check of the N-batch graph failed: max_err={max_err:.2e} (Reshape {n_reshape}, Concat {n_concat})")
    # 8) 原子的に保存(INV-PUBLISH-001)
    tmp = out_path.with_name(out_path.name + ".tmp")
    onnx.save(model, str(tmp))
    onnx.checker.check_model(str(tmp))
    tmp.replace(out_path)
    return {"reshape_rewritten": n_reshape, "concat_broadcast": n_concat, "max_err": max_err,
            "batch_tensors": len(batch), "symbols_canonicalized": replaced,
            "unproven_symbols": len(unproven)}


def eliminate_qkv_rank5(model: onnx.ModelProto) -> int:
    """qkv 分割の 5 次元テンソルを排除する graph 書き換え(2026-08-29、045 §6 追記 4)。

    torch の `qkv.reshape(B,N,3,H,D)` 系の分割は ONNX に rank 5 の Reshape / Split / Squeeze を残す。チャネル順が
    [3][H][D] なので「末尾軸で 3 分割 → 各 Reshape(B,N,H,D)」と数値的に同一で、rank 5 を作らない。
    対応パターン(Reshape の target は定数、rank 5、dim[2] == 3):
      A) Reshape → Split(axis=2, 3 出力) → Squeeze(axes=[2]) ×3               … hub DINOv3(unbind(2))
      B) Reshape → Transpose(perm 2,0,3,1,4) → Split(axis=0, 3 出力) → Squeeze(axes=[0]) ×3 … permute(2,0,3,1,4).unbind(0)
         (出力は (B,H,N,D) なので Reshape(B,N,H,D) → Transpose(0,2,1,3) を挿入)
    Squeeze の出力名は維持するので下流は変更不要。戻り値は書き換えたブロック数。
    """
    consts = _const_map(model)
    cons = _consumers(model)
    dims = _infer_dims(model)                            # 生 export は target が Shape 由来のことがある → 推論形状で補う
    nodes = list(model.graph.node)
    index = {id(n): i for i, n in enumerate(nodes)}

    def _target(r):
        d = dims.get(r.output[0])
        if d and len(d) == 5 and all(v is not None for v in d):
            return [int(v) for v in d]
        if r.input[1] in consts:
            return [int(v) for v in np.asarray(consts[r.input[1]][0]).reshape(-1)]
        return None

    def _axes(node):
        if len(node.input) > 1 and node.input[1] in consts:
            return [int(v) for v in np.asarray(consts[node.input[1]][0]).reshape(-1)]
        a = _attr(node, "axes")
        return [int(v) for v in a] if a is not None else None

    def _only_consumer(name, op_type):
        c = cons.get(name, [])
        return c[0] if len(c) == 1 and c[0].op_type == op_type else None

    rewrites = []
    for r in nodes:
        if r.op_type != "Reshape":
            continue
        t = _target(r)
        if t is None or len(t) != 5 or t[2] != 3 or t[3] <= 0 or t[4] <= 0:
            continue
        nxt = cons.get(r.output[0], [])
        if len(nxt) != 1:
            continue
        pattern, split = None, None
        if nxt[0].op_type == "Split" and _attr(nxt[0], "axis", 0) == 2 and len(nxt[0].output) == 3:
            pattern, split, tail = "A", nxt[0], [r, nxt[0]]
        elif nxt[0].op_type == "Transpose" and list(_attr(nxt[0], "perm") or []) == [2, 0, 3, 1, 4]:
            s = _only_consumer(nxt[0].output[0], "Split")
            if s is not None and _attr(s, "axis", 0) == 0 and len(s.output) == 3:
                pattern, split, tail = "B", s, [r, nxt[0], s]
        if pattern is None:
            continue
        squeezes = [_only_consumer(o, "Squeeze") for o in split.output]
        want = [2] if pattern == "A" else [0]
        if any(q is None or _axes(q) != want for q in squeezes):
            continue
        rewrites.append((pattern, r, tail + squeezes, t, squeezes))

    if not rewrites:
        return 0
    remove = set()
    inserts = []                                         # (挿入位置, 新ノード列)
    for k, (pattern, r, old_nodes, t, squeezes) in enumerate(rewrites):
        remove.update(id(n) for n in old_nodes)
        base = r.name or f"qkv_{k}"
        c = t[3] * t[4]
        split_sizes = numpy_helper.from_array(np.asarray([c, c, c], dtype=np.int64), name=f"{base}/qkv4d/split")
        model.graph.initializer.append(split_sizes)
        parts = [f"{base}/qkv4d/part{i}" for i in range(3)]
        new = [onnx.helper.make_node("Split", [r.input[0], split_sizes.name], parts, name=f"{base}/qkv4d/Split", axis=-1)]
        for i, q in enumerate(squeezes):
            shape = numpy_helper.from_array(np.asarray([t[0], t[1], t[3], t[4]], dtype=np.int64), name=f"{base}/qkv4d/shape{i}")
            model.graph.initializer.append(shape)
            if pattern == "A":
                new.append(onnx.helper.make_node("Reshape", [parts[i], shape.name], [q.output[0]], name=f"{base}/qkv4d/Reshape{i}"))
            else:
                mid = f"{base}/qkv4d/bnhd{i}"
                new.append(onnx.helper.make_node("Reshape", [parts[i], shape.name], [mid], name=f"{base}/qkv4d/Reshape{i}"))
                new.append(onnx.helper.make_node("Transpose", [mid], [q.output[0]], name=f"{base}/qkv4d/Transpose{i}", perm=[0, 2, 1, 3]))
        inserts.append((index[id(r)], new))
    kept = []
    ins_at = {pos: new for pos, new in inserts}
    for i, n in enumerate(nodes):
        if i in ins_at:
            kept.extend(ins_at[i])
        if id(n) not in remove:
            kept.append(n)
    del model.graph.node[:]
    model.graph.node.extend(kept)
    # 使われなくなった定数(旧 shape / split / axes)を除去。If などのサブグラフが外側の Constant を参照する
    # (生 export の RoPE)ため、使用判定はサブグラフも含めて行う
    used = _all_inputs(model.graph)
    dangling = [n for n in model.graph.node if n.op_type == "Constant" and n.output[0] not in used]
    for n in dangling:
        model.graph.node.remove(n)
    _remove_unused_initializers(model)
    # 旧 Shape 由来の target を作っていた Shape/Gather/Concat 等が孤立していれば除去(出力を持たないノード)
    used = _all_inputs(model.graph)
    changed = True
    while changed:
        changed = False
        for n in list(model.graph.node):
            if n.op_type in _SHAPE_OPS | {"Gather", "Concat", "Unsqueeze", "Cast", "Slice"} and not any(o in used for o in n.output):
                model.graph.node.remove(n); changed = True
        used = _all_inputs(model.graph)
    _remove_unused_initializers(model)
    del model.graph.value_info[:]                        # 先に消す(infer_shapes は既存の value_info を残すため、消した
    inferred = _infer_model(model)                       # テンソルの rank-5 エントリが残ってしまう)
    model.graph.value_info.extend(inferred.graph.value_info)
    onnx.checker.check_model(model)
    return len(rewrites)
