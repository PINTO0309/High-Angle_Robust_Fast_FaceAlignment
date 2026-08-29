"""history/045 D5〜D8(多層特徴・デコーダ局所項・局所畳み込み・POPoS デコード)の単体テスト。

- multilateration が真の距離から座標を厳密復元する
- 既定設定のモデルは A1 の checkpoint を strict に読める(構造不変)
- D6 / D7 は追加直後に基準モデルと同一出力(zero-init の同一性)
- D5 / D8 の形状・損失・逆伝播
"""

from __future__ import annotations

import glob
import os
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hrffa.model.popos import distance_loss, multilaterate, distance_targets  # noqa: E402
from hrffa.model.teacher import TeacherModel  # noqa: E402

HAVE_VITT = Path("ckpts/vitt_distill.pt").exists()


def _student(**kw):
    torch.manual_seed(0)
    return TeacherModel("vitt", d_model=256, dec_layers=3, patch_instance_norm=True,
                        input_norm="center05", **kw).eval()


class TestMultilateration(unittest.TestCase):
    def test_exact_recovery(self):
        h = w = 16
        torch.manual_seed(1)
        pts = torch.rand(2, 5, 2)                       # 正規化座標
        dist = distance_targets(pts, h, w)              # 真の距離(セル単位)
        rec = multilaterate(dist, h, w, k=6)
        self.assertLess((rec - pts).abs().max().item() * w, 2e-3)   # セル単位で 0.002 未満

    def test_gradient_flows(self):
        h = w = 8
        dist = (torch.rand(1, 3, h * w) * 10).requires_grad_()   # leaf にしてから使う
        multilaterate(dist, h, w, k=5).sum().backward()
        self.assertIsNotNone(dist.grad)
        self.assertTrue(torch.isfinite(dist.grad).all())

    def test_distance_loss_finite(self):
        h = w = 8
        pts = torch.rand(2, 4, 2); vis = torch.full((2, 4), 2)
        dist = torch.rand(2, 4, h * w) * 10
        loss = distance_loss(dist, pts, vis, h, w, radius=3.0)
        self.assertTrue(torch.isfinite(loss))


@unittest.skipUnless(HAVE_VITT, "ckpts/vitt_distill.pt がない環境")
class TestArms(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(2, 3, 64, 64)               # 4×4 トークン(軽量)

    def test_default_loads_a1_checkpoint(self):
        cks = sorted(glob.glob("runs/student_s256_96gb*/student_s256_96gb_best_*.pt"))
        if not cks:
            self.skipTest("A1 checkpoint なし")
        ck = torch.load(cks[-1], map_location="cpu", weights_only=False)
        _student().load_state_dict(ck["ema"], strict=True)

    def _same_as_base(self, **kw):
        base = _student()
        arm = _student(**kw)
        missing, unexpected = arm.load_state_dict(base.state_dict(), strict=False)
        self.assertFalse(unexpected)
        with torch.no_grad():
            a, b = base(self.x, "ibug68"), arm(self.x, "ibug68")
        for k in ("points", "vis_logits", "dec_tokens"):
            self.assertLess((a[k] - b[k]).abs().max().item(), 1e-5, k)
        return missing

    def test_d7_local_conv_identity_at_init(self):
        missing = self._same_as_base(local_conv=True)
        self.assertTrue(all(".local." in k for k in missing))

    def test_d6_dec_local_identity_at_init(self):
        missing = self._same_as_base(dec_local_iters=2)
        self.assertTrue(all(k.split(".")[0] in ("local_proj", "local_gate", "delta_head") for k in missing))

    def test_d6_delta_changes_after_perturb(self):
        arm = _student(dec_local_iters=2)
        with torch.no_grad():
            arm.delta_head[-1].weight.normal_(0, 0.01)
            out = arm(self.x, "wflw98")
        self.assertEqual(tuple(out["points"].shape), (2, 98, 2))

    def test_d5_multilayer_shapes(self):
        arm = _student(feat_layers=(5, 8, 11))
        self.assertEqual(arm.backbone.embed_dim, 192 * 3)
        with torch.no_grad():
            out = arm(self.x, "cofw29")
        self.assertEqual(tuple(out["points"].shape), (2, 29, 2))
        self.assertEqual(tuple(out["memory"].shape), (2, 16, 256))

    def test_d8_popos_forward_backward(self):
        arm = _student(head="popos", popos_topk=6).train()
        out = arm(self.x, "ibug68")
        self.assertEqual(tuple(out["dist_pred"].shape), (2, 68, 16))
        self.assertEqual(out["grid_hw"], (4, 4))
        pts = torch.rand(2, 68, 2); vis = torch.full((2, 68), 2)
        loss = distance_loss(out["dist_pred"], pts, vis, 4, 4, radius=2.0) + out["points"].abs().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(arm.dist_q.weight.grad).all())

    def test_invalid_combo(self):
        with self.assertRaises(ValueError):
            _student(head="popos", dec_local_iters=1)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(Path("ckpts/PPHGNetV2_B0_stage1.pth").exists(), "PP-HGNetV2 B0 の重みがない環境")
class TestCnnStudent(unittest.TestCase):
    """history/049: 軽量 CNN 学生(PP-HGNetV2-B0 + FPN ネック + 小型デコーダ)。"""

    def test_pretrained_strict_load_and_strides(self):
        from hrffa.model.hgnetv2 import HGNetV2
        net = HGNetV2("hgnetv2_b0")
        net.load_pretrained(Path("ckpts/PPHGNetV2_B0_stage1.pth"))
        with torch.no_grad():
            outs = net(torch.randn(1, 3, 128, 128))
        self.assertEqual([tuple(o.shape) for o in outs],
                         [(1, 64, 32, 32), (1, 256, 16, 16), (1, 512, 8, 8), (1, 1024, 4, 4)])

    def test_student_budget_and_shapes(self):
        torch.manual_seed(0)
        m = TeacherModel("hgnetv2_b0", d_model=128, dec_layers=2, n_heads=4, ffn_dim=512,
                         input_norm="imagenet", feat_stride=8, cnn_feat_ch=128).eval()
        n = sum(p.numel() for p in m.parameters())
        self.assertLess(n, 2.5e6)
        with torch.no_grad():
            out = m(torch.randn(2, 3, 256, 256), "ibug68")
        self.assertEqual(tuple(out["memory"].shape), (2, 32 * 32, 128))
        self.assertEqual(tuple(out["points"].shape), (2, 68, 2))
        self.assertEqual(tuple(out["dec_tokens"].shape), (2, 68, 128))

    def test_center05_equivalence(self):
        """center05 入力の CNN 学生は、同じ画像を ImageNet 正規化で入れた imagenet 版と厳密に一致する。"""
        torch.manual_seed(0)
        m_im = TeacherModel("hgnetv2_b0", d_model=128, dec_layers=2, n_heads=4, ffn_dim=512,
                            input_norm="imagenet", feat_stride=8).eval()
        m_c5 = TeacherModel("hgnetv2_b0", d_model=128, dec_layers=2, n_heads=4, ffn_dim=512,
                            input_norm="center05", feat_stride=8).eval()
        # 両者とも同じ事前学習重みから構築(center05 版は stem1 conv + BN に折り込み済み)。
        # デコーダ等の乱数初期化を揃えるため、backbone 以外は imagenet 版の重みをコピーする
        sd = {k: v for k, v in m_im.state_dict().items() if not k.startswith("backbone.")}
        m_c5.load_state_dict(sd, strict=False)
        img = torch.rand(2, 3, 128, 128)                          # [0,1] の画像
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x_im, x_c5 = (img - mean) / std, (img - 0.5) / 0.5
        with torch.no_grad():
            s1_im = m_im.backbone.inner.net.stem.stem1(x_im)     # 折り込み前後の stem1 出力(BN+ReLU+LAB 込み)
            s1_c5 = m_c5.backbone.inner.net.stem.stem1(x_c5)
        # 内部画素は厳密一致(ゼロ padding の境界 1 画素の帯だけが等価でない。RF が画像全体に及ぶため
        # 以降の層・最終座標は近似一致にとどまり、学習で吸収される。049 §4)
        self.assertLess((s1_im - s1_c5)[..., 1:-1, 1:-1].abs().max().item(), 1e-4)
        self.assertGreater((s1_im - s1_c5).abs().max().item(), 1e-4)   # 境界には差が残る(折り込みの限界の記録)
        # 折り込みで変わるのは stem1 の conv 重みと BN running_mean だけ
        a, b = m_im.backbone.state_dict(), m_c5.backbone.state_dict()
        changed = [k for k in a if k.startswith("inner.net.") and not torch.equal(a[k], b[k])]   # ネックは乱数初期化で別物
        self.assertEqual(sorted(changed), ["inner.net.stem.stem1.bn.running_mean", "inner.net.stem.stem1.conv.weight"])

    def test_patch_in_for_cnn(self):
        m = TeacherModel("hgnetv2_b0", d_model=128, dec_layers=2, n_heads=4, ffn_dim=512,
                         input_norm="center05", feat_stride=8, patch_instance_norm=True).eval()
        self.assertIsNotNone(m.backbone.inner.patch_in)
        self.assertIn("backbone.inner.patch_in.weight", m.state_dict())
        with torch.no_grad():
            out = m(torch.randn(1, 3, 256, 256), "ibug68")
        self.assertEqual(tuple(out["points"].shape), (1, 68, 2))

    def test_invalid_options_for_cnn(self):
        with self.assertRaises(AssertionError):
            TeacherModel("hgnetv2_b0", d_model=128, input_norm="center05", feat_stride=8, local_conv=True)


class TestExportModules(unittest.TestCase):
    """export 用デコーダ(バッチ軸を潰さない Reshape)が nn.TransformerDecoder と同一計算であること。"""

    def test_decoder_equivalence(self):
        from hrffa.model.export_modules import ExportDecoder
        for d, h, ffn, layers in ((256, 8, 1024, 3), (128, 4, 512, 2)):
            torch.manual_seed(0)
            layer = torch.nn.TransformerDecoderLayer(d, h, dim_feedforward=ffn, dropout=0.0,
                                                     batch_first=True, norm_first=True)
            dec = torch.nn.TransformerDecoder(layer, layers, norm=torch.nn.LayerNorm(d)).eval()
            exp = ExportDecoder(dec).eval()
            q, mem = torch.randn(3, 68, d), torch.randn(3, 256, d)
            with torch.no_grad():
                a, b = dec(q, mem), exp(q, mem)
            self.assertLess((a - b).abs().max().item(), 1e-5)

    @unittest.skipUnless(HAVE_VITT, "ckpts/vitt_distill.pt がない環境")
    def test_teacher_model_equivalence(self):
        from hrffa.model.export_modules import to_export_model
        m = _student()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            a = m(x, "wflw98")
        for static in (False, True):
            e = to_export_model(m, static=static)
            with torch.no_grad():
                b = e(x, "wflw98")
            for k in ("points", "vis_logits", "dec_tokens"):
                self.assertLess((a[k] - b[k]).abs().max().item(), 1e-5, f"{k} static={static}")
        to_export_model(m, static=False)   # グローバルフラグを戻す

    def test_eliminate_qkv_rank5_graph_rewrite(self):
        """nbatch.eliminate_qkv_rank5: hub 形(Reshape→Split(axis 2)→Squeeze)と permute 形(Reshape→Transpose→Split(axis 0)
        →Squeeze)の qkv 分割を 4 次元に書き換え、数値が一致し rank 5 が消える(2026-08-29)。"""
        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper, numpy_helper, shape_inference
        from hrffa.export.nbatch import eliminate_qkv_rank5
        B, N, H, D = 1, 7, 2, 4
        C = H * D

        def build(pattern):
            nodes, inits = [], []
            inits.append(numpy_helper.from_array(np.asarray([B, N, 3, H, D], dtype=np.int64), "shape5"))
            nodes.append(helper.make_node("Reshape", ["x", "shape5"], ["r5"], name="attn/Reshape"))
            if pattern == "A":
                inits.append(numpy_helper.from_array(np.asarray([1, 1, 1], dtype=np.int64), "split1"))
                nodes.append(helper.make_node("Split", ["r5", "split1"], ["s0", "s1", "s2"], name="attn/Split", axis=2))
                inits.append(numpy_helper.from_array(np.asarray([2], dtype=np.int64), "ax"))
                for i in range(3):
                    nodes.append(helper.make_node("Squeeze", [f"s{i}", "ax"], [f"q{i}"], name=f"attn/Squeeze_{i}"))
                    nodes.append(helper.make_node("Transpose", [f"q{i}"], [f"y{i}"], name=f"attn/Transpose_{i}", perm=[0, 2, 1, 3]))
            else:
                nodes.append(helper.make_node("Transpose", ["r5"], ["t5"], name="attn/Transpose", perm=[2, 0, 3, 1, 4]))
                inits.append(numpy_helper.from_array(np.asarray([1, 1, 1], dtype=np.int64), "split1"))
                nodes.append(helper.make_node("Split", ["t5", "split1"], ["s0", "s1", "s2"], name="attn/Split", axis=0))
                inits.append(numpy_helper.from_array(np.asarray([0], dtype=np.int64), "ax"))
                for i in range(3):
                    nodes.append(helper.make_node("Squeeze", [f"s{i}", "ax"], [f"y{i}"], name=f"attn/Squeeze_{i}"))
            g = helper.make_graph(nodes, "qkv", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [B, N, 3 * C])],
                                  [helper.make_tensor_value_info(f"y{i}", TensorProto.FLOAT, [B, H, N, D]) for i in range(3)], inits)
            mm = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
            mm.ir_version = 10                      # onnx 1.19 の既定 IR 12 は ORT 1.22 が読めない
            return mm

        x = np.random.default_rng(0).standard_normal((B, N, 3 * C)).astype(np.float32)
        for pattern in ("A", "B"):
            m = build(pattern)
            ref = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})
            self.assertEqual(eliminate_qkv_rank5(m), 1)
            got = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})
            for a, b in zip(ref, got):
                self.assertTrue(np.array_equal(a, b), pattern)
            gi = shape_inference.infer_shapes(m).graph
            self.assertEqual([vi.name for vi in gi.value_info if len(vi.type.tensor_type.shape.dim) == 5], [], pattern)
            self.assertEqual(sum(1 for n in m.graph.node if n.op_type == "Squeeze"), 0, pattern)
            self.assertEqual(eliminate_qkv_rank5(m), 0, pattern)          # 冪等

    def test_attention_qkv_4d(self):
        """vit_tiny の注意機構: 4 次元だけの qkv 分割が旧実装(reshape(B,N,3,H,Dh) → permute → unbind)と一致し、
        静的 export の ONNX に 5 次元テンソルが残らない(2026-08-29)。"""
        import tempfile
        import onnx
        from onnx import shape_inference
        from hrffa.model import vit_tiny
        import torch.nn.functional as F
        from hrffa.model.export_modules import to_export_model
        from hrffa.export.export_onnx import ExportWrapper
        torch.manual_seed(0)
        attn = vit_tiny._Attention(192, 3).eval()
        x = torch.randn(2, 17, 192)
        rope = attn_rope = vit_tiny.Rope2D(64)
        attn_rope.periods.copy_(torch.linspace(1.0, 100.0, 16))
        sin_cos = rope(4, 4)
        with torch.no_grad():
            new = attn(x, sin_cos, prefix=1)
            b, n, c = x.shape
            qkv = attn.qkv(x).reshape(b, n, 3, attn.num_heads, attn.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            sin, cos = sin_cos
            q = torch.cat([q[:, :, :1], vit_tiny._rope_apply(q[:, :, 1:], sin, cos)], dim=2)
            k = torch.cat([k[:, :, :1], vit_tiny._rope_apply(k[:, :, 1:], sin, cos)], dim=2)
            ref = attn.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c))
        self.assertLess((new - ref).abs().max().item(), 1e-6)
        m = to_export_model(_student(), static=True)
        wrapper = ExportWrapper(m, "ibug68").eval()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.onnx"
            torch.onnx.export(wrapper, (torch.randn(1, 3, 64, 64),), str(path), opset_version=17,
                              input_names=["images"], output_names=["points", "vis_logits"], dynamo=False)
            g = shape_inference.infer_shapes(onnx.load(str(path))).graph
            rank5 = [vi.name for vi in list(g.value_info) + list(g.input) + list(g.output)
                     if len(vi.type.tensor_type.shape.dim) == 5]
            self.assertEqual(rank5, [])
        to_export_model(m, static=False)


@unittest.skipUnless(HAVE_VITT, "ckpts/vitt_distill.pt がない環境")
class TestNBatch(unittest.TestCase):
    """固定バッチ 1 の ONNX → N バッチ化(export/nbatch.py)。変換関数の内部で N=3 一括と 1 枚ずつの一致を検証する。"""

    def test_fixed_to_n(self):
        import tempfile
        from hrffa.export.export_onnx import ExportWrapper
        from hrffa.model.export_modules import to_export_model
        from hrffa.export.nbatch import convert_fixed_batch_to_n
        m = to_export_model(_student(), static=True)
        wrapper = ExportWrapper(m, "ibug68").eval()
        with tempfile.TemporaryDirectory() as d:
            fixed, nb = Path(d) / "fixed.onnx", Path(d) / "fixed_n.onnx"
            torch.onnx.export(wrapper, (torch.randn(1, 3, 64, 64),), str(fixed), opset_version=17,
                              input_names=["images"], output_names=["points", "vis_logits"],
                              dynamo=False)
            info = convert_fixed_batch_to_n(fixed, nb, n_check=3, strict=False)  # 生 export は RoPE の If 由来の未解決記号が残る
            self.assertLess(info["max_err"], 1e-5)
            self.assertGreater(info["reshape_rewritten"], 0)
        to_export_model(m, static=False)
