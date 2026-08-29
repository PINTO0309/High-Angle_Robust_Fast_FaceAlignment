"""300W-LP Flip 画像の左右番号入替(history/020)の整合性テスト。

- permutation の妥当性(対合・meta との一致)
- 元データ .mat との照合(データがある環境のみ)
- unified jsonl の Flip レコードが入替済みであること(unified がある環境のみ)
"""

from __future__ import annotations

import json
import random
import unittest
from pathlib import Path

import numpy as np

from hrffa.dataset.converters.c300wlp import ibug68_flip_perm

_DATA = Path("data/300W_LP_w_masked")
_UNIFIED = Path("datasets/unified")


class TestFlipPerm(unittest.TestCase):
    def test_perm_is_involution(self):
        perm = ibug68_flip_perm()
        self.assertTrue(np.array_equal(perm[perm], np.arange(68)))
        # 中央線上の 10 点(27-30, 33, 51, 57, 62, 66, 8)は不動
        for j in [8, 27, 28, 29, 30, 33, 51, 57, 62, 66]:
            self.assertEqual(perm[j], j)
        # 代表対
        self.assertEqual(perm[36], 45)  # 目外眼角
        self.assertEqual(perm[0], 16)   # 輪郭端
        self.assertEqual(perm[48], 54)  # 口角

    @unittest.skipUnless(_DATA.exists(), "300W_LP_w_masked なし")
    def test_flip_mat_is_plain_mirror(self):
        """前提の再確認: Flip の pt2d は番号入替なしの単純ミラーである。"""
        from scipy.io import loadmat
        pairs = 0
        for flip_dir in sorted(_DATA.glob("*_Flip_*"))[:2]:
            orig_dir = _DATA / flip_dir.name.replace("_Flip", "")
            for mat_f in sorted(flip_dir.glob("*.mat"))[:3]:
                if mat_f.stem.endswith("_masked"):
                    continue
                mat_o = orig_dir / mat_f.name
                if not mat_o.exists():
                    continue
                pf = loadmat(str(mat_f), variable_names=["pt2d"])["pt2d"]
                po = loadmat(str(mat_o), variable_names=["pt2d"])["pt2d"]
                mirror = np.stack([450.0 - po[0], po[1]])
                d = np.linalg.norm(pf - mirror, axis=0).mean()
                self.assertLess(d, 3.0, f"{mat_f}: 単純ミラー前提が崩れた (d={d:.1f})")
                pairs += 1
        self.assertGreater(pairs, 0)

    @unittest.skipUnless((_UNIFIED / "annotations/300wlp.jsonl").exists(),
                         "unified なし")
    def test_unified_flip_records_are_remapped(self):
        """unified の Flip レコードは .mat の補正座標に perm を適用した値と一致する。"""
        from scipy.io import loadmat

        from hrffa.dataset.geometry import correct_300wlp_pt2d
        perm = ibug68_flip_perm()
        rng = random.Random(0)
        flips, nonflips = [], []
        with open(_UNIFIED / "annotations/300wlp.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                (flips if rec["attributes"]["flip_baked"] else nonflips).append(rec)
                if len(flips) > 3000 and len(nonflips) > 3000:
                    break
        for rec in rng.sample(flips, 20) + rng.sample(nonflips, 20):
            folder, stem = rec["record_id"].split("/")[1:]
            stem = stem.removesuffix("_masked")
            mat = loadmat(str(_DATA / folder / f"{stem}.mat"),
                          variable_names=["pt2d"])["pt2d"]
            pts, _ = correct_300wlp_pt2d(np.asarray(mat, dtype=np.float64))
            expect = pts.T[perm] if rec["attributes"]["flip_baked"] else pts.T
            got = np.asarray(rec["landmarks"]["points"])
            d = np.linalg.norm(got - expect, axis=1).max()
            self.assertLess(d, 0.01, rec["record_id"])
            if rec["attributes"]["flip_baked"]:
                self.assertTrue(rec["attributes"].get("flip_index_fixed"))


if __name__ == "__main__":
    unittest.main()
