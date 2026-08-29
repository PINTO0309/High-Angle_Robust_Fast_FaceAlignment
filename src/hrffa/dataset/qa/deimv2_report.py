"""DEIMv2 擬似ラベル適用結果の検証レポート。

- ソースごとの適用結果内訳・包含率分布・head スコア分布
- 300wlp: GT yaw と direction8(front/side 系)のクロス集計。
  弱ラベルの信頼性を GT で検証する(D2 の受け入れ判定)。

使い方:
    PYTHONPATH=src python3 -m hrffa.dataset.qa.deimv2_report --unified datasets/unified
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# 300W-LP の yaw(度)→ 期待される direction8 セクタ。
# 正面 ±22.5° を front、±(22.5-67.5)° を right/left_front、
# ±67.5° 超を right/left_side とする(back 系は 300W-LP には存在しない)。
# 注意: DEIMv2 の right/left は「画像内で頭部がどちらを向いて見えるか」であり、
# 300W-LP の yaw 正方向(被写体の左向き = 画像では右向きに見える)との対応は
# クロス集計自体で確認する。


def yaw_sector(yaw_deg: float) -> str:
    a = abs(yaw_deg)
    side = "L" if yaw_deg > 0 else "R"  # 符号→左右はクロス集計で意味付けを確認
    if a <= 22.5:
        return "front"
    if a <= 67.5:
        return f"{side}_front"
    return f"{side}_side"


def main() -> None:
    ap = argparse.ArgumentParser(description="Verification report for the DEIMv2 pseudo-label application.")
    ap.add_argument("--unified", type=Path, default=Path("datasets/unified"))
    ap.add_argument("--sources", nargs="+",
                    default=["300wlp", "wflw", "300w", "cofw"])
    args = ap.parse_args()

    for source in args.sources:
        path = args.unified / "annotations" / f"{source}.jsonl"
        if not path.exists():
            continue
        result: Counter = Counter()
        containment: list[float] = []
        head_score: list[float] = []
        crosstab: dict[str, Counter] = defaultdict(Counter)
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                q = rec["quality"]
                result[q.get("deimv2", "not_applied")] += 1
                if q.get("deimv2") == "matched":
                    containment.append(q["deimv2_containment"])
                    head_score.append(q["deimv2_head_score"])
                if source == "300wlp" and rec["direction8"] and rec["pose"]:
                    sec = yaw_sector(rec["pose"]["euler_deg"]["yaw"])
                    crosstab[sec][rec["direction8"]] += 1

        print(f"\n=== {source} ===")
        total = sum(result.values())
        for k, v in result.most_common():
            print(f"  {k}: {v} ({v / total:.1%})")
        if containment:
            c = np.array(containment)
            s = np.array(head_score)
            print(f"  containment: p1={np.percentile(c, 1):.3f} p50={np.percentile(c, 50):.3f} "
                  f"min={c.min():.3f} | head_score: p50={np.percentile(s, 50):.3f} min={s.min():.3f}")
        if crosstab:
            dirs = ["front", "right_front", "right_side", "right_back", "back",
                    "left_back", "left_side", "left_front"]
            print("  yaw_sector \\ direction8:")
            print("    " + " ".join(f"{d[:7]:>7}" for d in dirs))
            for sec in ["front", "R_front", "R_side", "L_front", "L_side"]:
                row = crosstab.get(sec)
                if row is None:
                    continue
                print(f"    {sec:>8}: " + " ".join(f"{row.get(d, 0):>7}" for d in dirs))


if __name__ == "__main__":
    main()
