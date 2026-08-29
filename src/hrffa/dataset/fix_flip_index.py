"""300W-LP Flip 画像のランドマーク左右番号入替を既存 unified jsonl に適用する移行ツール。

背景(history/020): 300W-LP の Flip フォルダの pt2d は座標ミラーのみで
ibug68 の左右番号対応が入れ替えられておらず、Flip レコード(300wlp の 50%)と
その派生(synth_dwarp 系)が左右逆の教師信号になっていた。
変換器(converters/c300wlp.py)は修正済みだが、unified は DEIMv2 bbox 適用後の
成果物のため、再変換ではなく本ツールでインプレース修正する
(番号の並べ替えは bbox・姿勢・direction8 と独立)。

対象:
- 300wlp.jsonl: attributes.flip_baked が真のレコード
- synth_dwarp*.jsonl: attributes.parent_record に "Flip" を含むレコード

冪等: 適用済みレコードには attributes.flip_index_fixed=true を付け、再実行時はスキップ。

usage: uv run python -m hrffa.dataset.fix_flip_index [--unified datasets/unified] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .converters.c300wlp import ibug68_flip_perm

_MARK = "flip_index_fixed"


def _needs_fix(rec: dict, kind: str) -> bool:
    if rec["attributes"].get(_MARK):
        return False
    if kind == "300wlp":
        return bool(rec["attributes"].get("flip_baked"))
    return "Flip" in rec["attributes"].get("parent_record", "")


def fix_file(path: Path, kind: str, perm: list[int], dry_run: bool) -> tuple[int, int]:
    """1 ファイルを移行し (総数, 修正数) を返す。tmp 書き出し後に置換する。"""
    tmp = path.with_suffix(".jsonl.tmp")
    total = fixed = 0
    with open(path) as fin, open(tmp, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            total += 1
            if _needs_fix(rec, kind):
                lm = rec["landmarks"]
                assert lm["scheme"] == "ibug68", rec["record_id"]
                lm["points"] = [lm["points"][j] for j in perm]
                lm["visibility"] = [lm["visibility"][j] for j in perm]
                rec["attributes"][_MARK] = True
                fixed += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if dry_run:
        tmp.unlink()
    else:
        tmp.replace(path)
    return total, fixed


def main() -> None:
    ap = argparse.ArgumentParser(description="Migration tool: apply the left/right landmark index swap to 300W-LP flip images in the unified jsonl.".splitlines()[0])
    ap.add_argument("--unified", type=Path, default=Path("datasets/unified"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    perm = ibug68_flip_perm().tolist()
    targets = [("300wlp.jsonl", "300wlp"),
               ("synth_dwarp.jsonl", "synth"),
               ("synth_dwarp10k.jsonl", "synth")]
    for name, kind in targets:
        path = args.unified / "annotations" / name
        if not path.exists():
            print(f"{name}: missing (skipped)")
            continue
        total, fixed = fix_file(path, kind, perm, args.dry_run)
        tag = "(dry-run)" if args.dry_run else ""
        print(f"{name}: swapped left/right indices for {fixed}/{total} records {tag}")


if __name__ == "__main__":
    main()
