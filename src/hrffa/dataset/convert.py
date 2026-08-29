"""統合データセット変換 CLI。

使い方:
    python3 -m hrffa.dataset.convert --source all --out datasets/unified
    python3 -m hrffa.dataset.convert --source 300wlp --limit 5   # サンプル動作確認

出力:
    <out>/annotations/<source>.jsonl        レコード本体
    <out>/annotations/<source>.stats.json   統計
    <out>/annotations/meta/<scheme>.json    scheme メタ(flip/edge)
    <out>/images/<source>                   データ実体へのシンボリックリンク
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from tqdm import tqdm

from .converters import c300wlp, cofw, w300, wflw
from .converters.base import JsonlWriter, write_stats
from .materialize import materialize_source

_META_DIR = Path(__file__).parent / "meta"

# source 名 → (データルート相対パス, 画像リンク先相対パス, iter_records)
_SOURCES = {
    "300wlp": ("300W_LP_w_masked", "300W_LP_w_masked", c300wlp.iter_records),
    "wflw": ("ORFormer/WFLW", "ORFormer/WFLW/WFLW_images", wflw.iter_records),
    "300w": ("ORFormer/300W", "ORFormer/300W", w300.iter_records),
    "cofw": ("ORFormer/COFW", "ORFormer/COFW/images", cofw.iter_records),
}

def convert_source(
    name: str, data_root: Path, out_root: Path, limit: int | None, workers: int
) -> dict:
    src_rel, _img_rel, iter_fn = _SOURCES[name]
    src_root = data_root / src_rel
    if not src_root.exists():
        raise FileNotFoundError(src_root)

    t0 = time.time()
    out_path = out_root / "annotations" / f"{name}.jsonl"
    with JsonlWriter(out_path) as w:
        for rec in tqdm(iter_fn(src_root, limit=limit, workers=workers),
                        desc=name, unit="rec"):
            w.write(rec)
        stats = w.stats()
    stats["elapsed_sec"] = round(time.time() - t0, 1)
    stats["source"] = name
    write_stats(out_root / "annotations" / f"{name}.stats.json", stats)
    # 画像を実コピーで実体化(シンボリックリンクは廃止: history/007)
    materialize_source(name, out_root, data_root)
    return stats


def copy_scheme_meta(out_root: Path) -> None:
    meta_out = out_root / "annotations" / "meta"
    meta_out.mkdir(parents=True, exist_ok=True)
    for f in _META_DIR.glob("*.json"):
        shutil.copy2(f, meta_out / f.name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert the source datasets into the unified dataset format.")
    ap.add_argument("--source", default="all",
                    choices=["all", *_SOURCES.keys()])
    ap.add_argument("--data-root", default="data", type=Path)
    ap.add_argument("--out", default="datasets/unified", type=Path)
    ap.add_argument("--limit", type=int, default=None,
                    help="conversion limit per source (per folder for 300wlp)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    names = list(_SOURCES.keys()) if args.source == "all" else [args.source]
    copy_scheme_meta(args.out)
    for name in names:
        stats = convert_source(name, args.data_root, args.out, args.limit, args.workers)
        errs = stats.pop("validation_errors")
        print(f"[{name}] records={stats['n_records']} splits={stats['splits']} "
              f"pose={stats['pose_available']} offset={stats['offset_check']} "
              f"elapsed={stats['elapsed_sec']}s")
        if errs:
            print(f"[{name}] VALIDATION ERRORS ({len(errs)} shown):")
            for e in errs[:10]:
                print("  -", e)


if __name__ == "__main__":
    main()
