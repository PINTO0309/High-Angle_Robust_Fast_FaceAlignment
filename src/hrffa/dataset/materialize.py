"""統合データセットの画像を datasets/ 配下へ実コピーで実体化する。

D1〜D2 では images/<source> を data/ へのディレクトリ・シンボリックリンクとして
いたが、他マシンへの移行時に絶対パスリンクが壊れるため、参照画像を実コピーして
`datasets/unified` を自己完結にする(決定の経緯は history/007)。

- 各ソースの JSONL からユニークな image_path を列挙し、data/ 配下の対応ファイルを
  `datasets/unified/images/...` へ copy2 する(既存・同サイズならスキップ = resumable)
- 既存のディレクトリ・シンボリックリンクは実体化前に除去する
- synth_dwarp は生成時から実体のため対象外

使い方:
    PYTHONPATH=src python3 -m hrffa.dataset.materialize \
        --unified datasets/unified --data-root data
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from tqdm import tqdm

# source 名 → (images/ 配下のリンク名, data ルートからの画像ベース相対パス)
SOURCES = {
    "300wlp": ("300wlp", "300W_LP_w_masked"),
    "wflw": ("wflw", "ORFormer/WFLW/WFLW_images"),
    "300w": ("300w", "ORFormer/300W"),
    "cofw": ("cofw", "ORFormer/COFW/images"),
}


def materialize_source(name: str, unified: Path, data_root: Path) -> dict:
    link_name, src_rel = SOURCES[name]
    img_root = unified / "images" / link_name
    src_base = data_root / src_rel

    if img_root.is_symlink():
        img_root.unlink()

    jsonl = unified / "annotations" / f"{name}.jsonl"
    seen: set[str] = set()
    n_copied = n_skipped = 0
    bytes_copied = 0
    with open(jsonl, encoding="utf-8") as f:
        for line in tqdm(f, desc=f"materialize:{name}", unit="rec"):
            p = json.loads(line)["image_path"]
            if p in seen:
                continue
            seen.add(p)
            rel = Path(p).relative_to(Path("images") / link_name)
            src = src_base / rel
            dst = unified / p
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                n_skipped += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n_copied += 1
            bytes_copied += src.stat().st_size
    return {"source": name, "unique_images": len(seen), "copied": n_copied,
            "skipped_existing": n_skipped, "bytes_copied": bytes_copied}


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize the unified dataset images as real copies under datasets/.")
    ap.add_argument("--unified", type=Path, default=Path("datasets/unified"))
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--source", default="all", choices=["all", *SOURCES.keys()])
    args = ap.parse_args()

    names = list(SOURCES.keys()) if args.source == "all" else [args.source]
    total = 0
    for name in names:
        stats = materialize_source(name, args.unified, args.data_root)
        total += stats["bytes_copied"]
        print(f"[{name}] unique={stats['unique_images']:,} copied={stats['copied']:,} "
              f"skipped={stats['skipped_existing']:,} ({stats['bytes_copied']/1e9:.2f} GB)")
    print(f"total copied: {total/1e9:.2f} GB")


if __name__ == "__main__":
    main()
