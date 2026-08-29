"""D3: ソース横断統計レポート生成。

統合アノテーション全体を走査し、以下を生成する:
  - 統計 JSON(<out>/dataset_report.json)
  - 図(<out>/*.png): 姿勢分布 / |pitch| ギャップ / head bbox 相対サイズ
  - stdout に markdown 表(history へ転記する要約)

使い方:
    PYTHONPATH=src python3 -m hrffa.dataset.qa.dataset_report \
        --unified datasets/unified --out history/assets/004
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_SOURCES = ["300wlp", "wflw", "300w", "cofw"]

# dataviz 既定パレット(light)
_BLUE = "#2a78d6"
_CAT4 = {"300wlp": "#2a78d6", "wflw": "#eb6834", "300w": "#1baf7a", "cofw": "#eda100"}
_SURFACE = "#fcfcfb"
_TEXT = "#0b0b0b"
_TEXT2 = "#52514e"

_PITCH_BINS = [0, 15, 30, 45, 60, 90, 120, 180]


def _style_ax(ax):
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_TEXT2)
    ax.tick_params(colors=_TEXT2, labelsize=9)
    ax.grid(axis="y", color="#e6e5e2", linewidth=0.8)
    ax.set_axisbelow(True)


def collect(unified: Path) -> dict:
    agg: dict = {
        "records": defaultdict(Counter),          # source -> split -> n
        "scheme": {},                              # source -> scheme
        "visibility": defaultdict(Counter),        # source -> vis -> n
        "head_source": defaultdict(Counter),       # source -> head_bbox_source -> n
        "direction8": defaultdict(Counter),        # source -> dir -> n
        "pose": {"pitch": [], "yaw": [], "roll": []},        # 300wlp のみ
        "no_head_pose": {"pitch": [], "yaw": [], "roll": []},
        "head_area_ratio": defaultdict(list),      # source -> [head_area/img_area]
        "head_aspect": defaultdict(list),          # source -> [w/h]
        "mask_worn": Counter(),
    }
    for source in _SOURCES:
        path = unified / "annotations" / f"{source}.jsonl"
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                agg["records"][source][rec["split"]] += 1
                agg["scheme"][source] = rec["landmarks"]["scheme"]
                for v in rec["landmarks"]["visibility"]:
                    agg["visibility"][source][v] += 1
                agg["head_source"][source][rec["head_bbox_source"]] += 1
                if rec["direction8"]:
                    agg["direction8"][source][rec["direction8"]] += 1
                if rec["attributes"].get("mask_worn"):
                    agg["mask_worn"][source] += 1

                hb = rec["head_bbox"]
                w, h = rec["image_size"]
                bw, bh = hb[2] - hb[0], hb[3] - hb[1]
                agg["head_area_ratio"][source].append(bw * bh / (w * h))
                agg["head_aspect"][source].append(bw / bh)

                if rec["pose"] is not None:
                    e = rec["pose"]["euler_deg"]
                    dst = (agg["no_head_pose"]
                           if rec["quality"].get("deimv2") == "no_head" else agg["pose"])
                    dst["pitch"].append(e["pitch"])
                    dst["yaw"].append(e["yaw"])
                    dst["roll"].append(e["roll"])
    return agg


def pitch_gap_table(pitch: np.ndarray) -> list[dict]:
    rows = []
    total = len(pitch)
    a = np.abs(pitch)
    for lo, hi in zip(_PITCH_BINS[:-1], _PITCH_BINS[1:]):
        n = int(((a >= lo) & (a < hi)).sum())
        rows.append({"bin": f"{lo}-{hi}", "count": n, "share": n / total})
    return rows


def make_figures(agg: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    p = np.array(agg["pose"]["pitch"] + agg["no_head_pose"]["pitch"])
    y = np.array(agg["pose"]["yaw"] + agg["no_head_pose"]["yaw"])
    r = np.array(agg["pose"]["roll"] + agg["no_head_pose"]["roll"])

    # 1) 300wlp 姿勢 3 分布
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), facecolor=_SURFACE)
    for ax, arr, name in zip(axes, [p, y, r], ["pitch", "yaw", "roll"]):
        _style_ax(ax)
        ax.hist(arr, bins=90, range=(-180, 180), color=_BLUE, linewidth=0)
        ax.set_title(f"{name} (deg)", color=_TEXT, fontsize=11)
        ax.set_xlim(-180, 180)
    fig.suptitle("300W_LP pose distribution (all records with GT pose)",
                 color=_TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "pose_hist.png", dpi=110, facecolor=_SURFACE)
    plt.close(fig)

    # 2) |pitch| 対数スケール(ギャップ可視化)
    fig, ax = plt.subplots(figsize=(7, 3.4), facecolor=_SURFACE)
    _style_ax(ax)
    ax.hist(np.abs(p), bins=60, range=(0, 180), color=_BLUE, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlabel("|pitch| (deg)", color=_TEXT2)
    ax.set_title("300W_LP |pitch| — log scale (target range: up to 120)",
                 color=_TEXT, fontsize=11)
    for x in (60, 90, 120):
        ax.axvline(x, color=_TEXT2, linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(out / "pitch_abs_log.png", dpi=110, facecolor=_SURFACE)
    plt.close(fig)

    # 3) head bbox 相対面積(ソース別)
    fig, ax = plt.subplots(figsize=(7, 3.4), facecolor=_SURFACE)
    _style_ax(ax)
    for source in _SOURCES:
        arr = np.array(agg["head_area_ratio"][source])
        ax.hist(arr, bins=60, range=(0, 1.2), histtype="step",
                linewidth=2, color=_CAT4[source], label=source, density=True)
    ax.set_xlabel("head bbox area / image area", color=_TEXT2)
    ax.set_title("Head bbox relative size by source (density)", color=_TEXT, fontsize=11)
    ax.legend(frameon=False, labelcolor=_TEXT, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "headbbox_area.png", dpi=110, facecolor=_SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="D3: cross-source statistics report.")
    ap.add_argument("--unified", type=Path, default=Path("datasets/unified"))
    ap.add_argument("--out", type=Path, default=Path("history/assets/004"))
    args = ap.parse_args()

    agg = collect(args.unified)
    make_figures(agg, args.out)

    pitch = np.array(agg["pose"]["pitch"] + agg["no_head_pose"]["pitch"])
    gap = pitch_gap_table(pitch)

    stats = {
        "records": {s: dict(c) for s, c in agg["records"].items()},
        "scheme": agg["scheme"],
        "visibility": {s: {str(k): v for k, v in sorted(c.items())}
                       for s, c in agg["visibility"].items()},
        "head_bbox_source": {s: dict(c) for s, c in agg["head_source"].items()},
        "direction8": {s: dict(c) for s, c in agg["direction8"].items()},
        "mask_worn": dict(agg["mask_worn"]),
        "pitch_gap_bins": gap,
        "head_area_ratio_p50": {
            s: round(float(np.percentile(agg["head_area_ratio"][s], 50)), 4)
            for s in _SOURCES},
        "no_head_pose_summary": {
            k: {"n": len(v),
                "abs_p50": round(float(np.percentile(np.abs(v), 50)), 1) if v else None,
                "abs_p90": round(float(np.percentile(np.abs(v), 90)), 1) if v else None}
            for k, v in agg["no_head_pose"].items()},
        "matched_pose_abs_p50": {
            k: round(float(np.percentile(np.abs(agg["pose"][k]), 50)), 1)
            for k in ("pitch", "yaw", "roll")},
    }
    with open(args.out / "dataset_report.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)

    # markdown 要約
    print("\n## |pitch| gap (300W_LP, all records with GT pose)\n")
    print("| bin (deg) | count | share |")
    print("|---|---|---|")
    for row in gap:
        print(f"| {row['bin']} | {row['count']:,} | {row['share']:.3%} |")
    print("\n## pose of no_head records (median)\n")
    print(json.dumps(stats["no_head_pose_summary"], indent=1))
    print("\n## head bbox area ratio p50:", stats["head_area_ratio_p50"])
    print("\nfigures + JSON ->", args.out)


if __name__ == "__main__":
    main()
